"""The classes an episode author writes, and the objects the cookbook hands them.

An ``Episode`` is one object per attempt. The cookbook builds it with the attempt's
``Task`` and validated options. It then awaits ``setup(setup)``, then ``run(sample)``
under the agent's deadline, then ``grade(grading)`` once the rollout is sealed. Last,
it awaits ``close(closing)`` exactly once, however the attempt ended.

``close()`` is skipped in two cases. If ``__init__`` raised or never ran, there is no
episode to close. If ``setup()``, ``run()`` or ``grade()`` ignored cancellation past
its grace period, the cookbook keeps the attempt live and reports it instead.

A retry is a fresh attempt. It gets a new ``Episode`` and a new sandbox, and the agent
runs again.

The policy is in scope only inside ``run()``. The ``Model`` arrives in the ``Sampling``
argument, and it refuses every call once ``run()`` has ended. ``grade()`` never gets
the policy, so an LLM judge brings its own client.

An episode class may also score a whole group at once with the classmethod
``grade_group()``. Once the group is complete, ``grade_group()`` gets each member's
recorded evidence as a ``GroupMember``, together with the episode's options. It
returns every member's final reward.
"""

from __future__ import annotations

import asyncio
import typing
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import NoneType, get_original_bases
from typing import Any, ClassVar, Literal, get_args, get_origin

from pydantic import BaseModel, SecretStr

from loops_cookbook.episodes.tasks import Task
from loops_cookbook.evidence import JsonValue, Rollout
from loops_cookbook.llm.client import Conversation, LoopsLLM
from loops_cookbook.llm.transcript import SealedTranscript
from loops_cookbook.model_access.api import (
    ModelAccess,
    ModelAccessError,
    ModelAccessMode,
    ModelConnection,
)
from loops_cookbook.resources.api import (
    CleanupIncomplete,
    CleanupReport,
    ResourceScope,
    SandboxKind,
)
from loops_cookbook.runtime.api import (
    AttemptProcesses,
    EpisodeResources,
    GradeResult,
    ProcessCommand,
    ProcessResult,
    RuntimeMetrics,
)


class AgentTurn:
    """The one span in which the policy is in scope, which is ``run()`` under the agent's deadline.

    When the deadline expires, the cookbook cancels ``run()``. The episode still counts.
    It is sealed with the generations delivered so far, and then graded. An exception
    that ``run()`` raises after the cut is ignored. If ``run()`` catches the
    cancellation and returns, the cookbook keeps what it returned.
    """

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._open = False
        self._timed_out = False

    @property
    def open(self) -> bool:
        return self._open

    @property
    def timed_out(self) -> bool:
        return self._timed_out

    async def run[T](self, call: Callable[[], Awaitable[T]]) -> T | None:
        """Await ``call()`` under the deadline, and return ``None`` if the deadline cut it first."""
        deadline = asyncio.timeout(self._seconds)
        self._open = True
        try:
            async with deadline:
                result = await call()
        except Exception:
            # A cancellation from outside, such as the rollout clock, is never caught.
            if not deadline.expired():
                raise
            self._timed_out = True
            return None
        finally:
            self._open = False
        # run() may have caught the deadline's cancellation and returned anyway.
        self._timed_out = deadline.expired()
        return result


class Model:
    """The policy for this attempt, usable only inside ``run()``.

    ``conversation()`` and ``llm()`` serve ``model_access = "direct"``.
    ``http_connection()`` serves ``"openai_http"``. Each takes an optional producer name
    from ``[rollout.producers]``. Once ``run()`` has ended, all three refuse. The
    attempt's model access also refuses whatever was taken from them. A direct client
    refuses once the attempt is sealed, and an HTTP credential is revoked at that point.
    """

    def __init__(self, access: ModelAccess, turn: AgentTurn) -> None:
        self._access = access
        self._turn = turn

    def llm(self, producer: str | None = None) -> LoopsLLM:
        """Return the direct client that samples as ``producer``, or as the default producer."""
        self._check()
        return self._access.llm(producer)

    def conversation(
        self, *, producer: str | None = None, conversation_id: str | None = None
    ) -> Conversation:
        """Start a direct generation session, and pass the history on each generate() call.

        The session records against this attempt and producer. Its model access ends
        when the attempt seals, as for a session taken through llm().
        """
        return self.llm(producer).conversation(conversation_id=conversation_id)

    async def http_connection(self, producer: str | None = None) -> ModelConnection:
        """Return the attempt's OpenAI-compatible endpoint and a credential for ``producer``."""
        self._check()
        return await self._access.http_connection(producer)

    def _check(self) -> None:
        if not self._turn.open:
            raise ModelAccessError("model access for this attempt is sealed")


class SandboxAttachment:
    """Holds the one client for an attempt's sandbox, on the rollout runtime's side.

    The client is attached on first use, and the same object is returned from then on.
    The rollout runtime closes it after the episode's ``close()``. After that, the
    sandbox gives no client again.
    """

    def __init__(self, resource_id: str, resources: EpisodeResources) -> None:
        self._resource_id = resource_id
        self._resources = resources
        self._lock = asyncio.Lock()
        self._client: object = None
        self._attached = False
        self._closed = False

    async def client(self) -> object:
        async with self._lock:
            if self._closed:
                raise RuntimeError(
                    "the attempt's sandbox access has ended, and its sandbox's client is closed"
                )
            if not self._attached:
                # Store the client in the same step as the call returns. Otherwise a
                # cancellation could land in between and leave a client nobody closes.
                self._client = await self._resources.sandbox_client(self._resource_id)
                self._attached = True
            return self._client

    async def alive(self) -> bool:
        async with self._lock:
            if self._closed:
                raise RuntimeError(
                    "the attempt's sandbox access has ended, and its sandbox's client is closed"
                )
            return await self._resources.sandbox_alive(self._resource_id)

    async def close(self) -> None:
        """Close the client, if one was attached. The sandbox gives no client after this."""
        async with self._lock:
            self._closed = True
            if self._attached:
                await self._resources.close_sandbox_client(self._resource_id, self._client)
                self._attached = False

    async def release(self, attempt_id: str) -> None:
        """Close the attachment, then require proof that the registered resource was released.

        If the release fails, final attempt cleanup and journal recovery retry it.
        Client access stays closed even when the release cannot be proven.
        """
        await self.close()
        result = await self._resources.release(self._resource_id)
        report = CleanupReport(ResourceScope.EPISODE, attempt_id, (result,))
        if not report.complete:
            raise CleanupIncomplete(report)


@dataclass(frozen=True, slots=True)
class GroupMember[TaskT: BaseModel | None]:
    """Holds the recorded evidence for one member of a complete group.

    A group member is never a live attempt.
    """

    task: Task[TaskT]
    transcript: SealedTranscript
    reward: float | None
    metadata: Mapping[str, JsonValue]


class Processes:
    """Runs supervised local processes for this attempt.

    A process runs in the attempt's working directory unless ``cwd`` names another.
    """

    def __init__(self, processes: AttemptProcesses, work_dir: Path) -> None:
        self._processes = processes
        self._work_dir = work_dir

    async def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str | SecretStr] | None = None,
        cwd: Path | None = None,
    ) -> ProcessResult:
        """Run ``argv`` to the end, and return its exit status and the tails of its output.

        A nonzero exit is returned, not raised. ``env`` is the process's whole
        environment beyond a small non-secret allowlist, and a ``SecretStr`` value is
        redacted from the output the cookbook keeps. The process is killed when the
        caller is cancelled or the attempt ends.
        """
        where = cwd if cwd is not None else self._work_dir
        where.mkdir(parents=True, exist_ok=True)
        return await self._processes.run(
            ProcessCommand(argv=tuple(argv), cwd=where, env=dict(env or {}))
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Phase:
    attempt_id: str
    metrics: RuntimeMetrics
    processes: Processes

    def _require_sandbox[KindT: SandboxKind[Any]](
        self, sandbox: SandboxKind[Any] | None, kind: type[KindT], name: str
    ) -> KindT:
        where = f"{type(self).__name__}.{name} for attempt {self.attempt_id}"
        if sandbox is None:
            raise RuntimeError(f"{where} is unavailable in this phase")
        if not isinstance(sandbox, kind):
            raise TypeError(f"{where} requires {kind.__name__}, got {type(sandbox).__name__}")
        return sandbox


@dataclass(frozen=True, slots=True, kw_only=True)
class Setup(_Phase):
    """The phase record that ``setup()`` receives before sampling.

    It gives the execution sandbox and host services. It gives no model access.
    """

    sandbox: SandboxKind[Any] | None

    def require_sandbox[KindT: SandboxKind[Any]](self, kind: type[KindT]) -> KindT:
        """Return the execution sandbox as ``kind``, or raise if it is absent or another kind."""
        return self._require_sandbox(self.sandbox, kind, "sandbox")


@dataclass(frozen=True, slots=True, kw_only=True)
class Sampling(_Phase):
    """The phase record that ``run()`` receives, inside sampling capacity and the agent's deadline.

    Its model access ends when ``run()`` does.
    """

    model: Model
    sandbox: SandboxKind[Any] | None
    # The run's trajectory budget, from ``RunLimits.trajectory_tokens``. A call's
    # max_tokens is lowered to the room its prompt leaves, and a prompt that leaves no
    # room is refused. None means the run sets no budget.
    trajectory_tokens: int | None = None

    def require_sandbox[KindT: SandboxKind[Any]](self, kind: type[KindT]) -> KindT:
        """Return the execution sandbox as ``kind``, or raise if it is absent or another kind."""
        return self._require_sandbox(self.sandbox, kind, "sandbox")


@dataclass(frozen=True, slots=True, kw_only=True)
class Grading(_Phase):
    """The phase record that ``grade()`` receives for one grading try.

    It holds the sealed rollout and the sandboxes still available. It gives no policy
    model. ``timed_out`` is true when the agent's deadline cut ``run()``.
    """

    rollout: Rollout
    timed_out: bool
    execution_sandbox: SandboxKind[Any] | None
    grading_sandbox: SandboxKind[Any] | None

    def require_execution_sandbox[KindT: SandboxKind[Any]](self, kind: type[KindT]) -> KindT:
        """Return the execution sandbox as ``kind``, or raise if it is unavailable.

        It is unavailable when the episode sets ``sandbox_lifetime = "sampling"``.
        """
        return self._require_sandbox(self.execution_sandbox, kind, "execution_sandbox")

    def require_grading_sandbox[KindT: SandboxKind[Any]](self, kind: type[KindT]) -> KindT:
        """Return the separate grading sandbox as ``kind``, or raise if it is unavailable."""
        return self._require_sandbox(self.grading_sandbox, kind, "grading_sandbox")


@dataclass(frozen=True, slots=True, kw_only=True)
class Closing(_Phase):
    """The phase record that ``close()`` receives for final cleanup.

    It offers only the sandboxes that were initialized and are still accessible.
    """

    timed_out: bool
    execution_sandbox: SandboxKind[Any] | None
    grading_sandbox: SandboxKind[Any] | None

    def require_execution_sandbox[KindT: SandboxKind[Any]](self, kind: type[KindT]) -> KindT:
        """Return the execution sandbox as ``kind`` if it is still accessible, or raise."""
        return self._require_sandbox(self.execution_sandbox, kind, "execution_sandbox")

    def require_grading_sandbox[KindT: SandboxKind[Any]](self, kind: type[KindT]) -> KindT:
        """Return the grading sandbox as ``kind`` if it is still accessible, or raise."""
        return self._require_sandbox(self.grading_sandbox, kind, "grading_sandbox")


class Episode[TaskT: BaseModel | None, OptionsT: BaseModel | None]:
    """One attempt at a task, built from the validated task and options.

    Subclass it as ``Episode[MyTask, MyOptions]``, with ``None`` for a model you don't
    need. Declare ``execution_kind`` and ``grading_kind`` separately from those two
    types. Each hook receives an immutable phase record, and the rollout runtime keeps
    ownership of every resource. Initialize any state that ``grade()`` reads in
    ``__init__``, because the deadline can cut ``run()`` before it produces output.
    """

    execution_kind: ClassVar[type[SandboxKind[Any]] | None] = None
    grading_kind: ClassVar[type[SandboxKind[Any]] | None] = None
    sandbox_lifetime: ClassVar[Literal["attempt", "sampling"]] = "attempt"
    model_access: ClassVar[ModelAccessMode] = ModelAccessMode.DIRECT
    grading_retries: ClassVar[int] = 0
    _episode_models: ClassVar[tuple[object, object]] = (None, None)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for base in get_original_bases(cls):
            origin = get_origin(base)
            if isinstance(origin, type) and issubclass(origin, Episode):
                cls._episode_models = _substitute(
                    typing.cast(type[Episode[Any, Any]], origin), get_args(base)
                )
                return

    def __init__(self, task: Task[TaskT], options: OptionsT) -> None:
        self.task = task
        self.options = options

    async def setup(self, setup: Setup) -> None:
        """Prepare execution resources before the attempt holds sampling capacity."""

    async def run(self, sample: Sampling) -> Mapping[str, JsonValue] | None:
        """Run the agent, and return the metadata to record, or ``None``.

        The metadata may include partial output.
        """
        raise NotImplementedError

    async def grade(self, grading: Grading) -> float | GradeResult | None:
        """Score this attempt.

        ``grading.timed_out`` is true when the agent's deadline cut ``run()``.
        """
        raise NotImplementedError

    @classmethod
    async def grade_group(
        cls, members: Sequence[GroupMember[TaskT]], options: OptionsT
    ) -> Sequence[float]:
        """Score a complete group from its recorded evidence and the validated options only."""
        raise NotImplementedError

    async def close(self, closing: Closing) -> None:
        """Clean up once, after all of the attempt's work has stopped.

        The cookbook calls it after a failed setup or sampling, too.
        """


def _substitute(origin: type[Episode[Any, Any]], args: tuple[object, ...]) -> tuple[object, object]:
    if origin is Episode:
        found = args
    else:
        parameters = getattr(origin, "__parameters__", ())
        bound: dict[object, object] = dict(zip(parameters, args, strict=True))
        found = tuple(
            bound.get(model, model) if isinstance(model, typing.TypeVar) else model
            for model in origin._episode_models  # pyright: ignore[reportPrivateUsage]
        )
    task, options = (None if model is NoneType else model for model in found)
    return task, options


def episode_models(episode: type[Episode[Any, Any]]) -> tuple[object, object]:
    """Resolve the task and options type arguments, including those bound by generic bases."""

    task, options = (
        _default(model)
        for model in episode._episode_models  # pyright: ignore[reportPrivateUsage]
    )
    return task, options


def _default(model: object) -> object:
    if isinstance(model, typing.TypeVar) and getattr(model, "has_default", lambda: False)():
        return getattr(model, "__default__")  # noqa: B009
    return model
