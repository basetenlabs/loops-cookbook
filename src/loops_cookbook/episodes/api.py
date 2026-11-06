"""What an episode author writes, and what the cookbook hands it.

An ``Episode`` is one object per attempt. The cookbook builds it with the attempt's
``Task`` and validated options, awaits ``setup(setup)``, then ``run(sample)`` under the
agent deadline, then ``grade(grading)`` once the rollout is sealed, and finally
``close(closing)``, exactly once, however the attempt ended -- except when ``__init__``
raised or never ran, so there is no episode to close, or when ``setup()``,
``run()`` or ``grade()`` ignored cancellation past its grace, so the attempt is
kept live and reported instead. A retry is a fresh attempt: a new ``Episode``, a new sandbox, and
the agent runs again.

The policy is in scope only inside ``run()``: the ``Model`` is in its Sampling argument, and
refuses once ``run()`` has ended. ``grade()`` never gets the policy; an LLM judge
brings its own client.

An episode class may also score a whole group at once: the classmethod
``grade_group()`` gets each member's recorded evidence as a ``GroupMember`` once the
group is complete, with the episode's options, and gives every member its final
reward.
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
    """The one span in which the policy is in scope: ``run()``, under the agent deadline.

    When the deadline expires, ``run()`` is cancelled and the episode is still an
    episode: it is sealed with what was delivered, and graded. Once the deadline has
    cut ``run()``, how ``run()`` then ends is not the outcome, unless it returns.
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
        """``call()`` until it returns or the deadline cuts it; ``None`` if cut before returning."""
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
        self._timed_out = deadline.expired()  # run() caught the cancellation and returned
        return result


class Model:
    """The policy, for this attempt, only inside ``run()``.

    ``conversation()`` and ``llm()`` serve ``model_access = "direct"``; ``http_connection()`` serves
    ``"openai_http"``; each names an optional producer from ``[rollout.producers]``.
    Once ``run()`` has ended all refuse, and the attempt's own model access refuses
    whatever was taken from them: a direct client once the attempt is sealed, and an
    HTTP credential, which is revoked then.
    """

    def __init__(self, access: ModelAccess, turn: AgentTurn) -> None:
        self._access = access
        self._turn = turn

    def llm(self, producer: str | None = None) -> LoopsLLM:
        self._check()
        return self._access.llm(producer)

    def conversation(
        self, *, producer: str | None = None, conversation_id: str | None = None
    ) -> Conversation:
        """Start a direct generation session; supply history on each generate() call.

        The session records against this attempt and producer, and its model access
        ends when the attempt seals, just like a session taken through llm().
        """
        return self.llm(producer).conversation(conversation_id=conversation_id)

    async def http_connection(self, producer: str | None = None) -> ModelConnection:
        self._check()
        return await self._access.http_connection(producer)

    def _check(self) -> None:
        if not self._turn.open:
            raise ModelAccessError("model access for this attempt is sealed")


class SandboxAttachment:
    """The layer's side of an attempt's sandbox: its one client, closed after ``close()``.

    The client is attached on first use and the same object is given from then on.
    Once the layer has closed it, the sandbox gives no client again.
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
                # Kept in the same step as the call returns, so no cancellation can
                # land between them and leave a client nobody closes.
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
        """Close the client, if one was attached; the sandbox gives none after this."""
        async with self._lock:
            self._closed = True
            if self._attached:
                await self._resources.close_sandbox_client(self._resource_id, self._client)
                self._attached = False

    async def release(self, attempt_id: str) -> None:
        """Close the attachment, then require proof of the registered resource's release.

        A failure leaves final attempt cleanup and journal recovery responsible for
        retrying. Client access stays closed even when release cannot be proven.
        """
        await self.close()
        result = await self._resources.release(self._resource_id)
        report = CleanupReport(ResourceScope.EPISODE, attempt_id, (result,))
        if not report.complete:
            raise CleanupIncomplete(report)


@dataclass(frozen=True, slots=True)
class GroupMember[TaskT: BaseModel | None]:
    """Recorded evidence for one member of a complete group; never a live attempt."""

    task: Task[TaskT]
    transcript: SealedTranscript
    reward: float | None
    metadata: Mapping[str, JsonValue]


class Processes:
    """Supervised local processes, defaulting to this attempt's working directory."""

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
    """Before sampling: execution resources and host services, without model access."""

    sandbox: SandboxKind[Any] | None

    def require_sandbox[KindT: SandboxKind[Any]](self, kind: type[KindT]) -> KindT:
        """Return the execution sandbox as kind, or raise if absent or incompatible."""
        return self._require_sandbox(self.sandbox, kind, "sandbox")


@dataclass(frozen=True, slots=True, kw_only=True)
class Sampling(_Phase):
    """Inside sampling capacity and the agent deadline; model access ends with run()."""

    model: Model
    sandbox: SandboxKind[Any] | None
    # The run's trajectory budget (``RunLimits.trajectory_tokens``): a call's max_tokens
    # is lowered to the room its prompt leaves, and a prompt that leaves no room is
    # refused. None: no run-level budget.
    trajectory_tokens: int | None = None

    def require_sandbox[KindT: SandboxKind[Any]](self, kind: type[KindT]) -> KindT:
        """Return the execution sandbox as kind, or raise if absent or incompatible."""
        return self._require_sandbox(self.sandbox, kind, "sandbox")


@dataclass(frozen=True, slots=True, kw_only=True)
class Grading(_Phase):
    """Sealed rollout and resources available for this grading try; no policy model."""

    rollout: Rollout
    timed_out: bool
    execution_sandbox: SandboxKind[Any] | None
    grading_sandbox: SandboxKind[Any] | None

    def require_execution_sandbox[KindT: SandboxKind[Any]](self, kind: type[KindT]) -> KindT:
        """Return the execution sandbox; unavailable after a sampling-only lifetime."""
        return self._require_sandbox(self.execution_sandbox, kind, "execution_sandbox")

    def require_grading_sandbox[KindT: SandboxKind[Any]](self, kind: type[KindT]) -> KindT:
        """Return the separate grading sandbox as kind, or raise if unavailable."""
        return self._require_sandbox(self.grading_sandbox, kind, "grading_sandbox")


@dataclass(frozen=True, slots=True, kw_only=True)
class Closing(_Phase):
    """Final cleanup: only resources actually initialized and still accessible."""

    timed_out: bool
    execution_sandbox: SandboxKind[Any] | None
    grading_sandbox: SandboxKind[Any] | None

    def require_execution_sandbox[KindT: SandboxKind[Any]](self, kind: type[KindT]) -> KindT:
        """Return the execution sandbox when it was initialized and remains accessible."""
        return self._require_sandbox(self.execution_sandbox, kind, "execution_sandbox")

    def require_grading_sandbox[KindT: SandboxKind[Any]](self, kind: type[KindT]) -> KindT:
        """Return the grading sandbox when it was initialized and remains accessible."""
        return self._require_sandbox(self.grading_sandbox, kind, "grading_sandbox")


class Episode[TaskT: BaseModel | None, OptionsT: BaseModel | None]:
    """One attempt, constructed with validated task/options data.

    Declare execution_kind and grading_kind independently of the two data types.
    Fixed hooks receive immutable phase records; resources remain runtime-owned.
    Initialize grading state here because run() can be cut before producing output.
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
        """Prepare execution resources without holding sampling capacity."""

    async def run(self, sample: Sampling) -> Mapping[str, JsonValue] | None:
        """Run the agent; return recorded metadata or None, including partial output."""
        raise NotImplementedError

    async def grade(self, grading: Grading) -> float | GradeResult | None:
        """Score this attempt; grading.timed_out records an agent deadline cut."""
        raise NotImplementedError

    @classmethod
    async def grade_group(
        cls, members: Sequence[GroupMember[TaskT]], options: OptionsT
    ) -> Sequence[float]:
        """Score a complete group using recorded evidence and validated options only."""
        raise NotImplementedError

    async def close(self, closing: Closing) -> None:
        """Close once after a quiescent attempt, including failed setup or sampling."""


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
    """Resolve task/options arguments, including bound generic intermediate bases."""
    task, options = (
        _default(model)
        for model in episode._episode_models  # pyright: ignore[reportPrivateUsage]
    )
    return task, options


def _default(model: object) -> object:
    if isinstance(model, typing.TypeVar) and getattr(model, "has_default", lambda: False)():
        return getattr(model, "__default__")  # noqa: B009
    return model
