"""``EpisodeRuntime``: a ``RolloutRuntime`` built from an ``Episode`` class.

It holds the attempt's phases itself, as the Harbor runtime does::

    admit()                       backpressure and the pre-start places apply here
      provision the sandbox       a resource operation: off both clocks, no sampling slot
      Episode(task, options)
      await episode.setup(Setup(...))       the setup clock; no sampling slot, no model
    hold(sampling):
      await episode.run(Sampling(...))    the rollout clock; cut at agent_timeout_seconds
    release

``grade()`` delegates to the attempt's ``Episode``, under the driver's staleness
check, verification slot, deadline and retries. ``finish()`` and ``abort()`` call
its ``close()`` once, under the cleanup deadline; the driver then releases the
sandbox with the rest of the attempt's resources.

An episode class that defines ``grade_group()`` makes the runtime a group grader:
it declares ``pending_group_grade``, and ``grade_group()`` receives each member's
evidence as plain data once the pipeline has a complete group to score.
"""

from __future__ import annotations

import inspect
import shutil
import typing
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import NoneType, UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import typing_extensions
from annotated_types import BaseMetadata, GroupedMetadata
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainValidator,
    ValidationError,
    WrapValidator,
)
from pydantic import JsonValue as PydanticJsonValue
from pydantic.fields import FieldInfo

from loops_cookbook.episodes.api import (
    AgentTurn,
    Closing,
    Episode,
    Grading,
    GroupMember,
    Model,
    Processes,
    Sampling,
    SandboxAttachment,
    Setup,
    episode_models,
)
from loops_cookbook.episodes.tasks import Task, load_tasks, task_digest, validated_dump
from loops_cookbook.evidence import JsonValue, Rollout, TaskRef, json_mapping
from loops_cookbook.resources.api import ResourceRequest, SandboxKind, SandboxSpec
from loops_cookbook.runtime.api import (
    AttemptAbortReason,
    AttemptContext,
    EpisodeDisposition,
    EpisodeResult,
    GradeResult,
    MemberEvidence,
    PermanentRolloutFailure,
    Phase,
    RuntimeCapabilities,
    RuntimeContext,
)
from loops_cookbook.runtime.context import RuntimeConfigurationError, import_attribute
from loops_cookbook.runtime.processes import GROUP_EXIT_SECONDS
from loops_cookbook.threads import in_thread

# Observed once per run(): 1 when the agent deadline cut it, else 0.
TIMED_OUT = "episodes/agent_timed_out"


class InvalidEpisodeMetadata(PermanentRolloutFailure):
    """``run()`` returned something other than a mapping or ``None``; it would every time."""


class _Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SandboxSettings(BaseModel):
    """``[runtime.config.sandbox]``: its provider, and the keys its kind's spec types."""

    # The keys beside provider are the provided kind's spec's to validate.
    model_config = ConfigDict(extra="allow", frozen=True)

    provider: str = Field(min_length=1, description="The [providers] entry that provides it.")

    def spec(self, kind: type[SandboxKind[Any]], *, table: str = "sandbox") -> SandboxSpec:
        """The keys beside provider, validated with ``kind``'s spec; one it does not
        declare is refused."""
        try:
            return kind.spec.model_validate(self.model_extra or {})
        except ValidationError as error:
            raise RuntimeConfigurationError(
                f"[runtime.config.{table}] is not a valid {kind.spec.__name__} for "
                f"[providers.{self.provider}]: {error}"
            ) from None

    def request(self, spec: SandboxSpec, *, logical_key: str = "sandbox") -> ResourceRequest:
        """Each attempt's sandbox, as ``spec`` asks its provider for it."""
        return ResourceRequest(
            provider=self.provider,
            kind=spec.resource_kind,
            logical_key=logical_key,
            specification=spec.specification(),
        )


class EpisodeSettings(_Settings):
    """``[runtime.config]`` for ``loops_cookbook.episodes:create_runtime``."""

    episode: str = Field(min_length=1, description="The Episode subclass, as module:Class.")
    agent_timeout_seconds: float = Field(
        gt=0, description="How long run() may take before it is cut and graded."
    )
    sandbox: SandboxSettings | None = Field(
        default=None, description="One sandbox per attempt, provisioned before setup()."
    )
    grading_sandbox: SandboxSettings | None = Field(
        default=None,
        description="Optional separate sandbox, provisioned on first grading admission.",
    )
    # Written to config.json with the rest of the config, so never a secret.
    options: dict[str, JsonValue] | None = Field(
        default=None,
        description="The episode's own non-secret settings, validated with its options model.",
    )


@dataclass(frozen=True, slots=True)
class SandboxPlan:
    """What each attempt's sandbox is provisioned from, and the kind the episode gets."""

    request: ResourceRequest
    kind: type[SandboxKind[Any]]


@dataclass
class _EpisodeState:
    episode: Episode[Any, Any]
    turn: AgentTurn
    processes: Processes
    execution_sandbox: SandboxKind[Any] | None
    grading_sandbox: SandboxKind[Any] | None = None


class EpisodeRuntime:
    def __init__(
        self,
        episode: type[Episode[Any, Any]],
        tasks: Sequence[Task[Any]],
        *,
        agent_timeout_seconds: float,
        rollout_timeout_seconds: float,
        sandbox: SandboxPlan | None,
        grading_sandbox: SandboxPlan | None = None,
        options: BaseModel | None,
        work_dir: Path,
        trajectory_tokens: int | None = None,
    ) -> None:
        self._episode = episode
        self._sandbox_lifetime = episode.sandbox_lifetime
        self._tasks = {task.id: task for task in tasks}
        self._agent_timeout = agent_timeout_seconds
        self._rollout_timeout = rollout_timeout_seconds
        self._sandbox = sandbox
        self._grading_sandbox = grading_sandbox
        self._options = options
        self._work_dir = work_dir
        self._trajectory_tokens = trajectory_tokens
        self._grades = defines(episode, "grade")
        # Each attempt's Episode, from its construction until its close().
        self._live: dict[str, _EpisodeState] = {}
        # Each attempt's sandbox client, from its sandbox's provision until after close().
        self._attachments: dict[str, SandboxAttachment] = {}
        self._grading_attachments: dict[str, SandboxAttachment] = {}
        self._grading_ready: set[str] = set()

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            model_access_modes=frozenset({self._episode.model_access}),
            # Without grade(), grading is a no-op that takes no verification slot:
            # the runtime acquires verification itself, and never needs to.
            internally_acquired_phases=frozenset(
                {Phase.SAMPLING} if self._grades else {Phase.SAMPLING, Phase.VERIFICATION}
            ),
            grading_retries=self._episode.grading_retries,
            pending_group_grade=defines(self._episode, "grade_group"),
            rollout_timeout_seconds=self._rollout_timeout,
            model_access_source=f"{getattr(self._episode.model_access, 'value', '?')} "
            f"({self._episode.__name__}.model_access)",
            grading_retries_source=f"{self._episode.grading_retries} "
            f"({self._episode.__name__}.grading_retries)",
        )

    async def prepare(self) -> list[TaskRef]:
        return [TaskRef(task.id, task_digest(task.data)) for task in self._tasks.values()]

    async def sample(self, attempt: AttemptContext) -> EpisodeResult:
        services = attempt.services
        await services.capacity.admit()
        sandbox = (
            None
            if self._sandbox is None
            else await self._provision(attempt, self._sandbox, self._attachments)
        )
        turn = AgentTurn(self._agent_timeout)
        episode = self._episode(self._tasks[attempt.spec.task.id].copy(), self._attempt_options())
        state = _EpisodeState(
            episode, turn, Processes(services.processes, self._attempt_dir(attempt)), sandbox
        )
        self._live[attempt.spec.identity.attempt_id] = state
        phase = self._phase_services(attempt, state)
        await episode.setup(Setup(**phase, sandbox=sandbox))
        model = Model(services.model_access, turn)
        async with services.capacity.hold(Phase.SAMPLING):
            metadata = await turn.run(
                lambda: episode.run(
                    Sampling(
                        **phase,
                        model=model,
                        sandbox=sandbox,
                        trajectory_tokens=self._trajectory_tokens,
                    )
                )
            )
        services.metrics.observe(TIMED_OUT, float(turn.timed_out))
        if metadata is not None and not isinstance(metadata, Mapping):
            raise InvalidEpisodeMetadata(
                f"run() returned a {type(metadata).__name__}, not episode metadata"
            )
        if self._sandbox_lifetime == "sampling":
            attachment = self._attachments.get(attempt.spec.identity.attempt_id)
            if attachment is not None:
                try:
                    await attachment.release(attempt.spec.identity.attempt_id)
                finally:
                    # Attachment access closes even if verified release fails.
                    state.execution_sandbox = None
        return EpisodeResult(metadata=metadata or {})

    async def grade(self, attempt: AttemptContext, rollout: Rollout) -> GradeResult:
        if not self._grades:  # only with grade_group(), which gives the reward
            return GradeResult(reward=None)
        attempt_id = attempt.spec.identity.attempt_id
        state = self._live[attempt_id]
        if self._grading_sandbox is not None and attempt_id not in self._grading_ready:
            sandbox = await self._provision(
                attempt, self._grading_sandbox, self._grading_attachments
            )
            state.grading_sandbox = sandbox
            self._grading_ready.add(attempt_id)
        result = await state.episode.grade(
            Grading(
                **self._phase_services(attempt, state),
                rollout=rollout,
                timed_out=state.turn.timed_out,
                execution_sandbox=state.execution_sandbox,
                grading_sandbox=state.grading_sandbox,
            )
        )
        # The driver refuses a reward that is not a finite number, as a grader bug,
        # and no reward at all unless grade_group() gives it.
        return result if isinstance(result, GradeResult) else GradeResult(reward=result)

    async def grade_group(self, members: Sequence[MemberEvidence]) -> Sequence[float]:
        group = [
            GroupMember(
                task=self._tasks[member.rollout.task.id].copy(),
                transcript=member.transcript,
                reward=member.rollout.reward,
                # A private copy: the trained rollout keeps its own, whatever the judge does.
                metadata=json_mapping(member.rollout.metadata, where="metadata"),
            )
            for member in members
        ]
        return await self._episode.grade_group(group, self._attempt_options())

    async def finish(
        self, attempt: AttemptContext, rollout: Rollout, disposition: EpisodeDisposition
    ) -> None:
        await self._close(attempt)

    async def abort(self, attempt: AttemptContext, reason: AttemptAbortReason) -> None:
        await self._close(attempt)

    async def close(self) -> None:
        return None

    async def _close(self, attempt: AttemptContext) -> None:
        """The attempt is over: its episode closes, then its sandbox's client, then its
        own directory goes; each after the one before, however that one ended."""
        attempt_id = attempt.spec.identity.attempt_id
        state = self._live.pop(attempt_id, None)
        attachment = self._attachments.pop(attempt_id, None)
        grading_attachment = self._grading_attachments.pop(attempt_id, None)
        self._grading_ready.discard(attempt_id)
        try:
            if state is not None:
                await state.episode.close(
                    Closing(
                        **self._phase_services(attempt, state),
                        timed_out=state.turn.timed_out,
                        execution_sandbox=state.execution_sandbox,
                        grading_sandbox=state.grading_sandbox,
                    )
                )
        finally:
            try:
                try:
                    if attachment is not None:
                        await attachment.close()
                finally:
                    if grading_attachment is not None:
                        await grading_attachment.close()
            finally:
                # Off the loop: a large tree (or a slow disk) would otherwise stall
                # every other rollout and clock while it goes.
                directory = self._attempt_dir(attempt)
                if directory.exists():
                    await in_thread(shutil.rmtree, directory)

    def _phase_services(self, attempt: AttemptContext, state: _EpisodeState) -> dict[str, Any]:
        return {
            "attempt_id": attempt.spec.identity.attempt_id,
            "metrics": attempt.services.metrics,
            "processes": state.processes,
        }

    def _attempt_dir(self, attempt: AttemptContext) -> Path:
        """Where the attempt's processes run by default; no other attempt shares it."""
        return self._work_dir / "attempts" / attempt.spec.identity.attempt_id

    async def _provision(
        self,
        attempt: AttemptContext,
        plan: SandboxPlan,
        attachments: dict[str, SandboxAttachment],
    ) -> SandboxKind[Any]:
        resources = attempt.services.resources
        handle = await resources.provision(plan.request)
        resource_id = handle.intent.resource_id
        attempt_id = attempt.spec.identity.attempt_id
        attachment = attachments.get(attempt_id)
        if attachment is None:
            attachment = SandboxAttachment(resource_id, resources)
            attachments[attempt_id] = attachment
        environment = resources.usage_environment(resource_id)
        return plan.kind(handle.resource.name, environment, attachment)

    def _attempt_options(self) -> BaseModel | None:
        """A private copy of the options, so no episode or group grader can change another's."""
        return None if self._options is None else self._options.model_copy(deep=True)


def create_runtime(config: Mapping[str, JsonValue], *, context: RuntimeContext) -> EpisodeRuntime:
    """The factory ``[runtime] factory = "loops_cookbook.episodes:create_runtime"`` names."""
    try:
        settings = EpisodeSettings.model_validate(config)
    except ValidationError as error:
        raise RuntimeConfigurationError(f"[runtime.config]: {error}") from None
    limits = context.limits
    # A cut run() returns once its processes have exited after SIGTERM, or been
    # killed after the kill grace and waited for, and then it has its own
    # cancellation grace.
    stopped = (
        settings.agent_timeout_seconds
        + limits.process_kill_grace_seconds
        + GROUP_EXIT_SECONDS
        + limits.cancellation_grace_seconds
    )
    configured = limits.rollout_timeout_seconds
    if configured is not None and configured < stopped:
        raise RuntimeConfigurationError(
            f"[rollout] rollout_timeout_seconds ({configured:g}s) is below "
            f"{stopped:g}s: [runtime.config] agent_timeout_seconds "
            f"({settings.agent_timeout_seconds:g}s), [rollout.cleanup] "
            f"process_kill_grace_seconds ({limits.process_kill_grace_seconds:g}s), the "
            f"{GROUP_EXIT_SECONDS:g}s a killed process group is waited for, and "
            f"cancellation_grace_seconds ({limits.cancellation_grace_seconds:g}s); it is the "
            "backstop that discards a run() the agent deadline could not stop. Leave it "
            "unset to use that sum"
        )
    sandbox = settings.sandbox
    grading_sandbox = settings.grading_sandbox
    for table, value in (("sandbox", sandbox), ("grading_sandbox", grading_sandbox)):
        if value is not None and value.provider not in limits.provider_deadlines:
            raise RuntimeConfigurationError(
                f"[runtime.config.{table}] provider {value.provider!r} is not configured in "
                f"[providers]; configured: {sorted(limits.provider_deadlines)}"
            )
    episode = episode_class(settings.episode)
    context.evidence.record_digest(
        "sandbox_lifetime", task_digest({"sandbox_lifetime": episode.sandbox_lifetime})
    )
    models = declared_models(episode, settings.episode)
    provided = _provided_sandbox(
        models.sandbox, settings.episode, sandbox, context.sandbox_providers
    )
    grading_provided = _provided_sandbox(
        models.grading_sandbox,
        settings.episode,
        grading_sandbox,
        context.sandbox_providers,
        table="grading_sandbox",
    )
    if grading_sandbox is not None and not defines(episode, "grade"):
        raise RuntimeConfigurationError(
            "[runtime.config.grading_sandbox] requires a per-attempt grade(); "
            "grade_group() receives recorded evidence only"
        )
    options = episode_options(models.options, settings.options, settings.episode)
    if options is not None:
        # What ran, defaults included, which config.json's table as written is not. A
        # digest only: a default may come from the environment, and must not reach disk.
        context.evidence.record_digest(
            "options", _validated_digest(options, "[runtime.config.options]")
        )
    plan = None
    if sandbox is not None and provided is not None:
        spec = sandbox.spec(provided)
        # As the options: what each sandbox is asked for, defaults included.
        context.evidence.record_digest(
            "sandbox", _validated_digest(spec, "[runtime.config.sandbox]")
        )
        plan = SandboxPlan(sandbox.request(spec), provided)
    grading_plan = None
    if grading_sandbox is not None and grading_provided is not None:
        spec = grading_sandbox.spec(grading_provided, table="grading_sandbox")
        context.evidence.record_digest(
            "grading_sandbox", _validated_digest(spec, "[runtime.config.grading_sandbox]")
        )
        grading_plan = SandboxPlan(
            grading_sandbox.request(spec, logical_key="grading_sandbox"), grading_provided
        )
    return EpisodeRuntime(
        episode,
        load_tasks(context.files.path("tasks"), models.task),
        agent_timeout_seconds=settings.agent_timeout_seconds,
        rollout_timeout_seconds=stopped if configured is None else configured,
        sandbox=plan,
        grading_sandbox=grading_plan,
        options=options,
        work_dir=context.work_dir,
        trajectory_tokens=limits.trajectory_tokens,
    )


def episode_options(
    model: type[BaseModel] | None, table: Mapping[str, JsonValue] | None, path: str
) -> BaseModel | None:
    """``[runtime.config.options]`` validated with the episode's options model.

    Options no model validates are refused rather than passed on unread.
    """
    if model is None:
        if table is not None:
            raise RuntimeConfigurationError(
                f"[runtime.config.options] is set, and episode {path!r} declares no options "
                "model to validate it"
            )
        return None
    try:
        return model.model_validate(dict(table or {}))
    except ValidationError as error:
        raise RuntimeConfigurationError(
            f"[runtime.config.options] is not a valid {model.__name__}: {error}"
        ) from None


def _validated_digest(value: BaseModel, where: str) -> str:
    """The digest of what ran: the validated model's dump."""
    return task_digest(validated_dump(value, where=where))


def _provided_sandbox(
    kind: type[SandboxKind[Any]] | None,
    path: str,
    sandbox: SandboxSettings | None,
    sandbox_providers: Mapping[str, type[SandboxKind[Any]]],
    *,
    table: str = "sandbox",
) -> type[SandboxKind[Any]] | None:
    """The kind of sandbox the table's provider gives the episode: one of the kind the
    episode is written for, from an adapter that can back it.

    Matched by type, never by name: whatever a provider is called in [providers], its
    adapter declares the kind it provides. The provided kind's spec is what validates
    the table, so it is the one the allow list checks.
    """
    if sandbox is None:
        if kind is not None:
            raise RuntimeConfigurationError(
                f"episode {path!r} is written for a {kind.provider!r} sandbox, and "
                f"[runtime.config] has no {table} table"
            )
        return None
    if kind is None:
        raise RuntimeConfigurationError(
            f"[runtime.config.{table}] provisions a sandbox for every attempt, and episode "
            f"{path!r} declares no sandbox kind to use it"
        )
    provided = sandbox_providers.get(sandbox.provider)
    if provided is None:
        raise RuntimeConfigurationError(
            f"the [providers.{sandbox.provider}] adapter is not a SandboxProvider (sandbox_kind, "
            "client, close_client, alive), so it cannot back an episode's sandbox"
        )
    if not issubclass(provided, kind):
        raise RuntimeConfigurationError(
            f"the [providers.{sandbox.provider}] adapter provides {provided.__name__} "
            f"sandboxes, and episode {path!r} is written for {kind.__name__}"
        )
    _declared_spec(provided, path)
    return provided


def _declared_spec(kind: type[SandboxKind[Any]], path: str) -> None:
    """Refuse ``kind``'s spec unless it is a ``SandboxSpec`` the allow list allows.

    The table it validates is written to config.json, and what it validated is run
    evidence, as the options are, so the one allow list holds for it too.
    """
    role = f"{kind.__name__} sandbox spec"
    if not (inspect.isclass(kind.spec) and issubclass(kind.spec, SandboxSpec)):
        raise RuntimeConfigurationError(
            f"episode {path!r}: {role} {getattr(kind.spec, '__name__', kind.spec)!r} is not a "
            "SandboxSpec"
        )
    _declared_model(kind.spec, role, path)


def defines(episode: type[Episode[Any, Any]], hook: str) -> bool:
    """Whether the class, or a base of its own, overrides ``Episode``'s ``hook``."""
    return inspect.getattr_static(episode, hook) is not inspect.getattr_static(Episode, hook)


def episode_class(path: str) -> type[Episode[Any, Any]]:
    """The ``Episode`` subclass ``module:Class`` names, defining the hooks it must, each async.

    ``run()`` is required; so is ``grade()``, unless ``grade_group()`` gives the
    rewards, and then it must be an async classmethod: it sees no live episode.
    """
    found = import_attribute(path, "episode")
    if not inspect.isclass(found) or not issubclass(found, Episode):
        raise RuntimeConfigurationError(f"episode {path!r} is not a subclass of Episode")
    required = ["run"] if defines(found, "grade_group") else ["run", "grade"]
    missing = [name for name in required if not defines(found, name)]
    if missing:
        raise RuntimeConfigurationError(f"episode {path!r} does not define {', '.join(missing)}()")
    for name in ("setup", "run", "grade", "close"):
        if not inspect.iscoroutinefunction(getattr(found, name)):
            raise RuntimeConfigurationError(
                f"episode {path!r}: its {name}() is not an async def, and the cookbook awaits it"
            )
        try:
            inspect.signature(getattr(found, name)).bind(None, None)
        except (TypeError, ValueError) as error:
            raise RuntimeConfigurationError(
                f"episode {path!r}: its {name}() must accept self and one phase argument: {error}"
            ) from None
    try:
        inspect.signature(found).bind(None, None)
    except (TypeError, ValueError) as error:
        raise RuntimeConfigurationError(
            f"episode {path!r}: its constructor must accept task and options: {error}"
        ) from None
    if defines(found, "grade_group"):
        grader = inspect.getattr_static(found, "grade_group")
        if not isinstance(grader, classmethod) or not inspect.iscoroutinefunction(
            grader.__func__  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        ):
            raise RuntimeConfigurationError(
                f"episode {path!r} must define grade_group() as an async classmethod"
            )
    if found.sandbox_lifetime not in ("attempt", "sampling"):
        raise RuntimeConfigurationError(
            f"episode {path!r}: sandbox_lifetime must be 'attempt' or 'sampling'"
        )
    # model_access and grading_retries are checked as the runtime's capabilities.
    return found


@dataclass(frozen=True, slots=True)
class EpisodeModels:
    """Validated task/options types and independently declared sandbox requirements."""

    task: type[BaseModel] | None
    options: type[BaseModel] | None
    sandbox: type[SandboxKind[Any]] | None
    grading_sandbox: type[SandboxKind[Any]] | None


def declared_models(episode: type[Episode[Any, Any]], path: str) -> EpisodeModels:
    """The models ``episode`` declares, each refused unless every model it reaches is allowed.

    One allow list holds for task and options models alike, and for a sandbox kind's
    spec (``_declared_spec``). A model forbids keys it
    does not declare, so a misspelt one is refused rather than dropped unread; and it
    dumps what it validated, since a task is digested by its dump and the run records
    its options' digest. So a model computes and serializes nothing of its own; each
    field is read and dumped by its own name and never excluded; and each field's
    type dumps as its value: a JSON scalar, a Literal or Enum of JSON scalars, a list
    or ``dict[str, ...]`` of allowed types, a union of them, a type alias of one, a
    recursive JSON type, or an allowed model.
    Anything else, a secret or an untyped dict among them, is refused by model and
    field. A task's ``id`` is the cookbook's, so no task model may declare one.
    """
    task_argument, options_argument = episode_models(episode)
    task, options = (
        _declared_model(argument, role, path)
        for argument, role in ((task_argument, "task model"), (options_argument, "options model"))
    )
    if task is not None and "id" in task.model_fields:
        raise RuntimeConfigurationError(
            f"episode {path!r}: task model {task.__name__} declares a field 'id'; a task's "
            "id names its line, and is read as task.id"
        )
    return EpisodeModels(
        task,
        options,
        _declared_sandbox(episode.execution_kind, path),
        _declared_sandbox(episode.grading_kind, path),
    )


def _declared_sandbox(argument: object, path: str) -> type[SandboxKind[Any]] | None:
    """The sandbox kind a class declaration names: a subclass of SandboxKind naming its provider."""
    if argument is None:
        return None
    if isinstance(argument, typing.TypeVar):
        raise RuntimeConfigurationError(
            f"episode {path!r} leaves its sandbox kind open: {argument} is a type parameter"
        )
    kind = get_origin(argument) or argument
    if not (inspect.isclass(kind) and issubclass(kind, SandboxKind)):
        raise RuntimeConfigurationError(
            f"episode {path!r}: its sandbox kind {argument!r} is not a SandboxKind"
        )
    provider = getattr(kind, "provider", None)
    if not isinstance(provider, str) or not provider:
        raise RuntimeConfigurationError(
            f"episode {path!r}: sandbox kind {kind.__name__} names no provider; a provider's "
            'kind sets provider, such as provider = "blaxel"'
        )
    return kind  # pyright: ignore[reportUnknownVariableType]


def _declared_model(argument: object, role: str, path: str) -> type[BaseModel] | None:
    if argument is None:
        return None
    if isinstance(argument, typing.TypeVar):
        raise RuntimeConfigurationError(
            f"episode {path!r} leaves its {role} open: {argument} is a type parameter"
        )
    if not (inspect.isclass(argument) and issubclass(argument, BaseModel)):
        raise RuntimeConfigurationError(
            f"episode {path!r}: its {role} {argument!r} is not a pydantic model"
        )
    refusal = _refusal(argument)
    if refusal is not None:
        raise RuntimeConfigurationError(
            f"episode {path!r}: {role} {argument.__name__} reaches {refusal}"
        )
    return argument


# The types a field may dump as its value, and the values a Literal may name.
_SCALARS: tuple[object, ...] = (str, int, float, bool, NoneType)
_LITERALS = (str, int, bool, NoneType)
# Pydantic's recursive JSON type, which dumps as the JSON it validated. It is a
# tagged union of forward references the walk does not read; the cookbook's
# JsonValue, and any JSON type alias of one's own, the walk reads as what it names.
_JSON_VALUES: tuple[object, ...] = (PydanticJsonValue,)
_ALIASES = (typing.TypeAliasType, typing_extensions.TypeAliasType)
# What Annotated may add to a type: constraints, validators and Field(), which change
# what validates and never what dumps.
_METADATA = (
    BaseMetadata,
    GroupedMetadata,
    FieldInfo,
    AfterValidator,
    BeforeValidator,
    PlainValidator,
    WrapValidator,
)


@dataclass(frozen=True, slots=True)
class _Refused:
    """Why a field is refused, as the words that follow its name."""

    why: str


def _refusal(root: type[BaseModel]) -> str | None:
    """The first model ``root`` reaches, itself included, that is not allowed, and why."""
    seen: set[type[BaseModel]] = set()
    pending = [root]
    while pending:
        model = pending.pop()
        if model in seen:
            continue
        seen.add(model)
        why = _model_refusal(model)
        if why is not None:
            return f"{model.__name__}, which {why}"
        for name, field in model.model_fields.items():
            for found in _field_reaches(field):
                if isinstance(found, _Refused):
                    return f"{model.__name__}, whose field {name!r} {found.why}"
                pending.append(found)
    return _serialization_refusal(root.__pydantic_core_schema__, root.__name__, None)


def _serialization_refusal(schema: object, model: str, field: str | None) -> str | None:
    """Where in ``schema`` a serialization of its own is set, if one is anywhere.

    Every way pydantic lets a model dump other than its values, a serializer
    decorator, an annotated serializer or a config's ``json_encoders``, is a
    ``serialization`` key in its core schema, a nested model's and a type
    definition's included. ``model`` and ``field`` name where the walk is.
    """
    parts: list[tuple[object, str | None]] = []
    if isinstance(schema, Mapping):
        entries = typing.cast(Mapping[str, object], schema)
        if entries.get("type") == "model":
            model, field = typing.cast(type[BaseModel], entries["cls"]).__name__, None
        if "serialization" in entries:
            where = (
                "which serializes itself" if field is None else f"whose field {field!r} serializes"
            )
            return f"{model}, {where} its own way"
        for key, item in entries.items():
            if key == "fields" and entries.get("type") == "model-fields":
                named = typing.cast(Mapping[str, object], item)
                parts.extend((part, name) for name, part in named.items())
            else:
                parts.append((item, field))
    elif isinstance(schema, list | tuple):
        parts.extend((item, field) for item in typing.cast(Sequence[object], schema))
    for part, at in parts:
        found = _serialization_refusal(part, model, at)
        if found is not None:
            return found
    return None


def _model_refusal(model: type[BaseModel]) -> str | None:
    if model.model_config.get("extra") != "forbid":
        return (
            "does not set model_config extra='forbid', so a key it does not declare "
            "would be dropped unread"
        )
    if model.model_computed_fields:
        return f"computes {', '.join(map(repr, model.model_computed_fields))}"
    return None


def _field_reaches(field: FieldInfo) -> Iterator[type[BaseModel] | _Refused]:
    """The models ``field`` reaches, and why it is refused if it is."""
    if field.exclude or field.exclude_if is not None:
        yield _Refused("is excluded from its dump")
    for alias in (field.alias, field.validation_alias, field.serialization_alias):
        if alias is not None:
            yield _Refused(f"is read or dumped as {alias!r}, not by its name")
    yield from _metadata_refusals(field.metadata)
    yield from _type_reaches(field.annotation)


def _type_reaches(
    annotation: object, aliases: frozenset[object] = frozenset()
) -> Iterator[type[BaseModel] | _Refused]:
    """The models ``annotation`` reaches, and a refusal for each part not allowed.

    A type alias is what it names; ``aliases`` are those already being checked, so
    a recursive one, such as a JSON type of one's own, is checked once.
    """
    origin, arguments = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        yield from _metadata_refusals(arguments[1:])
        yield from _type_reaches(arguments[0], aliases)
    elif origin in (Union, UnionType) or (origin is list and len(arguments) == 1):
        for argument in arguments:
            yield from _type_reaches(argument, aliases)
    elif origin is dict and arguments[:1] == (str,):
        yield from _type_reaches(arguments[1], aliases)
    elif origin is Literal:
        if not all(type(value) in _LITERALS for value in arguments):
            yield _not_allowed(annotation)
    elif annotation in _SCALARS or annotation in _JSON_VALUES or _is_scalar_enum(annotation):
        return
    elif isinstance(annotation, _ALIASES):
        if annotation not in aliases:
            yield from _type_reaches(annotation.__value__, aliases | {annotation})
    elif inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        yield annotation
    else:
        yield _not_allowed(annotation)


def _metadata_refusals(metadata: Iterable[object]) -> Iterator[_Refused]:
    for item in metadata:
        if not isinstance(item, _METADATA):
            yield _Refused(
                f"is annotated with {type(item).__name__}, which is not a constraint, a "
                "validator or Field()"
            )


def _is_scalar_enum(annotation: object) -> bool:
    """An Enum whose members dump as JSON scalars, each naming one member."""
    return (
        inspect.isclass(annotation)
        and issubclass(annotation, Enum)
        and all(type(member.value) in _SCALARS for member in annotation)
    )


def _not_allowed(part: object) -> _Refused:
    shown = repr(part) if get_origin(part) is not None else getattr(part, "__name__", repr(part))
    return _Refused(
        f"is typed with {shown}, which is not a JSON scalar, a Literal or Enum of them, a "
        "list or dict[str, ...] of allowed types, a union of them, or a pydantic model"
    )
