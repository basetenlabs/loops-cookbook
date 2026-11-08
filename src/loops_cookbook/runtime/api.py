"""Define the rollout runtime contract, which is the one place where users supply episodes.

The names and signatures here are normative. A rollout runtime implements
``RolloutRuntime``. A synchronous ``RuntimeFactory`` builds it from its opaque
configuration and a ``RuntimeContext``. The cookbook owns scheduling, model access,
capture, grading admission, the order of cleanup, and training.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, SecretStr

from loops_cookbook.evidence import AttemptIdentity, JsonValue, Rollout, TaskRef
from loops_cookbook.llm.transcript import SealedTranscript
from loops_cookbook.model_access.api import ModelAccess, ModelAccessMode
from loops_cookbook.resources.api import (
    CleanupResult,
    ProviderDeadlines,
    ResourceHandle,
    ResourceLease,
    ResourceRequest,
    SandboxCapacity,
    SandboxKind,
)


class Phase(str, Enum):
    """The kinds of capacity an attempt can hold: sampling, verification and provisioning.

    A rollout runtime holds sampling or verification itself only for a phase it declares
    in ``internally_acquired_phases``. Otherwise the cookbook holds sampling around all
    of ``sample()`` and verification around all of ``grade()``. Only the cookbook's
    resource service holds provisioning.
    """

    SAMPLING = "sampling"
    VERIFICATION = "verification"
    PROVISIONING = "provisioning"


class EpisodeDisposition(str, Enum):
    """How a sealed rollout ended, as ``finish()`` receives it.

    ``COMPLETED`` means ``grade()`` gave its result. ``DISCARDED`` means the cookbook
    dropped it before grading, for example as stale. ``FAILED`` means grading failed or
    never ran, as at shutdown.
    """

    COMPLETED = "completed"
    DISCARDED = "discarded"
    FAILED = "failed"


class AttemptAbortReason(str, Enum):
    """Why an attempt left no valid rollout, as ``abort()`` receives it."""

    SAMPLE_FAILED = "sample_failed"  # sample() raised, or the lease was poisoned.
    CAPTURE_FAILED = "capture_failed"  # Capture broke, or proved a future policy version.
    ROLLOUT_INVALID = "rollout_invalid"  # The sealed evidence failed rollout validation.
    ROLLOUT_TIMEOUT = "rollout_timeout"  # The rollout clock expired.
    SETUP_TIMEOUT = "setup_timeout"  # The setup clock expired.
    STALE = "stale"  # The lease raised a freshness or policy-delta error.
    CANCELLED = "cancelled"


class PermanentRolloutFailure(Exception):
    """Raise this when running the episode again cannot help.

    The cookbook discards the rollout at once, without rollout retries.
    """


class PermanentGradingFailure(Exception):
    """Raise this when grading the rollout again cannot help.

    The cookbook discards the rollout at once, without grading retries.
    """


class ExcludedReward(PermanentGradingFailure):
    """Raise this to leave a reward out of training on purpose.

    The cookbook discards the rollout, but never counts it toward the
    consecutive-failure stop, because excluding many rewards in a row is not an outage.
    """


class NamedFailure(Exception):
    """A failure that names itself in its discard metric, as in ``grading_failed/<name>``.

    The cookbook names any other failure by its type, snake-cased, so
    ``VerifierTimeoutError`` becomes ``verifier_timeout``. Give ``name`` in the same
    snake-cased form. Without a ``name``, the type names this failure too.
    """

    def __init__(self, *args: object, name: str | None = None) -> None:
        super().__init__(*args)
        self.name = name


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    """What a rollout runtime declares about itself, which the cookbook checks before any attempt.

    ``model_access_modes`` lists the model access modes it supports. With more than one,
    ``[rollout_runtime] model_access`` picks. ``internally_acquired_phases`` names the
    phases whose capacity it holds itself through the ``CapacityBroker``. It may not
    name provisioning.
    """

    model_access_modes: frozenset[ModelAccessMode]
    internally_acquired_phases: frozenset[Phase] = frozenset()
    # Extra tries after a failed grade() or group grade. The rollout runtime sets this,
    # because only it knows whether grading can safely run again. 0 means grading is
    # never retried.
    grading_retries: int = 0
    # When true, a complete group's rewards come from the rollout runtime's
    # grade_group(), so grade() may give none. The rollout runtime then implements
    # GroupGrader.
    pending_group_grade: bool = False
    # The rollout clock that the rollout runtime derives from its own agent deadline.
    # The run uses it when [rollout] rollout_timeout_seconds is unset. None means the
    # rollout runtime derives none, so the setting is required.
    rollout_timeout_seconds: float | None = None
    # Where the rollout runtime's model access and grading retries are stated, so that
    # check can report the source with each value. Examples are
    # "direct (MyEpisode.model_access)" and "4 (Harbor job retry)". None means the
    # rollout runtime reports only the values.
    model_access_source: str | None = None
    grading_retries_source: str | None = None


@dataclass(frozen=True, slots=True)
class AttemptSpec:
    """Which attempt this is, and which task it runs."""

    identity: AttemptIdentity
    task: TaskRef


class CapacityBroker(Protocol):
    """The attempt's capacity, for a rollout runtime that holds sampling or verification itself.

    Each call is bound to the attempt. It is allowed only for a phase declared in
    ``internally_acquired_phases``, and only while ``sample()`` or ``grade()`` runs.
    Any other call is refused.
    """

    def hold(self, phase: Phase) -> AbstractAsyncContextManager[None]:
        """Return a context manager that waits for a slot of ``phase`` and holds it until exit.

        An attempt's sampling holds must not overlap. Time inside a sampling hold,
        outside resource operations, runs the rollout clock. The first sampling hold
        also admits the attempt, if ``admit()`` has not.
        """
        ...

    async def admit(self) -> None:
        """Wait until the attempt is admitted to start, before it starts an environment to sample.

        Admission happens once per attempt, so a second call returns at once. Admitting
        here, rather than at the first sampling hold, lets backpressure stop
        environments from starting. It needs sampling in ``internally_acquired_phases``.
        """
        ...


class RuntimeMetrics(Protocol):
    """Report the rollout runtime's own numeric metrics, under the ``runtime/`` prefix.

    The cookbook reports these and never reads them back, so they cannot affect
    scheduling. A name must be nonempty and relative, and a value a finite number.
    """

    def set_gauge(self, name: str, value: float) -> None:
        """Set the gauge ``name`` to ``value``, replacing its last value."""
        ...

    def observe(self, name: str, value: float) -> None:
        """Record one observation of ``name``, which is reported as a count, mean and maximum."""
        ...


class EpisodeResources(Protocol):
    """One attempt's resources, which are provisioned for it and released with it."""

    async def provision(self, request: ResourceRequest) -> ResourceHandle:
        """Provision ``request`` as one of this attempt's own resources, and return its handle.

        The cookbook releases it with the attempt, unless ``release()`` already did. The
        time spent here counts toward neither of the attempt's clocks.
        """
        ...

    async def acquire(self, resource_id: str) -> ResourceLease:
        """Borrow a run resource by its ID, and return the lease.

        The lease ends when it is closed or when the attempt ends. An attempt cannot
        lease one of its own resources, or hold two leases on one resource.
        """
        ...

    async def release(self, resource_id: str) -> CleanupResult:
        """Release a resource early. Only RELEASED or ALREADY_ABSENT proves it gone."""
        ...

    def usage_environment(self, resource_id: str) -> Mapping[str, str | SecretStr]:
        """Return the environment a separate process needs to use a resource the attempt holds."""
        ...

    async def sandbox_client(self, resource_id: str) -> object:
        """Return the provider's own client for a sandbox whose adapter is a ``SandboxProvider``."""
        ...

    async def close_sandbox_client(self, resource_id: str, client: object) -> None:
        """Close a client that ``sandbox_client()`` returned for ``resource_id``."""
        ...

    async def sandbox_alive(self, resource_id: str) -> bool:
        """Return the provider's fact of whether a sandbox the attempt holds is still usable."""
        ...


class RunResources(Protocol):
    """Resources that outlive every episode of the run."""

    async def provision(self, request: ResourceRequest) -> ResourceHandle:
        """Provision ``request`` as a run resource, and return its handle.

        The run releases it at shutdown.
        """
        ...

    async def acquire(self, resource_id: str) -> ResourceLease:
        """Borrow a run resource by its ID for the run itself, and return the lease.

        The run holds at most one open lease on a resource.
        """
        ...


class RuntimeFiles(Protocol):
    """Resolve the file aliases that ``[rollout_runtime.files]`` declares."""

    def path(self, alias: str) -> Path:
        """Return the path that ``alias`` names.

        A relative path in the config resolves against the config directory. An alias
        that ``[rollout_runtime.files]`` does not declare is refused.
        """
        ...


class RuntimeEvidence(Protocol):
    """Record digests of what the rollout runtime made of its configuration, with the run.

    It takes only a ``sha256:`` digest, never a value, so nothing that a rollout runtime
    derives reaches disk. Each name is recorded once, while the rollout runtime is
    built. The run writes the digests beside ``config.json`` and into its checkpoints,
    and a resume compares them.
    """

    def record_digest(self, name: str, digest: str) -> None:
        """Record ``digest`` under ``name``.

        A repeated name, or anything but a ``sha256:`` digest, is refused.
        """
        ...


@dataclass(frozen=True, slots=True)
class ProcessCommand:
    """One process for ``AttemptProcesses.run()`` to start, with its arguments, cwd and environment.

    The process inherits only a short non-secret allowlist from the cookbook's
    environment, plus exactly ``env``. Pass a secret as a ``SecretStr`` in ``env``, never
    in ``argv``.
    """

    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str | SecretStr] = field(default_factory=dict[str, str | SecretStr])


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """How a supervised process ended, with bounded tails of its output and its secrets redacted."""

    exit_code: int  # Meaningful only when signal is None.
    signal: int | None  # The terminating signal number, if the process was signalled.
    stdout_tail: str
    stderr_tail: str
    elapsed_seconds: float


class AttemptProcesses(Protocol):
    """Run separate processes for one attempt, under the cookbook's supervision.

    The cookbook terminates the attempt's processes when its rollout is sealed, and any
    still running before ``finish()`` or ``abort()``.
    """

    async def run(self, command: ProcessCommand) -> ProcessResult:
        """Start ``command``, wait for its process group to exit, and return how it ended.

        A nonzero exit is returned, not raised. Cancelling the call terminates the group
        with ``SIGTERM``, then with ``SIGKILL`` after ``process_kill_grace_seconds``.
        """
        ...


@dataclass(frozen=True, slots=True)
class RunLimits:
    max_concurrent_rollouts: int
    max_concurrent_verifiers: int
    # [rollout] rollout_timeout_seconds as configured. It is None when the rollout
    # runtime is to derive it.
    rollout_timeout_seconds: float | None
    grading_timeout_seconds: float
    # How long a cancelled hook may take to stop before the run fails.
    cancellation_grace_seconds: float
    # How long a cancelled agent process may take to exit after SIGTERM before SIGKILL.
    process_kill_grace_seconds: float
    provider_deadlines: Mapping[str, ProviderDeadlines]
    # How many admitted rollouts may start their environment ahead of a free sampling slot.
    prestart_places: int = 0
    # The most tokens one trajectory holds, counting the prompt and completion of any
    # call. It is [sampling] max_trajectory_tokens, or else [model] max_seq_len. It is
    # None when neither is set.
    trajectory_tokens: int | None = None


type ImageLookup = Callable[[str], Awaitable[frozenset[str] | None]]


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """What a ``RuntimeFactory`` receives to build the rollout runtime, for the whole run.

    ``work_dir`` is the rollout runtime's own directory in the run's output.
    ``project_root`` is the directory the run started in, and ``config_dir`` is the
    directory of its config.
    """

    work_dir: Path
    project_root: Path
    config_dir: Path
    files: RuntimeFiles
    limits: RunLimits
    run_resources: RunResources
    metrics: RuntimeMetrics
    evidence: RuntimeEvidence
    # The sandbox kind each configured provider provides, for each whose adapter is a
    # SandboxProvider.
    sandbox_providers: Mapping[str, type[SandboxKind[object]]]
    # The region each configured provider creates in, for each whose adapter is a
    # RegionalProvider.
    provider_regions: Mapping[str, str] = field(default_factory=dict[str, str])
    # Default sandbox sizes and declared limits, for each provider whose adapter is a
    # CapacityProvider.
    provider_capacity: Mapping[str, SandboxCapacity] = field(
        default_factory=dict[str, SandboxCapacity]
    )
    # The validated [providers.<name>] table, for each provider whose adapter is a
    # ConfiguredProvider. Only that provider's own code reads it.
    provider_settings: Mapping[str, BaseModel] = field(default_factory=dict[str, BaseModel])
    # Each ImageProvider's image_tags, so a rollout runtime can check the images its
    # tasks name.
    provider_images: Mapping[str, ImageLookup] = field(default_factory=dict[str, "ImageLookup"])


@dataclass(frozen=True, slots=True)
class AttemptServices:
    """The cookbook's services for one attempt, from model access to processes and metrics."""

    model_access: ModelAccess
    capacity: CapacityBroker
    resources: EpisodeResources
    processes: AttemptProcesses
    metrics: RuntimeMetrics


@dataclass(frozen=True, slots=True)
class AttemptContext:
    """What every hook receives for one attempt, which is its spec and its services.

    The same context reaches each of the attempt's hooks.
    """

    spec: AttemptSpec
    services: AttemptServices


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """What ``sample()`` returns for an episode that happened.

    ``metadata`` is the rollout runtime's own report about the episode. The cookbook
    keeps it on the rollout verbatim and never interprets it.
    """

    metadata: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class GradeResult:
    """What ``grade()`` returns, which is the rollout's reward and metadata to add to the episode's.

    The metadata may not repeat a key that the episode's metadata holds.
    """

    # None means no reward yet. Only a rollout runtime that declares
    # pending_group_grade may return None.
    reward: float | None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class MemberEvidence:
    """One member of a complete group, as its group grader receives it.

    ``rollout.reward`` is what ``grade()`` gave it, or ``None``. ``transcript`` is its
    conversation, as recorded at the model boundary and sealed with its evidence.
    """

    rollout: Rollout
    transcript: SealedTranscript


class RolloutRuntime(Protocol):
    """The contract a rollout runtime implements to supply tasks and run and grade their episodes.

    The cookbook calls ``prepare()`` once, then reads ``capabilities``. For each attempt
    it calls ``sample()``, then ``grade()`` for a sealed rollout it keeps. It then calls
    ``finish()`` for a sealed rollout, or ``abort()`` for an attempt that left none.
    It calls ``close()`` once, at shutdown.
    """

    @property
    def capabilities(self) -> RuntimeCapabilities:
        """What this rollout runtime declares about itself.

        The cookbook reads it after ``prepare()``, and checks it before any attempt is
        admitted.
        """
        ...

    async def prepare(self) -> Sequence[TaskRef]:
        """Load the tasks, and return a ``TaskRef`` for each.

        The cookbook calls it once, before any attempt. The list must not be empty, and a
        task ID that repeats must keep its digest. A failure here stops the run before it
        starts.
        """
        ...

    async def sample(self, attempt: AttemptContext) -> EpisodeResult:
        """Run one episode. How it ends decides what happens next.

        - If it returns, an episode happened, however badly the agent did. The cookbook
          grades it.
        - If it raises, the infrastructure failed and the episode didn't happen. The
          cookbook retries it as a fresh attempt up to ``max_rollout_retries`` times.
          Then it discards the rollout as ``rollout_failed/<type>``, named by the
          exception's type. A ``CancelledError`` that the cookbook did not cause counts
          as such a failure, ``rollout_failed/cancelled``.
        - If it raises ``PermanentRolloutFailure``, the episode can never succeed. The
          cookbook discards it at once, without a retry.

        An agent that failed, crashed or ran out of its own time still had an episode.
        Return, and let ``grade()`` score what it left. An agent that never called the
        policy left nothing to train, and the cookbook discards its rollout as
        ``unversioned``.

        The one outcome that the cookbook decides itself is its rollout clock. When the
        clock cancels ``sample()``, there is no episode to grade. The cookbook discards
        the rollout as ``agent_timeout/rollout_clock``, does not retry it, and counts it
        as a sampling failure toward ``max_consecutive_rollout_failures``. The clock is a
        backstop, not the agent's deadline. A rollout runtime should give its own agent a
        deadline below ``rollout_timeout_seconds``. A slow agent then returns and is
        graded, rather than expiring the clock and counting toward the stop.

        The cookbook decides a failed model call, broken capture or stale attempt from
        its own evidence, whatever ``sample()`` raised or swallowed, and never retries
        it. It decides a completion that the renderer could not read the same way, but
        retries that one as a fresh attempt, because the model and sampler did their
        part.
        """
        ...

    async def grade(self, attempt: AttemptContext, rollout: Rollout) -> GradeResult:
        """Score one sealed rollout. How it ends decides what happens next.

        - If it returns a ``GradeResult``, that holds the rollout's reward.
        - If it raises, the grader failed this time. The cookbook grades the rollout
          again up to the capabilities' ``grading_retries`` times. Then it discards the
          rollout as ``grading_failed/<type>``. A ``CancelledError`` that the cookbook
          did not cause counts as such a failure, ``grading_failed/cancelled``.
        - If it raises ``PermanentGradingFailure``, grading this rollout again cannot
          help. The cookbook discards it at once, without a retry.

        A non-finite or non-numeric reward, or metadata that repeats the episode's keys,
        is a grader bug. It fails as ``PermanentGradingFailure`` does. A reward of
        ``None`` is allowed only with ``pending_group_grade``, because the group grader
        then gives the reward.
        """
        ...

    async def finish(
        self,
        attempt: AttemptContext,
        rollout: Rollout,
        disposition: EpisodeDisposition,
    ) -> None:
        """Clean up after a sealed rollout.

        Any failure, including a ``CancelledError`` that the cookbook did not cause, is a
        cleanup failure. The cookbook reports it and counts it toward cleanup failures.
        """
        ...

    async def abort(self, attempt: AttemptContext, reason: AttemptAbortReason) -> None:
        """Clean up an attempt that left no valid rollout.

        Its failures count in the same way as ``finish()``'s.
        """
        ...

    async def close(self) -> None:
        """Release what the rollout runtime itself holds. The cookbook calls it once, at shutdown.

        It runs under the cleanup deadline. A ``close()`` that does not stop within that
        deadline and its grace is reported, and the run does not end cleanly.
        """
        ...


class GroupGrader(Protocol):
    """What a rollout runtime that declares ``pending_group_grade`` implements beside its hooks."""

    async def grade_group(self, members: Sequence[MemberEvidence]) -> Sequence[float]:
        """Score a complete group at once, returning one final reward per member, in order.

        It runs once the group's members are sealed and graded, after its stale members
        are dropped, and only for a group still large enough to train.

        - If it returns the rewards, they replace whatever ``grade()`` gave.
        - If it raises, the grader failed this time. The cookbook tries again up to the
          capabilities' ``grading_retries`` times. Then it discards the group's members
          as ``grading_failed/<type>``.
        - If it raises ``PermanentGradingFailure``, the cookbook discards the members at
          once, without a retry.

        A result that is not one finite number per member is a grader bug. It fails as
        ``PermanentGradingFailure`` does.
        """
        ...


class RuntimeFactory(Protocol):
    def __call__(
        self, config: Mapping[str, JsonValue], *, context: RuntimeContext
    ) -> RolloutRuntime: ...
