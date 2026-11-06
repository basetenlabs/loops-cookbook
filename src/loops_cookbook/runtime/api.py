"""The rollout runtime contract: the one seam through which users supply episodes.

Names and signatures here are normative. A runtime implements ``RolloutRuntime``
and is built by a synchronous ``RuntimeFactory`` from its opaque configuration and
a ``RuntimeContext``; the cookbook owns scheduling, model access, capture, grading
admission, cleanup ordering, and training.
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
    SAMPLING = "sampling"
    VERIFICATION = "verification"
    PROVISIONING = "provisioning"


class EpisodeDisposition(str, Enum):
    COMPLETED = "completed"
    DISCARDED = "discarded"
    FAILED = "failed"


class AttemptAbortReason(str, Enum):
    SAMPLE_FAILED = "sample_failed"  # sample() raised, or the lease was poisoned
    CAPTURE_FAILED = "capture_failed"  # capture broke, or proved a future policy version
    ROLLOUT_INVALID = "rollout_invalid"  # sealed evidence failed rollout validation
    ROLLOUT_TIMEOUT = "rollout_timeout"  # the rollout clock expired
    SETUP_TIMEOUT = "setup_timeout"  # the setup clock expired
    STALE = "stale"  # the lease raised a freshness or policy-delta error
    CANCELLED = "cancelled"


class PermanentRolloutFailure(Exception):
    """Re-running this episode cannot help; bypasses rollout retries."""


class PermanentGradingFailure(Exception):
    """Grading this rollout again cannot help; bypasses grading retries."""


class ExcludedReward(PermanentGradingFailure):
    """A reward left out of training on purpose: discarded, but never counted toward the
    consecutive-failure stop, since excluding many in a row is not an outage."""


class NamedFailure(Exception):
    """A failure that names itself in its discard metric, as in ``grading_failed/<name>``.

    Any other failure is named by its type. ``name`` is snake-cased like a type name,
    so ``VerifierTimeoutError`` is ``verifier_timeout``; without one, the type names it.
    """

    def __init__(self, *args: object, name: str | None = None) -> None:
        super().__init__(*args)
        self.name = name


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    model_access_modes: frozenset[ModelAccessMode]
    internally_acquired_phases: frozenset[Phase] = frozenset()
    # Extra tries after a failed grade() or group grade: the runtime's to say, since
    # only it knows whether grading can safely run again. 0: never retried.
    grading_retries: int = 0
    # A complete group's rewards come from the runtime's grade_group(), so grade()
    # may give none; the runtime implements GroupGrader.
    pending_group_grade: bool = False
    # The rollout clock the runtime derives from its own agent deadline: the run uses
    # it when [rollout] rollout_timeout_seconds is unset. None: the runtime derives
    # none, and the setting is required.
    rollout_timeout_seconds: float | None = None
    # Where the runtime's model access and grading retries are stated, for check to
    # report with each: "direct (MyEpisode.model_access)", "4 (Harbor job retry)".
    # None: the runtime says nothing more than the values.
    model_access_source: str | None = None
    grading_retries_source: str | None = None


@dataclass(frozen=True, slots=True)
class AttemptSpec:
    identity: AttemptIdentity
    task: TaskRef


class CapacityBroker(Protocol):
    def hold(self, phase: Phase) -> AbstractAsyncContextManager[None]: ...
    async def admit(self) -> None: ...


class RuntimeMetrics(Protocol):
    def set_gauge(self, name: str, value: float) -> None: ...
    def observe(self, name: str, value: float) -> None: ...


class EpisodeResources(Protocol):
    """One attempt's resources: provisioned for it, released with it."""

    async def provision(self, request: ResourceRequest) -> ResourceHandle: ...
    async def acquire(self, resource_id: str) -> ResourceLease: ...
    async def release(self, resource_id: str) -> CleanupResult:
        """Release early; only RELEASED or ALREADY_ABSENT proves it gone."""
        ...

    def usage_environment(self, resource_id: str) -> Mapping[str, str | SecretStr]: ...
    async def sandbox_client(self, resource_id: str) -> object:
        """The provider's own client for a sandbox, when its adapter is a ``SandboxProvider``."""
        ...

    async def close_sandbox_client(self, resource_id: str, client: object) -> None: ...
    async def sandbox_alive(self, resource_id: str) -> bool: ...


class RunResources(Protocol):
    """Resources that outlive every episode of the run."""

    async def provision(self, request: ResourceRequest) -> ResourceHandle: ...
    async def acquire(self, resource_id: str) -> ResourceLease: ...


class RuntimeFiles(Protocol):
    def path(self, alias: str) -> Path: ...


class RuntimeEvidence(Protocol):
    """Digests of what the runtime made of its configuration, recorded with the run.

    Only a ``sha256:`` digest, never a value, so nothing a runtime derives reaches
    disk. Each name is recorded once, while the runtime is built; the run writes the
    digests beside ``config.json`` and into its checkpoints, and a resume compares them.
    """

    def record_digest(self, name: str, digest: str) -> None: ...


@dataclass(frozen=True, slots=True)
class ProcessCommand:
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str | SecretStr] = field(default_factory=dict[str, str | SecretStr])


@dataclass(frozen=True, slots=True)
class ProcessResult:
    exit_code: int  # meaningful only when signal is None
    signal: int | None  # terminating signal number, if the process was signalled
    stdout_tail: str
    stderr_tail: str
    elapsed_seconds: float


class AttemptProcesses(Protocol):
    async def run(self, command: ProcessCommand) -> ProcessResult: ...


@dataclass(frozen=True, slots=True)
class RunLimits:
    max_concurrent_rollouts: int
    max_concurrent_verifiers: int
    # [rollout] rollout_timeout_seconds as configured; None when the runtime is to
    # derive it.
    rollout_timeout_seconds: float | None
    grading_timeout_seconds: float
    # How long a cancelled hook may take to stop before the run fails.
    cancellation_grace_seconds: float
    # How long a cancelled agent process may take to exit after SIGTERM before SIGKILL.
    process_kill_grace_seconds: float
    provider_deadlines: Mapping[str, ProviderDeadlines]
    # Admitted rollouts that may start their environment ahead of a free sampling slot.
    prestart_places: int = 0
    # The most tokens one trajectory holds, prompt and completion of any call:
    # [sampling] max_trajectory_tokens, else [model] max_seq_len; None when neither is set.
    trajectory_tokens: int | None = None


type ImageLookup = Callable[[str], Awaitable[frozenset[str] | None]]


@dataclass(frozen=True, slots=True)
class RuntimeContext:
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
    # ConfiguredProvider; read only by that provider's own code.
    provider_settings: Mapping[str, BaseModel] = field(default_factory=dict[str, BaseModel])
    # Each ImageProvider's image_tags, so a runtime can check the images its tasks name.
    provider_images: Mapping[str, ImageLookup] = field(default_factory=dict[str, "ImageLookup"])


@dataclass(frozen=True, slots=True)
class AttemptServices:
    model_access: ModelAccess
    capacity: CapacityBroker
    resources: EpisodeResources
    processes: AttemptProcesses
    metrics: RuntimeMetrics


@dataclass(frozen=True, slots=True)
class AttemptContext:
    spec: AttemptSpec
    services: AttemptServices


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    metadata: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class GradeResult:
    # None, no reward yet, only from a runtime that declares pending_group_grade.
    reward: float | None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class MemberEvidence:
    """One member of a complete group, as its group grader receives it.

    ``rollout.reward`` is what ``grade()`` gave it, or ``None``; ``transcript`` is its
    conversation as recorded at the model boundary and sealed with its evidence.
    """

    rollout: Rollout
    transcript: SealedTranscript


class RolloutRuntime(Protocol):
    @property
    def capabilities(self) -> RuntimeCapabilities: ...
    async def prepare(self) -> Sequence[TaskRef]: ...
    async def sample(self, attempt: AttemptContext) -> EpisodeResult:
        """Run one episode; how it ends decides what happens next.

        - Returns: an episode happened, however badly the agent did. Graded.
        - Raises: infrastructure, the episode didn't happen. Retried as a fresh
          attempt up to ``max_rollout_retries``, then discarded as
          ``rollout_failed/<type>``, named by the exception's type. A
          ``CancelledError`` the cookbook did not cause is one: ``rollout_failed/cancelled``.
        - Raises ``PermanentRolloutFailure``: this can never succeed. Discarded now,
          not retried.

        An agent that failed, crashed or ran out of its own time is an episode:
        return, and let ``grade()`` score what it left. One that never called the
        policy left nothing to train, and is discarded as ``unversioned``. The one outcome the cookbook
        decides itself is its rollout clock: when that cancels ``sample()`` there is
        no episode to grade, and the rollout is discarded as
        ``agent_timeout/rollout_clock``, not retried, and counted as a sampling
        failure toward ``max_consecutive_rollout_failures``. The clock is a backstop,
        not the agent's deadline: a runtime bounds its own agent below
        ``rollout_timeout_seconds``, so that a slow agent returns and is graded
        rather than expiring the clock and counting toward the stop. A failed model
        call, broken capture or stale attempt is decided by the cookbook's own
        evidence, whatever ``sample()`` raised or swallowed, and is never retried. A
        completion the renderer could not read is decided the same way, but retried
        as a fresh attempt: the model and sampler did their part.
        """
        ...

    async def grade(self, attempt: AttemptContext, rollout: Rollout) -> GradeResult:
        """Score one sealed rollout; how it ends decides what happens next.

        - Returns a ``GradeResult``: the rollout's reward.
        - Raises: the grader failed this time. Graded again up to the capabilities'
          ``grading_retries`` times, then
          discarded as ``grading_failed/<type>``. A ``CancelledError`` the cookbook
          did not cause is one: ``grading_failed/cancelled``.
        - Raises ``PermanentGradingFailure``: grading this rollout again cannot help.
          Discarded now, not retried.

        A non-finite or non-numeric reward, or metadata that repeats the episode's
        keys, is a grader bug, and fails as ``PermanentGradingFailure`` does. A reward
        of ``None`` is allowed only with ``pending_group_grade``: the group grader
        gives it.
        """
        ...

    async def finish(
        self,
        attempt: AttemptContext,
        rollout: Rollout,
        disposition: EpisodeDisposition,
    ) -> None:
        """Clean up after a sealed rollout. A failure, a ``CancelledError`` the cookbook
        did not cause included, is a cleanup failure: reported, and counted toward cleanup."""
        ...

    async def abort(self, attempt: AttemptContext, reason: AttemptAbortReason) -> None:
        """Clean up an attempt that left no valid rollout; its failures count as ``finish()``'s do."""
        ...

    async def close(self) -> None: ...


class GroupGrader(Protocol):
    """What a runtime that declares ``pending_group_grade`` implements beside its hooks."""

    async def grade_group(self, members: Sequence[MemberEvidence]) -> Sequence[float]:
        """Score a complete group at once: one final reward per member, in order.

        It runs once the group's members are sealed and graded, after its stale
        members are dropped, and only for a group still large enough to train.
        - Returns the rewards; they replace whatever ``grade()`` gave.
        - Raises: the grader failed this time. Tried again up to the capabilities'
          ``grading_retries`` times; then the
          group's members are discarded as ``grading_failed/<type>``.
        - Raises ``PermanentGradingFailure``: discarded now, not retried.

        A result that is not one finite number per member is a grader bug, and fails
        as ``PermanentGradingFailure`` does.
        """
        ...


class RuntimeFactory(Protocol):
    def __call__(
        self, config: Mapping[str, JsonValue], *, context: RuntimeContext
    ) -> RolloutRuntime: ...
