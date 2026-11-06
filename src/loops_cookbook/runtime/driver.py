"""The authoritative in-process attempt lifecycle.

For every admitted rollout the driver owns this sequence::

    live attempt
      -> attempt identity, phase state, model-access lease
      -> sample()            rollout and setup clocks, bounded quiescence
      -> seal the lease      settle, drop undelivered generations, reject a failed capture
      -> Rollout(reward=None)
      -> grade() tries       per-try deadline, opt-in retries, staleness rechecks
      -> commit              the pipeline receives the graded rollout, and its
                             transcript when a group grader will score its group
      -> finish()            cleanup only, in the background, under its deadline
    attempt settled

When no valid rollout exists, ``abort()`` runs instead of ``finish()``, and an
eligible failure is re-dispatched as a fresh attempt under the same rollout ID.
A runtime hook that does not stop within the cancellation grace after it is
cancelled runs in this process and cannot be killed, so its attempt is kept live.
A kept ``sample()`` or ``grade()`` discards its rollout as ``unquiesced/<hook>``, keeps
what it holds until it ends and is refused anything more; a kept ``finish()`` or
``abort()`` is a cleanup failure. More kept hooks running than 5% of
``max_concurrent_rollouts``, or kept hooks holding every slot of a phase or every
admission place, is a ``SystemicFailure``.

A sampling failure makes the driver ask whether the Loops run itself has ended, at
most once per ``run_status_interval_seconds``. Only that answer, never the failure,
stops the run at once.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import itertools
import logging
import math
import numbers
import re
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, NoReturn
from uuid import uuid4

from loops_cookbook.cancellation import Stopped, Waited, run_until, see_through
from loops_cookbook.evidence import (
    AttemptIdentity,
    InternalPolicyDeltaError,
    Rollout,
    StaleRolloutError,
    TaskRef,
    TrajectoryError,
    json_mapping,
)
from loops_cookbook.llm.rendering import RendererFailure
from loops_cookbook.llm.transcript import SealedTranscript
from loops_cookbook.model_access.api import (
    AttemptModelAccess,
    CaptureIntegrityFailure,
    FuturePolicyVersion,
    ModelAccessError,
    ModelTransportFailure,
    SamplingFailure,
    UndeclaredProducer,
)
from loops_cookbook.model_access.lease import AttemptLease
from loops_cookbook.resources.api import CleanupIncomplete
from loops_cookbook.resources.services import EpisodeResourceService
from loops_cookbook.runtime.api import (
    AttemptAbortReason,
    AttemptContext,
    AttemptServices,
    AttemptSpec,
    EpisodeDisposition,
    EpisodeResult,
    ExcludedReward,
    GradeResult,
    NamedFailure,
    PermanentGradingFailure,
    PermanentRolloutFailure,
    Phase,
    RolloutRuntime,
    RuntimeCapabilities,
    RuntimeMetrics,
)
from loops_cookbook.runtime.capacity import (
    AttemptPhases,
    Hook,
    RunCapacity,
    RuntimeContractError,
)
from loops_cookbook.runtime.processes import AttemptProcessSupervisor, ProcessKillFailure

logger = logging.getLogger(__name__)


class SystemicFailure(RuntimeError):
    """A contract violation or repeated infrastructure failures prevent safe training."""


class LoopsRunEnded(SystemicFailure):
    """The Loops run ended under the runtime: a spot reclaim, a failed or stopped
    trainer, a deactivation. ``ended`` is what Loops reported."""

    def __init__(self, ended: str) -> None:
        super().__init__(
            f"the Loops run has ended, so the run stops: {ended}. Every sampling "
            "request would fail from now on. Start a new run that resumes from the "
            "latest checkpoint manifest"
        )
        self.ended = ended


class NonQuiescentHook(SystemicFailure):
    """A runtime hook kept running after cancellation; its episode holds everything it had.

    ``waited`` is the stop's wait for the hook. A cancellation of the waiter during it
    is kept, and ``reraise()`` raises it in place of this report, with the report as
    its cause, once the caller has kept the attempt.
    """

    def __init__(self, message: str, waited: Waited) -> None:
        super().__init__(message)
        self.waited = waited

    def reraise(self) -> NoReturn:
        self.waited.propagate(self)
        raise self


class RuntimeExecutionFailure(RuntimeError):
    """A runtime hook raised an exception the cookbook has no type for."""

    def __init__(self, phase: str, identity: AttemptIdentity | None, error: BaseException):
        self.phase = phase
        self.identity = identity
        where = f" for {identity.rollout_id} attempt {identity.attempt_id}" if identity else ""
        detail = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
        super().__init__(f"runtime {phase}(){where} failed: {detail}")


class _RolloutClockExpired(TimeoutError):
    """The cookbook's rollout clock cancelled sample(): the one outcome the cookbook decides."""


def sub_reason(error: BaseException) -> str:
    """What a failure was, in its discard metric: ``<discard>/<sub_reason>``.

    A failure's type names it, in snake case without its ``Error``, ``Exception`` or
    ``Failure`` suffix, so ``SandboxLost`` is ``sandbox_lost``; a ``NamedFailure``
    with a name is named by it. A failure that only wraps or marks another, such as
    ``PermanentRolloutFailure``, is named by its cause.
    """
    if isinstance(error, _RolloutClockExpired):
        return "rollout_clock"
    if isinstance(error, NamedFailure) and error.name:
        return _snake(error.name)
    if isinstance(error, _WRAPPERS) and error.__cause__ is not None:
        return sub_reason(error.__cause__)
    return _snake(type(error).__name__)


def _group_grading_cause(error: BaseException) -> str:
    """The ``cause`` of the members of a group its grader failed."""
    return f"{Discard.GRADING_FAILED.value}/{sub_reason(error)}"


_WRAPPERS = (RuntimeExecutionFailure, PermanentRolloutFailure, PermanentGradingFailure)


def _snake(name: str) -> str:
    name = re.sub(r"(?<=.)(Error|Exception|Failure)$", "", name)
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


class HookDeadline(TimeoutError):
    """The cookbook cancelled a runtime hook at its deadline."""

    def __init__(self, reason: AttemptAbortReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class Discard(str, Enum):
    INTERNAL_DELTA = "internal_delta"
    STALE_BEFORE_GRADING = "stale_before_grading"
    STALE_BEFORE_TRAINING = "stale_before_training"
    ROLLOUT_FAILED = "rollout_failed"
    AGENT_TIMEOUT = "agent_timeout"
    SETUP_TIMEOUT = "setup_timeout"
    GRADING_FAILED = "grading_failed"
    UNVERSIONED = "unversioned"
    GROUP_TOO_SMALL = "group_too_small"
    NO_VARIANCE = "no_variance"
    NOTHING_TO_TRAIN = "nothing_to_train"  # every member with an advantage discarded its turns
    CANCELLED = "cancelled"
    TRAINING_ABORTED = "training_aborted"
    UNQUIESCED = "unquiesced"  # its sample() or grade() ignored cancellation past its grace


# Typed failures keep their own class; anything else from a hook is wrapped once.
_TYPED = (
    FuturePolicyVersion,
    ModelAccessError,
    PermanentRolloutFailure,
    PermanentGradingFailure,
    CaptureIntegrityFailure,
    SamplingFailure,
    ModelTransportFailure,
    StaleRolloutError,
    InternalPolicyDeltaError,
    RuntimeContractError,
)
# Rollout retry is for episode failures; the Loops SDK has already retried every
# model-call failure it treats as transient.
_NEVER_RETRIED = (
    PermanentRolloutFailure,
    ModelAccessError,
    CaptureIntegrityFailure,
    SamplingFailure,
    ModelTransportFailure,
    StaleRolloutError,
    InternalPolicyDeltaError,
)


@dataclass(frozen=True, slots=True)
class DriverSettings:
    run_id: str
    rollout_timeout_seconds: float
    setup_timeout_seconds: float
    max_rollout_retries: int
    grading_timeout_seconds: float
    cleanup_timeout_seconds: float
    cancellation_grace_seconds: float
    process_kill_grace_seconds: float
    max_steps_off_policy: int
    max_policy_delta_in_trajectory: int
    max_consecutive_failures: int
    # Least time between two questions of whether the Loops run has ended: a dead run
    # fails every sampling request, and one answer serves them all.
    run_status_interval_seconds: float = 45.0

    def __post_init__(self) -> None:
        for name in (
            "rollout_timeout_seconds",
            "setup_timeout_seconds",
            "grading_timeout_seconds",
            "cleanup_timeout_seconds",
            "cancellation_grace_seconds",
            "process_kill_grace_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_rollout_retries < 0:
            raise ValueError("the rollout retry budget must be nonnegative")
        if (
            not math.isfinite(self.run_status_interval_seconds)
            or self.run_status_interval_seconds < 0
        ):
            raise ValueError("run_status_interval_seconds must be finite and nonnegative")


class FailureBudget:
    """Consecutive failures per stage; reaching the limit is systemic.

    Each stage's failures are its infrastructure's, except a group grader's, which
    are the judge's own: an outage or a bad key fails every group alike.
    """

    FAILURES: ClassVar[Mapping[str, str]] = {
        "sampling": "sampling infrastructure failures",
        "grading": "grading infrastructure failures",
        "group_grading": "group-grading failures",
        "cleanup": "cleanup infrastructure failures",
    }

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._counts: dict[str, int] = dict.fromkeys(self.FAILURES, 0)

    def success(self, stage: str) -> None:
        self._counts[stage] = 0

    def failure(self, stage: str, error: BaseException) -> SystemicFailure | None:
        self._counts[stage] += 1
        count = self._counts[stage]
        if self._limit and count >= self._limit:
            return SystemicFailure(f"{count} consecutive {self.FAILURES[stage]}: {error}")
        return None


class FailureLog:
    """One traceback per distinct fault; every other occurrence is one line.

    An outage fails every rollout in flight for the same reason, and at high
    concurrency that is a hundred identical tracebacks burying the single line that
    says what broke. ``--verbose`` keeps every stack.
    """

    def __init__(self) -> None:
        self._seen: set[tuple[str, str]] = set()

    def report(self, stage: str, rollout_id: str, error: BaseException) -> None:
        kind = (stage, type(error).__name__)
        first = kind not in self._seen
        self._seen.add(kind)
        detail = f"{kind[1]}: {error}" if str(error) else kind[1]
        logger.error(
            "%s failed: %s: %s",
            stage,
            rollout_id,
            detail,
            exc_info=error if first or logger.isEnabledFor(logging.DEBUG) else None,
        )


@dataclass(slots=True)
class _Attempt:
    context: AttemptContext
    phases: AttemptPhases
    lease: AttemptLease
    access: AttemptModelAccess
    resources: EpisodeResourceService
    processes: AttemptProcessSupervisor
    retained: bool = False  # a hook ignored cancellation; the attempt stays live
    recorded: bool = False  # its rollout's record is written, so its note was taken


@dataclass(slots=True)
class _Kept:
    """A hook kept running past its grace, and what its attempt is owed once it ends."""

    attempt: _Attempt
    # The cleanup hook its attempt still needs: none for a kept finish() or abort(),
    # whose attempt ends with it.
    cleanup: tuple[str, Callable[[], Coroutine[Any, Any, None]]] | None
    settles: bool = False  # it settles its rollout, not a later attempt
    ended: list[Callable[[], None]] = field(default_factory=list)  # called once it ends


@dataclass(slots=True)
class Admitted:
    """A sealed rollout awaiting grading; it stays live until its cleanup settles."""

    attempt: _Attempt
    rollout: Rollout
    # Recorded when a group grader will read it, or the run keeps transcripts.
    transcript: SealedTranscript | None = None
    _settled: bool = field(default=False, repr=False)


@dataclass(slots=True)
class RolloutNote:
    retries: int = 0  # attempts retried, sampling and grading
    # ``<discard>`` or ``<discard>/<sub_reason>``; ``cleanup_failed/<sub_reason>``
    # when its abort() or finish() failed and nothing before it did.
    cause: str | None = None


@dataclass(frozen=True, slots=True)
class _Failed:
    reason: AttemptAbortReason
    error: BaseException
    discard: Discard
    retryable: bool
    counts: bool  # toward the sampling stage's consecutive-failure stop

    @property
    def cause(self) -> str:
        return f"{self.discard.value}/{sub_reason(self.error)}"


async def _hook[T](phase: str, identity: AttemptIdentity, call: Awaitable[T]) -> T:
    """Await one runtime hook, wrapping an exception the cookbook has no type for."""
    try:
        return await call
    except _TYPED:
        raise
    except Exception as error:
        raise RuntimeExecutionFailure(phase, identity, error) from error


_CLEANUP = "loops_cleanup_failures"
_CLEANUP_FAILED = "cleanup_failed"


def cleanup_failures(error: BaseException) -> tuple[BaseException, ...]:
    """The ``finish()`` or ``abort()`` failures that followed ``error``, in order.

    A cleanup failure never replaces the failure that caused the cleanup; it is
    recorded on that failure, whatever its type, and as a note on it.
    """
    found: object = getattr(error, _CLEANUP, ())
    return found if isinstance(found, tuple) else ()  # pyright: ignore[reportUnknownVariableType]


def _stopped_by(still_running: NonQuiescentHook) -> AttemptAbortReason:
    """Why a kept hook was stopped: its deadline's reason, or a cancellation."""
    waited = still_running.waited
    if isinstance(waited, Stopped) and isinstance(waited.expiry, AttemptAbortReason):
        return waited.expiry
    return AttemptAbortReason.CANCELLED


def _attach_cleanup(primary: BaseException, failure: BaseException) -> None:
    primary.add_note(f"cleanup then failed: {type(failure).__name__}: {failure}")
    setattr(primary, _CLEANUP, (*cleanup_failures(primary), failure))


def _abort_reason(failure: BaseException) -> AttemptAbortReason:
    """A lost generation aborts as a sampling failure unless capture itself broke."""
    if isinstance(failure, CaptureIntegrityFailure):
        return AttemptAbortReason.CAPTURE_FAILED
    return AttemptAbortReason.SAMPLE_FAILED


LeaseFactory = Callable[[AttemptIdentity], AttemptLease]
AccessFactory = Callable[[AttemptLease], AttemptModelAccess]
# An attempt's own resource services, given the phases whose clocks exclude them.
ResourcesFactory = Callable[[AttemptIdentity, AttemptPhases], EpisodeResourceService]


class AttemptDriver:
    def __init__(
        self,
        runtime: RolloutRuntime,
        capabilities: RuntimeCapabilities,
        settings: DriverSettings,
        *,
        capacity: RunCapacity,
        leases: LeaseFactory,
        access: AccessFactory,
        resources: ResourcesFactory,
        metrics: RuntimeMetrics,
        policy_version: Callable[[], int],
        on_systemic: Callable[[SystemicFailure], None],
        run_ended: Callable[[], Awaitable[str | None]],
        on_sealed: Callable[[], None] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._runtime = runtime
        self._capabilities = capabilities
        self._settings = settings
        self._capacity = capacity
        self._leases = leases
        self._access = access
        self._resources = resources
        self._metrics = metrics
        self._policy_version = policy_version
        self._on_systemic = on_systemic
        self._on_sealed = on_sealed
        self._run_ended = run_ended
        self._status_check: asyncio.Task[None] | None = None
        self._status_asked: float | None = None
        self._sleep = sleep
        self._live = 0
        self._budget = FailureBudget(settings.max_consecutive_failures)
        self._log = FailureLog()
        self._attempts: dict[str, _Attempt] = {}
        self._finishing: set[asyncio.Task[None]] = set()
        self._unquiesced: list[str] = []
        # Hooks kept running past their grace, by attempt ID; each leaves once it ends
        self._kept: dict[str, _Kept] = {}
        # Hook tasks still running past their grace: the loop holds tasks only weakly,
        # so one waiting on a private future would be collected, unwinding its holds.
        self._still_running: set[asyncio.Task[Any]] = set()
        self._systemic: SystemicFailure | None = None
        # By ``<discard>/<sub_reason>``: rollouts discarded, and attempts and group
        # grader tries retried.
        self._discarded: dict[str, int] = {}
        self._retried: dict[str, int] = {}
        self._retried_groups: dict[str, int] = {}
        # Per rollout until its record is written: attempts retried, and why it ended.
        self._notes: dict[str, RolloutNote] = {}

    # ----------------------------------------------------------------- public

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return self._capabilities

    @property
    def live_attempts(self) -> int:
        """Attempts from dispatch until their cleanup settles, including grading waits."""
        return self._live

    @property
    def unquiesced(self) -> tuple[str, ...]:
        """Hooks that ignored cancellation; each leaves its attempt live."""
        return tuple(self._unquiesced)

    def after_kept(self, rollout_id: str, callback: Callable[[], None]) -> bool:
        """Call ``callback`` once the hook of ``rollout_id`` kept past its grace ends.

        False, and ``callback`` is not called, when no hook of it is kept.
        """
        for kept in self._kept.values():
            if kept.attempt.phases.identity.rollout_id == rollout_id:
                kept.ended.append(callback)
                return True
        return False

    def gauges(self) -> dict[str, float]:
        """Discarded rollouts, and retried attempts and groups, by reason and sub-reason."""
        return {
            **{f"discard/rollouts/{key}": float(n) for key, n in self._discarded.items()},
            **{f"retried/attempts/{key}": float(n) for key, n in self._retried.items()},
            **{f"retried/groups/{key}": float(n) for key, n in self._retried_groups.items()},
        }

    def note(self, rollout_id: str) -> RolloutNote:
        """What the driver saw of a rollout, once: its retries and its discard's cause.

        Its record is written with it, so a cleanup of the rollout's attempt that fails
        after this notes nothing: no record would read it.
        """
        # A kept attempt is no longer in ``_attempts``, and its cleanup is still to come.
        kept = (entry.attempt for entry in self._kept.values())
        for attempt in itertools.chain(self._attempts.values(), kept):
            if attempt.phases.identity.rollout_id == rollout_id:
                attempt.recorded = True
        return self._notes.pop(rollout_id, RolloutNote())

    # The group stage's reports, one per way a group grader try ends.

    def group_scored(self) -> None:
        """A group grader scored a group: the ``group_grading`` count starts again."""
        self._budget.success("group_grading")

    def group_retrying(self, group_id: str, error: Exception) -> None:
        """A group grader try failed and is tried again; tallied, never counted."""
        self._log.report("group_grading", group_id, error)
        _tally(self._retried_groups, _group_grading_cause(error))

    def group_failed(self, group_id: str, error: Exception, rollout_ids: Sequence[str]) -> None:
        """A group's last grader try failed: its members are discarded, and it counts.

        Each member's noted cause is ``grading_failed/<type>``. A failure that reaches
        the ``group_grading`` limit stops the run.
        """
        self._log.report("group_grading", group_id, error)
        cause = _group_grading_cause(error)
        for rollout_id in rollout_ids:
            self._note(rollout_id).cause = cause
        self._discarded[cause] = self._discarded.get(cause, 0) + len(rollout_ids)
        self._count("group_grading", error)

    def group_unquiesced(self, group_id: str, error: NonQuiescentHook) -> None:
        """A group grader call still running after cancellation stops the run unsettled."""
        self._unquiesced.append(f"grade_group:{group_id}")
        self._systemic_error(error)

    async def back_off(self, number: int) -> None:
        """The wait before grading try ``number + 1``: 0.5s, doubling, at most 8s.

        ``grade()`` and the group stage both back off here, on the driver's ``sleep``.
        """
        await self._sleep(min(0.5 * 2 ** (number - 1), 8.0))

    def stale(self, rollout: Rollout) -> bool:
        oldest = rollout.minimum_policy_version
        return (
            oldest is None or self._policy_version() - oldest > self._settings.max_steps_off_policy
        )

    async def sample(self, task: TaskRef, group_id: str, rollout_id: str) -> Admitted | Discard:
        """Run attempts for one rollout until one seals, or the rollout is discarded."""
        self._raise_if_stopped()
        self._live += 1
        keep_live = False
        attempt: _Attempt | None = None
        try:
            retries = self._settings.max_rollout_retries
            while True:
                attempt = self._open(task, group_id, rollout_id)
                outcome = await self._sample_attempt(attempt)
                # A finished attempt's outcome is its rollout's even once the run has
                # stopped, by its own failure or another's: a sealed rollout still
                # reaches finish(), a failed one keeps its cause and is not retried.
                self._count_sampling(outcome)
                if isinstance(outcome, Admitted):
                    keep_live = True
                    return outcome
                if isinstance(outcome, Discard):
                    if outcome is not Discard.UNQUIESCED:  # noted with its hook when kept
                        self._note(rollout_id).cause = outcome.value
                    return outcome
                if outcome.retryable and retries > 0 and self._systemic is None:
                    retries -= 1
                    self._note(rollout_id).retries += 1
                    _tally(self._retried, outcome.cause)
                    logger.info(
                        "Retrying %s after %s (%d retr%s left)",
                        rollout_id,
                        outcome.cause,
                        retries,
                        "y" if retries == 1 else "ies",
                    )
                    continue
                self._discard(rollout_id, outcome.cause)
                return outcome.discard
        finally:
            if not keep_live:
                if attempt is not None and attempt.phases.identity.attempt_id in self._kept:
                    self._kept[attempt.phases.identity.attempt_id].settles = True
                else:
                    self._settle()

    async def grade(self, admitted: Admitted) -> Rollout | Discard:
        """Grade a sealed rollout; the committed rollout is returned before ``finish()``."""
        attempt, rollout = admitted.attempt, admitted.rollout
        tries = 1 + self._capabilities.grading_retries
        try:
            for number in range(1, tries + 1):
                if self.stale(rollout):
                    self._finalize(admitted, rollout, EpisodeDisposition.DISCARDED)
                    self._note(
                        rollout.identity.rollout_id
                    ).cause = Discard.STALE_BEFORE_GRADING.value
                    return Discard.STALE_BEFORE_GRADING
                try:
                    result = await self._grade_try(attempt, rollout)
                except (asyncio.CancelledError, NonQuiescentHook):
                    raise
                except SystemicFailure as error:
                    return self._grading_stopped(admitted, rollout, _origin(error))
                except Exception as error:  # noqa: BLE001 - every failure is an outcome
                    if (violation := attempt.phases.contract_error) is not None:
                        self._log.report("grading", rollout.identity.rollout_id, error)
                        return self._grading_stopped(admitted, rollout, violation)
                    retryable = not isinstance(
                        error, (PermanentGradingFailure, CaptureIntegrityFailure, ModelAccessError)
                    )
                    cause = f"{Discard.GRADING_FAILED.value}/{sub_reason(error)}"
                    if number < tries and retryable:
                        _tally(self._retried, cause)
                        self._note(rollout.identity.rollout_id).retries += 1
                        self._log.report("grading", rollout.identity.rollout_id, error)
                        await self.back_off(number)
                        continue
                    excluded = isinstance(error, ExcludedReward)
                    # Reported once its finish() has run, so a cleanup failure shows on it.
                    self._finalize(
                        admitted,
                        rollout,
                        EpisodeDisposition.FAILED,
                        error,
                        report=None if excluded else "grading",
                    )
                    self._discard(rollout.identity.rollout_id, cause)
                    if not excluded:
                        self._count("grading", error)
                    return Discard.GRADING_FAILED
                self._budget.success("grading")
                graded = Rollout(
                    identity=rollout.identity,
                    task=rollout.task,
                    samples=rollout.samples,
                    reward=None if result.reward is None else float(result.reward),
                    metadata={**rollout.metadata, **result.metadata},
                    discarded_versions=rollout.discarded_versions,
                )
                self._finalize(admitted, graded, EpisodeDisposition.COMPLETED)
                return graded
            raise AssertionError("unreachable")
        except NonQuiescentHook as still_running:
            # grade() is still running: no finish(), and the attempt stays live.
            self._retain(attempt, "grade", release_holds=False)
            admitted._settled = True  # pyright: ignore[reportPrivateUsage]
            finish = functools.partial(
                self._runtime.finish, attempt.context, rollout, EpisodeDisposition.FAILED
            )
            self._keep(attempt, still_running, ("finish", finish), settles=True)
            self._discard(rollout.identity.rollout_id, f"{Discard.UNQUIESCED.value}/grade")
            return Discard.UNQUIESCED
        except asyncio.CancelledError:
            if not admitted._settled:  # pyright: ignore[reportPrivateUsage]
                await self._finish_now(admitted, rollout, EpisodeDisposition.FAILED)
            raise

    def _grading_stopped(
        self, admitted: Admitted, rollout: Rollout, error: BaseException
    ) -> Discard:
        """The attempt's own violation stops the run; it is still its rollout's outcome."""
        self._systemic_error(error)
        self._finalize(admitted, rollout, EpisodeDisposition.FAILED)
        self._discard(
            rollout.identity.rollout_id, f"{Discard.GRADING_FAILED.value}/{sub_reason(error)}"
        )
        return Discard.GRADING_FAILED

    def discard_admitted(self, admitted: Admitted) -> None:
        """A sealed rollout that will never be graded, such as at shutdown."""
        self._finalize(admitted, admitted.rollout, EpisodeDisposition.FAILED)

    async def drain(self) -> None:
        """Wait for background finalization; the caller bounds how long.

        A run-status check in flight is only advice, so it is cancelled, not awaited.
        """
        if self._status_check is not None:
            self._status_check.cancel()
            await asyncio.gather(self._status_check, return_exceptions=True)
        while self._finishing:
            finishing = list(self._finishing)
            await asyncio.gather(*finishing, return_exceptions=True)
            # Awaiting tasks that are already done does not yield, so their own
            # done callbacks may not have run yet: they are dropped here.
            self._finishing.difference_update(finishing)

    # --------------------------------------------------------------- sampling

    def _open(self, task: TaskRef, group_id: str, rollout_id: str) -> _Attempt:
        identity = AttemptIdentity(
            run_id=self._settings.run_id,
            group_id=group_id,
            rollout_id=rollout_id,
            attempt_id=uuid4().hex,
        )
        phases = AttemptPhases(identity, self._capabilities.internally_acquired_phases)
        phases.on_violation = self._violation
        capacity = self._capacity.track(phases)
        lease = self._leases(identity)
        access = self._access(lease)
        resources = self._resources(identity, phases)
        processes = AttemptProcessSupervisor(self._settings.process_kill_grace_seconds)
        context = AttemptContext(
            spec=AttemptSpec(identity=identity, task=task),
            services=AttemptServices(
                model_access=access,
                capacity=capacity,
                resources=resources,
                processes=processes,
                metrics=self._metrics,
            ),
        )
        attempt = _Attempt(context, phases, lease, access, resources, processes)
        self._attempts[identity.attempt_id] = attempt
        return attempt

    def _note(self, rollout_id: str) -> RolloutNote:
        return self._notes.setdefault(rollout_id, RolloutNote())

    def _discard(self, rollout_id: str, cause: str) -> None:
        _tally(self._discarded, cause)
        self._note(rollout_id).cause = cause

    def _close(self, attempt: _Attempt) -> None:
        self._capacity.forget(attempt.phases)
        self._attempts.pop(attempt.phases.identity.attempt_id, None)

    async def _sample_attempt(self, attempt: _Attempt) -> Admitted | Discard | _Failed:
        identity = attempt.phases.identity
        try:
            result = await self._run_sample(attempt)
        except asyncio.CancelledError:
            attempt.lease.revoke()
            await self._abort(attempt, AttemptAbortReason.CANCELLED)
            raise
        except NonQuiescentHook as still_running:
            # sample() is still running: no abort(), and nothing it holds is released.
            attempt.lease.revoke()
            self._retain(attempt, "sample", release_holds=False)
            reason = _stopped_by(still_running)
            abort = functools.partial(self._runtime.abort, attempt.context, reason)
            try:
                self._keep(attempt, still_running, ("abort", abort))
                if (fatal := attempt.lease.fatal) is not None:
                    self._systemic_error(fatal)  # fatal to the run, kept or not
                self._discard(
                    attempt.phases.identity.rollout_id, f"{Discard.UNQUIESCED.value}/sample"
                )
            finally:
                # As far as it got: sample() still runs, but model access has ended. Not
                # left to the cleanup, which waits for a hook that may never end.
                await attempt.lease.keep_unsealed()
            return Discard.UNQUIESCED
        except SystemicFailure as error:
            attempt.lease.revoke()
            await self._abort(attempt, AttemptAbortReason.SAMPLE_FAILED)
            return self._sampling_stopped(AttemptAbortReason.SAMPLE_FAILED, _origin(error))
        except BaseException as error:  # noqa: BLE001 - classified; not a failure is re-raised
            attempt.lease.revoke()
            if (fatal := attempt.lease.fatal) is not None:
                reason = (
                    AttemptAbortReason.SAMPLE_FAILED
                    if isinstance(fatal, UndeclaredProducer)
                    else AttemptAbortReason.CAPTURE_FAILED
                )
                await self._abort(attempt, reason)
                return self._sampling_stopped(reason, fatal)
            if (violation := attempt.phases.contract_error) is not None:
                # The runtime broke its contract, whatever sample() then raised.
                await self._abort(attempt, AttemptAbortReason.SAMPLE_FAILED)
                return self._sampling_stopped(AttemptAbortReason.SAMPLE_FAILED, violation)
            failure = self._classify_sample_failure(attempt, error)
            if failure.discard is Discard.STALE_BEFORE_GRADING or failure.discard is (
                Discard.INTERNAL_DELTA
            ):
                await self._abort(attempt, AttemptAbortReason.STALE)
                return failure.discard
            await self._abort(attempt, failure.reason, failure.error)
            if not isinstance(error, HookDeadline):
                self._log.report("sampling", identity.rollout_id, failure.error)
            return failure
        return await self._seal(attempt, result)

    async def _run_sample(self, attempt: _Attempt) -> EpisodeResult:
        phases = attempt.phases
        runner_owns = Phase.SAMPLING not in phases.declared

        async def sample() -> EpisodeResult:
            return await _hook("sample", phases.identity, self._runtime.sample(attempt.context))

        async def body() -> EpisodeResult:
            if runner_owns:
                async with self._capacity.runner_hold(phases, Phase.SAMPLING):
                    return await sample()
            return await sample()

        def budget() -> tuple[float, AttemptAbortReason] | None:
            candidates: list[tuple[float, AttemptAbortReason]] = []
            if phases.rollout_running:
                candidates.append(
                    (
                        self._settings.rollout_timeout_seconds - phases.rollout_elapsed(),
                        AttemptAbortReason.ROLLOUT_TIMEOUT,
                    )
                )
            if phases.outside_holds:
                candidates.append(
                    (
                        self._settings.setup_timeout_seconds - phases.setup_elapsed(),
                        AttemptAbortReason.SETUP_TIMEOUT,
                    )
                )
            return min(candidates, key=lambda item: item[0], default=None)

        phases.start_hook(Hook.SAMPLE)
        try:
            # A stopped attempt loses model access at once, not after its grace, so
            # an agent outside sample()'s own tasks cannot keep sampling.
            result = await self._supervise(
                body(),
                "sample",
                phases,
                budget,
                on_stop=attempt.lease.revoke,
                processes=attempt.processes,
            )
        except NonQuiescentHook:
            phases.end_hook()
            raise
        except BaseException:
            phases.end_hook()
            _release_leftovers(phases)
            raise
        phases.end_hook()
        self._check_hook_return(phases, "sample")
        if not isinstance(result, EpisodeResult):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise self._systemic_error(
                RuntimeContractError(f"sample() returned {type(result).__name__}")
            )
        if not phases.sampling_entered:
            raise self._systemic_error(
                RuntimeContractError("sample() returned without entering a sampling hold")
            )
        return result

    def _sampling_stopped(self, reason: AttemptAbortReason, error: BaseException) -> _Failed:
        """The attempt's own violation stops the run; it is still its rollout's outcome."""
        self._systemic_error(error)
        return _Failed(reason, error, Discard.ROLLOUT_FAILED, False, False)

    def _classify_sample_failure(self, attempt: _Attempt, error: BaseException) -> _Failed:
        lease = attempt.lease
        # A stale attempt is discarded whether or not the runtime propagated it.
        stale = lease.stale or (
            error if isinstance(error, (StaleRolloutError, InternalPolicyDeltaError)) else None
        )
        if stale is not None:
            discard = (
                Discard.INTERNAL_DELTA
                if isinstance(stale, InternalPolicyDeltaError)
                else Discard.STALE_BEFORE_GRADING
            )
            return _Failed(AttemptAbortReason.STALE, stale, discard, False, False)
        if isinstance(error, HookDeadline):
            if error.reason is AttemptAbortReason.ROLLOUT_TIMEOUT:
                # A sampling failure all the same: the clock fires when sample() outlived
                # every deadline of its own, and a run of expiries is systemic.
                timeout = _RolloutClockExpired("the rollout clock expired")
                return _Failed(error.reason, timeout, Discard.AGENT_TIMEOUT, False, True)
            return _Failed(error.reason, error, Discard.SETUP_TIMEOUT, True, True)
        if not isinstance(error, Exception):
            raise error
        if (failure := lease.failure) is not None:
            # A lost generation's own cause decides the outcome, whatever the runtime
            # raised or swallowed. Only a renderer's failure is retried as a rollout:
            # the model and sampler did their part, and a fresh attempt samples anew.
            retry = isinstance(failure, RendererFailure)
            return _Failed(_abort_reason(failure), failure, Discard.ROLLOUT_FAILED, retry, True)
        return _Failed(
            AttemptAbortReason.SAMPLE_FAILED,
            error,
            Discard.ROLLOUT_FAILED,
            not isinstance(error, _NEVER_RETRIED),
            True,
        )

    async def _seal(self, attempt: _Attempt, result: EpisodeResult) -> Admitted | Discard | _Failed:
        phases, lease = attempt.phases, attempt.lease
        identity = phases.identity
        try:
            # The episode is over: every process it started ends; grading starts its own.
            await attempt.processes.terminate()
            # A transport's handlers may still be finishing a response the agent has read;
            # they get the grace to settle, and the seal keeps only what was delivered.
            await attempt.access.quiesce(self._settings.cancellation_grace_seconds)
            sealed = await lease.seal()
        except asyncio.CancelledError:
            await self._abort(attempt, AttemptAbortReason.CANCELLED)
            raise
        except ProcessKillFailure as error:
            # A group that outlives SIGKILL may still be acting for the episode.
            await self._abort(attempt, AttemptAbortReason.SAMPLE_FAILED, error)
            self._log.report("sampling", identity.rollout_id, error)
            return _Failed(
                AttemptAbortReason.SAMPLE_FAILED, error, Discard.ROLLOUT_FAILED, False, True
            )
        except FuturePolicyVersion as error:
            await self._abort(attempt, AttemptAbortReason.CAPTURE_FAILED)
            return self._sampling_stopped(AttemptAbortReason.CAPTURE_FAILED, error)
        except UndeclaredProducer as error:
            await self._abort(attempt, AttemptAbortReason.SAMPLE_FAILED)
            return self._sampling_stopped(AttemptAbortReason.SAMPLE_FAILED, error)
        except (StaleRolloutError, InternalPolicyDeltaError) as error:
            await self._abort(attempt, AttemptAbortReason.STALE)
            return (
                Discard.INTERNAL_DELTA
                if isinstance(error, InternalPolicyDeltaError)
                else Discard.STALE_BEFORE_GRADING
            )
        except (
            CaptureIntegrityFailure,
            SamplingFailure,
            ModelTransportFailure,
            RendererFailure,
        ) as error:
            reason = _abort_reason(error)
            await self._abort(attempt, reason, error)
            self._log.report("sampling", identity.rollout_id, error)
            retry = isinstance(error, RendererFailure)
            return _Failed(reason, error, Discard.ROLLOUT_FAILED, retry, True)
        phases.sealed = True
        self._capacity.notify()
        try:
            rollout = Rollout(
                identity=identity,
                task=attempt.context.spec.task,
                samples=sealed.samples,
                reward=None,
                metadata=json_mapping(result.metadata, where="episode metadata"),
                discarded_versions=sealed.discarded_versions,
            )
        except TrajectoryError as error:
            await self._abort(attempt, AttemptAbortReason.ROLLOUT_INVALID, error)
            self._log.report("sampling", identity.rollout_id, error)
            # The runtime's own value: the same every attempt, so never retried.
            return _Failed(
                AttemptAbortReason.ROLLOUT_INVALID, error, Discard.ROLLOUT_FAILED, False, True
            )
        if self._on_sealed is not None:
            self._on_sealed()
        # A rollout whose agent discarded every generation is still graded: dated by
        # what it discarded, it counts in its group's baseline and trains nothing.
        if rollout.minimum_policy_version is None or (
            rollout.internal_delta > self._settings.max_policy_delta_in_trajectory
        ):
            await self._terminal(
                attempt,
                "finish",
                lambda: self._runtime.finish(
                    attempt.context, rollout, EpisodeDisposition.DISCARDED
                ),
                settle=False,
                primary=None,
            )
            unversioned = rollout.minimum_policy_version is None
            return Discard.UNVERSIONED if unversioned else Discard.INTERNAL_DELTA
        self._budget.success("sampling")
        return Admitted(attempt, rollout, sealed.transcript)

    # ---------------------------------------------------------------- grading

    async def _grade_try(self, attempt: _Attempt, rollout: Rollout) -> GradeResult:
        phases = attempt.phases
        runner_owns = Phase.VERIFICATION not in phases.declared
        deadline = self._settings.grading_timeout_seconds
        loop = asyncio.get_running_loop()
        started: float | None = None
        finished: float | None = None

        async def grade() -> GradeResult:
            nonlocal started, finished
            started = loop.time()
            result = await _hook(
                "grade", phases.identity, self._runtime.grade(attempt.context, rollout)
            )
            finished = loop.time()  # when grade() returned, not when the driver resumes
            return result

        # The runner's slot is taken inside the supervised hook, as sample()'s is, so
        # a grade() still running after its grace still holds it.
        async def body() -> GradeResult:
            if runner_owns:
                async with self._capacity.runner_hold(phases, Phase.VERIFICATION):
                    return await grade()
            return await grade()

        def budget() -> tuple[float, AttemptAbortReason] | None:
            if started is None:
                return None  # waiting for its slot: the try's clock has not started
            return (deadline - (loop.time() - started), AttemptAbortReason.CANCELLED)

        phases.start_hook(Hook.GRADE)
        try:
            result = await self._supervise(
                body(), "grade", phases, budget, processes=attempt.processes
            )
        except NonQuiescentHook:
            phases.end_hook()
            raise
        except BaseException:
            phases.end_hook()
            _release_leftovers(phases)
            raise
        phases.end_hook()
        self._check_hook_return(phases, "grade")
        # A grade() that blocked the loop past its deadline returns before the deadline's
        # timer could stop it: its result is late, and refused as one that ran out.
        if started is not None and finished is not None and finished - started >= deadline:
            raise HookDeadline(AttemptAbortReason.CANCELLED, "grade() exceeded its deadline")
        if not isinstance(result, GradeResult):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise self._systemic_error(
                RuntimeContractError(f"grade() returned {type(result).__name__}")
            )
        reward = result.reward
        # Only a rollout whose group grader gives it its reward may have none yet.
        pending = reward is None and self._capabilities.pending_group_grade
        if not pending and not finite_reward(reward):
            # A grader bug: the same result every try, so never retried.
            raise PermanentGradingFailure(
                f"grade() returned a non-finite or non-numeric reward {reward!r}"
            )
        try:
            json_mapping(result.metadata, where="grade metadata")
        except TrajectoryError as error:
            raise PermanentGradingFailure(str(error)) from error
        clashes = sorted(set(result.metadata) & set(rollout.metadata))
        if clashes:
            raise PermanentGradingFailure(
                f"grade() metadata repeats episode metadata keys {clashes}"
            )
        return result

    # -------------------------------------------------------------- terminals

    def _finalize(
        self,
        admitted: Admitted,
        rollout: Rollout,
        disposition: EpisodeDisposition,
        primary: BaseException | None = None,
        *,
        report: str | None = None,
    ) -> None:
        task = asyncio.create_task(
            self._finish_now(admitted, rollout, disposition, primary, report=report),
            name=f"finish:{rollout.identity.rollout_id}",
        )
        self._finishing.add(task)
        task.add_done_callback(self._finishing.discard)

    async def _finish_now(
        self,
        admitted: Admitted,
        rollout: Rollout,
        disposition: EpisodeDisposition,
        primary: BaseException | None = None,
        *,
        report: str | None = None,
    ) -> None:
        """Run ``finish()``; ``report`` names the stage to report ``primary`` under after it."""
        attempt = admitted.attempt
        if admitted._settled:  # pyright: ignore[reportPrivateUsage]
            return
        admitted._settled = True  # pyright: ignore[reportPrivateUsage]
        try:
            await self._terminal(
                attempt,
                "finish",
                lambda: self._runtime.finish(attempt.context, rollout, disposition),
                settle=True,
                primary=primary,
            )
        finally:
            if report is not None and primary is not None:
                self._log.report(report, rollout.identity.rollout_id, primary)

    async def _abort(
        self,
        attempt: _Attempt,
        reason: AttemptAbortReason,
        primary: BaseException | None = None,
    ) -> None:
        await self._terminal(
            attempt,
            "abort",
            lambda: self._runtime.abort(attempt.context, reason),
            settle=False,
            primary=primary,
        )

    async def _terminal(
        self,
        attempt: _Attempt,
        name: str,
        hook: Callable[[], Coroutine[Any, Any, None]],
        *,
        settle: bool,
        primary: BaseException | None,
    ) -> None:
        """Run ``finish()`` or ``abort()`` to completion even if our caller is cancelled.

        A failure of the hook is recorded on ``primary``, the failure that caused the
        cleanup, when there is one.
        """
        attempt.lease.revoke()
        # Shutdown cancels attempts, once or more, and their cleanup still runs to
        # its own deadline; the shutdown drain bounds how long anyone waits for it.
        await see_through(asyncio.create_task(self._cleanup(attempt, name, hook, settle, primary)))

    async def _cleanup(
        self,
        attempt: _Attempt,
        name: str,
        hook: Callable[[], Coroutine[Any, Any, None]],
        settle: bool,
        primary: BaseException | None,
    ) -> None:
        phases = attempt.phases
        # Every process the attempt still runs ends first, outside the hook's deadline.
        failure: Exception | None = None
        try:
            await attempt.processes.terminate()
        except ProcessKillFailure as error:
            failure = error
        # An attempt that ends without sealing keeps its transcript as far as it got.
        await attempt.lease.keep_unsealed()
        loop = asyncio.get_running_loop()
        started = loop.time()
        excluded = phases.excluded_seconds()
        deadline = self._settings.cleanup_timeout_seconds

        def budget() -> tuple[float, AttemptAbortReason] | None:
            # Resource-service time, such as a sandbox release, is not the hook's.
            spent = loop.time() - started - (phases.excluded_seconds() - excluded)
            return (deadline - spent, AttemptAbortReason.CANCELLED)

        systemic = False
        try:
            await self._supervise(
                _hook(name, phases.identity, hook()),
                name,
                phases,
                budget,
                stop_on_contract_error=False,
                processes=attempt.processes,
            )
        except NonQuiescentHook as still_running:
            # A hook that ignored cancellation leaves its attempt live: the
            # episode may still be running, and is reported rather than settled.
            self._retain(attempt, name, release_holds=True)
            self._cleanup_failed(attempt, still_running, primary)
            try:
                # raises a cancellation of its waiter
                self._keep(attempt, still_running, None, settles=settle)
            finally:
                await _closed(attempt.processes)
            return
        except SystemicFailure:
            systemic = True  # already reported; cleanup still completes below
        except Exception as error:  # noqa: BLE001 - reported and budgeted
            failure = failure or error
        # The attempt's last hook has returned: whatever it or a task of its
        # runtime started since is ended, and nothing starts again.
        closing = await _closed(attempt.processes)
        failure = failure or closing
        # The episode's resources end with it, whatever its hook did.
        try:
            report = await attempt.resources.close()
        except Exception as error:  # noqa: BLE001 - reported with the cleanup, never lost
            if failure is None:
                failure = error
            else:
                _attach_cleanup(failure, error)
        else:
            if failure is None and not report.complete:
                failure = CleanupIncomplete(report)
        if failure is not None:
            self._cleanup_failed(attempt, failure, primary)
        elif not systemic:
            self._budget.success("cleanup")
        self._close(attempt)
        if settle:
            self._settle()

    def _cleanup_failed(
        self, attempt: _Attempt, failure: BaseException, primary: BaseException | None
    ) -> None:
        if primary is not None:
            _attach_cleanup(primary, failure)
        rollout_id = attempt.phases.identity.rollout_id
        self._log.report("cleanup", rollout_id, failure)
        # Its rollout's cause, unless the failure that caused the cleanup is one or
        # its record is already written.
        if not attempt.recorded:
            note = self._note(rollout_id)
            note.cause = note.cause or f"{_CLEANUP_FAILED}/{sub_reason(failure)}"
        self._count("cleanup", failure)

    # ------------------------------------------------------------ supervision

    async def _supervise[T](
        self,
        coroutine: Coroutine[Any, Any, T],
        name: str,
        phases: AttemptPhases,
        budget: Callable[[], tuple[float, AttemptAbortReason] | None],
        *,
        stop_on_contract_error: bool = True,
        on_stop: Callable[[], None] | None = None,
        processes: AttemptProcessSupervisor,
    ) -> T:
        """Await a runtime hook, cancelling it at its deadline with bounded quiescence.

        A contract violation stops ``sample()`` and ``grade()`` at once; a cleanup
        hook still runs to its own deadline so the runtime can unwind. ``on_stop``
        runs the moment the hook is cancelled, before its grace. The attempt's agent
        processes are terminated before the hook is cancelled, so the hook's grace
        starts only once they have exited.
        """
        task = asyncio.create_task(coroutine, name=f"{name}:{phases.identity.rollout_id}")

        async def halting() -> None:
            # Nothing new starts while the hook stops, and its processes end before
            # its grace starts.
            if on_stop is not None:
                on_stop()
            processes.pause()
            with contextlib.suppress(ProcessKillFailure):  # cleanup kills again, and reports it
                await processes.terminate()

        grace = self._settings.cancellation_grace_seconds
        ran = await run_until(
            task,
            _expiry(phases, budget, stop_on_contract_error),
            grace,
            own_failure=lambda leaked: RuntimeExecutionFailure(name, phases.identity, leaked),
            before_stop=halting,
        )
        if not isinstance(ran, Stopped):
            return ran
        if not ran.settled:
            # Only the grace's end tells a hook that stopped, whose holds may go,
            # from one still running inside them. Its processes stay paused: it
            # starts none while it runs on.
            self._still_running.add(task)
            task.add_done_callback(functools.partial(self._hook_ended, name, phases))
            # Whether that stops the run is the caller's to decide.
            raise NonQuiescentHook(
                f"runtime {name}() for {phases.identity.rollout_id} ignored cancellation "
                f"for {grace:g}s; its attempt and runtime state remain live",
                ran,
            )
        processes.resume()
        ran.propagate()
        if stop_on_contract_error and phases.contract_error is not None:
            raise self._systemic_error(phases.contract_error)
        assert ran.expiry is not None  # not interrupted: the expiry stopped it
        raise HookDeadline(ran.expiry, f"{name}() exceeded its deadline")

    def _retain(self, attempt: _Attempt, name: str, *, release_holds: bool) -> None:
        """Record a hook that is still running; its attempt stays live.

        Holds a still-running sample() or grade() is inside stay held, because the
        work they bound is still happening. A cleanup hook holds no capacity.
        """
        attempt.retained = True
        self._unquiesced.append(f"{name}:{attempt.phases.identity.rollout_id}")
        if release_holds:
            self._close(attempt)
        else:
            self._attempts.pop(attempt.phases.identity.attempt_id, None)

    def _keep(
        self,
        attempt: _Attempt,
        still_running: NonQuiescentHook,
        cleanup: tuple[str, Callable[[], Coroutine[Any, Any, None]]] | None,
        *,
        settles: bool = False,
    ) -> None:
        """A kept hook: the run goes on unless the kept hooks still running pass 5% of
        ``max_concurrent_rollouts``, rounded up, or hold every slot of a phase or every
        start admission place, which no other rollout could then get.

        Each holds its capacity until it ends, so the bound caps what is lost meanwhile,
        and anything more it asks for is refused in its own task. A hook kept while its
        waiter was cancelled, as at shutdown, stops the run as before, and the
        cancellation is raised.
        """
        attempt.phases.kept = True
        self._kept[attempt.phases.identity.attempt_id] = _Kept(attempt, cleanup, settles)
        kept = len(self._kept)
        snapshot = self._capacity.snapshot()
        limit = max(1, -(-int(snapshot["slots/sampling_limit"]) // 20))  # 5%, rounded up
        if still_running.waited.interrupted:
            self._systemic_error(still_running)
            still_running.reraise()
        sampling = int(snapshot["slots/sampling_limit"])
        lost = (
            (Phase.SAMPLING, sampling, "sampling slots"),
            (Phase.VERIFICATION, int(snapshot["slots/verification_limit"]), "verification slots"),
            # Admitted and not yet sampling, as a setup() that ignored cancellation is.
            (None, sampling + int(snapshot["admission/prestart_limit"]), "admission places"),
        )
        for phase, places, what in lost:
            held = sum(
                self._capacity.admitted(phases) if phase is None else phases.holding.get(phase, 0)
                for phases in (kept.attempt.phases for kept in self._kept.values())
            )
            if held >= places:
                self._systemic_error(
                    NonQuiescentHook(
                        f"{kept} hooks ignored cancellation past their "
                        f"grace and hold all {places} {what} until they end, so no other "
                        f"rollout can get one. Latest: {still_running}",
                        still_running.waited,
                    )
                )
                return
        if kept > limit:
            self._systemic_error(
                NonQuiescentHook(
                    f"{kept} hooks ignored cancellation past their "
                    f"grace, more than the {limit} tolerated (5% of max_concurrent_rollouts, "
                    f"rounded up); each holds its capacity until it ends. "
                    f"Latest: {still_running}",
                    still_running.waited,
                )
            )
        else:
            logger.warning(
                "%s; it holds its capacity until it ends (%d of %d tolerated)",
                still_running,
                kept,
                limit,
            )

    def _hook_ended(self, name: str, phases: AttemptPhases, task: asyncio.Task[Any]) -> None:
        """A hook kept past its grace has ended: what it still holds goes back, its
        attempt gets the cleanup it was owed, and it no longer counts toward the bound."""
        self._still_running.discard(task)
        identity = phases.identity
        kept = self._kept.pop(identity.attempt_id, None)
        if kept is None:
            return
        self._capacity.forget(phases)
        self._unquiesced.remove(f"{name}:{identity.rollout_id}")
        for callback in kept.ended:
            callback()
        if kept.cleanup is None:
            # A kept finish() or abort(): its processes are closed; its resources go now.
            # Run to its end, as any cleanup does, even if shutdown stops waiting.
            owed = see_through(asyncio.create_task(self._release_kept(kept.attempt)))
            if kept.settles:
                self._settle()
        else:
            # Its abort() or finish(), which may start processes as any may.
            kept.attempt.processes.resume()
            hook_name, hook = kept.cleanup
            owed = self._terminal(kept.attempt, hook_name, hook, settle=kept.settles, primary=None)
        cleaning = asyncio.create_task(owed, name=f"cleanup:{identity.rollout_id}")
        self._finishing.add(cleaning)
        cleaning.add_done_callback(self._finishing.discard)
        failure = None if task.cancelled() else task.exception()
        logger.warning(
            "%s() for %s ended after it was kept: %s; its attempt is cleaned up",
            name,
            identity.rollout_id,
            "cancelled" if task.cancelled() else "returned" if failure is None else repr(failure),
        )

    async def _release_kept(self, attempt: _Attempt) -> None:
        """Release a kept attempt's resources; a failure is reported, not counted again."""
        try:
            report = await attempt.resources.close()
        except Exception as error:  # noqa: BLE001 - reported, never lost
            failure: BaseException | None = error
        else:
            failure = None if report.complete else CleanupIncomplete(report)
        if failure is not None:
            self._log.report("cleanup", attempt.phases.identity.rollout_id, failure)

    def _check_hook_return(self, phases: AttemptPhases, name: str) -> None:
        if phases.contract_error is not None:
            raise self._systemic_error(phases.contract_error)
        if phases.open_holds:
            open_phases = sorted(hold.phase.value for hold in phases.open_holds)
            for hold in list(phases.open_holds):
                hold.release()
            raise self._systemic_error(
                RuntimeContractError(f"{name}() returned with holds still open: {open_phases}")
            )

    # ---------------------------------------------------------------- budgets

    def _violation(self, error: RuntimeContractError) -> None:
        self._systemic_error(error)

    def _systemic_error(self, error: BaseException) -> SystemicFailure:
        systemic = error if isinstance(error, SystemicFailure) else SystemicFailure(f"{error}")
        if systemic is not error:
            systemic.__cause__ = error
        if self._systemic is None:
            self._systemic = systemic
            self._on_systemic(systemic)
        return systemic

    def _count_sampling(self, outcome: Admitted | Discard | _Failed) -> None:
        """Count a finished attempt toward the sampling stop."""
        if outcome is Discard.UNVERSIONED:
            self._count("sampling", ValueError("rollout has no trainable policy evidence"))
        elif isinstance(outcome, _Failed) and outcome.counts:
            self._count("sampling", outcome.error)

    def _count(self, stage: str, error: BaseException) -> None:
        """Report the stop, never raise it: the outcome that trips it is still returned."""
        systemic = self._budget.failure(stage, error)
        if systemic is not None and self._systemic is None:
            self._systemic_error(systemic)
        if stage == "sampling":
            self._ask_run_status()

    def _ask_run_status(self) -> None:
        """Ask in the background whether the Loops run has ended: one check at a time,
        at most once per ``run_status_interval_seconds``.

        The failure that prompted it is still counted as usual: only the answer, when
        it says the run has ended, stops the run here.
        """
        if self._systemic is not None:
            return
        if self._status_check is not None and not self._status_check.done():
            return
        now = asyncio.get_running_loop().time()
        if (
            self._status_asked is not None
            and now - self._status_asked < self._settings.run_status_interval_seconds
        ):
            return
        self._status_asked = now
        self._status_check = asyncio.create_task(self._check_run_status(), name="loops-run-status")

    async def _check_run_status(self) -> None:
        try:
            reason = await self._run_ended()
        except Exception:
            logger.warning("Could not read whether the Loops run has ended", exc_info=True)
            return
        if reason is None or self._systemic is not None:
            return
        self._systemic_error(LoopsRunEnded(reason))

    def _raise_if_stopped(self) -> None:
        if self._systemic is not None:
            raise self._systemic

    def _settle(self) -> None:
        self._live -= 1


def finite_reward(value: object) -> bool:
    """A reward the cookbook can train on: a real, finite number that is not a bool."""
    numeric = isinstance(value, numbers.Real) and not isinstance(value, bool)
    return numeric and math.isfinite(float(value))  # pyright: ignore[reportArgumentType]


def _origin(stop: SystemicFailure) -> BaseException:
    """The violation a stop was raised for: its cause, when it wraps one."""
    return stop.__cause__ or stop


def _tally(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _release_leftovers(phases: AttemptPhases) -> None:
    """A hook that raised gives back any hold it left open; the error is the outcome."""
    for hold in list(phases.open_holds):
        hold.release()


async def _expiry(
    phases: AttemptPhases,
    budget: Callable[[], tuple[float, AttemptAbortReason] | None],
    stop_on_contract_error: bool,
) -> AttemptAbortReason:
    """Returns once a hook must stop: its budget is spent, or the runtime broke its contract.

    The budget is read again whenever the attempt's phases change, since a hold
    taken or released can start or stop a clock.
    """
    while True:
        if stop_on_contract_error and phases.contract_error is not None:
            return AttemptAbortReason.CANCELLED
        remaining = budget()
        if remaining is not None and remaining[0] <= 0:
            return remaining[1]
        phases.changed.clear()
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(None if remaining is None else remaining[0]):
                await phases.changed.wait()


async def _closed(processes: AttemptProcessSupervisor) -> ProcessKillFailure | None:
    """Close the attempt's processes; a group that would not die is returned, not raised."""
    try:
        await processes.close()
    except ProcessKillFailure as error:
        return error
    return None
