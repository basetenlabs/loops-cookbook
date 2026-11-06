"""One eval of one set: its tasks, each rolled out ``samples_per_task`` times on a
sampler pinned to one step's weights, graded, and summarized. Nothing here trains.

An eval needs far less than training: no admission window, no staleness, no batches,
no group selection. Each rollout goes through the run's own ``AttemptDriver`` --
``sample()``, then ``grade()`` -- at most ``max_concurrent_rollouts`` at once. With a
group grader, a task's graded samples are then scored together, retried as the
training pipeline retries them.
"""

from __future__ import annotations

import asyncio
import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from loops_cookbook.cancellation import outlast
from loops_cookbook.evidence import JsonValue, Rollout, TaskRef
from loops_cookbook.llm.transcript import SealedTranscript
from loops_cookbook.rl.config import EvalSetSettings
from loops_cookbook.runtime.api import (
    GroupGrader,
    MemberEvidence,
    PermanentGradingFailure,
    RolloutRuntime,
    RuntimeCapabilities,
)
from loops_cookbook.runtime.capacity import RunCapacity
from loops_cookbook.runtime.context import MetricsSink
from loops_cookbook.runtime.driver import AttemptDriver, Discard, SystemicFailure
from loops_cookbook.runtime.groups import GroupGrading, GroupGradingSettings


class SetSlots:
    """A set's bound on its rollouts at once, across every eval of the set. A hook kept
    past its grace holds its slot until it ends, into the set's later evals; ``kept``
    counts those slots."""

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.free = asyncio.Semaphore(cap)
        self.kept = 0

    @property
    def all_kept(self) -> bool:
        """Kept hooks hold every slot: no rollout of the set can start until one ends."""
        return self.kept >= self.cap

    def kept_ended(self) -> None:
        """A kept hook ended: its slot is free again."""
        self.kept -= 1
        self.free.release()


def _all_kept(eval_set: EvalSet, slots: SetSlots) -> SystemicFailure:
    return SystemicFailure(
        f"hooks of eval set {eval_set.name!r} ignored cancellation past their grace and "
        f"hold all {slots.cap} of its slots until they end, so none of its rollouts can "
        "start"
    )


@dataclass(frozen=True, slots=True)
class EvalSet:
    name: str
    settings: EvalSetSettings
    runtime: RolloutRuntime
    capabilities: RuntimeCapabilities
    group_grader: GroupGrader | None
    tasks: tuple[TaskRef, ...]
    """Already cut to ``max_tasks``."""
    rollout_timeout_seconds: float
    """The set's rollout clock: the one configured, else its own runtime's."""
    metrics: MetricsSink = field(default_factory=MetricsSink)
    """What its runtime reports, kept apart from training's."""
    slots: SetSlots | None = None
    """The bound on its rollouts at once, across every eval of the set. None: each
    eval's own."""


@dataclass(frozen=True, slots=True)
class EvalResult:
    rows: tuple[dict[str, JsonValue], ...]
    """One per rollout, in task then sample order."""
    gauges: dict[str, float]
    """Every name under ``eval/<set>/``."""
    transcripts: tuple[tuple[str, SealedTranscript], ...] = ()
    """Each sealed rollout's conversation, by rollout ID: what the policy was asked and
    what it replied."""


class _AdmitAll:
    """Start admission for an eval: a semaphore bounds it, so admit up to the cap."""

    def __init__(self, cap: int) -> None:
        self._cap = cap

    def target(self, group_id: str) -> int:
        return self._cap

    def started(self, group_id: str) -> None:
        pass

    def order(self, group_id: str) -> int:
        return 0


class PinnedSampler:
    """An eval's standalone sampler, whose samples are labelled with the eval's version.

    It serves exactly the checkpoint it was created on. The policy version it reports
    is its own counter, which only weight sync moves: 0 on a dedicated sampler,
    whatever step it serves (a serverless one reports none). Dropped, the lease labels
    each sample with the version its handle was pinned at, the eval's.
    """

    def __init__(self, sampler: Any) -> None:
        self._sampler = sampler

    async def sample_async(self, *args: Any, **kwargs: Any) -> Any:
        response = copy.copy(await self._sampler.sample_async(*args, **kwargs))
        response.policy_version = None
        return response


DriverFactory = Callable[[RunCapacity, Callable[[SystemicFailure], None]], AttemptDriver]

# One rollout's outcome: graded, or why it was discarded; and its transcript, which the
# eval records and a group grader reads.
_Outcome = tuple[Rollout | Discard | str, SealedTranscript | None]


async def evaluate(
    eval_set: EvalSet,
    *,
    step: int,
    verifiers: int,
    driver: DriverFactory,
    group_grading: GroupGradingSettings,
    drain_seconds: float,
    on_unsettled: Callable[[], None],
) -> EvalResult:
    """Roll out, grade and summarize every task of ``eval_set`` once per sample, at
    most ``verifiers`` grading at once, as training is bounded.

    A ``SystemicFailure`` or a cancellation cancels what is still running and is
    raised. Waiting for the cancelled rollouts and the driver's drain is bounded by
    ``drain_seconds``; a wait cut off, or a hook still running after cancellation, is
    reported to ``on_unsettled``.
    """
    cap = eval_set.settings.max_concurrent_rollouts
    label = "base" if step < 0 else str(step)  # in rollout IDs: the base model's is -1
    capacity = RunCapacity(
        sampling_slots=cap,
        verification_slots=min(cap, verifiers),
        admission=_AdmitAll(cap),
        on_change=lambda: None,
    )
    stopped = asyncio.Event()
    failures: list[SystemicFailure] = []

    def on_systemic(failure: SystemicFailure) -> None:
        failures.append(failure)
        stopped.set()

    attempts = driver(capacity, on_systemic)
    grading = (
        None
        if eval_set.group_grader is None
        else GroupGrading(
            eval_set.group_grader,
            capacity,
            group_grading,
            observe=lambda _: None,
            unquiesced=attempts.group_unquiesced,
        )
    )
    # The bound on work in flight: RunCapacity gates only the sampling holds, and a
    # runtime may acquire that phase itself. One per set, shared by its evals.
    slots = eval_set.slots or SetSlots(cap)
    if slots.all_kept:
        raise _all_kept(eval_set, slots)
    samples = eval_set.settings.samples_per_task

    async def rollout(task: TaskRef, group_id: str, index: int) -> _Outcome:
        rollout_id = f"{group_id}-s{index}"
        await slots.free.acquire()
        try:
            outcome = await attempts.sample(task, group_id, rollout_id)
            if isinstance(outcome, Discard):
                return outcome, None
            return await attempts.grade(outcome), outcome.transcript
        finally:
            # A hook kept past its grace is still running: its slot stays taken until
            # it ends, into the set's later evals. Once kept hooks hold every slot, the
            # rollouts waiting for one would wait without end, so the eval stops.
            if attempts.after_kept(rollout_id, slots.kept_ended):
                slots.kept += 1
                if slots.all_kept:
                    on_systemic(_all_kept(eval_set, slots))
            else:
                slots.free.release()

    async def task_outcomes(task_index: int, task: TaskRef) -> list[_Outcome]:
        group_id = f"eval-{eval_set.name}-{label}-{task_index}"
        async with asyncio.TaskGroup() as group:
            runs = [group.create_task(rollout(task, group_id, i)) for i in range(samples)]
        outcomes = [run.result() for run in runs]
        if grading is not None:
            outcomes = await _grade_group(attempts, grading, eval_set, group_id, outcomes)
        return outcomes

    async def roll_out() -> list[asyncio.Task[list[_Outcome]]]:
        try:
            async with asyncio.TaskGroup() as group:
                runs = [
                    group.create_task(task_outcomes(index, task))
                    for index, task in enumerate(eval_set.tasks)
                ]
        except* SystemicFailure as caught:
            raise caught.exceptions[0] from None
        return runs

    async def settle() -> None:
        # A cancelled attempt waits out its abort() before its task ends, so the
        # rollouts' join and the driver's drain share one bound, as in the pipeline.
        rolling.cancel()
        await asyncio.wait({rolling})
        await attempts.drain()

    rolling = asyncio.ensure_future(roll_out())
    failed = asyncio.ensure_future(stopped.wait())
    try:
        await asyncio.wait({rolling, failed}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        failed.cancel()
        draining = asyncio.ensure_future(settle())
        waited = await outlast(draining, drain_seconds)
        if not waited.settled:
            # Stop waiting, not stop cleaning: each abort() runs to its own deadline.
            draining.cancel()
        if not waited.settled or attempts.unquiesced:
            on_unsettled()
        waited.propagate()
    if failures:
        raise failures[0]
    runs = rolling.result()
    outcomes = [
        (task, f"eval-{eval_set.name}-{label}-{task_index}-s{index}", index, outcome)
        for task_index, (task, run) in enumerate(zip(eval_set.tasks, runs, strict=True))
        for index, outcome in enumerate(run.result())
    ]
    rows = tuple(
        _row(task, rollout_id, index, outcome) for task, rollout_id, index, outcome in outcomes
    )
    transcripts = tuple(
        (rollout_id, outcome[1]) for _, rollout_id, _, outcome in outcomes if outcome[1] is not None
    )
    return EvalResult(rows, _gauges(eval_set.name, rows), transcripts)


async def _grade_group(
    attempts: AttemptDriver,
    grading: GroupGrading,
    eval_set: EvalSet,
    group_id: str,
    outcomes: list[_Outcome],
) -> list[_Outcome]:
    """The task's graded samples scored together, as ``Pipeline._grade_group`` tries
    a group: retried within the runtime's ``grading_retries``, the members discarded
    as ``grading_failed`` after the last."""
    graded = [i for i, (outcome, _) in enumerate(outcomes) if isinstance(outcome, Rollout)]
    if not graded:
        return outcomes
    members: list[MemberEvidence] = []
    for i in graded:
        rollout, transcript = outcomes[i]
        if transcript is None:
            raise RuntimeError("a group grader's rollout was sealed without its transcript")
        assert isinstance(rollout, Rollout)
        members.append(MemberEvidence(rollout, transcript))
    tries = 1 + eval_set.capabilities.grading_retries
    for number in range(1, tries + 1):
        try:
            rewards = await grading.score(group_id, members)
        except SystemicFailure:
            raise
        except Exception as error:  # noqa: BLE001 - the grader's failure, recorded and budgeted
            if number < tries and not isinstance(error, PermanentGradingFailure):
                attempts.group_retrying(group_id, error)
                await attempts.back_off(number)
                continue
            ids = [member.rollout.identity.rollout_id for member in members]
            attempts.group_failed(group_id, error, ids)
            scored = list(outcomes)
            for i in graded:
                scored[i] = (Discard.GRADING_FAILED.value, outcomes[i][1])
            return scored
        attempts.group_scored()
        scored = list(outcomes)
        for i, member, reward in zip(graded, members, rewards, strict=True):
            scored[i] = (replace(member.rollout, reward=reward), member.transcript)
        return scored
    raise AssertionError("unreachable")


def _row(task: TaskRef, rollout_id: str, index: int, outcome: _Outcome) -> dict[str, JsonValue]:
    result = outcome[0]
    graded = isinstance(result, Rollout)
    return {
        "task_id": task.id,
        "task_digest": task.digest,
        "sample": index,
        "rollout_id": rollout_id,
        "reward": result.reward if isinstance(result, Rollout) else None,
        "discarded": None
        if graded
        else (result.value if isinstance(result, Discard) else str(result)),
        "metadata": dict(result.metadata) if isinstance(result, Rollout) else {},
    }


def _gauges(name: str, rows: Sequence[dict[str, JsonValue]]) -> dict[str, float]:
    prefix = f"eval/{name}/"
    rewards = [float(row["reward"]) for row in rows if isinstance(row["reward"], (int, float))]
    gauges = {
        f"{prefix}rollouts": float(len(rows)),
        f"{prefix}graded": float(len(rewards)),
    }
    for row in rows:
        if isinstance(reason := row["discarded"], str):
            key = f"{prefix}discarded/{reason}"
            gauges[key] = gauges.get(key, 0.0) + 1
    if rewards:
        mean = sum(rewards) / len(rewards)
        gauges[f"{prefix}reward_mean"] = mean
        gauges[f"{prefix}reward_std"] = math.sqrt(
            sum((r - mean) ** 2 for r in rewards) / len(rewards)
        )
    return gauges
