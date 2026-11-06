"""One eval of one set, through the real driver, lease and capacity broker; only the
runtime, sampler and renderer are scripted."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from hashlib import sha256
from typing import Any

import pytest
from support.harness import Renderer, Runtime, Sampler, generate
from support.resources import no_resources

from loops_cookbook.evidence import TaskRef
from loops_cookbook.llm.client import Renderers, SamplingPolicy
from loops_cookbook.model_access.api import ModelAccessMode
from loops_cookbook.model_access.lease import (
    AttemptLease,
    DirectModelAccess,
    Freshness,
    LeaseWiring,
)
from loops_cookbook.model_access.producers import ProducerSettings
from loops_cookbook.rl.config import EvalSetSettings
from loops_cookbook.rl.evals import EvalSet, SetSlots, evaluate
from loops_cookbook.runtime.api import (
    AttemptAbortReason,
    EpisodeResult,
    GradeResult,
    RuntimeCapabilities,
)
from loops_cookbook.runtime.context import MetricsSink
from loops_cookbook.runtime.driver import AttemptDriver, DriverSettings, SystemicFailure
from loops_cookbook.runtime.groups import GroupGradingSettings
from loops_cookbook.trainer import Capacity

VERSION = 5  # the step's weights: every eval sample is at this version


def tasks(n: int) -> tuple[TaskRef, ...]:
    return tuple(
        TaskRef(f"t{i}", f"sha256:{sha256(f't{i}'.encode()).hexdigest()}") for i in range(n)
    )


async def odd_tasks_fail(attempt, rollout) -> GradeResult:
    """t0 and t2 pass, t1 fails: a score from the task alone."""
    return GradeResult(reward=0.0 if attempt.spec.task.id == "t1" else 1.0)


def build(
    runtime: Runtime,
    *,
    n_tasks: int = 3,
    samples: int = 1,
    cap: int = 16,
    group_grading: bool = False,
    grading_retries: int = 0,
    max_failures: int = 64,
    rollout_timeout: float = 5.0,
):
    runtime.capabilities = RuntimeCapabilities(
        model_access_modes=frozenset({ModelAccessMode.DIRECT}),
        grading_retries=grading_retries,
        pending_group_grade=group_grading,
    )
    sampler = Sampler(lambda: VERSION)
    wiring = LeaseWiring(
        sampler=lambda: sampler,  # pyright: ignore[reportArgumentType]
        capacity=Capacity.DEDICATED,
        sampler_version=lambda: VERSION,
        policy_version=lambda: VERSION,
        producers=ProducerSettings(),
        freshness=Freshness(max_steps_off_policy=0, max_policy_delta_in_trajectory=0),
        sampling_policy=SamplingPolicy(max_generation_tokens=2),
        on_sample=_nothing,
        on_discard=lambda identity, sample: None,
        on_undelivered=lambda identity, sample: None,
        records_transcripts=group_grading,
    )
    settings = DriverSettings(
        run_id="run",
        rollout_timeout_seconds=rollout_timeout,
        setup_timeout_seconds=5.0,
        max_rollout_retries=0,
        grading_timeout_seconds=5.0,
        cleanup_timeout_seconds=5.0,
        cancellation_grace_seconds=0.1,
        process_kill_grace_seconds=0.1,
        max_steps_off_policy=0,
        max_policy_delta_in_trajectory=0,
        max_consecutive_failures=max_failures,
    )

    async def no_wait(seconds: float) -> None:
        return None

    async def live() -> str | None:
        return None

    def driver(capacity, on_systemic) -> AttemptDriver:
        return AttemptDriver(
            runtime,
            runtime.capabilities,
            settings,
            capacity=capacity,
            leases=lambda identity: AttemptLease(identity, wiring, affinity_key=None),
            access=lambda lease: DirectModelAccess(lease, Renderers(Renderer())),  # pyright: ignore[reportArgumentType]
            resources=no_resources,
            metrics=MetricsSink(),
            policy_version=lambda: VERSION,
            on_systemic=on_systemic,
            run_ended=live,
            sleep=no_wait,
        )

    eval_set = EvalSet(
        name="val",
        settings=EvalSetSettings(
            runtime={"files": {"tasks": "val.jsonl"}},
            samples_per_task=samples,
            max_concurrent_rollouts=cap,
        ),
        runtime=runtime,  # pyright: ignore[reportArgumentType]
        capabilities=runtime.capabilities,
        group_grader=runtime if group_grading else None,  # pyright: ignore[reportArgumentType]
        tasks=tasks(n_tasks),
        rollout_timeout_seconds=5.0,
    )
    return eval_set, driver


async def _nothing(identity, sample) -> None:
    return None


async def run(
    eval_set,
    driver,
    *,
    unsettled: list[bool] | None = None,
    drain: float = 5.0,
    verifiers: int = 64,
):
    return await evaluate(
        eval_set,
        step=3,
        verifiers=verifiers,
        driver=driver,
        group_grading=GroupGradingSettings(timeout_seconds=5.0, cancellation_grace_seconds=0.1),
        drain_seconds=drain,
        on_unsettled=lambda: (unsettled if unsettled is not None else []).append(True),
    )


async def test_every_task_and_sample_is_one_row_in_order_and_the_mean_is_over_them():
    eval_set, driver = build(Runtime(on_grade=odd_tasks_fail), n_tasks=3, samples=2)
    result = await run(eval_set, driver)
    assert [(row["task_id"], row["sample"], row["reward"]) for row in result.rows] == [
        ("t0", 0, 1.0), ("t0", 1, 1.0), ("t1", 0, 0.0), ("t1", 1, 0.0), ("t2", 0, 1.0), ("t2", 1, 1.0),
    ]  # fmt: skip
    assert result.gauges == {
        "eval/val/rollouts": 6.0,
        "eval/val/graded": 6.0,
        "eval/val/reward_mean": pytest.approx(2 / 3),
        "eval/val/reward_std": pytest.approx((2 / 9) ** 0.5),
    }


async def test_a_rollout_that_fails_is_a_discarded_row_left_out_of_the_mean():
    async def fail_t1(attempt) -> EpisodeResult:
        if attempt.spec.task.id == "t1":
            raise RuntimeError("the episode broke")
        await generate(attempt)
        return EpisodeResult()

    async def full_marks(attempt, rollout) -> GradeResult:
        return GradeResult(reward=1.0)

    eval_set, driver = build(Runtime(on_sample=fail_t1, on_grade=full_marks), n_tasks=3)
    result = await run(eval_set, driver)
    failed = result.rows[1]
    assert failed["task_id"] == "t1" and failed["reward"] is None
    assert failed["discarded"] == "rollout_failed"
    assert result.gauges["eval/val/reward_mean"] == 1.0
    assert result.gauges["eval/val/graded"] == 2.0
    assert result.gauges["eval/val/discarded/rollout_failed"] == 1.0


async def test_no_more_rollouts_than_the_cap_are_between_sampling_and_the_end_of_grading():
    live = [0]
    peak = [0]

    async def start(attempt) -> EpisodeResult:
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        await generate(attempt)
        return EpisodeResult()

    async def slow_grade(attempt, rollout) -> GradeResult:
        await asyncio.sleep(0.01)
        live[0] -= 1
        return GradeResult(reward=1.0)

    eval_set, driver = build(Runtime(on_sample=start, on_grade=slow_grade), n_tasks=8, cap=2)
    await run(eval_set, driver)
    assert peak[0] == 2


async def test_no_more_rollouts_grade_at_once_than_the_runs_verifier_limit():
    """A verifier limit, a judge's rate limit say, holds in an eval as in training,
    however many rollouts the set's cap lets run."""
    grading = [0]
    peak = [0]

    async def slow_grade(attempt, rollout) -> GradeResult:
        grading[0] += 1
        peak[0] = max(peak[0], grading[0])
        await asyncio.sleep(0.01)
        grading[0] -= 1
        return GradeResult(reward=1.0)

    eval_set, driver = build(Runtime(on_grade=slow_grade), n_tasks=8, cap=4)
    await run(eval_set, driver, verifiers=1)
    assert peak[0] == 1


class GroupRuntime(Runtime):
    """A runtime that is its own group grader; ``tries`` scripts each call's answer."""

    def __init__(self, tries: list[Any]) -> None:
        super().__init__(on_grade=odd_tasks_fail)
        self.tries = tries
        self.seen: list[int] = []

    async def grade_group(self, members):
        self.seen.append(len(members))
        answer = self.tries.pop(0) if len(self.tries) > 1 else self.tries[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


async def test_a_group_grader_scores_each_tasks_samples_together():
    runtime = GroupRuntime([[10.0, 20.0]])
    eval_set, driver = build(runtime, n_tasks=2, samples=2, group_grading=True)
    result = await run(eval_set, driver)
    assert runtime.seen == [2, 2]
    assert [row["reward"] for row in result.rows] == [10.0, 20.0, 10.0, 20.0]


async def test_a_failing_group_grader_is_retried_then_its_rows_are_discarded():
    runtime = GroupRuntime([RuntimeError("judge down"), [3.0]])
    eval_set, driver = build(runtime, n_tasks=1, group_grading=True, grading_retries=1)
    assert [row["reward"] for row in (await run(eval_set, driver)).rows] == [3.0]

    runtime = GroupRuntime([RuntimeError("judge down")])
    eval_set, driver = build(runtime, n_tasks=1, samples=2, group_grading=True, grading_retries=1)
    rows = (await run(eval_set, driver)).rows
    assert runtime.seen == [2, 2]
    assert [(row["reward"], row["discarded"]) for row in rows] == [
        (None, "grading_failed"),
        (None, "grading_failed"),
    ]


async def test_a_systemic_failure_cancels_what_is_still_running_and_is_raised():
    cancelled: list[str] = []

    async def t0_fails_the_rest_wait(attempt) -> EpisodeResult:
        if attempt.spec.task.id == "t0":
            raise RuntimeError("sandbox provider down")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(attempt.spec.task.id)
            raise
        return EpisodeResult()

    eval_set, driver = build(Runtime(on_sample=t0_fails_the_rest_wait), n_tasks=3, max_failures=1)
    with pytest.raises(SystemicFailure, match="1 consecutive"):
        await asyncio.wait_for(run(eval_set, driver), 5)
    assert sorted(cancelled) == ["t1", "t2"]


async def test_a_hook_that_ignores_cancellation_leaves_the_eval_unsettled():
    async def stubborn(attempt) -> EpisodeResult:
        if attempt.spec.task.id == "t0":
            raise RuntimeError("sandbox provider down")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.5)  # past its cancellation grace
        return EpisodeResult()

    eval_set, driver = build(Runtime(on_sample=stubborn), n_tasks=2, max_failures=1)
    unsettled: list[bool] = []
    with pytest.raises(SystemicFailure):
        await asyncio.wait_for(run(eval_set, driver, unsettled=unsettled, drain=0.05), 5)
    assert unsettled == [True]


async def test_a_slow_abort_does_not_hold_the_eval_past_its_drain():
    aborted = asyncio.Event()

    async def t0_fails_the_rest_wait(attempt) -> EpisodeResult:
        if attempt.spec.task.id == "t0":
            raise RuntimeError("sandbox provider down")
        await asyncio.Event().wait()
        return EpisodeResult()

    async def slow_abort(attempt, reason) -> None:
        if reason is AttemptAbortReason.CANCELLED:
            await asyncio.sleep(1.0)  # inside its cleanup deadline, past the eval's drain
            aborted.set()

    runtime = Runtime(on_sample=t0_fails_the_rest_wait, on_abort=slow_abort)
    eval_set, driver = build(runtime, n_tasks=2, max_failures=1)
    unsettled: list[bool] = []
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(SystemicFailure):
        await asyncio.wait_for(run(eval_set, driver, unsettled=unsettled, drain=0.1), 5)
    assert loop.time() - started < 0.5
    assert unsettled == [True]
    await asyncio.wait_for(aborted.wait(), 5)  # its cleanup still runs to the end


async def test_a_hook_kept_past_its_grace_holds_its_sets_slot_into_the_next_eval():
    """The set's cap spans its evals, so a hook still running counts against the next."""
    live = [0]
    peak = [0]
    stubborn = [True]

    async def sample(attempt) -> EpisodeResult:
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        try:
            if stubborn.pop() if stubborn else False:
                try:
                    await asyncio.Event().wait()  # until its rollout clock cancels it
                except asyncio.CancelledError:
                    await asyncio.sleep(10.0)  # past its cancellation grace: kept
            await generate(attempt)
            await asyncio.sleep(1.0)
            return EpisodeResult()
        finally:
            live[0] -= 1

    eval_set, driver = build(Runtime(on_sample=sample, on_grade=odd_tasks_fail), n_tasks=1, cap=2)
    shared = dataclasses.replace(eval_set, slots=SetSlots(2))
    unsettled: list[bool] = []
    first = await run(shared, driver, unsettled=unsettled)
    assert [row["discarded"] for row in first.rows] == ["unquiesced"]
    assert unsettled == [True] and live == [1]
    await run(dataclasses.replace(shared, tasks=tasks(2)), driver)
    assert peak == [2]


@pytest.mark.parametrize(
    ("cap", "stubborn"), [(1, [True]), (2, [True, True])], ids=["as it starts", "as it runs"]
)
async def test_an_eval_whose_slots_kept_hooks_all_hold_stops_rather_than_waits(cap, stubborn):
    """Hooks kept from the set's earlier evals, and this one's, can come to hold every
    slot; the rollouts waiting for one would wait without end, so the eval stops."""

    async def sample(attempt) -> EpisodeResult:
        if stubborn.pop() if stubborn else False:
            try:
                await asyncio.Event().wait()  # until its rollout clock cancels it
            except asyncio.CancelledError:
                await asyncio.sleep(10.0)  # past its cancellation grace: kept
        await generate(attempt)
        return EpisodeResult()

    eval_set, driver = build(Runtime(on_sample=sample), n_tasks=1, cap=cap, rollout_timeout=0.3)
    shared = dataclasses.replace(eval_set, slots=SetSlots(cap))
    with contextlib.suppress(SystemicFailure):  # one slot, kept: that eval stops too
        await run(shared, driver)
    later = dataclasses.replace(shared, tasks=tasks(2))
    with pytest.raises(SystemicFailure, match=f"hold all {cap} of its slots"):
        await asyncio.wait_for(run(later, driver, drain=0.2), 5)
