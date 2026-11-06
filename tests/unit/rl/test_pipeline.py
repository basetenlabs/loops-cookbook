"""Group admission, start admission, the async window, freshness, and commit boundaries.

Every pipeline here runs the real attempt driver, capacity broker, and lease over a
scripted runtime, so these are the scheduling rules as training sees them.
"""

import asyncio
import collections
import itertools
import logging
import math
import time
from dataclasses import replace
from fractions import Fraction

import pytest
from support.builders import task
from support.cancellation import (
    Landing,
    Response,
    Trigger,
    Work,
    cancellation_matrix,
    outcome_within,
    settle_loop,
)
from support.harness import Renderer, Runtime, Sampler, generate
from support.resources import no_resources

from loops_cookbook.llm.client import Renderers, SamplingPolicy
from loops_cookbook.model_access.api import ModelAccessMode
from loops_cookbook.model_access.lease import (
    AttemptLease,
    DirectModelAccess,
    Freshness,
    LeaseWiring,
)
from loops_cookbook.model_access.producers import ProducerSettings
from loops_cookbook.rl.pipeline import (
    Counters,
    Pipeline,
    TaskStream,
    UnusableGroupsError,
    _admission_limit,
    _admission_window,
)
from loops_cookbook.runtime.api import (
    EpisodeDisposition,
    EpisodeResult,
    GradeResult,
    PermanentGradingFailure,
    Phase,
    RuntimeCapabilities,
)
from loops_cookbook.runtime.context import MetricsSink
from loops_cookbook.runtime.driver import (
    AttemptDriver,
    Discard,
    DriverSettings,
    RolloutNote,
    SystemicFailure,
)
from loops_cookbook.trainer import Capacity

TASKS = [task("a"), task("b")]


async def no_sleep(seconds):
    return None


class Harness:
    def __init__(self, config, runtime=None, phases=frozenset()):
        self.config = config
        self.version = 0
        self.calls = []
        self.runtime = runtime or Runtime(on_sample=self.sample, on_grade=self.grade)
        self.runtime.capabilities = RuntimeCapabilities(
            model_access_modes=frozenset({ModelAccessMode.DIRECT}),
            internally_acquired_phases=phases,
        )
        self.sampler = Sampler(lambda: self.version)
        self.records_transcripts = False  # on for a runtime with a group grader
        self.sleep = no_sleep  # the grading backoff: none, unless a test holds it
        self.stopped = asyncio.Event()
        """Set once the driver stops the run."""

    async def sample(self, attempt):
        self.calls.append((attempt.spec.identity.group_id, self.version))
        await generate(attempt)
        return EpisodeResult()

    async def grade(self, attempt, rollout):
        return GradeResult(reward=float(int(rollout.identity.rollout_id.rsplit("r", 1)[1]) % 2))

    def pipeline(self, tasks=TASKS, **kwargs):
        config = self.config
        training, rollout = config.training, config.rollout

        async def record(identity, sample):
            return None

        wiring = LeaseWiring(
            sampler=lambda: self.sampler,
            capacity=Capacity.DEDICATED,
            policy_version=lambda: self.version,
            producers=ProducerSettings(
                {name: spec.trainable for name, spec in rollout.producers.items()}
            ),
            freshness=Freshness(training.steps_off_policy, training.policy_delta_in_trajectory),
            sampling_policy=SamplingPolicy(max_generation_tokens=2),
            on_sample=record,
            on_discard=lambda identity, sample: None,
            on_undelivered=lambda identity, sample: None,
            records_transcripts=self.records_transcripts,
        )
        settings = DriverSettings(
            run_id="run",
            rollout_timeout_seconds=rollout.rollout_timeout_seconds,
            setup_timeout_seconds=rollout.setup_timeout_seconds,
            max_rollout_retries=rollout.max_rollout_retries,
            grading_timeout_seconds=rollout.grading.timeout_seconds,
            cleanup_timeout_seconds=rollout.cleanup.timeout_seconds,
            cancellation_grace_seconds=rollout.cleanup.cancellation_grace_seconds,
            process_kill_grace_seconds=rollout.cleanup.process_kill_grace_seconds,
            max_steps_off_policy=training.steps_off_policy,
            max_policy_delta_in_trajectory=training.policy_delta_in_trajectory,
            max_consecutive_failures=rollout.max_consecutive_rollout_failures,
        )

        async def live():
            return None

        def driver(capacity, on_systemic, on_sealed):
            def stopping(error):
                self.stopped.set()
                on_systemic(error)

            return AttemptDriver(
                self.runtime,
                self.runtime.capabilities,
                settings,
                capacity=capacity,
                leases=lambda identity: AttemptLease(identity, wiring, affinity_key=None),
                access=lambda lease: DirectModelAccess(lease, Renderers(Renderer())),
                resources=no_resources,
                metrics=MetricsSink(),
                policy_version=lambda: self.version,
                on_systemic=stopping,
                run_ended=live,
                on_sealed=on_sealed,
                sleep=self.sleep,
            )

        return Pipeline(config, tasks=list(tasks), driver=driver, **kwargs)


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


def judging(h, judge, *, retries=0):
    """A pipeline harness whose runtime has ``judge`` as its group grader, tried
    ``retries`` more times after a failure."""
    h.runtime.capabilities = replace(
        h.runtime.capabilities, pending_group_grade=True, grading_retries=retries
    )
    h.records_transcripts = True
    h.runtime.grade_group = judge


async def test_a_rollout_clock_wave_stops_the_run_naming_the_clock(config):
    h = Harness(
        config(
            max_consecutive_rollout_failures=2,
            max_consecutive_unusable_groups=100,
            rollout_timeout_seconds=0.02,
        )
    )

    async def hangs(attempt):
        await asyncio.Event().wait()  # the rollout clock cancels it

    h.runtime.on_sample = hangs
    async with h.pipeline() as pipeline:
        with pytest.raises(SystemicFailure, match="the rollout clock expired"):
            await asyncio.wait_for(pipeline.next_batch(), 3)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("undersized", [False, True])
async def test_consecutive_unusable_groups_abort(config, mode, undersized, caplog):
    h = Harness(
        config(mode=mode, max_consecutive_unusable_groups=3, max_consecutive_rollout_failures=0)
    )

    async def grade(attempt, rollout):
        if undersized:
            raise RuntimeError("grading unavailable")
        return GradeResult(reward=0.0)

    h.runtime.on_grade = grade
    async with h.pipeline() as pipeline:
        with pytest.raises(UnusableGroupsError, match="3 consecutive") as stopped:
            await asyncio.wait_for(pipeline.next_batch(), timeout=3)
        assert pipeline.snapshot()["groups/selected"] == 0
    assert "consecutive unusable groups" in caplog.text
    if undersized:
        assert "3 had too few usable rollouts" in str(stopped.value)
    else:
        assert "3 gave every rollout one reward (0 in 3)" in str(stopped.value)


def _discarding(h, which):
    """Sampling that discards its one generation in the rollouts ``which`` picks by
    number, as an agent drops a reply the token budget cut off."""

    async def sample(attempt):
        conversation = attempt.services.model_access.llm().conversation()
        reply = await conversation.generate(messages=[{"role": "user", "content": "hi"}])
        if which(int(attempt.spec.identity.rollout_id.rsplit("r", 1)[1])):
            conversation.discard(reply)
        return EpisodeResult()

    h.runtime.on_sample = sample


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_a_rollout_that_discarded_every_generation_counts_in_its_group_and_trains_nothing(
    config, mode
):
    h = Harness(config(mode=mode, group_size=4))
    _discarding(h, lambda number: number % 2 == 0)  # the rollouts scored 0
    async with h.pipeline() as pipe:
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
    [group] = batch
    assert sorted(r.reward for r in group.members) == [0.0, 0.0, 1.0, 1.0]
    assert sorted(r.trains for r in group.members) == [False, False, True, True]


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_a_group_whose_advantages_are_all_on_discarded_rollouts_cannot_train(config, mode):
    h = Harness(
        config(mode=mode, max_consecutive_unusable_groups=3, max_consecutive_rollout_failures=0)
    )
    _discarding(h, lambda number: True)
    async with h.pipeline() as pipe:
        with pytest.raises(UnusableGroupsError, match="3 consecutive") as stopped:
            await asyncio.wait_for(pipe.next_batch(), 3)
        assert pipe.counters.discarded_groups[Discard.NOTHING_TO_TRAIN.value] == 3
    assert "3 had an advantage only on rollouts whose agent discarded" in str(stopped.value)


def test_the_stop_reports_the_rewards_the_unusable_groups_shared():
    """A solved task, a stuck policy, and a constant verifier all look alike; say which rewards."""
    from collections import Counter

    from loops_cookbook.rl.pipeline import _unusable_streak

    message = _unusable_streak(20, Counter({0.5: 8, 1.0: 12}), 0)
    assert message.startswith("20 consecutive unusable groups: 20 gave every rollout one reward")
    assert "(1 in 12, 0.5 in 8)" in message and "too few" not in message
    mixed = _unusable_streak(5, Counter({0.0: 3}), 2)
    assert "3 gave every rollout one reward (0 in 3)" in mixed
    assert "2 had too few usable rollouts" in mixed


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("limit", [0, 2])
async def test_usable_group_resets_rejection_streak(config, mode, limit, caplog):
    """Every other group is unusable, so each one starts a streak of its own."""
    h = Harness(config(mode=mode, groups_per_step=1, max_consecutive_unusable_groups=limit))

    async def grade(attempt, rollout):
        index = int(rollout.identity.group_id.split("-")[0][1:])
        if index % 2 == 0:
            return GradeResult(reward=0.0)
        return await h.grade(attempt, rollout)

    h.runtime.on_grade = grade
    async with h.pipeline() as pipeline:
        async with asyncio.timeout(3):
            for _ in range(3):
                groups = await pipeline.next_batch()
                pipeline.commit_batch(groups)
        assert pipeline.counters.discarded_groups["no_variance"] == 3
    started = [r for r in caplog.records if "1 consecutive unusable groups" in r.getMessage()]
    assert len(started) == 3


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_guard_aborts_partially_selected_batch(config, mode):
    h = Harness(config(mode=mode, groups_per_step=2, max_consecutive_unusable_groups=2))

    async def grade(attempt, rollout):
        if rollout.identity.group_id.startswith("g0000000-"):
            return await h.grade(attempt, rollout)
        return GradeResult(reward=0.0)

    h.runtime.on_grade = grade
    async with h.pipeline() as pipeline:
        with pytest.raises(UnusableGroupsError):
            await asyncio.wait_for(pipeline.next_batch(), timeout=3)
        assert pipeline.snapshot()["groups/selected"] == 0
        assert pipeline.counters.discarded["training_aborted"] == h.config.training.group_size


async def test_a_future_policy_version_stops_the_run_without_training(config):
    h = Harness(config(mode="sync", group_size=2, max_consecutive_rollout_failures=0))

    async def sample(attempt):
        h.sampler.version = lambda: 1
        return await h.sample(attempt)

    h.runtime.on_sample = sample
    async with h.pipeline() as pipe:
        with pytest.raises(SystemicFailure):
            await asyncio.wait_for(pipe.next_batch(), 3)
        assert pipe.counters.trained == 0 and pipe.counters.graded == 0


async def test_sync_admission_stops_across_training_and_publish(config):
    h = Harness(config(mode="sync", group_size=2, groups_per_step=2, max_concurrent_rollouts=16))
    async with h.pipeline() as pipe:
        await asyncio.sleep(0)
        assert not h.calls
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
        assert len(h.calls) == 4
        await settle_loop(rounds=50)
        assert len(h.calls) == 4
        assert pipe.counters.trained == 0
        h.version = 1
        pipe.commit_batch(batch)
        await settle_loop(rounds=50)
        assert len(h.calls) == 4
        second = await asyncio.wait_for(pipe.next_batch(), 3)
        assert all(
            member.minimum_policy_version == 1 for group in second for member in group.members
        )
        assert pipe.counters.groups_trained == 2


async def _entered(h, gate):
    """Wrap sampling so each attempt counts on entry, then waits on ``gate``."""
    entered = []

    async def sample(attempt):
        entered.append(attempt.spec.identity.group_id)
        await gate.wait()
        return await h.sample(attempt)

    h.runtime.on_sample = sample
    return entered


@pytest.mark.parametrize("lag", [1, 3])
async def test_async_starts_only_groups_that_can_train_fresh(config, lag):
    """Idle, the window is the batch being assembled and ``lag`` more, at full concurrency.

    Nothing has sealed: a started group holds its place from its first attempt.
    """
    h = Harness(
        config(
            mode="async",
            group_size=2,
            groups_per_step=2,
            max_concurrent_rollouts=32,
            max_steps_off_policy=lag,
        )
    )
    gate = asyncio.Event()
    entered = await _entered(h, gate)
    async with h.pipeline() as pipe:
        window = (lag + 1) * 2
        await until(lambda: len(entered) == 2 * window)
        await settle_loop(rounds=50)
        assert len(entered) == 2 * window
        assert pipe.snapshot()["admission/committed_groups"] == window
        assert pipe.snapshot()["admission/window_groups"] == window
        gate.set()


@pytest.mark.parametrize("lag", [1, 3])
@pytest.mark.parametrize("finish", ["commit_batch", "abort_batch"])
async def test_a_finished_batch_frees_its_places_in_the_window(config, lag, finish):
    h = Harness(
        config(
            mode="async",
            group_size=2,
            groups_per_step=1,
            max_concurrent_rollouts=2,
            max_steps_off_policy=lag,
        )
    )
    async with h.pipeline() as pipe:
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
        await until(lambda: pipe.counters.graded == 2 * (lag + 1))
        pipe.start_training(batch)
        # Training, the window is the ``lag`` batches after this one, already started.
        snapshot = pipe.snapshot()
        assert snapshot["admission/committed_groups"] == lag
        assert snapshot["admission/window_groups"] == lag
        with pytest.raises(ValueError, match="complete selected batch"):
            pipe.start_training(batch)
        await settle_loop(rounds=50)
        assert pipe.counters.graded == 2 * (lag + 1)
        h.version = 1
        getattr(pipe, finish)(batch)
        await until(lambda: pipe.counters.graded == 2 * (lag + 2))
        assert h.calls[-1][1] == 1  # the freed place samples the published policy
        next_batch = await asyncio.wait_for(pipe.next_batch(), 3)
        assert all(
            not pipe.driver.stale(member) for group in next_batch for member in group.members
        )
        pipe.abort_batch(next_batch)


@pytest.mark.parametrize(
    ("ahead", "batches"), [(None, 3), (1, 1), (2, 2), (0.5, 0.5), (0.25, 0.25)]
)
async def test_async_admits_only_the_batches_ahead_it_is_set_to(config, ahead, batches):
    """Idle, the batch being assembled and ``batches`` more, fractions of one too; unset,
    the cap's worth. Only whole groups start, and the gauges report the window exactly."""
    h = Harness(
        config(
            mode="async",
            group_size=2,
            groups_per_step=2,
            max_concurrent_rollouts=32,
            max_steps_off_policy=3,
            admission_steps_ahead=ahead,
        )
    )
    gate = asyncio.Event()
    entered = await _entered(h, gate)
    async with h.pipeline() as pipe:
        window = (batches + 1) * 2
        await until(lambda: len(entered) == 2 * int(window))
        await settle_loop(rounds=50)
        assert len(entered) == 2 * int(window)
        snapshot = pipe.snapshot()
        assert snapshot["admission/window_groups"] == window
        assert snapshot["admission/window_rollouts"] == window * 2
        gate.set()


async def test_a_group_older_than_the_batches_admitted_ahead_still_trains_within_the_cap(
    config,
):
    """Admitting one batch ahead does not tighten staleness: the cap alone drops groups."""
    h = Harness(
        config(
            mode="async",
            group_size=2,
            groups_per_step=1,
            max_concurrent_verifiers=4,
            max_steps_off_policy=2,
            admission_steps_ahead=1,
        )
    )
    gate = asyncio.Event()
    slow = "g0000001-"

    async def grade(attempt, rollout):
        if rollout.identity.group_id.startswith(slow):
            await gate.wait()
        return await h.grade(attempt, rollout)

    h.runtime.on_grade = grade
    async with h.pipeline() as pipe:
        for version in (1, 2):  # two batches train past the slow group, sampled at 0
            batch = await asyncio.wait_for(pipe.next_batch(), 3)
            assert not batch[0].group_id.startswith(slow)
            pipe.start_training(batch)
            h.version = version
            pipe.commit_batch(batch)
        gate.set()
        await until(lambda: pipe.counters.graded == 8)
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
        assert batch[0].group_id.startswith(slow)
        assert {member.minimum_policy_version for member in batch[0].members} == {0}
        assert not pipe.counters.discarded
        pipe.commit_batch(batch)


async def test_partially_selected_batch_cannot_start_training(config):
    h = Harness(
        config(
            mode="async",
            group_size=2,
            groups_per_step=2,
            max_concurrent_rollouts=1,
            max_steps_off_policy=1,
        )
    )
    gate = asyncio.Event()

    async def sample(attempt):
        if len(h.calls) >= 2:
            await gate.wait()
        return await h.sample(attempt)

    h.runtime.on_sample = sample
    async with h.pipeline() as pipe:
        pending = asyncio.create_task(pipe.next_batch())
        try:
            await until(lambda: len(pipe._selected) == 1)
            selected = [pipe._groups[group_id] for group_id in pipe._selected]
            before = pipe.snapshot()["admission/window_groups"]
            for attempt in (selected, selected * 2):
                with pytest.raises(ValueError, match="complete selected batch"):
                    pipe.start_training(attempt)
            assert pipe.snapshot()["admission/window_groups"] == before
            gate.set()
            batch = await asyncio.wait_for(pending, 3)
            pipe.start_training(batch)
            assert pipe.snapshot()["admission/window_groups"] == 2
            pipe.abort_batch(batch)
        finally:
            gate.set()
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


def _stragglers(h, gate, groups):
    """Hold the last member of each of ``groups`` (by index) on ``gate``."""

    async def sample(attempt):
        identity = attempt.spec.identity
        index = int(identity.group_id[1:].split("-")[0])
        last = f"-r{h.config.training.group_size - 1}"
        if index in groups and identity.rollout_id.endswith(last):
            await gate.wait()
        return await h.sample(attempt)

    h.runtime.on_sample = sample


@pytest.mark.parametrize("ahead", [2, None], ids=["ahead-below-the-cap", "ahead-at-the-cap"])
async def test_a_groups_finished_members_free_its_place_for_new_groups(config, ahead):
    """Counted by groups, a group whose last member ran long kept its place while the
    rest sat finished; below the cap its finished members' slots go to new groups.

    Two groups of four wait on one member each. Admitted two batches ahead, the pace is
    12 rollouts and those two hold 1 each, so more than the 3 groups a count by groups
    allowed start; a group filling holds only what it has left, so one completing can
    carry the count past the pace, but never past the cap's bound of 5 groups. At the
    cap (unset) that bound is all that holds, as before.
    """
    h = Harness(
        config(
            mode="async",
            group_size=4,
            groups_per_step=1,
            max_concurrent_rollouts=32,
            max_steps_off_policy=4,
            admission_steps_ahead=ahead,
        )
    )
    gate = asyncio.Event()
    _stragglers(h, gate, {0, 1})
    async with h.pipeline() as pipe:

        def settled() -> bool:  # all but the two stragglers resolved
            snapshot = pipe.snapshot()
            groups = snapshot["admission/committed_groups"]
            return groups > 3 and snapshot["admission/committed_rollouts"] == 2 + 4 * (groups - 2)

        await until(settled)
        await settle_loop(rounds=50)  # and no more start
        assert settled()
        started = pipe.snapshot()["admission/committed_groups"]
        assert started == 5 if ahead is None else 3 < started <= 5
        gate.set()


@pytest.mark.parametrize("trainer", ["waits", "has_groups_to_spare"])
async def test_async_reports_the_trainer_wait_and_the_groups_spare_when_it_asked(config, trainer):
    """What admission_steps_ahead is chosen from: a trainer that waits wants more ahead,
    one that always finds groups to spare wants fewer."""
    h = Harness(
        config(
            mode="async",
            group_size=2,
            groups_per_step=1,
            max_concurrent_rollouts=32,
            max_steps_off_policy=3,
        )
    )
    gate = asyncio.Event()

    async def grade(attempt, rollout):
        await gate.wait()
        return await h.grade(attempt, rollout)

    h.runtime.on_grade = grade
    async with h.pipeline() as pipe:
        assert "admission/batch_wait_seconds" not in pipe.snapshot()
        if trainer == "waits":
            asyncio.get_running_loop().call_later(0.2, gate.set)
        else:
            gate.set()
            await until(lambda: pipe.snapshot()["groups/ready"] == 4)  # the idle window
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
        pipe.start_training(batch)
        snapshot = pipe.snapshot()
        if trainer == "waits":
            assert snapshot["admission/batch_wait_seconds"] >= 0.2
            assert snapshot["admission/ready_spare_groups"] == -1
        else:
            assert snapshot["admission/batch_wait_seconds"] < 0.2
            assert snapshot["admission/ready_spare_groups"] == 3
        pipe.commit_batch(batch)


async def test_no_group_starts_past_the_cap_however_few_members_it_holds(config):
    """Every group waiting on a straggler holds one rollout of the pace's eight, but a
    batch place each: three, what the cap of 2 leaves room for, start and no more."""
    h = Harness(
        config(
            mode="async",
            group_size=4,
            groups_per_step=1,
            max_concurrent_rollouts=32,
            max_steps_off_policy=2,
            admission_steps_ahead=1,
        )
    )
    gate = asyncio.Event()
    _stragglers(h, gate, set(range(100)))
    async with h.pipeline() as pipe:
        await until(lambda: pipe.snapshot()["admission/committed_rollouts"] == 3)
        await settle_loop(rounds=50)
        snapshot = pipe.snapshot()
        assert snapshot["admission/committed_groups"] == 3
        assert snapshot["admission/window_rollouts"] == 8
        gate.set()


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("max_steps", [1, 2])
async def test_admission_never_outruns_the_steps_left_to_train(config, mode, max_steps):
    """A window wider than the run admits only the groups its steps train (#99)."""
    h = Harness(
        config(
            mode=mode,
            group_size=2,
            groups_per_step=2,
            max_concurrent_rollouts=32,
            max_steps=max_steps,
            **({"max_steps_off_policy": 3} if mode == "async" else {}),
        )
    )
    gate = asyncio.Event()
    entered = await _entered(h, gate)
    async with h.pipeline() as pipe:
        selecting = asyncio.create_task(pipe.next_batch())
        needed = 2 * (max_steps if mode == "async" else 1)
        await until(lambda: len(entered) == 2 * needed)
        assert pipe.counters.admitted_groups == needed
        assert pipe.counters.admitted_rollouts == 2 * needed
        gate.set()
        pipe.commit_batch(await asyncio.wait_for(selecting, 3))


@pytest.mark.parametrize("max_steps", [2, 5])
@pytest.mark.parametrize("resumed_at", [0, 7], ids=["fresh", "resumed"])
async def test_a_run_admits_exactly_the_groups_its_steps_train(config, max_steps, resumed_at):
    """Its last batches too: once the last one trains, nothing more is admitted. A
    resumed pipeline counts its steps from where it starts, as the runner's limit does."""
    h = Harness(
        config(
            mode="async",
            group_size=2,
            groups_per_step=2,
            max_concurrent_rollouts=32,
            max_steps_off_policy=3,
            max_steps=max_steps,
        )
    )
    async with h.pipeline(initial_group_index=resumed_at) as pipe:
        for step in range(max_steps):
            batch = await asyncio.wait_for(pipe.next_batch(), 3)
            pipe.start_training(batch)
            assert pipe.counters.admitted_groups <= 2 * max_steps
            h.version = step + 1
            pipe.commit_batch(batch)
        assert pipe.counters.admitted_groups == pipe.counters.groups_trained == 2 * max_steps
        assert pipe.next_group_index == resumed_at + 2 * max_steps
        assert not pipe.snapshot()["groups/outstanding"]


async def test_groups_dropped_near_the_end_are_still_replaced(config):
    """Every even group has no variance: the last batch still fills, and what is
    admitted beyond it is at most the drops and the window's allowance for them."""
    h = Harness(
        config(
            mode="async",
            group_size=2,
            groups_per_step=2,
            max_concurrent_rollouts=32,
            max_steps_off_policy=3,
            max_steps=1,
            max_consecutive_unusable_groups=100,
        )
    )
    _no_variance_in_even_groups(h)
    async with h.pipeline() as pipe:
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
        assert len(batch) == 2
        pipe.start_training(batch)
        pipe.commit_batch(batch)
        dropped = pipe.counters.discarded_groups.get(Discard.NO_VARIANCE.value, 0)
        assert dropped
        assert pipe.counters.admitted_groups <= 2 + dropped + 2


async def test_stalled_grading_does_not_admit_an_infinite_stream(config):
    h = Harness(config(mode="async", group_size=2, max_steps_off_policy=1))
    gate = asyncio.Event()

    async def grade(attempt, rollout):
        await gate.wait()
        return await h.grade(attempt, rollout)

    h.runtime.on_grade = grade
    async with h.pipeline() as pipe:
        await until(lambda: pipe.snapshot()["admission/committed_groups"] == 2)
        await settle_loop(rounds=50)
        admitted = pipe.counters.admitted_groups
        await settle_loop(rounds=50)
        assert pipe.counters.admitted_groups == admitted
        gate.set()
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
        pipe.commit_batch(batch)


@pytest.mark.parametrize(
    ("slots", "held"),
    [(8, True), (4, False)],
    ids=["window-full-slots-free", "slots-full"],
)
async def test_the_window_reports_the_time_it_holds_free_slots_idle(config, slots, held):
    """Sampler capacity left unused by freshness, not by the slots, is what is reported."""
    h = Harness(
        config(
            mode="async",
            group_size=2,
            groups_per_step=1,
            max_concurrent_rollouts=slots,
            max_steps_off_policy=1,
        )
    )
    gate = asyncio.Event()
    entered = await _entered(h, gate)
    async with h.pipeline() as pipe:
        await until(lambda: len(entered) == 4)  # the idle window: two groups of two
        pipe.take_window_held_share()
        before = pipe.snapshot()["admission/window_held_seconds"]
        await asyncio.sleep(0.1)
        grew = pipe.snapshot()["admission/window_held_seconds"] - before
        share = pipe.take_window_held_share()
        if held:
            assert grew >= 0.09 and share > 0.9
        else:
            assert grew == 0 and share == 0
        gate.set()


async def test_a_wholly_held_interval_reports_a_share_of_exactly_one(config, monkeypatch):
    """The held total and the interval's end are read at one instant, so the time
    between two clock reads lands in neither interval twice nor in the wrong one."""
    import loops_cookbook.rl.pipeline as pipeline_module

    h = Harness(
        config(
            mode="async",
            group_size=2,
            groups_per_step=1,
            max_concurrent_rollouts=8,
            max_steps_off_policy=1,
        )
    )
    gate = asyncio.Event()
    entered = await _entered(h, gate)
    async with h.pipeline() as pipe:
        await until(lambda: len(entered) == 4)  # held from here on: window full, slots free
        start, step = time.monotonic(), itertools.count(1)
        clock = itertools.accumulate(step, initial=start)  # each read later by more
        monkeypatch.setattr(pipeline_module.time, "monotonic", lambda: next(clock))
        pipe.take_window_held_share()
        share = pipe.take_window_held_share()
        monkeypatch.undo()
        assert share == 1
        gate.set()


async def test_a_judgment_that_closes_the_window_starts_the_held_clock(config, monkeypatch):
    """Survival is a window input like any other: a group judged usable can close the
    window with slots free, and the clock must run from then, not from the next change."""
    import loops_cookbook.rl.pipeline as pipeline_module

    h = Harness(
        config(
            mode="async",
            group_size=2,
            groups_per_step=6,
            max_concurrent_rollouts=64,
            max_steps_off_policy=1,
            max_consecutive_unusable_groups=100,
        )
    )

    async def grade(attempt, rollout):
        if int(rollout.identity.group_id[1:].split("-")[0]) in {0, 7}:
            return GradeResult(reward=0.0)  # unusable
        return await h.grade(attempt, rollout)

    h.runtime.on_grade = grade
    async with h.pipeline() as pipe:
        pipe.commit_batch(await asyncio.wait_for(pipe.next_batch(), 3))  # F T T T T T T
        await until(lambda: pipe.snapshot()["groups/ready"] == 14)  # the window, all idle
        # All are ready, so this selects without yielding: g7 unusable opens the
        # window to 16, then six usable, at survival 6/7, close it at 14.
        await pipe.next_batch()
        later = time.monotonic() + 1
        monkeypatch.setattr(pipeline_module.time, "monotonic", lambda: later)
        held = pipe.snapshot()["admission/window_held_seconds"]
        monkeypatch.undo()
        assert held >= 1  # the second since the judgment, not 0 from this read
        assert (
            pipe.snapshot()["admission/window_groups"]
            <= pipe.snapshot()["admission/committed_groups"]
        )


def _no_variance_in_even_groups(h):
    async def grade(attempt, rollout):
        if int(rollout.identity.group_id[1:].split("-")[0]) % 2 == 0:
            return GradeResult(reward=0.0)
        return await h.grade(attempt, rollout)

    h.runtime.on_grade = grade


async def test_the_window_widens_by_the_groups_expected_to_drop(config):
    h = Harness(
        config(
            mode="async",
            group_size=2,
            groups_per_step=2,
            max_concurrent_rollouts=4,
            max_steps_off_policy=1,
            max_consecutive_unusable_groups=100,
        )
    )
    _no_variance_in_even_groups(h)
    async with h.pipeline() as pipe:
        for version in (1, 2):
            batch = await asyncio.wait_for(pipe.next_batch(), 3)
            h.version = version
            pipe.commit_batch(batch)
        snapshot = pipe.snapshot()
        survival, window = snapshot["admission/group_survival"], snapshot["admission/window_groups"]
        assert 0.25 <= survival < 1
        fresh = 2 * 2  # idle: the batch being assembled and one more
        expected_lost = min(fresh / survival - fresh, 2)  # counted up to a batch
        assert window == math.ceil(fresh + expected_lost)


@pytest.mark.parametrize("ahead", [None, 1], ids=["unset", "at-the-cap"])
async def test_a_fraction_of_a_group_expected_to_drop_starts_a_whole_replacement(config, ahead):
    """A replacement is a whole group: at survival 3/4 a batch expects a third of a group
    to drop, and one more group starts for it, ahead set or not. With sampling held,
    every group holds its whole place, so no finished member lends a slot."""
    h = Harness(
        config(
            mode="async",
            group_size=2,
            groups_per_step=1,
            max_concurrent_rollouts=32,
            max_steps_off_policy=1,
            admission_steps_ahead=ahead,
            max_consecutive_unusable_groups=100,
        )
    )
    gate = asyncio.Event()
    gate.set()
    await _entered(h, gate)

    async def grade(attempt, rollout):
        if int(rollout.identity.group_id[1:].split("-")[0]) % 3 == 0:
            return GradeResult(reward=0.0)  # unusable
        return await h.grade(attempt, rollout)

    h.runtime.on_grade = grade
    async with h.pipeline() as pipe:
        for _ in range(2):  # the policy held, so no group goes stale
            pipe.commit_batch(await asyncio.wait_for(pipe.next_batch(), 3))
        gate.clear()
        pipe.commit_batch(await asyncio.wait_for(pipe.next_batch(), 3))
        assert 0 < 1 / pipe.snapshot()["admission/group_survival"] - 1 < 1
        # Idle: the batch being assembled and one more, and a whole group for the drops.
        await until(lambda: pipe.snapshot()["admission/committed_groups"] == 3)
        gate.set()


@pytest.mark.parametrize(
    ("fresh", "survival", "window"),
    [
        (Fraction(21), Fraction(7, 10), 30),
        (Fraction(10), Fraction(1, 2), 20),
        (Fraction(4), Fraction(1), 4),
        (Fraction(1), Fraction(3, 4), 2),
        (Fraction(3, 2), Fraction(3, 4), Fraction(5, 2)),
    ],
)
def test_the_window_admits_exactly_the_groups_expected_to_drop(fresh, survival, window):
    """Float error once rounded 21 / 0.7 - 21 = 9 up to 10, admitting a group too many."""
    assert _admission_window(fresh, survival, batch=24) == window


def test_the_expected_drops_count_up_to_a_batch():
    """A collapsing policy drops most groups: at survival 1/4, 48 fresh would add 144
    more, each training later than the last."""
    assert _admission_window(Fraction(48), Fraction(1, 4), batch=24) == 72


@pytest.mark.parametrize(
    ("needed", "judged", "survival", "limit"),
    [
        (2, 0, Fraction(1), 2),
        (2, 0, Fraction(1, 2), 4),
        (2, 2, Fraction(1, 2), 2),
        (4, 2, Fraction(1, 2), 6),
        (2, 3, Fraction(1, 2), 3),
    ],
    ids=["no-drops", "none-judged", "batch-selected", "half-judged", "judged-beyond"],
)
def test_the_steps_left_allow_drops_only_for_groups_not_yet_judged(needed, judged, survival, limit):
    """A judged group can no longer drop as unusable: with the last batch selected,
    an allowance for it admitted a whole batch that never trains."""
    assert _admission_limit(needed, judged, survival, batch=2) == limit


@pytest.mark.parametrize(
    ("spoil", "counted"),
    [("stale", False), ("no_variance", True)],
)
async def test_groups_lost_to_staleness_do_not_widen_the_window(config, spoil, counted):
    """Otherwise stale drops would lower survival, widen the window, and make more stale."""
    h = Harness(
        config(
            mode="async",
            group_size=2,
            groups_per_step=1,
            max_group_losses=0,
            max_steps_off_policy=1,
            max_consecutive_unusable_groups=100,
        )
    )
    if spoil == "no_variance":
        _no_variance_in_even_groups(h)
    async with h.pipeline() as pipe:
        if spoil == "stale":
            await until(lambda: pipe.snapshot()["groups/ready"] >= 1)
            h.version = 2  # stale before selection first judges it
        await asyncio.wait_for(pipe.next_batch(), 3)
        assert (pipe.snapshot()["admission/group_survival"] < 1) is counted


@pytest.mark.parametrize("judge", ["per_rollout", "group_grader"])
async def test_survival_counts_a_good_group_as_soon_as_a_bad_one(config, judge):
    """Counted only once trained, good groups lagged drops, and the first batch read
    every outcome as a drop: the widest window, just when nothing had trained yet."""
    h = Harness(
        config(
            mode="async",
            group_size=2,
            groups_per_step=2,
            max_concurrent_rollouts=4,
            max_steps_off_policy=1,
            max_consecutive_unusable_groups=100,
        )
    )
    kwargs = {}
    if judge == "per_rollout":
        _no_variance_in_even_groups(h)
    else:

        async def grade_group(members):
            group_id = members[0].rollout.identity.group_id
            return [0.0, 0.0 if int(group_id[1:].split("-")[0]) % 2 == 0 else 1.0]

        judging(h, grade_group)
        kwargs["group_grader"] = h.runtime
    async with h.pipeline(**kwargs) as pipe:
        if judge == "group_grader":  # judged when scored, before selection takes any

            def judged() -> tuple[int, int]:
                ready = int(pipe.snapshot()["groups/ready"])
                return ready, pipe.counters.discarded_groups.get("no_variance", 0)

            await until(lambda: min(judged()) >= 1)
            ready, dropped = judged()
            survival = pipe.snapshot()["admission/group_survival"]
            assert survival == max(0.25, ready / (ready + dropped))
        await asyncio.wait_for(pipe.next_batch(), 3)
        snapshot = pipe.snapshot()
        assert snapshot["admission/group_survival"] == 0.5
        assert snapshot["admission/window_groups"] == 6  # 4 fresh, and 2: drops up to a batch


async def test_a_group_retained_for_a_replacement_is_counted_once(config):
    """A retained group is judged again when its batch is rebuilt: one outcome still."""
    h = Harness(
        config(
            mode="async",
            group_size=2,
            groups_per_step=2,
            max_group_losses=0,
            max_steps_off_policy=1,
            max_consecutive_unusable_groups=100,
        )
    )

    async def grade(attempt, rollout):
        if rollout.identity.group_id.startswith("g0000000-"):
            return GradeResult(reward=0.0)  # the one unusable group
        return await h.grade(attempt, rollout)

    h.runtime.on_grade = grade
    async with h.pipeline() as pipe:
        kept, spoiled = await asyncio.wait_for(pipe.next_batch(), 3)  # after g0's drop
        kept.members = [
            replace(member, samples=tuple(replace(s, policy_version=2) for s in member.samples))
            for member in kept.members
        ]
        h.version = 3  # the other goes stale: the batch needs a replacement
        assert not pipe.refresh_batch([kept, spoiled])
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
        assert batch[0].group_id == kept.group_id
        # g0 dropped; kept, spoiled (before it went stale) and the replacement trained
        assert pipe.snapshot()["admission/group_survival"] == 0.75


def test_discard_units_and_surviving_members_are_not_double_counted():
    counters = Counters()
    counters.discard(Discard.ROLLOUT_FAILED, 2)
    counters.discard_group(Discard.GROUP_TOO_SMALL, 6)
    gauges = counters.gauges()
    assert gauges["discard/rollouts/total"] == 8
    assert gauges["discard/groups/total"] == 1
    assert gauges["discard/rollouts/group_too_small"] == 6


async def test_rollout_and_verifier_concurrency_are_independent(config):
    h = Harness(
        config(
            mode="sync",
            group_size=4,
            groups_per_step=2,
            max_concurrent_rollouts=2,
            max_concurrent_verifiers=1,
        )
    )
    gate = asyncio.Event()
    active_sample = active_grade = peak_sample = peak_grade = 0

    async def sample(attempt):
        nonlocal active_sample, peak_sample
        active_sample += 1
        peak_sample = max(peak_sample, active_sample)
        await asyncio.sleep(0.005)
        value = await h.sample(attempt)
        active_sample -= 1
        return value

    async def grade(attempt, rollout):
        nonlocal active_grade, peak_grade
        active_grade += 1
        peak_grade = max(peak_grade, active_grade)
        await gate.wait()
        active_grade -= 1
        return await h.grade(attempt, rollout)

    h.runtime.on_sample, h.runtime.on_grade = sample, grade
    async with h.pipeline(tasks=[task("a")]) as pipe:
        pending = asyncio.create_task(pipe.next_batch())
        await until(lambda: pipe.counters.sampled == 8)
        assert peak_sample == 2 and peak_grade == 1
        gate.set()
        await asyncio.wait_for(pending, 3)


async def test_runtime_acquired_sampling_is_limited_by_holds_not_by_sample_entry(config):
    h = Harness(
        config(mode="sync", group_size=4, max_concurrent_rollouts=1),
        phases=frozenset({Phase.SAMPLING}),
    )
    gate = asyncio.Event()
    entered = holding = peak = 0

    async def sample(attempt):
        nonlocal entered, holding, peak
        entered += 1
        await gate.wait()
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            holding += 1
            peak = max(peak, holding)
            await generate(attempt)
            await asyncio.sleep(0.002)
            holding -= 1
        return EpisodeResult()

    h.runtime.on_sample = sample
    async with h.pipeline() as pipe:
        pending = asyncio.create_task(pipe.next_batch())
        await until(lambda: entered == 4)
        gate.set()
        await asyncio.wait_for(pending, 3)
        assert peak == 1


async def test_zero_variance_is_not_an_infrastructure_failure(config):
    h = Harness(config(mode="sync", group_size=2, max_consecutive_rollout_failures=1))
    seen = 0

    async def grade(attempt, rollout):
        nonlocal seen
        seen += 1
        return GradeResult(reward=0 if seen <= 12 else float(seen % 2))

    h.runtime.on_grade = grade
    async with h.pipeline() as pipe:
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
        assert len(batch) == 1
        assert pipe.counters.discarded_groups[Discard.NO_VARIANCE.value] == 6


async def test_infrastructure_failures_wake_waiting_consumer(config):
    h = Harness(config(mode="sync", group_size=2, max_consecutive_rollout_failures=2))

    async def broken(attempt):
        raise RuntimeError("provider failed")

    h.runtime.on_sample = broken
    async with h.pipeline() as pipe:
        with pytest.raises(SystemicFailure, match="infrastructure"):
            await asyncio.wait_for(pipe.next_batch(), 3)


async def test_successful_sampling_does_not_hide_a_broken_verifier(config):
    h = Harness(config(mode="sync", group_size=2, max_consecutive_rollout_failures=3))

    async def broken(attempt, rollout):
        raise RuntimeError("verifier unavailable")

    h.runtime.on_grade = broken
    async with h.pipeline() as pipe:
        with pytest.raises(SystemicFailure, match="grading infrastructure"):
            await asyncio.wait_for(pipe.next_batch(), 3)


class _Fails:
    """Hooks that fail every rollout at one stage, and name each rollout they failed.

    The second rollout to start seals only once the run has stopped, as a sibling
    finishing at the moment another trips the stop does.
    """

    def __init__(self, stage: str, stopped: asyncio.Event) -> None:
        self.stage = stage
        self.stopped = stopped
        self.failed: set[str] = set()
        self.started = 0

    async def sample(self, attempt):
        rollout_id = attempt.spec.identity.rollout_id
        self.started += 1
        if self.started == 2:
            await self.stopped.wait()
            await generate(attempt)
            return EpisodeResult()
        if self.stage in ("sample", "cleanup"):
            self.failed.add(rollout_id)
            raise RuntimeError("provider failed")
        if self.stage == "untrained":
            await generate(attempt, "critic")
            self.failed.add(rollout_id)
            return EpisodeResult()
        if self.stage == "sample_contract":
            await generate(attempt)
            self.failed.add(rollout_id)
            return "not an episode"
        await generate(attempt)
        return EpisodeResult()

    async def grade(self, attempt, rollout):
        if self.stage not in ("grade", "grade_finish", "grade_contract"):
            return GradeResult(reward=0.0)
        self.failed.add(rollout.identity.rollout_id)
        if self.stage == "grade_contract":
            return "not a grade"
        raise RuntimeError("verifier unavailable")

    async def abort(self, attempt, reason):
        if self.stage == "cleanup":
            raise RuntimeError("sandbox release failed")

    async def finish(self, attempt, rollout, disposition):
        if self.stage not in ("finish", "grade_finish"):
            return
        if self.stage == "finish" and disposition is EpisodeDisposition.COMPLETED:
            self.failed.add(rollout.identity.rollout_id)  # graded, so it has a record
        raise RuntimeError("sandbox release failed")


@pytest.mark.parametrize("retries", [0, 2])
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    ("stage", "reason", "cause", "stop"),
    [
        ("sample", Discard.ROLLOUT_FAILED, "rollout_failed", "2 consecutive"),
        ("grade", Discard.GRADING_FAILED, "grading_failed", "2 consecutive"),
        ("untrained", Discard.UNVERSIONED, "unversioned", "2 consecutive"),
        # abort() fails too: the cleanup budget trips before the sampling one is counted
        ("cleanup", Discard.ROLLOUT_FAILED, "rollout_failed", "2 consecutive"),
        # a graded rollout's finish() fails: its group is cancelled, for that cause
        ("finish", Discard.CANCELLED, "cleanup_failed", "2 consecutive"),
        # finish() fails after grade(): the grading failure stays the cause
        ("grade_finish", Discard.GRADING_FAILED, "grading_failed", "2 consecutive"),
        # a contract violation stops the run at once, from its own rollout
        ("sample_contract", Discard.ROLLOUT_FAILED, "rollout_failed", "sample\\(\\) returned str"),
        ("grade_contract", Discard.GRADING_FAILED, "grading_failed", "grade\\(\\) returned str"),
    ],
)
async def test_the_rollout_that_stops_the_run_is_counted_once_under_its_cause(
    config, mode, stage, reason, cause, stop, retries
):
    """The failure that trips the stop is that rollout's cause, not a cancellation.

    So is a failure that finishes once the run has stopped. Every
    attempt still ends in exactly one abort() or finish(), a sibling that sealed as
    the run stopped included.
    """
    producers = {"main": {"trainable": True}, "critic": {"trainable": False}}
    h = Harness(
        config(
            mode=mode,
            group_size=4,
            max_consecutive_rollout_failures=2,
            max_rollout_retries=retries,
            producers=producers,
        )
    )
    fails = _Fails(stage, h.stopped)
    h.runtime.on_sample, h.runtime.on_grade = fails.sample, fails.grade
    h.runtime.on_abort, h.runtime.on_finish = fails.abort, fails.finish
    records = []
    with pytest.raises(SystemicFailure, match=stop):
        async with h.pipeline(on_rollout=records.append) as pipe:
            try:
                await asyncio.wait_for(pipe.next_batch(), 3)
            finally:
                # The attempts in flight at the stop finish before shutdown cancels them.
                await until(lambda: pipe.snapshot()["pipeline_calls/sample"] == 0)
    assert fails.failed
    assert collections.Counter(record["rollout_id"] for record in records).most_common(1)[0][1] == 1
    failed = [record for record in records if record["rollout_id"] in fails.failed]
    assert {record["rollout_id"] for record in failed} == fails.failed
    assert {(record["reason"], record["cause"].partition("/")[0]) for record in failed} == {
        (reason.value, cause)
    }
    if reason is not Discard.CANCELLED:  # which also counts the rollouts that never ran
        assert pipe.counters.discarded[reason.value] == len(fails.failed)
    assert sum(pipe.counters.discarded.values()) == pipe.counters.admitted_rollouts
    started = collections.Counter(id(call[1]) for call in h.runtime.calls if call[0] == "sample")
    ended = collections.Counter(
        id(call[1]) for call in h.runtime.calls if call[0] in ("abort", "finish")
    )
    assert ended == started


async def test_a_finish_that_fails_after_its_rollout_was_recorded_leaves_the_driver_nothing(
    config,
):
    """Its cause could only reach a record already written, so none is kept for it."""
    h = Harness(config(mode="sync", group_size=2, max_consecutive_rollout_failures=100))
    recorded = asyncio.Event()

    async def fails_after_the_record(attempt, rollout, disposition):
        await recorded.wait()
        raise RuntimeError("sandbox release failed")

    h.runtime.on_finish = fails_after_the_record
    async with h.pipeline() as pipe:
        (group,) = await asyncio.wait_for(pipe.next_batch(), 3)
        for member in group.members:
            pipe.trained_record(member, 0)  # the runner writes each record
        recorded.set()
        await until(lambda: pipe.driver.live_attempts == 0)
        assert [pipe.driver.note(m.identity.rollout_id) for m in group.members] == [
            RolloutNote(),
            RolloutNote(),
        ]


async def test_missing_trainable_evidence_is_a_provenance_failure(config):
    producers = {"main": {"trainable": True}, "critic": {"trainable": False}}
    h = Harness(
        config(mode="sync", group_size=2, max_consecutive_rollout_failures=2, producers=producers)
    )

    async def untrained(attempt):
        await generate(attempt, "critic")
        return EpisodeResult()

    h.runtime.on_sample = untrained
    async with h.pipeline() as pipe:
        with pytest.raises(SystemicFailure, match="no trainable policy evidence"):
            await asyncio.wait_for(pipe.next_batch(), 3)
        assert pipe.counters.discarded[Discard.UNVERSIONED.value] >= 1


async def test_refresh_filters_stale_members_and_preserves_fresh_siblings(config):
    h = Harness(config(mode="async", group_size=4, max_steps_off_policy=1))
    async with h.pipeline() as pipe:
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
        group = batch[0]
        group.members[:2] = [
            replace(
                member,
                samples=tuple(replace(sample, policy_version=1) for sample in member.samples),
            )
            for member in group.members[:2]
        ]
        h.version = 2
        assert pipe.refresh_batch(batch)
        assert len(group.members) == 2
        assert pipe.counters.discarded[Discard.STALE_BEFORE_TRAINING.value] == 2


async def test_minimum_version_of_whole_rollout_controls_freshness(config):
    h = Harness(
        config(
            mode="async", group_size=2, max_steps_off_policy=1, max_policy_delta_in_trajectory=10
        )
    )
    async with h.pipeline() as pipe:
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
        for index, member in enumerate(batch[0].members):
            extra = replace(member.samples[0], sample_id=f"later-{index}", policy_version=3)
            batch[0].members[index] = replace(member, samples=member.samples + (extra,))
        h.version = 3
        assert not pipe.refresh_batch(batch)
        assert pipe.snapshot()["groups/selected"] == 0


async def test_generation_staleness_is_not_infrastructure_failure(config):
    # One episode at a time, so each one's policy bump affects only itself.
    h = Harness(
        config(
            mode="sync",
            group_size=2,
            max_consecutive_rollout_failures=1,
            max_concurrent_rollouts=1,
        )
    )
    seen = 0

    async def sample(attempt):
        nonlocal seen
        seen += 1
        await generate(attempt)
        if seen < 7:
            h.version += 5
            try:
                await generate(attempt)
            finally:
                h.version -= 5
        return EpisodeResult()

    h.runtime.on_sample = sample
    async with h.pipeline() as pipe:
        await asyncio.wait_for(pipe.next_batch(), 3)
        assert pipe.counters.discarded[Discard.INTERNAL_DELTA.value] == 6


async def test_shutdown_drains_workers_and_clears_ownership(config):
    h = Harness(config(mode="async", group_size=2))
    pipe = h.pipeline()
    async with pipe:
        await until(lambda: pipe.counters.dispatched >= 2)
    assert all(task.done() for task in pipe._tasks)
    assert pipe.snapshot()["groups/outstanding"] == 0
    assert pipe.driver.live_attempts == 0


async def test_shutdown_cancels_in_flight_attempts_and_runs_their_abort(config):
    h = Harness(config(mode="sync", group_size=2))
    entered = asyncio.Event()

    async def hangs(attempt):
        entered.set()
        await asyncio.Event().wait()

    h.runtime.on_sample = hangs
    async with h.pipeline() as pipe:
        pending = asyncio.create_task(pipe.next_batch())
        await entered.wait()
        await asyncio.sleep(0.01)
        pending.cancel()
    assert len(h.runtime.named("abort")) == 2
    assert pipe.driver.live_attempts == 0


async def test_resume_uses_next_deterministic_group_identity(config):
    cfg = config(mode="sync", group_size=2)
    stream = TaskStream(TASKS, group_size=2, seed=cfg.rollout.shuffle_seed).groups()
    for _ in range(7):
        next(stream)
    expected = next(stream)
    async with Harness(cfg).pipeline(initial_group_index=7) as pipe:
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
        assert batch[0].group_id == expected.group_id
        assert pipe.next_group_index == 8


async def test_start_admission_orders_groups_as_they_were_dispatched(config):
    cfg = config(mode="sync", group_size=2, groups_per_step=2)
    stream = TaskStream(TASKS, group_size=2, seed=cfg.rollout.shuffle_seed).groups()
    for _ in range(3):
        next(stream)
    expected = [next(stream).group_id, next(stream).group_id]
    async with Harness(cfg).pipeline(initial_group_index=3) as pipe:
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
        assert sorted(group.group_id for group in batch) == sorted(expected)
        assert [pipe.order(group_id) for group_id in expected] == [3, 4]
        pipe.commit_batch(batch)
        # A released group has no attempt left to admit; it sorts after every live one.
        assert pipe.order(expected[0]) == pipe.next_group_index


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_a_resume_samples_the_groups_left_untrained_first_in_task_order(config, mode):
    """A checkpoint's untrained groups come first, by their place in the task stream, and
    then the stream goes on where it was: none of them is skipped or sampled twice."""
    h = Harness(config(mode=mode, group_size=2, max_steps=3))
    stream = TaskStream(TASKS, group_size=2, seed=h.config.rollout.shuffle_seed).groups()
    ids = [next(stream).group_id for _ in range(6)]
    async with h.pipeline(initial_group_index=5, replay_group_indices=[2, 4]) as pipe:
        assert pipe.pending_group_indices == [2, 4]
        trained = []
        for step in range(3):
            batch = await asyncio.wait_for(pipe.next_batch(), 3)
            pipe.start_training(batch)
            h.version = step + 1
            pipe.commit_batch(batch)
            trained += [group.group_id for group in batch]
        assert list(dict.fromkeys(group_id for group_id, _ in h.calls)) == [ids[2], ids[4], ids[5]]
        assert sorted(trained) == [ids[2], ids[4], ids[5]]
        assert (pipe.next_group_index, pipe.pending_group_indices) == (6, [])


async def test_groups_that_did_not_train_stay_pending_and_dropped_ones_do_not(config):
    """An aborted update's group and a group cancelled at shutdown are pending; a group
    dropped as unusable is not: a resume would only sample it to drop it again."""
    h = Harness(config(mode="sync", group_size=2))
    _no_variance_in_even_groups(h)
    async with h.pipeline() as pipe:
        aborted = await asyncio.wait_for(pipe.next_batch(), 3)
        assert [pipe.order(group.group_id) for group in aborted] == [1]
        pipe.start_training(aborted)
        pipe.abort_batch(aborted)
        pipe.commit_batch(await asyncio.wait_for(pipe.next_batch(), 3))  # group 3
        assert pipe.pending_group_indices == [1]
        entered = asyncio.Event()
        held = asyncio.Event()

        async def hangs(attempt):
            entered.set()
            await held.wait()
            return EpisodeResult()

        h.runtime.on_sample = hangs
        selecting = asyncio.create_task(pipe.next_batch())
        await entered.wait()
        selecting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await selecting
    assert (pipe.next_group_index, pipe.pending_group_indices) == (5, [1, 4])


async def test_double_commit_is_refused(config):
    async with Harness(config(mode="sync", group_size=2)).pipeline() as pipe:
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
        pipe.commit_batch(batch)
        with pytest.raises(ValueError, match="committed"):
            pipe.commit_batch(batch)


async def test_refresh_requeues_valid_groups_when_one_needs_replacement(config):
    h = Harness(config(mode="sync", group_size=2, groups_per_step=2))
    async with h.pipeline() as pipe:
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
        retained = batch[0].group_id
        batch[0].members = [
            replace(member, samples=tuple(replace(s, policy_version=1) for s in member.samples))
            for member in batch[0].members
        ]
        h.version = 1
        assert not pipe.refresh_batch(batch)
        replacement = await asyncio.wait_for(pipe.next_batch(), 3)
        # The valid group keeps its turn, ahead of groups that became ready after it.
        assert replacement[0].group_id == retained
        assert all(
            member.minimum_policy_version == 1 for group in replacement for member in group.members
        )


async def test_no_consumer_stops_admission_and_bounds_prefetch(config):
    h = Harness(config(mode="async", group_size=2, max_steps_off_policy=1))
    async with h.pipeline() as pipe:
        await until(lambda: pipe.snapshot()["groups/ready"] >= 1)
        await settle_loop(rounds=50)
        admitted, sampled = pipe.counters.admitted_groups, pipe.counters.sampled
        await settle_loop(rounds=50)
        assert pipe.counters.admitted_groups == admitted
        assert pipe.counters.sampled == sampled
        assert pipe.snapshot()["queue/sample_dispatch_rollouts"] <= 2
        batch = await pipe.next_batch()
        h.version = 1
        pipe.commit_batch(batch)
        await until(lambda: pipe.counters.sampled > sampled)


@pytest.mark.parametrize("primary", [None, ValueError("original failure")])
async def test_shutdown_timeout_reports_incomplete_cleanup(config, primary, caplog):
    h = Harness(config(mode="async", group_size=2, shutdown_drain_seconds=0.05))
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_abort(attempt, reason):
        started.set()
        await release.wait()

    async def hangs(attempt):
        await asyncio.Event().wait()

    h.runtime.on_sample, h.runtime.on_abort = hangs, slow_abort
    pipe = h.pipeline()
    await pipe.__aenter__()
    await until(lambda: pipe.counters.dispatched >= 1)
    try:
        await asyncio.wait_for(pipe.__aexit__(type(primary), primary, None), timeout=2)
        # abort() cannot end before release, so the drain always times out; whether it
        # had started by then depends on load, and it runs on in the background either way.
        await asyncio.wait_for(started.wait(), timeout=2)
        assert "runtime cleanup is incomplete" in caplog.text
        assert not pipe.settled
    finally:
        release.set()
        await pipe.driver.drain()


@cancellation_matrix(
    triggers=tuple(Trigger),
    points={"drain": (Landing.NOTHING, Landing.CANCEL)},
    responses=tuple(Response),
)
async def test_a_cancellation_during_the_shutdown_drain_neither_cuts_it_short_nor_is_lost(
    config, caplog, case
):
    """A pipeline left mid-sample, cancelled or ending on its own (the DEADLINE
    trigger), drains its attempts for up to shutdown_drain_seconds; a cancellation
    during that drain waits it out, reports what is still live, and still propagates."""
    grace = 0.05 if case.still_running else 30
    h = Harness(
        config(
            mode="async",  # samples ahead with no consumer
            group_size=2,
            shutdown_drain_seconds=5,
            **{"rollout.cleanup.cancellation_grace_seconds": grace},
        )
    )
    work = Work(case.response)
    h.runtime.on_sample = work
    pipe = h.pipeline()
    ended = asyncio.Event()

    async def run():
        async with pipe:
            await ended.wait()

    running = asyncio.create_task(run())
    try:
        async with asyncio.timeout(5):
            await work.running.wait()
            if case.trigger is Trigger.CANCEL:
                running.cancel()
            else:
                ended.set()
            await work.cancelled.wait()
            if case.at("drain") is Landing.CANCEL:
                running.cancel()
                await settle_loop()
            assert not running.done()  # a cancellation never cuts the drain short
            work.let_go.set()
        outcome = await outcome_within(running, 5)
        if case.cancelled:
            assert isinstance(outcome, asyncio.CancelledError)
        else:
            assert outcome is None
        if case.still_running:
            assert not pipe.settled
            assert "ignored cancellation" in caplog.text
        else:
            assert pipe.settled
            assert pipe.driver.live_attempts == 0
            assert "runtime cleanup is incomplete" not in caplog.text
    finally:
        work.end()
        await settle_loop()  # a kept hook let go ends, and its cleanup starts
        await pipe.driver.drain()


@pytest.mark.parametrize("hook", ["on_sample", "on_grade"])
@pytest.mark.parametrize("drain", ["settles", "times-out"])
async def test_every_rollout_in_flight_at_shutdown_gets_one_cancelled_record(config, hook, drain):
    """A rollout still in sample() or grade() when the pipeline closes is recorded as
    cancelled, once, whether or not the drain settles within shutdown_drain_seconds."""
    seconds = 5 if drain == "settles" else 0.05
    h = Harness(config(mode="async", group_size=2, shutdown_drain_seconds=seconds))
    work, started = Work(Response.STOPS), set()

    async def in_flight(attempt, *rest):
        started.add(attempt.spec.identity.rollout_id)
        await work()

    setattr(h.runtime, hook, in_flight)
    records = []
    pipe = h.pipeline(on_rollout=records.append)
    await pipe.__aenter__()
    try:
        await until(lambda: len(started) >= 2)
        if drain == "settles":
            work.let_go.set()
        await outcome_within(asyncio.ensure_future(pipe.__aexit__()), 5)
        assert pipe.settled is (drain == "settles")
    finally:
        work.end()
        await pipe.driver.drain()
    await settle_loop()
    ids = [record["rollout_id"] for record in records]
    assert sorted(ids) == sorted(set(ids))
    cancelled = {r["rollout_id"]: r for r in records if r["reason"] == "cancelled"}
    assert started <= cancelled.keys()
    if hook == "on_grade":  # sealed, so the record names the attempt
        assert all(cancelled[rollout_id]["attempt_id"] for rollout_id in started)


async def test_unexpected_drain_error_surfaces(config):
    pipe = Harness(config()).pipeline()

    async def broken_drain():
        raise TypeError("broken cleanup implementation")

    pipe._drain = broken_drain
    with pytest.raises(TypeError, match="broken cleanup implementation"):
        await pipe.__aexit__()


def test_first_fatal_error_wins(config):
    pipe = Harness(config()).pipeline()
    primary = RuntimeError("first failure")
    pipe._stop(primary)
    pipe._stop(RuntimeError("later failure"))
    assert pipe._fatal is primary


async def test_one_outage_prints_one_traceback_not_one_per_rollout(config, caplog):
    """A sandbox outage fails every rollout in flight with the same error."""
    caplog.set_level(logging.ERROR)
    h = Harness(config(mode="sync", group_size=2, max_consecutive_rollout_failures=20))

    async def broken(attempt):
        raise OSError(24, "Too many open files", f"/cache/{attempt.spec.identity.rollout_id}")

    h.runtime.on_sample = broken
    async with h.pipeline() as pipe:
        with pytest.raises(SystemicFailure, match="infrastructure"):
            await asyncio.wait_for(pipe.next_batch(), 3)
    failures = [r for r in caplog.records if "sampling failed" in r.message]
    assert len(failures) >= 5, "the test needs several failures to be meaningful"
    assert len([r for r in failures if r.exc_info]) == 1
    assert all("Too many open files" in r.getMessage() for r in failures)


async def test_the_live_group_states_partition_every_group_held(config):
    h = Harness(config(mode="sync", group_size=2, groups_per_step=1))
    async with h.pipeline() as pipe:
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
        states, outstanding = live_group_states(pipe)
        assert states["selected"] == len(batch)
        assert sum(states.values()) == outstanding
        pipe.commit_batch(batch)
        assert pipe.snapshot()["groups/selected"] == 0


def live_group_states(pipe):
    """The live group states from one snapshot, and the total they must partition."""
    gauges = pipe.snapshot()
    states = ("filling", "judging", "ready", "selected")
    return {state: gauges[f"groups/{state}"] for state in states}, gauges["groups/outstanding"]


async def test_a_group_being_judged_is_judging_not_filling(config):
    h = Harness(config(mode="sync", group_size=2, max_concurrent_verifiers=1))
    judging_now, release = asyncio.Event(), asyncio.Event()

    async def judge(members):
        judging_now.set()
        await release.wait()
        return [0.0, 1.0]

    judging(h, judge)
    async with h.pipeline(group_grader=h.runtime) as pipe:
        batch = asyncio.create_task(pipe.next_batch())  # sync admits on demand
        await asyncio.wait_for(judging_now.wait(), 3)
        states, outstanding = live_group_states(pipe)
        # Being judged, or complete and waiting for the judge's one slot.
        assert states["judging"] == 1 + pipe.snapshot()["queue/group_grading_groups"]
        assert states["filling"] == 0  # sync admitted this one group, and it is complete
        assert sum(states.values()) == outstanding
        release.set()
        batch = await asyncio.wait_for(batch, 3)
        states, outstanding = live_group_states(pipe)
        assert states["selected"] == 1 and sum(states.values()) == outstanding
        pipe.commit_batch(batch)


async def test_a_batchs_valid_groups_keep_their_turn_ahead_of_newer_ready_groups(config):
    h = Harness(config(mode="async", group_size=2, groups_per_step=2, max_steps_off_policy=2))
    async with h.pipeline() as pipe:
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
        await until(lambda: pipe.snapshot()["groups/ready"] >= 1)  # a newer group is ready
        retained = batch[0].group_id
        batch[1].members = batch[1].members[:1]  # too few survivors: needs a replacement
        assert not pipe.refresh_batch(batch)
        replacement = await asyncio.wait_for(pipe.next_batch(), 3)
        assert replacement[0].group_id == retained
        pipe.commit_batch(replacement)


def first_group_finishes_last(h, *, others_at=0):
    """The first group dispatched samples at once, then holds until ``release`` is set;
    every other group samples once the policy is at ``others_at``."""
    release, sampled = asyncio.Event(), []

    async def sample(attempt):
        if attempt.spec.identity.group_id.startswith("g0000000-"):
            result = await h.sample(attempt)
            sampled.append(attempt.spec.identity.rollout_id)
            await release.wait()
            return result
        await until(lambda: h.version >= others_at)
        return await h.sample(attempt)

    h.runtime.on_sample = sample
    return release, sampled


async def test_the_ready_group_sampled_at_the_oldest_policy_trains_first_and_not_stale(config):
    h = Harness(config(mode="async", group_size=2, max_steps_off_policy=1))
    release, sampled = first_group_finishes_last(h, others_at=1)
    async with h.pipeline() as pipe:
        await until(lambda: len(sampled) == 2)  # the first group sampled at version 0
        h.version = 1
        await until(lambda: pipe.snapshot()["groups/ready"] >= 1)  # a version-1 group
        release.set()
        await until(lambda: pipe.snapshot()["groups/ready"] >= 2)
        # Completion order would train the version-1 group, and leave the first to go
        # stale at the next update.
        (oldest,) = await asyncio.wait_for(pipe.next_batch(), 3)
        assert oldest.group_id.startswith("g0000000-")
        assert {m.minimum_policy_version for m in oldest.members} == {0}
        pipe.commit_batch([oldest])
        h.version = 2
        (newer,) = await asyncio.wait_for(pipe.next_batch(), 3)
        assert {m.minimum_policy_version for m in newer.members} == {1}
        pipe.commit_batch([newer])
        assert Discard.STALE_BEFORE_TRAINING.value not in pipe.counters.discarded


async def test_ready_groups_at_one_policy_version_train_in_dispatch_order(config):
    # A one-step window: once the later group is ready, no third group starts.
    h = Harness(config(mode="async", group_size=2, max_steps_off_policy=1))
    release, sampled = first_group_finishes_last(h)
    async with h.pipeline() as pipe:
        await until(lambda: len(sampled) == 2)
        await until(lambda: pipe.snapshot()["groups/ready"] >= 1)  # a later group
        release.set()
        await until(lambda: pipe.snapshot()["groups/ready"] >= 2)
        (first,) = await asyncio.wait_for(pipe.next_batch(), 3)
        assert first.group_id.startswith("g0000000-")
        pipe.commit_batch([first])


async def test_held_over_groups_go_ahead_of_ready_groups_sampled_at_older_policies(config):
    h = Harness(config(mode="async", group_size=2, groups_per_step=2, max_steps_off_policy=2))
    async with h.pipeline() as pipe:
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
        await until(lambda: pipe.snapshot()["groups/ready"] >= 1)  # sampled at version 0
        h.version = 1
        retained = batch[0]
        retained.members = [
            replace(member, samples=tuple(replace(s, policy_version=1) for s in member.samples))
            for member in retained.members
        ]
        batch[1].members = batch[1].members[:1]  # too few survivors: needs a replacement
        assert not pipe.refresh_batch(batch)
        replacement = await asyncio.wait_for(pipe.next_batch(), 3)
        assert replacement[0].group_id == retained.group_id
        assert {m.minimum_policy_version for m in replacement[1].members} == {0}
        pipe.commit_batch(replacement)


async def test_async_a_member_gone_stale_after_the_judge_is_dropped_and_its_siblings_keep_ranks(
    config,
):
    """The async caveat: ranks computed with a member present stand once it is dropped."""
    h = Harness(
        config(
            mode="async",
            group_size=4,
            max_group_losses=2,
            max_concurrent_rollouts=1,
            max_steps_off_policy=1,
        )
    )
    judged: dict[str, dict[str, float]] = {}

    async def sample(attempt):
        if not attempt.spec.identity.rollout_id.endswith("-r0"):
            # r0 alone is sampled a policy version earlier; the judge's version stands.
            h.version = max(h.version, 1)
        return await h.sample(attempt)

    async def grade_group(members):
        rewards = [0.1, 0.2, 0.3, 0.4][: len(members)]
        group_id = members[0].rollout.identity.group_id
        judged[group_id] = {
            member.rollout.identity.rollout_id: reward
            for member, reward in zip(members, rewards, strict=True)
        }
        h.version = 2  # the policy moves on between the judge and selection
        return rewards

    h.runtime.on_sample = sample
    judging(h, grade_group)
    async with h.pipeline(group_grader=h.runtime) as pipe:
        (group,) = await asyncio.wait_for(pipe.next_batch(), 3)
        ranks = judged[group.group_id]
        assert len(ranks) == 4  # judged with every member present
        survivors = {member.identity.rollout_id: member.reward for member in group.members}
        assert f"{group.group_id}-r0" not in survivors
        assert survivors == {key: value for key, value in ranks.items() if not key.endswith("-r0")}
        assert pipe.counters.discarded[Discard.STALE_BEFORE_TRAINING.value] == 1
        pipe.commit_batch([group])


async def test_a_group_dropped_by_selections_recheck_counts_toward_the_unusable_streak(config):
    h = Harness(
        config(
            mode="async", group_size=2, max_steps_off_policy=1, max_consecutive_unusable_groups=1
        )
    )
    async with h.pipeline() as pipe:
        batch = await asyncio.wait_for(pipe.next_batch(), 3)
        h.version = 5  # every member is stale by the time advantages are computed
        with pytest.raises(UnusableGroupsError, match="1 consecutive unusable groups"):
            pipe.refresh_batch(batch)
        assert pipe.counters.discarded_groups[Discard.GROUP_TOO_SMALL.value] == 1


async def test_a_group_grader_call_holds_a_verification_slot(config):
    h = Harness(config(mode="sync", group_size=2, max_concurrent_verifiers=1))
    seen: list[dict[str, float]] = []
    holder: list[Pipeline] = []

    async def grade_group(members):
        seen.append(holder[0].capacity.snapshot())
        return [0.0, 1.0][: len(members)]

    judging(h, grade_group)
    async with h.pipeline(group_grader=h.runtime) as pipe:
        holder.append(pipe)
        (group,) = await asyncio.wait_for(pipe.next_batch(), 3)
        pipe.commit_batch([group])
    assert seen and seen[0]["slots/verification"] == 1 == seen[0]["slots/verification_limit"]


@pytest.mark.parametrize(("losses", "tries"), [(2, 2), (0, 1)], ids=["tried-again", "too-small"])
async def test_staleness_is_rechecked_before_every_try_of_the_group_grader(config, losses, tries):
    """A member gone stale during the backoff is dropped before the next try, and a group
    it leaves too small is dropped without paying the grader again."""
    h = Harness(
        config(
            mode="async",
            group_size=4,
            max_group_losses=losses,
            max_concurrent_rollouts=1,
            max_steps_off_policy=1,
            max_consecutive_unusable_groups=0,
        )
    )
    seen: list[list[str]] = []

    async def sample(attempt):
        if not attempt.spec.identity.rollout_id.endswith("-r0"):
            h.version = max(h.version, 1)  # r0 alone is sampled a policy version earlier
        return await h.sample(attempt)

    async def grade_group(members):
        seen.append([member.rollout.identity.rollout_id for member in members])
        if len(seen) == 1:
            h.version = 2  # r0 goes stale during the backoff
            raise RuntimeError("a blip")
        return [float(index) for index in range(len(members))]

    h.runtime.on_sample = sample
    judging(h, grade_group, retries=1)

    def tries_of_first():
        first = seen[0][0].rsplit("-r", 1)[0]
        return [ids for ids in seen if ids[0].startswith(first)]

    async with h.pipeline(group_grader=h.runtime) as pipe:
        await until(lambda: seen)
        if tries == 2:
            await until(lambda: len(tries_of_first()) == 2)
        else:  # dropped once r0 is gone, so never tried again
            await until(lambda: pipe.counters.discarded_groups.get(Discard.GROUP_TOO_SMALL.value))
        assert pipe.counters.discarded[Discard.STALE_BEFORE_TRAINING.value] == 1
        judged = tries_of_first()
        assert len(judged) == tries
        assert len(judged[0]) == 4
        if tries == 2:
            assert judged[1] == [i for i in judged[0] if not i.endswith("-r0")]


class Clock:
    """The group stage's clock, moved only by the test."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr("loops_cookbook.runtime.groups.time", clock)
    return clock


async def test_shutdown_during_a_judge_that_stops_releases_its_slot_and_ends_settled(config):
    h = Harness(config(mode="sync", group_size=2, max_concurrent_verifiers=1))
    started = asyncio.Event()

    async def judge(members):
        started.set()
        await asyncio.sleep(30)  # cancelled by the shutdown, and stops
        return [0.0, 1.0]

    judging(h, judge)
    records = []
    async with h.pipeline(group_grader=h.runtime, on_rollout=records.append) as pipe:
        batch = asyncio.create_task(pipe.next_batch())  # sync admits on demand
        await asyncio.wait_for(started.wait(), 3)
    batch.cancel()
    assert pipe.settled and not pipe.driver.unquiesced
    assert pipe.capacity.snapshot()["slots/verification"] == 0
    assert len(records) == 2
    assert {(record["reason"], record["reward"]) for record in records} == {("cancelled", None)}


async def test_a_drain_timeout_during_a_judge_still_records_its_group_as_cancelled(config):
    """The judge stops within its grace, but not within the drain: its group's records
    are written all the same, not lost with the drain that stopped waiting."""
    h = Harness(
        config(
            mode="sync",
            group_size=2,
            max_concurrent_verifiers=1,
            shutdown_drain_seconds=0.05,
            **{"rollout.cleanup.cancellation_grace_seconds": 5},
        )
    )
    started, stopped = asyncio.Event(), asyncio.Event()

    async def judge(members):
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.sleep(0.3)  # past the drain, inside its grace
            stopped.set()
            raise
        return [0.0, 1.0]

    judging(h, judge)
    records = []
    async with h.pipeline(group_grader=h.runtime, on_rollout=records.append) as pipe:
        batch = asyncio.create_task(pipe.next_batch())  # sync admits on demand
        await asyncio.wait_for(started.wait(), 3)
    batch.cancel()
    assert not pipe.settled  # the drain did time out
    assert len(records) == 2
    assert {(record["reason"], record["reward"]) for record in records} == {("cancelled", None)}
    assert pipe.counters.discarded_groups == {Discard.CANCELLED.value: 1}
    await asyncio.wait_for(stopped.wait(), 3)
    assert len(records) == 2  # once each, when the judge does stop


async def test_the_group_grader_gauges_measure_its_calls_not_the_backoff(config, clock):
    h = Harness(config(mode="sync", group_size=2))
    backing_off, resume = asyncio.Event(), asyncio.Event()
    tries = 0

    async def backoff(seconds):
        backing_off.set()
        await resume.wait()

    async def judge(members):
        nonlocal tries
        tries += 1
        if tries == 1:
            raise RuntimeError("a blip")  # then the stage backs off
        return [0.0, 1.0]

    h.sleep = backoff
    judging(h, judge, retries=1)
    async with h.pipeline(group_grader=h.runtime) as pipe:
        batch = asyncio.create_task(pipe.next_batch())
        await asyncio.wait_for(backing_off.wait(), 3)
        during = pipe.snapshot()
        assert during["pipeline_calls/grade_group"] == 0  # no call runs in the backoff
        assert during["time/pipeline/group_grading/count"] == 1  # the first ended before it
        clock.now += 100  # the backoff takes its time
        resume.set()
        (group,) = await asyncio.wait_for(batch, 3)
        timed = pipe.snapshot()
        assert timed["time/pipeline/group_grading/count"] == 2  # one per call
        assert timed["time/pipeline/group_grading/max_seconds"] == 0  # never the backoff
        pipe.commit_batch([group])


async def test_sync_replaces_a_group_the_group_grader_failed(config):
    h = Harness(config(mode="sync", group_size=2, max_consecutive_rollout_failures=2))
    judged: list[str] = []

    async def judge(members):
        judged.append(members[0].rollout.identity.group_id)
        if len(judged) == 1:
            raise PermanentGradingFailure("this group cannot be judged")
        return [0.0, 1.0]

    judging(h, judge)
    async with h.pipeline(group_grader=h.runtime) as pipe:
        (group,) = await asyncio.wait_for(pipe.next_batch(), 3)
        assert judged == [judged[0], group.group_id] and group.group_id != judged[0]
        assert pipe.counters.discarded_groups == {Discard.GRADING_FAILED.value: 1}
        states, outstanding = live_group_states(pipe)  # the failed group left the stage
        assert states["judging"] == 0 and sum(states.values()) == outstanding
        pipe.commit_batch([group])


async def test_a_grade_that_raises_its_own_cancellation_fails_only_its_rollout(config):
    h = Harness(config(mode="sync", group_size=4, max_consecutive_rollout_failures=2))
    grade = h.grade

    async def leaks(attempt, rollout):
        if rollout.identity.rollout_id.endswith("-r0"):
            raise asyncio.CancelledError  # nothing cancelled it
        return await grade(attempt, rollout)

    h.runtime.on_grade = leaks
    records = []
    async with h.pipeline(on_rollout=records.append) as pipe:
        (group,) = await asyncio.wait_for(pipe.next_batch(), 3)  # the worker lived on
        assert len(group.members) == 3
        (failed,) = [record for record in records if record["reason"] == "grading_failed"]
        assert failed["rollout_id"].endswith("-r0")
        assert failed["cause"] == "grading_failed/cancelled"
        pipe.commit_batch([group])


async def test_a_worker_lost_to_a_cancellation_nothing_asked_for_stops_the_pipeline(config):
    """No runtime hook's cancellation reaches a worker; the pipeline's own record sink
    runs on one, so it stands in for anything that would."""
    h = Harness(config(mode="sync", group_size=2, max_consecutive_rollout_failures=0))

    async def fails(attempt):
        raise ConnectionError("the environment went away")

    def sink(record):
        raise asyncio.CancelledError

    h.runtime.on_sample = fails
    async with h.pipeline(on_rollout=sink) as pipe:
        with pytest.raises(RuntimeError, match=r"pipeline worker sample-\d+ was cancelled"):
            await asyncio.wait_for(pipe.next_batch(), 3)


async def test_a_group_waiting_for_a_verification_slot_is_not_a_group_grader_call(config, clock):
    h = Harness(
        config(mode="sync", group_size=2, max_concurrent_verifiers=1),
        phases=frozenset({Phase.VERIFICATION}),  # grade() takes no slot: only the judge does
    )

    async def judge(members):
        return [0.0, 1.0]

    judging(h, judge)
    async with h.pipeline(group_grader=h.runtime) as pipe:
        async with pipe.capacity.group_verification():  # the only slot, held elsewhere
            batch = asyncio.create_task(pipe.next_batch())
            await until(lambda: pipe.capacity.snapshot()["queue/verification_waiting"] == 1)
            assert pipe.snapshot()["pipeline_calls/grade_group"] == 0
            clock.now += 100  # the wait takes its time
        (group,) = await asyncio.wait_for(batch, 3)
        timed = pipe.snapshot()
        assert timed["time/pipeline/group_grading/count"] == 1
        assert timed["time/pipeline/group_grading/max_seconds"] == 0  # not the wait
        pipe.commit_batch([group])


async def test_a_sample_kept_past_its_grace_costs_its_group_one_member_not_the_run(config):
    """nwx9k5w: one sample() that ignores cancellation discards only its own rollout."""
    h = Harness(
        config(
            mode="sync",
            group_size=4,
            max_group_losses=2,
            rollout_timeout_seconds=0.05,
            **{"rollout.cleanup.cancellation_grace_seconds": 0.05},
        )
    )
    release, stubborn = asyncio.Event(), set()

    async def sample(attempt):
        if not stubborn:
            stubborn.add(attempt.spec.identity.rollout_id)
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
        return await h.sample(attempt)

    h.runtime.on_sample = sample
    records = []
    try:
        async with h.pipeline(on_rollout=records.append) as pipe:
            (group,) = await asyncio.wait_for(pipe.next_batch(), 3)
            [kept] = stubborn
            assert kept not in {member.identity.rollout_id for member in group.members}
            assert len(group.members) == 3
            assert not h.stopped.is_set()
            assert pipe.counters.discarded == {Discard.UNQUIESCED.value: 1}
            assert [(r["rollout_id"], r["cause"]) for r in records] == [(kept, "unquiesced/sample")]
            pipe.commit_batch([group])
            release.set()  # it ends inside the run: the run's drain waits for its abort()
        assert pipe.settled
    finally:
        release.set()
