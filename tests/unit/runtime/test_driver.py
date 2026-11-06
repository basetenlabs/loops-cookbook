"""The attempt driver's lifecycle: grading, terminal hooks, deadlines, retries, quiescence."""

import asyncio
import contextlib
import gc
import logging
import signal
import sys
import time
from fractions import Fraction

import pytest
from support.cancellation import settle_loop
from support.harness import TASK, Renderer, Runtime, default_grade, default_sample, generate, rig

from loops_cookbook.cancellation import Waited
from loops_cookbook.evidence import AttemptIdentity, Rollout, StaleRolloutError
from loops_cookbook.llm.client import Renderers
from loops_cookbook.model_access.api import CaptureIntegrityFailure, ModelTransportFailure
from loops_cookbook.model_access.lease import DirectModelAccess, LeaseState
from loops_cookbook.model_access.producers import ProducerSettings
from loops_cookbook.runtime.api import (
    AttemptAbortReason,
    EpisodeDisposition,
    EpisodeResult,
    ExcludedReward,
    GradeResult,
    NamedFailure,
    PermanentGradingFailure,
    PermanentRolloutFailure,
    Phase,
    ProcessCommand,
)
from loops_cookbook.runtime.capacity import (
    AttemptPhases,
    CapacityRevoked,
    Hook,
    RuntimeContractError,
)
from loops_cookbook.runtime.driver import (
    Discard,
    LoopsRunEnded,
    NonQuiescentHook,
    RolloutNote,
    RuntimeExecutionFailure,
    SystemicFailure,
    cleanup_failures,
)
from loops_cookbook.runtime.processes import ProcessesClosed

SAMPLING = frozenset({Phase.SAMPLING})
BOTH = frozenset({Phase.SAMPLING, Phase.VERIFICATION})


async def test_every_valid_rollout_is_graded_then_finished_with_its_original_context():
    r = rig()
    graded = await r.rollout()
    (_, sampled), (_, graded_ctx, offered), (_, finished_ctx, final, disposition) = (
        r.runtime.named("sample")[0],
        r.runtime.named("grade")[0],
        r.runtime.named("finish")[0],
    )
    assert offered.reward is None
    assert graded.reward == 1.0 and final is graded
    assert disposition is EpisodeDisposition.COMPLETED
    assert sampled is graded_ctx is finished_ctx
    assert dict(graded.metadata) == {"answer": "episode", "verdict": "grade"}
    assert graded.samples == offered.samples and graded.identity == offered.identity
    assert not r.runtime.named("abort")


async def test_the_committed_rollout_does_not_wait_for_finish():
    release = asyncio.Event()

    async def slow_finish(*args):
        await release.wait()

    r = rig(Runtime(on_finish=slow_finish))
    admitted = await r.driver.sample(TASK, "g", "g-r1")
    graded = await asyncio.wait_for(r.driver.grade(admitted), 1)
    assert graded.reward == 1.0
    assert r.driver.live_attempts == 1  # the permit is held until cleanup settles
    release.set()
    await r.driver.drain()
    assert r.driver.live_attempts == 0


async def test_a_sample_failure_aborts_once_and_never_finishes():
    async def failing(attempt):
        raise KeyError("runtime bug")

    r = rig(Runtime(on_sample=failing))
    assert await r.rollout() is Discard.ROLLOUT_FAILED
    assert [call[2] for call in r.runtime.named("abort")] == [AttemptAbortReason.SAMPLE_FAILED]
    assert not r.runtime.named("finish") and not r.runtime.named("grade")
    assert r.driver.live_attempts == 0


async def test_a_generation_the_runtime_abandons_leaves_no_evidence():
    async def hanging(prompt, **kwargs):
        await asyncio.Event().wait()

    async def swallowing(attempt):
        r.sampler.sample_async = hanging  # type: ignore[method-assign]
        try:
            await asyncio.wait_for(generate(attempt), 0.02)
        except TimeoutError:
            pass
        return EpisodeResult()

    r = rig(Runtime(on_sample=swallowing))
    assert await r.rollout() is Discard.UNVERSIONED  # nothing delivered, nothing to train
    assert not r.runtime.named("abort")  # sealed and finished, not a capture failure


async def test_a_rollout_without_trainable_evidence_is_unversioned_and_finished():
    async def critic_only(attempt):
        await generate(attempt, "critic")
        return EpisodeResult()

    r = rig(
        Runtime(on_sample=critic_only),
        producers=ProducerSettings({"main": True, "critic": False}),
    )
    assert await r.rollout() is Discard.UNVERSIONED
    assert r.runtime.named("finish")[0][3] is EpisodeDisposition.DISCARDED
    assert not r.runtime.named("grade")


async def test_a_rollout_whose_agent_discarded_every_generation_is_graded_and_trains_nothing():
    async def cut_off(attempt):
        conversation = attempt.services.model_access.llm().conversation()
        reply = await conversation.generate(messages=[{"role": "user", "content": "hi"}])
        conversation.discard(reply)
        return EpisodeResult()

    r = rig(Runtime(on_sample=cut_off))
    r.version[0] = 3
    rollout = await r.rollout()
    assert isinstance(rollout, Rollout) and rollout.reward is not None
    assert not rollout.trains and rollout.samples == ()
    assert rollout.minimum_policy_version == 3  # dated by what it discarded
    assert r.runtime.named("finish")[0][3] is EpisodeDisposition.COMPLETED
    r.version[0] = 3 + 3 + 1  # past the rig's max_steps_off_policy of 3
    assert r.driver.stale(rollout)  # and goes stale like any other


async def test_a_discarded_generation_from_an_untrained_producer_dates_nothing():
    async def critic_only(attempt):
        conversation = attempt.services.model_access.llm("critic").conversation()
        reply = await conversation.generate(messages=[{"role": "user", "content": "hi"}])
        conversation.discard(reply)
        return EpisodeResult()

    r = rig(
        Runtime(on_sample=critic_only),
        producers=ProducerSettings({"main": True, "critic": False}),
    )
    assert await r.rollout() is Discard.UNVERSIONED


@pytest.mark.parametrize("propagate", [True, False], ids=["propagated", "swallowed"])
async def test_a_stale_generation_is_discarded_whether_or_not_it_propagates(propagate):
    async def stale(attempt):
        await generate(attempt)
        r.version[0] = 5
        try:
            await generate(attempt)
        except StaleRolloutError:
            if propagate:
                raise
        return EpisodeResult()

    r = rig(
        Runtime(on_sample=stale),
        max_steps_off_policy=1,
        max_policy_delta_in_trajectory=9,
        max_rollout_retries=3,
    )
    assert await r.rollout() is Discard.STALE_BEFORE_GRADING
    assert [call[2] for call in r.runtime.named("abort")] == [AttemptAbortReason.STALE]
    assert len(r.runtime.named("sample")) == 1  # never retried


async def test_an_internal_policy_delta_during_the_episode_is_discarded():
    async def spread(attempt):
        await generate(attempt)
        r.version[0] = 3
        await generate(attempt)
        return EpisodeResult()

    r = rig(Runtime(on_sample=spread), max_policy_delta_in_trajectory=1, max_steps_off_policy=9)
    assert await r.rollout() is Discard.INTERNAL_DELTA
    assert r.runtime.named("abort")[0][2] is AttemptAbortReason.STALE


@pytest.mark.parametrize("swallowed", [False, True], ids=["propagated", "swallowed"])
async def test_a_future_policy_version_aborts_before_stopping_the_run(swallowed):
    async def future(attempt):
        r.sampler.version = lambda: 9
        if swallowed:
            with contextlib.suppress(Exception):
                await generate(attempt)
        else:
            await generate(attempt)
        return EpisodeResult()

    r = rig(Runtime(on_sample=future))
    assert await r.rollout() is Discard.ROLLOUT_FAILED  # its own outcome, and it stops the run
    assert r.systemic
    assert r.runtime.named("abort")[0][2] is AttemptAbortReason.CAPTURE_FAILED
    assert r.driver.live_attempts == 0


# ------------------------------------------------------------------ deadlines


async def test_the_rollout_clock_counts_only_sampling_holds():
    async def slow_outside_holds(attempt):
        await asyncio.sleep(0.15)  # setup, outside any hold
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            await generate(attempt)
        return EpisodeResult()

    r = rig(
        Runtime(on_sample=slow_outside_holds),
        phases=SAMPLING,
        rollout_timeout_seconds=0.1,
        setup_timeout_seconds=1.0,
    )
    graded = await r.rollout()
    assert graded.reward == 1.0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
async def test_abort_may_run_processes_after_the_sample_it_follows_was_stopped(tmp_path):
    """Processes are refused only while a hook is being stopped, not after."""
    exits = []

    async def hangs(attempt):
        await asyncio.Event().wait()

    async def aborts(attempt, reason):
        command = ProcessCommand(argv=(sys.executable, "-c", "pass"), cwd=tmp_path)
        exits.append((await attempt.services.processes.run(command)).exit_code)

    r = rig(Runtime(on_sample=hangs, on_abort=aborts), rollout_timeout_seconds=0.05)
    assert await r.rollout() is Discard.AGENT_TIMEOUT
    assert exits == [0]


async def test_the_rollout_clock_cancels_sample_and_is_a_non_retried_agent_timeout():
    cancelled = asyncio.Event()

    async def hangs(attempt):
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    r = rig(
        Runtime(on_sample=hangs),
        phases=SAMPLING,
        rollout_timeout_seconds=0.05,
        max_rollout_retries=3,
    )
    assert await r.rollout() is Discard.AGENT_TIMEOUT
    assert cancelled.is_set()
    assert [call[2] for call in r.runtime.named("abort")] == [AttemptAbortReason.ROLLOUT_TIMEOUT]
    assert len(r.runtime.named("sample")) == 1


async def test_a_runner_owned_sampling_hold_counts_setup_inside_sample_on_the_rollout_clock():
    async def slow_setup(attempt):
        await asyncio.sleep(10)
        return EpisodeResult()

    r = rig(Runtime(on_sample=slow_setup), rollout_timeout_seconds=0.05)
    assert await r.rollout() is Discard.AGENT_TIMEOUT
    assert r.driver.gauges() == {"discard/rollouts/agent_timeout/rollout_clock": 1.0}


async def test_the_setup_clock_bounds_time_outside_holds_and_is_retry_eligible():
    async def slow_setup(attempt):
        await asyncio.sleep(1)

    r = rig(
        Runtime(on_sample=slow_setup),
        phases=SAMPLING,
        setup_timeout_seconds=0.05,
        max_rollout_retries=1,
    )
    assert await r.rollout() is Discard.SETUP_TIMEOUT
    assert [call[2] for call in r.runtime.named("abort")] == [AttemptAbortReason.SETUP_TIMEOUT] * 2


async def test_rollout_clock_expiries_between_real_failures_count_toward_the_stop():
    calls = 0

    async def clock_or_own_deadline(attempt):
        nonlocal calls
        calls += 1
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            if calls % 2 == 0:
                await asyncio.Event().wait()  # the rollout clock cancels it
            raise TimeoutError("the runtime's own deadline")

    r = rig(
        Runtime(on_sample=clock_or_own_deadline),
        phases=SAMPLING,
        rollout_timeout_seconds=0.05,
        max_consecutive_failures=4,
    )
    outcomes = [await r.driver.sample(TASK, "g", f"g-r{i}") for i in range(3)]
    # a runtime's own timeout is a rollout failure; only the clock is an agent timeout
    assert outcomes == [Discard.ROLLOUT_FAILED, Discard.AGENT_TIMEOUT, Discard.ROLLOUT_FAILED]
    assert not r.systemic
    assert await r.driver.sample(TASK, "g", "g-r3") is Discard.AGENT_TIMEOUT
    assert "4 consecutive sampling" in str(r.systemic[0])


@pytest.mark.parametrize("retries", [0, 2])
async def test_consecutive_rollout_clock_expiries_stop_the_run_naming_the_clock(retries):
    async def hangs_past_the_clock(attempt):
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            await asyncio.Event().wait()

    r = rig(
        Runtime(on_sample=hangs_past_the_clock),
        phases=SAMPLING,
        rollout_timeout_seconds=0.05,
        max_rollout_retries=retries,
        max_consecutive_failures=3,
    )
    assert [await r.driver.sample(TASK, "g", f"g-r{i}") for i in range(2)] == [
        Discard.AGENT_TIMEOUT
    ] * 2
    assert await r.driver.sample(TASK, "g", "g-r2") is Discard.AGENT_TIMEOUT
    assert "3 consecutive sampling" in str(r.systemic[0])
    assert "the rollout clock expired" in str(r.systemic[0])
    assert len(r.runtime.named("sample")) == 3  # never retried, whatever the budget


@contextlib.contextmanager
def bounded(seconds: float):
    """Fail a block that spins without yielding, which no asyncio timeout can stop."""

    def expire(signum, frame):
        raise TimeoutError(f"still running after {seconds}s")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


async def test_drain_returns_whenever_it_is_called_after_finish():
    # Wherever drain() lands relative to finish() completing, it returns; one of
    # these points is after the finish task is done but before its done callback.
    for yields in range(12):
        r = rig()
        await r.driver.grade(await r.driver.sample(TASK, "g", "g-r1"))
        for _ in range(yields):
            await asyncio.sleep(0)
        with bounded(2):
            await r.driver.drain()
        assert len(r.runtime.named("finish")) == 1


async def test_each_grading_try_has_a_deadline_and_a_try_that_times_out_fails():
    async def hangs(attempt, rollout):
        await asyncio.Event().wait()

    r = rig(Runtime(on_grade=hangs), grading_timeout_seconds=0.05)
    assert await r.rollout() is Discard.GRADING_FAILED
    assert r.runtime.named("finish")[0][3] is EpisodeDisposition.FAILED


async def test_a_grade_that_returns_after_its_deadline_fails_as_one_that_ran_out():
    """One that blocks the loop past its deadline returns before the deadline's timer can
    run: its result is late, and refused, never trained."""

    async def blocks(attempt, rollout):
        time.sleep(0.1)  # noqa: ASYNC251 - the point: no await, so no timer runs
        return GradeResult(reward=1.0)

    r = rig(Runtime(on_grade=blocks), grading_timeout_seconds=0.05)
    assert await r.rollout() is Discard.GRADING_FAILED
    assert list(r.driver.gauges()) == ["discard/rollouts/grading_failed/hook_deadline"]


async def test_a_timely_grade_is_kept_though_the_loop_is_blocked_before_it_is_read():
    """Its lateness is when grade() returned, not when the driver got round to reading it."""

    async def timely(attempt, rollout):
        # Runs after grade() returns, before the driver resumes: a busy loop, not a late grade.
        asyncio.get_running_loop().call_soon(time.sleep, 0.1)
        return GradeResult(reward=1.0)

    r = rig(Runtime(on_grade=timely), grading_timeout_seconds=0.05)
    assert (await r.rollout()).reward == 1.0


async def test_excluded_rewards_are_discarded_without_counting_toward_the_stop():
    async def excludes(attempt, rollout):
        raise ExcludedReward("left out on purpose")

    r = rig(Runtime(on_grade=excludes), max_consecutive_failures=2)
    assert [await r.rollout() for _ in range(3)] == [Discard.GRADING_FAILED] * 3
    assert not r.systemic


async def test_cleanup_hooks_obey_their_deadline_and_count_as_cleanup_failures():
    async def hangs(*args):
        await asyncio.Event().wait()

    r = rig(Runtime(on_finish=hangs), cleanup_timeout_seconds=0.05, max_consecutive_failures=2)
    first = await r.rollout("g-r1")
    assert first.reward == 1.0  # a committed rollout keeps its eligibility
    assert not r.systemic
    await r.rollout("g-r3")
    assert r.systemic and "cleanup" in str(r.systemic[0])


@pytest.mark.parametrize("between", ["finish", "abort"])
async def test_a_successful_cleanup_hook_resets_the_cleanup_count(between):
    """Two cleanup failures, with a successful finish() or abort() between them."""
    plan = iter(["fails", between, "fails"])
    rollout = []

    async def sample(attempt):
        rollout[:] = [next(plan)]
        if rollout[0] == "abort":
            raise KeyError("episode setup failed")
        await generate(attempt)
        return EpisodeResult()

    async def finish(*args):
        if rollout[0] == "fails":
            raise OSError("sandbox stop failed")

    r = rig(Runtime(on_sample=sample, on_finish=finish), max_consecutive_failures=2)
    for number in range(3):
        await r.rollout(f"g-r{number}")
    assert not r.systemic


# ----------------------------------------------------------------- quiescence


async def let_go(r, release):
    """Let a kept hook end, then wait for the cleanup its attempt is owed."""
    release.set()
    await settle_loop()
    await r.driver.drain()


def stubborn_until(release):
    """A hook that ignores cancellation until the test lets it go."""

    async def hook(*args):
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue

    return hook


async def test_a_finish_that_ignores_cancellation_is_a_cleanup_failure_and_the_run_goes_on():
    release = asyncio.Event()
    r = rig(Runtime(on_finish=stubborn_until(release)), cleanup_timeout_seconds=0.05)
    try:
        assert (await asyncio.wait_for(r.rollout(), 2)).reward == 1.0  # its rollout stands
        await r.driver.drain()
        assert not r.systemic
        assert r.driver.live_attempts == 1
        assert r.driver.unquiesced == ("finish:g-r1",)
        release.set()
        await settle_loop()
        assert r.driver.live_attempts == 0
        assert r.driver.unquiesced == ()
    finally:
        await let_go(r, release)


@pytest.mark.parametrize("hook", ["finish", "abort"])
async def test_cleanup_hooks_that_all_ignore_cancellation_stop_the_run_by_the_cleanup_count(hook):
    release = asyncio.Event()
    hooks = {f"on_{hook}": stubborn_until(release)}
    if hook == "abort":
        hooks["on_sample"] = _fails
    r = rig(
        Runtime(**hooks),
        sampling_slots=100,  # 5 tolerated: the cleanup count is reached first
        cleanup_timeout_seconds=0.05,
        max_consecutive_failures=2,
    )
    try:
        for number in range(2):
            await asyncio.wait_for(r.rollout(f"g-r{number}"), 2)
        await r.driver.drain()
        [stop] = r.systemic
        assert "2 consecutive cleanup" in str(stop)
        assert "ignored cancellation" in str(stop)
    finally:
        await let_go(r, release)


async def test_cleanup_hooks_kept_running_count_toward_the_bound_with_sample_and_grade():
    release = asyncio.Event()
    r = rig(
        Runtime(on_finish=stubborn_until(release)),
        sampling_slots=20,  # 1 tolerated
        cleanup_timeout_seconds=0.05,
    )
    try:
        await asyncio.wait_for(r.rollout("g-r1"), 2)
        await r.driver.drain()
        assert not r.systemic
        await asyncio.wait_for(r.rollout("g-r2"), 2)
        await r.driver.drain()
        [stop] = r.systemic
        assert "2 hooks ignored cancellation past their grace, more than the 1 tolerated" in str(
            stop
        )
    finally:
        await let_go(r, release)


async def test_a_grade_that_ignores_cancellation_gets_no_second_try():
    tries = 0
    release = asyncio.Event()
    hold_out = stubborn_until(release)

    async def stubborn(attempt, rollout):
        nonlocal tries
        tries += 1
        await hold_out()

    r = rig(
        Runtime(on_grade=stubborn),
        grading_retries=3,
        grading_timeout_seconds=0.05,
    )
    try:
        assert await asyncio.wait_for(r.rollout(), 2) is Discard.UNQUIESCED
        assert tries == 1
        assert r.driver.note("g-r1").cause == "unquiesced/grade"
        assert r.driver.unquiesced == ("grade:g-r1",)
        assert not r.systemic  # one kept hook does not stop the run
    finally:
        await let_go(r, release)


def stubborn_sample(release):
    async def sample(attempt):
        await generate(attempt)
        await stubborn_until(release)()

    return sample


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
async def test_a_kept_sample_starts_no_process_and_is_aborted_once_it_ends(tmp_path):
    # While it runs it is refused new processes; once it ends its abort() runs, and
    # may run processes as any abort() may.
    release = asyncio.Event()
    refused: list[ProcessesClosed] = []
    aborts: list[tuple[AttemptAbortReason, int]] = []
    command = ProcessCommand(argv=(sys.executable, "-c", "pass"), cwd=tmp_path)

    async def sample(attempt):
        await generate(attempt)
        await stubborn_until(release)()
        try:
            await attempt.services.processes.run(command)
        except ProcessesClosed as error:
            refused.append(error)

    async def abort(attempt, reason):
        aborts.append((reason, (await attempt.services.processes.run(command)).exit_code))

    r = rig(Runtime(on_sample=sample, on_abort=abort), rollout_timeout_seconds=0.05)
    try:
        assert await asyncio.wait_for(r.rollout("g-r1"), 2) is Discard.UNQUIESCED
        assert aborts == []  # no cleanup while it runs
        release.set()
        await settle_loop()
        await r.driver.drain()
        assert len(refused) == 1
        assert aborts == [(AttemptAbortReason.ROLLOUT_TIMEOUT, 0)]
        assert r.driver.live_attempts == 0
        assert not r.systemic
    finally:
        await let_go(r, release)


@pytest.mark.parametrize("call", ["hold(verification)", "hold(provisioning)", "admit()"])
async def test_a_kept_hook_that_breaks_the_capacity_contract_still_stops_the_run(call):
    # A kept hook is refused capacity, but a call no hook may make is a contract
    # violation first, kept or not. Nothing is declared here.
    release = asyncio.Event()
    raised: list[BaseException] = []

    async def sample(attempt):
        await generate(attempt)
        await stubborn_until(release)()
        capacity = attempt.services.capacity
        try:
            if call == "admit()":
                await capacity.admit()
            else:
                capacity.hold(Phase(call[len("hold(") : -1]))
        except Exception as error:  # noqa: BLE001 - what it raised is the observation
            raised.append(error)

    r = rig(Runtime(on_sample=sample), sampling_slots=20, rollout_timeout_seconds=0.05)
    try:
        assert await asyncio.wait_for(r.rollout("g-r1"), 2) is Discard.UNQUIESCED
        assert not r.systemic
        await let_go(r, release)
        assert [type(error) for error in raised] == [RuntimeContractError]
        [stop] = r.systemic
        assert str(raised[0]) in str(stop)
    finally:
        await let_go(r, release)


async def test_a_kept_sample_that_hit_a_run_fatal_model_access_error_still_stops_the_run():
    # Kept or not, a sample() whose lease went fatal (here an undeclared producer,
    # which it swallowed) stops the run; its abort() still comes once it ends.
    release = asyncio.Event()

    async def sample(attempt):
        await generate(attempt)
        with contextlib.suppress(Exception):
            await generate(attempt, producer="not-declared")
        await stubborn_until(release)()

    r = rig(Runtime(on_sample=sample), rollout_timeout_seconds=0.05)
    try:
        assert await asyncio.wait_for(r.rollout("g-r1"), 2) is Discard.UNQUIESCED
        [stop] = r.systemic
        assert "not-declared" in str(stop)
        await let_go(r, release)
        assert len(r.runtime.named("abort")) == 1
    finally:
        await let_go(r, release)


async def test_a_kept_sample_whose_late_abort_fails_leaves_no_note_behind():
    # Its record is written with its discard; the abort() that fails once it ends is
    # reported, and notes nothing no record would read.
    release = asyncio.Event()

    async def abort(attempt, reason):
        raise RuntimeError("teardown failed")

    r = rig(
        Runtime(on_sample=stubborn_sample(release), on_abort=abort), rollout_timeout_seconds=0.05
    )
    try:
        assert await asyncio.wait_for(r.rollout("g-r1"), 2) is Discard.UNQUIESCED
        assert r.driver.note("g-r1").cause == "unquiesced/sample"  # its record takes it
        await let_go(r, release)
        assert r.driver.note("g-r1") == RolloutNote()
    finally:
        await let_go(r, release)


@pytest.mark.parametrize("cancelled", [False, True], ids=["deadline", "shutdown"])
async def test_a_kept_grade_is_finished_and_settled_once_it_ends(cancelled):
    release, started = asyncio.Event(), asyncio.Event()
    hold_out = stubborn_until(release)

    async def grade(attempt, rollout):
        started.set()
        await hold_out()

    r = rig(Runtime(on_grade=grade), grading_timeout_seconds=30 if cancelled else 0.05)
    try:
        admitted = await r.driver.sample(TASK, "g", "g-r1")
        grading = asyncio.create_task(r.driver.grade(admitted))
        if cancelled:
            await started.wait()
            grading.cancel()  # as at shutdown
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(grading, 2)
        else:
            assert await asyncio.wait_for(grading, 2) is Discard.UNQUIESCED
        assert r.runtime.named("finish") == []
        assert r.driver.live_attempts == 1
        release.set()
        await settle_loop()
        await r.driver.drain()
        assert [call[3] for call in r.runtime.named("finish")] == [EpisodeDisposition.FAILED]
        assert r.driver.live_attempts == 0
    finally:
        await let_go(r, release)


async def test_a_sample_that_ignores_cancellation_is_discarded_and_the_run_goes_on():
    release = asyncio.Event()
    stubborn = stubborn_sample(release)

    async def first_only(attempt):
        if attempt.spec.identity.rollout_id == "g-r1":
            await stubborn(attempt)
        await generate(attempt)
        return EpisodeResult()

    r = rig(Runtime(on_sample=first_only), rollout_timeout_seconds=0.05, max_rollout_retries=2)
    try:
        assert await asyncio.wait_for(r.rollout("g-r1"), 2) is Discard.UNQUIESCED
        assert not r.systemic
        assert r.driver.note("g-r1").cause == "unquiesced/sample"
        assert r.driver.gauges()["discard/rollouts/unquiesced/sample"] == 1
        assert len(r.runtime.named("sample")) == 1  # never retried beside itself
        assert r.runtime.named("abort") == []
        assert r.capacity.snapshot()["slots/sampling"] == 1  # held while it runs
        assert r.driver.live_attempts == 1
        assert (await asyncio.wait_for(r.rollout("g-r2"), 2)).reward == 0.0  # the run goes on
    finally:
        await let_go(r, release)


def kept_for_g_r1(release, phase):
    """A runtime whose g-r1 ignores cancellation in the hook that holds ``phase``."""
    hold_out = stubborn_until(release)

    async def sample(attempt):
        if phase == "sampling" and attempt.spec.identity.rollout_id == "g-r1":
            await generate(attempt)
            await hold_out()
        return await default_sample(attempt)

    async def grade(attempt, rollout):
        if phase == "verification" and rollout.identity.rollout_id == "g-r1":
            await hold_out()
        return await default_grade(attempt, rollout)

    return Runtime(on_sample=sample, on_grade=grade)


@pytest.mark.parametrize("phase", ["sampling", "verification"])
async def test_a_kept_hook_holding_a_phases_last_slot_stops_the_run(phase):
    # One kept hook is within 5% of 20 rollouts, but with one slot it would leave
    # every later rollout waiting for good.
    release = asyncio.Event()
    r = rig(
        kept_for_g_r1(release, phase),
        sampling_slots=1 if phase == "sampling" else 20,
        verification_slots=1,
        rollout_timeout_seconds=0.05,
        grading_timeout_seconds=0.05,
    )
    try:
        assert await asyncio.wait_for(r.rollout("g-r1"), 2) is Discard.UNQUIESCED
        [stop] = r.systemic
        assert isinstance(stop, NonQuiescentHook)
        assert f"hold all 1 {phase} slots until they end" in str(stop)
        with pytest.raises(NonQuiescentHook):
            await asyncio.wait_for(r.rollout("g-r2"), 2)  # refused, not left waiting
    finally:
        await let_go(r, release)


async def test_a_phase_filled_beside_a_kept_hook_by_a_healthy_holder_goes_on():
    # Only what kept hooks hold counts: the other slot is busy, not lost.
    release = asyncio.Event()
    r = rig(
        kept_for_g_r1(release, "verification"),
        sampling_slots=20,
        verification_slots=2,
        grading_timeout_seconds=0.05,
    )
    try:
        async with r.capacity.group_verification():
            assert await asyncio.wait_for(r.rollout("g-r1"), 2) is Discard.UNQUIESCED
            assert r.capacity.snapshot()["slots/verification"] == 2
            assert not r.systemic
        assert (await asyncio.wait_for(r.rollout("g-r2"), 2)).reward == 0.0
    finally:
        await let_go(r, release)


@pytest.mark.parametrize(("prestart", "stops"), [(0, True), (1, False)])
async def test_a_kept_setup_holding_the_last_admission_place_stops_the_run(prestart, stops):
    # Admitted, then stuck outside any hold: no slot is held, but its admission
    # place is, and the next rollout would wait for admission for good.
    release = asyncio.Event()

    async def sample(attempt):
        await attempt.services.capacity.admit()
        if attempt.spec.identity.rollout_id == "g-r1":
            await stubborn_until(release)()
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            await generate(attempt)
        return EpisodeResult()

    r = rig(
        Runtime(on_sample=sample),
        phases=SAMPLING,
        sampling_slots=1,
        prestart=prestart,
        setup_timeout_seconds=0.05,
    )
    r.admission.value = 1  # the pipeline's target: max_concurrent_rollouts
    try:
        assert await asyncio.wait_for(r.rollout("g-r1"), 2) is Discard.UNQUIESCED
        assert r.capacity.snapshot()["admission/active_rollouts"] == 1
        if stops:
            [stop] = r.systemic
            assert f"hold all {1 + prestart} admission places until they end" in str(stop)
            with pytest.raises(NonQuiescentHook):
                await asyncio.wait_for(r.rollout("g-r2"), 2)  # refused, not left waiting
        else:
            assert not r.systemic  # a pre-start place is still free
            assert await asyncio.wait_for(r.rollout("g-r2"), 2)
    finally:
        await let_go(r, release)


async def test_an_admission_granted_after_its_hook_was_kept_is_refused_and_given_back():
    # A setup that outlives its clock asks for the only admission place while a
    # healthy rollout has it; kept with no place yet, it is granted one later.
    release, free, holding, settled = (asyncio.Event() for _ in range(4))

    async def sample(attempt):
        if attempt.spec.identity.rollout_id == "g-r1":
            async with attempt.services.capacity.hold(Phase.SAMPLING):
                holding.set()
                await free.wait()
                await generate(attempt)
            return EpisodeResult()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            try:
                await attempt.services.capacity.admit()
            finally:
                settled.set()
            await stubborn_until(release)()

    r = rig(
        Runtime(on_sample=sample),
        phases=SAMPLING,
        sampling_slots=1,
        setup_timeout_seconds=0.05,
    )
    r.admission.value = 1
    try:
        healthy = asyncio.create_task(r.rollout("g-r1"))
        await asyncio.wait_for(holding.wait(), 2)
        assert await asyncio.wait_for(r.rollout("g-r2"), 2) is Discard.UNQUIESCED
        assert not r.systemic  # kept while it held no place
        free.set()
        assert await asyncio.wait_for(healthy, 2)
        await asyncio.wait_for(settled.wait(), 2)
        assert r.capacity.snapshot()["admission/active_rollouts"] == 0
        assert not r.systemic  # refused in its own task
    finally:
        await let_go(r, release)


async def test_a_kept_hook_is_not_collected_while_it_runs(monkeypatch):
    # The loop holds tasks only weakly; a hook waiting on a future nothing else holds
    # would be collected, giving back a slot it still runs inside.
    # A captured log record of the overrun would hold it alive here, as nothing does in a run.
    monkeypatch.setattr(logging.getLogger("loops_cookbook.runtime.driver"), "disabled", True)
    stop = False

    async def sample(attempt):
        await generate(attempt)
        while True:
            try:
                await asyncio.get_running_loop().create_future()
            except asyncio.CancelledError:
                if stop:
                    raise

    r = rig(Runtime(on_sample=sample), sampling_slots=20, rollout_timeout_seconds=0.05)
    assert await asyncio.wait_for(r.rollout("g-r1"), 2) is Discard.UNQUIESCED
    gc.collect()
    await asyncio.sleep(0)
    assert r.capacity.snapshot()["slots/sampling"] == 1
    assert r.driver.unquiesced == ("sample:g-r1",)
    stop = True
    for task in asyncio.all_tasks():
        if task.get_name() == "sample:g-r1":
            task.cancel()
    await settle_loop()
    await r.driver.drain()


@pytest.mark.parametrize("kept", [True, False], ids=["kept", "cancelled"])
async def test_a_slot_granted_after_its_hook_ended_is_given_back(kept):
    # A runtime task the cancellation did not reach waits for the only verification
    # slot while a healthy holder has it; by the time it is free the hook is over.
    release, waiting = asyncio.Event(), asyncio.Event()
    late: list[asyncio.Task[None]] = []

    async def sample(attempt):
        async def background():
            waiting.set()
            async with attempt.services.capacity.hold(Phase.VERIFICATION):
                await asyncio.Event().wait()

        late.append(asyncio.create_task(background()))
        await generate(attempt)
        if kept:
            await stubborn_until(release)()
        else:
            await asyncio.Event().wait()

    r = rig(
        Runtime(on_sample=sample),
        phases=frozenset({Phase.VERIFICATION}),
        verification_slots=1,
        rollout_timeout_seconds=0.05,
    )
    try:
        async with r.capacity.group_verification():
            outcome = await asyncio.wait_for(r.rollout("g-r1"), 2)
            assert await waiting.wait()
        assert outcome is (Discard.UNQUIESCED if kept else Discard.AGENT_TIMEOUT)
        # A kept hook's is refused in its own task; any other ended hook's breaks the contract.
        with pytest.raises(CapacityRevoked if kept else RuntimeContractError):
            await asyncio.wait_for(late[0], 2)
        assert r.capacity.snapshot()["slots/verification"] == 0
        assert bool(r.systemic) is not kept
    finally:
        await let_go(r, release)
        for task in late:
            task.cancel()


async def test_a_kept_hook_that_asks_for_more_capacity_is_refused_and_the_run_goes_on():
    # Like a Harbor trial whose agent ends after the grace: it gives back SAMPLING,
    # then asks for VERIFICATION.
    advance = asyncio.Event()
    refused: list[BaseException] = []

    async def sample(attempt):
        hold = attempt.services.capacity.hold(Phase.SAMPLING)
        await hold.__aenter__()
        await generate(attempt)
        await stubborn_until(advance)()
        await hold.__aexit__(None, None, None)
        try:
            async with attempt.services.capacity.hold(Phase.VERIFICATION):
                pass
        except CapacityRevoked as error:
            refused.append(error)
            raise

    r = rig(
        Runtime(on_sample=sample),
        phases=BOTH,
        sampling_slots=20,
        verification_slots=4,
        rollout_timeout_seconds=0.05,
    )
    assert await asyncio.wait_for(r.rollout("g-r1"), 2) is Discard.UNQUIESCED
    advance.set()
    await settle_loop()
    assert len(refused) == 1
    assert not r.systemic
    assert r.capacity.snapshot()["slots/verification"] == 0


async def test_a_kept_hook_gives_back_its_capacity_and_its_place_in_the_bound_when_it_ends(
    caplog,
):
    # One kept hook is tolerated with 20 slots; one that has ended no longer counts.
    release = asyncio.Event()

    async def sample(attempt):
        await attempt.services.capacity.hold(Phase.SAMPLING).__aenter__()  # never exited
        await generate(attempt)
        await stubborn_until(release)()
        raise RuntimeError("woke up")

    r = rig(
        Runtime(on_sample=sample), phases=SAMPLING, sampling_slots=20, rollout_timeout_seconds=0.05
    )
    try:
        assert await asyncio.wait_for(r.rollout("g-r1"), 2) is Discard.UNQUIESCED
        assert r.capacity.snapshot()["slots/sampling"] == 1
        release.set()
        await settle_loop()
        await r.driver.drain()  # its abort() runs, then it settles
        snapshot = r.capacity.snapshot()
        assert snapshot["slots/sampling"] == 0
        assert snapshot["admission/active_rollouts"] == 0
        assert r.driver.unquiesced == ()
        assert r.driver.live_attempts == 0
        assert "sample() for g-r1 ended after it was kept" in caplog.text
        assert "RuntimeError: woke up" in caplog.text
        assert r.driver.note("g-r1").cause == "unquiesced/sample"  # still discarded
        release.clear()
        assert await asyncio.wait_for(r.rollout("g-r2"), 2) is Discard.UNQUIESCED
        assert not r.systemic  # one kept now, as before
    finally:
        await let_go(r, release)


@pytest.mark.parametrize(("slots", "tolerated"), [(8, 1), (20, 1), (21, 2), (40, 2), (60, 3)])
async def test_more_kept_hooks_than_5_percent_of_the_rollout_slots_stop_the_run(slots, tolerated):
    release = asyncio.Event()
    r = rig(
        Runtime(on_sample=stubborn_sample(release)),
        sampling_slots=slots,
        rollout_timeout_seconds=0.05,
    )
    try:
        for number in range(tolerated):
            assert await asyncio.wait_for(r.rollout(f"g-r{number}"), 2) is Discard.UNQUIESCED
        assert not r.systemic
        # The rollout that passes the bound is still discarded as its own outcome.
        assert await asyncio.wait_for(r.rollout("g-rx"), 2) is Discard.UNQUIESCED
        [stop] = r.systemic
        assert isinstance(stop, NonQuiescentHook)
        assert f"{tolerated + 1} hooks ignored cancellation" in str(stop)
        assert f"more than the {tolerated} tolerated" in str(stop)
        assert "holds its capacity until it ends" in str(stop)
    finally:
        await let_go(r, release)


# -------------------------------------------------------------------- retries


async def test_an_eligible_failure_is_retried_as_a_fresh_attempt_after_cleanup():
    order = []

    async def once_then_fine(attempt):
        order.append(("sample", attempt.spec.identity.attempt_id))
        await generate(attempt)
        if len(order) == 1:
            raise OSError("sandbox died")
        return EpisodeResult()

    async def abort(attempt, reason):
        order.append(("abort", attempt.spec.identity.attempt_id))

    r = rig(Runtime(on_sample=once_then_fine, on_abort=abort), max_rollout_retries=2)
    graded = await r.rollout()
    first, second = order[0][1], order[2][1]
    assert order == [("sample", first), ("abort", first), ("sample", second)]
    assert first != second
    assert graded.identity.attempt_id == second and graded.identity.rollout_id == "g-r1"
    assert len(graded.samples) == 1  # nothing from the failed attempt


async def _permanent_sample(attempt):
    raise PermanentRolloutFailure("no point")


async def _flaky_sample(attempt):
    raise OSError("once")


async def _permanent_grade(attempt, rollout):
    raise PermanentGradingFailure("missing result files")


async def _flaky_grade(attempt, rollout):
    raise OSError("verifier flake")


@pytest.mark.parametrize(
    ("hooks", "settings", "outcome", "hook"),
    [
        (
            {"on_sample": _permanent_sample},
            {"max_rollout_retries": 5},
            Discard.ROLLOUT_FAILED,
            "sample",
        ),
        ({"on_sample": _flaky_sample}, {}, Discard.ROLLOUT_FAILED, "sample"),
        ({"on_grade": _permanent_grade}, {"grading_retries": 3}, Discard.GRADING_FAILED, "grade"),
        ({"on_grade": _flaky_grade}, {}, Discard.GRADING_FAILED, "grade"),
    ],
    ids=["sample-permanent", "sample-by-default", "grade-permanent", "grade-by-default"],
)
async def test_a_failure_is_tried_once_when_permanent_or_when_no_retry_is_configured(
    hooks, settings, outcome, hook
):
    r = rig(Runtime(**hooks), **settings)
    assert await r.rollout() is outcome
    assert len(r.runtime.named(hook)) == 1


async def test_a_deterministic_failure_retried_by_configuration_is_still_systemic():
    async def failing(attempt):
        raise OSError("always")

    r = rig(Runtime(on_sample=failing), max_rollout_retries=10, max_consecutive_failures=3)
    assert await r.rollout() is Discard.ROLLOUT_FAILED
    assert "3 consecutive sampling" in str(r.systemic[0])
    assert len(r.runtime.named("sample")) == 3


async def test_grading_retries_only_after_the_try_quiesces_with_capacity_released():
    tries = []
    held = []

    async def flaky(attempt, rollout):
        tries.append(attempt)
        held.append(r.capacity.snapshot()["slots/verification"])
        if len(tries) < 3:
            raise OSError("verifier flake")
        return GradeResult(reward=0.5)

    r = rig(Runtime(on_grade=flaky), grading_retries=2)
    graded = await r.rollout()
    assert graded.reward == 0.5
    assert held == [1.0, 1.0, 1.0]  # one runner-owned permit per try, never two
    assert r.sleeps == [0.5, 1.0]
    assert r.capacity.snapshot()["slots/verification"] == 0


async def test_the_grading_deadline_does_not_count_the_wait_for_a_verification_slot():
    grading, let_go = asyncio.Event(), asyncio.Event()

    async def grade(attempt, rollout):
        grading.set()
        await let_go.wait()
        return GradeResult(reward=1.0)

    deadline = 0.2  # only a few loop turns of the try's own run inside it
    r = rig(Runtime(on_grade=grade), verification_slots=1, grading_timeout_seconds=deadline)
    admitted = await r.driver.sample(TASK, "g", "g-r1")
    other = AttemptPhases(AttemptIdentity("run", "g", "g-r2", "a2"), frozenset())
    r.capacity.track(other)
    other.start_hook(Hook.GRADE)
    async with r.capacity.runner_hold(other, Phase.VERIFICATION):  # the one slot, elsewhere
        trying = asyncio.create_task(r.driver.grade(admitted))
        await asyncio.sleep(1.25 * deadline)  # the try waits for the slot past its deadline
    async with asyncio.timeout(5):
        await grading.wait()
        await settle_loop()  # had the wait counted, its deadline would fire here
        let_go.set()
        assert (await trying).reward == 1.0
    await r.driver.drain()


async def test_grading_retries_recheck_staleness():
    async def flaky(attempt, rollout):
        r.version[0] = 99
        raise OSError("verifier flake")

    r = rig(Runtime(on_grade=flaky), grading_retries=2)
    assert await r.rollout() is Discard.STALE_BEFORE_GRADING
    assert r.runtime.named("finish")[0][3] is EpisodeDisposition.DISCARDED
    assert len(r.runtime.named("grade")) == 1


async def test_a_rollout_stale_at_grading_admission_is_discarded_before_grading():
    r = rig()
    admitted = await r.driver.sample(TASK, "g", "g-r1")
    r.version[0] = 99
    assert await r.driver.grade(admitted) is Discard.STALE_BEFORE_GRADING
    await r.driver.drain()
    assert not r.runtime.named("grade")


def numpy_value(kind, value):
    numpy = pytest.importorskip("numpy")
    return getattr(numpy, kind)(value)


@pytest.mark.parametrize(
    "reward",
    [float("nan"), float("inf"), "1", True, None, ("float64", "nan"), ("bool_", True)],
)
async def test_a_non_finite_or_non_numeric_reward_is_a_grading_failure(reward):
    if isinstance(reward, tuple):
        reward = numpy_value(*reward)

    async def bad(attempt, rollout):
        return GradeResult(reward=reward)  # pyright: ignore[reportArgumentType]

    r = rig(Runtime(on_grade=bad))
    assert await r.rollout() is Discard.GRADING_FAILED


@pytest.mark.parametrize(
    "reward", [1, 0.5, Fraction(1, 2), ("float64", 0.5), ("float32", 0.5), ("int64", 1)]
)
async def test_any_finite_real_reward_is_accepted_as_a_float(reward):
    if isinstance(reward, tuple):
        reward = numpy_value(*reward)

    async def graded(attempt, rollout):
        return GradeResult(reward=reward)

    rollout = await rig(Runtime(on_grade=graded)).rollout()
    assert type(rollout.reward) is float and rollout.reward == float(reward)


@pytest.mark.parametrize(("reward", "graded"), [(None, True), (float("nan"), False)])
async def test_with_a_group_grader_only_no_reward_at_all_is_left_for_it_to_give(reward, graded):
    async def pending(attempt, rollout):
        return GradeResult(reward=reward, metadata={"verdict": "pending"})

    result = await rig(Runtime(on_grade=pending), pending_group_grade=True).rollout()
    if graded:
        assert result.reward is None and result.metadata["verdict"] == "pending"
    else:
        assert result is Discard.GRADING_FAILED


def test_group_grading_failures_are_budgeted_and_a_scored_group_resets_the_count():
    r = rig(max_consecutive_failures=2)
    r.driver.group_retrying("g1", RuntimeError("blip"))  # not counted
    r.driver.group_failed("g1", RuntimeError("judge down"), ["g1-r0"])
    r.driver.group_scored()  # the count starts again
    r.driver.group_failed("g3", RuntimeError("judge down"), ["g3-r0"])
    assert not r.systemic
    r.driver.group_failed("g4", RuntimeError("judge down"), ["g4-r0"])  # stops the run
    assert [str(error) for error in r.systemic] == [
        "2 consecutive group-grading failures: judge down"
    ]
    assert r.driver.gauges()["retried/groups/grading_failed/runtime"] == 1.0


def test_a_group_grader_still_running_after_cancellation_leaves_the_run_unsettled():
    r = rig()
    r.driver.group_unquiesced(
        "g1", NonQuiescentHook("still running", Waited(settled=False, interrupted=False))
    )
    assert r.driver.unquiesced == ("grade_group:g1",)
    assert [str(error) for error in r.systemic] == ["still running"]


async def test_grade_metadata_that_repeats_an_episode_key_is_a_grading_failure(caplog):
    async def clashing(attempt, rollout):
        return GradeResult(reward=1.0, metadata={"answer": "grade", "verdict": "grade"})

    r = rig(Runtime(on_grade=clashing))
    assert await r.rollout() is Discard.GRADING_FAILED
    assert "['answer']" in caplog.text
    [(_, _, final, disposition)] = r.runtime.named("finish")
    assert disposition is EpisodeDisposition.FAILED
    assert dict(final.metadata) == {"answer": "episode"}  # neither value is chosen silently


# ------------------------------------------------------------ phase capacity


async def test_internal_verification_holds_are_released_between_grading_tries():
    tries = 0

    async def holds(attempt, rollout):
        nonlocal tries
        tries += 1
        async with attempt.services.capacity.hold(Phase.VERIFICATION):
            if tries == 1:
                raise OSError("flake")
        return GradeResult(reward=1.0)

    r = rig(Runtime(on_grade=holds), phases=BOTH, grading_retries=1)

    async def sample(attempt):
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            await generate(attempt)
        return EpisodeResult()

    r.runtime.on_sample = sample
    assert (await r.rollout()).reward == 1.0
    assert r.capacity.snapshot()["slots/verification"] == 0


@pytest.mark.parametrize("phase", [Phase.PROVISIONING, Phase.VERIFICATION])
async def test_holding_an_undeclared_or_provisioning_phase_is_a_contract_error(phase):
    async def sample(attempt):
        async with attempt.services.capacity.hold(phase):
            pass

    r = rig(Runtime(on_sample=sample), phases=SAMPLING)
    assert await r.rollout() is Discard.ROLLOUT_FAILED  # its own outcome, and it stops the run
    assert r.systemic


async def test_a_hold_taken_while_no_hook_runs_is_a_contract_error():
    leaked = []

    async def sample(attempt):
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            await generate(attempt)
        leaked.append(attempt)
        return EpisodeResult()

    r = rig(Runtime(on_sample=sample), phases=SAMPLING)
    admitted = await r.driver.sample(TASK, "g", "g-r1")
    with pytest.raises(RuntimeContractError):
        async with leaked[0].services.capacity.hold(Phase.SAMPLING):
            pass
    assert r.systemic
    del admitted


async def test_a_hold_still_open_when_its_hook_returns_is_a_contract_error():
    async def sample(attempt):
        hold = attempt.services.capacity.hold(Phase.SAMPLING)
        await hold.__aenter__()
        await generate(attempt)
        return EpisodeResult()

    r = rig(Runtime(on_sample=sample), phases=SAMPLING)
    assert await r.rollout() is Discard.ROLLOUT_FAILED  # its own outcome, and it stops the run
    assert r.systemic and "holds still open" in str(r.systemic[0])
    assert r.capacity.snapshot()["slots/sampling"] == 0


async def test_an_internal_sampling_runtime_must_enter_a_sampling_hold():
    r = rig(phases=SAMPLING)
    assert await r.rollout() is Discard.ROLLOUT_FAILED  # its own outcome, and it stops the run
    assert r.systemic and "without entering a sampling hold" in str(r.systemic[0])


async def test_overlapping_sampling_holds_are_a_contract_error():
    async def sample(attempt):
        async with (
            attempt.services.capacity.hold(Phase.SAMPLING),
            attempt.services.capacity.hold(Phase.SAMPLING),
        ):
            pass

    r = rig(Runtime(on_sample=sample), phases=SAMPLING)
    assert await r.rollout() is Discard.ROLLOUT_FAILED  # its own outcome, and it stops the run
    assert r.systemic


async def test_cancellation_during_sampling_aborts_with_cancelled_and_propagates():
    entered = asyncio.Event()

    async def hangs(attempt):
        entered.set()
        await asyncio.Event().wait()

    r = rig(Runtime(on_sample=hangs))
    task = asyncio.create_task(r.rollout())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert r.runtime.named("abort")[0][2] is AttemptAbortReason.CANCELLED
    assert r.driver.live_attempts == 0


async def test_cancellation_during_grading_finishes_as_failed():
    entered = asyncio.Event()

    async def hangs(attempt, rollout):
        entered.set()
        await asyncio.Event().wait()

    r = rig(Runtime(on_grade=hangs))
    task = asyncio.create_task(r.rollout())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert r.runtime.named("finish")[0][3] is EpisodeDisposition.FAILED
    assert r.driver.live_attempts == 0


# A client library can leak a CancelledError that nothing here asked for: it is the
# hook's own failure, not a cancellation of the rollout.


async def _leaks(*args):
    raise asyncio.CancelledError


async def _generates_then_leaks(attempt):
    await generate(attempt)
    raise asyncio.CancelledError


async def test_a_sample_that_raises_its_own_cancellation_is_a_retried_rollout_failure():
    r = rig(Runtime(on_sample=_generates_then_leaks), max_rollout_retries=1)
    async with asyncio.timeout(5):
        assert await r.rollout() is Discard.ROLLOUT_FAILED
    assert [call[2] for call in r.runtime.named("abort")] == [AttemptAbortReason.SAMPLE_FAILED] * 2
    assert r.driver.gauges() == {
        "retried/attempts/rollout_failed/cancelled": 1.0,
        "discard/rollouts/rollout_failed/cancelled": 1.0,
    }
    assert r.driver.live_attempts == 0


async def test_samples_that_raise_their_own_cancellation_count_toward_the_stop():
    r = rig(Runtime(on_sample=_generates_then_leaks), max_consecutive_failures=2)
    async with asyncio.timeout(5):
        assert await r.rollout("g-r1") is Discard.ROLLOUT_FAILED
        assert await r.rollout("g-r2") is Discard.ROLLOUT_FAILED
    assert "2 consecutive sampling" in str(r.systemic[0])


@pytest.mark.parametrize("retries", [0, 2])
async def test_the_rollout_that_trips_the_sampling_stop_is_discarded_with_its_cause(retries):
    """Stopped, not cancelled: its failure is its outcome, whether or not it had retries left."""
    r = rig(
        Runtime(on_sample=_generates_then_leaks),
        max_rollout_retries=retries,
        max_consecutive_failures=2,
    )
    tripping = "g-r1" if retries else "g-r2"
    async with asyncio.timeout(5):
        if not retries:
            assert await r.rollout("g-r1") is Discard.ROLLOUT_FAILED
        assert await r.rollout(tripping) is Discard.ROLLOUT_FAILED
        with pytest.raises(SystemicFailure, match="2 consecutive sampling"):
            await r.rollout("g-r3")
    assert r.driver.note(tripping) == RolloutNote(1 if retries else 0, "rollout_failed/cancelled")
    assert r.driver.gauges()["discard/rollouts/rollout_failed/cancelled"] == (1 if retries else 2)


@pytest.mark.parametrize("retries", [0, 2])
async def test_a_rollout_that_fails_once_another_stopped_the_run_keeps_its_cause_unretried(
    retries,
):
    """Its attempt finished: its failure is its outcome, and no attempt starts after the stop."""
    release = asyncio.Event()

    async def fails(attempt):
        if attempt.spec.identity.rollout_id == "g-r1":
            await release.wait()
        await _generates_then_leaks(attempt)

    r = rig(Runtime(on_sample=fails), max_rollout_retries=retries, max_consecutive_failures=1)
    async with asyncio.timeout(5):
        waiting = asyncio.create_task(r.driver.sample(TASK, "g", "g-r1"))
        while not r.runtime.named("sample"):
            await asyncio.sleep(0)
        assert await r.driver.sample(TASK, "g", "g-r2") is Discard.ROLLOUT_FAILED
        assert r.systemic
        release.set()
        assert await waiting is Discard.ROLLOUT_FAILED
    assert r.driver.note("g-r1") == RolloutNote(0, "rollout_failed/cancelled")
    assert len(r.runtime.named("sample")) == 2
    assert r.driver.live_attempts == 0


async def test_the_rollout_that_trips_the_grading_stop_is_discarded_with_its_cause():
    r = rig(Runtime(on_grade=_leaks), max_consecutive_failures=2)
    async with asyncio.timeout(5):
        assert await r.rollout("g-r1") is Discard.GRADING_FAILED
        assert await r.rollout("g-r2") is Discard.GRADING_FAILED
        with pytest.raises(SystemicFailure, match="2 consecutive grading"):
            await r.rollout("g-r3")
    assert r.driver.note("g-r2") == RolloutNote(0, "grading_failed/cancelled")
    assert r.driver.gauges()["discard/rollouts/grading_failed/cancelled"] == 2


@pytest.mark.parametrize("retries", [0, 2])
async def test_a_grade_that_raises_its_own_cancellation_is_a_grading_failure(retries):
    r = rig(Runtime(on_grade=_leaks), grading_retries=retries)
    async with asyncio.timeout(5):
        assert await r.rollout() is Discard.GRADING_FAILED
    assert len(r.runtime.named("grade")) == 1 + retries
    assert r.runtime.named("finish")[0][3] is EpisodeDisposition.FAILED
    assert r.driver.gauges()["discard/rollouts/grading_failed/cancelled"] == 1.0
    assert r.driver.live_attempts == 0


@pytest.mark.parametrize("hook", ["finish", "abort"])
async def test_a_cleanup_hook_that_raises_its_own_cancellation_is_a_cleanup_failure(hook):
    fails = {"on_sample": _generates_then_leaks} if hook == "abort" else {}
    r = rig(Runtime(**{f"on_{hook}": _leaks}, **fails), max_consecutive_failures=2)
    async with asyncio.timeout(5):
        await r.rollout("g-r1")
        assert r.driver.live_attempts == 0 and not r.systemic
        await r.rollout("g-r2")  # the rollout whose cleanup stops the run keeps its outcome
    assert r.systemic and "2 consecutive cleanup" in str(r.systemic[0])


async def test_a_request_rejected_before_sampling_does_not_poison_the_attempt():
    from loops_cookbook.llm.client import ContextWindowExceeded

    async def oversized_then_fine(attempt):
        conversation = attempt.services.model_access.llm().conversation()
        with pytest.raises(ContextWindowExceeded):
            await conversation.generate(
                messages=[{"role": "user", "content": "x"}], max_trajectory_tokens=1
            )
        await generate(attempt)
        return EpisodeResult()

    r = rig(Runtime(on_sample=oversized_then_fine))
    graded = await r.rollout()
    assert graded.reward == 1.0 and len(graded.samples) == 1


# ------------------------------------------------------- systemic stop paths


async def test_exhausting_the_grading_budget_still_finishes_and_frees_the_episode():
    async def failing(attempt, rollout):
        raise OSError("verifier down")

    r = rig(Runtime(on_grade=failing), max_consecutive_failures=1)
    admitted = await r.driver.sample(TASK, "g", "g-r1")
    assert await r.driver.grade(admitted) is Discard.GRADING_FAILED
    assert "grading" in str(r.systemic[0])
    await r.driver.drain()
    assert r.runtime.named("finish")[0][3] is EpisodeDisposition.FAILED
    assert r.driver.live_attempts == 0


async def test_a_contract_error_in_grade_finishes_the_episode_without_another_try():
    async def illegal(attempt, rollout):
        async with attempt.services.capacity.hold(Phase.PROVISIONING):
            pass

    r = rig(Runtime(on_grade=illegal), grading_retries=3)
    admitted = await r.driver.sample(TASK, "g", "g-r1")
    assert await r.driver.grade(admitted) is Discard.GRADING_FAILED
    assert r.systemic
    assert r.driver.note("g-r1").cause == "grading_failed/runtime_contract"
    await r.driver.drain()
    assert len(r.runtime.named("grade")) == 1
    assert r.runtime.named("finish")[0][3] is EpisodeDisposition.FAILED
    assert r.driver.live_attempts == 0


async def test_a_contract_error_in_sample_still_lets_abort_unwind():
    async def illegal(attempt):
        async with attempt.services.capacity.hold(Phase.PROVISIONING):
            pass

    r = rig(Runtime(on_sample=illegal))
    assert await r.rollout() is Discard.ROLLOUT_FAILED  # its own outcome, and it stops the run
    assert r.systemic
    assert r.runtime.named("abort")
    assert r.driver.live_attempts == 0


@pytest.mark.parametrize(
    "raised",
    [
        ModelTransportFailure("down"),
        KeyError("runtime bug"),
        PermanentRolloutFailure("trial"),
        None,
    ],
    ids=["propagated", "replaced", "replaced-permanent", "returned"],
)
async def test_a_poisoned_leases_cause_decides_the_failure_and_is_never_retried(caplog, raised):
    """Whatever the runtime raises instead, or if it returns, the lost generation's cause wins."""

    async def broken(prompt, **kwargs):
        raise ConnectionError("down")

    async def fails(attempt):
        r.sampler.sample_async = broken  # type: ignore[method-assign]
        with contextlib.suppress(ModelTransportFailure):
            await generate(attempt)
        if raised is None:
            return EpisodeResult()
        raise raised

    r = rig(Runtime(on_sample=fails), max_rollout_retries=2)
    assert await r.rollout() is Discard.ROLLOUT_FAILED
    assert len(r.runtime.named("sample")) == 1
    assert [call[2] for call in r.runtime.named("abort")] == [AttemptAbortReason.SAMPLE_FAILED]
    assert "ModelTransportFailure" in caplog.text


async def test_a_terminated_sampler_is_an_ordinary_sampling_failure_the_budget_stops():
    from baseten.loops.sampling_client import SamplerTerminatedError

    async def terminated(prompt, **kwargs):
        raise SamplerTerminatedError("dep", "UNHEALTHY")

    async def propagates(attempt):
        r.sampler.sample_async = terminated  # type: ignore[method-assign]
        await generate(attempt)
        return EpisodeResult()

    r = rig(Runtime(on_sample=propagates), max_rollout_retries=3, max_consecutive_failures=2)
    assert await r.rollout() is Discard.ROLLOUT_FAILED
    assert not r.systemic and len(r.runtime.named("sample")) == 1
    assert await r.rollout() is Discard.ROLLOUT_FAILED
    assert "2 consecutive sampling" in str(r.systemic[0])


async def test_sampling_holds_that_both_wait_for_a_permit_still_count_as_overlap():
    r = rig(phases=SAMPLING, sampling_slots=1)
    blocker_ready, release = asyncio.Event(), asyncio.Event()

    async def blocker(attempt):
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            await generate(attempt)
            blocker_ready.set()
            await release.wait()
        return EpisodeResult()

    async def overlapping(attempt):
        first = asyncio.create_task(attempt.services.capacity.hold(Phase.SAMPLING).__aenter__())
        await asyncio.sleep(0)
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            pass
        await first
        return EpisodeResult()

    calls = iter([blocker, overlapping])
    r.runtime.on_sample = lambda attempt: next(calls)(attempt)
    held = asyncio.create_task(r.driver.sample(TASK, "g", "g-r1"))
    await blocker_ready.wait()
    try:
        assert await asyncio.wait_for(r.driver.sample(TASK, "g", "g-r2"), 2) is (
            Discard.ROLLOUT_FAILED
        )
        assert "overlap" in str(r.systemic[0])
    finally:
        await let_go(r, release)
        await asyncio.gather(held, return_exceptions=True)


# ------------------------------------------------------------ failure contract


def _logged(caplog, stage: str) -> list[BaseException]:
    return [
        record.exc_info[1]
        for record in caplog.records
        if record.exc_info and record.getMessage().startswith(f"{stage} failed")
    ]


async def _fails(*args):
    raise ValueError("runtime bug")


@pytest.mark.parametrize(
    ("hooks", "stage", "phase"),
    [
        ({"on_sample": _fails}, "sampling", "sample"),
        ({"on_grade": _fails}, "grading", "grade"),
        ({"on_finish": _fails}, "cleanup", "finish"),
        ({"on_sample": _fails, "on_abort": _fails}, "cleanup", "abort"),
    ],
)
async def test_an_unknown_hook_exception_keeps_its_phase_identity_and_cause(
    caplog, hooks, stage, phase
):
    r = rig(Runtime(**hooks))
    await r.rollout()
    (failure,) = _logged(caplog, stage)
    assert isinstance(failure, RuntimeExecutionFailure)
    assert failure.phase == phase
    assert failure.identity is not None and failure.identity.rollout_id == "g-r1"
    assert isinstance(failure.__cause__, ValueError)


async def test_a_cleanup_failure_is_attached_to_the_failure_that_caused_it(caplog):
    async def stop_fails(attempt, reason):
        raise OSError("sandbox stop failed")

    r = rig(Runtime(on_sample=_fails, on_abort=stop_fails))
    assert await r.rollout() is Discard.ROLLOUT_FAILED
    (primary,) = _logged(caplog, "sampling")  # the cleanup failure did not replace it
    assert isinstance(primary.__cause__, ValueError)
    (cleanup,) = cleanup_failures(primary)
    assert cleanup.phase == "abort" and isinstance(cleanup.__cause__, OSError)
    assert any("sandbox stop failed" in note for note in primary.__notes__)


async def test_abort_obeys_the_cleanup_deadline_and_counts_as_a_cleanup_failure():
    async def hangs(attempt, reason):
        await asyncio.Event().wait()

    r = rig(
        Runtime(on_sample=_fails, on_abort=hangs),
        cleanup_timeout_seconds=0.05,
        max_consecutive_failures=1,
    )
    # the rollout whose cleanup stops the run keeps its own outcome
    assert await asyncio.wait_for(r.rollout(), 2) is Discard.ROLLOUT_FAILED
    assert "abort() exceeded its deadline" in str(r.systemic[0])
    assert r.driver.live_attempts == 0


async def test_an_abort_that_ignores_cancellation_keeps_its_attempt_live_and_the_run_goes_on():
    release = asyncio.Event()
    r = rig(
        Runtime(on_sample=_fails, on_abort=stubborn_until(release)), cleanup_timeout_seconds=0.05
    )
    try:
        assert await asyncio.wait_for(r.rollout(), 2) is Discard.ROLLOUT_FAILED
        assert not r.systemic
        assert r.driver.live_attempts == 1
        assert r.driver.unquiesced == ("abort:g-r1",)
        release.set()
        await settle_loop()
        assert r.driver.live_attempts == 0
        assert r.driver.unquiesced == ()
    finally:
        await let_go(r, release)


async def test_a_rollout_retried_past_an_abort_that_ignores_cancellation_settles_once():
    release = asyncio.Event()
    hold_out = stubborn_until(release)
    aborts = 0

    async def abort(attempt, reason):
        nonlocal aborts
        aborts += 1
        if aborts == 1:
            await hold_out()

    r = rig(
        Runtime(on_sample=_fails, on_abort=abort),
        cleanup_timeout_seconds=0.05,
        max_rollout_retries=1,
    )
    try:
        assert await asyncio.wait_for(r.rollout(), 2) is Discard.ROLLOUT_FAILED
        assert len(r.runtime.named("sample")) == 2
        assert not r.systemic
        assert r.driver.live_attempts == 0  # settled by its last attempt
        release.set()
        await settle_loop()
        assert r.driver.live_attempts == 0  # not settled again by the kept one
        assert r.driver.unquiesced == ()
    finally:
        await let_go(r, release)


async def test_abort_receives_the_original_context_with_its_lease_already_revoked():
    revoked: list[BaseException] = []

    async def abort(attempt, reason):
        try:
            await generate(attempt)
        except CaptureIntegrityFailure as error:
            revoked.append(error)

    r = rig(Runtime(on_sample=_fails, on_abort=abort))
    await r.rollout()
    (sampled,) = r.runtime.named("sample")
    (aborted,) = r.runtime.named("abort")
    assert aborted[1] is sampled[1]
    assert revoked and "revoked" in str(revoked[0])


async def test_runtime_metrics_have_no_effect_on_scheduling():
    async def noisy(attempt):
        # Names that shadow the broker's own gauges change nothing it decides.
        for name in ("slots/sampling", "pressure", "live_attempts"):
            attempt.services.metrics.set_gauge(name, 1e9)
            attempt.services.metrics.observe(name, -1e9)
        return await Runtime().on_sample(attempt)

    quiet, loud = rig(), rig(Runtime(on_sample=noisy))
    results = [await each.rollout() for each in (quiet, loud)]
    assert [result.reward for result in results] == [1.0, 1.0]
    assert quiet.capacity.snapshot() == loud.capacity.snapshot()
    assert quiet.admission.groups == loud.admission.groups


async def test_a_typed_primary_also_carries_its_cleanup_failure(caplog):
    raised = PermanentRolloutFailure("no point")

    async def fails(attempt):
        raise raised

    async def stop_fails(attempt, reason):
        raise OSError("sandbox stop failed")

    r = rig(Runtime(on_sample=fails, on_abort=stop_fails))
    await r.rollout()
    (cleanup,) = cleanup_failures(raised)
    assert isinstance(cleanup.__cause__, OSError)
    assert any("sandbox stop failed" in note for note in raised.__notes__)


async def test_a_final_grading_failure_is_reported_with_its_finish_failure(caplog):
    async def grading_fails(attempt, rollout):
        raise ValueError("verifier crashed")

    async def stop_fails(*args):
        raise OSError("sandbox stop failed")

    r = rig(Runtime(on_grade=grading_fails, on_finish=stop_fails))
    assert await r.rollout() is Discard.GRADING_FAILED
    (primary,) = _logged(caplog, "grading")
    assert any("sandbox stop failed" in note for note in primary.__notes__)
    assert cleanup_failures(primary)


async def test_a_stopped_sample_loses_model_access_before_its_grace():
    seen = []

    async def keeps_sampling(attempt):
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Still inside the grace: the lease must already refuse.
                try:
                    await generate(attempt)
                except CaptureIntegrityFailure as error:
                    seen.append(str(error))
                raise

    r = rig(Runtime(on_sample=keeps_sampling), phases=SAMPLING, rollout_timeout_seconds=0.05)
    assert await r.rollout() is Discard.AGENT_TIMEOUT
    assert seen and "revoked" in seen[0]


async def test_sealing_quiesces_the_attempts_model_access_first():
    order = []

    def recording(lease):
        class Recording(DirectModelAccess):
            async def quiesce(self, grace_seconds):
                order.append(("quiesce", lease.state, grace_seconds))

        return Recording(lease, Renderers(Renderer()))

    r = rig(access=recording)
    await r.rollout()
    assert order == [("quiesce", LeaseState.OPEN, 0.2)]  # before the lease sealed


# ------------------------------------------------------------- sub-reasons


class SandboxLost(RuntimeError):
    """A runtime's own failure type, named by its type."""


class Named(NamedFailure):
    """A runtime's own failure, named by the runtime (Harbor names Harbor's exception)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, name="VerifierTimeoutError")


class Unnamed(NamedFailure):
    """A failure that could name itself but does not: its type names it."""


class Attributed(RuntimeError):
    """A ``sub_reason`` attribute is no contract; only ``NamedFailure`` names a failure."""

    sub_reason = "VerifierTimeoutError"


def _wrapped() -> PermanentRolloutFailure:
    try:
        raise Named("the verifier timed out")
    except Named as cause:
        try:
            raise PermanentRolloutFailure("excluded from retry") from cause
        except PermanentRolloutFailure as error:
            return error


@pytest.mark.parametrize(
    ("error", "cause", "retried"),
    [
        (OSError("sandbox died"), "rollout_failed/os", True),
        (SandboxLost("gone"), "rollout_failed/sandbox_lost", True),
        (Named("timed out"), "rollout_failed/verifier_timeout", True),
        (Unnamed("timed out"), "rollout_failed/unnamed", True),
        (Attributed("timed out"), "rollout_failed/attributed", True),
        (_wrapped(), "rollout_failed/verifier_timeout", False),
        (TimeoutError("the agent's own deadline"), "rollout_failed/timeout", True),
    ],
    ids=[
        "type",
        "runtime-type",
        "runtime-named",
        "runtime-unnamed",
        "attribute-ignored",
        "wrapped",
        "runtime-deadline",
    ],
)
async def test_every_discard_and_retry_is_counted_by_its_sub_reason(error, cause, retried):
    async def failing(attempt):
        await generate(attempt)
        raise error

    r = rig(Runtime(on_sample=failing), max_rollout_retries=1)
    await r.rollout()
    expected = {f"discard/rollouts/{cause}": 1.0}
    if retried:
        expected[f"retried/attempts/{cause}"] = 1.0
    assert r.driver.gauges() == expected


async def test_a_grading_failure_and_its_retries_are_counted_by_its_sub_reason():
    async def failing(attempt, rollout):
        raise Named("timed out")

    r = rig(Runtime(on_grade=failing), grading_retries=2)
    assert await r.rollout() is Discard.GRADING_FAILED
    assert r.driver.gauges() == {
        "retried/attempts/grading_failed/verifier_timeout": 2.0,
        "discard/rollouts/grading_failed/verifier_timeout": 1.0,
    }


@pytest.mark.parametrize(
    "result",
    [GradeResult(reward=float("nan")), GradeResult(reward=1.0, metadata={"answer": "grade"})],
    ids=["non-finite", "metadata-clash"],
)
async def test_a_grader_bug_is_not_retried_even_when_grading_is_retry_safe(result):
    async def buggy(attempt, rollout):
        return result

    r = rig(Runtime(on_grade=buggy), grading_retries=3)
    assert await r.rollout() is Discard.GRADING_FAILED
    assert len(r.runtime.named("grade")) == 1


# Whether the Loops run has ended: a sampling failure is only a reason to ask.


class RunEnded:
    """A fake ``run_ended``: answers in turn, the last for every later question."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked = 0

    async def __call__(self):
        self.asked += 1
        answer = self.answers[min(self.asked, len(self.answers)) - 1]
        if isinstance(answer, BaseException):
            raise answer
        return answer


def _sampling_fails(run_ended, *, swallowed=False, **settings):
    runtime = Runtime()
    r = rig(runtime, run_ended=run_ended, **settings)

    async def failed(prompt, **kwargs):
        raise ConnectionError("Model version w55zd67 is deactivated")

    async def samples(attempt):
        r.sampler.sample_async = failed  # type: ignore[method-assign]
        if swallowed:
            with contextlib.suppress(ModelTransportFailure):
                await generate(attempt)
            return EpisodeResult()
        await generate(attempt)
        return EpisodeResult()

    runtime.on_sample = samples
    return r


@pytest.mark.parametrize("swallowed", [False, True], ids=["propagated", "swallowed"])
async def test_a_run_loops_says_has_ended_stops_at_the_failure_that_asked(swallowed):
    run_ended = RunEnded("Loops reports run 4w50xmq INACTIVE")
    # The consecutive-failure stop is off, as it was in the run that found this.
    r = _sampling_fails(run_ended, swallowed=swallowed, max_consecutive_failures=0)
    assert await r.rollout() is Discard.ROLLOUT_FAILED
    await settle_loop()
    [stop] = r.systemic
    assert isinstance(stop, LoopsRunEnded) and stop.ended == "Loops reports run 4w50xmq INACTIVE"


async def test_a_failure_while_the_run_is_live_is_left_to_the_budget():
    run_ended = RunEnded(None)
    r = _sampling_fails(run_ended, max_consecutive_failures=2)
    assert await r.rollout("g-r1") is Discard.ROLLOUT_FAILED
    await settle_loop()
    assert run_ended.asked == 1 and not r.systemic
    assert await r.rollout("g-r2") is Discard.ROLLOUT_FAILED
    await settle_loop()
    [stop] = r.systemic
    assert "2 consecutive sampling infrastructure failures" in str(stop)


async def test_a_status_check_that_fails_does_not_stop_the_run():
    run_ended = RunEnded(RuntimeError("api.baseten.co unreachable"))
    r = _sampling_fails(run_ended, max_consecutive_failures=0)
    assert await r.rollout() is Discard.ROLLOUT_FAILED
    await settle_loop()
    assert run_ended.asked == 1 and not r.systemic


@pytest.mark.parametrize(("interval", "asked"), [(45.0, 1), (0.0, 3)])
async def test_the_run_is_asked_about_at_most_once_per_interval(interval, asked):
    run_ended = RunEnded(None)
    r = _sampling_fails(run_ended, max_consecutive_failures=0, run_status_interval_seconds=interval)
    for n in range(1, 4):
        assert await r.rollout(f"g-r{n}") is Discard.ROLLOUT_FAILED
        await settle_loop()
    assert run_ended.asked == asked and not r.systemic
