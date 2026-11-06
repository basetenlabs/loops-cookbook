"""A Harbor trial stopped by a deadline or a cancel: what it keeps, and until when."""

import asyncio

import pytest
from support.cancellation import (
    Landing,
    Response,
    Trigger,
    Work,
    cancellation_matrix,
    outcome_within,
    settle_loop,
)

from loops_cookbook.harbor import attempts
from loops_cookbook.runtime.api import (
    AttemptAbortReason,
)
from loops_cookbook.runtime.driver import Discard

from .jobs import HILL_CLIMB, harbor, live_trials, sampled, separate_steps_task


async def test_a_regrade_cancelled_at_its_deadline_returns_promptly(tmp_path, monkeypatch):
    """grade() returns at its deadline, without waiting for the verifier sandbox's stop."""
    import baseline_environment

    grading, stop_released = asyncio.Event(), asyncio.Event()
    stop, run = (
        baseline_environment.LocalEnvironment.stop,
        baseline_environment.LocalEnvironment.exec,
    )

    async def slow_stop(self, delete):
        if grading.is_set():
            await stop_released.wait()
        return await stop(self, delete)

    async def hanging_verifier(self, command, *args, **kwargs):
        if grading.is_set() and "test" in str(command):
            await asyncio.Event().wait()
        return await run(self, command, *args, **kwargs)

    monkeypatch.setattr(baseline_environment.LocalEnvironment, "stop", slow_stop)
    monkeypatch.setattr(baseline_environment.LocalEnvironment, "exec", hanging_verifier)
    r, runtime, task = await harbor(tmp_path, HILL_CLIMB, grading_timeout_seconds=0.3)
    admitted = await r.driver.sample(task, "g", "g-r1")
    grading.set()
    graded = await asyncio.wait_for(r.driver.grade(admitted), 10)  # the stop is still held
    assert graded is Discard.GRADING_FAILED
    stop_released.set()
    await r.driver.drain()
    assert not r.systemic and live_trials(runtime) == 0
    assert r.driver.live_attempts == 0


async def test_a_trial_that_swallows_a_cancel_is_still_discarded(tmp_path, monkeypatch):
    """Harbor can swallow one cancel in teardown and park at the next verification."""
    import baseline_environment

    stop = baseline_environment.LocalEnvironment.stop

    async def slow_agent_stop(self, delete):
        if "verifier" not in self.session_id:
            await asyncio.sleep(1.0)
        return await stop(self, delete)

    monkeypatch.setattr(baseline_environment.LocalEnvironment, "stop", slow_agent_stop)
    r, runtime, task = await harbor(
        tmp_path,
        separate_steps_task(tmp_path),
        setup_timeout_seconds=0.6,
        cleanup_timeout_seconds=10.0,
    )
    outcome = await asyncio.wait_for(r.driver.sample(task, "g", "g-r1"), 15)
    assert outcome is Discard.SETUP_TIMEOUT  # the agent's slow stop ran out the setup clock
    await r.driver.drain()
    assert not r.systemic
    assert live_trials(runtime) == 0
    assert r.driver.live_attempts == 0


async def test_a_trial_that_outlives_its_cleanup_deadline_stays_owned_until_it_ends(
    tmp_path, monkeypatch
):
    """A discard cut off at the cleanup deadline keeps the trial until Harbor lets it go.

    The sandbox start swallows every cancel for a while, then Harbor carries the trial
    on to its agent: the kept abort() is a cleanup failure, the escaped trial's
    sampling hold is refused, and the trial is forgotten once it ends.
    """
    import baseline_environment

    start = baseline_environment.LocalEnvironment.start

    async def stubborn_start(self, force_build):
        until = asyncio.get_running_loop().time() + 1.5
        while (left := until - asyncio.get_running_loop().time()) > 0:
            try:
                await asyncio.sleep(left)
            except asyncio.CancelledError:
                pass
        return await start(self, force_build)

    monkeypatch.setattr(baseline_environment.LocalEnvironment, "start", stubborn_start)
    r, runtime, task = await harbor(
        tmp_path, HILL_CLIMB, setup_timeout_seconds=0.3, cleanup_timeout_seconds=0.3
    )
    sampling = sampled(runtime)
    outcome = await asyncio.wait_for(r.driver.sample(task, "g", "g-r1"), 15)
    [attempt_id] = [attempt.spec.identity.attempt_id for attempt in sampling]
    trials = [t for t in asyncio.all_tasks() if t.get_name().startswith("harbor:")]
    await asyncio.wait_for(asyncio.wait(trials), 15)
    await asyncio.sleep(0.1)  # the discard's own wake-up after the trial ends
    assert outcome is Discard.SETUP_TIMEOUT
    assert not r.systemic
    assert r.driver.unquiesced == ()  # its abort() ended with the trial
    assert r.driver.live_attempts == 0
    assert live_trials(runtime) == 0
    with pytest.raises(RuntimeError, match="no live attempt"):
        attempts.attempt_for(attempt_id)


@cancellation_matrix(
    triggers=(Trigger.CANCEL,),
    points={"trial": (Landing.NOTHING, Landing.CANCEL)},
    responses=tuple(Response),
)
async def test_a_cancelled_abort_keeps_its_trial_until_harbor_ends_it(tmp_path, monkeypatch, case):
    """However the trial takes its cancel, abort() forgets it only once it has ended,
    and then raises the cancellation it waited through."""
    import baseline_environment

    work = Work(case.response)

    async def starting(self, force_build):
        await work()

    monkeypatch.setattr(baseline_environment.LocalEnvironment, "start", starting)
    r, runtime, task = await harbor(tmp_path, HILL_CLIMB, setup_timeout_seconds=60)
    attempts_sampled = sampled(runtime)
    sampling = asyncio.create_task(r.driver.sample(task, "g", "g-r1"))
    try:
        await asyncio.wait_for(work.running.wait(), 10)
        [attempt] = attempts_sampled
        attempt_id = attempt.spec.identity.attempt_id
        aborting = asyncio.create_task(runtime.abort(attempt, AttemptAbortReason.CANCELLED))
        await asyncio.wait_for(work.cancelled.wait(), 5)
        aborting.cancel()
        await settle_loop()
        if case.at("trial") is Landing.CANCEL:
            aborting.cancel()
            await settle_loop()
        assert not aborting.done()
        assert live_trials(runtime) == 1  # Harbor still has the trial
        assert attempts.attempt_for(attempt_id) is not None
        work.end()
        assert isinstance(await outcome_within(aborting, 15), asyncio.CancelledError)
        assert live_trials(runtime) == 0
        with pytest.raises(RuntimeError, match="no live attempt"):
            attempts.attempt_for(attempt_id)
    finally:
        work.end()
        await outcome_within(sampling, 15)
        await r.driver.drain()
