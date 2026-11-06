"""The shielded waits every stop ends in: what each guarantees for one, two, or no cancellations."""

import asyncio

import pytest
from support.cancellation import (
    Gate,
    Landing,
    Response,
    Trigger,
    Work,
    WorkFailed,
    cancellation_matrix,
    outcome_within,
    settle_loop,
)

from loops_cookbook.cancellation import Stopped, Waited, outlast, run_until, see_through, stop


async def _cancel(waiter: asyncio.Task[Waited], times: int) -> None:
    for _ in range(times):
        waiter.cancel()
        await settle_loop()


@cancellation_matrix(
    triggers=(Trigger.DEADLINE,),
    points={"grace": (Landing.NOTHING, Landing.CANCEL)},
    responses=tuple(Response),
)
async def test_stop_waits_out_the_grace_whatever_cancels_the_waiter(case):
    work = Work(case.response)
    task = asyncio.create_task(work())
    await work.running.wait()
    # Long enough never to end first for work that does stop; short for work that won't.
    waiter = asyncio.create_task(stop(task, 0.2 if case.still_running else 30))
    try:
        async with asyncio.timeout(5):
            await work.cancelled.wait()
            await _cancel(waiter, 2 if case.at("grace") is Landing.CANCEL else 0)
            assert not waiter.done()  # a cancellation never ends the wait early
            work.let_go.set()
        waited = await outcome_within(waiter, 5)
        assert waited == Waited(settled=not case.still_running, interrupted=case.cancelled)
        assert task.done() is waited.settled
        assert work.cancellations == 1  # cancelled once; never again, by the waiter's own
    finally:
        work.end()
        await asyncio.wait({task})


@cancellation_matrix(
    # Unbounded: work that ignored its cancellation would be waited for forever.
    triggers=(Trigger.DEADLINE,),
    points={"grace": (Landing.NOTHING, Landing.CANCEL)},
    responses=(Response.STOPS, Response.FAILS),
)
async def test_an_unbounded_stop_cancels_once_and_waits_the_task_out(case):
    work = Work(case.response)
    task = asyncio.create_task(work())
    await work.running.wait()
    waiter = asyncio.create_task(stop(task, None))
    try:
        async with asyncio.timeout(5):
            await work.cancelled.wait()
            await _cancel(waiter, 2 if case.at("grace") is Landing.CANCEL else 0)
            assert not waiter.done()  # a cancellation never ends the wait early
            work.let_go.set()
        waited = await outcome_within(waiter, 5)
        assert waited == Waited(settled=True, interrupted=case.cancelled)
        assert work.cancellations == 1
    finally:
        work.end()
        await asyncio.wait({task})


@pytest.mark.parametrize("cancels", [0, 1, 2])
@pytest.mark.parametrize("landing", [Landing.NOTHING, Landing.FAILURE], ids=lambda x: x.value)
async def test_outlast_waits_for_the_work_and_leaves_its_outcome_on_it(cancels, landing):
    gate = Gate(WorkFailed("the work failed"))
    work = asyncio.ensure_future(gate.passage())
    waiter = asyncio.create_task(outlast(work))
    async with asyncio.timeout(5):
        await gate.entered.wait()
        await _cancel(waiter, cancels)
        assert not waiter.done()
        gate.open(landing)
    waited = await outcome_within(waiter, 5)
    assert waited == Waited(settled=True, interrupted=cancels > 0)
    assert not work.cancelled()  # the waiter's cancellation never reached the work
    failure = work.exception()
    assert isinstance(failure, WorkFailed) if landing is Landing.FAILURE else failure is None


@pytest.mark.parametrize("cancels", [0, 2])
async def test_outlast_ends_at_its_timeout_with_the_work_still_running(cancels):
    gate = Gate(WorkFailed("unused"))
    work = asyncio.ensure_future(gate.passage())
    waiter = asyncio.create_task(outlast(work, 0.05))
    try:
        async with asyncio.timeout(5):
            await gate.entered.wait()
            await _cancel(waiter, cancels)
        waited = await outcome_within(waiter, 5)
        assert waited == Waited(settled=False, interrupted=cancels > 0)
        assert not work.done()
    finally:
        gate.open(Landing.NOTHING)
        await work


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("cause", [None, WorkFailed("kept")], ids=["alone", "after-failure"])
def test_propagate_raises_the_waiters_cancellation_only_with_the_failure_as_cause(
    interrupted, cause
):
    waited = Waited(settled=True, interrupted=interrupted)
    if not interrupted:
        waited.propagate(cause)
        return
    with pytest.raises(asyncio.CancelledError) as raised:
        waited.propagate(cause)
    assert raised.value.__cause__ is cause


@pytest.mark.parametrize("cancels", [0, 2])
@pytest.mark.parametrize("landing", [Landing.NOTHING, Landing.FAILURE], ids=lambda x: x.value)
async def test_see_through_gives_the_result_then_raises_what_cancelled_the_waiter(cancels, landing):
    gate = Gate(WorkFailed("the work failed"))

    async def result() -> str:
        await gate.passage()
        return "done"

    waiter = asyncio.create_task(see_through(asyncio.ensure_future(result())))
    async with asyncio.timeout(5):
        await gate.entered.wait()
        await _cancel(waiter, cancels)
        assert not waiter.done()  # a cancellation never ends the wait early
        gate.open(landing)
    outcome = await outcome_within(waiter, 5)
    failure = gate.failure if landing is Landing.FAILURE else None
    if cancels:
        assert isinstance(outcome, asyncio.CancelledError) and outcome.__cause__ is failure
    else:
        assert outcome == "done" if failure is None else outcome is failure


class ExpiryFailed(RuntimeError):
    """A bug in the caller's expiry."""


@cancellation_matrix(
    triggers=tuple(Trigger),
    # "expiry": the expiry returns, or fails, once it is due; the waiter's cancellation
    # cancels it before then.
    points={
        "expiry": (Landing.NOTHING, Landing.FAILURE),
        "grace": (Landing.NOTHING, Landing.CANCEL),
    },
    responses=tuple(Response),
)
async def test_run_until_stops_its_task_at_the_expiry_or_the_waiters_cancellation(case):
    """A failing expiry still stops the task; its failure is raised then, as a cause."""
    work = Work(case.response)
    task = asyncio.create_task(work())
    expired = asyncio.Event()
    fails = case.trigger is Trigger.DEADLINE and case.at("expiry") is Landing.FAILURE

    async def expiry() -> str:
        await expired.wait()
        if fails:
            raise ExpiryFailed("the expiry failed")
        return "expired"

    grace = 0.2 if case.still_running else 30
    waiter = asyncio.create_task(run_until(task, expiry(), grace, own_failure=RuntimeError))
    try:
        async with asyncio.timeout(5):
            await work.running.wait()
            if case.trigger is Trigger.CANCEL:
                waiter.cancel()
            else:
                expired.set()
            await work.cancelled.wait()
            await _cancel(waiter, 2 if case.at("grace") is Landing.CANCEL else 0)
            assert not waiter.done()
            work.let_go.set()
        outcome = await outcome_within(waiter, 5)
        if not fails:
            assert outcome == Stopped(
                settled=not case.still_running,
                interrupted=case.cancelled,
                expiry="expired" if case.trigger is Trigger.DEADLINE else None,
            )
        elif case.cancelled:
            assert isinstance(outcome, asyncio.CancelledError), outcome
            assert isinstance(outcome.__cause__, ExpiryFailed)
        else:
            assert isinstance(outcome, ExpiryFailed), outcome
        assert task.done() is not case.still_running
        assert work.cancellations == 1
    finally:
        work.end()
        await asyncio.wait({task})


async def test_run_until_gives_a_tasks_result_or_failure_and_its_own_cancellation_as_failure():
    async def fails() -> None:
        raise WorkFailed("the work failed")

    async def leaks() -> None:
        raise asyncio.CancelledError

    def own(leaked: asyncio.CancelledError) -> WorkFailed:
        return WorkFailed("its own cancellation")

    async def run(coroutine):
        task = asyncio.create_task(coroutine)
        return await run_until(task, asyncio.Event().wait(), 30, own_failure=own)

    async with asyncio.timeout(5):
        assert await run(asyncio.sleep(0, "done")) == "done"
        with pytest.raises(WorkFailed, match="the work failed"):
            await run(fails())
        with pytest.raises(WorkFailed, match="its own cancellation") as raised:
            await run(leaks())
    assert isinstance(raised.value.__cause__, asyncio.CancelledError)
