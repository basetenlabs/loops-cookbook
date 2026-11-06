"""The cancellation matrix for the judge stage: every landing at every await of its stop.

A ``grade_group()`` call is stopped by an outside cancellation, as on shutdown, or by
its grading deadline. The stage then gives it its grace (``grace``), where nothing or
a second cancellation can land; that wait is the stop's only await, so no step of the
stage's own can fail there: the judge failing while it stops is the ``fails``
response. Whatever lands, a judge still running keeps its verification slot and is
counted as a call; a judge that stopped gives both back; an outside cancellation
propagates.
"""

import asyncio

from support.cancellation import (
    Case,
    Landing,
    Outcome,
    Response,
    Trigger,
    Work,
    cancellation_matrix,
    settle_loop,
)
from support.harness import Admission

from loops_cookbook.runtime.capacity import RunCapacity
from loops_cookbook.runtime.driver import HookDeadline, NonQuiescentHook
from loops_cookbook.runtime.groups import GroupGrading, GroupGradingSettings

DEADLINE = 0.05
GRACE = 0.2


class Judge:
    def __init__(self, work: Work) -> None:
        self.work = work

    async def grade_group(self, members: object) -> object:
        await self.work()
        return []


async def play(case: Case, caller: asyncio.Task, work: Work) -> object:
    """Land the case's events at each await of the stop; the caller's outcome, raised or not."""
    async with asyncio.timeout(5):
        await work.running.wait()
        if case.trigger is Trigger.CANCEL:
            caller.cancel()
        await work.cancelled.wait()
        if case.at("grace") is Landing.CANCEL:
            caller.cancel()
            await settle_loop()
        work.let_go.set()
        try:
            return await caller
        except BaseException as error:  # noqa: BLE001 - the outcome under test
            return error


@cancellation_matrix(
    triggers=tuple(Trigger),
    points={"grace": (Landing.NOTHING, Landing.CANCEL)},
    responses=tuple(Response),
)
async def test_a_stopped_judge_keeps_its_slot_and_its_call_iff_it_is_still_running(case):
    work = Work(case.response)
    capacity = RunCapacity(
        sampling_slots=1, verification_slots=1, admission=Admission(), on_change=lambda: None
    )
    observed: list[float] = []
    unquiesced: list[str] = []
    grading = GroupGrading(
        Judge(work),
        capacity,
        GroupGradingSettings(
            timeout_seconds=DEADLINE if case.trigger is Trigger.DEADLINE else 30,
            cancellation_grace_seconds=GRACE if case.still_running else 30,
        ),
        observe=observed.append,
        unquiesced=lambda group_id, error: unquiesced.append(group_id),
    )
    try:
        outcome = await play(case, asyncio.create_task(grading.score("g", [])), work)
        if case.outcome is Outcome.NON_QUIESCENT:
            assert isinstance(outcome, NonQuiescentHook), outcome
        elif case.outcome is Outcome.CANCELLED:
            assert isinstance(outcome, asyncio.CancelledError), outcome
        else:
            assert isinstance(outcome, HookDeadline), outcome
        running = case.still_running
        assert work.finished is not running
        assert work.cancellations == 1  # cancelled once, however often the stage was
        assert capacity.snapshot()["slots/verification"] == (1 if running else 0)
        assert grading.calls == (1 if running else 0)  # a kept call is still a call
        assert len(observed) == 1
        assert unquiesced == (["g"] if running else [])  # reported, cancelled or not
    finally:
        work.end()
