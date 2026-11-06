"""The cancellation matrix for the driver's hooks: every landing at every await of a stop.

A hook is stopped by an outside cancellation or its own deadline. The driver then
ends the attempt's processes (``kill``) and gives the hook its grace (``grace``);
at each, nothing, a second cancellation, or a failure can land. Whatever lands,
a hook still running keeps everything it holds and is reported unquiesced; a hook
that stopped releases everything exactly once; an outside cancellation propagates,
with a hook still running as its cause.
"""

import asyncio

import pytest
from support.cancellation import (
    ALL_LANDINGS,
    Case,
    Gate,
    Landing,
    Outcome,
    Response,
    Trigger,
    Work,
    WorkFailed,
    cancellation_matrix,
    outcome_within,
    settle_loop,
)
from support.harness import TASK, Rig, Runtime, rig

from loops_cookbook.runtime.api import AttemptAbortReason, EpisodeDisposition, Phase
from loops_cookbook.runtime.driver import Discard, NonQuiescentHook
from loops_cookbook.runtime.processes import AttemptProcessSupervisor, ProcessKillFailure

DEADLINE = 0.05
GRACE = 0.2


@pytest.fixture
def kill(monkeypatch) -> Gate:
    """The kill of a stopped hook's processes, gated: the first one made while stopping."""
    gate = Gate(ProcessKillFailure("the group outlived SIGKILL"))
    pause, terminate = AttemptProcessSupervisor.pause, AttemptProcessSupervisor.terminate
    stopping: set[int] = set()

    def paused(self):
        pause(self)
        stopping.add(id(self))

    async def gated(self):
        if id(self) in stopping and not gate.entered.is_set():
            await gate.passage()
        else:
            await terminate(self)

    monkeypatch.setattr(AttemptProcessSupervisor, "pause", paused)
    monkeypatch.setattr(AttemptProcessSupervisor, "terminate", gated)
    return gate


async def play(case: Case, caller: asyncio.Task, work: Work, kill: Gate) -> object:
    """Land the case's events at each await of the stop; the caller's outcome, raised or not."""
    async with asyncio.timeout(5):
        await work.running.wait()
        if case.trigger is Trigger.CANCEL:
            caller.cancel()
        await kill.entered.wait()
        if case.at("kill") is Landing.CANCEL:
            caller.cancel()
            await settle_loop()
        kill.open(case.at("kill"))
        await work.cancelled.wait()
        if case.at("grace") is Landing.CANCEL:
            caller.cancel()
            await settle_loop()
        work.let_go.set()
    return await outcome_within(caller, 5)


def assert_outcome(case: Case, outcome: object, own: object, kept: object = None) -> None:
    if case.outcome is Outcome.NON_QUIESCENT:
        # A kept sample() or grade() discards its rollout; a kept cleanup hook stops the run.
        assert outcome == kept if kept is not None else isinstance(outcome, NonQuiescentHook)
    elif case.outcome is Outcome.CANCELLED:
        assert isinstance(outcome, asyncio.CancelledError), outcome
        # the hook still running is reported on the cancellation, not in its place
        assert isinstance(outcome.__cause__, NonQuiescentHook) is case.still_running
    else:
        assert outcome == own


def assert_kept_iff_running(case: Case, r: Rig, work: Work, hook: str, slot: str | None) -> None:
    running = case.still_running
    assert work.finished is not running
    assert work.cancellations == 1  # cancelled once, however often the driver was
    assert r.driver.unquiesced == ((f"{hook}:g-r1",) if running else ())
    assert r.driver.live_attempts == (1 if running else 0)  # never settled twice
    if slot is not None:
        assert r.capacity.snapshot()[f"slots/{slot}"] == (1 if running else 0)


STOPPED_BY_EITHER = cancellation_matrix(
    triggers=tuple(Trigger),
    points={"kill": ALL_LANDINGS, "grace": (Landing.NOTHING, Landing.CANCEL)},
    responses=tuple(Response),
)


@STOPPED_BY_EITHER
async def test_a_stopped_sample_keeps_its_hold_iff_it_is_still_running(case, kill):
    work = Work(case.response)

    async def sample(attempt):
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            await work()

    kept = []

    async def keep(transcript):
        kept.append(transcript)

    r = rig(
        Runtime(on_sample=sample),
        phases=frozenset({Phase.SAMPLING}),
        rollout_timeout_seconds=DEADLINE if case.trigger is Trigger.DEADLINE else 30,
        cancellation_grace_seconds=GRACE if case.still_running else 30,
        keep_transcript=keep,
    )
    try:
        caller = asyncio.create_task(r.driver.sample(TASK, "g", "g-r1"))
        outcome = await play(case, caller, work, kill)
        assert_outcome(case, outcome, own=Discard.AGENT_TIMEOUT, kept=Discard.UNQUIESCED)
        assert_kept_iff_running(case, r, work, "sample", "sampling")
        if case.still_running:
            # Written while sample() still runs, cancelled or not: its cleanup waits
            # for a hook that may never end.
            assert [transcript.sealed for transcript in kept] == [False]
        aborts = [call[2] for call in r.runtime.named("abort")]
        if case.still_running:
            assert aborts == []  # no further hook while it runs
        elif case.cancelled:
            assert aborts == [AttemptAbortReason.CANCELLED]
        else:
            assert aborts == [AttemptAbortReason.ROLLOUT_TIMEOUT]
    finally:
        work.end()
        await settle_loop()
        await r.driver.drain()  # a kept hook's cleanup runs once it ends


@STOPPED_BY_EITHER
async def test_a_stopped_grade_keeps_its_slot_iff_it_is_still_running(case, kill):
    work = Work(case.response)
    r = rig(
        Runtime(on_grade=work),
        grading_timeout_seconds=DEADLINE if case.trigger is Trigger.DEADLINE else 30,
        cancellation_grace_seconds=GRACE if case.still_running else 30,
    )
    try:
        admitted = await r.driver.sample(TASK, "g", "g-r1")
        caller = asyncio.create_task(r.driver.grade(admitted))
        outcome = await play(case, caller, work, kill)
        await r.driver.drain()
        assert_outcome(case, outcome, own=Discard.GRADING_FAILED, kept=Discard.UNQUIESCED)
        assert_kept_iff_running(case, r, work, "grade", "verification")
        finishes = [call[3] for call in r.runtime.named("finish")]
        assert finishes == ([] if case.still_running else [EpisodeDisposition.FAILED])
    finally:
        work.end()
        await settle_loop()
        await r.driver.drain()  # a kept hook's cleanup runs once it ends


@cancellation_matrix(
    # A cleanup hook runs shielded from its caller's cancellation: only its deadline stops it.
    triggers=(Trigger.DEADLINE,),
    points={"kill": (Landing.NOTHING, Landing.FAILURE)},
    responses=tuple(Response),
)
async def test_a_stopped_finish_keeps_its_attempt_live_iff_it_is_still_running(case, kill):
    work = Work(case.response)
    r = rig(
        Runtime(on_finish=work),
        cleanup_timeout_seconds=DEADLINE,
        cancellation_grace_seconds=GRACE if case.still_running else 30,
    )
    try:
        admitted = await r.driver.sample(TASK, "g", "g-r1")
        await r.driver.grade(admitted)
        caller = asyncio.create_task(r.driver.drain())
        assert await play(case, caller, work, kill) is None
        assert_kept_iff_running(case, r, work, "finish", None)
        assert r.systemic == []  # one kept cleanup hook is a cleanup failure
    finally:
        work.end()


@cancellation_matrix(
    # The seal outlasts its transcript write and never cancels it; the abort that
    # follows runs to its end. A cancellation starts the stop; then, at the write and
    # at the abort, a second one or the write's failure can land.
    triggers=(Trigger.CANCEL,),
    points={"write": ALL_LANDINGS, "abort": (Landing.NOTHING, Landing.CANCEL)},
    responses=(Response.STOPS,),
)
async def test_a_seal_cancelled_while_keeping_its_transcript_aborts_once_then_propagates(case):
    write, abort = Gate(OSError("no space left on device")), Gate(AssertionError())
    kept = []

    async def keep(transcript):
        await write.passage()
        kept.append(transcript)

    async def aborting(attempt, reason):
        await abort.passage()

    r = rig(Runtime(on_abort=aborting), keep_transcript=keep)
    caller = asyncio.create_task(r.driver.sample(TASK, "g", "g-r1"))
    async with asyncio.timeout(5):
        await write.entered.wait()
        caller.cancel()
        await settle_loop()
        if case.at("write") is Landing.CANCEL:
            caller.cancel()
            await settle_loop()
        write.open(case.at("write"))
        aborted = asyncio.create_task(abort.entered.wait())
        await asyncio.wait({caller, aborted}, return_when=asyncio.FIRST_COMPLETED)
        if case.at("abort") is Landing.CANCEL:
            caller.cancel()
            await settle_loop()
        abort.open(Landing.NOTHING)
        outcome = (await asyncio.gather(caller, return_exceptions=True))[0]
        live = r.driver.live_attempts  # as the caller saw it end
        aborted.cancel()

    assert isinstance(outcome, asyncio.CancelledError), outcome
    assert [call[2] for call in r.runtime.named("abort")] == [AttemptAbortReason.CANCELLED]
    assert r.runtime.named("finish") == []
    assert live == 0  # the abort ran to its end before the cancellation went on
    # Aborted, so the transcript it leaves last says it never sealed, and never trains.
    if case.at("write") is Landing.FAILURE:
        assert kept == []
    else:
        assert [transcript.sealed for transcript in kept] == [True, False]
        assert not any(turn.trainable for turn in kept[-1].turns)


@cancellation_matrix(
    triggers=tuple(Trigger),
    points={"abort": ALL_LANDINGS},
    responses=(Response.STOPS,),
)
async def test_a_stopped_samples_abort_runs_out_whatever_cancels_its_caller(case):
    """Shutdown cancels again, twice, while abort() runs: the caller waits it out."""
    work = Work(case.response)
    abort = Gate(WorkFailed("abort failed"))

    async def aborts(*args):
        await abort.passage()

    r = rig(
        Runtime(on_sample=work, on_abort=aborts),
        rollout_timeout_seconds=DEADLINE if case.trigger is Trigger.DEADLINE else 30,
    )
    try:
        caller = asyncio.create_task(r.driver.sample(TASK, "g", "g-r1"))
        async with asyncio.timeout(5):
            await work.running.wait()
            if case.trigger is Trigger.CANCEL:
                caller.cancel()
            await work.cancelled.wait()
            work.let_go.set()
            await abort.entered.wait()
            if case.at("abort") is Landing.CANCEL:
                for _ in range(2):
                    caller.cancel()
                    await settle_loop()
            assert not caller.done()  # abort() is still running
            abort.open(case.at("abort"))
        outcome = await outcome_within(caller, 5)
        assert_outcome(case, outcome, own=Discard.AGENT_TIMEOUT)
        assert len(r.runtime.named("abort")) == 1
        assert r.driver.live_attempts == 0
    finally:
        work.end()
        abort.open(Landing.NOTHING)
