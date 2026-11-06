"""Start admission and phase permits: develop's ``wait_to_start``, now in the broker."""

import asyncio

import pytest
from support.harness import Admission

from loops_cookbook.evidence import AttemptIdentity
from loops_cookbook.runtime.api import Phase
from loops_cookbook.runtime.capacity import AttemptPhases, Hook, RunCapacity


def run_capacity(admission=None, *, sampling_slots=1, prestart=0):
    """The run's capacity broker, with one verification slot."""
    return RunCapacity(
        sampling_slots=sampling_slots,
        verification_slots=1,
        admission=admission or Admission(),
        on_change=lambda: None,
        prestart=prestart,
    )


def phases(rollout, group="g"):
    state = AttemptPhases(AttemptIdentity("run", group, rollout, f"a-{rollout}"), frozenset())
    state.start_hook(Hook.SAMPLE)
    return state


class GroupTargets(Admission):
    """develop's rule: started groups fill every slot; new ones follow the target."""

    def __init__(self, slots, target):
        super().__init__(target)
        self.slots = slots

    def target(self, group_id):
        return self.slots if group_id in self.groups else self.value


async def test_a_lower_target_never_interrupts_an_active_rollout_and_started_groups_finish():
    admission = GroupTargets(slots=2, target=2)
    broker = run_capacity(admission, sampling_slots=2)
    a, b, a2, c = phases("a", "ga"), phases("b", "gb"), phases("a2", "ga"), phases("c", "gc")
    for state in (a, b, a2, c):
        broker.track(state)
    first, second = broker.runner_hold(a, Phase.SAMPLING), broker.runner_hold(b, Phase.SAMPLING)
    await first.__aenter__()
    await second.__aenter__()
    admission.value = 0  # downstream pressure rose
    assert broker.snapshot()["admission/active_rollouts"] == 2  # nothing active was interrupted
    partial = asyncio.create_task(broker.runner_hold(a2, Phase.SAMPLING).__aenter__())
    fresh = asyncio.create_task(broker.runner_hold(c, Phase.SAMPLING).__aenter__())
    await asyncio.sleep(0.01)
    assert not partial.done()  # the physical slot cap still applies
    first.release()
    await asyncio.wait_for(partial, 1)  # a started group is admitted regardless of target
    await asyncio.sleep(0.01)
    assert not fresh.done()  # a new group waits for the target
    admission.value = 2
    second.release()
    broker.notify()
    await asyncio.wait_for(fresh, 1)


async def test_later_sampling_holds_skip_admission():
    admission = Admission(target=1)
    broker = run_capacity(admission, sampling_slots=4)
    state = phases("r")
    broker.track(state)
    async with broker.runner_hold(state, Phase.SAMPLING):
        pass
    admission.value = 0
    async with broker.runner_hold(state, Phase.SAMPLING):
        assert broker.snapshot()["admission/active_rollouts"] == 1
    assert admission.groups == ["g"]


async def test_cancelling_an_admission_wait_releases_its_reservation():
    admission = Admission(target=0)
    broker = run_capacity(admission)
    state = phases("r")
    broker.track(state)
    waiting = asyncio.create_task(broker.runner_hold(state, Phase.SAMPLING).__aenter__())
    await asyncio.sleep(0.01)
    assert broker.snapshot()["queue/sampling_waiting"] == 1
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    assert broker.snapshot()["queue/sampling_waiting"] == 0
    assert broker.snapshot()["admission/active_rollouts"] == 0


async def test_the_rollout_clock_counts_only_time_inside_sampling_holds():
    now = [0.0]
    broker = run_capacity()
    state = AttemptPhases(AttemptIdentity("run", "g", "r", "a"), frozenset(), clock=lambda: now[0])
    broker.track(state)
    state.start_hook(Hook.SAMPLE)
    now[0] = 5  # setup outside any hold
    async with broker.runner_hold(state, Phase.SAMPLING):
        now[0] = 7
        assert state.rollout_elapsed() == 2
    now[0] = 10
    assert state.rollout_elapsed() == 2
    assert state.setup_elapsed() == 8  # 5 before the hold and 3 after it


async def test_the_rollout_clock_stops_during_a_resource_operation_inside_a_sampling_hold():
    now = [0.0]
    broker = run_capacity()
    state = AttemptPhases(AttemptIdentity("run", "g", "r", "a"), frozenset(), clock=lambda: now[0])
    broker.track(state)
    state.start_hook(Hook.SAMPLE)
    async with broker.runner_hold(state, Phase.SAMPLING):
        now[0] = 1
        with state.resource_operation():
            now[0] = 11  # a sandbox create
            with state.resource_operation():
                now[0] = 21  # nested: still one stop
            now[0] = 31
            assert state.rollout_elapsed() == 1
        now[0] = 33
        assert state.rollout_elapsed() == 3
    now[0] = 40
    assert state.rollout_elapsed() == 3
    assert state.setup_elapsed() == 7  # after the hold; the create was excluded too


# ------------------------------------------------ environment start and pre-start


def sampling_owned(rollout, group="g"):
    """An attempt whose runtime acquires sampling itself, as Harbor does."""
    identity = AttemptIdentity("run", group, rollout, f"a-{rollout}")
    state = AttemptPhases(identity, frozenset({Phase.SAMPLING}))
    state.start_hook(Hook.SAMPLE)
    return state


async def admitting(broker, state):
    """``state``'s admission at environment start, and whether it is through already."""
    gate = asyncio.create_task(broker.track(state).admit())
    await asyncio.sleep(0.01)
    return gate


async def test_admission_at_environment_start_allows_the_target_plus_the_prestart_places():
    admission = Admission(target=2)
    broker = run_capacity(admission, sampling_slots=2, prestart=1)
    a, b, c, d = (sampling_owned(name) for name in "abcd")
    gates = [await admitting(broker, state) for state in (a, b, c, d)]
    assert [gate.done() for gate in gates] == [True, True, True, False]  # 2 to sample, 1 ahead
    first = broker.track(a).hold(Phase.SAMPLING)
    await first.__aenter__()
    async with broker.track(b).hold(Phase.SAMPLING):
        assert broker.snapshot()["admission/prestarted"] == 1  # c is up, waiting for a slot
        first.release()  # a's sampling ended: its admission is done, so d is admitted
        await asyncio.wait_for(gates[3], 1)


@pytest.mark.parametrize("prestart", [0, 2])
async def test_under_backpressure_no_new_group_starts_even_with_prestart_places(prestart):
    broker = run_capacity(Admission(target=0), sampling_slots=4, prestart=prestart)
    gates = [await admitting(broker, sampling_owned(name)) for name in "abc"]
    assert not any(gate.done() for gate in gates)
    for gate in gates:
        gate.cancel()
    await asyncio.gather(*gates, return_exceptions=True)


async def test_a_prestart_place_never_starts_a_group_the_target_holds_back():
    """With the pipeline's own group rule: a started group's siblings fill regardless of
    the target, so a pre-start place spent on a new group would bring all of it."""
    admission = GroupTargets(slots=4, target=1)  # backpressure: the target is below 4
    broker = run_capacity(admission, sampling_slots=4, prestart=2)
    await broker.track(sampling_owned("a", "ga")).admit()  # the target's one place
    held = [await admitting(broker, sampling_owned(f"b{i}", "gb")) for i in range(3)]
    assert not any(gate.done() for gate in held)  # gb never starts
    siblings = [await admitting(broker, sampling_owned(f"a{i}", "ga")) for i in range(6)]
    # A started group fills the limit, plus the pre-start places ahead of free slots.
    assert sum(gate.done() for gate in siblings) == 4 - 1 + 2
    assert set(admission.groups) == {"ga"}
    for gate in held + siblings:
        gate.cancel()
    await asyncio.gather(*held, *siblings, return_exceptions=True)


async def test_an_attempt_admitted_at_environment_start_takes_its_slot_without_waiting_again():
    admission = Admission(target=1)
    broker = run_capacity(admission)
    state = sampling_owned("r")
    await broker.track(state).admit()
    admission.value = 0  # pressure rose after it was admitted: it is not interrupted
    async with broker.track(state).hold(Phase.SAMPLING):
        assert broker.snapshot()["slots/sampling"] == 1
    assert admission.groups == ["g"]  # admitted once


async def test_admitting_needs_runtime_acquired_sampling():
    from loops_cookbook.runtime.capacity import RuntimeContractError

    broker = run_capacity()
    with pytest.raises(RuntimeContractError, match="internally_acquired_phases"):
        await broker.track(phases("r")).admit()


# ------------------------------------------------ admission is first come, first served


def fifo_broker(target=1, slots=4):
    admission = GroupTargets(slots=slots, target=target)
    broker = run_capacity(admission, sampling_slots=slots)
    return broker, admission


async def test_waiting_attempts_are_admitted_in_arrival_order_and_nobody_jumps_the_queue():
    """A place that frees goes to the attempt waiting longest, not one arriving just then."""
    broker, _ = fifo_broker(target=1)
    first = sampling_owned("x", "gx")
    await broker.track(first).admit()
    a = await admitting(broker, sampling_owned("a", "ga"))
    b = await admitting(broker, sampling_owned("b", "gb"))

    async def arrives_as_a_place_frees(freed):
        broker.forget(freed)  # in the same step, before any waiter has run
        await broker.track(sampling_owned("c", "gc")).admit()

    racer = asyncio.create_task(arrives_as_a_place_frees(first))
    await asyncio.wait_for(a, 1)
    await asyncio.sleep(0.01)
    assert not b.done() and not racer.done()  # b waits behind a, c behind b
    for waiting in (b, racer):
        waiting.cancel()
    await asyncio.gather(b, racer, return_exceptions=True)


async def test_a_retried_attempt_of_an_older_group_is_admitted_before_newer_groups():
    """The retry arrives after a newer group's attempt; its group was dispatched first."""
    broker, admission = fifo_broker(target=1, slots=1)
    admission.orders.update({"gx": 0, "g0": 1, "g5": 6})
    running = sampling_owned("x", "gx")
    await broker.track(running).admit()  # every place is taken
    newer = await admitting(broker, sampling_owned("n", "g5"))
    retried = await admitting(broker, sampling_owned("r", "g0"))
    assert not newer.done() and not retried.done()
    broker.forget(running)
    await asyncio.wait_for(retried, 1)
    await asyncio.sleep(0.01)
    assert not newer.done()  # the one free place went to the older group
    assert admission.groups == ["gx", "g0"]
    newer.cancel()
    await asyncio.gather(newer, return_exceptions=True)


async def test_attempts_of_one_group_are_admitted_in_arrival_order():
    broker, _ = fifo_broker(target=1, slots=1)
    running = sampling_owned("x", "gx")
    await broker.track(running).admit()
    first = await admitting(broker, sampling_owned("a", "g"))
    second = await admitting(broker, sampling_owned("b", "g"))
    broker.forget(running)
    await asyncio.wait_for(first, 1)
    await asyncio.sleep(0.01)
    assert not second.done()
    second.cancel()
    await asyncio.gather(second, return_exceptions=True)


async def test_a_started_groups_sibling_passes_new_groups_waiting_ahead_of_it():
    broker, admission = fifo_broker(target=1)
    await broker.track(sampling_owned("x", "gx")).admit()
    fresh = await admitting(broker, sampling_owned("u", "gu"))
    sibling = await admitting(broker, sampling_owned("x2", "gx"))
    assert sibling.done() and not fresh.done()  # the started group fills its slots
    assert admission.groups == ["gx", "gx"]
    fresh.cancel()
    await asyncio.gather(fresh, return_exceptions=True)


async def test_an_attempt_cancelled_as_it_is_admitted_gives_its_place_to_the_next():
    broker, _ = fifo_broker(target=1)
    first = sampling_owned("x", "gx")
    await broker.track(first).admit()
    a = await admitting(broker, sampling_owned("a", "ga"))
    b = await admitting(broker, sampling_owned("b", "gb"))
    broker.forget(first)  # admits a ...
    a.cancel()  # ... which is cancelled before it runs again
    await asyncio.gather(a, return_exceptions=True)
    await asyncio.wait_for(b, 1)
    assert broker.snapshot()["admission/active_rollouts"] == 1


async def test_a_waiter_cancelled_just_before_a_place_frees_is_skipped():
    broker, _ = fifo_broker(target=1)
    first = sampling_owned("x", "gx")
    await broker.track(first).admit()
    a = await admitting(broker, sampling_owned("a", "ga"))
    b = await admitting(broker, sampling_owned("b", "gb"))
    a.cancel()  # still queued when ...
    broker.forget(first)  # ... the place frees: it goes to b
    await asyncio.gather(a, return_exceptions=True)
    await asyncio.wait_for(b, 1)
    assert broker.snapshot()["admission/active_rollouts"] == 1


async def test_a_group_grader_call_takes_one_of_the_same_verification_slots_as_grade():
    broker = run_capacity()
    state = phases("r")
    state.start_hook(Hook.GRADE)
    broker.track(state)
    async with broker.group_verification():
        snapshot = broker.snapshot()
        assert snapshot["slots/verification"] == 1
        grade = asyncio.create_task(broker.runner_hold(state, Phase.VERIFICATION).__aenter__())
        judge = asyncio.create_task(broker.group_verification().__aenter__())
        await asyncio.sleep(0.01)
        assert not grade.done() and not judge.done()  # one slot, held by the judge
        assert broker.snapshot()["queue/verification_waiting"] == 2
    await asyncio.wait_for(asyncio.wait({grade, judge}, return_when="FIRST_COMPLETED"), 1)
    assert grade.done() != judge.done()  # the freed slot went to exactly one of them
    for waiter in (grade, judge):
        waiter.cancel()
