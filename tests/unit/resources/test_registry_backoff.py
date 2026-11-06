"""The resource registry under refusals: a refused create waits and retries, and each
provider's create window narrows and widens."""

import asyncio
import dataclasses
import logging

import pytest
from support.cancellation import settle_loop
from support.resources import FAST, Backoff, FakeProvider, not_created, refused, registry_at

from loops_cookbook.resources.api import CreationDisposition, ProvisioningFailure, ResourceScope

from .rig import events, provision

# Half of it, 10 s, is the backoff budget: waits of 1, 2 and 4 s fit, 8 more does not.
PATIENT = dataclasses.replace(FAST, create_seconds=20)


@dataclasses.dataclass
class TimedProvider(FakeProvider):
    """A ``FakeProvider`` whose creates take ``durations`` seconds, one each, in order."""

    durations: list[float] = dataclasses.field(default_factory=list)

    async def create(self, intent):
        self.delay = self.durations.pop(0)  # applied once the call is recorded
        return await super().create(intent)


def window(registry, provider="fake"):
    return registry._windows[provider].size


@dataclasses.dataclass
class SharedProvider(FakeProvider):
    """A ``FakeProvider`` with room for ``capacity`` creates in flight, shared with
    ``background`` creates of other clients: a create past that room is refused."""

    capacity: int = 8
    background: int = 0
    duration: float = 0.01  # seconds each create it takes is in flight
    ours: int = 0
    refusals: int = 0

    async def create(self, intent):
        if self.ours + self.background >= self.capacity:
            self.calls.append(("create", intent.desired_name))
            self.refusals += 1
            raise refused()
        self.ours += 1
        try:
            await asyncio.sleep(self.duration)
            return await super().create(intent)
        finally:
            self.ours -= 1


async def burst(registry, keys):
    return await asyncio.gather(*(provision(registry, key) for key in keys), return_exceptions=True)


@dataclasses.dataclass
class NamedTimedProvider(FakeProvider):
    """A ``FakeProvider`` whose creates of each name take ``durations[name]`` seconds."""

    durations: dict[str, list[float]] = dataclasses.field(default_factory=dict)

    async def create(self, intent):
        self.delay = self.durations[intent.desired_name].pop(0)
        return await super().create(intent)


async def test_a_refused_create_is_sent_again_after_a_growing_wait(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="loops_cookbook.resources.registry")
    provider = FakeProvider(deadlines=PATIENT, create_errors=[refused(), refused()])
    backoff = Backoff()
    registry = registry_at(tmp_path, provider, backoff=backoff)
    handle = await provision(registry)
    assert backoff.waits == [1, 2]
    assert provider.calls.count(("create", "fake-a")) == 3
    assert events(registry, handle.intent.resource_id) == ["intent", "created"]
    retries = [r for r in caplog.records if "sending it again" in r.getMessage()]
    assert [r.levelno for r in retries] == [logging.DEBUG, logging.DEBUG]
    landed = [r for r in caplog.records if r.levelno >= logging.INFO]
    assert [r.getMessage() for r in landed] == [
        "fake: refused (rate limited); fewer of its creates are sent at once until it takes more",
        "fake-a: created after 2 refusals and 3.0s of waits",
    ]
    await registry.close()


async def test_a_create_still_refused_when_its_waits_would_pass_half_its_deadline_fails(
    tmp_path, caplog
):
    provider = FakeProvider(deadlines=PATIENT, create_errors=[refused() for _ in range(10)])
    backoff = Backoff()
    registry = registry_at(tmp_path, provider, backoff=backoff)
    with pytest.raises(ProvisioningFailure, match="rate limited") as failure:
        await provision(registry)
    assert failure.value.disposition is CreationDisposition.DEFINITELY_NOT_CREATED
    assert backoff.waits == [1, 2, 4]
    assert provider.calls.count(("create", "fake-a")) == 4
    [failed] = [r for r in registry.journal.records if r.event == "create_failed"]
    assert failed.payload["disposition"] == "definitely_not_created"
    assert not registry.unresolved.results
    assert any("refused 4 times" in r.getMessage() for r in caplog.records)
    await registry.close()


async def test_a_create_that_failed_without_being_refused_is_not_sent_again(tmp_path):
    provider = FakeProvider(deadlines=PATIENT, create_errors=[not_created("quota")])
    backoff = Backoff()
    registry = registry_at(tmp_path, provider, backoff=backoff)
    with pytest.raises(ProvisioningFailure, match="quota"):
        await provision(registry)
    assert backoff.waits == [] and provider.calls.count(("create", "fake-a")) == 1
    await registry.close()


async def test_a_stop_during_a_wait_ends_it_at_once_as_never_created(tmp_path):
    provider = FakeProvider(deadlines=PATIENT, create_errors=[refused()])
    backoff = Backoff(hold=asyncio.Event())  # the wait never ends by itself
    registry = registry_at(tmp_path, provider, backoff=backoff)
    creating = asyncio.create_task(provision(registry))
    await settle_loop(rounds=20)
    assert backoff.waits == [1]
    creating.cancel()
    done, _ = await asyncio.wait({creating}, timeout=1)
    backoff.hold.set()  # lets a stop that waited the wait out finish, so the test ends
    assert done, "the stop waited for a wait with nothing in flight"
    with pytest.raises(asyncio.CancelledError):
        await creating
    [failed] = [r for r in registry.journal.records if r.event == "create_failed"]
    assert failed.payload["disposition"] == "definitely_not_created"
    assert provider.calls.count(("create", "fake-a")) == 1
    assert not registry.unresolved.results
    await registry.close()


async def test_a_create_refused_after_its_caller_stopped_is_not_sent_again(tmp_path):
    provider = FakeProvider(deadlines=PATIENT, gate=asyncio.Event())
    backoff = Backoff()
    registry = registry_at(tmp_path, provider, backoff=backoff)
    creating = asyncio.create_task(provision(registry))
    await settle_loop(rounds=20)
    creating.cancel()  # while the create is in flight, so the stop waits for it
    await settle_loop(rounds=20)
    provider.create_errors.append(refused())
    provider.gate.set()
    with pytest.raises(asyncio.CancelledError):
        await creating
    assert backoff.waits == [] and provider.calls.count(("create", "fake-a")) == 1
    assert not registry.unresolved.results
    await registry.close()


async def test_a_wait_holds_no_provisioning_permit(tmp_path):
    provider = FakeProvider(deadlines=PATIENT, create_errors=[refused()])
    backoff = Backoff(hold=asyncio.Event())
    registry = registry_at(tmp_path, provider, limit=1, backoff=backoff)
    waiting = asyncio.create_task(provision(registry, "a"))
    await settle_loop(rounds=20)
    assert backoff.waits == [1]
    await provision(registry, "b")  # the only permit, and window slot, are free while "a" waits
    backoff.hold.set()
    await waiting
    assert sorted(r.name for r in provider.store.values()) == ["fake-a", "fake-b"]
    await registry.close()


async def test_every_try_of_a_refused_create_shares_its_one_deadline(tmp_path):
    # Refused after 0.1 s of a 0.5 s deadline, then a try of 0.45 s: it could finish
    # only if its deadline started again.
    deadlines = dataclasses.replace(FAST, create_seconds=0.5)
    provider = TimedProvider(deadlines=deadlines, durations=[0.1, 0.45], create_errors=[refused()])
    registry = registry_at(tmp_path, provider, backoff=Backoff(scale=0.01))
    with pytest.raises(ProvisioningFailure):
        await provision(registry)
    assert provider.calls.count(("create", "fake-a")) == 2
    assert provider.store == {}
    await registry.close()


async def test_a_retry_is_never_sent_with_less_than_half_its_deadline_left(tmp_path):
    # Refused after 0.2 s of a 1 s deadline; the permit comes back only after 0.85 s.
    deadlines = dataclasses.replace(FAST, create_seconds=1.0)
    provider = TimedProvider(deadlines=deadlines, durations=[0.2, 0.2], create_errors=[refused()])
    slow = FakeProvider(name="slow", deadlines=PATIENT, gate=asyncio.Event())
    backoff = Backoff(hold=asyncio.Event(), scale=0.01)
    registry = registry_at(tmp_path, provider, slow, limit=1, backoff=backoff)
    refused_once = asyncio.create_task(provision(registry, "a"))
    await settle_loop(rounds=20)
    holder = asyncio.create_task(provision(registry, "b", provider="slow"))  # the only permit
    await settle_loop(rounds=20)
    backoff.hold.set()
    await asyncio.sleep(0.85)  # past half of "a"'s deadline while it waits for the permit
    slow.gate.set()
    with pytest.raises(ProvisioningFailure, match="rate limited"):
        await refused_once
    await holder
    assert provider.calls.count(("create", "fake-a")) == 1
    [failed] = [r for r in registry.journal.records if r.event == "create_failed"]
    assert failed.payload["disposition"] == "definitely_not_created"
    await registry.close()


async def test_a_burst_into_a_shared_limit_converges_to_the_room_left(tmp_path):
    # Room for 5 of ours, beside 3 of other clients', each in flight 0.05 s; 50 creates
    # at once, through a window of 16, with a 4 s deadline: 2 s of waits per create.
    deadlines = dataclasses.replace(FAST, create_seconds=4)
    provider = SharedProvider(deadlines=deadlines, capacity=8, background=3, duration=0.05)
    registry = registry_at(tmp_path, provider, limit=16, backoff=Backoff(real=True, scale=0.01))
    results = await burst(registry, [f"k{i}" for i in range(50)])
    assert not [r for r in results if isinstance(r, BaseException)]  # nobody gave up
    assert provider.refusals < 25  # fewer than one for every two creates
    assert 1 <= window(registry) <= 8  # about the room left, not the 16 it started at
    await registry.release_scope(ResourceScope.EPISODE, "attempt-1")
    await registry.close()


async def test_the_window_narrows_as_others_load_rises_and_recovers_once_it_falls(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="loops_cookbook.resources.registry")
    provider = SharedProvider(deadlines=PATIENT, capacity=8)
    registry = registry_at(tmp_path, provider, limit=8, backoff=Backoff(real=True, scale=0.01))
    await burst(registry, [f"quiet-{i}" for i in range(20)])
    assert window(registry) == 8 and provider.refusals == 0
    provider.background = 6
    busy = await burst(registry, [f"busy-{i}" for i in range(20)])
    assert not [r for r in busy if isinstance(r, BaseException)]
    assert window(registry) < 5  # about the room of 2 left
    provider.background = 0
    await burst(registry, [f"calm-{i}" for i in range(60)])
    assert window(registry) == 8
    lines = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert sum("fewer of its creates" in line for line in lines) == 1
    assert sum("back to 8 creates at once" in line for line in lines) == 1
    await registry.release_scope(ResourceScope.EPISODE, "attempt-1")
    await registry.close()


async def test_a_create_waiting_longest_is_sent_first_and_a_retry_keeps_its_place(tmp_path):
    provider = FakeProvider(deadlines=PATIENT, create_errors=[refused()])
    backoff = Backoff(hold=asyncio.Event())
    registry = registry_at(tmp_path, provider, limit=1, backoff=backoff)
    b = asyncio.create_task(provision(registry, "b"))  # refused, and waiting to retry
    await settle_loop(rounds=20)
    provider.gate = asyncio.Event()
    a = asyncio.create_task(provision(registry, "a"))  # in flight: the one slot
    await settle_loop(rounds=20)
    c = asyncio.create_task(provision(registry, "c"))  # in line
    await settle_loop(rounds=20)
    backoff.hold.set()  # "b" is back in line, after "c" but first in it
    await settle_loop(rounds=20)
    provider.gate.set()
    await asyncio.gather(a, b, c)
    creates = [subject for operation, subject in provider.calls if operation == "create"]
    assert creates == ["fake-b", "fake-a", "fake-b", "fake-c"]
    await registry.close()


async def test_one_burst_of_refusals_halves_the_window_once(tmp_path):
    provider = FakeProvider(
        deadlines=PATIENT, gate=asyncio.Event(), create_errors=[refused() for _ in range(8)]
    )
    backoff = Backoff(hold=asyncio.Event())
    registry = registry_at(tmp_path, provider, limit=8, backoff=backoff)
    creating = [asyncio.create_task(provision(registry, f"k{i}")) for i in range(8)]
    await settle_loop(rounds=20)
    assert provider.in_flight == 8
    provider.gate.set()
    await settle_loop(rounds=20)
    assert len(backoff.waits) == 8
    provider.gate = asyncio.Event()
    probes = [asyncio.create_task(provision(registry, f"p{i}")) for i in range(8)]
    await settle_loop(rounds=20)
    assert provider.in_flight == 4
    provider.gate.set()
    await asyncio.gather(*probes)
    for task in creating:
        task.cancel()
    await asyncio.gather(*creating, return_exceptions=True)
    await registry.close()


async def test_a_halved_window_of_three_still_sends_two_creates_at_once(tmp_path):
    provider = FakeProvider(deadlines=PATIENT, create_errors=[refused()])
    backoff = Backoff(hold=asyncio.Event())
    registry = registry_at(tmp_path, provider, limit=3, backoff=backoff)
    waiting = asyncio.create_task(provision(registry, "r"))  # refused, and waiting to retry
    await settle_loop(rounds=20)
    provider.gate = asyncio.Event()
    creating = [asyncio.create_task(provision(registry, key)) for key in "abc"]
    await settle_loop(rounds=20)
    assert provider.in_flight == 2  # 1.5 rounds up, not down to 1
    provider.gate.set()
    await asyncio.gather(*creating)
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    await registry.close()


async def test_answers_that_are_not_refusals_widen_a_narrowed_window_back_to_its_cap(tmp_path):
    answers = [
        ProvisioningFailure("deployed as FAILED", CreationDisposition.DEFINITELY_CREATED),
        ProvisioningFailure("504 gateway timeout", CreationDisposition.AMBIGUOUS),
        TimeoutError("no answer"),
        ProvisioningFailure("image build failed", CreationDisposition.DEFINITELY_CREATED),
        ProvisioningFailure("502 bad gateway", CreationDisposition.AMBIGUOUS),
        ConnectionError("reset"),
    ]
    provider = FakeProvider(deadlines=PATIENT, create_errors=[refused(), *answers])
    backoff = Backoff(hold=asyncio.Event())
    registry = registry_at(tmp_path, provider, limit=4, backoff=backoff)
    waiting = asyncio.create_task(provision(registry, "r"))  # refused, and waiting to retry
    await settle_loop(rounds=20)
    sizes = [window(registry)]
    for i in range(len(answers)):
        [failed] = await asyncio.gather(provision(registry, f"k{i}"), return_exceptions=True)
        assert isinstance(failed, Exception)
        sizes.append(window(registry))
    assert sizes[0] == 2 and sizes == sorted(sizes) and sizes[-1] == 4
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    await registry.close()


async def test_a_stop_while_a_create_waits_for_a_slot_sends_nothing(tmp_path):
    provider = FakeProvider(deadlines=PATIENT, gate=asyncio.Event())
    registry = registry_at(tmp_path, provider, limit=1)
    sent = asyncio.create_task(provision(registry, "a"))  # in flight: the one slot
    await settle_loop(rounds=20)
    waiting = asyncio.create_task(provision(registry, "b"))
    await settle_loop(rounds=20)
    waiting.cancel()
    done, _ = await asyncio.wait({waiting}, timeout=1)
    assert done, "the stop waited for a slot"
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert provider.calls.count(("create", "fake-b")) == 0
    [never] = [r for r in registry.journal.records if r.event == "create_failed"]
    assert never.payload["disposition"] == "definitely_not_created"
    provider.gate.set()
    await sent
    await registry.close()


async def test_a_wait_for_a_slot_before_the_first_send_costs_none_of_the_deadline(tmp_path):
    # "b" waits 0.4 s for the one slot, then takes 0.45 s of its 0.5 s deadline.
    deadlines = dataclasses.replace(FAST, create_seconds=0.5)
    provider = NamedTimedProvider(
        deadlines=deadlines, durations={"fake-a": [0.4], "fake-b": [0.45]}
    )
    registry = registry_at(tmp_path, provider, limit=1)
    first = asyncio.create_task(provision(registry, "a", kind="image_build"))  # 3 s
    await settle_loop(rounds=20)
    handle = await provision(registry, "b")
    await first
    assert handle.resource.name == "fake-b"
    assert provider.calls.count(("create", "fake-b")) == 1
    await registry.close()
