"""The resource registry closes: within its bound, recording what it stopped, and
closing every adapter once."""

import asyncio
import dataclasses
import gc
import time

import pytest
from support.cancellation import (
    ALL_LANDINGS,
    Landing,
    Response,
    Trigger,
    cancellation_matrix,
    settle_loop,
)
from support.resources import (
    FAST,
    FakeProvider,
    FakeSandboxProvider,
    episode,
    not_created,
    refused,
    registry_at,
    request,
    run_resources,
)

from loops_cookbook.resources.api import (
    CleanupStatus,
    CreationDisposition,
    ProvisioningFailure,
    ResourceScope,
)
from loops_cookbook.resources.registry import ResourceContractError

from .rig import provision

# A builder create may take minutes, as an image build does: far past any close bound.
SLOW_BUILDS = dataclasses.replace(FAST, create_seconds_by_kind={"builder": 600})


@dataclasses.dataclass
class HeldBuilds(FakeProvider):
    """A ``FakeProvider`` whose builder creates wait for ``builds`` to be set."""

    builds: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)

    async def create(self, intent):
        if intent.request.kind == "builder":
            self.calls.append(("build", intent.desired_name))
            await self.builds.wait()
        return await super().create(intent)


def outcomes(registry):
    """Each intent's last record: an intent with nothing after it was left pending."""
    last = {}
    for record in registry.journal.records:
        last[record.resource_id] = record
    return last


async def _in_prerequisites(registry, provider):
    provider.blocked.add("prerequisites")
    return [asyncio.create_task(provision(registry, "a"))]


async def _in_create(registry, provider):
    provider.blocked.add("create")
    return [asyncio.create_task(provision(registry, "a"))]


async def _in_reconcile(registry, provider):
    provider.ambiguous_after_create = True  # the create lands, then its answer is lost
    provider.blocked.add("reconcile")
    return [asyncio.create_task(provision(registry, "a"))]


async def _in_validate(registry, provider):
    provider.ambiguous_after_create = True  # found by the reconcile, then validated
    provider.blocked.add("validate")
    return [asyncio.create_task(provision(registry, "a"))]


async def _releasing(registry, provider, operation):
    handle = await provision(registry, "a")
    provider.blocked.add(operation)
    return [asyncio.create_task(registry.release(handle.intent.resource_id))]


async def _in_release(registry, provider):
    return await _releasing(registry, provider, "release")


async def _in_confirm_absent(registry, provider):
    return await _releasing(registry, provider, "confirm_absent")


async def _in_proof_of_absence(registry, provider):
    old = await provision(registry, "a")
    provider.stubborn.add("fake-a")
    await registry.release(old.intent.resource_id)  # not proven gone
    provider.stubborn.clear()
    provider.blocked.add("release")  # the key's next provision must prove it gone first
    return [asyncio.create_task(provision(registry, "a"))]


async def _in_client(registry, provider):
    handle = await provision(registry, "a")
    provider.blocked.add("client")
    return [asyncio.create_task(registry.sandbox_client(handle.intent.resource_id))]


async def _in_alive(registry, provider):
    handle = await provision(registry, "a")
    provider.blocked.add("alive")
    return [asyncio.create_task(registry.sandbox_alive(handle.intent.resource_id))]


async def _in_close_client(registry, provider):
    resource_id = (await provision(registry, "a")).intent.resource_id
    client = await registry.sandbox_client(resource_id)
    provider.blocked.add("close_client")
    return [asyncio.create_task(registry.close_sandbox_client(resource_id, client))]


IN_FLIGHT = [
    _in_prerequisites,
    _in_create,
    _in_reconcile,
    _in_validate,
    _in_release,
    _in_confirm_absent,
    _in_proof_of_absence,
    _in_client,
    _in_alive,
    _in_close_client,
]


def _assert_closed_cleanly(tmp_path, registry, provider):
    """No adapter call once closed, the adapter closed once, the journal unlocked, and
    every intent ended with an outcome."""
    assert provider.used_after_close == [] and provider.closed_under == []
    assert provider.closed == 1
    assert not journal_is_locked(tmp_path)
    pending = [rid for rid, records in _by_resource(registry).items() if len(records) == 1]
    assert pending == []


def journal_is_locked(state_dir):
    from loops_cookbook.resources.journal import Journal, JournalLocked

    try:
        Journal.open(state_dir, kind=lambda path: "ext4").close()
    except JournalLocked:
        return True
    return False


def _by_resource(registry):
    grouped = {}
    for record in registry.journal.records:
        grouped.setdefault(record.resource_id, []).append(record)
    return grouped


async def test_provider_clients_close_exactly_once(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    await registry.close()
    await registry.close()
    assert provider.closed == 1


async def test_a_closed_registry_refuses_further_work_and_writes_nothing(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    service = episode(registry)
    handle = await service.provision(request())
    await registry.close()
    records = registry.journal.path.read_bytes()
    for late in (
        service.release(handle.intent.resource_id),
        service.close(),
        registry.release_all(),
    ):
        with pytest.raises(ResourceContractError, match="closed"):
            await late
    with pytest.raises(ResourceContractError, match="closed"):
        registry.lease(handle.intent.resource_id, "x")
    assert registry.journal.path.read_bytes() == records
    assert list(provider.store)  # kept for recovery, not released behind the lock


async def test_every_adapter_closes_even_when_one_fails(tmp_path):
    broken, second = FakeProvider(name="broken"), FakeProvider(name="second")

    async def fails():
        raise RuntimeError("close failed")

    broken.close = fails
    registry = registry_at(tmp_path, broken, second)
    with pytest.raises(RuntimeError, match="close failed"):
        await registry.close()
    assert second.closed == 1
    registry_at(tmp_path, FakeProvider()).journal.close()  # the journal was unlocked


async def test_an_adapter_that_never_closes_holds_up_neither_the_close_nor_the_others(
    tmp_path, monkeypatch
):
    from loops_cookbook.resources import registry as registry_module

    monkeypatch.setattr(registry_module, "CUT_OFF_GRACE_SECONDS", 0.1)
    hung, second = FakeProvider(name="hung"), FakeProvider(name="second")
    gate = asyncio.Event()

    async def never_closes():
        try:
            await gate.wait()
        except asyncio.CancelledError:
            await gate.wait()

    hung.close = never_closes
    registry = registry_at(tmp_path, hung, second)
    started = time.monotonic()
    await asyncio.wait_for(registry.close(timeout=0.1), timeout=2)
    assert time.monotonic() - started < 1
    assert second.closed == 1 and not journal_is_locked(tmp_path)
    gate.set()
    await settle_loop(rounds=20)


async def test_an_adapter_close_that_is_cancelled_is_not_reported_as_closed():
    from loops_cookbook.resources.registry import close_adapters

    cancelled, second = FakeProvider(name="cancelled"), FakeProvider(name="second")

    async def cancels():
        raise asyncio.CancelledError

    cancelled.close = cancels
    with pytest.raises(asyncio.CancelledError):
        await close_adapters({"cancelled": cancelled, "second": second})
    assert second.closed == 1  # the others still close


@cancellation_matrix(
    triggers=(Trigger.CANCEL,),
    points={"create": ALL_LANDINGS},
    responses=(Response.STOPS,),
)
async def test_a_stop_mid_create_lets_the_create_land_then_the_resource_is_released(tmp_path, case):
    """A create cut off mid-request could never be proven absent; so it is not cut off,
    by a second cancellation either."""
    provider = FakeProvider(gate=asyncio.Event())
    registry = registry_at(tmp_path, provider)
    creating = asyncio.create_task(provision(registry))
    await settle_loop(rounds=20)
    assert provider.calls == [("create", "fake-a")]  # sent, and not yet answered
    creating.cancel()
    await settle_loop(rounds=20)
    if case.at("create") is Landing.CANCEL:
        creating.cancel()
        await settle_loop(rounds=20)
    assert not creating.done()  # the stop waits for the create it cannot take back
    if case.at("create") is Landing.FAILURE:
        provider.create_errors.append(ConnectionError("reset"))
    provider.gate.set()
    with pytest.raises(asyncio.CancelledError):
        await creating
    report = await registry.release_scope(ResourceScope.EPISODE, "attempt-1")
    created = case.at("create") is not Landing.FAILURE  # a failed create is proven absent
    assert ("created" in [r.event for r in registry.journal.records]) is created
    assert [r.status for r in report.results] == ([CleanupStatus.RELEASED] if created else [])
    assert provider.store == {} and not registry.unresolved.results
    await registry.close()


async def test_a_stop_before_the_create_is_sent_records_that_nothing_was_created(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider, limit=1)
    held = asyncio.Event()
    provider.gate = held
    first = asyncio.create_task(provision(registry, "a"))
    await settle_loop(rounds=20)
    queued = asyncio.create_task(provision(registry, "b"))  # waits for the only permit
    await settle_loop(rounds=20)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    [never] = [r for r in registry.journal.records if r.event == "create_failed"]
    assert never.payload["disposition"] == "definitely_not_created"
    assert ("create", "fake-b") not in provider.calls
    held.set()
    await first
    unresolved = {r.resource_id for r in registry.unresolved.results}
    assert never.resource_id not in unresolved  # b needs no cleanup; only a is live
    await registry.release_scope(ResourceScope.EPISODE, "attempt-1")
    await registry.close()


async def test_a_close_with_builds_queued_and_in_flight_ends_within_its_bound(tmp_path):
    """A build in flight is cut off at the bound and kept for cleanup; callers queued
    on it, and a create still waiting to be sent, are stopped as never created."""
    builder = request("builder-1", kind="builder", release_after_create=True)
    provider = HeldBuilds(deadlines=SLOW_BUILDS, needs={"sandbox": [builder]})
    registry = registry_at(tmp_path, provider, limit=1)
    sandboxes = [asyncio.create_task(provision(registry, key, kind="sandbox")) for key in "abc"]
    await settle_loop(rounds=20)
    unsent = asyncio.create_task(provision(registry, "x"))  # waits for the only slot
    await settle_loop(rounds=20)
    assert provider.calls == [("build", "fake-builder-1")]  # sent, and never answered
    started = time.monotonic()
    registry.begin_shutdown()
    await registry.close(timeout=0.2)
    assert time.monotonic() - started < 2
    failures = await asyncio.gather(*sandboxes, unsent, return_exceptions=True)
    assert all(isinstance(failure, ProvisioningFailure) for failure in failures), failures
    assert [f.disposition for f in failures].count(CreationDisposition.DEFINITELY_NOT_CREATED) == 3
    # One build was ever started, and nothing was sent after the close began.
    assert provider.calls == [("build", "fake-builder-1")]
    last = outcomes(registry)
    assert [r.event for r in last.values()] == ["create_failed", "create_failed"]
    by_name = {r.resource_id: r.payload["disposition"] for r in last.values()}
    [built] = [r for r in registry.unresolved.results]
    assert by_name.pop(built.resource_id) == "ambiguous"  # kept for `loops cleanup`
    assert list(by_name.values()) == ["definitely_not_created"]


async def test_a_close_records_a_create_stopped_before_it_was_sent(tmp_path):
    provider = FakeProvider(gate=asyncio.Event())
    registry = registry_at(tmp_path, provider, limit=1)
    first = asyncio.create_task(provision(registry, "a"))
    await settle_loop(rounds=20)
    queued = asyncio.create_task(provision(registry, "b"))
    await settle_loop(rounds=20)
    closing = asyncio.create_task(registry.close())
    await settle_loop(rounds=20)
    assert not closing.done()  # waits for the create already sent
    provider.gate.set()
    await closing
    handle = await first
    with pytest.raises(ProvisioningFailure) as stopped:
        await queued
    assert stopped.value.disposition is CreationDisposition.DEFINITELY_NOT_CREATED
    assert ("create", "fake-b") not in provider.calls
    events_by_id = {rid: [r.event for r in rs] for rid, rs in _by_resource(registry).items()}
    assert events_by_id.pop(handle.intent.resource_id) == ["intent", "created"]
    assert list(events_by_id.values()) == [["intent", "create_failed"]]


async def test_a_refused_create_is_not_retried_after_shutdown_begins(tmp_path):
    provider = FakeProvider(gate=asyncio.Event(), create_errors=[refused()])
    registry = registry_at(tmp_path, provider)
    creating = asyncio.create_task(provision(registry))
    await settle_loop(rounds=20)
    assert provider.calls == [("create", "fake-a")]

    registry.begin_shutdown()
    provider.gate.set()

    with pytest.raises(ProvisioningFailure, match="rate limited"):
        await creating
    assert provider.calls == [("create", "fake-a")]
    await registry.close()


async def test_close_keeps_journal_and_adapters_open_for_a_release_that_ignores_cancellation(
    tmp_path, monkeypatch
):
    """Past the grace, close returns; the journal and the adapter stay open for the
    release still running, and close once it stops."""
    from loops_cookbook.resources import registry as registry_module

    monkeypatch.setattr(registry_module, "CUT_OFF_GRACE_SECONDS", 0.01)
    provider = HeldBuilds(deadlines=SLOW_BUILDS)
    provider.builds.set()
    release_started, release_gate = asyncio.Event(), asyncio.Event()
    release = provider.release

    async def ignore_release_cancellation(resource):
        release_started.set()
        try:
            await release_gate.wait()
        except asyncio.CancelledError:
            await release_gate.wait()
        await release(resource)

    provider.release = ignore_release_cancellation
    registry = registry_at(tmp_path, provider)
    creating = asyncio.create_task(provision(registry, kind="builder", release_after_create=True))
    await asyncio.wait_for(release_started.wait(), timeout=1)

    await registry.close(timeout=0)

    with pytest.raises(ProvisioningFailure):
        await creating
    assert journal_is_locked(tmp_path) and provider.closed == 0
    assert not any(record.event == "released" for record in registry.journal.records)

    release_gate.set()
    await settle_loop(rounds=20)
    assert any(record.event == "released" for record in registry.journal.records)
    assert not journal_is_locked(tmp_path) and provider.closed == 1


async def test_a_caller_whose_prerequisite_is_released_at_shutdown_is_not_provisioned(tmp_path):
    """The shared prerequisite lands after shutdown began and the run scope is released:
    its waiter fails as never provisioned, not as a contract violation."""
    shared = request("shared", kind="base")
    provider = FakeProvider(gate=asyncio.Event(), needs={"sandbox": [shared]})
    registry = registry_at(tmp_path, provider)
    waiting = asyncio.create_task(provision(registry, "a", kind="sandbox"))
    await settle_loop(rounds=20)
    registry.begin_shutdown()
    releasing = asyncio.create_task(run_resources(registry).close())
    await settle_loop(rounds=20)
    provider.gate.set()
    with pytest.raises(ProvisioningFailure) as stopped:
        await waiting
    assert stopped.value.disposition is CreationDisposition.DEFINITELY_NOT_CREATED
    await releasing
    await registry.close()


async def test_a_cancelled_close_records_a_create_stopped_before_it_started(tmp_path):
    slow, fast = FakeProvider(name="slow", gate=asyncio.Event()), FakeProvider()
    registry = registry_at(tmp_path, slow, fast)
    held = asyncio.create_task(provision(registry, "held", provider="slow"))
    await settle_loop(rounds=20)  # sent and held, so the close waits for it
    unstarted = asyncio.create_task(provision(registry, "a"))
    while len(registry.journal.records) < 2:  # held's intent, then a's, and no further
        await asyncio.sleep(0)
    registry.begin_shutdown()
    closing = asyncio.create_task(registry.close(timeout=60))
    await settle_loop(rounds=20)
    closing.cancel()  # a second interrupt
    await asyncio.gather(closing, return_exceptions=True)
    with pytest.raises(ProvisioningFailure, match="registry is closing") as stopped:
        await unstarted
    assert stopped.value.disposition is CreationDisposition.DEFINITELY_NOT_CREATED
    slow.gate.set()
    await held
    await settle_loop(rounds=20)
    events = {rid: [r.event for r in rs] for rid, rs in _by_resource(registry).items()}
    assert sorted(events.values()) == [["intent", "create_failed"], ["intent", "created"]]
    assert ("create", "fake-a") not in fast.calls
    assert not journal_is_locked(tmp_path)


async def test_a_close_waits_for_the_release_a_failed_create_cascades(tmp_path):
    """A dependant whose create fails frees the dependency it alone held; the close
    under way waits for that release and journals it before the journal closes."""
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    base = (await provision(registry, "base")).intent.resource_id
    provider.gate, provider.create_errors = asyncio.Event(), [not_created("oom")]
    dependant = asyncio.create_task(provision(registry, "a", depends_on=(base,)))
    await settle_loop(rounds=20)
    assert (await registry.release(base)).status is CleanupStatus.DEFERRED  # a holds it
    release_gate, release = asyncio.Event(), provider.release

    async def held_release(resource):
        await release_gate.wait()
        await release(resource)

    provider.release = held_release
    closing = asyncio.create_task(registry.close())
    await settle_loop(rounds=20)
    provider.gate.set()  # the dependant fails, and its dependency is released
    await settle_loop(rounds=20)
    assert not closing.done()  # waiting for that release
    release_gate.set()
    await closing
    with pytest.raises(ProvisioningFailure):
        await dependant
    events = [r.event for r in _by_resource(registry)[base]]
    assert events == ["intent", "created", "release_deferred", "released"]
    assert not journal_is_locked(tmp_path)


# Every provider operation, in flight when the registry closes. Each starts the
# operation with ``blocked`` naming it, and returns the tasks waiting on it.


@pytest.mark.parametrize("start", IN_FLIGHT, ids=lambda start: start.__name__[1:])
async def test_a_close_past_its_deadline_cuts_off_every_operation(tmp_path, monkeypatch, start):
    from loops_cookbook.resources import registry as registry_module

    monkeypatch.setattr(registry_module, "CUT_OFF_GRACE_SECONDS", 0.2)
    provider = FakeSandboxProvider()
    registry = registry_at(tmp_path, provider)
    waiting = await start(registry, provider)
    await settle_loop(rounds=20)
    started = time.monotonic()
    await registry.close(timeout=0.1)
    assert time.monotonic() - started < 1
    outcomes = await asyncio.gather(*waiting, return_exceptions=True)
    # Their callers were never cancelled, so none is told it was.
    assert not [o for o in outcomes if isinstance(o, asyncio.CancelledError)], outcomes
    await settle_loop(rounds=20)
    _assert_closed_cleanly(tmp_path, registry, provider)


@pytest.mark.parametrize("start", IN_FLIGHT, ids=lambda start: start.__name__[1:])
async def test_a_cancelled_close_finishes_once_every_operation_ends(tmp_path, start):
    """A second interrupt cancels the close: the closing goes on, and closes the journal
    and the adapter only once the operation in flight has ended."""
    provider = FakeSandboxProvider()
    registry = registry_at(tmp_path, provider)
    waiting = await start(registry, provider)
    await settle_loop(rounds=20)
    closing = asyncio.create_task(registry.close(timeout=60))
    await settle_loop(rounds=20)
    closing.cancel()
    await asyncio.gather(closing, return_exceptions=True)
    if provider.busy:  # still under way: the journal and the adapter stay open for it
        assert journal_is_locked(tmp_path) and provider.closed == 0
    provider.unblocked.set()
    await asyncio.gather(*waiting, return_exceptions=True)
    await settle_loop(rounds=20)
    _assert_closed_cleanly(tmp_path, registry, provider)


async def test_sandbox_usage_once_the_registry_is_closed_is_refused(tmp_path):
    """Usage after close would reach a closed adapter: it is refused, as a release is."""
    provider = FakeSandboxProvider()
    registry = registry_at(tmp_path, provider)
    resource_id = (await provision(registry, "a")).intent.resource_id
    await registry.close()
    with pytest.raises(ResourceContractError, match="closed"):
        await registry.sandbox_client(resource_id)
    with pytest.raises(ResourceContractError, match="closed"):
        await registry.sandbox_alive(resource_id)
    assert provider.used_after_close == []


async def test_the_adapters_close_even_when_recording_the_unsent_creates_fails(tmp_path):
    slow, fast = FakeProvider(name="slow", gate=asyncio.Event()), FakeProvider()
    registry = registry_at(tmp_path, slow, fast)
    held = asyncio.create_task(provision(registry, "held", provider="slow"))
    await settle_loop(rounds=20)
    unstarted = asyncio.create_task(provision(registry, "a"))
    while len(registry.journal.records) < 2:  # held's intent, then a's, and no further
        await asyncio.sleep(0)
    registry.begin_shutdown()  # a's create stops before it starts: the sweep records it
    slow.gate.set()
    await asyncio.gather(held, unstarted, return_exceptions=True)
    append = registry.journal.append

    def failing(event, resource_id, payload):
        if event == "create_failed":
            raise OSError("fsync failed")
        return append(event, resource_id, payload)

    registry.journal.append = failing
    with pytest.raises(OSError, match="fsync failed"):
        await registry.close()
    assert slow.closed == fast.closed == 1
    assert not journal_is_locked(tmp_path)


async def test_a_release_its_caller_abandoned_has_its_failure_seen(tmp_path):
    """Proving an old identity gone runs on when its caller is cancelled; when it then
    fails, the failure is retrieved, not left for asyncio to report at exit."""
    unseen = []
    asyncio.get_running_loop().set_exception_handler(
        lambda loop, context: unseen.append(context["message"])
    )
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    first = await provision(registry, "a")
    provider.stubborn.add("fake-a")
    await registry.release(first.intent.resource_id)  # not proven gone
    provider.gate = asyncio.Event()
    again = asyncio.create_task(provision(registry, "a"))  # proves the old one gone first
    await settle_loop(rounds=20)
    again.cancel()
    await asyncio.gather(again, return_exceptions=True)
    provider.gate.set()
    await settle_loop(rounds=20)
    await registry.close()
    gc.collect()
    await settle_loop(rounds=20)
    assert unseen == []


async def test_a_provision_on_a_closed_registry_fails_as_never_created(tmp_path):
    """As on a closing one, such as a caller's next prerequisite once the close is done."""
    registry = registry_at(tmp_path, FakeProvider())
    await registry.close()
    with pytest.raises(ProvisioningFailure) as failure:
        await provision(registry, "a")
    assert failure.value.disposition is CreationDisposition.DEFINITELY_NOT_CREATED


async def test_a_build_that_fails_while_closing_is_not_started_again_for_its_waiters(tmp_path):
    builder = request("builder-1", kind="builder", release_after_create=True)
    provider = HeldBuilds(
        deadlines=SLOW_BUILDS, needs={"sandbox": [builder]}, create_errors=[not_created("oom")]
    )
    registry = registry_at(tmp_path, provider)
    sandboxes = [asyncio.create_task(provision(registry, key, kind="sandbox")) for key in "abc"]
    await settle_loop(rounds=20)
    registry.begin_shutdown()
    provider.builds.set()
    failures = await asyncio.gather(*sandboxes, return_exceptions=True)
    assert all(isinstance(failure, ProvisioningFailure) for failure in failures), failures
    assert provider.calls.count(("build", "fake-builder-1")) == 1
    await registry.close()
    assert registry.unresolved.results == ()


async def test_a_helper_that_lands_while_the_registry_closes_is_released(tmp_path):
    builder = request("builder-1", kind="builder", release_after_create=True)
    provider = HeldBuilds(deadlines=SLOW_BUILDS, needs={"sandbox": [builder]})
    registry = registry_at(tmp_path, provider)
    sandbox = asyncio.create_task(provision(registry, "a", kind="sandbox"))
    await settle_loop(rounds=20)
    closing = asyncio.create_task(registry.close(timeout=5))
    await settle_loop(rounds=20)
    assert not closing.done()
    provider.builds.set()  # the build lands after the close began
    await closing
    with pytest.raises(ProvisioningFailure, match="closing"):
        await sandbox  # the sandbox itself is never created
    [helper] = _by_resource(registry).values()
    assert [r.event for r in helper] == ["intent", "created", "released"]
    assert provider.store == {} and registry.unresolved.results == ()
