"""The resource registry recovers: a crashed or stopped run's journal is reconciled by identity."""

import asyncio
import time
from datetime import datetime

import pytest
from support.cancellation import settle_loop
from support.resources import FakeProvider, episode, registry_at, request, run_resources

from loops_cookbook.resources.api import (
    CleanupStatus,
    CreationDisposition,
    ProviderResource,
    ProvisioningFailure,
    ReconcileResult,
)
from loops_cookbook.resources.registry import (
    SETTLE_SECONDS,
    ResourceContractError,
    ResourceRegistry,
)

from .rig import QUICK_LOOKUPS, TIGHT, provision

MINUTE_NS = 60 * 1_000_000_000


async def test_a_crash_after_intent_recovers_by_reconciling_the_identity(tmp_path):
    provider = FakeProvider(gate=asyncio.Event())
    registry = registry_at(tmp_path, provider)
    creating = asyncio.create_task(provision(registry))
    await settle_loop(rounds=20)
    creating.cancel()  # the process dies mid-create, after the intent is durable
    await settle_loop(rounds=20)
    registry.journal.close()
    [intent] = registry.journal.records
    # The create reached the provider before the process died.
    labels = intent.payload["ownership_labels"]
    created = ProviderResource(provider_id="x", name=intent.payload["desired_name"], labels=labels)
    survivor = FakeProvider(store={"x": created})
    recovered = registry_at(tmp_path, survivor)
    report = await recovered.release_all()
    assert [(r.resource_id, r.status) for r in report.results] == [
        (intent.resource_id, CleanupStatus.RELEASED)
    ]
    assert survivor.store == {}
    await recovered.close()


@pytest.mark.parametrize(
    ("inherited", "minutes", "status"),
    [
        (True, 11, CleanupStatus.ALREADY_ABSENT),  # settled: the not-found is authoritative
        (True, 5, CleanupStatus.UNCONFIRMED),  # a create may still be becoming visible
        (False, 11, CleanupStatus.UNCONFIRMED),  # this process's own create never settles
    ],
    ids=["inherited_and_settled", "inherited_too_recent", "this_process"],
)
async def test_a_create_a_stopped_run_left_is_absent_only_once_settled(
    tmp_path, inherited, minutes, status
):
    """A lookup that cannot prove absence can once no create for the intent can land."""

    def later():
        return time.time_ns() + minutes * MINUTE_NS

    provider = FakeProvider(gate=asyncio.Event(), hide_absence=True, deadlines=QUICK_LOOKUPS)
    registry = registry_at(tmp_path, provider, clock=None if inherited else later)
    creating = asyncio.create_task(provision(registry))
    await settle_loop(rounds=20)
    creating.cancel()  # the run stops mid-create, after the intent is durable
    await asyncio.gather(creating, return_exceptions=True)
    provider.gate = None
    if inherited:
        registry.journal.close()
        registry = registry_at(tmp_path, provider, clock=later)
    [result] = (await registry.release_all()).results
    assert result.status is status
    await registry.close()


async def test_crash_recovery_is_deterministic_for_every_recorded_state(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    released = await provision(registry, "released")
    await registry.release(released.intent.resource_id)
    await provision(registry, "owned")
    provider.hide_absence = True
    provider.create_errors.append(ConnectionError("reset"))
    with pytest.raises(ProvisioningFailure):
        await provision(registry, "ambiguous")
    unconfirmed = await provision(registry, "unconfirmed")
    provider.release_errors.append(ConnectionError("reset"))
    await registry.release(unconfirmed.intent.resource_id)
    registry.journal.close()  # crash: nothing else runs

    first = registry_at(tmp_path, provider, mode="reconcile")
    second_view = sorted((r.resource_id, r.status) for r in first.unresolved.results)
    first.journal.close()
    again = registry_at(tmp_path, provider, mode="reconcile")
    assert sorted((r.resource_id, r.status) for r in again.unresolved.results) == second_view
    assert len(second_view) == 3  # owned, ambiguous, unconfirmed; released is closed
    provider.hide_absence = False
    report = await again.release_all()
    assert report.complete
    assert provider.store == {}
    await again.close()


async def test_reconcile_mode_does_not_provision(tmp_path):
    registry = registry_at(tmp_path, FakeProvider(), mode="reconcile")
    with pytest.raises(ResourceContractError, match="does not provision"):
        await provision(registry)
    await registry.close()


async def test_a_damaged_journal_refuses_write_mode(tmp_path):
    registry = registry_at(tmp_path, FakeProvider())
    await provision(registry)
    path = registry.journal.path
    await registry.close()
    path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(ResourceContractError, match="loops cleanup"):
        registry_at(tmp_path, FakeProvider())
    reconciling = registry_at(tmp_path, FakeProvider(), mode="reconcile")
    assert isinstance(reconciling, ResourceRegistry)
    await reconciling.close()


async def test_an_ownership_check_that_fails_proves_nothing_and_stops_nothing(tmp_path):
    provider = FakeProvider(ambiguous_after_create=True)

    async def broken(intent, resource):
        raise RuntimeError("ownership lookup failed")

    provider.validate_ownership = broken
    registry = registry_at(tmp_path, provider)
    service = episode(registry)
    with pytest.raises(ProvisioningFailure) as failure:
        await service.provision(request("a"))
    assert failure.value.disposition is CreationDisposition.AMBIGUOUS
    await service.provision(request("b"))
    report = await service.close()
    statuses = sorted(result.status.value for result in report.results)
    assert statuses == ["released", "unconfirmed"]  # b released; a kept, not aborted
    await registry.close()


async def test_an_unconfigured_provider_is_kept_unconfirmed_while_the_rest_is_released(tmp_path):
    first, second = FakeProvider(), FakeProvider(name="other")
    registry = registry_at(tmp_path, first, second)
    service = episode(registry)
    await service.provision(request("a"))
    await service.provision(request("b", provider="other"))
    registry.journal.close()
    recovered = registry_at(tmp_path, first, mode="reconcile")
    report = await recovered.release_all()
    by_status = {result.status for result in report.results}
    assert by_status == {CleanupStatus.RELEASED, CleanupStatus.UNCONFIRMED}
    [kept] = report.unresolved
    assert "providers.other" in (kept.detail or "")
    assert first.store == {} and list(second.store)
    await recovered.close()


@pytest.mark.parametrize("event", ["mystery", "created"])
async def test_a_journal_this_version_cannot_replay_fails_to_open_and_keeps_no_lock(
    tmp_path, event
):
    from loops_cookbook.resources.journal import Journal, JournalError

    journal = Journal.open(tmp_path, kind=lambda path: "ext4")
    journal.append(event, "nobody", {"resource": {}})
    journal.close()
    with pytest.raises(JournalError):
        registry_at(tmp_path, FakeProvider(), mode="reconcile")
    Journal.open(tmp_path, kind=lambda path: "ext4").close()  # the lock was released


@pytest.mark.parametrize(
    ("blocked", "detail"),
    [
        ("reconcile", "(TimeoutError: no answer within 0.05s)"),
        ("release", "release failed: TimeoutError: no answer within 0.05s"),
        ("confirm_absent", "absence unknown: TimeoutError: no answer within 0.05s"),
    ],
)
async def test_an_operation_out_of_time_says_how_long_it_waited(tmp_path, blocked, detail):
    provider = FakeProvider(deadlines=TIGHT)
    registry = registry_at(tmp_path, provider)
    if blocked == "reconcile":  # the create's outcome is unknown, so recovery looks it up
        provider.gate = asyncio.Event()
        creating = asyncio.create_task(provision(registry))
        await settle_loop(rounds=20)
        creating.cancel()
        await asyncio.gather(creating, return_exceptions=True)
        provider.gate = None
    else:
        await provision(registry)
    registry.journal.close()
    provider.blocked.add(blocked)
    recovered = registry_at(tmp_path, provider)
    report = await recovered.release_all()
    [result] = report.results
    assert result.status is CleanupStatus.UNCONFIRMED
    assert result.detail == detail if blocked != "reconcile" else result.detail.endswith(detail)
    await recovered.close()


async def test_recovery_ends_the_leases_a_crashed_process_recorded(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    shared = await run_resources(registry).provision(request("cache"))
    await episode(registry).acquire(shared.intent.resource_id)
    registry.journal.close()  # the process dies holding the lease
    recovered = registry_at(tmp_path, provider, mode="reconcile")
    assert (await recovered.release_all()).complete and provider.store == {}
    await recovered.close()


async def test_a_reconciled_resource_that_is_unusable_is_owned_released_and_fails(tmp_path):
    provider = FakeProvider(ambiguous_after_create=True)
    lookup = provider.reconcile_create

    async def unusable(intent, *, settled=False):
        found = await lookup(intent, settled=settled)
        return ReconcileResult(found.status, found.resource, unusable="it deployed dead")

    provider.reconcile_create = unusable
    registry = registry_at(tmp_path, provider)
    service = episode(registry)
    with pytest.raises(ProvisioningFailure, match="deployed dead") as failure:
        await service.provision(request())
    assert failure.value.disposition is CreationDisposition.DEFINITELY_CREATED
    assert (await service.close()).complete and provider.store == {}
    await registry.close()


async def test_an_unsettled_create_says_when_cleanup_can_prove_it_absent(tmp_path):
    provider = FakeProvider(gate=asyncio.Event(), deadlines=TIGHT)
    registry = registry_at(tmp_path, provider)
    creating = asyncio.create_task(provision(registry))
    await settle_loop(rounds=20)
    registry.journal.close()  # the process dies mid-create, after its intent is durable
    [intent] = registry.journal.records
    provider.blocked.add("reconcile")
    recovered = registry_at(tmp_path, provider)
    [result] = (await recovered.release_all()).results
    settles = datetime.fromtimestamp(intent.recorded_at_ns / 1e9 + SETTLE_SECONDS).astimezone()
    assert result.status is CleanupStatus.UNCONFIRMED
    assert f"until the create settles at {settles:%Y-%m-%d %H:%M:%S}" in result.detail
    assert "run `loops cleanup` again after that" in result.detail
    await recovered.close()
    provider.gate.set()
    creating.cancel()
    await asyncio.gather(creating, return_exceptions=True)
