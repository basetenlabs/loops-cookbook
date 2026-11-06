"""The resource registry provisions: concurrency, prerequisites, and what a create's
outcome proves."""

import asyncio

import pytest
from support.cancellation import settle_loop
from support.resources import (
    Backoff,
    FakeProvider,
    episode,
    not_created,
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
from loops_cookbook.resources.registry import FAILED_HELPER_COOLDOWN_SECONDS, ResourceContractError

from .rig import QUICK_LOOKUPS, TIGHT, provision


def image_provider(**options):
    builder = request("builder-1", kind="builder", release_after_create=True)
    return FakeProvider(needs={"sandbox": [builder]}, **options)


async def test_two_slow_independent_creates_overlap(tmp_path):
    provider = FakeProvider(gate=asyncio.Event())
    registry = registry_at(tmp_path, provider)
    both = asyncio.gather(provision(registry, "a"), provision(registry, "b"))
    await settle_loop(rounds=20)
    assert provider.in_flight == 2
    provider.gate.set()
    first, second = await both
    assert first.intent.resource_id != second.intent.resource_id
    await registry.close()


async def test_operations_for_one_resource_serialize(tmp_path):
    provider = FakeProvider(gate=asyncio.Event())
    registry = registry_at(tmp_path, provider)
    both = asyncio.gather(provision(registry, "a"), provision(registry, "a"))
    await settle_loop(rounds=20)
    assert provider.in_flight == 1
    provider.gate.set()
    first, second = await both
    assert first == second
    assert [call for call in provider.calls if call[0] == "create"] == [("create", "fake-a")]
    await registry.close()


async def test_a_limit_of_two_admits_exactly_two_of_three_blocking_operations(tmp_path):
    provider = FakeProvider(gate=asyncio.Event())
    registry = registry_at(tmp_path, provider, limit=2)
    three = asyncio.gather(*(provision(registry, key) for key in "abc"))
    await settle_loop(rounds=20)
    assert provider.in_flight == 2
    provider.gate.set()
    await three
    assert provider.most_in_flight == 2
    assert len(provider.store) == 3
    await registry.close()


async def test_a_shared_prerequisite_is_provisioned_once_and_waiters_hold_no_permit(tmp_path):
    provider = image_provider(gate=asyncio.Event())
    registry = registry_at(tmp_path, provider, limit=1)
    waiters = asyncio.gather(*(provision(registry, key, kind="sandbox") for key in "abc"))
    await settle_loop(rounds=20)
    # One builder create holds the only permit; the other two wait on its key.
    assert provider.calls == [("create", "fake-builder-1")]
    provider.gate.set()
    handles = await waiters
    assert len(handles) == 3
    creates = [subject for operation, subject in provider.calls if operation == "create"]
    assert creates.count("fake-builder-1") == 1
    assert sorted(creates) == ["fake-a", "fake-b", "fake-builder-1", "fake-c"]
    await registry.close()


async def test_a_release_after_create_prerequisite_is_released_at_once_and_never_a_dependency(
    tmp_path,
):
    provider = image_provider()
    registry = registry_at(tmp_path, provider)
    handle = await provision(registry, "a", kind="sandbox")
    assert provider.calls[:3] == [
        ("create", "fake-builder-1"),
        ("release", "fake-builder-1"),
        ("confirm_absent", "fake-builder-1"),
    ]
    assert handle.intent.request.depends_on == ()
    assert [resource.name for resource in provider.store.values()] == ["fake-a"]
    await provision(registry, "b", kind="sandbox")
    creates = [subject for operation, subject in provider.calls if operation == "create"]
    assert creates.count("fake-builder-1") == 1  # reused, not built again
    await registry.close()


async def test_a_failed_release_after_create_prerequisite_is_retried_by_the_next_request(
    tmp_path,
):
    provider = image_provider(create_errors=[not_created("quota")])
    backoff = Backoff()
    registry = registry_at(tmp_path, provider, backoff=backoff)
    with pytest.raises(ProvisioningFailure, match="quota"):
        await provision(registry, "a", kind="sandbox")
    backoff.time += FAILED_HELPER_COOLDOWN_SECONDS
    await provision(registry, "a", kind="sandbox")
    creates = [subject for operation, subject in provider.calls if operation == "create"]
    assert creates == ["fake-builder-1", "fake-builder-1", "fake-a"]
    await registry.close()


async def test_requests_queued_behind_a_failed_helper_fail_at_once_until_its_cooldown_passes(
    tmp_path,
):
    provider = image_provider(gate=asyncio.Event(), create_errors=[not_created("bad image")])
    backoff = Backoff()
    registry = registry_at(tmp_path, provider, backoff=backoff)
    waiters = asyncio.gather(
        *(provision(registry, key, kind="sandbox") for key in "abc"), return_exceptions=True
    )
    await settle_loop(rounds=20)
    provider.gate.set()
    results = await waiters
    assert all(isinstance(r, ProvisioningFailure) and "bad image" in str(r) for r in results)
    builds = [subject for operation, subject in provider.calls if operation == "create"]
    assert builds == ["fake-builder-1"]  # built once, not once for each request
    backoff.time += FAILED_HELPER_COOLDOWN_SECONDS - 1
    with pytest.raises(ProvisioningFailure, match="not trying again for 1s: .*bad image"):
        await provision(registry, "d", kind="sandbox")
    backoff.time += 1
    await provision(registry, "d", kind="sandbox")
    creates = [subject for operation, subject in provider.calls if operation == "create"]
    assert creates == ["fake-builder-1", "fake-builder-1", "fake-d"]
    await registry.close()


async def test_an_expired_helper_failure_is_forgotten_on_any_helpers_request(tmp_path):
    """So a long run's failed helpers do not pile up: a later request for another
    helper drops every failure past its cooldown."""
    provider = image_provider(create_errors=[not_created("bad image")])
    backoff = Backoff()
    registry = registry_at(tmp_path, provider, backoff=backoff)
    with pytest.raises(ProvisioningFailure, match="bad image"):
        await provision(registry, "a", kind="sandbox")
    backoff.time += FAILED_HELPER_COOLDOWN_SECONDS
    await registry.provision(
        request("another", kind="builder", release_after_create=True),
        scope=ResourceScope.RUN,
        owner_id="run-1",
    )
    assert not registry._failed_helpers
    await registry.close()


async def test_an_ambiguous_create_is_reconciled_by_identity(tmp_path):
    provider = FakeProvider(ambiguous_after_create=True)
    registry = registry_at(tmp_path, provider)
    handle = await provision(registry)
    assert handle.resource.name == "fake-a"
    registry.journal.close()  # a crash now: the journal alone knows the sandbox is owned
    recovered = registry_at(tmp_path, provider)
    assert (await recovered.release_all()).complete and provider.store == {}
    await recovered.close()


async def test_a_negative_lookup_is_unknown_without_authoritative_evidence(tmp_path):
    provider = FakeProvider(create_errors=[ConnectionError("reset")], hide_absence=True)
    registry = registry_at(tmp_path, provider)
    with pytest.raises(ProvisioningFailure) as failure:
        await provision(registry)
    assert failure.value.disposition is CreationDisposition.AMBIGUOUS
    [unresolved] = registry.unresolved.results
    assert unresolved.status is CleanupStatus.UNCONFIRMED
    await registry.close()


async def test_proven_absence_closes_a_failed_create(tmp_path):
    provider = FakeProvider(create_errors=[ConnectionError("reset")])
    registry = registry_at(tmp_path, provider)
    with pytest.raises(ProvisioningFailure) as failure:
        await provision(registry)
    assert failure.value.disposition is CreationDisposition.DEFINITELY_NOT_CREATED
    assert registry.unresolved.results == ()
    await registry.close()


async def test_a_reused_key_with_a_different_specification_is_rejected(tmp_path):
    registry = registry_at(tmp_path, FakeProvider())
    await provision(registry, specification={"memory": 1})
    with pytest.raises(ResourceContractError, match="different resource"):
        await provision(registry, specification={"memory": 2})
    await registry.close()


async def test_an_unknown_provider_is_a_contract_error(tmp_path):
    registry = registry_at(tmp_path, FakeProvider())
    with pytest.raises(ResourceContractError, match="providers.other"):
        await provision(registry, provider="other")
    await registry.close()


async def test_a_reused_helper_key_with_a_different_specification_is_rejected(tmp_path):
    registry = registry_at(tmp_path, FakeProvider())
    helper = request("builder", kind="builder", release_after_create=True, specification={"v": 1})
    await registry.provision(helper, scope=ResourceScope.RUN, owner_id="run-1")
    changed = request("builder", kind="builder", release_after_create=True, specification={"v": 2})
    with pytest.raises(ResourceContractError, match="different resource"):
        await registry.provision(changed, scope=ResourceScope.RUN, owner_id="run-1")
    await registry.close()


@pytest.mark.parametrize(
    ("known", "helper"),
    [(True, False), (False, False), (True, True)],
    ids=["identity_known", "identity_reconciled", "transient_helper"],
)
async def test_a_create_that_happened_but_failed_is_owned_released_and_still_fails(
    tmp_path, known, helper
):
    provider = FakeProvider()

    async def dead(intent):
        resource = await FakeProvider.create(provider, intent)
        raise ProvisioningFailure(
            "deployed as FAILED",
            CreationDisposition.DEFINITELY_CREATED,
            resource if known else None,
        )

    provider.create = dead
    registry = registry_at(tmp_path, provider)
    service = run_resources(registry) if helper else episode(registry)
    wanted = (
        request("builder-1", kind="builder", release_after_create=True) if helper else request()
    )
    with pytest.raises(ProvisioningFailure, match="FAILED"):
        await service.provision(wanted)
    if helper:
        assert provider.store == {}  # at once, not with its scope
    assert (await service.close()).complete and provider.store == {}
    await registry.close()


@pytest.mark.parametrize(
    ("blocked", "outcome"),
    [
        ("prerequisites", CreationDisposition.DEFINITELY_NOT_CREATED),
        ("create", CreationDisposition.AMBIGUOUS),  # and its lookup, so nothing is proven
        ("reconcile", CreationDisposition.AMBIGUOUS),
        ("validate", CreationDisposition.AMBIGUOUS),
        ("release", CleanupStatus.UNCONFIRMED),
        ("confirm_absent", CleanupStatus.UNCONFIRMED),
    ],
)
async def test_every_provider_operation_is_bounded_and_keeps_its_evidence(
    tmp_path, blocked, outcome
):
    provider = FakeProvider(deadlines=TIGHT, blocked={blocked})
    if blocked == "create":
        provider.blocked.add("reconcile")
    if blocked in ("reconcile", "validate"):
        provider.create_errors.append(ConnectionError("reset"))  # so the create is looked up
    if blocked == "validate":
        provider.ambiguous_after_create, provider.create_errors = True, []
    registry = registry_at(tmp_path, provider)
    if isinstance(outcome, CreationDisposition):
        with pytest.raises(ProvisioningFailure) as failure:
            await provision(registry)
        assert failure.value.disposition is outcome
        kept = outcome is CreationDisposition.AMBIGUOUS
        assert len(registry.unresolved.results) == (1 if kept else 0)
    else:
        handle = await provision(registry)
        result = await registry.release(handle.intent.resource_id)
        assert result.status is outcome and "TimeoutError" in (result.detail or "")
        assert registry.unresolved.results == (result,)
    await registry.close()


async def test_a_create_cut_by_its_deadline_is_closed_by_the_next_runs_recovery(tmp_path):
    provider = FakeProvider(deadlines=TIGHT, blocked={"create", "reconcile"})
    registry = registry_at(tmp_path, provider)
    with pytest.raises(ProvisioningFailure) as failure:
        await provision(registry)
    assert failure.value.disposition is CreationDisposition.AMBIGUOUS
    provider.unblocked.set()  # the provider answers again
    await settle_loop(rounds=20)
    registry.journal.close()
    recovered = registry_at(tmp_path, provider)
    assert (await recovered.release_all()).complete and provider.store == {}
    await recovered.close()


@pytest.mark.parametrize("blocked", ["reconcile", "confirm_absent"])
async def test_lookups_and_absence_checks_hold_the_provisioning_permit(tmp_path, blocked):
    provider = FakeProvider(blocked={blocked}, deadlines=QUICK_LOOKUPS)
    registry = registry_at(tmp_path, provider, limit=1)
    if blocked == "reconcile":
        provider.create_errors.append(ConnectionError("reset"))
        holding = asyncio.create_task(provision(registry, "a"))
    else:
        handle = await provision(registry, "a")
        holding = asyncio.create_task(registry.release(handle.intent.resource_id))
    await settle_loop(rounds=20)
    waiting = asyncio.create_task(provision(registry, "b"))
    await settle_loop(rounds=20)
    assert ("create", "fake-b") not in provider.calls  # the one permit is taken
    for task in (holding, waiting):
        task.cancel()
    await asyncio.gather(holding, waiting, return_exceptions=True)
    provider.blocked.clear()
    await registry.close()
