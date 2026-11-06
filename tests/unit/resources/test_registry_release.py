"""The resource registry releases: absence proven before reuse, dependants first, leases
and scopes."""

import asyncio

import pytest
from support.cancellation import settle_loop
from support.resources import (
    FakeClient,
    FakeProvider,
    FakeSandboxProvider,
    episode,
    not_created,
    registry_at,
    request,
    run_resources,
)

from loops_cookbook.resources.api import (
    CleanupStatus,
    ProvisioningFailure,
    ReconcileResult,
    ReconcileStatus,
    ResourceScope,
)
from loops_cookbook.resources.registry import ResourceContractError

from .rig import provision

SANDBOX_CALLS = {
    "client": lambda service, resource_id: service.sandbox_client(resource_id),
    "close_client": lambda service, resource_id: service.close_sandbox_client(
        resource_id, FakeClient("fake-a")
    ),
    "alive": lambda service, resource_id: service.sandbox_alive(resource_id),
}


async def test_a_released_key_is_created_again_only_after_confirmed_absence(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    first = await provision(registry)
    provider.gate = asyncio.Event()
    releasing = asyncio.create_task(registry.release(first.intent.resource_id))
    await settle_loop(rounds=20)
    again = asyncio.create_task(provision(registry))
    await settle_loop(rounds=20)
    assert [operation for operation, _ in provider.calls] == ["create", "release"]
    provider.gate.set()
    await releasing
    second = await again
    assert second.intent.resource_id != first.intent.resource_id
    assert [operation for operation, _ in provider.calls] == [
        "create",
        "release",
        "confirm_absent",
        "create",
    ]
    await registry.release(second.intent.resource_id)
    assert provider.store == {}
    await registry.close()


async def test_a_key_whose_absence_is_unproven_is_not_created_again(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    first = await provision(registry)
    provider.release_errors.append(ConnectionError("reset"))
    result = await registry.release(first.intent.resource_id)
    assert result.status is CleanupStatus.UNCONFIRMED
    provider.release_errors.append(ConnectionError("reset"))
    with pytest.raises(ProvisioningFailure, match="proven gone"):
        await provision(registry)
    await registry.close()


async def test_release_is_deferred_while_a_dependant_lives(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    base = await provision(registry, "base")
    child = await provision(registry, "child", depends_on=(base.intent.resource_id,))
    deferred = await registry.release(base.intent.resource_id)
    assert deferred.status is CleanupStatus.DEFERRED
    statuses = {r.resource_id: r.status for r in registry.unresolved.results}
    assert statuses[base.intent.resource_id] is CleanupStatus.DEFERRED
    assert len(provider.store) == 2
    await registry.release(child.intent.resource_id)
    assert provider.store == {}
    events = [record.event for record in registry.journal.records]
    assert events.count("release_deferred") == 1
    await registry.close()


async def test_a_repeated_release_does_no_provider_io(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    handle = await provision(registry)
    first = await registry.release(handle.intent.resource_id)
    calls = len(provider.calls)
    second = await registry.release(handle.intent.resource_id)
    assert first == second
    assert len(provider.calls) == calls
    await registry.close()


async def test_an_in_flight_release_survives_its_callers_cancellation(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    handle = await provision(registry)
    provider.gate = asyncio.Event()
    caller = asyncio.create_task(registry.release(handle.intent.resource_id))
    await settle_loop(rounds=20)
    caller.cancel()
    await settle_loop(rounds=20)
    provider.gate.set()
    await registry.close()  # waits for releases in flight
    assert provider.store == {} and registry.unresolved.results == ()


async def test_run_resources_outlive_episode_leases_and_release_after_dependants(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    run = run_resources(registry)
    shared = await run.provision(request("cache"))
    first, second = episode(registry, "attempt-1"), episode(registry, "attempt-2")
    await first.acquire(shared.intent.resource_id)
    await second.acquire(shared.intent.resource_id)
    await first.close()
    run_report = asyncio.create_task(run.close())
    await settle_loop(rounds=20)
    assert run_report.done()
    assert run_report.result().results[0].status is CleanupStatus.DEFERRED
    assert len(provider.store) == 1  # the second episode still holds it
    await second.close()
    assert provider.store == {}
    await registry.close()


async def test_episode_code_cannot_provision_run_resources_or_release_others(tmp_path):
    registry = registry_at(tmp_path, FakeProvider())
    service = episode(registry)
    handle = await service.provision(request())
    assert handle.intent.scope is ResourceScope.EPISODE
    assert handle.intent.owner_id == "attempt-1"
    other = await episode(registry, "attempt-2").provision(request("b"))
    with pytest.raises(ResourceContractError):
        await service.release(other.intent.resource_id)
    with pytest.raises(ResourceContractError, match="only a run resource"):
        await service.acquire(other.intent.resource_id)
    with pytest.raises(ResourceContractError):
        service.usage_environment(other.intent.resource_id)
    with pytest.raises(ResourceContractError, match="needs no lease"):
        await service.acquire(handle.intent.resource_id)
    await registry.close()


async def test_a_run_resource_cannot_depend_on_an_episode_resource(tmp_path):
    registry = registry_at(tmp_path, FakeProvider())
    owned = await episode(registry).provision(request())
    with pytest.raises(ResourceContractError, match="depend on an episode"):
        await run_resources(registry).provision(
            request("b", depends_on=(owned.intent.resource_id,))
        )
    await registry.close()


async def test_usage_environment_is_the_adapters_for_owned_and_leased_resources(tmp_path):
    registry = registry_at(tmp_path, FakeProvider())
    shared = await run_resources(registry).provision(request("cache"))
    service = episode(registry)
    owned = await service.provision(request())
    assert service.usage_environment(owned.intent.resource_id)["FAKE_NAME"] == "fake-a"
    with pytest.raises(ResourceContractError):
        service.usage_environment(shared.intent.resource_id)
    lease = await service.acquire(shared.intent.resource_id)
    assert service.usage_environment(shared.intent.resource_id)["FAKE_NAME"] == "fake-cache"
    with pytest.raises(ResourceContractError, match="already holds"):
        await service.acquire(shared.intent.resource_id)
    await lease.close()
    with pytest.raises(ResourceContractError):
        service.usage_environment(shared.intent.resource_id)
    await registry.close()


@pytest.mark.parametrize("call", sorted(SANDBOX_CALLS))
async def test_a_sandbox_is_used_only_through_an_owned_or_leased_resource(tmp_path, call):
    registry = registry_at(tmp_path, FakeSandboxProvider())
    other = await episode(registry, "attempt-2").provision(request())
    with pytest.raises(ResourceContractError, match="neither owned nor leased"):
        await SANDBOX_CALLS[call](episode(registry), other.intent.resource_id)
    await registry.close()


@pytest.mark.parametrize("call", sorted(SANDBOX_CALLS))
async def test_an_adapter_that_is_not_a_sandbox_provider_backs_no_sandbox(tmp_path, call):
    registry = registry_at(tmp_path, FakeProvider())
    service = episode(registry)
    handle = await service.provision(request())
    with pytest.raises(ResourceContractError, match="is not a SandboxProvider"):
        await SANDBOX_CALLS[call](service, handle.intent.resource_id)
    await registry.close()


@pytest.mark.parametrize("call", ["client", "alive"])
async def test_a_released_sandbox_is_neither_reached_nor_read(tmp_path, call):
    provider = FakeSandboxProvider()
    registry = registry_at(tmp_path, provider)
    service = episode(registry)
    handle = await service.provision(request())
    await service.release(handle.intent.resource_id)
    with pytest.raises(ResourceContractError, match="is not a live resource"):
        await SANDBOX_CALLS[call](service, handle.intent.resource_id)
    assert [op for op, _ in provider.calls if op == call] == []
    await registry.close()


async def test_a_liveness_that_is_not_a_bool_is_refused(tmp_path):
    provider = FakeSandboxProvider()

    async def unsure(resource):
        return None

    provider.alive = unsure  # pyright: ignore[reportAttributeAccessIssue]
    registry = registry_at(tmp_path, provider)
    service = episode(registry)
    handle = await service.provision(request())
    with pytest.raises(ResourceContractError, match="alive\\(\\) returned None, not a bool"):
        await service.sandbox_alive(handle.intent.resource_id)
    await registry.close()


async def test_releases_run_under_the_provisioning_limit(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider, limit=1)
    service = episode(registry)
    handles = [await service.provision(request(key)) for key in "ab"]
    provider.gate = asyncio.Event()
    releases = asyncio.gather(*(service.release(h.intent.resource_id) for h in handles))
    await settle_loop(rounds=20)
    assert provider.in_flight == 1
    provider.gate.set()
    await releases
    assert provider.most_in_flight == 1
    assert provider.store == {}
    await registry.close()


async def test_episode_release_is_a_noop_when_repeated_and_counts_as_excluded_time(tmp_path):
    registry = registry_at(tmp_path, FakeProvider())
    entered = []

    class Excluded:
        def __enter__(self):
            entered.append(True)

        def __exit__(self, *exc):
            return None

    from loops_cookbook.resources.services import EpisodeResourceService

    service = EpisodeResourceService(registry, "attempt-1", Excluded)
    handle = await service.provision(request())
    await service.release(handle.intent.resource_id)
    await service.release(handle.intent.resource_id)
    report = await service.close()
    assert report.complete
    assert len(entered) == 4
    await registry.close()


async def test_nothing_new_may_depend_on_or_lease_a_resource_being_released(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    run = run_resources(registry)
    base = await run.provision(request("base"))
    provider.gate = asyncio.Event()
    releasing = asyncio.create_task(registry.release(base.intent.resource_id))
    await settle_loop(rounds=20)
    assert provider.in_flight == 1  # the release is at the provider
    service = episode(registry)
    with pytest.raises(ResourceContractError, match="not a live resource"):
        await service.acquire(base.intent.resource_id)
    with pytest.raises(ResourceContractError, match="not a live resource"):
        await service.provision(request("child", depends_on=(base.intent.resource_id,)))
    provider.gate.set()
    assert (await releasing).status is CleanupStatus.RELEASED
    assert provider.store == {}
    await registry.close()


async def test_the_run_holds_one_lease_per_resource_and_ends_them_at_close(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    run = run_resources(registry)
    shared = await run.provision(request("cache"))
    await run.acquire(shared.intent.resource_id)
    with pytest.raises(ResourceContractError, match="already holds"):
        await run.acquire(shared.intent.resource_id)
    report = await run.close()
    assert report.complete and provider.store == {}
    await registry.close()


async def test_a_dependant_that_fails_to_create_frees_its_deferred_dependency(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    service = episode(registry)
    base = await service.provision(request("base"))
    provider.gate = asyncio.Event()
    child = asyncio.create_task(
        service.provision(request("child", depends_on=(base.intent.resource_id,)))
    )
    await settle_loop(rounds=20)
    assert (await service.release(base.intent.resource_id)).status is CleanupStatus.DEFERRED
    provider.create_errors.append(not_created("quota"))
    provider.gate.set()
    with pytest.raises(ProvisioningFailure, match="quota"):
        await child
    assert provider.store == {}  # the deferred base went as soon as nothing needed it
    await registry.close()


@pytest.mark.parametrize("seen", [ReconcileStatus.UNKNOWN, ReconcileStatus.FOUND])
async def test_a_release_counts_only_once_absence_is_proven(tmp_path, seen):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    handle = await provision(registry)

    async def unproven(resource):
        if seen is ReconcileStatus.FOUND:
            return ReconcileResult(seen, resource)
        return ReconcileResult(seen)

    provider.confirm_absent = unproven
    result = await registry.release(handle.intent.resource_id)
    assert result.status is CleanupStatus.UNCONFIRMED
    assert [r.resource_id for r in registry.unresolved.results] == [handle.intent.resource_id]
    await registry.close()


async def test_cleanup_releases_each_owners_resources_concurrently(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    for owner in ("attempt-1", "attempt-2", "attempt-3"):
        await registry.provision(request("a"), scope=ResourceScope.EPISODE, owner_id=owner)
    registry.journal.close()
    provider.delay = 0.05
    recovered = registry_at(tmp_path, provider)
    report = await recovered.release_all()
    assert [r.status for r in report.results] == [CleanupStatus.RELEASED] * 3
    assert provider.most_in_flight >= 3
    await recovered.close()


async def test_a_scope_is_released_together_dependants_first(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    run = run_resources(registry)
    base = await run.provision(request("base"))
    for key in "ab":
        await run.provision(request(key, depends_on=(base.intent.resource_id,)))
    provider.gate = asyncio.Event()
    releasing = asyncio.create_task(run.close())
    await settle_loop(rounds=20)
    assert sorted(provider.calls[-2:]) == [("release", "fake-a"), ("release", "fake-b")]
    assert provider.in_flight == 2  # both at once; the dependency waits for them
    provider.gate.set()
    report = await releasing
    assert [r.status for r in report.results] == [CleanupStatus.RELEASED] * 3
    assert provider.calls[-2:] == [("release", "fake-base"), ("confirm_absent", "fake-base")]
    await registry.close()
