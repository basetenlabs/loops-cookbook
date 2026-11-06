"""The Daytona adapter through the real registry, against a fake control plane.

Each test asserts what Daytona is asked to do and what the resource services report,
never how the adapter gets there.
"""

import asyncio
import dataclasses
import time

import pytest

pytest.importorskip("daytona")

from daytona import CreateSandboxFromSnapshotParams
from support.daytona_api import (
    CREDENTIALS,
    ENVIRON,
    REQUIRED,
    SETTINGS,
    FakeApi,
    Sandbox,
    StatusError,
    never_connected,
    sandbox_request,
)
from support.fakes import no_wait
from support.resources import episode, local

from loops_cookbook.providers import create_adapters
from loops_cookbook.providers.daytona import Daytona, DaytonaSandbox, create_adapter
from loops_cookbook.providers.daytona.adapter import DaytonaAdapter
from loops_cookbook.providers.daytona.api import create_params
from loops_cookbook.providers.daytona.settings import DaytonaSettings
from loops_cookbook.resources.api import (
    CapacityProvider,
    CleanupStatus,
    CreationDisposition,
    ProviderResource,
    ProvisioningFailure,
    ReconcileStatus,
    ResourceRequest,
)
from loops_cookbook.resources.registry import ResourceRegistry
from loops_cookbook.runtime.context import RuntimeConfigurationError


@pytest.fixture
def api():
    return FakeApi()


@pytest.fixture
def adapter(api):
    return DaytonaAdapter(SETTINGS, CREDENTIALS, api, sleep=no_wait)


def registry_for(path, adapter, **options):
    return ResourceRegistry.open(
        path,
        {"daytona": adapter},
        mode="write",
        run_id="run-1",
        kind=local,
        sleep=no_wait,
        jitter=lambda: 0.5,  # a first wait of 0.5 s, inside half of SETTINGS' 2 s create
        **options,
    )


@pytest.fixture
async def registry(tmp_path, adapter):
    opened = registry_for(tmp_path, adapter)
    yield opened
    await opened.close()


@pytest.fixture
def service(registry):
    return episode(registry)


def called(api, verb):
    return [subject for call, subject in api.calls if call == verb]


# ------------------------------------------------------------ configuration


def test_the_credential_comes_from_the_environment_and_is_kept_secret():
    adapter = create_adapter(REQUIRED, environ={**ENVIRON, "DAYTONA_API_URL": "https://d.example"})
    env = adapter.usage_environment(ProviderResource("id-1", "box", {}))
    assert env["DAYTONA_SANDBOX_ID"] == "id-1" and env["DAYTONA_API_URL"] == "https://d.example"
    assert env["DAYTONA_API_KEY"].get_secret_value() == "dtn-key-123"  # pyright: ignore[reportAttributeAccessIssue]
    assert "dtn-key-123" not in repr(env)


def test_a_missing_credential_fails_before_any_daytona_call():
    with pytest.raises(RuntimeConfigurationError, match="DAYTONA_API_KEY"):
        create_adapter({}, environ={})


@pytest.mark.parametrize(
    ("table", "allowed"), [({}, False), ({"target": "us"}, False), ({"target": "eu"}, True)]
)
def test_the_target_is_the_tables_and_an_environment_target_must_agree(table, allowed):
    """The SDK reads DAYTONA_TARGET when no target is configured; the table owns it."""
    environ = {**ENVIRON, "DAYTONA_TARGET": "eu"}
    if allowed:
        create_adapter({**REQUIRED, **table}, environ=environ)
    else:
        with pytest.raises(RuntimeConfigurationError, match="DAYTONA_TARGET"):
            create_adapter({**REQUIRED, **table}, environ=environ)


@pytest.mark.parametrize(
    "table",
    [
        {"api_key": "DAYTONA_API_KEY"},  # credentials are not settings
        {"create_timeout_seconds": 0},
        {"cpu": 0},
        {"auto_stop_minutes": 0, "ttl_minutes": 0},  # a leak would never end
    ],
    ids=["api_key", "zero_deadline", "zero_cpu", "immortal"],
)
def test_an_invalid_table_is_refused_naming_it(table):
    with pytest.raises(RuntimeConfigurationError, match=r"\[providers.daytona\] is invalid"):
        create_adapter({**REQUIRED, **table}, environ=ENVIRON)


def test_a_table_of_only_the_required_keys_has_every_default_deadline():
    adapter = create_adapter(REQUIRED, environ=ENVIRON)
    deadlines = adapter.deadlines
    assert (
        deadlines.create_seconds,
        deadlines.reconcile_seconds,
        deadlines.release_seconds,
        deadlines.confirm_absent_seconds,
        deadlines.list_seconds,
        deadlines.attach_seconds,
    ) == (600, 60, 120, 60, 60, 30)


async def test_daytona_is_a_built_in_provider(monkeypatch):
    monkeypatch.setenv("DAYTONA_API_KEY", "dtn-key-123")
    adapters = await create_adapters({"daytona": {"target": "us", **REQUIRED}})
    try:
        assert isinstance(adapters["daytona"], DaytonaAdapter)
        # How a run finds the provider whose capacity bounds its rollouts.
        assert isinstance(adapters["daytona"], CapacityProvider)
    finally:
        for adapter in adapters.values():
            await adapter.close()


def test_the_adapter_backs_daytona_sandboxes(adapter):
    assert adapter.sandbox_kind is Daytona


# ---------------------------------------------------------------- creates


async def test_a_sandbox_is_created_with_ownership_labels_and_a_lifecycle_that_ends_it(
    service, api
):
    handle = await service.provision(sandbox_request())
    [config] = api.created
    assert config["name"] == f"loops-{handle.intent.resource_id}"
    labels = config["labels"]
    assert labels["harbor.session_id"] == "trial__env"  # Harbor's labels are kept
    assert labels["owner"] == "team-a"  # [providers.daytona] labels
    assert labels["loops.run_id"] == "run-1"
    assert labels["loops.owner_id"] == "attempt-1"  # the attempt
    assert labels["loops.resource_id"] == handle.intent.resource_id
    assert config["resources"] == {"cpu": 2, "memory": 4, "disk": 3}  # the table's
    assert (config["auto_stop_interval"], config["auto_delete_interval"]) == (120, 0)
    assert config["ttl_minutes"] == 1440
    assert handle.resource.provider_id in api.sandboxes
    assert (await service.close()).complete and api.sandboxes == {}


async def test_what_the_request_states_wins_over_the_tables_defaults(service, api):
    await service.provision(
        sandbox_request(
            resources={"cpu": 1},
            auto_stop_interval=60,
            ttl_minutes=90,
            ephemeral=True,
        )
    )
    [config] = api.created
    assert config["resources"] == {"cpu": 1, "memory": 4, "disk": 3}
    assert (config["auto_stop_interval"], config["ttl_minutes"]) == (60, 90)
    assert "auto_delete_interval" not in config  # ephemeral is Daytona's own delete-on-stop


async def test_an_image_sandbox_always_names_its_size_from_the_table(tmp_path, api):
    """The table's sizes are sent, never left to a Daytona default."""
    adapter = DaytonaAdapter(DaytonaSettings(**REQUIRED), CREDENTIALS, api, sleep=no_wait)
    registry = registry_for(tmp_path / "empty", adapter)
    await episode(registry).provision(sandbox_request())
    [config] = api.created
    assert config["resources"] == {"cpu": 1, "memory": 1, "disk": 3}
    defaults = adapter.capacity.defaults
    assert (defaults.cpu, defaults.memory_gib, defaults.disk_gib) == (1, 1, 3)
    await registry.close()


async def test_a_snapshot_sandbox_is_not_sized(service, api):
    await service.provision(sandbox_request(image=None, snapshot="harbor__abc__snapshot"))
    [config] = api.created
    assert config["snapshot"] == "harbor__abc__snapshot" and "resources" not in config


@pytest.mark.parametrize(
    ("sandbox", "reason"),
    [
        ({"region": "us"}, "takes no region"),
        ({"snapshot": "s"}, "from an image or a snapshot"),
        ({"resources": {"cpus": 2}}, "sandbox.resources takes only"),
        ({"auto_stop_interval": 0, "ttl_minutes": 0}, "would never stop"),
        ({"auto_stop_interval": -1}, "auto_stop_interval must be"),
        ({"auto_delete_interval": -1}, "auto_delete_interval must be"),
        ({"auto_archive_interval": -1}, "auto_archive_interval must be"),
        ({"ttl_minutes": -1}, "ttl_minutes must be"),
        ({"labels": ["a"]}, "sandbox.labels must be"),
        ({"labels": {"a": 1}}, "sandbox.labels must be"),
        ({"image": ["python"]}, "sandbox.image must be"),
    ],
    ids=[
        "unknown_key",
        "image_and_snapshot",
        "unknown_resource",
        "never_ends",
        "negative_auto_stop",
        "kept_forever",
        "negative_auto_archive",
        "negative_ttl",
        "list_labels",
        "non_string_label",
        "list_image",
    ],
)
async def test_a_request_daytona_cannot_take_is_refused_before_it_is_journaled(
    service, api, sandbox, reason
):
    with pytest.raises(ValueError, match=reason):
        await service.provision(sandbox_request(**sandbox))
    assert api.calls == []
    assert (await service.close()).results == ()  # no intent was journaled


@pytest.mark.parametrize("key", ["image", "resources", "labels"])
async def test_a_null_object_is_refused_before_it_is_journaled(service, api, key):
    request = sandbox_request()
    specification = {"sandbox": {**request.specification["sandbox"], key: None}}
    with pytest.raises(ValueError, match=f"sandbox.{key} must be"):
        await service.provision(dataclasses.replace(request, specification=specification))
    assert api.calls == []
    assert (await service.close()).results == ()


async def test_a_sandbox_with_no_name_gets_one_unique_to_its_resource(service, api):
    first = await service.provision(sandbox_request(key="a"))
    second = await service.provision(sandbox_request(key="b"))
    names = {first.resource.name, second.resource.name}
    assert names == {f"loops-{h.intent.resource_id}" for h in (first, second)}


# ------------------------------------------------------------ dispositions


@pytest.mark.parametrize(
    "error",
    [
        StatusError(429, "Too many requests"),
        StatusError(400, "bad request"),
        StatusError(401, "unauthorized"),
        StatusError(403, "forbidden"),
        never_connected(),
    ],
    ids=["rate_limited", "bad_request", "unauthorized", "forbidden", "never_connected"],
)
async def test_a_failure_after_the_create_landed_owns_the_sandbox(service, api, error):
    """The SDK's create goes on calling Daytona after the create; a failure of those
    calls is not a create that left nothing."""
    api.create_errors, api.land_before_error = [error], True
    handle = await service.provision(sandbox_request())
    assert len(called(api, "create")) == 1 and handle.resource.provider_id in api.sandboxes
    assert (await service.close()).complete and api.sandboxes == {}


@pytest.mark.parametrize("found", ["started", "creating", "pending_build", "starting"])
async def test_an_ambiguous_create_found_by_name_is_owned_and_released(service, api, found):
    api.create_errors, api.land_before_error = [StatusError(502, "bad gateway")], True
    api.get_errors = [StatusError(404), StatusError(503)]  # not readable at first
    api.start = [found]  # a sandbox still building is found as it is
    handle = await service.provision(sandbox_request())
    assert handle.resource.provider_id in api.sandboxes
    assert len(called(api, "create")) == 1  # never posted twice
    assert (await service.close()).complete and api.sandboxes == {}  # owned, released
    assert called(api, "delete") == [handle.resource.provider_id]


@pytest.mark.parametrize("dead", ["error", "build_failed", "destroyed"])
async def test_an_ambiguous_create_found_dead_fails_and_is_released(service, api, dead):
    api.create_errors, api.land_before_error = [StatusError(502, "bad gateway")], True
    api.start = [dead]
    with pytest.raises(ProvisioningFailure, match=dead) as failure:
        await service.provision(sandbox_request())
    assert failure.value.disposition is CreationDisposition.DEFINITELY_CREATED
    assert len(called(api, "create")) == 1
    assert (await service.close()).complete and api.sandboxes == {}


async def test_a_sandbox_found_still_building_fails_at_attach_once_it_dies(service, api):
    api.create_errors, api.land_before_error = [StatusError(502, "bad gateway")], True
    api.get_errors = [StatusError(503)]  # the create's own lookup fails: a reconcile finds it
    api.start = ["pending_build", "build_failed"]
    handle = await service.provision(sandbox_request())
    with pytest.raises(ProvisioningFailure, match="build_failed"):
        await service.sandbox_client(handle.intent.resource_id)
    assert (await service.close()).complete and api.sandboxes == {}


async def test_a_create_cut_off_mid_build_is_found_and_released(tmp_path, api):
    """A cold build outlasting the create deadline: the reconcile owns what it finds."""
    tight = SETTINGS.model_copy(
        update={"create_timeout_seconds": 0.05, "reconcile_timeout_seconds": 0.05}
    )
    registry = registry_for(
        tmp_path / "tight", DaytonaAdapter(tight, CREDENTIALS, api, sleep=no_wait)
    )
    service = episode(registry)
    api.start = ["pending_build"]  # it never finishes building within the test
    api.answer_gate = asyncio.Event()  # and the create's answer never comes
    handle = await service.provision(sandbox_request())
    assert handle.resource.provider_id in api.sandboxes
    assert (await service.close()).complete
    assert called(api, "delete") == [handle.resource.provider_id] and api.sandboxes == {}
    await registry.close()


async def test_a_create_cut_off_by_its_deadline_is_reconciled_not_repeated(tmp_path, api):
    """Harbor's own create retries a timed-out create; the registry looks it up instead."""
    tight = SETTINGS.model_copy(update={"create_timeout_seconds": 0.05})
    registry = registry_for(
        tmp_path / "tight", DaytonaAdapter(tight, CREDENTIALS, api, sleep=no_wait)
    )
    service = episode(registry)
    api.answer_gate = asyncio.Event()  # Daytona creates it, and its answer never comes
    handle = await service.provision(sandbox_request())
    assert len(called(api, "create")) == 1  # read back by name, not created again
    assert handle.resource.provider_id in api.sandboxes
    assert (await service.close()).complete and api.sandboxes == {}
    await registry.close()


# ------------------------------------------------------------------ release


@pytest.mark.parametrize("transient", [[], [StatusError(503)]], ids=["clean", "503_first"])
async def test_a_release_is_complete_only_once_daytona_answers_404_for_its_id(
    service, api, transient
):
    handle = await service.provision(sandbox_request())
    api.linger = 2  # the delete is accepted; the sandbox reads destroying for a while
    original = api.get

    async def get(id_or_name):
        if transient:
            raise transient.pop(0)  # a transient lookup failure proves nothing
        return await original(id_or_name)

    api.get = get
    [result] = (await service.close()).results
    assert result.status is CleanupStatus.RELEASED and api.sandboxes == {}
    assert called(api, "delete") == [handle.resource.provider_id]  # by id, not name
    assert "404 after its delete" in (result.detail or "")


async def test_a_destroyed_sandbox_is_absent(service, adapter, api):
    handle = await service.provision(sandbox_request())
    api.sandboxes[handle.resource.provider_id].states = ["destroyed"]
    found = await adapter.confirm_absent(handle.resource)
    assert found.status is ReconcileStatus.PROVEN_ABSENT and "destroyed" in (found.evidence or "")


async def test_a_delete_daytona_cannot_answer_leaves_the_sandbox_unresolved(service, api):
    await service.provision(sandbox_request())
    api.delete_errors = [StatusError(500)] * 10
    [result] = (await service.close()).results
    assert result.status is not CleanupStatus.RELEASED and len(api.sandboxes) == 1


async def test_a_closing_registry_releases_daytona_sandboxes_within_its_bound(
    tmp_path, api, adapter
):
    """Shutdown stops new creates, releases what exists, and is bounded."""
    registry = registry_for(tmp_path / "closing", adapter)
    services = [episode(registry, f"attempt-{n}") for n in range(3)]
    for n, service in enumerate(services[:2]):
        await service.provision(sandbox_request(key=f"k{n}"))
    api.create_gate = asyncio.Event()  # a third create is in flight and never answers
    stuck = asyncio.create_task(services[2].provision(sandbox_request(key="k2")))
    await asyncio.sleep(0.01)
    started = time.monotonic()
    registry.begin_shutdown()
    with pytest.raises(ProvisioningFailure, match="closing"):
        await services[0].provision(sandbox_request(key="late"))
    reports = [await service.close() for service in services[:2]]
    await registry.close(timeout=0.2)
    assert time.monotonic() - started < 2
    assert all(report.complete for report in reports) and api.sandboxes == {}
    with pytest.raises(ProvisioningFailure):
        await stuck
    assert len(called(api, "create")) == 3  # the late request sent nothing
    [kept] = registry.unresolved.results  # the cut-off create, for `loops cleanup`
    assert kept.status is not CleanupStatus.RELEASED


# ---------------------------------------------------------------- sweeps


async def test_list_owned_returns_only_live_sandboxes_with_every_label_page_by_page(adapter, api):
    api.list_page_size = 2
    run = {"loops.run_id": "run-1", "loops.scope": "run"}
    api.sandboxes["a"] = Sandbox("a", "a", dict(run), ["started"])
    api.sandboxes["b"] = Sandbox("b", "b", {"loops.run_id": "run-1"}, ["started"])
    api.sandboxes["c"] = Sandbox("c", "c", {**run, "loops.run_id": "run-2"}, ["started"])
    api.sandboxes["d"] = Sandbox("d", "d", dict(run), ["destroyed"])
    api.sandboxes["e"] = Sandbox("e", "e", dict(run), ["stopped"])
    pages = [[r.name for r in page] async for page in adapter.list_owned(run)]
    assert pages == [["a"], ["e"]]


async def test_a_journal_replay_releases_what_a_crashed_run_left(tmp_path, api, adapter):
    from loops_cookbook.resources.cleanup import reconcile_journal

    registry = registry_for(tmp_path / "crashed", adapter)
    await episode(registry).provision(sandbox_request())
    registry.journal.close()  # the process dies: nothing is released
    report = await reconcile_journal(tmp_path / "crashed", {"daytona": adapter}, kind=local)
    assert report.complete and api.sandboxes == {}


# ------------------------------------------------------- an Episode's sandbox


async def test_a_sandbox_is_alive_only_while_it_is_ours_and_started(service, api):
    handle = await service.provision(sandbox_request())
    resource_id = handle.intent.resource_id
    assert await service.sandbox_alive(resource_id) is True
    api.sandboxes[handle.resource.provider_id].states = ["stopped"]
    assert await service.sandbox_alive(resource_id) is False
    del api.sandboxes[handle.resource.provider_id]
    assert await service.sandbox_alive(resource_id) is False


async def test_the_client_is_the_sdks_sandbox_checked_against_its_ownership(service, api):
    handle = await service.provision(sandbox_request())
    client = await service.sandbox_client(handle.intent.resource_id)
    assert client.id == handle.resource.provider_id
    api.sandboxes[client.id].labels["loops.resource_id"] = "someone-else"
    with pytest.raises(ProvisioningFailure, match="ownership"):
        await service.sandbox_client(handle.intent.resource_id)


def test_an_episodes_daytona_sandbox_is_sized_only_from_an_image():
    spec = DaytonaSandbox.model_validate({"image": "python:3.12-slim", "cpu": 2, "memory_gib": 4})
    assert spec.specification() == {
        "sandbox": {"image": {"base": "python:3.12-slim"}, "resources": {"cpu": 2, "memory": 4}}
    }
    assert DaytonaSandbox.model_validate({}).specification() == {"sandbox": {}}
    with pytest.raises(ValueError, match="name an image"):
        DaytonaSandbox.model_validate({"cpu": 2})


async def test_an_episodes_daytona_sandbox_starts_from_a_named_snapshot(service, api):
    spec = DaytonaSandbox.model_validate({"snapshot": "harbor__abc__snapshot"})
    assert spec.specification() == {"sandbox": {"snapshot": "harbor__abc__snapshot"}}
    await service.provision(
        ResourceRequest(
            provider="daytona",
            kind="sandbox",
            logical_key="trial",
            specification=spec.specification(),
        )
    )
    [config] = api.created
    params = create_params(config)
    assert isinstance(params, CreateSandboxFromSnapshotParams)
    assert params.snapshot == "harbor__abc__snapshot"


@pytest.mark.parametrize(
    ("sandbox", "reason"),
    [
        ({"image": "python:3.12-slim", "snapshot": "s"}, "an image or a snapshot"),
        ({"snapshot": "s", "cpu": 2}, "name an image"),
        ({"snapshot": "s", "disk_gib": 10}, "name an image"),
    ],
    ids=["image_and_snapshot", "sized_snapshot", "sized_snapshot_disk"],
)
def test_an_episodes_daytona_snapshot_is_neither_an_image_nor_sized(sandbox, reason):
    with pytest.raises(ValueError, match=reason):
        DaytonaSandbox.model_validate(sandbox)
