"""The Blaxel adapter through the real registry, against a fake control plane.

Each test asserts what Blaxel is asked to do and what the resource services report,
never how the adapter gets there.
"""

import asyncio
import dataclasses
from types import SimpleNamespace as NS

import httpx
import pytest

pytest.importorskip("blaxel")

from support.blaxel_api import (
    CREDENTIALS,
    ENVIRON,
    SETTINGS,
    TABLE,
    FakeApi,
    Sandbox,
    StatusError,
    sandbox_request,
)
from support.fakes import no_wait
from support.resources import episode, local

from loops_cookbook.providers.blaxel import create_adapter, usage
from loops_cookbook.providers.blaxel.adapter import BlaxelAdapter
from loops_cookbook.providers.blaxel.settings import IMAGE_BUILD
from loops_cookbook.resources.api import (
    CleanupStatus,
    CreationDisposition,
    ProviderResource,
    ProvisioningFailure,
    ReconcileStatus,
    ResourceIntent,
    ResourceRequest,
    ResourceScope,
)
from loops_cookbook.resources.registry import FAILED_HELPER_COOLDOWN_SECONDS, ResourceRegistry
from loops_cookbook.runtime.context import RuntimeConfigurationError

from .sdk import sdk_sandbox

SANDBOX = "harbor-env-trial-env-abc123"
IMAGE = "harbor-img-env-abc123"


@pytest.fixture
def api():
    return FakeApi()


@pytest.fixture
def adapter(api):
    return BlaxelAdapter(SETTINGS, CREDENTIALS, api, sleep=no_wait)


@pytest.fixture
async def registry(tmp_path, adapter):
    opened = ResourceRegistry.open(
        tmp_path,
        {"blaxel": adapter},
        mode="write",
        run_id="run-1",
        kind=local,
        sleep=no_wait,
        jitter=lambda: 0.5,  # a first wait of 0.5 s, inside half of SETTINGS' 2 s create
    )
    yield opened
    await opened.close()


@pytest.fixture
def service(registry):
    return episode(registry)


def called(api, verb):
    return [name for call, name in api.calls if call == verb]


# ------------------------------------------------------------ configuration


def test_credentials_come_from_the_environment_as_on_develop():
    adapter = create_adapter(TABLE, environ=ENVIRON)
    env = adapter.usage_environment(ProviderResource("team/box", "box", {}))
    assert env["BL_WORKSPACE"] == "team" and env["BLAXEL_SANDBOX_NAME"] == "box"
    assert env["BL_API_KEY"].get_secret_value() == "bl-key-123"  # pyright: ignore[reportAttributeAccessIssue]
    assert "bl-key-123" not in repr(env)


@pytest.mark.parametrize("missing", ["BL_WORKSPACE", "BL_API_KEY"])
def test_a_missing_credential_fails_before_any_blaxel_call(missing):
    environ = {name: value for name, value in ENVIRON.items() if name != missing}
    with pytest.raises(RuntimeConfigurationError, match=missing):
        create_adapter(TABLE, environ=environ)


@pytest.mark.parametrize(
    "table",
    [
        {**TABLE, "api_key": "BLAXEL_API_KEY"},  # credentials are not settings
        {**TABLE, "workspace": "team"},
        {**TABLE, "create_timeout_seconds": 0},
        {**TABLE, "image_build_memory_mb": 0},  # Blaxel's range: 1 to 32768 MB
        {**TABLE, "image_build_volume_mb": 131073},  # 0 to 131072 MB
    ],
    ids=["api_key", "workspace", "zero_deadline", "zero_build_memory", "build_volume_past_blaxels"],
)
def test_provider_settings_are_the_region_the_deadlines_and_labels_only(table):
    with pytest.raises(RuntimeConfigurationError, match=r"\[providers.blaxel\]"):
        create_adapter(table, environ=ENVIRON)


def test_a_region_alone_is_a_whole_blaxel_table_with_every_deadline_defaulted():
    deadlines = create_adapter({"region": "us-was-1"}, environ=ENVIRON).deadlines
    assert (
        deadlines.create_for("image_build"),
        deadlines.create_seconds,
        deadlines.attach_seconds,
        deadlines.reconcile_seconds,
        deadlines.release_seconds,
        deadlines.confirm_absent_seconds,
        deadlines.list_seconds,
    ) == (420, 120, 30, 60, 120, 60, 60)


@pytest.mark.parametrize("kind", ["attach", "image_build"])
def test_a_deadline_set_in_the_table_is_the_one_used(kind):
    table = {"region": "us-was-1", f"{kind}_timeout_seconds": 45}
    deadlines = create_adapter(table, environ=ENVIRON).deadlines
    used = deadlines.attach_seconds if kind == "attach" else deadlines.create_for("image_build")
    assert used == 45
    assert deadlines.create_for("sandbox") == 120  # an image build's deadline is its own


def test_a_blaxel_table_without_a_region_is_refused_naming_it():
    with pytest.raises(
        RuntimeConfigurationError,
        match=r"(?s)\[providers.blaxel\] is invalid.*region\n  Field required",
    ):
        create_adapter({"attach_timeout_seconds": 45}, environ=ENVIRON)


@pytest.mark.parametrize(
    "name", ["Harbor-Upper", "trailing-", "-leading", "under_score", "x" * 41, ""]
)
async def test_only_valid_blaxel_names_are_created(service, api, name):
    with pytest.raises(ValueError, match="not a Blaxel name"):
        await service.provision(sandbox_request(name=name))
    assert api.calls == []


async def test_a_sandbox_asked_for_without_a_name_gets_one_of_its_own(service, api):
    """A runtime that asks by specification alone, as a native one does, never collides."""
    first = await service.provision(sandbox_request(key="a", name=None))
    second = await service.provision(sandbox_request(key="b", name=None))
    names = [handle.resource.name for handle in (first, second)]
    assert names == [f"loops-{h.intent.resource_id}" for h in (first, second)]
    assert len(set(names)) == 2 and set(names) <= set(api.sandboxes)


# ---------------------------------------------------------------- sandboxes


async def test_a_sandbox_is_created_as_harbor_configured_it_and_owned_until_released(service, api):
    api.deploy = ["DEPLOYING", "DEPLOYED"]
    handle = await service.provision(sandbox_request())
    config = api.sandboxes[SANDBOX].config
    assert config["name"] == SANDBOX and config["region"] == "us-pdx-1"
    assert config["labels"]["session-id"] == "trial__env"  # Harbor's labels are kept
    assert config["labels"]["owner"] == "team-a"  # [providers.blaxel] labels
    assert config["labels"]["loops.resource_id"] == handle.intent.resource_id
    assert handle.resource.metadata["status"] == "DEPLOYED"  # not before
    assert (await service.close()).complete and api.sandboxes == {}


@pytest.mark.parametrize("final", ["DEPLOYED", "FAILED", "TERMINATED", "DELETING"])
async def test_an_ambiguous_create_is_read_back_by_name_and_must_deploy(service, api, final):
    api.create_errors, api.land_before_error = [StatusError(500)], True
    api.get_errors = [StatusError(404), StatusError(503)]  # not readable at first
    api.deploy = ["DEPLOYING", final]
    if final == "DEPLOYED":
        handle = await service.provision(sandbox_request())
        assert handle.resource.name == SANDBOX
    else:
        with pytest.raises(ProvisioningFailure, match=final) as failure:
            await service.provision(sandbox_request())
        assert failure.value.disposition is CreationDisposition.DEFINITELY_CREATED
    assert called(api, "create") == [SANDBOX]  # never posted twice
    assert (await service.close()).complete and api.sandboxes == {}  # owned, released


async def test_a_sandbox_that_cannot_be_read_back_is_owned_released_and_fails(service, api):
    api.deploy = ["DEPLOYING"]
    api.get_errors = [StatusError(400)]  # not a transient lookup failure
    with pytest.raises(ProvisioningFailure) as failure:
        await service.provision(sandbox_request())
    assert failure.value.disposition is CreationDisposition.DEFINITELY_CREATED
    assert (await service.close()).complete and api.sandboxes == {}


# ------------------------------------------------------------------- images


async def test_an_existing_image_is_adopted_and_owned_by_no_one(service, api):
    api.images.add(IMAGE)
    await service.provision(sandbox_request(image=True))
    report = await service.close()
    assert called(api, "build") == [] and len(report.results) == 1  # the sandbox only
    assert IMAGE in api.images


@pytest.mark.parametrize(
    "blip",
    [StatusError(502), StatusError(429), httpx.ConnectError("no route")],
    ids=["bad_gateway", "rate_limited", "unreachable"],
)
async def test_an_image_lookup_that_fails_for_now_is_asked_again_not_rebuilt(service, api, blip):
    api.images.add(IMAGE)
    api.image_errors = [blip, blip]
    await service.provision(sandbox_request(image=True))
    assert called(api, "build") == []  # a rebuild that fails leaves :latest unstartable


async def test_an_image_lookup_with_no_answer_fails_the_sandbox_without_a_build(service, api):
    api.image_errors = [ValueError("not json")]
    with pytest.raises(ProvisioningFailure, match="not json"):
        await service.provision(sandbox_request(image=True))
    assert called(api, "build") == [] and api.sandboxes == {}


async def test_an_image_is_built_once_for_concurrent_sandboxes_then_its_builder_released(
    registry, api
):
    api.build_gate = asyncio.Event()
    services = [episode(registry, f"attempt-{n}") for n in range(3)]
    starting = asyncio.gather(
        *(
            service.provision(
                sandbox_request(key=f"t{n}__env", name=f"harbor-env-t{n}-env-abc123", image=True)
            )
            for n, service in enumerate(services)
        )
    )
    await asyncio.sleep(0.01)
    api.build_gate.set()
    handles = await starting
    assert called(api, "build") == [IMAGE] and called(api, "delete") == [IMAGE]
    assert api.build_labels[0]["owner"] == "team-a"  # [providers.blaxel] labels
    assert api.build_labels[0]["loops.run_id"] == "run-1"
    assert IMAGE in api.images  # the image outlives its builder
    assert all(handle.intent.request.depends_on == () for handle in handles)
    for service in services:
        assert (await service.close()).complete
    assert api.sandboxes == {}


async def test_one_image_whose_builds_differ_only_in_source_paths_and_memory_is_built_once(
    registry, api
):
    def request(n):
        made = sandbox_request(key=f"t{n}__env", name=f"harbor-env-t{n}-env-abc123", image=True)
        image = dict(made.specification["image"])
        image["build"] = {
            **image["build"],
            "local_files": [
                {
                    "source_path": f"/tasks/t{n}/app",
                    "destination_path": "/app",
                    "context_name": "app",
                }
            ],
            "memory": 4096 * (n + 1),
        }
        return dataclasses.replace(made, specification={**made.specification, "image": image})

    api.build_gate = asyncio.Event()
    services = [episode(registry, f"attempt-{n}") for n in range(2)]
    starting = asyncio.gather(
        *(service.provision(request(n)) for n, service in enumerate(services))
    )
    await asyncio.sleep(0.01)
    api.build_gate.set()
    await starting
    assert called(api, "build") == [IMAGE]
    for service in services:
        assert (await service.close()).complete


@pytest.mark.parametrize(
    ("sizes", "sent"),
    [
        ({}, (None, None)),
        ({"image_build_memory_mb": 16384, "image_build_volume_mb": 65536}, (16384, 65536)),
    ],
    ids=["blaxels_defaults", "sized"],
)
async def test_an_image_is_built_with_the_configured_build_memory_and_disk(
    tmp_path, api, sizes, sent
):
    adapter = BlaxelAdapter(SETTINGS.model_copy(update=sizes), CREDENTIALS, api, sleep=no_wait)
    registry = ResourceRegistry.open(
        tmp_path, {"blaxel": adapter}, mode="write", run_id="run-1", kind=local, sleep=no_wait
    )
    try:
        await episode(registry).provision(sandbox_request(image=True))
        assert api.build_sizes == [sent]
    finally:
        await registry.close()


async def test_a_forced_build_rebuilds_an_existing_image(service, api):
    api.images.add(IMAGE)
    await service.provision(sandbox_request(image=True, force_build=True))
    assert called(api, "build") == [IMAGE]


async def test_a_build_that_loses_a_race_uses_the_other_owners_image_and_leaves_its_builder(
    service, api
):
    api.build_errors, api.built_elsewhere = [RuntimeError("builder already exists")], True
    api.image_errors = [None, StatusError(502)]  # the wait for its image survives a blip
    await service.provision(sandbox_request(image=True))
    assert IMAGE in api.images and IMAGE in api.sandboxes
    assert IMAGE not in called(api, "delete")  # the other owner's builder is theirs


async def test_a_build_blaxel_turns_away_before_its_builder_exists_is_sent_again(service, api):
    api.build_refusals = [StatusError(429)]
    await service.provision(sandbox_request(image=True))
    assert called(api, "build") == [IMAGE, IMAGE]
    assert IMAGE in api.images and SANDBOX in api.sandboxes


async def test_a_rate_limited_build_beside_another_owners_builder_uses_its_image(service, api):
    api.build_errors, api.built_elsewhere = [StatusError(429)], True
    await service.provision(sandbox_request(image=True))
    assert called(api, "build") == [IMAGE]  # not sent again over the other owner's build
    assert IMAGE in api.images and IMAGE not in called(api, "delete")


async def test_a_build_of_ours_that_fails_releases_its_builder_and_fails(service, api):
    api.build_errors = [RuntimeError("build did not finish")]
    with pytest.raises(ProvisioningFailure, match="build did not finish"):
        await service.provision(sandbox_request(image=True))
    assert IMAGE not in api.sandboxes and IMAGE not in api.images


async def test_a_build_that_times_out_is_a_failure_not_a_found_image(tmp_path, api):
    tight = SETTINGS.model_copy(update={"image_build_timeout_seconds": 0.05})
    now = [0.0]
    registry = ResourceRegistry.open(
        tmp_path / "tight",
        {"blaxel": BlaxelAdapter(tight, CREDENTIALS, api, sleep=no_wait)},
        mode="write",
        run_id="run-1",
        kind=local,
        monotonic=lambda: now[0],
    )
    api.build_gate = asyncio.Event()  # the builder exists; the build never finishes
    service = episode(registry, "attempt-9")
    with pytest.raises(ProvisioningFailure, match="not built"):
        await service.provision(sandbox_request(image=True))
    assert IMAGE not in api.sandboxes  # the builder was ours, and is released
    api.build_gate.set()
    api.build_gate = None
    now[0] += FAILED_HELPER_COOLDOWN_SECONDS  # past the failed build's cooldown
    await service.provision(sandbox_request(key="t2__env", name="harbor-env-t2-abc123", image=True))
    assert called(api, "build") == [IMAGE, IMAGE]  # the failure was not cached for good
    await registry.close()


@pytest.mark.parametrize(
    "blip",
    [StatusError(502), StatusError(429), httpx.ConnectError("no route")],
    ids=["bad_gateway", "rate_limited", "unreachable"],
)
async def test_an_ambiguous_build_whose_image_lookup_fails_for_now_is_asked_again(
    adapter, api, blip
):
    labels = {"loops.resource_id": "build-1"}
    request = ResourceRequest(
        provider="blaxel", kind=IMAGE_BUILD, logical_key=f"image-build:{IMAGE}", specification={}
    )
    intent = ResourceIntent("build-1", ResourceScope.RUN, "run-1", request, "d", IMAGE, labels)
    api.sandboxes[IMAGE] = Sandbox(IMAGE, dict(labels), ["DEPLOYED"])
    api.images.add(IMAGE)
    api.image_errors = [blip, blip]
    found = await adapter.reconcile_create(intent)
    assert found.status is ReconcileStatus.FOUND and found.unusable is None


async def test_an_unreadable_builder_is_not_taken_to_be_someone_elses(service, api):
    api.build_errors, api.built_elsewhere = [RuntimeError("builder already exists")], True
    api.get_errors = [StatusError(503)]
    with pytest.raises(ProvisioningFailure):
        await service.provision(sandbox_request(image=True))
    assert IMAGE not in called(api, "delete")


# ------------------------------------------------------------------ release


@pytest.mark.parametrize("transient", [[], [StatusError(503)]], ids=["clean", "503_first"])
async def test_a_release_is_complete_only_once_the_name_answers_404(service, api, transient):
    await service.provision(sandbox_request())
    api.linger = 2  # the delete is accepted; the sandbox lingers as DELETING
    original = api.get

    async def get(name):
        if transient:
            raise transient.pop(0)  # a transient lookup failure proves nothing
        api.tick()
        return await original(name)

    api.get = get
    [result] = (await service.close()).results
    assert result.status is CleanupStatus.RELEASED and api.sandboxes == {}
    assert "404 after its delete" in (result.detail or "")


@pytest.mark.parametrize(
    ("deleted", "owner", "absent"),
    [
        (True, "ours", True),
        (False, "ours", False),  # terminated, but not by our delete
        (True, "someone-else", False),  # the name is another sandbox's now
    ],
    ids=["ours_after_delete", "ours_without_delete", "someone_elses"],
)
async def test_terminated_proves_absence_only_for_our_sandbox_after_our_delete(
    service, adapter, api, deleted, owner, absent
):
    handle = await service.provision(sandbox_request())
    ours = handle.resource.labels["loops.resource_id"]
    labels = {"loops.resource_id": ours if owner == "ours" else owner}
    if deleted:
        api.terminated = True  # observed Blaxel: the delete leaves it readable, TERMINATED
        await adapter.release(handle.resource)
    api.sandboxes[SANDBOX] = Sandbox(SANDBOX, labels, ["TERMINATED"])
    try:
        async with asyncio.timeout(0.2):
            found = await adapter.confirm_absent(handle.resource)
    except TimeoutError:
        found = None  # still waiting: absence is not proven
    assert (found is not None and found.status is ReconcileStatus.PROVEN_ABSENT) is absent
    if absent:
        assert "observed, not documented" in (found.evidence or "")
        await api.create({"name": SANDBOX, "labels": {}})  # the name is free again


async def test_a_name_another_owner_now_holds_does_not_prove_ours_gone(service, api):
    await service.provision(sandbox_request())
    api.sandboxes[SANDBOX] = Sandbox(SANDBOX, {"loops.resource_id": "someone-else"}, ["DEPLOYED"])
    api.delete = lambda name: asyncio.sleep(0)  # accepted, yet the name stays taken
    [result] = (await service.close()).results
    assert result.status is CleanupStatus.UNCONFIRMED and SANDBOX in api.sandboxes


async def test_list_owned_returns_only_sandboxes_with_every_label_page_by_page(adapter, api):
    api.list_page_size = 2
    api.sandboxes["a"] = Sandbox("a", {"loops.run_id": "run-1", "loops.scope": "run"}, ["DEPLOYED"])
    api.sandboxes["b"] = Sandbox("b", {"loops.run_id": "run-1"}, ["DEPLOYED"])
    api.sandboxes["c"] = Sandbox("c", {"loops.run_id": "run-2", "loops.scope": "run"}, ["DEPLOYED"])
    api.sandboxes["d"] = Sandbox("d", {"loops.run_id": "run-1", "loops.scope": "run"}, ["DEPLOYED"])
    pages = [
        [resource.name for resource in page]
        async for page in adapter.list_owned({"loops.run_id": "run-1", "loops.scope": "run"})
    ]
    assert pages == [["a"], ["d"]]


# ------------------------------------------------------- an Episode's sandbox


def lost(api, how):
    """What becomes of the sandbox, as Blaxel shows it afterwards."""
    sandbox = api.sandboxes[SANDBOX]
    if how == "deleted":
        del api.sandboxes[SANDBOX]
    elif how == "reused":  # deleted, and its name taken by a sandbox that is not ours
        sandbox.labels = {**sandbox.labels, "loops.resource_id": "someone-else"}
    else:
        sandbox.statuses = [how]


@pytest.mark.parametrize("how", ["deleted", "reused", "TERMINATED", "FAILED", "DELETING"])
async def test_a_sandbox_is_alive_only_while_it_is_ours_and_deployed(api, service, how):
    handle = await service.provision(sandbox_request())
    assert await service.sandbox_alive(handle.intent.resource_id) is True
    lost(api, how)
    assert await service.sandbox_alive(handle.intent.resource_id) is False


async def test_a_liveness_blaxel_cannot_answer_is_raised_not_guessed(api, service):
    handle = await service.provision(sandbox_request())
    api.get_errors.append(StatusError(503, "unavailable"))
    with pytest.raises(StatusError, match="unavailable"):
        await service.sandbox_alive(handle.intent.resource_id)


async def test_the_client_is_the_sdks_own_for_the_sandbox_and_closing_it_closes_its_pools(
    api, service, monkeypatch
):
    handle = await service.provision(sandbox_request())
    reads = []

    async def get(name):
        observed = api.sandboxes[name].read()
        reads.append(sdk_sandbox(name, observed.labels, observed.status))
        return reads[-1]

    monkeypatch.setattr(usage, "SandboxInstance", NS(get=get))
    client = await service.sandbox_client(handle.intent.resource_id)
    assert reads == [client]
    pools = client.fs.get_client(), client.process.get_client()
    await service.close_sandbox_client(handle.intent.resource_id, client)
    assert all(pool.is_closed for pool in pools)
