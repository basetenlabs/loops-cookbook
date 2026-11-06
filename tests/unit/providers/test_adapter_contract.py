"""What every sandbox provider's adapter does through the real registry, held once for each.

Each provider runs against its own fake control plane. ``Rig`` names what differs
between them: how a request is spelled, how a sandbox is planted, where it is stored,
and which errors the provider's SDK raises.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from support.fakes import no_wait
from support.resources import episode, local

from loops_cookbook.resources.api import (
    CleanupStatus,
    CreationDisposition,
    ProvisioningFailure,
    ReconcileStatus,
    ResourceIntent,
    ResourceRequest,
    ResourceScope,
)
from loops_cookbook.resources.cleanup import sweep_labels
from loops_cookbook.resources.registry import ResourceRegistry


@dataclass
class Rig:
    provider: str
    api: Any
    adapter: Any
    request: Callable[..., ResourceRequest]  # request(labels=None)
    # A sandbox owned by `labels`, already in the provider: (its key in api.sandboxes, it).
    sandbox: Callable[[str, dict[str, str]], tuple[str, Any]]
    key: Callable[[Any], str]  # a resource's key in api.sandboxes
    # Another owner's sandbox under the name the request asks for, and that request.
    rival: Callable[[Any], ResourceRequest]
    errors: dict[str, BaseException]  # by name: turned_away_*, rejected_*
    failures: dict[str, tuple[dict[str, Any], CreationDisposition]]  # api setup, disposition


def blaxel() -> Rig:
    pytest.importorskip("blaxel")
    import httpx
    from support.blaxel_api import CREDENTIALS, SETTINGS, FakeApi, Sandbox, StatusError
    from support.blaxel_api import sandbox_request as request

    from loops_cookbook.providers.blaxel.adapter import BlaxelAdapter

    api = FakeApi()
    name = request().specification["sandbox"]["name"]

    def labelled(labels=None):
        made = request()
        if labels:
            made.specification["sandbox"]["labels"] = labels
        return made

    def rival(api):
        api.sandboxes[name] = Sandbox(name, {"loops.resource_id": "not-ours"}, ["DEPLOYED"])
        return request()

    return Rig(
        provider="blaxel",
        api=api,
        adapter=BlaxelAdapter(SETTINGS, CREDENTIALS, api, sleep=no_wait),
        request=labelled,
        sandbox=lambda name, labels: (name, Sandbox(name, labels, ["DEPLOYED"])),
        key=lambda resource: resource.name,
        rival=rival,
        errors={
            "turned_away_rate_limited": StatusError(429),
            "turned_away_never_dispatched": httpx.ConnectError("no route"),
            "rejected_400": StatusError(400),
            "rejected_401": StatusError(401),
            "rejected_403": StatusError(403),
        },
        failures={
            "refused": (
                {"create_errors": [StatusError(403)]},
                CreationDisposition.DEFINITELY_NOT_CREATED,
            ),
            "deployed_dead": (
                {"deploy": ["DEPLOYING", "FAILED"]},
                CreationDisposition.DEFINITELY_CREATED,
            ),
        },
    )


def daytona() -> Rig:
    pytest.importorskip("daytona")
    from support.daytona_api import (
        CREDENTIALS,
        SETTINGS,
        FakeApi,
        Sandbox,
        StatusError,
        never_connected,
        sandbox_request,
    )

    from loops_cookbook.providers.daytona.adapter import DaytonaAdapter

    api = FakeApi()

    def rival(api):
        api.sandboxes["theirs"] = Sandbox(
            "theirs", "taken", {"loops.resource_id": "x"}, ["started"]
        )
        return sandbox_request(name="taken")

    return Rig(
        provider="daytona",
        api=api,
        adapter=DaytonaAdapter(SETTINGS, CREDENTIALS, api, sleep=no_wait),
        request=lambda labels=None: sandbox_request(labels=labels),
        sandbox=lambda name, labels: (name, Sandbox(name, name, labels, ["started"])),
        key=lambda resource: resource.provider_id,
        rival=rival,
        errors={
            "turned_away_rate_limited": StatusError(429, "Too many requests"),
            "turned_away_never_dispatched": never_connected(),
            "rejected_per_sandbox_limit": StatusError(
                400, "CPU request 5 exceeds maximum allowed per sandbox (4)."
            ),
            "rejected_unauthorized": StatusError(401, "unauthorized"),
            "rejected_forbidden": StatusError(403, "forbidden: missing write:sandboxes"),
        },
        failures={
            "forbidden": (
                {"create_errors": [StatusError(403)]},
                CreationDisposition.DEFINITELY_NOT_CREATED,
            ),
            "build_failed": (
                {"start": ["creating", "build_failed"]},
                CreationDisposition.DEFINITELY_CREATED,
            ),
            "error": ({"start": ["starting", "error"]}, CreationDisposition.DEFINITELY_CREATED),
        },
    )


RIGS = {"blaxel": blaxel, "daytona": daytona}
TURNED_AWAY = ["turned_away_rate_limited", "turned_away_never_dispatched"]
REJECTED = {
    "blaxel": ["rejected_400", "rejected_401", "rejected_403"],
    "daytona": ["rejected_per_sandbox_limit", "rejected_unauthorized", "rejected_forbidden"],
}
FAILURES = {
    "blaxel": ["refused", "deployed_dead"],
    "daytona": ["forbidden", "build_failed", "error"],
}


def each_provider(cases: dict[str, list[str]] | list[str]):
    """``(provider, case)`` for every provider and each of its cases."""
    return [
        pytest.param(provider, case, id=f"{provider}-{case}")
        for provider in RIGS
        for case in (cases[provider] if isinstance(cases, dict) else cases)
    ]


@pytest.fixture
def rig(provider):
    return RIGS[provider]()


@pytest.fixture
async def registry(tmp_path, rig):
    opened = ResourceRegistry.open(
        tmp_path,
        {rig.provider: rig.adapter},
        mode="write",
        run_id="run-1",
        kind=local,
        sleep=no_wait,
        jitter=lambda: 0.5,  # a first wait of 0.5 s, inside half of each SETTINGS' 2 s create
    )
    yield opened
    await opened.close()


@pytest.fixture
def service(registry):
    return episode(registry)


def called(api, verb):
    return [subject for call, subject in api.calls if call == verb]


@pytest.fixture(params=sorted(RIGS))
def provider(request):
    return request.param


async def test_the_configured_labels_win_over_the_tasks_and_ownership_over_both(rig, service):
    handle = await service.provision(
        rig.request(labels={"owner": "job", "loops.resource_id": "job", "kept": "job"})
    )
    [config] = rig.api.created
    assert config["labels"]["owner"] == "team-a"  # the provider table's labels
    assert config["labels"]["loops.resource_id"] == handle.intent.resource_id
    assert config["labels"]["kept"] == "job"


@pytest.mark.parametrize(("provider", "failure"), each_provider(FAILURES))
async def test_a_failed_create_leaves_nothing_behind(rig, service, failure):
    setup, disposition = rig.failures[failure]
    for field, value in setup.items():
        setattr(rig.api, field, value)
    with pytest.raises(ProvisioningFailure) as failed:
        await service.provision(rig.request())
    assert failed.value.disposition is disposition
    assert (await service.close()).complete and rig.api.sandboxes == {}


@pytest.mark.parametrize(("provider", "error"), each_provider(TURNED_AWAY))
async def test_a_create_the_provider_turns_away_for_now_is_sent_again(rig, service, error):
    rig.api.create_errors = [rig.errors[error]]
    handle = await service.provision(rig.request())
    assert len(called(rig.api, "create")) == 2
    assert list(rig.api.sandboxes) == [rig.key(handle.resource)]


@pytest.mark.parametrize(("provider", "error"), each_provider(REJECTED))
async def test_a_create_the_provider_rejects_is_never_sent_again(rig, service, error):
    rig.api.create_errors = [rig.errors[error]]
    with pytest.raises(ProvisioningFailure) as failed:
        await service.provision(rig.request())
    assert failed.value.disposition is CreationDisposition.DEFINITELY_NOT_CREATED
    assert len(called(rig.api, "create")) == 1


@pytest.mark.parametrize("settled", [True, False], ids=["settled", "not_settled"])
async def test_a_404_proves_a_create_absent_only_once_it_is_settled(rig, settled):
    labels = {"loops.resource_id": "resource-1"}
    intent = ResourceIntent(
        "resource-1", ResourceScope.EPISODE, "attempt-1", rig.request(), "d", "loops-r1", labels
    )
    try:
        async with asyncio.timeout(0.2):  # there is no such sandbox: every read is a 404
            found = await rig.adapter.reconcile_create(intent, settled=settled)
    except TimeoutError:
        found = None  # still waiting: a 404 may be a create not yet readable
    if settled:
        assert found is not None and found.status is ReconcileStatus.PROVEN_ABSENT
        assert "404 for a settled create" in (found.evidence or "")
    else:
        assert found is None


async def test_a_same_named_sandbox_someone_else_owns_is_never_adopted_or_deleted(
    rig, registry, service
):
    request = rig.rival(rig.api)
    planted = set(rig.api.sandboxes)
    with pytest.raises(ProvisioningFailure):
        await service.provision(request)
    await service.close()
    assert set(rig.api.sandboxes) == planted and called(rig.api, "delete") == []
    assert [r.status for r in registry.unresolved.results] == [CleanupStatus.UNCONFIRMED]


async def test_a_sandbox_already_gone_releases_cleanly(rig, service):
    handle = await service.provision(rig.request())
    del rig.api.sandboxes[rig.key(handle.resource)]  # its lifetime ran out first
    [result] = (await service.close()).results
    assert result.status is CleanupStatus.RELEASED


async def test_a_label_sweep_releases_the_runs_sandboxes_and_nothing_else(rig, service):
    await service.provision(rig.request())
    key, other = rig.sandbox("other", {"loops.run_id": "run-2"})
    rig.api.sandboxes[key] = other
    report = await sweep_labels({rig.provider: rig.adapter}, "run-1")
    assert CleanupStatus.RELEASED in {result.status for result in report.results}
    assert list(rig.api.sandboxes) == ["other"]
