"""Blaxel usage code: the client for a provisioned sandbox, and only for that one."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

pytest.importorskip("blaxel")

from support.fakes import no_wait

from loops_cookbook.providers.blaxel import usage
from loops_cookbook.resources.api import ProviderResource, ProvisioningFailure

from .sdk import sdk_sandbox

OWNED = {"loops.resource_id": "r1", "loops.run_id": "run-1"}
RESOURCE = ProviderResource(provider_id="sb", name="sb", labels=OWNED)


@pytest.fixture
def blaxel(monkeypatch):
    """The SDK's view of one sandbox: each read reports the next status."""
    state = NS(statuses=["DEPLOYED"], labels=dict(OWNED), name="sb", reads=[])

    async def get(name):
        status = state.statuses.pop(0) if len(state.statuses) > 1 else state.statuses[0]
        state.reads.append(sdk_sandbox(state.name, state.labels, status))
        return state.reads[-1]

    monkeypatch.setattr(usage, "SandboxInstance", NS(get=get))
    return state


def closed(sandbox):
    return sandbox.fs.get_client().is_closed and sandbox.process.get_client().is_closed


@pytest.mark.parametrize(
    "statuses", [["DEPLOYED"], ["DEPLOYING", "DEPLOYING", "DEPLOYED"]], ids=["ready", "later"]
)
async def test_attach_gives_the_owned_sandbox_once_deployed_with_network_deadlines(
    blaxel, statuses
):
    blaxel.statuses = statuses
    sandbox = await usage.attach(RESOURCE, timeout_seconds=1, sleep=no_wait)
    assert sandbox is blaxel.reads[-1] and sandbox.status == "DEPLOYED"
    request = sandbox.process.get_client().build_request("POST", "https://sandbox/process")
    request.extensions["timeout"] = {"connect": None}
    for hook in sandbox.process.get_client().event_hooks["request"]:
        await hook(request)
    assert request.extensions["timeout"]["connect"] == 10.0
    assert all(closed(read) for read in blaxel.reads[:-1])  # no client left behind
    await sandbox.process.get_client().aclose()
    await sandbox.fs.get_client().aclose()


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"labels": {**OWNED, "loops.resource_id": "someone-else"}}, "ownership"),
        ({"name": "another"}, "ownership"),
        ({"statuses": ["FAILED"]}, "entered FAILED"),
    ],
    ids=["relabelled", "renamed", "dead"],
)
async def test_attach_refuses_a_sandbox_that_is_not_the_one_provisioned(blaxel, change, match):
    for key, value in change.items():
        setattr(blaxel, key, value)
    with pytest.raises(ProvisioningFailure, match=match):
        await usage.attach(RESOURCE, timeout_seconds=1, sleep=no_wait)
    assert all(closed(read) for read in blaxel.reads)  # a refused client is not left open


async def test_attach_is_bounded_by_its_deadline(blaxel):
    blaxel.statuses = ["DEPLOYING"]
    with pytest.raises(TimeoutError):
        await usage.attach(RESOURCE, timeout_seconds=0.05, sleep=no_wait)
    assert all(closed(read) for read in blaxel.reads)
