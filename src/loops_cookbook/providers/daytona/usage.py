"""Reach a Daytona sandbox that the run's registry already owns.

Nothing here creates or deletes a sandbox. A caller passes the ``ProviderResource``
that its handle carries, and a Daytona client. It gets back the SDK's sandbox for
exactly that sandbox, or a failure. The adapter's ``client`` reaches a sandbox this way,
with its own Daytona client, for an Episode and for Harbor's environment alike.

A sandbox is usable only while it is ``started``. The ``started`` function holds that
rule, and the adapter's provisioning, its liveness check and ``attach`` all use it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

from loops_cookbook.resources.api import CreationDisposition, ProviderResource, ProvisioningFailure

if TYPE_CHECKING:
    from daytona import AsyncSandbox

STARTED = "started"
# The state of a deleted sandbox, until Daytona answers 404 for it.
DESTROYED = "destroyed"
# States a sandbox never leaves for ``started`` on its own. A stopped sandbox is gone
# for the run, because an ephemeral sandbox is deleted as it stops.
DEAD = frozenset(
    {
        "error",
        "build_failed",
        DESTROYED,
        "destroying",
        "stopped",
        "stopping",
        "archived",
        "archiving",
    }
)
POLL_SECONDS = 1.0
SANDBOX_ID = "DAYTONA_SANDBOX_ID"


class Client(Protocol):
    async def get(self, sandbox_id_or_name: str) -> Any: ...


def started(state: object) -> bool:
    """Return whether a sandbox in ``state``, Daytona's wire value, is usable."""
    return state == STARTED


def state_of(sandbox: AsyncSandbox) -> str:
    """Return an SDK sandbox's state as Daytona's wire value."""
    return "unknown" if sandbox.state is None else sandbox.state.value


async def attach(
    resource: ProviderResource,
    client: Client,
    *,
    timeout_seconds: float,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Any:
    """Return the SDK sandbox for the provisioned resource, once it is ``started``.

    ``timeout_seconds``, the provider's attach deadline, bounds the wait. The sandbox must
    carry the ownership labels it was created with.
    """
    async with asyncio.timeout(timeout_seconds):
        while True:
            sandbox = await client.get(resource.provider_id)
            state = state_of(sandbox)
            if started(state):
                break
            if state in DEAD:
                raise ProvisioningFailure(
                    f"Daytona sandbox {resource.name} is {state}",
                    CreationDisposition.DEFINITELY_CREATED,
                    resource,
                )
            await sleep(POLL_SECONDS)
    labels = sandbox.labels or {}
    if any(labels.get(key) != value for key, value in resource.labels.items()):
        raise ProvisioningFailure(
            f"Daytona sandbox {resource.name} does not carry the ownership it was created with",
            CreationDisposition.DEFINITELY_CREATED,
            resource,
        )
    return sandbox
