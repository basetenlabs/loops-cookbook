"""Reach a Blaxel sandbox that the run's registry already owns.

Nothing here creates or deletes a sandbox, or holds an adapter. A caller passes the
``ProviderResource`` that a provisioned sandbox's handle carries. It gets back the
SDK's usage client for exactly that sandbox, or a failure. Code outside the registry,
such as Harbor's environment, can therefore use a sandbox without bypassing the
registry's ownership.

A sandbox is usable only while it is ``DEPLOYED``. The ``deployed`` function holds that
rule, and the adapter's provisioning, its liveness check and ``attach`` all use it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from blaxel.core.sandbox import SandboxInstance

from loops_cookbook.providers.blaxel.incidents import DEAD, close_clients, configure_clients
from loops_cookbook.resources.api import CreationDisposition, ProviderResource, ProvisioningFailure

DEPLOYED = "DEPLOYED"
POLL_SECONDS = 2.0


def deployed(status: object) -> bool:
    """Return whether a sandbox in ``status`` is usable.

    Being readable proves nothing, because a deleted sandbox stays readable, as
    ``TERMINATED``, for many minutes.
    """
    return status == DEPLOYED


async def attach(
    resource: ProviderResource,
    *,
    timeout_seconds: float,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> SandboxInstance:
    """Return the usage client for the provisioned sandbox, once it is ``DEPLOYED``.

    ``timeout_seconds``, the provider's attach deadline, bounds the wait. The sandbox must
    carry the ownership labels it was created with. The client has the network deadlines
    installed, and the caller closes it with ``close_clients``.
    """
    sandbox: SandboxInstance | None = None
    try:
        async with asyncio.timeout(timeout_seconds):
            while True:
                sandbox = await SandboxInstance.get(resource.name)
                configure_clients(sandbox)
                status = sandbox.status
                if deployed(status):
                    break
                read, sandbox = sandbox, None
                await close_clients(read)
                if status in DEAD:
                    raise ProvisioningFailure(
                        f"Blaxel sandbox {resource.name} entered {status}",
                        CreationDisposition.DEFINITELY_CREATED,
                        resource,
                    )
                await sleep(POLL_SECONDS)
    except BaseException:
        # A client read just before the deadline still needs closing.
        if sandbox is not None:
            await close_clients(sandbox)
        raise
    observed = sandbox.metadata.to_dict()
    labels = observed.get("labels") or {}
    if observed.get("name") != resource.name or any(
        labels.get(key) != value for key, value in resource.labels.items()
    ):
        await close_clients(sandbox)
        raise ProvisioningFailure(
            f"Blaxel sandbox {resource.name} does not carry the ownership it was created with",
            CreationDisposition.DEFINITELY_CREATED,
            resource,
        )
    return sandbox
