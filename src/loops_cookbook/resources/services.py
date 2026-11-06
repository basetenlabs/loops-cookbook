"""The scoped resource services runtime code is given: one per attempt, one per run.

Each service is bound to its scope and owner, so runtime code supplies a request
and never a scope, an owner, a digest, or an adapter. An episode can provision and
release its own resources and borrow run resources; it cannot create or release a
run resource.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Self

from pydantic import SecretStr

from loops_cookbook.resources.api import (
    CleanupReport,
    CleanupResult,
    ResourceHandle,
    ResourceRequest,
    ResourceScope,
)
from loops_cookbook.resources.registry import ResourceContractError, ResourceRegistry

# Wraps each call so the caller's clocks leave out resource-service time.
Excluded = Callable[[], AbstractContextManager[None]]


class _Lease:
    def __init__(self, registry: ResourceRegistry, handle: ResourceHandle, holder: str) -> None:
        self._registry = registry
        self._handle = handle
        self._holder = holder
        self._closed = False

    @property
    def handle(self) -> ResourceHandle:
        return self._handle

    @property
    def closed(self) -> bool:
        return self._closed

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._registry.unlease(self._handle.intent.resource_id, self._holder)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()


class RunResourceService:
    """``RunResources``: resources that outlive every episode of the run."""

    def __init__(self, registry: ResourceRegistry, run_id: str) -> None:
        self._registry = registry
        self._run_id = run_id
        self._leases: dict[str, _Lease] = {}

    async def provision(self, request: ResourceRequest) -> ResourceHandle:
        return await self._registry.provision(
            request, scope=ResourceScope.RUN, owner_id=self._run_id
        )

    async def acquire(self, resource_id: str) -> _Lease:
        held = self._leases.get(resource_id)
        if held is not None and not held.closed:
            raise ResourceContractError(f"the run already holds a lease on {resource_id}")
        handle = self._registry.lease(resource_id, self._run_id)
        lease = _Lease(self._registry, handle, self._run_id)
        self._leases[resource_id] = lease
        return lease

    async def close(self) -> CleanupReport:
        """Shutdown step 6: end the run's leases, then release the run scope."""
        for lease in list(self._leases.values()):
            await lease.close()
        return await self._registry.release_scope(ResourceScope.RUN, self._run_id)


class EpisodeResourceService:
    """``EpisodeResources``: one attempt's own resources and the run resources it borrows."""

    def __init__(self, registry: ResourceRegistry, attempt_id: str, excluded: Excluded) -> None:
        self._registry = registry
        self._attempt_id = attempt_id
        self._excluded = excluded
        self._owned: set[str] = set()
        self._leases: dict[str, _Lease] = {}

    async def provision(self, request: ResourceRequest) -> ResourceHandle:
        with self._excluded():
            handle = await self._registry.provision(
                request, scope=ResourceScope.EPISODE, owner_id=self._attempt_id
            )
        self._owned.add(handle.intent.resource_id)
        return handle

    async def acquire(self, resource_id: str) -> _Lease:
        if resource_id in self._owned:
            raise ResourceContractError(f"{resource_id} is this attempt's own; it needs no lease")
        held = self._leases.get(resource_id)
        if held is not None and not held.closed:
            raise ResourceContractError(f"this attempt already holds a lease on {resource_id}")
        handle = self._registry.lease(resource_id, self._attempt_id)
        lease = _Lease(self._registry, handle, self._attempt_id)
        self._leases[resource_id] = lease
        return lease

    async def release(self, resource_id: str) -> CleanupResult:
        """End a resource early; return the registry's proof, including unresolved results."""
        if resource_id not in self._owned:
            raise ResourceContractError(f"{resource_id} is not an episode resource of this attempt")
        with self._excluded():
            return await self._registry.release(resource_id)

    def usage_environment(self, resource_id: str) -> Mapping[str, str | SecretStr]:
        self._check_reachable(resource_id)
        return self._registry.usage_environment(resource_id)

    async def sandbox_client(self, resource_id: str) -> object:
        self._check_reachable(resource_id)
        return await self._registry.sandbox_client(resource_id)

    async def close_sandbox_client(self, resource_id: str, client: object) -> None:
        self._check_reachable(resource_id)
        await self._registry.close_sandbox_client(resource_id, client)

    async def sandbox_alive(self, resource_id: str) -> bool:
        self._check_reachable(resource_id)
        return await self._registry.sandbox_alive(resource_id)

    def _check_reachable(self, resource_id: str) -> None:
        lease = self._leases.get(resource_id)
        if resource_id not in self._owned and (lease is None or lease.closed):
            raise ResourceContractError(
                f"{resource_id} is neither owned nor leased by this attempt"
            )

    async def close(self) -> CleanupReport:
        """End the episode: close its leases and release what it still owns."""
        with self._excluded():
            for lease in list(self._leases.values()):
                await lease.close()
            return await self._registry.release_scope(ResourceScope.EPISODE, self._attempt_id)
