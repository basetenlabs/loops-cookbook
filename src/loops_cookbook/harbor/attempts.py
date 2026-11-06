"""The driver's attempt directory, as the Harbor-facing environments see it.

Harbor constructs environments itself, passing only ``environment.kwargs``; the
Harbor adapter writes the attempt's ID there as ``loops_attempt_id`` and registers
the attempt's ``EpisodeResources`` and the run's provider deadlines and provider
settings here while it holds the attempt's trial. The directory is private to the run and holds
no adapter, so an environment reaches providers only through its resources. It
imports no provider, so a job without one needs no provider extra. Every provider's
environment releases its sandbox on stop through ``release_sandbox``, so one rule
decides when a stop failed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import NoReturn

from pydantic import BaseModel

from loops_cookbook.resources.api import (
    CleanupStatus,
    ProviderDeadlines,
    ResourceHandle,
    ResourceRequest,
)
from loops_cookbook.runtime.api import EpisodeResources

ATTEMPT_ID = "loops_attempt_id"


class SandboxNotReleased(RuntimeError):
    """A stop's release could not prove the sandbox gone; the registry still holds it."""


@dataclass(frozen=True, slots=True)
class Attempt:
    """What an environment of the attempt may use: its resources, their deadlines, and
    each provider's validated table, which only that provider's environment reads."""

    resources: EpisodeResources
    provider_deadlines: Mapping[str, ProviderDeadlines]
    provider_settings: Mapping[str, BaseModel] = field(default_factory=dict)


_ATTEMPTS: dict[str, Attempt] = {}


def bind(
    attempt_id: str,
    resources: EpisodeResources,
    provider_deadlines: Mapping[str, ProviderDeadlines],
    provider_settings: Mapping[str, BaseModel] | None = None,
) -> None:
    _ATTEMPTS[attempt_id] = Attempt(resources, provider_deadlines, provider_settings or {})


def forget(attempt_id: str) -> None:
    _ATTEMPTS.pop(attempt_id, None)


async def release_sandbox(resources: EpisodeResources, handle: ResourceHandle) -> None:
    """Release a stopped environment's sandbox; raise ``SandboxNotReleased`` unless the
    release proves it gone. The registry keeps an unproven one for the attempt's cleanup."""
    result = await resources.release(handle.intent.resource_id)
    if result.status not in (CleanupStatus.RELEASED, CleanupStatus.ALREADY_ABSENT):
        raise SandboxNotReleased(
            f"{handle.intent.request.provider} sandbox {handle.resource.name} is not proven "
            f"released ({result.status.value}: {result.detail}); the attempt's cleanup tries "
            "again"
        )


def attempt_for(attempt_id: object) -> Attempt:
    """The live attempt, or a failure naming why there is none."""
    if not isinstance(attempt_id, str) or not attempt_id:
        raise TypeError(f"{ATTEMPT_ID} must be the attempt ID the Harbor adapter wrote")
    attempt = _ATTEMPTS.get(attempt_id)
    if attempt is None:
        raise RuntimeError(
            f"no live attempt {attempt_id!r}; this environment is created only by the "
            "cookbook's Harbor adapter"
        )
    return attempt


class _Preflight:
    """The resources of a trial Harbor builds only to read its deadlines: none at all."""

    async def provision(self, request: ResourceRequest) -> NoReturn:
        _refuse()

    async def acquire(self, resource_id: str) -> NoReturn:
        _refuse()

    async def release(self, resource_id: str) -> NoReturn:
        _refuse()

    def usage_environment(self, resource_id: str) -> NoReturn:
        _refuse()

    async def sandbox_client(self, resource_id: str) -> NoReturn:
        _refuse()

    async def close_sandbox_client(self, resource_id: str, client: object) -> NoReturn:
        _refuse()

    async def sandbox_alive(self, resource_id: str) -> NoReturn:
        _refuse()


def _refuse() -> NoReturn:
    raise RuntimeError("a preflight trial provisions nothing")


PREFLIGHT: EpisodeResources = _Preflight()
