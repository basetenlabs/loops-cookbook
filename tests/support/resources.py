"""A fake second provider and resource-service helpers, for the resource contract tests.

``FakeProvider`` is a real ``ResourceAdapter`` over an in-memory store. Switches
make it block, fail, leave a create ambiguous, lose track of absence, or declare
prerequisites, so every boundary of the registry can be exercised without a
provider.
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
import time
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import SecretStr

from loops_cookbook.resources.api import (
    CreationDisposition,
    ProviderDeadlines,
    ProviderResource,
    ProvisioningFailure,
    ReconcileResult,
    ReconcileStatus,
    ResourceIntent,
    ResourceRequest,
    SandboxKind,
    SandboxSpec,
)
from loops_cookbook.resources.registry import ResourceRegistry
from loops_cookbook.resources.services import EpisodeResourceService, RunResourceService

FAST = ProviderDeadlines(
    create_seconds=2,
    reconcile_seconds=2,
    release_seconds=2,
    confirm_absent_seconds=2,
    list_seconds=2,
    attach_seconds=2,
    create_seconds_by_kind={"image_build": 3},
)


def local(path: Path) -> str:
    return "ext4"


# Every FakeProvider a test built: an autouse fixture fails a test that closed one
# under a call.
LIVE: list[FakeProvider] = []


@dataclass
class FakeProvider:
    """Resources live in ``store`` by provider id; every call is recorded."""

    name: str = "fake"
    deadlines: ProviderDeadlines = FAST
    store: dict[str, ProviderResource] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)
    created: list[ResourceRequest] = field(default_factory=list)  # each create's request
    in_flight: int = 0  # provider I/O calls, which concurrency limits count
    most_in_flight: int = 0
    busy: int = 0  # every call under way, I/O or not, which a close must wait for
    gate: asyncio.Event | None = None  # when set, provider I/O waits for it
    delay: float = 0.0  # seconds every provider call takes
    blocked: set[str] = field(default_factory=set)  # operations that wait for `unblocked`
    unblocked: asyncio.Event = field(default_factory=asyncio.Event)
    create_errors: list[BaseException] = field(default_factory=list)
    ambiguous_after_create: bool = False  # the create happens, then the call fails
    release_errors: list[BaseException] = field(default_factory=list)
    hide_absence: bool = False  # confirm_absent cannot prove anything
    stubborn: set[str] = field(default_factory=set)  # names whose release always fails
    needs: dict[str, list[ResourceRequest]] = field(default_factory=dict)  # kind -> prerequisites
    closed: int = 0
    # Closes made with calls still in flight: never right, so every test fails on one.
    closed_under: list[str] = field(default_factory=list)
    # Calls made once closed. A test that reuses its provider for a later run, as a
    # new process would build anew, makes these legitimately; one that does not, must not.
    used_after_close: list[str] = field(default_factory=list)
    list_page_size: int = 0  # resources per listed page; 0 lists them in one page
    list_delay: float = 0.0  # seconds each listed page takes

    def __post_init__(self) -> None:
        LIVE.append(self)

    def _check_open(self, operation: str) -> None:
        if self.closed:
            self.used_after_close.append(operation)

    @contextlib.asynccontextmanager
    async def _io(self, operation: str, subject: str):
        self._check_open(operation)
        self.calls.append((operation, subject))
        self.in_flight += 1
        self.busy += 1
        self.most_in_flight = max(self.most_in_flight, self.in_flight)
        try:
            if self.gate is not None:
                await self.gate.wait()
            if self.delay:
                await asyncio.sleep(self.delay)
            if operation in self.blocked:
                await self.unblocked.wait()
            yield
        finally:
            self.in_flight -= 1
            self.busy -= 1
            self._check_open(operation)

    async def _unless_blocked(self, operation: str) -> None:
        self._check_open(operation)
        self.busy += 1
        try:
            if operation in self.blocked:
                await self.unblocked.wait()
        finally:
            self.busy -= 1
            self._check_open(operation)

    async def prerequisites(self, request: ResourceRequest) -> Sequence[ResourceRequest]:
        await self._unless_blocked("prerequisites")
        return list(self.needs.get(request.kind, ()))

    def desired_name(self, request: ResourceRequest, resource_id: str) -> str:
        return f"{self.name}-{request.logical_key}"

    async def create(self, intent: ResourceIntent) -> ProviderResource:
        async with self._io("create", intent.desired_name):
            self.created.append(intent.request)
            if self.create_errors:
                raise self.create_errors.pop(0)
            resource = ProviderResource(
                provider_id=f"id-{intent.resource_id}",
                name=intent.desired_name,
                labels=dict(intent.ownership_labels),
            )
            self.store[resource.provider_id] = resource
            if self.ambiguous_after_create:
                self.ambiguous_after_create = False
                raise ConnectionError("the response was lost after the create")
            return resource

    async def reconcile_create(
        self, intent: ResourceIntent, *, settled: bool = False
    ) -> ReconcileResult:
        async with self._io("reconcile", intent.desired_name):
            for resource in self.store.values():
                if resource.name == intent.desired_name:
                    return ReconcileResult(ReconcileStatus.FOUND, resource)
            if self.hide_absence and not settled:  # a lookup proves nothing until settled
                return ReconcileResult(ReconcileStatus.UNKNOWN)
            return ReconcileResult(
                ReconcileStatus.PROVEN_ABSENT, evidence="authoritative lookup: 404"
            )

    async def validate_ownership(self, intent: ResourceIntent, resource: ProviderResource) -> bool:
        await self._unless_blocked("validate")
        return resource.labels.get("loops.resource_id") == intent.resource_id

    async def release(self, resource: ProviderResource) -> None:
        async with self._io("release", resource.name):
            if self.release_errors:
                raise self.release_errors.pop(0)
            if resource.name in self.stubborn:
                raise ConnectionError(f"{resource.name} will not go")
            self.store.pop(resource.provider_id, None)

    async def confirm_absent(self, resource: ProviderResource) -> ReconcileResult:
        async with self._io("confirm_absent", resource.name):
            if resource.provider_id in self.store:
                return ReconcileResult(ReconcileStatus.FOUND, resource)
            if self.hide_absence:
                return ReconcileResult(ReconcileStatus.UNKNOWN)
            return ReconcileResult(
                ReconcileStatus.PROVEN_ABSENT, evidence="authoritative lookup: 404"
            )

    async def list_owned(
        self, labels: Mapping[str, str]
    ) -> AsyncGenerator[Sequence[ProviderResource], None]:
        matching = [
            resource
            for resource in self.store.values()
            if all(resource.labels.get(k) == v for k, v in labels.items())
        ]
        size = self.list_page_size or max(len(matching), 1)
        for start in range(0, max(len(matching), 1), size):
            async with self._io("list", ",".join(sorted(labels.values()))):
                await asyncio.sleep(self.list_delay)
                page = matching[start : start + size]
            yield page

    def usage_environment(self, resource: ProviderResource) -> Mapping[str, str | SecretStr]:
        return {"FAKE_NAME": resource.name, "FAKE_KEY": SecretStr("secret")}

    async def close(self) -> None:
        if self.busy:
            self.closed_under.append(f"closed with {self.busy} call(s) under way")
        self.closed += 1


@dataclass
class FakeClient:
    """A sandbox's client, as ``FakeSandboxProvider`` gives one."""

    name: str
    closed: int = 0  # how often it was closed


class FakeSpec(SandboxSpec):
    """``[rollout_runtime.config.sandbox]`` for a ``Fake`` sandbox: a ``box`` with an image."""

    resource_kind: ClassVar[str] = "box"

    image: str = "fake-image"


class Fake(SandboxKind[FakeClient]):
    """The sandbox kind of ``FakeSandboxProvider``, configured as ``[providers.fake]``."""

    provider = "fake"
    spec = FakeSpec


@dataclass
class FakeSandboxProvider(FakeProvider):
    """A ``FakeProvider`` that is also a ``SandboxProvider``: a new ``FakeClient`` per
    ``client`` call, and alive while its resource is in the store and not ``dead``."""

    sandbox_kind: ClassVar[type[Fake]] = Fake
    clients: list[FakeClient] = field(default_factory=list)
    dead: set[str] = field(default_factory=set)  # names alive() reports gone
    closing: Any = None  # a support.cancellation Gate that close_client passes, when set

    async def client(self, resource: ProviderResource) -> FakeClient:
        async with self._io("client", resource.name):
            self.clients.append(FakeClient(resource.name))
            return self.clients[-1]

    async def close_client(self, client: FakeClient) -> None:
        async with self._io("close_client", client.name):
            if self.closing is not None:
                await self.closing.passage()
            client.closed += 1

    async def alive(self, resource: ProviderResource) -> bool:
        async with self._io("alive", resource.name):
            return resource.provider_id in self.store and resource.name not in self.dead


def not_created(message: str = "refused") -> ProvisioningFailure:
    return ProvisioningFailure(message, CreationDisposition.DEFINITELY_NOT_CREATED)


def refused(message: str = "rate limited") -> ProvisioningFailure:
    """A create the provider turned away for now, as a rate limit does."""
    return ProvisioningFailure(message, CreationDisposition.DEFINITELY_NOT_CREATED, refused=True)


@dataclass
class Backoff:
    """The registry's waits between refused creates, recorded, on a fake clock each wait
    moves on to when it ends; with ``hold`` set, each wait lasts until it is released."""

    waits: list[float] = field(default_factory=list)
    hold: asyncio.Event | None = None
    scale: float = 1.0  # the jitter: 1.0 waits the longest each time
    time: float = 0.0  # the fake monotonic clock, in seconds
    real: bool = False  # each wait takes its seconds, on the real clock

    def now(self) -> float:
        return time.monotonic() if self.real else self.time

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        if self.real:
            await asyncio.sleep(seconds)
            return
        wakes = self.time + seconds
        if self.hold is not None:
            await self.hold.wait()
        else:
            await asyncio.sleep(0)
        self.time = max(self.time, wakes)


def request(key: str = "a", kind: str = "box", **fields: Any) -> ResourceRequest:
    return ResourceRequest(
        provider=fields.pop("provider", "fake"), kind=kind, logical_key=key, **fields
    )


def registry_at(
    state_dir: Path,
    *providers: FakeProvider,
    mode: str = "write",
    limit: int | None = None,
    run_id: str = "run-1",
    clock: Callable[[], int] | None = None,
    backoff: Backoff | None = None,
) -> ResourceRegistry:
    permit = _permit(limit) if limit is not None else None
    options: dict[str, Any] = {"kind": local}
    if clock is not None:
        options["clock"] = clock
    if backoff is not None:
        options["sleep"], options["jitter"] = backoff.sleep, lambda: backoff.scale
        options["monotonic"] = backoff.now
    if permit is not None:
        options["permit"] = permit
        options["create_window"] = limit
    if mode == "write":
        options["run_id"] = run_id
    return ResourceRegistry.open(
        state_dir, {provider.name: provider for provider in providers}, mode=mode, **options
    )


def _permit(limit: int):
    permits = asyncio.Semaphore(limit)

    @contextlib.asynccontextmanager
    async def permit():
        async with permits:
            yield

    return permit


def episode(registry: ResourceRegistry, attempt_id: str = "attempt-1") -> EpisodeResourceService:
    return EpisodeResourceService(registry, attempt_id, contextlib.nullcontext)


def run_resources(registry: ResourceRegistry, run_id: str = "run-1") -> RunResourceService:
    return RunResourceService(registry, run_id)


def scratch_registry(*providers: FakeProvider, run_id: str = "run") -> ResourceRegistry:
    """A writing registry in a fresh directory, for rigs that are not given one."""
    return registry_at(Path(tempfile.mkdtemp(prefix="loops-resources-")), *providers, run_id=run_id)


_EMPTY: list[ResourceRegistry] = []


def no_resources(identity: Any, phases: Any) -> EpisodeResourceService:
    """An attempt's resource service for a runtime that provisions nothing.

    Every such attempt shares one empty registry, with no providers, so rigs that
    never touch resources do not each hold a journal open.
    """
    if not _EMPTY:
        _EMPTY.append(scratch_registry())
    return EpisodeResourceService(_EMPTY[0], identity.attempt_id, phases.resource_operation)


def episodes(registry: ResourceRegistry):
    """A driver ``resources=`` factory over ``registry``, with the driver's clocks."""

    def factory(identity: Any, phases: Any) -> EpisodeResourceService:
        return EpisodeResourceService(registry, identity.attempt_id, phases.resource_operation)

    return factory


# Providers a ``[providers.<name>].adapter`` factory hands out, by ``instance``, so
# a test can reach the adapter a config built.
PROVIDERS: dict[str, FakeProvider] = {}
FACTORY_CALLS: list[dict[str, Any]] = []


def create_fake(settings: Mapping[str, Any]) -> FakeProvider:
    """A ``[providers.<name>].adapter`` factory: ``adapter = "support.resources:create_fake"``."""
    FACTORY_CALLS.append(dict(settings))
    return PROVIDERS[str(settings["instance"])]
