"""Track who owns each provider resource, with a state machine over the resource journal.

The state machine works the same for every provider. Every provider resource the
cookbook owns moves through it. Each transition is journaled and fsynced at the point
named below. What the journal already holds then fixes the recovery that a crash at
that point needs.

1. ``intent`` records the immutable creation intent, before any provider call. A crash
   after it leaves an identity whose create may or may not have happened. Recovery
   reconciles it by name and ownership labels.
2. The adapter's ``create`` runs under the provisioning permit and the create
   deadline. A create that the provider refuses for now is sent again within that
   deadline, after a wait that holds no permit. How many of a provider's creates are in
   flight at once follows its refusals, as ``_CreateWindow`` describes.
3. ``created`` records the exact provider identity, or ``create_failed`` records the
   known disposition. A crash between steps 2 and 3 is the same as a crash after step
   1. A create that happened but left the resource unusable is recorded as
   ``created``, so it is released, and the provision still fails.
4. ``reconciled`` records an ambiguous create looked up by identity. ``FOUND`` makes the
   resource owned, ``PROVEN_ABSENT`` closes it, and ``UNKNOWN`` keeps it unresolved. A
   lookup's not-found answer proves nothing, except for a settled create. A settled
   create is one inherited from a stopped process and older than ``SETTLE_SECONDS``.
5. ``leased`` and ``unleased`` record a scope borrowing a run resource. A crash leaves
   the lease recorded, and recovery releases the lease's scope anyway.
6. ``release_deferred`` is recorded while a dependant or lease is live, and then
   ``released`` with its cleanup status. Only ``RELEASED`` and ``ALREADY_ABSENT``
   close an identity. ``UNCONFIRMED`` keeps it for ``loops cleanup``. A crash during a
   release repeats the release, which a provider release must tolerate.

Operations on one resource identity are serialized, and nothing takes a registry-wide
lock across provider calls. The provisioning permit is held only while an operation
talks to the provider, so a caller waiting on another identity or a prerequisite holds
none.

Every operation that uses an adapter or writes the journal runs as tracked work,
through ``_track``. That covers creates, releases, prerequisite lookups and sandbox
usage alike, and it makes closing safe and bounded. ``begin_shutdown`` stops
provisioning. Work that has sent nothing yet stops there, and its create is recorded
as never created. A caller still waiting on a key or a prerequisite gets a
``DEFINITELY_NOT_CREATED`` failure, and no create is started for it. ``close`` then
waits for the tracked work, up to its deadline. Past the deadline the work is cut off,
and each operation is recorded as far as it got. What is left stays in the journal,
unresolved, for ``loops cleanup``. The journal and the adapters close only once no
work is left.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
import logging
import math
import random
import time
import uuid
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

from pydantic import SecretStr

from loops_cookbook.cancellation import outlast
from loops_cookbook.evidence import JsonValue, json_mapping
from loops_cookbook.resources.api import (
    CleanupReport,
    CleanupResult,
    CleanupStatus,
    CreationDisposition,
    ProviderResource,
    ProvisioningFailure,
    ReconcileResult,
    ReconcileStatus,
    ResourceAdapter,
    ResourceHandle,
    ResourceIntent,
    ResourceRequest,
    ResourceScope,
    SandboxProvider,
)
from loops_cookbook.resources.journal import (
    Journal,
    JournalError,
    Record,
    digest,
    filesystem_type,
)

Permit = Callable[[], AbstractAsyncContextManager[None]]

logger = logging.getLogger(__name__)


class _SandboxAdapter(ResourceAdapter, SandboxProvider[Any], Protocol):
    """An adapter that can back an Episode's sandbox."""


OWNERSHIP_LABELS = ("loops.run_id", "loops.scope", "loops.owner_id", "loops.resource_id")
# How long after its intent a create that a stopped process left unresolved counts as
# settled. The time is past any provider's visibility lag, so a not-found lookup then
# means absent.
SETTLE_SECONDS = 600
# The wait before sending a refused create again. It starts at up to 1 s and doubles to
# at most 30 s. Each wait is scaled by a random jitter, because a refusal usually comes
# from a limit that other clients share.
REFUSED_FIRST_WAIT_SECONDS = 1.0
REFUSED_MAX_WAIT_SECONDS = 30.0
# The most creates a provider has in flight at once, and the number it starts with, when
# the registry is not told the provisioning limit.
DEFAULT_CREATE_WINDOW = 4
# How long a closing registry waits, once its deadline has passed and it has cut off
# the creates and releases still in flight, for them and its adapters to stop.
CUT_OFF_GRACE_SECONDS = 5.0
# How long after a transient helper, such as an image build, fails to create a request
# for the same key fails at once with that failure, instead of creating it again.
# Otherwise the requests queued behind a failed build would each rebuild it, one by one.
FAILED_HELPER_COOLDOWN_SECONDS = 90.0


class ResourceContractError(RuntimeError):
    """Raised when a caller uses the resource services outside their contract."""


class _State(str, Enum):
    PENDING = "pending"  # The intent is recorded, and the create's outcome is not known.
    OWNED = "owned"
    AMBIGUOUS = "ambiguous"  # The create may have happened, and is not yet reconciled.
    DEFERRED = "deferred"  # A release was requested while a dependant or lease is live.
    UNCONFIRMED = "unconfirmed"  # Released or lost, but absence is not proven.
    CLOSED = "closed"  # Proven gone, or never created.


@dataclass(slots=True)
class _CreateWindow:
    """Limit how many of one provider's creates may be in flight, meaning sent and not yet answered.

    It works like TCP's congestion window. A refusal usually comes from a limit that the
    provider's other callers share, and the room under it changes with their load. The
    window therefore follows the refusals. It halves on a refusal. It grows by
    ``1 / size`` for each create the provider answers without refusing, which is about
    one per window. It stays between 1 and ``cap``.

    Any answer that is not a refusal, even a failure or a timeout, shows room that the
    provider did not refuse. If only successes widened the window, a provider that
    fails creates would hold it narrow for good. A burst refused together is one
    signal, so only a refusal of a create sent since the last halving halves the window
    again. A fractional size admits the next whole create, so 1.5 means 2 slots. One
    halving of a small window therefore leaves room to recover. Creates wait for a slot
    in the order they first asked for one, so a retry keeps its place ahead of creates
    that arrived after it.
    """

    provider: str
    cap: int
    size: float = 0.0  # Set to ``cap`` at first.
    in_flight: int = 0
    # The number of halvings so far. A create remembers the epoch it was sent in.
    epoch: int = 0
    waiting: list[tuple[int, asyncio.Future[None]]] = field(
        default_factory=list[tuple[int, asyncio.Future[None]]]
    )  # A heap, ordered by place in line.
    refusals: int = 0  # The refusals since the window last left its cap.
    # When the window left its cap, while it is below the cap.
    narrowed_at: float | None = None

    def __post_init__(self) -> None:
        self.size = float(self.cap)

    def wake(self) -> None:
        """Give free slots to the creates longest in line."""
        while self.waiting and self.in_flight < max(1, math.ceil(self.size)):
            _, waiter = heapq.heappop(self.waiting)
            if waiter.done():  # The waiter stopped waiting.
                continue
            self.in_flight += 1
            waiter.set_result(None)


@dataclass(slots=True)
class _Entry:
    intent: ResourceIntent
    state: _State = _State.PENDING
    resource: ProviderResource | None = None
    holders: set[str] = field(default_factory=set[str])
    result: CleanupResult | None = None
    # A release was requested and has not yet deferred or finished.
    releasing: bool = False
    sent: bool = False  # The create may have reached the provider.
    # The caller stopped, so a refused create is not sent again.
    abandoned: bool = False
    written_at_ns: int | None = None  # When this process recorded the intent.
    # When a create that this process inherited, with no recorded outcome, is settled.
    # After that, no create for it can still be in flight. It is None for an intent
    # this process wrote.
    settles_at_ns: int | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def live(self) -> bool:
        return self.state is not _State.CLOSED


def _no_permit() -> AbstractAsyncContextManager[None]:
    return contextlib.nullcontext()


async def close_adapters(adapters: Mapping[str, ResourceAdapter]) -> None:
    """Close every adapter at once, so one that fails or hangs holds up none of the others.

    The first failure or cancellation, in the adapters' order, is raised. Whoever holds
    the adapters when their next owner cannot take them closes them here. That is a
    registry when it closes, or a composition root whose run, registry or sweep never
    started.
    """
    outcomes = await asyncio.gather(
        *(adapter.close() for adapter in adapters.values()), return_exceptions=True
    )
    for outcome in outcomes:
        if isinstance(outcome, BaseException):  # A cancelled close is not a closed one.
            raise outcome


class ResourceRegistry:
    """Own every resource of one run, including its journal, identities and cleanup."""

    def __init__(
        self,
        journal: Journal,
        adapters: Mapping[str, ResourceAdapter],
        *,
        mode: Literal["write", "reconcile"],
        run_id: str | None,
        permit: Permit,
        clock: Callable[[], int] = time.time_ns,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
        monotonic: Callable[[], float] = time.monotonic,
        create_window: int = DEFAULT_CREATE_WINDOW,
    ) -> None:
        if create_window < 1:
            raise ValueError("the create window must be positive")
        self._journal = journal
        self._clock = clock
        self._sleep = sleep
        self._jitter = jitter
        self._monotonic = monotonic
        self._create_window = create_window
        self._windows: dict[str, _CreateWindow] = {}
        self._in_line = itertools.count()  # Each create's place in its window's line.
        self._adapters = dict(adapters)
        self._mode = mode
        self._run_id = run_id
        self._permit = permit
        self._entries: dict[str, _Entry] = {}
        self._keys: dict[tuple[ResourceScope, str, str], str] = {}
        self._key_locks: dict[tuple[ResourceScope, str, str], asyncio.Lock] = {}
        # Helpers that were created and then released.
        self._built: dict[tuple[ResourceScope, str, str], ResourceHandle] = {}
        # Helpers whose create failed, with when and the error's details, until the
        # cooldown ends.
        self._failed_helpers: dict[tuple[ResourceScope, str, str], tuple[float, str]] = {}
        # Every operation that uses an adapter or writes the journal, while it runs.
        self._work: set[asyncio.Task[Any]] = set()
        self._unsent: set[asyncio.Task[Any]] = set()  # Work that has sent nothing yet.
        # Creates in flight, keyed by resource id.
        self._creates: dict[str, asyncio.Future[ProviderResource]] = {}
        # Provisioning has stopped, because the registry is closing.
        self._stopping = False
        self._closed = False
        self._finishing: asyncio.Task[None] | None = None  # The close, once begun.
        # The close deadline passed, so work is cancelled at once.
        self._cutting_off = False
        # The close found no work left, so no new work may start.
        self._idle = False
        # A journal damaged before this process opened it is cleanup-only, and its
        # releases are not recorded. A journal that fails later refuses every append.
        self._cleanup_only = journal.damage is not None
        for record in journal.records:
            self._replay(record)

    @classmethod
    def open(
        cls,
        state_dir: Path,
        adapters: Mapping[str, ResourceAdapter],
        *,
        mode: Literal["write", "reconcile"],
        run_id: str | None = None,
        permit: Permit = _no_permit,
        kind: Callable[[Path], str] = filesystem_type,
        clock: Callable[[], int] = time.time_ns,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
        monotonic: Callable[[], float] = time.monotonic,
        create_window: int = DEFAULT_CREATE_WINDOW,
    ) -> ResourceRegistry:
        """Lock and replay the run's journal.

        ``write`` is the run's mode, and it needs the run's ``run_id``. ``reconcile``
        replays without provisioning, for ``loops cleanup``. Both modes fail while
        another process holds the journal.
        """
        if mode == "write" and not run_id:
            raise ValueError("a writing registry needs the run's run_id")
        journal = Journal.open(state_dir, kind=kind)
        if mode == "write" and journal.damage is not None:
            journal.close()
            raise ResourceContractError(
                f"the resource journal is damaged ({journal.damage}); run `loops cleanup` "
                "to release what it records before starting the run again"
            )
        try:
            return cls(
                journal,
                adapters,
                mode=mode,
                run_id=run_id,
                permit=permit,
                clock=clock,
                sleep=sleep,
                jitter=jitter,
                monotonic=monotonic,
                create_window=create_window,
            )
        except BaseException:
            journal.close()
            raise

    def _record(self, event: str, resource_id: str, payload: Mapping[str, JsonValue]) -> None:
        """Journal one transition. A damaged journal is cleanup-only and takes no records.

        Cleanup of a damaged journal still reconciles and releases what it replayed. A
        rerun repeats that from the same records, which is safe because a provider
        release tolerates repetition.
        """
        if not self._cleanup_only:
            self._journal.append(event, resource_id, payload)

    # ---------------------------------------------------------------- replay

    def _replay(self, record: Record) -> None:
        payload = record.payload
        if record.event == "intent":
            intent = _intent_from(payload)
            # The record is replayed, so a process that the journal lock says has
            # stopped wrote it.
            settles = record.recorded_at_ns + SETTLE_SECONDS * 1_000_000_000
            self._entries[record.resource_id] = _Entry(intent, settles_at_ns=settles)
            key = (intent.scope, intent.owner_id, intent.request.logical_key)
            self._keys[key] = intent.resource_id
            return
        entry = self._entries.get(record.resource_id)
        if entry is None:
            raise JournalError(
                f"record {record.sequence} names {record.resource_id}, which has no intent"
            )
        if record.event in ("created", "reconciled_found"):
            entry.resource = _resource_from(payload["resource"])
            entry.state = _State.OWNED
        elif record.event == "create_failed":
            if payload["disposition"] == CreationDisposition.DEFINITELY_NOT_CREATED.value:
                entry.state = _State.CLOSED
            else:
                known = payload.get("resource")
                entry.resource = _resource_from(known) if isinstance(known, dict) else None
                entry.state = _State.AMBIGUOUS
        elif record.event == "reconciled_absent":
            entry.state = _State.CLOSED
        elif record.event == "leased":
            entry.holders.add(str(payload["holder"]))
        elif record.event == "unleased":
            entry.holders.discard(str(payload["holder"]))
        elif record.event == "release_deferred":
            entry.state = _State.DEFERRED
        elif record.event == "released":
            status = CleanupStatus(payload["status"])
            detail = payload.get("detail")
            entry.result = CleanupResult(
                record.resource_id, status, detail if isinstance(detail, str) else None
            )
            entry.state = _State.CLOSED if status in _COMPLETE else _State.UNCONFIRMED
        else:
            raise JournalError(
                f"record {record.sequence} has an event this version does not know: "
                f"{record.event!r}"
            )

    # ----------------------------------------------------------- provisioning

    def _adapter(self, provider: str) -> ResourceAdapter:
        adapter = self._adapters.get(provider)
        if adapter is None:
            raise ResourceContractError(
                f"no provider {provider!r} is registered; configure [providers.{provider}]"
            )
        return adapter

    async def provision(
        self, request: ResourceRequest, *, scope: ResourceScope, owner_id: str
    ) -> ResourceHandle:
        """Provision ``request`` for ``owner_id`` in ``scope``, after its prerequisites."""
        if self._mode != "write":
            raise ResourceContractError("this registry does not provision")
        self._check_provisioning(request)  # A closed registry is also a closing one.
        adapter = self._adapter(request.provider)
        depends_on = list(request.depends_on)
        prerequisites = await self._awaited(
            self._track(_prerequisites(adapter, request), unsent=True),
            lambda: self._closing(request),
        )
        for prerequisite in prerequisites:
            run_id = self._run_id or ""
            # Every request that needs the prerequisite shares it, and the run owns it.
            # It is therefore provisioned to completion whoever waits, because one
            # waiter's cancellation must not abandon a create the others wait for.
            handle = await _detached(
                self.provision(prerequisite, scope=ResourceScope.RUN, owner_id=run_id)
            )
            if not prerequisite.release_after_create:
                depends_on.append(handle.intent.resource_id)
        if depends_on != list(request.depends_on):
            request = replace(request, depends_on=tuple(depends_on))
        self._check_provisioning(request)
        self._check_dependencies(request, scope)
        key = (scope, owner_id, request.logical_key)
        lock = self._key_locks.setdefault(key, asyncio.Lock())
        async with lock:
            # Check again once the key is this caller's. A caller that waited on it
            # while the registry began closing starts nothing.
            self._check_provisioning(request)
            if request.release_after_create:
                self._raise_if_cooling(key)
            reused = await self._reuse(key, request)
            if reused is not None:
                return reused
            try:
                return await self._create(adapter, request, scope, owner_id)
            except Exception as error:
                if request.release_after_create:
                    self._failed_helpers[key] = (
                        self._monotonic(),
                        f"{type(error).__name__}: {error}",
                    )
                raise

    def _raise_if_cooling(self, key: tuple[ResourceScope, str, str]) -> None:
        """Fail at once if the helper that ``key`` names failed to create within the cooldown."""
        now = self._monotonic()
        for expired_key, (failed_at, _) in tuple(self._failed_helpers.items()):
            if now - failed_at >= FAILED_HELPER_COOLDOWN_SECONDS:
                del self._failed_helpers[expired_key]
        failed = self._failed_helpers.get(key)
        if failed is None:
            return
        failed_at, error = failed
        ago = now - failed_at
        raise ProvisioningFailure(
            f"{key[2]!r} failed to create {ago:.0f}s ago; not trying again for "
            f"{FAILED_HELPER_COOLDOWN_SECONDS - ago:.0f}s: {error}",
            CreationDisposition.DEFINITELY_NOT_CREATED,
        )

    def _check_provisioning(self, request: ResourceRequest) -> None:
        if self._stopping:
            raise self._closing(request)

    def _closing(self, request: ResourceRequest) -> ProvisioningFailure:
        return ProvisioningFailure(
            f"{request.logical_key!r} was not created: the resource registry is closing",
            CreationDisposition.DEFINITELY_NOT_CREATED,
        )

    def _check_dependencies(self, request: ResourceRequest, scope: ResourceScope) -> None:
        for resource_id in request.depends_on:
            entry = self._entries.get(resource_id)
            if entry is None or entry.state is not _State.OWNED or entry.releasing:
                raise ResourceContractError(f"dependency {resource_id} is not a live resource")
            if scope is ResourceScope.RUN and entry.intent.scope is ResourceScope.EPISODE:
                raise ResourceContractError("a run resource cannot depend on an episode resource")

    async def _reuse(
        self, key: tuple[ResourceScope, str, str], request: ResourceRequest
    ) -> ResourceHandle | None:
        """Return the live handle that ``key`` already names, or ``None`` to create a new one.

        A released or releasing key starts a new identity, but only once the old one is
        proven gone. A name derived from the key therefore never collides with it.
        """
        wanted = (request.provider, request.kind, _specification_digest(request))
        built = self._built.get(key) if request.release_after_create else None
        if built is not None:
            request_made = built.intent.request
            if (request_made.provider, request_made.kind, built.intent.specification_digest) != (
                wanted
            ):
                raise ResourceContractError(
                    f"logical_key {request.logical_key!r} already names a different resource"
                )
            return built
        previous = self._keys.get(key)
        if previous is None:
            return None
        entry = self._entries[previous]
        found = (
            entry.intent.request.provider,
            entry.intent.request.kind,
            entry.intent.specification_digest,
        )
        if entry.state is _State.DEFERRED:
            raise ResourceContractError(
                f"logical_key {request.logical_key!r} names a resource still held by a "
                "dependant or lease; it cannot be created again until that ends"
            )
        if entry.state is _State.OWNED and entry.resource is not None and not entry.releasing:
            if wanted != found:
                raise ResourceContractError(
                    f"logical_key {request.logical_key!r} already names a different resource"
                )
            return ResourceHandle(entry.intent, entry.resource)
        if entry.state is not _State.CLOSED:
            # This is a release, so it runs on whoever waits for it.
            await self._awaited(
                self._track(self._wait_until_gone(entry)),
                lambda: self._closing(request),
                shield=True,
            )
        return None

    async def _wait_until_gone(self, entry: _Entry) -> None:
        adapter = self._adapter(entry.intent.request.provider)
        async with entry.lock:
            if entry.state is _State.CLOSED:
                return
            if entry.state is _State.DEFERRED or self._blocked(entry):
                raise ResourceContractError(
                    f"{entry.intent.desired_name} is still held by a dependant or lease; "
                    "it cannot be created again until that ends"
                )
            result = await self._release_now(entry, adapter)
        await self._cascade(entry)
        if result.status not in _COMPLETE:
            raise ProvisioningFailure(
                f"{entry.intent.desired_name} could not be proven gone before it is created "
                f"again ({result.status.value}: {result.detail})",
                CreationDisposition.DEFINITELY_NOT_CREATED,
            )

    async def _create(
        self,
        adapter: ResourceAdapter,
        request: ResourceRequest,
        scope: ResourceScope,
        owner_id: str,
    ) -> ResourceHandle:
        # Check again with no await before the entry exists. A dependency then cannot
        # begin releasing between the check and this resource counting as its dependant.
        # The registry also cannot begin closing between the check and the intent.
        self._check_provisioning(request)
        self._check_dependencies(request, scope)
        resource_id = uuid.uuid4().hex
        labels = {
            **request.labels,
            "loops.run_id": self._run_id or "",
            "loops.scope": scope.value,
            "loops.owner_id": owner_id,
            "loops.resource_id": resource_id,
        }
        intent = ResourceIntent(
            resource_id=resource_id,
            scope=scope,
            owner_id=owner_id,
            request=request,
            specification_digest=_specification_digest(request),
            desired_name=adapter.desired_name(request, resource_id),
            ownership_labels=MappingProxyType(labels),
        )
        self._record("intent", resource_id, _intent_payload(intent))
        entry = _Entry(intent, written_at_ns=self._clock())
        self._entries[resource_id] = entry
        self._keys[(scope, owner_id, request.logical_key)] = resource_id
        resource = await self._uncancellable_create(adapter, entry)
        handle = ResourceHandle(intent, resource)
        if request.release_after_create:
            self._built[(scope, owner_id, request.logical_key)] = handle
        return handle

    async def _uncancellable_create(
        self, adapter: ResourceAdapter, entry: _Entry
    ) -> ProviderResource:
        """Run the create, which a cancellation cannot cut off once it may have been sent.

        A create cut off mid-request leaves a resource that nobody can prove exists or
        not until the create settles. A cancellation that arrives once the create is sent
        therefore waits for it to land, bounded by its own deadline, and only then
        propagates. The resource is then owned, and released and proven gone like any
        other. A cancellation that arrives while the create still waits for a permit, or
        for its next try after a refusal, stops it there, as never sent. A create refused
        after its caller stopped is not sent again. The entry's lock is held until the
        outcome is recorded, so a release waits for it.

        A transient helper, one with ``release_after_create``, is released before this
        returns if it exists once its create ends, whether it is usable or not. It is
        released from the create's own task, which a closing registry waits for or cuts
        off. A helper that lands while the registry closes is therefore released before
        the journal closes. A create that failed also releases, from that task, each
        deferred dependency that it alone was holding. A create that the closing
        registry stops fails as a ``ProvisioningFailure``.
        """
        resource_id = entry.intent.resource_id

        async def locked() -> ProviderResource:
            try:
                async with entry.lock:
                    return await self._create_or_reconcile(adapter, entry)
            except asyncio.CancelledError as error:
                if not self._stopping:
                    raise
                # The closing registry stopped the create, and records how far it got.
                # A caller that stopped it still raises its own cancellation.
                raise ProvisioningFailure(
                    f"creating {entry.intent.desired_name} stopped: the resource registry "
                    "is closing",
                    CreationDisposition.AMBIGUOUS
                    if entry.sent
                    else CreationDisposition.DEFINITELY_NOT_CREATED,
                    entry.resource,
                ) from error
            finally:
                if entry.intent.request.release_after_create and entry.state is _State.OWNED:
                    await asyncio.shield(self._start_release(resource_id))
                # A create that failed frees the dependencies it alone was holding.
                await self._cascade(entry)

        creating = self._track(locked())
        self._creates[resource_id] = creating
        creating.add_done_callback(lambda _: self._creates.pop(resource_id, None))
        try:
            return await asyncio.shield(creating)
        except asyncio.CancelledError:
            caller_cancelled = _caller_cancelled()
            entry.abandoned = True
            if not entry.sent:
                creating.cancel()
            await outlast(creating)
            if self._stopping and not caller_cancelled and creating.cancelled():
                disposition = (
                    CreationDisposition.DEFINITELY_CREATED
                    if entry.state is _State.OWNED
                    else CreationDisposition.AMBIGUOUS
                    if entry.sent
                    else CreationDisposition.DEFINITELY_NOT_CREATED
                )
                raise ProvisioningFailure(
                    f"creating {entry.intent.desired_name} stopped: the resource registry "
                    "is closing",
                    disposition,
                    entry.resource,
                )
            raise

    async def _create_or_reconcile(
        self, adapter: ResourceAdapter, entry: _Entry
    ) -> ProviderResource:
        try:
            resource = await self._send_create(adapter, entry)
        except asyncio.CancelledError:
            # A create stopped while waiting for a permit or a try has nothing in flight.
            if not entry.sent:
                self._record_failure(entry, CreationDisposition.DEFINITELY_NOT_CREATED, None)
            # A create that a closing registry cut off in flight may exist.
            elif self._stopping:
                self._record_failure(entry, CreationDisposition.AMBIGUOUS, None)
            raise
        except ProvisioningFailure as error:
            if error.disposition is CreationDisposition.DEFINITELY_NOT_CREATED:
                self._record_failure(entry, error.disposition, None)
                raise
            if error.disposition is CreationDisposition.DEFINITELY_CREATED:
                # The resource exists but is not usable, such as a sandbox that deployed
                # into a failed state. It is owned, so it is released, and the
                # provision still fails.
                if error.resource is not None:
                    self._own(entry, "created", error.resource)
                    raise
                self._record_failure(entry, CreationDisposition.AMBIGUOUS, None)
                await self._reconcile(adapter, entry)
                raise
            self._record_failure(entry, CreationDisposition.AMBIGUOUS, error.resource)
            return await self._reconcile_or_fail(adapter, entry, error)
        except (TimeoutError, Exception) as error:  # noqa: BLE001 - a create that may have run
            self._record_failure(entry, CreationDisposition.AMBIGUOUS, None)
            return await self._reconcile_or_fail(adapter, entry, error)
        self._own(entry, "created", resource)
        return resource

    async def _send_create(self, adapter: ResourceAdapter, entry: _Entry) -> ProviderResource:
        """Send the adapter's create, and send it again while the provider refuses it.

        Every try first waits for a slot in the provider's create window, holding no
        permit. Every try shares the kind's one create deadline, which starts at the
        first permit, so a wait for a slot before the first send costs none of it. A
        refusal left nothing, so between tries nothing is in flight and the wait holds
        no permit. A retry is sent only while half the deadline is left, so the create
        that lands has time to finish. Past that point, the last refusal becomes the
        create's failure.
        """
        name = entry.intent.desired_name
        budget = adapter.deadlines.create_for(entry.intent.request.kind)
        provider = entry.intent.request.provider
        window = self._windows.get(provider)
        if window is None:
            window = self._windows[provider] = _CreateWindow(provider, self._create_window)
        place = next(self._in_line)
        loop = asyncio.get_running_loop()
        deadline: float | None = None
        refusal: ProvisioningFailure | None = None
        waited = 0.0
        refused_at = 0.0
        for tries in itertools.count(1):
            await self._take_slot(window, place)
            epoch, refused = window.epoch, None
            try:
                if refusal is not None:
                    waited += self._monotonic() - refused_at
                async with self._permit():
                    if deadline is None:
                        deadline = loop.time() + budget
                    elif refusal is not None and loop.time() > deadline - budget / 2:
                        # It waited past half its deadline to be sent, so it was never sent.
                        raise refusal
                    entry.sent = True
                    try:
                        async with asyncio.timeout_at(deadline):
                            resource = await adapter.create(entry.intent)
                    except ProvisioningFailure as error:
                        refused = error.refused
                        if not refused:
                            raise
                        refusal = error
                    except Exception:  # A failure or timeout, but not a refusal.
                        refused = False
                        raise
                    else:
                        refused = False
                        if refusal is not None:
                            logger.info(
                                "%s: created after %d refusals and %.1fs of waits",
                                name,
                                tries - 1,
                                waited,
                            )
                        return resource
                    entry.sent = False
            finally:
                self._answered(window, epoch, refused, refusal)
            refused_at = self._monotonic()
            wait = self._jitter() * min(
                REFUSED_FIRST_WAIT_SECONDS * 2 ** (tries - 1), REFUSED_MAX_WAIT_SECONDS
            )
            if entry.abandoned:
                raise refusal  # The caller stopped, and nothing is in flight to wait for.
            if waited + wait > budget / 2 or loop.time() + wait > deadline - budget / 2:
                logger.warning("%s: refused %d times; giving up: %s", name, tries, refusal)
                raise refusal
            logger.debug(
                "%s: refused (%s); sending it again in %.1fs (try %d)", name, refusal, wait, tries
            )
            await self._sleep(wait)
        raise AssertionError("unreachable")

    async def _take_slot(self, window: _CreateWindow, place: int) -> None:
        """Wait for a slot in ``window``, in line at ``place``, holding no permit."""
        waiter = asyncio.get_running_loop().create_future()
        heapq.heappush(window.waiting, (place, waiter))
        window.wake()
        try:
            await waiter
        except asyncio.CancelledError:
            # The waiter was given a slot just as it stopped.
            if waiter.done() and not waiter.cancelled():
                window.in_flight -= 1
                window.wake()
            else:
                waiter.cancel()  # Its entry in the line is skipped.
            raise

    def _answered(
        self,
        window: _CreateWindow,
        epoch: int,
        refused: bool | None,
        refusal: ProvisioningFailure | None,
    ) -> None:
        """Free a create's slot. A refusal narrows the window, and any other answer widens it.

        ``refused`` is ``None`` for a create that was not sent, or was stopped before its
        answer. That says nothing about the provider's room. Any other answer, such as a
        success, a failure or a timeout, was not a refusal, so it widens the window.
        """
        window.in_flight -= 1
        before = window.size
        if refused and epoch == window.epoch:
            window.size = max(1.0, window.size / 2)
            window.epoch += 1
        elif refused is False:
            window.size = min(float(window.cap), window.size + 1 / window.size)
        if refused:
            window.refusals += 1
        if window.size != before:
            logger.debug(
                "%s: %.2f creates in flight at most (was %.2f)",
                window.provider,
                window.size,
                before,
            )
        if window.size < window.cap and window.narrowed_at is None:
            window.narrowed_at = self._monotonic()
            logger.info(
                "%s: refused (%s); fewer of its creates are sent at once until it takes more",
                window.provider,
                refusal,
            )
        elif window.size >= window.cap and window.narrowed_at is not None:
            logger.info(
                "%s: back to %d creates at once, after %d refusals over %.1fs",
                window.provider,
                window.cap,
                window.refusals,
                self._monotonic() - window.narrowed_at,
            )
            window.narrowed_at, window.refusals = None, 0
        window.wake()

    def _own(self, entry: _Entry, event: str, resource: ProviderResource) -> None:
        self._record(event, entry.intent.resource_id, {"resource": _resource_payload(resource)})
        entry.resource, entry.state = resource, _State.OWNED

    def _record_failure(
        self, entry: _Entry, disposition: CreationDisposition, resource: ProviderResource | None
    ) -> None:
        payload: dict[str, JsonValue] = {"disposition": disposition.value}
        if resource is not None:
            payload["resource"] = _resource_payload(resource)
        self._record("create_failed", entry.intent.resource_id, payload)
        entry.resource = resource
        entry.state = (
            _State.CLOSED
            if disposition is CreationDisposition.DEFINITELY_NOT_CREATED
            else _State.AMBIGUOUS
        )

    async def _reconcile_or_fail(
        self, adapter: ResourceAdapter, entry: _Entry, cause: BaseException
    ) -> ProviderResource:
        found = await self._reconcile(adapter, entry)
        if found.status is ReconcileStatus.FOUND and found.resource is not None:
            if found.unusable is not None:
                # The lookup made it owned, so it is released. The provision still fails.
                raise ProvisioningFailure(
                    f"{entry.intent.desired_name} exists but cannot be used: {found.unusable}",
                    CreationDisposition.DEFINITELY_CREATED,
                    found.resource,
                ) from cause
            return found.resource
        if found.status is ReconcileStatus.PROVEN_ABSENT:
            raise ProvisioningFailure(
                f"creating {entry.intent.desired_name} failed: {cause}",
                CreationDisposition.DEFINITELY_NOT_CREATED,
            ) from cause
        raise ProvisioningFailure(
            f"creating {entry.intent.desired_name} failed and it may exist: {cause}",
            CreationDisposition.AMBIGUOUS,
            entry.resource,
        ) from cause

    async def _reconcile(self, adapter: ResourceAdapter, entry: _Entry) -> ReconcileResult:
        """Look up an unresolved create by identity. A failed lookup proves nothing.

        The exception is a settled create, one inherited from a stopped process and older
        than ``SETTLE_SECONDS``. For that create, the adapter may take a not-found answer
        as absence.
        """
        settled = entry.settles_at_ns is not None and self._clock() >= entry.settles_at_ns
        try:
            async with self._permit(), asyncio.timeout(adapter.deadlines.reconcile_seconds):
                found = await adapter.reconcile_create(entry.intent, settled=settled)
        except (TimeoutError, Exception) as error:  # noqa: BLE001 - reported as unknown
            evidence = describe(error, adapter.deadlines.reconcile_seconds)
            if isinstance(error, TimeoutError) and not settled:
                evidence = self._unsettled(entry, evidence)
            return ReconcileResult(ReconcileStatus.UNKNOWN, evidence=evidence)
        if found.status is ReconcileStatus.FOUND and found.resource is not None:
            if not await validate(adapter, entry.intent, found.resource, self._permit):
                return ReconcileResult(
                    ReconcileStatus.UNKNOWN, evidence="ownership did not validate"
                )
            self._own(entry, "reconciled_found", found.resource)
        elif found.status is ReconcileStatus.PROVEN_ABSENT:
            self._record(
                "reconciled_absent", entry.intent.resource_id, {"evidence": found.evidence}
            )
            entry.state = _State.CLOSED
        return found

    def _unsettled(self, entry: _Entry, evidence: str) -> str:
        """Explain why a lookup that found nothing proves nothing yet, and when it will."""
        settles = entry.settles_at_ns
        if settles is None and entry.written_at_ns is not None:
            settles = entry.written_at_ns + SETTLE_SECONDS * 1_000_000_000
        when = f" at {_local_time(settles)}" if settles is not None else ""
        return (
            f"its create may still land, so a lookup cannot prove it absent until the "
            f"create settles{when}; run `loops cleanup` again after that ({evidence})"
        )

    # --------------------------------------------------------------- leases

    def _check_open(self) -> None:
        if self._closed:
            raise ResourceContractError("the resource registry is closed")

    def lease(self, resource_id: str, holder: str) -> ResourceHandle:
        self._check_open()
        entry = self._entries.get(resource_id)
        if (
            entry is None
            or entry.state is not _State.OWNED
            or entry.resource is None
            or entry.releasing
        ):
            raise ResourceContractError(f"{resource_id} is not a live resource")
        if entry.intent.scope is not ResourceScope.RUN:
            raise ResourceContractError("only a run resource can be acquired")
        self._record("leased", resource_id, {"holder": holder})
        entry.holders.add(holder)
        return ResourceHandle(entry.intent, entry.resource)

    async def unlease(self, resource_id: str, holder: str) -> None:
        self._check_open()
        entry = self._entries[resource_id]
        if holder not in entry.holders:
            return
        self._record("unleased", resource_id, {"holder": holder})
        entry.holders.discard(holder)
        if entry.state is _State.DEFERRED:
            await self.release(resource_id)

    def handle(self, resource_id: str) -> ResourceHandle | None:
        entry = self._entries.get(resource_id)
        if entry is None or entry.resource is None or entry.state is not _State.OWNED:
            return None
        return ResourceHandle(entry.intent, entry.resource)

    def usage_environment(self, resource_id: str) -> Mapping[str, str | SecretStr]:
        entry = self._entries[resource_id]
        if entry.resource is None or entry.state is not _State.OWNED:
            raise ResourceContractError(f"{resource_id} is not a live resource")
        return self._adapter(entry.intent.request.provider).usage_environment(entry.resource)

    # These methods cover a sandbox's usage, not its ownership. Like the usage
    # environment, they write no journal record and take no provisioning permit. Each
    # call still has a deadline.

    async def sandbox_client(self, resource_id: str) -> object:
        """Return the provider's own client for a live sandbox, within the attach deadline."""
        self._check_open()
        provider, resource = self._sandbox(resource_id)

        async def attach() -> object:
            async with asyncio.timeout(provider.deadlines.attach_seconds):
                return await provider.client(resource)

        return await self._awaited(self._track(attach()), _closed_under)

    async def close_sandbox_client(self, resource_id: str, client: object) -> None:
        """Close a client that ``sandbox_client`` gave for ``resource_id``.

        The caller puts the deadline on this call.
        """
        self._check_open()
        entry = self._entries[resource_id]
        provider = self._sandbox_provider(entry.intent.request.provider)
        await self._awaited(self._track(provider.close_client(client)), _closed_under)

    async def sandbox_alive(self, resource_id: str) -> bool:
        """Return the provider's fact of whether a live sandbox is still usable.

        The attach deadline applies, because reaching a sandbox and reading whether it is
        there are one operation.
        """
        self._check_open()
        provider, resource = self._sandbox(resource_id)

        async def read() -> bool:
            async with asyncio.timeout(provider.deadlines.attach_seconds):
                return await provider.alive(resource)

        alive = await self._awaited(self._track(read()), _closed_under)
        if not isinstance(alive, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ResourceContractError(
                f"the [providers.{self._entries[resource_id].intent.request.provider}] adapter's "
                f"alive() returned {alive!r}, not a bool"
            )
        return alive

    def _sandbox(self, resource_id: str) -> tuple[_SandboxAdapter, ProviderResource]:
        entry = self._entries[resource_id]
        if entry.resource is None or entry.state is not _State.OWNED:
            raise ResourceContractError(f"{resource_id} is not a live resource")
        return self._sandbox_provider(entry.intent.request.provider), entry.resource

    def _sandbox_provider(self, provider: str) -> _SandboxAdapter:
        adapter = self._adapter(provider)
        if not isinstance(adapter, SandboxProvider):
            raise ResourceContractError(
                f"the [providers.{provider}] adapter is not a SandboxProvider, so it cannot "
                "back an Episode's sandbox"
            )
        return adapter

    # -------------------------------------------------------------- release

    async def release(self, resource_id: str) -> CleanupResult:
        """Release one resource, or defer it while something live still needs it.

        The release runs as its own task, so a caller cancelled mid-release does not
        cancel the release. A repeated release returns without calling the provider.
        """
        self._check_open()
        return await self._awaited(
            self._start_release(resource_id),
            lambda: CleanupResult(
                resource_id,
                CleanupStatus.UNCONFIRMED,
                "cut off: the resource registry closed before the release finished",
            ),
            shield=True,
        )

    def _start_release(self, resource_id: str) -> asyncio.Task[CleanupResult]:
        """Start releasing ``resource_id`` as a task that ``close`` waits for or cuts off."""
        entry = self._entries[resource_id]
        entry.releasing = entry.live
        return self._track(self._release(resource_id))

    async def _release(self, resource_id: str) -> CleanupResult:
        entry = self._entries[resource_id]
        try:
            return await self._release_entry(entry)
        finally:
            entry.releasing = False

    async def _release_entry(self, entry: _Entry) -> CleanupResult:
        resource_id = entry.intent.resource_id
        async with entry.lock:
            if entry.state is _State.CLOSED:
                return entry.result or CleanupResult(resource_id, CleanupStatus.ALREADY_ABSENT)
            if self._blocked(entry):
                if entry.state is not _State.DEFERRED:
                    self._record("release_deferred", resource_id, {})
                    entry.state = _State.DEFERRED
                return CleanupResult(
                    resource_id, CleanupStatus.DEFERRED, "a live resource or lease still needs it"
                )
            provider = entry.intent.request.provider
            adapter = self._adapters.get(provider)
            if adapter is None:
                # Keep it for a cleanup configured with this provider, rather than skip
                # it silently.
                result = self._finish(
                    entry,
                    CleanupStatus.UNCONFIRMED,
                    f"no [providers.{provider}] adapter is configured to release it",
                )
            else:
                result = await self._release_now(entry, adapter)
        await self._cascade(entry)
        return result

    async def _cascade(self, entry: _Entry) -> None:
        """Release each deferred dependency that ``entry`` was the last thing holding.

        ``close`` waits for or cuts off each such release, whoever cascades it.
        """
        if entry.live:
            return
        for dependency in entry.intent.request.depends_on:
            waiting = self._entries.get(dependency)
            if (
                waiting is not None
                and waiting.state is _State.DEFERRED
                and not self._blocked(waiting)
            ):
                await asyncio.shield(self._start_release(dependency))

    def _blocked(self, entry: _Entry) -> bool:
        if entry.holders:
            return True
        resource_id = entry.intent.resource_id
        return any(
            other.live and resource_id in other.intent.request.depends_on
            for other in self._entries.values()
        )

    async def _release_now(self, entry: _Entry, adapter: ResourceAdapter) -> CleanupResult:
        """Release under the permit and deadlines. Absence counts only when proven."""
        # An unresolved create, however far recovery got.
        if entry.resource is None:
            found = await self._reconcile(adapter, entry)
            if found.status is ReconcileStatus.PROVEN_ABSENT:
                return self._finish(entry, CleanupStatus.ALREADY_ABSENT, found.evidence)
            if found.status is not ReconcileStatus.FOUND:
                return self._finish(entry, CleanupStatus.UNCONFIRMED, found.evidence)
        resource = entry.resource
        if resource is None:
            return self._finish(entry, CleanupStatus.UNCONFIRMED, "no provider identity is known")
        status, detail = await release_resource(adapter, resource, self._permit)
        return self._finish(entry, status, detail)

    def _finish(self, entry: _Entry, status: CleanupStatus, detail: str | None) -> CleanupResult:
        result = CleanupResult(entry.intent.resource_id, status, detail)
        self._record("released", result.resource_id, {"status": status.value, "detail": detail})
        entry.result = result
        entry.state = _State.CLOSED if status in _COMPLETE else _State.UNCONFIRMED
        return result

    async def release_scope(self, scope: ResourceScope, owner_id: str) -> CleanupReport:
        """Release everything ``owner_id`` holds in ``scope``, dependants first.

        Resources that nothing else in the scope depends on are released together, so
        one waiting out its create does not hold up the rest.
        """
        self._check_open()
        for entry in list(self._entries.values()):
            if owner_id in entry.holders:
                await self.unlease(entry.intent.resource_id, owner_id)
        selected = [
            entry
            for entry in self._entries.values()
            if entry.intent.scope is scope and entry.intent.owner_id == owner_id and entry.live
        ]
        results: list[CleanupResult] = []
        for layer in _dependants_first(selected):
            results.extend(
                await asyncio.gather(*(self.release(entry.intent.resource_id) for entry in layer))
            )
        return CleanupReport(scope, owner_id, tuple(results))

    async def release_all(self) -> CleanupReport:
        """Release everything the journal still holds, episodes before the run.

        This is the recovery path. No lease survives the process that took it, so every
        recorded lease ends first.
        """
        self._check_open()
        for entry in list(self._entries.values()):
            for holder in sorted(entry.holders):
                self._record("unleased", entry.intent.resource_id, {"holder": holder})
            entry.holders.clear()
        results: list[CleanupResult] = []
        for scope in (ResourceScope.EPISODE, ResourceScope.RUN):
            owners = sorted(
                {
                    e.intent.owner_id
                    for e in self._entries.values()
                    if e.intent.scope is scope and e.live
                }
            )
            # Owners are independent, so their releases run together, limited by permits.
            reports = await asyncio.gather(*(self.release_scope(scope, owner) for owner in owners))
            for report in reports:
                results.extend(report.results)
        return CleanupReport(None, None, tuple(results))

    @property
    def unresolved(self) -> CleanupReport:
        """Return every identity whose absence is not proven, exactly as recorded."""
        return CleanupReport(
            None,
            None,
            tuple(self._outcome(entry) for entry in self._entries.values() if entry.live),
        )

    def _outcome(self, entry: _Entry) -> CleanupResult:
        resource_id = entry.intent.resource_id
        if entry.state is _State.DEFERRED:
            return CleanupResult(
                resource_id, CleanupStatus.DEFERRED, "a lease or dependant holds it"
            )
        return entry.result or CleanupResult(
            resource_id, CleanupStatus.UNCONFIRMED, entry.state.value
        )

    def _track[T](self, work: Coroutine[Any, Any, T], *, unsent: bool = False) -> asyncio.Task[T]:
        """Run ``work`` as tracked work.

        ``close`` waits for or cuts off tracked work before it closes the journal and the
        adapters. Every operation that uses an adapter or writes the journal runs this
        way, so nothing can outlive them. ``unsent`` work has sent the provider nothing,
        so ``begin_shutdown`` stops it. Work that starts after the close deadline passed
        is cut off as it starts. No work may start once the close found none left.
        """
        if self._idle:
            work.close()
            raise ResourceContractError("the resource registry is closed")
        task = asyncio.ensure_future(work)
        if self._cutting_off:
            task.cancel()
        self._work.add(task)
        task.add_done_callback(self._work.discard)
        # Retrieve the exception even when the caller is gone. A caller still waiting
        # gets it all the same.
        task.add_done_callback(_retrieve_exception)
        if unsent:
            self._unsent.add(task)
            task.add_done_callback(self._unsent.discard)
        return task

    async def _awaited[T](
        self,
        task: asyncio.Task[T],
        cut_off: Callable[[], T | BaseException],
        *,
        shield: bool = False,
    ) -> T:
        """Return ``task``'s result to its caller.

        When the closing registry stopped the task, and the caller did not, it returns
        what ``cut_off`` gives, which is a result or the error to raise. A caller that
        was itself cancelled still gets its cancellation. A caller that was not
        cancelled never gets a bare cancellation. ``shield`` keeps ``task`` running when
        its caller is cancelled, as a release needs.
        """
        try:
            return await (asyncio.shield(task) if shield else task)
        except asyncio.CancelledError:
            if not task.cancelled() or _caller_cancelled():
                raise
        outcome = cut_off()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def begin_shutdown(self) -> None:
        """Stop provisioning at the start of closing. Releases still run.

        A create not yet sent, or a prerequisite still being resolved, stops where it is.
        Such a create is recorded as never created. Its caller fails as
        ``DEFINITELY_NOT_CREATED`` without starting a create, and so does a caller waiting
        on a key or a prerequisite, and every later ``provision``. A create already sent
        runs on, for ``close`` to wait for.
        """
        if self._stopping:
            return
        self._stopping = True
        for resource_id, creating in list(self._creates.items()):
            self._entries[resource_id].abandoned = True
            if not self._entries[resource_id].sent:
                creating.cancel()
        for task in list(self._unsent):
            task.cancel()

    async def close(self, timeout: float | None = None) -> None:
        """Stop provisioning, wait for work in flight, then unlock the journal and close adapters.

        Each adapter is closed once.

        The closing itself is one task, which the first ``close`` starts. Once no work is
        left, it records what never started, unlocks the journal and closes the adapters.
        A ``close`` cancelled before that leaves the task running.

        With a ``timeout``, work still running when it passes is cut off. A create is then
        recorded as ambiguous, and a release as not made. Each is kept, unresolved, for
        ``loops cleanup``. The closing then has ``CUT_OFF_GRACE_SECONDS`` more, so
        ``close`` returns within ``timeout`` plus that grace. Work that ignores its
        cancellation is reported. It keeps the journal and the adapters open until it
        stops, because nothing in this process can stop it sooner.
        """
        if self._finishing is None:
            self.begin_shutdown()
            self._closed = True
            self._finishing = asyncio.create_task(self._close_when_idle())
            self._finishing.add_done_callback(_retrieve_exception)
        finishing = self._finishing
        if timeout is None:
            await asyncio.shield(finishing)
            return
        done, _ = await asyncio.wait({finishing}, timeout=timeout)
        if not done:
            self._cut_off()
            done, _ = await asyncio.wait({finishing}, timeout=CUT_OFF_GRACE_SECONDS)
        if done:
            finishing.result()
            return
        logger.error(
            "%d resource operation(s), or an adapter's close, did not stop within %gs of "
            "being cut off; the journal and the adapters stay open until they do",
            len(self._work),
            CUT_OFF_GRACE_SECONDS,
        )

    async def _close_when_idle(self) -> None:
        """Once no work is left, record what never started, unlock the journal, close adapters."""
        while self._work:
            await asyncio.wait(set(self._work))
        self._idle = True
        try:
            try:
                self._close_unsent()
            finally:
                self._journal.close()
        finally:
            await close_adapters(self._adapters)

    def _cut_off(self) -> None:
        """Cancel the work in flight when the close deadline passed, and any that starts after."""
        self._cutting_off = True
        logger.warning(
            "%d resource create(s) and %d other operation(s) were still in flight when "
            "the resource registry's close deadline passed; cutting them off",
            len(self._creates),
            len(self._work) - len(self._creates),
        )
        for task in list(self._work):
            task.cancel()

    def _close_unsent(self) -> None:
        """Record as never created each create that this process stopped before it started."""
        for resource_id, entry in self._entries.items():
            if (
                entry.state is _State.PENDING
                and entry.written_at_ns is not None
                and not entry.sent
                and resource_id not in self._creates
            ):
                self._record_failure(entry, CreationDisposition.DEFINITELY_NOT_CREATED, None)

    @property
    def journal(self) -> Journal:
        return self._journal


_COMPLETE = (CleanupStatus.RELEASED, CleanupStatus.ALREADY_ABSENT)


def _retrieve_exception(task: asyncio.Future[Any]) -> None:
    """Retrieve a background task's exception when no caller remains to report it."""
    if not task.cancelled():
        task.exception()


def _local_time(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def describe(error: BaseException, seconds: float) -> str:
    """Describe an operation's failure as evidence.

    A deadline that ran out says how long it was.
    """
    if isinstance(error, TimeoutError) and not str(error):
        return f"TimeoutError: no answer within {seconds:g}s"
    return f"{type(error).__name__}: {error}"


async def validate(
    adapter: ResourceAdapter,
    intent: ResourceIntent,
    resource: ProviderResource,
    permit: Permit = _no_permit,
) -> bool:
    """Return whether ``resource`` is the one that ``intent`` owns.

    A failed check proves nothing, so it returns ``False``.
    """
    try:
        async with permit(), asyncio.timeout(adapter.deadlines.reconcile_seconds):
            return await adapter.validate_ownership(intent, resource)
    except (TimeoutError, Exception):  # noqa: BLE001 - an unvalidated resource is not owned
        return False


async def release_resource(
    adapter: ResourceAdapter, resource: ProviderResource, permit: Permit = _no_permit
) -> tuple[CleanupStatus, str | None]:
    """Release ``resource``, then prove it gone, each step under the permit and its deadline.

    ``RELEASED`` needs authoritative absence. Anything less is ``UNCONFIRMED``.
    """
    deadlines = adapter.deadlines
    try:
        async with permit(), asyncio.timeout(deadlines.release_seconds):
            await adapter.release(resource)
    except (TimeoutError, Exception) as error:  # noqa: BLE001 - kept as unconfirmed
        return (
            CleanupStatus.UNCONFIRMED,
            f"release failed: {describe(error, deadlines.release_seconds)}",
        )
    try:
        async with permit(), asyncio.timeout(deadlines.confirm_absent_seconds):
            absent = await adapter.confirm_absent(resource)
    except (TimeoutError, Exception) as error:  # noqa: BLE001 - kept as unconfirmed
        detail = describe(error, deadlines.confirm_absent_seconds)
        return CleanupStatus.UNCONFIRMED, f"absence unknown: {detail}"
    if absent.status is ReconcileStatus.PROVEN_ABSENT:
        return CleanupStatus.RELEASED, absent.evidence
    return CleanupStatus.UNCONFIRMED, absent.evidence or absent.status.value


def _dependants_first(entries: list[_Entry]) -> list[list[_Entry]]:
    """Split ``entries`` into layers, each released before anything its members depend on."""
    remaining = {entry.intent.resource_id: entry for entry in entries}
    layers: list[list[_Entry]] = []
    while remaining:
        needed = {dep for entry in remaining.values() for dep in entry.intent.request.depends_on}
        ready = [entry for rid, entry in remaining.items() if rid not in needed] or list(
            remaining.values()
        )
        for entry in ready:
            remaining.pop(entry.intent.resource_id)
        layers.append(ready)
    return layers


def _specification_digest(request: ResourceRequest) -> str:
    identity = request.specification if request.identity is None else request.identity
    return digest(dict(json_mapping(identity, where="resource specification")))


def _intent_payload(intent: ResourceIntent) -> dict[str, JsonValue]:
    request = intent.request
    return {
        "resource_id": intent.resource_id,
        "scope": intent.scope.value,
        "owner_id": intent.owner_id,
        "specification_digest": intent.specification_digest,
        "desired_name": intent.desired_name,
        "ownership_labels": dict(intent.ownership_labels),
        "request": {
            "provider": request.provider,
            "kind": request.kind,
            "logical_key": request.logical_key,
            "specification": dict(request.specification),
            "depends_on": list(request.depends_on),
            "labels": dict(request.labels),
            "release_after_create": request.release_after_create,
        },
    }


def _intent_from(payload: Mapping[str, JsonValue]) -> ResourceIntent:
    raw = payload["request"]
    assert isinstance(raw, dict)
    request = ResourceRequest(
        provider=str(raw["provider"]),
        kind=str(raw["kind"]),
        logical_key=str(raw["logical_key"]),
        specification=_mapping(raw["specification"]),
        depends_on=tuple(str(item) for item in _list(raw["depends_on"])),
        labels={k: str(v) for k, v in _mapping(raw["labels"]).items()},
        release_after_create=bool(raw["release_after_create"]),
    )
    return ResourceIntent(
        resource_id=str(payload["resource_id"]),
        scope=ResourceScope(payload["scope"]),
        owner_id=str(payload["owner_id"]),
        request=request,
        specification_digest=str(payload["specification_digest"]),
        desired_name=str(payload["desired_name"]),
        ownership_labels=MappingProxyType(
            {k: str(v) for k, v in _mapping(payload["ownership_labels"]).items()}
        ),
    )


def _resource_payload(resource: ProviderResource) -> dict[str, JsonValue]:
    return {
        "provider_id": resource.provider_id,
        "name": resource.name,
        "labels": dict(resource.labels),
        "metadata": dict(resource.metadata),
    }


def _resource_from(payload: JsonValue) -> ProviderResource:
    data = _mapping(payload)
    return ProviderResource(
        provider_id=str(data["provider_id"]),
        name=str(data["name"]),
        labels={k: str(v) for k, v in _mapping(data["labels"]).items()},
        metadata=_mapping(data.get("metadata", {})),
    )


def _mapping(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TypeError("a journal payload field is not an object")
    return value


async def _prerequisites(
    adapter: ResourceAdapter, request: ResourceRequest
) -> list[ResourceRequest]:
    """Return what ``request`` needs created first, within the provider's reconcile deadline."""
    try:
        async with asyncio.timeout(adapter.deadlines.reconcile_seconds):
            return list(await adapter.prerequisites(request))
    except (TimeoutError, Exception) as error:
        raise ProvisioningFailure(
            f"resolving the prerequisites of {request.logical_key!r} failed: "
            f"{type(error).__name__}: {error}",
            CreationDisposition.DEFINITELY_NOT_CREATED,
        ) from error


def _closed_under() -> ResourceContractError:
    return ResourceContractError("the resource registry closed before this call finished")


def _caller_cancelled() -> bool:
    """Return whether the running task was itself asked to stop."""
    caller = asyncio.current_task()
    return caller is not None and caller.cancelling() > 0


async def _detached[T](work: Coroutine[Any, Any, T]) -> T:
    """Return ``work``'s result, and let ``work`` run to completion if the caller is cancelled."""
    task = asyncio.ensure_future(work)
    task.add_done_callback(lambda done: done.cancelled() or done.exception())
    return await asyncio.shield(task)


def _list(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        raise TypeError("a journal payload field is not a list")
    return value
