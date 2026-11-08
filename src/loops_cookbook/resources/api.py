"""Define the resource contracts that every provider shares.

The contracts cover requests, identities, adapters and reports. A rollout runtime asks
for a resource with a ``ResourceRequest``. The scoped service turns it into an
immutable ``ResourceIntent``, which the registry journals before any provider call. A
``ResourceAdapter`` then performs the provider's input and output. Nothing here knows
about any particular provider.
"""

from __future__ import annotations

import copy
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType, TracebackType
from typing import ClassVar, Protocol, Self, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, SecretStr

from loops_cookbook.evidence import JsonValue


class ResourceScope(str, Enum):
    EPISODE = "episode"
    RUN = "run"


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """What a rollout runtime asks a provider for: a kind of resource and its specification.

    ``logical_key`` names the resource within its owner, so asking again with the same
    key returns the live resource. Asking with a different specification, or a
    different ``identity`` when it is set, is refused. ``depends_on`` names, by resource
    ID, the resources that must not be released before this one.
    """

    provider: str
    kind: str
    logical_key: str
    specification: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    depends_on: tuple[str, ...] = ()
    labels: Mapping[str, str] = field(default_factory=dict[str, str])
    # True for transient helpers, such as image builders.
    release_after_create: bool = False
    # What makes two requests for one key the same resource. When it is unset, the whole
    # specification does.
    identity: Mapping[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        if not self.provider or not self.kind or not self.logical_key:
            raise ValueError("a resource request needs a provider, a kind, and a logical_key")
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))
        # Keep a private copy, so nothing the caller does later can change what was digested.
        specification = MappingProxyType(copy.deepcopy(dict(self.specification)))
        object.__setattr__(self, "specification", specification)
        if self.identity is not None:
            object.__setattr__(
                self, "identity", MappingProxyType(copy.deepcopy(dict(self.identity)))
            )


@dataclass(frozen=True, slots=True)
class ProviderDeadlines:
    """Hold the deadline, in seconds, for every provider operation.

    Each deadline bounds one call, except ``list_seconds``, which bounds each page of a
    listing. A whole listing grows with the provider's account, not with the run, so it
    has no deadline of its own.
    """

    create_seconds: float
    reconcile_seconds: float
    release_seconds: float
    confirm_absent_seconds: float
    list_seconds: float
    attach_seconds: float
    create_seconds_by_kind: Mapping[str, float] = field(default_factory=dict[str, float])

    def __post_init__(self) -> None:
        values = [
            self.create_seconds,
            self.reconcile_seconds,
            self.release_seconds,
            self.confirm_absent_seconds,
            self.list_seconds,
            self.attach_seconds,
            *self.create_seconds_by_kind.values(),
        ]
        if not all(value > 0 and value != float("inf") for value in values):
            raise ValueError("provider deadlines must be positive and finite")

    def create_for(self, kind: str) -> float:
        """Return the create deadline for ``kind``, or ``create_seconds`` if it has none."""
        return self.create_seconds_by_kind.get(kind, self.create_seconds)


@dataclass(frozen=True, slots=True)
class ResourceIntent:
    """The immutable record of a resource about to be created, journaled before any provider call.

    The adapter creates it as ``desired_name``, carrying ``ownership_labels``. A lookup
    by that name and those labels can then find it after a crash.
    """

    resource_id: str
    scope: ResourceScope
    owner_id: str
    request: ResourceRequest
    specification_digest: str
    desired_name: str
    ownership_labels: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ProviderResource:
    """A resource as its provider identifies it, by its provider ID, name and labels.

    ``metadata`` holds anything else the adapter records about it. The registry
    journals all of it.
    """

    provider_id: str
    name: str
    labels: Mapping[str, str]
    metadata: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])

    def __post_init__(self) -> None:
        if not self.provider_id or not self.name:
            raise ValueError("a provider resource needs a provider id and a name")


@dataclass(frozen=True, slots=True)
class ResourceHandle:
    """A live resource: the intent it was created from, and the provider's resource."""

    intent: ResourceIntent
    resource: ProviderResource


class CreationDisposition(str, Enum):
    """Whether a create that failed may have left a resource at the provider.

    Only ``DEFINITELY_NOT_CREATED`` lets the registry forget the resource. It looks up
    an ``AMBIGUOUS`` create by identity, and releases a ``DEFINITELY_CREATED`` resource.
    """

    DEFINITELY_NOT_CREATED = "definitely_not_created"
    DEFINITELY_CREATED = "definitely_created"
    AMBIGUOUS = "ambiguous"


class CleanupStatus(str, Enum):
    RELEASED = "released"
    ALREADY_ABSENT = "already_absent"
    DEFERRED = "deferred"
    UNCONFIRMED = "unconfirmed"


class ReconcileStatus(str, Enum):
    """What a provider lookup proved about a resource: found, proven absent, or unknown."""

    FOUND = "found"
    PROVEN_ABSENT = "proven_absent"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    """Record what a provider lookup proved. Only ``FOUND`` carries a resource.

    ``PROVEN_ABSENT`` needs authoritative negative evidence, tied to the immutable name
    and ownership identity. A negative or empty answer from an eventually consistent
    listing is ``UNKNOWN``.
    """

    status: ReconcileStatus
    resource: ProviderResource | None = None
    evidence: str | None = None
    # Set when the resource is FOUND and owned but not usable, such as a sandbox that
    # deployed dead. A provision then fails, and a release still releases it.
    unusable: str | None = None

    def __post_init__(self) -> None:
        if (self.status is ReconcileStatus.FOUND) != (self.resource is not None):
            raise ValueError("only a FOUND reconcile result carries a resource")
        if self.unusable is not None and self.status is not ReconcileStatus.FOUND:
            raise ValueError("only a FOUND reconcile result can be unusable")
        if self.status is ReconcileStatus.PROVEN_ABSENT and not self.evidence:
            raise ValueError("PROVEN_ABSENT requires the evidence it relies on")


@dataclass(frozen=True, slots=True)
class CleanupResult:
    resource_id: str
    status: CleanupStatus
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class CleanupReport:
    scope: ResourceScope | None  # None for a report that spans scopes.
    owner_id: str | None
    results: tuple[CleanupResult, ...]

    @property
    def complete(self) -> bool:
        return all(
            result.status in (CleanupStatus.RELEASED, CleanupStatus.ALREADY_ABSENT)
            for result in self.results
        )

    @property
    def unresolved(self) -> tuple[CleanupResult, ...]:
        """Return every result whose resource may still exist, with its exact identity."""
        return tuple(
            result
            for result in self.results
            if result.status not in (CleanupStatus.RELEASED, CleanupStatus.ALREADY_ABSENT)
        )


class ProvisioningFailure(RuntimeError):
    """Raised when a resource could not be provided.

    ``disposition`` says whether the provider may hold one anyway. ``refused`` means the
    provider turned the create away for now, as a rate limit does. Such a create left
    nothing, so the registry sends it again after a wait.
    """

    def __init__(
        self,
        message: str,
        disposition: CreationDisposition,
        resource: ProviderResource | None = None,
        *,
        refused: bool = False,
    ) -> None:
        if refused and disposition is not CreationDisposition.DEFINITELY_NOT_CREATED:
            raise ValueError("only a create that left nothing can be refused")
        super().__init__(message)
        self.disposition = disposition
        self.resource = resource
        self.refused = refused


class CleanupIncomplete(RuntimeError):
    """Raised when cleanup could not prove every resource gone.

    The report names each resource that may be left.
    """

    def __init__(self, report: CleanupReport) -> None:
        names = ", ".join(f"{r.resource_id} ({r.status.value})" for r in report.unresolved)
        super().__init__(f"cleanup is incomplete: {names}")
        self.report = report


class ResourceLease(Protocol):
    """A borrowed run resource, held until it is closed or the borrowing scope ends."""

    @property
    def handle(self) -> ResourceHandle:
        """The borrowed resource's handle."""
        ...

    async def close(self) -> None:
        """End the lease. A release that waited for this lease then goes ahead.

        Closing a lease again does nothing.
        """
        ...

    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


class ResourceAdapter(Protocol):
    """The contract a provider implements, so the cookbook can create, find and release resources.

    The registry puts the matching deadline from ``deadlines`` on each provider call. A
    crash during a release repeats the release, so an adapter's release must tolerate
    a resource that is already gone.
    """

    @property
    def deadlines(self) -> ProviderDeadlines:
        """The deadline for each of this provider's operations."""
        ...

    async def prerequisites(self, request: ResourceRequest) -> Sequence[ResourceRequest]:
        """Return the requests to provision before ``request``, such as a sandbox's image build.

        The registry asks under the reconcile deadline. It provisions each prerequisite
        as a run resource, which every request that needs it shares. Return nothing when
        ``request`` needs nothing first.
        """
        ...

    def desired_name(self, request: ResourceRequest, resource_id: str) -> str:
        """Return the name to create ``request``'s resource under, given its new ``resource_id``.

        The registry journals the name with the intent before any create, so a lookup
        after a crash can find the resource by it.
        """
        ...

    async def create(self, intent: ResourceIntent) -> ProviderResource:
        """Create ``intent``'s resource and return it.

        On failure, raise ``ProvisioningFailure`` with whether a resource may exist. Any
        other failure, or the create deadline, counts as ``AMBIGUOUS``. The registry
        then looks the resource up with ``reconcile_create()``. Raise a ``refused``
        failure when the provider turns the create away for now, and the registry sends
        it again after a wait.
        """
        ...

    async def reconcile_create(
        self, intent: ResourceIntent, *, settled: bool = False
    ) -> ReconcileResult:
        """Find ``intent``'s resource by identity, and never issue a second create.

        ``settled`` means the create cannot still be in flight, and a create that landed
        is visible. A not-found answer may then be returned as ``PROVEN_ABSENT``.
        """
        ...

    async def validate_ownership(self, intent: ResourceIntent, resource: ProviderResource) -> bool:
        """Return whether ``resource`` carries ``intent``'s identity, so the cookbook may own it.

        The registry asks before it owns a resource that a lookup found, and ``loops
        cleanup`` asks before it releases one. A failure counts as ``False``.
        """
        ...

    async def release(self, resource: ProviderResource) -> None:
        """Delete ``resource`` at the provider. The registry then calls ``confirm_absent()``."""
        ...

    async def confirm_absent(self, resource: ProviderResource) -> ReconcileResult:
        """Look ``resource`` up after its release, and return whether it is gone.

        Return ``PROVEN_ABSENT`` only on authoritative evidence. Anything else leaves the
        release unconfirmed, and the resource is kept for ``loops cleanup``.
        """
        ...

    def list_owned(
        self, labels: Mapping[str, str]
    ) -> AsyncGenerator[Sequence[ProviderResource], None]:
        """Yield every resource that carries all of ``labels``, one page at a time.

        The caller puts the ``deadlines.list_seconds`` deadline on each page, never on
        the whole listing.
        """
        ...

    def usage_environment(self, resource: ProviderResource) -> Mapping[str, str | SecretStr]:
        """Return the environment, credentials included, that a process needs for ``resource``."""
        ...

    async def close(self) -> None:
        """Close the adapter's own clients. The registry calls it once its work has stopped."""
        ...


@runtime_checkable
class RegionalProvider(Protocol):
    """A ``ResourceAdapter`` that creates everything in one region.

    A rollout runtime may need to hold what it describes to that region, such as a
    Harbor task's image build.
    """

    @property
    def region(self) -> str: ...


@runtime_checkable
class ImageProvider(Protocol):
    """A ``ResourceAdapter`` that holds images a sandbox can start from.

    A rollout runtime can then refuse, before the run, a task that names an image the
    provider does not hold, instead of failing on every trial.
    """

    async def image_tags(self, name: str) -> frozenset[str] | None:
        """Return the tags of the provider's image ``name``.

        It returns ``None`` when the provider holds no image by that name, and raises
        when it cannot tell.
        """
        ...


@dataclass(frozen=True, slots=True)
class SandboxResources:
    """A sandbox's size, or a limit on sizes, in vCPUs and GiB.

    A dimension is ``None`` where it is unknown.
    """

    cpu: float | None = None
    memory_gib: float | None = None
    disk_gib: float | None = None


@dataclass(frozen=True, slots=True)
class SandboxCapacity:
    """The size of a provider's sandbox when a request names none, and the most one may have.

    The limits are only those that the provider's table declares.
    """

    defaults: SandboxResources
    per_sandbox: SandboxResources = field(default_factory=SandboxResources)


@runtime_checkable
class CapacityProvider(Protocol):
    """A ``ResourceAdapter`` that sizes CPU, memory and disk independently.

    It can say how big one sandbox may be, so a rollout runtime can refuse a larger one
    before the run.
    """

    @property
    def capacity(self) -> SandboxCapacity:
        """The size of a sandbox whose request names none, and the most one sandbox may have."""
        ...


@runtime_checkable
class ConfiguredProvider(Protocol):
    """A ``ResourceAdapter`` that shares its validated ``[providers.<name>]`` table.

    The provider's own code elsewhere, such as its Harbor environment, then reads the
    settings that the adapter was built from.
    """

    @property
    def settings(self) -> BaseModel: ...


@runtime_checkable
class SandboxProvider[ClientT](Protocol):
    """A ``ResourceAdapter`` whose resources can back an Episode's sandbox.

    ``sandbox_kind`` is the ``SandboxKind`` of its sandboxes. It decides which Episodes
    the provider can back, which are those written for that kind or a base of it.

    ``client`` returns the provider's own client for a live resource. The registry puts
    the attach deadline on it. If the call fails or is cancelled after opening a
    client, the adapter closes that client. ``close_client`` closes one client that
    ``client`` returned.

    ``alive`` reports the provider's fact of whether the resource is still usable. It
    returns ``False`` for a resource that is gone, and raises only when it cannot tell.
    """

    @property
    def sandbox_kind(self) -> type[SandboxKind[ClientT]]:
        """The ``SandboxKind`` of this provider's sandboxes."""
        ...

    async def client(self, resource: ProviderResource) -> ClientT:
        """Return the provider's own client for the live ``resource``, under the attach deadline."""
        ...

    async def close_client(self, client: ClientT) -> None:
        """Close one client that ``client()`` returned."""
        ...

    async def alive(self, resource: ProviderResource) -> bool:
        """Return whether ``resource`` is still usable. Raise only when the provider cannot tell."""
        ...


class SandboxSession(Protocol):
    """An attempt's hold on its sandbox, which gives its one client and its liveness."""

    async def client(self) -> object:
        """Return the sandbox's one client, which is attached on the first call."""
        ...

    async def alive(self) -> bool:
        """Return the provider's fact of whether the sandbox is still usable."""
        ...


class SandboxSpec(BaseModel):
    """The ``[rollout_runtime.config.sandbox]`` table, typed by the sandbox's kind.

    The table sits beside its ``provider`` key. A kind's spec declares its keys flat,
    each with a default or marked required. It forbids any other key, so ``loops check``
    refuses a misspelt key. It validates strictly, so a value of another type is refused
    rather than converted. For example, ``true`` is not a memory size. Declare a choice
    as a ``Literal``, because a TOML string validates as a ``Literal`` but not as an
    ``Enum``. Each attempt's sandbox is provisioned as a ``resource_kind`` resource
    with ``specification()``, which returns the provider's own request format.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, strict=True)

    resource_kind: ClassVar[str] = "sandbox"

    def specification(self) -> dict[str, JsonValue]:
        """Return the provider's request for this sandbox, by default the table's values as JSON."""
        return self.model_dump(mode="json")


class SandboxGone(RuntimeError):
    """Raised when the sandbox went away, so what is missing from it proves nothing.

    Raised from ``grade()``, it fails that grading try. The grading is tried again only
    within the episode's ``grading_retries``.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"sandbox {name} is gone")


class SandboxKind[ClientT]:
    """An Episode's sandbox from one provider, whose client is a ``ClientT``.

    Each provider ships one subclass that names it, such as
    ``class Blaxel(SandboxKind[SandboxInstance]): provider = "blaxel"``. An Episode names
    that subclass as its third type argument, and the provider's adapter declares it as
    its ``sandbox_kind``. ``ctx.sandbox`` is an instance of the kind that the adapter
    provides. The run owns the sandbox, and releases it with the attempt after
    ``close()``.
    """

    # The provider that this kind's sandboxes come from, as a message names it.
    provider: ClassVar[str]
    # The model for what [rollout_runtime.config.sandbox] holds beside provider. By
    # default it holds nothing.
    spec: ClassVar[type[SandboxSpec]] = SandboxSpec

    def __init__(
        self,
        name: str,
        environment: Mapping[str, str | SecretStr],
        session: SandboxSession,
    ) -> None:
        self._name = name
        self._environment = environment
        self._session = session

    @property
    def name(self) -> str:
        """Return the sandbox's own name at its provider."""
        return self._name

    @property
    def environment(self) -> Mapping[str, str | SecretStr]:
        """Return what an agent process needs to reach the sandbox."""
        return self._environment

    async def client(self) -> ClientT:
        """Return the provider's own client, attached within the attach deadline on the first call.

        Every call returns the same object. The cookbook closes it after ``close()``.
        """
        # The provider's SandboxProvider gives this kind's client, because check matched
        # the two.
        return cast(ClientT, await self._session.client())

    async def alive(self) -> bool:
        """Return the provider's fact of whether the sandbox is still usable.

        For Blaxel, usable means DEPLOYED. A ``grade()`` that finds its answer missing
        asks this after the read, through ``ensure_alive()``.
        """
        return await self._session.alive()

    async def ensure_alive(self) -> None:
        """Return while the sandbox is usable, and raise ``SandboxGone`` once it went away.

        A ``grade()`` that finds its answer missing calls this after the read. A sandbox
        that is gone never comes back. A sandbox alive at this check was therefore there
        when the read missed, so the answer really is missing. A sandbox that is gone
        proves nothing.
        """
        if not await self.alive():
            raise SandboxGone(self.name)
