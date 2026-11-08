"""Provide Daytona sandboxes as resources that the registry owns.

One kind of request reaches this adapter, ``sandbox``. Its specification is
``{"sandbox": {...}}``, which takes any of ``SANDBOX_KEYS``, each optional.

``image`` is either ``{"dockerfile": <text>, "context": [<file>...]}`` or
``{"base": <image ref>}``. The first is a declarative build, which Daytona builds once
and caches by content. With neither ``image`` nor ``snapshot``, the sandbox uses
Daytona's default snapshot. ``resources`` is ``{"cpu", "memory", "disk"}``, in vCPUs
and GiB. The adapter fills each one left out from ``[providers.daytona]``, so an image
sandbox always names its size. It fills in the lifecycle the same way. A sandbox
always stops after idling or dies at a time to live (TTL), and it is deleted once
stopped, so a sandbox the run loses is not kept forever.

Every create carries the cookbook's ownership labels and a name unique to the
resource. Daytona names are unique in the organization, so a create is idempotent by
name. Reconciliation therefore looks the sandbox up by name, and never sends a second
create. Release deletes by Daytona's immutable id. Absence counts only on a ``404`` for
that id or a ``destroyed`` state.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from pydantic import SecretStr

from loops_cookbook.evidence import JsonValue
from loops_cookbook.providers.daytona import Daytona, usage
from loops_cookbook.providers.daytona.api import Observed
from loops_cookbook.providers.daytona.errors import (
    create_refused,
    creation_disposition,
    not_found,
    reconcile_retryable,
    rejected_before_sending,
    status_code,
)
from loops_cookbook.providers.daytona.settings import (
    SANDBOX,
    DaytonaSettings,
    lifecycle_minutes,
)
from loops_cookbook.resources.api import (
    CreationDisposition,
    ProviderDeadlines,
    ProviderResource,
    ProvisioningFailure,
    ReconcileResult,
    ReconcileStatus,
    ResourceIntent,
    ResourceRequest,
    SandboxCapacity,
    SandboxResources,
)

# The ownership label that names the registry resource a sandbox was created for.
RESOURCE_ID_LABEL = "loops.resource_id"
SANDBOX_KEYS = frozenset(
    {
        "name",
        "image",
        "snapshot",
        "resources",
        "labels",
        "env_vars",
        "secrets",
        "network_block_all",
        "network_allow_list",
        "domain_allow_list",
        "auto_stop_interval",
        "auto_delete_interval",
        "auto_archive_interval",
        "ttl_minutes",
        "ephemeral",
        "public",
        "os_user",
        "language",
    }
)
RESOURCE_KEYS = frozenset({"cpu", "memory", "disk"})
# Keys whose value must be an object. The adapter checks them before the intent is
# journaled.
OBJECT_KEYS = frozenset({"image", "resources", "labels"})
# Lifecycle intervals, in minutes. Daytona takes none below 0.
LIFECYCLE_KEYS = (
    "auto_stop_interval",
    "auto_delete_interval",
    "auto_archive_interval",
    "ttl_minutes",
)


class Api(Protocol):
    async def create(self, config: Mapping[str, Any], timeout: float) -> Observed: ...
    async def get(self, id_or_name: str) -> Observed: ...
    async def delete(self, sandbox_id: str) -> None: ...
    def list(self, labels: Mapping[str, str]) -> AsyncGenerator[Sequence[Observed], None]: ...
    def client(self) -> usage.Client: ...
    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Credentials:
    """Hold ``DAYTONA_API_KEY`` and, when set, ``DAYTONA_API_URL``, from the environment."""

    api_key: SecretStr
    api_url: str | None = None


class DaytonaContractError(ValueError):
    """Raised for a request that this adapter cannot turn into a Daytona sandbox."""


def _mapping(value: JsonValue | None, where: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise DaytonaContractError(f"{where} must be an object")
    return value


def _create_failure(intent: ResourceIntent, error: BaseException) -> ProvisioningFailure:
    return ProvisioningFailure(
        f"creating Daytona sandbox {intent.desired_name} failed: "
        f"{type(error).__name__}: {error} (status {status_code(error)})",
        creation_disposition(error),
        refused=create_refused(error),
    )


class DaytonaAdapter:
    """Implement ``ResourceAdapter`` and ``SandboxProvider`` for Daytona sandboxes."""

    sandbox_kind: ClassVar[type[Daytona]] = Daytona

    def __init__(
        self,
        settings: DaytonaSettings,
        credentials: Credentials,
        api: Api,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._credentials = credentials
        self._api = api
        self._sleep = sleep
        self._closed = False

    @property
    def deadlines(self) -> ProviderDeadlines:
        return self._settings.deadlines

    @property
    def settings(self) -> DaytonaSettings:
        return self._settings

    @property
    def capacity(self) -> SandboxCapacity:
        """Return the size a sandbox gets when it names none, and the most one may have."""
        settings = self._settings
        return SandboxCapacity(
            defaults=SandboxResources(
                cpu=settings.cpu, memory_gib=settings.memory_gib, disk_gib=settings.disk_gib
            ),
            per_sandbox=SandboxResources(
                cpu=settings.max_sandbox_cpu,
                memory_gib=settings.max_sandbox_memory_gib,
                disk_gib=settings.max_sandbox_disk_gib,
            ),
        )

    def _checked(self, request: ResourceRequest) -> dict[str, JsonValue]:
        """Return the request's sandbox configuration with the table's lifecycle filled in.

        A configuration that this adapter cannot create is refused before any journal
        entry or Daytona call.
        """
        if request.kind != SANDBOX:
            raise DaytonaContractError(f"Daytona provides no {request.kind!r} resource")
        config = dict(_mapping(request.specification.get(SANDBOX), SANDBOX))
        unknown = sorted(set(config) - SANDBOX_KEYS)
        if unknown:
            raise DaytonaContractError(f"a Daytona sandbox takes no {', '.join(unknown)}")
        objects = {
            key: _mapping(config[key], f"sandbox.{key}") for key in OBJECT_KEYS & set(config)
        }
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in objects.get("labels", {}).items()
        ):
            raise DaytonaContractError("sandbox.labels must be an object of strings")
        if "image" in config and "snapshot" in config:
            raise DaytonaContractError("a Daytona sandbox is created from an image or a snapshot")
        if "image" not in config and "resources" in config:
            raise DaytonaContractError(
                "only a sandbox built from an image takes resources; a snapshot fixes its own"
            )
        unsized = set(objects.get("resources", {})) - RESOURCE_KEYS
        if unsized:
            raise DaytonaContractError(f"sandbox.resources takes only {sorted(RESOURCE_KEYS)}")
        for key in LIFECYCLE_KEYS:
            value = config.get(key)
            if value is not None and not lifecycle_minutes(value):
                raise DaytonaContractError(
                    f"sandbox.{key} must be a whole number of minutes, at least 0"
                )
        settings = self._settings
        config.setdefault("auto_stop_interval", settings.auto_stop_minutes)
        if not config.get("ephemeral"):
            config.setdefault("auto_delete_interval", settings.auto_delete_minutes)
        if settings.ttl_minutes and "ttl_minutes" not in config:
            config["ttl_minutes"] = settings.ttl_minutes
        if not config["auto_stop_interval"] and not config.get("ttl_minutes"):
            raise DaytonaContractError(
                "a Daytona sandbox with auto_stop_interval 0 and no ttl_minutes would never "
                "stop on its own, so a leak would run forever"
            )
        return config

    # ------------------------------------------------------------- identity

    def desired_name(self, request: ResourceRequest, resource_id: str) -> str:
        """Return the caller's name for the sandbox, or a name unique to the resource.

        Harbor's Daytona environment names no sandbox, so each trial's sandbox is
        ``loops-<resource id>``, and no two names collide.
        """
        name = self._checked(request).get("name")
        if name is None:
            return f"loops-{resource_id}"
        if not isinstance(name, str) or not name:
            raise DaytonaContractError(f"{name!r} is not a Daytona sandbox name")
        return name

    def _resource(self, observed: Observed) -> ProviderResource:
        return ProviderResource(
            provider_id=observed.id,
            name=observed.name,
            labels=dict(observed.labels),
            metadata={"state": observed.state},
        )

    async def validate_ownership(self, intent: ResourceIntent, resource: ProviderResource) -> bool:
        return resource.name == intent.desired_name and all(
            resource.labels.get(key) == value for key, value in intent.ownership_labels.items()
        )

    # ---------------------------------------------------------------- create

    async def prerequisites(self, request: ResourceRequest) -> Sequence[ResourceRequest]:
        # Daytona builds and caches a declarative image inside the create.
        return []

    async def create(self, intent: ResourceIntent) -> ProviderResource:
        self._check_open()
        config = self.sandbox_config(intent)
        try:
            observed = await self._api.create(config, self.deadlines.create_seconds)
        except Exception as error:
            if rejected_before_sending(error):
                raise _create_failure(intent, error) from error
            # The SDK's create goes on calling Daytona after the create itself, so any
            # other failure may come after the sandbox exists. The name tells them apart.
            observed = await self._landed(intent, error)
        return await self._wait_started(self._resource(observed))

    async def _landed(self, intent: ResourceIntent, error: BaseException) -> Observed:
        """Return the sandbox that a failed create left behind, or raise the create's failure.

        After a ``404`` for the name, the failure means what it says. A refusal, which is
        a ``429`` or a connection never made, is sent again. A rejection, which is a
        ``400``, ``401`` or ``403``, is not. Anything else is still ambiguous. A lookup
        that also fails proves nothing, so the create is ambiguous.
        """
        try:
            observed = await self._api.get(intent.desired_name)
        except Exception as lookup:
            if not_found(lookup):
                raise _create_failure(intent, error) from lookup
            raise ProvisioningFailure(
                f"creating Daytona sandbox {intent.desired_name} failed "
                f"({type(error).__name__}: {error}), and looking it up failed: "
                f"{type(lookup).__name__}: {lookup}",
                CreationDisposition.AMBIGUOUS,
            ) from lookup
        if not await self.validate_ownership(intent, self._resource(observed)):
            # This is not the sandbox this create made, so the registry's reconcile
            # refuses to adopt it.
            raise ProvisioningFailure(
                f"Daytona sandbox {intent.desired_name} exists without this run's ownership",
                CreationDisposition.AMBIGUOUS,
            ) from error
        return observed

    def sandbox_config(self, intent: ResourceIntent) -> dict[str, Any]:
        """Return the create configuration.

        It is the request's configuration, with the table's defaults filled in, the
        ownership labels added and a lifecycle that ends a lost sandbox.
        """
        config: dict[str, Any] = self._checked(intent.request)
        settings = self._settings
        config["name"] = intent.desired_name
        labels = config.get("labels", {})
        config["labels"] = {**labels, **settings.labels, **intent.ownership_labels}
        if "image" in config:
            sized = dict(config.get("resources", {}))
            for key, default in (
                ("cpu", settings.cpu),
                ("memory", settings.memory_gib),
                ("disk", settings.disk_gib),
            ):
                if sized.get(key) is None:
                    sized[key] = default
            config["resources"] = sized
        return config

    async def _wait_started(self, resource: ProviderResource) -> ProviderResource:
        """Poll a created sandbox until it is ``started``, and fail if it dies.

        The caller's deadline bounds the wait. A sandbox that dies is known to exist, so
        the failure carries it, and the registry owns and releases it.
        """
        while True:
            state = resource.metadata.get("state")
            if usage.started(state):
                return resource
            if state in usage.DEAD:
                raise ProvisioningFailure(
                    f"Daytona sandbox {resource.name} entered {state}",
                    CreationDisposition.DEFINITELY_CREATED,
                    resource,
                )
            await self._sleep(usage.POLL_SECONDS)
            try:
                resource = self._resource(await self._api.get(resource.provider_id))
            except Exception as error:
                if not reconcile_retryable(error) or not_found(error):
                    raise ProvisioningFailure(
                        f"reading Daytona sandbox {resource.name} back failed: {error}",
                        CreationDisposition.DEFINITELY_CREATED,
                        resource,
                    ) from error

    # ------------------------------------------------------------ reconcile

    async def reconcile_create(
        self, intent: ResourceIntent, *, settled: bool = False
    ) -> ReconcileResult:
        """Read the uniquely named sandbox back, and never issue a second create.

        A ``404``, a ``429``, a ``5xx`` and transport errors are retried until the
        registry's reconcile deadline. A ``404`` for the name proves absence only once
        the create is ``settled``. A sandbox counts as found as soon as its name
        resolves, so the registry owns it and its release deletes it. ``attach`` waits
        for a sandbox that is still building or starting. A dead sandbox is found but
        unusable, so its provision fails.
        """
        while True:
            try:
                observed = await self._api.get(intent.desired_name)
            except Exception as error:
                if settled and not_found(error):
                    return ReconcileResult(
                        ReconcileStatus.PROVEN_ABSENT,
                        evidence=f"GET /sandbox/{intent.desired_name} answered 404 for a "
                        "settled create: none can still be in flight",
                    )
                if not reconcile_retryable(error):
                    raise
                await self._sleep(usage.POLL_SECONDS)
                continue
            found = self._resource(observed)
            if observed.state in usage.DEAD:
                return ReconcileResult(
                    ReconcileStatus.FOUND, found, unusable=f"it entered {observed.state}"
                )
            return ReconcileResult(ReconcileStatus.FOUND, found)

    # -------------------------------------------------------------- release

    async def release(self, resource: ProviderResource) -> None:
        try:
            await self._api.delete(resource.provider_id)
        except Exception as error:
            if not not_found(error):
                raise

    async def confirm_absent(self, resource: ProviderResource) -> ReconcileResult:
        """Wait until Daytona answers ``404`` for the sandbox's id, or reports it ``destroyed``.

        Ids are never reused, so neither answer can be about another sandbox. The
        registry's deadline bounds the wait.
        """
        while True:
            try:
                observed = await self._api.get(resource.provider_id)
            except Exception as error:
                if not_found(error):
                    return ReconcileResult(
                        ReconcileStatus.PROVEN_ABSENT,
                        evidence=f"GET /sandbox/{resource.provider_id} answered 404 after its "
                        "delete",
                    )
                if not reconcile_retryable(error):
                    raise
            else:
                if observed.state == usage.DESTROYED:
                    return ReconcileResult(
                        ReconcileStatus.PROVEN_ABSENT,
                        evidence=f"GET /sandbox/{resource.provider_id} shows it destroyed",
                    )
            await self._sleep(usage.POLL_SECONDS)

    async def list_owned(
        self, labels: Mapping[str, str]
    ) -> AsyncGenerator[Sequence[ProviderResource], None]:
        # Daytona filters by labels itself, and each page is checked again exactly.
        pages = self._api.list(labels)
        try:
            async for page in pages:
                yield [
                    self._resource(observed)
                    for observed in page
                    if observed.state != usage.DESTROYED
                    and all(observed.labels.get(key) == value for key, value in labels.items())
                ]
        finally:
            await pages.aclose()

    # ---------------------------------------------------------------- usage

    def usage_environment(self, resource: ProviderResource) -> Mapping[str, str | SecretStr]:
        """Return what a separate process needs to reach this sandbox.

        The API key is organization-wide.
        """
        environment: dict[str, str | SecretStr] = {
            "DAYTONA_API_KEY": self._credentials.api_key,
            usage.SANDBOX_ID: resource.provider_id,
        }
        if self._credentials.api_url:
            environment["DAYTONA_API_URL"] = self._credentials.api_url
        if self._settings.target:
            environment["DAYTONA_TARGET"] = self._settings.target
        return environment

    async def client(self, resource: ProviderResource) -> Any:
        """Return the SDK's ``AsyncSandbox`` for the sandbox, once it is ``started``."""
        return await usage.attach(
            resource,
            self._api.client(),
            timeout_seconds=self.deadlines.attach_seconds,
            sleep=self._sleep,
        )

    async def close_client(self, client: Any) -> None:
        """Do nothing, because an ``AsyncSandbox`` holds nothing of its own.

        It shares the provider's client.
        """

    async def alive(self, resource: ProviderResource) -> bool:
        """Return whether the sandbox is still ours and ``started``.

        A ``404`` for its id means it is not.
        """
        try:
            observed = await self._api.get(resource.provider_id)
        except Exception as error:
            if not_found(error):
                return False
            raise
        owner = observed.labels.get(RESOURCE_ID_LABEL)
        return owner == resource.labels.get(RESOURCE_ID_LABEL) and usage.started(observed.state)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("the Daytona adapter is closed")

    async def close(self) -> None:
        self._closed = True
        await self._api.close()
