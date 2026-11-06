"""Blaxel sandboxes and image builds as registry-owned resources.

Two kinds of request reach this adapter.

``sandbox``: one Blaxel sandbox. Its specification is::

    {"sandbox": <the SandboxInstance.create configuration: name, image, memory,
                 ttl, labels, and optionally region, network, extra_args>,
     "image": {"name": <the image the sandbox runs>, "force_build": bool,
               "build": <an image_build specification without its name>}}

``image`` is optional; when present and the image is not yet built, the sandbox
has one ``image_build`` prerequisite, keyed by image name, so concurrent sandboxes
of one image wait for one build. An image that already exists is adopted and
owned by no one: the cookbook never deletes an image.

``image_build``: one build, whose provider identity is the builder sandbox Blaxel
names after the image. Its specification is ``{"name", "base_image",
"instructions", "local_files", "has_entrypoint", "memory", "sandbox_version"}``,
and it is always ``release_after_create``: the builder is deleted once the image
is registered.

Every create carries the cookbook's ownership labels; a sandbox's name is the one
its caller chose, validated here. A failed create is classified by
``creation_disposition``; reconciliation finds the sandbox by name, and absence
counts only once its delete was accepted: a ``404`` for the name, or the sandbox
still shown as ours but ``TERMINATED``, which frees the name for a new create.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol

from blaxel.core.sandbox import SandboxInstance
from pydantic import SecretStr

from loops_cookbook.evidence import JsonValue
from loops_cookbook.providers.blaxel import Blaxel, usage
from loops_cookbook.providers.blaxel.api import Observed
from loops_cookbook.providers.blaxel.incidents import (
    DEAD,
    close_clients,
    create_refused,
    creation_disposition,
    reconcile_retryable,
    status_code,
)
from loops_cookbook.providers.blaxel.settings import IMAGE_BUILD, SANDBOX, BlaxelSettings
from loops_cookbook.resources.api import (
    CreationDisposition,
    ProviderDeadlines,
    ProviderResource,
    ProvisioningFailure,
    ReconcileResult,
    ReconcileStatus,
    ResourceIntent,
    ResourceRequest,
)

TERMINATED = "TERMINATED"
POLL_SECONDS = 2.0
IMAGE_POLL_SECONDS = 5.0
OTHER_BUILDER = "another owner"
# Harbor's own limit and alphabet for the names it derives.
MAX_NAME = 40
NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")


class Api(Protocol):
    async def create(self, config: Mapping[str, JsonValue]) -> Observed: ...
    async def get(self, name: str) -> Observed: ...
    async def delete(self, name: str) -> None: ...
    def list(self) -> AsyncGenerator[Sequence[Observed], None]: ...
    async def image_built(self, name: str) -> bool: ...
    async def image_tags(self, name: str) -> frozenset[str] | None: ...
    async def build_image(
        self,
        name: str,
        build: Mapping[str, JsonValue],
        labels: Mapping[str, str],
        timeout: float,
        *,
        region: str,
        build_memory: int | None = None,
        build_volume: int | None = None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class Credentials:
    """``BL_WORKSPACE`` and ``BL_API_KEY``, as the SDK reads them from the environment."""

    workspace: str
    api_key: SecretStr


class BlaxelContractError(ValueError):
    """A request this adapter cannot turn into a Blaxel resource."""


def _mapping(value: JsonValue | None, where: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise BlaxelContractError(f"{where} must be an object")
    return value


def valid_name(name: object) -> str:
    """A Blaxel resource name exactly as Harbor derives one, or a contract error."""
    if not isinstance(name, str) or len(name) > MAX_NAME or not NAME.match(name):
        raise BlaxelContractError(
            f"{name!r} is not a Blaxel name: at most {MAX_NAME} lowercase letters, digits, "
            "and inner hyphens"
        )
    return name


class BlaxelAdapter:
    """The ``ResourceAdapter`` for Blaxel sandboxes, and their ``SandboxProvider``."""

    sandbox_kind: ClassVar[type[Blaxel]] = Blaxel

    def __init__(
        self,
        settings: BlaxelSettings,
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
        self._deleted: set[str] = set()  # names whose delete Blaxel accepted

    @property
    def deadlines(self) -> ProviderDeadlines:
        return self._settings.deadlines

    @property
    def region(self) -> str:
        """``[providers.blaxel] region``: where every sandbox and image build runs."""
        return self._settings.region

    async def image_tags(self, name: str) -> frozenset[str] | None:
        """``ImageProvider``: the tags of the built image ``name``, or ``None`` when the
        workspace holds no built image by that name. One lookup within the reconcile
        deadline; a failure raises, and the caller says the image went unchecked."""
        async with asyncio.timeout(self.deadlines.reconcile_seconds):
            return await self._api.image_tags(name)

    # ------------------------------------------------------------- identity

    def desired_name(self, request: ResourceRequest, resource_id: str) -> str:
        """The caller's name for the sandbox, or one unique to the resource.

        Harbor names each trial's sandbox; a runtime that asks for sandboxes by
        specification alone gets ``loops-<resource id>``, so no two collide.
        """
        specification = dict(request.specification)
        if request.kind == SANDBOX:
            name = _mapping(specification.get("sandbox"), "sandbox").get("name")
            return valid_name(f"loops-{resource_id}" if name is None else name)
        if request.kind == IMAGE_BUILD:
            return valid_name(specification.get("name"))
        raise BlaxelContractError(f"Blaxel provides no {request.kind!r} resource")

    def _resource(self, observed: Observed) -> ProviderResource:
        return ProviderResource(
            provider_id=f"{self._credentials.workspace}/{observed.name}",
            name=observed.name,
            labels=dict(observed.labels),
            metadata={"status": observed.status},
        )

    async def validate_ownership(self, intent: ResourceIntent, resource: ProviderResource) -> bool:
        return resource.name == intent.desired_name and all(
            resource.labels.get(key) == value for key, value in intent.ownership_labels.items()
        )

    # ---------------------------------------------------------------- create

    async def prerequisites(self, request: ResourceRequest) -> Sequence[ResourceRequest]:
        if request.kind != SANDBOX or "image" not in request.specification:
            return []
        image = _mapping(request.specification["image"], "image")
        name = valid_name(image.get("name"))
        if not image.get("force_build") and await self._image_built(name):
            return []  # adopted: an existing image is owned by no one
        build = _mapping(image.get("build"), "image.build")
        return [
            ResourceRequest(
                provider=request.provider,
                kind=IMAGE_BUILD,
                logical_key=f"image-build:{name}",
                specification={**build, "name": name},
                # The name: an image is reused by name, so builds of it that differ only in
                # their source paths or builder memory are one build.
                identity={"name": name},
                release_after_create=True,
            )
        ]

    async def create(self, intent: ResourceIntent) -> ProviderResource:
        self._check_open()
        if intent.request.kind == IMAGE_BUILD:
            return await self._build(intent)
        config = self._sandbox_config(intent)
        try:
            observed = await self._api.create(config)
        except Exception as error:
            raise ProvisioningFailure(
                f"creating Blaxel sandbox {intent.desired_name} failed: "
                f"{type(error).__name__}: {error} (status {status_code(error)})",
                creation_disposition(error),
                refused=create_refused(error),
            ) from error
        return await self.wait_deployed(self._resource(observed))

    def _sandbox_config(self, intent: ResourceIntent) -> dict[str, JsonValue]:
        config = dict(_mapping(intent.request.specification.get("sandbox"), "sandbox"))
        region = config.get("region")
        if region not in (None, self._settings.region):
            raise ProvisioningFailure(
                f"sandbox {intent.desired_name} asks for region {region!r}, but "
                f"[providers.blaxel] configures {self._settings.region!r}",
                CreationDisposition.DEFINITELY_NOT_CREATED,
            )
        labels = _mapping(config.get("labels", {}), "sandbox.labels")
        config["name"] = intent.desired_name
        config["region"] = self._settings.region
        config["labels"] = {**labels, **self._settings.labels, **intent.ownership_labels}
        return config

    async def wait_deployed(self, resource: ProviderResource) -> ProviderResource:
        """Poll a created sandbox until it is ``DEPLOYED``; a dead one is a failure.

        The caller's deadline bounds the wait. A sandbox that dies is known to exist,
        so the failure carries it and the registry owns and releases it.
        """
        while True:
            status = resource.metadata.get("status")
            if usage.deployed(status):
                return resource
            if status in DEAD:
                raise ProvisioningFailure(
                    f"Blaxel sandbox {resource.name} entered {status}",
                    CreationDisposition.DEFINITELY_CREATED,
                    resource,
                )
            await self._sleep(POLL_SECONDS)
            try:
                resource = self._resource(await self._api.get(resource.name))
            except Exception as error:
                if not reconcile_retryable(error):
                    raise ProvisioningFailure(
                        f"reading Blaxel sandbox {resource.name} back failed: {error}",
                        CreationDisposition.DEFINITELY_CREATED,
                        resource,
                    ) from error

    async def _build(self, intent: ResourceIntent) -> ProviderResource:
        name = intent.desired_name
        build = dict(intent.request.specification)
        build.pop("name", None)
        resource = ProviderResource(
            provider_id=f"{self._credentials.workspace}/{name}",
            name=name,
            labels=dict(intent.ownership_labels),
            metadata={"status": "BUILT"},
        )
        try:
            await self._api.build_image(
                name,
                build,
                {**self._settings.labels, **intent.ownership_labels},
                self._settings.image_build_timeout_seconds,
                region=self._settings.region,
                build_memory=self._settings.image_build_memory_mb,
                build_volume=self._settings.image_build_volume_mb,
            )
        except Exception as error:
            failure = f"building Blaxel image {name} failed: {type(error).__name__}: {error}"
            builder = await self._builder(name)
            if (
                builder is not None
                and builder.labels.get("loops.resource_id") == intent.resource_id
            ):
                # Our own build failed: its builder exists and is ours to release.
                raise ProvisioningFailure(
                    failure, CreationDisposition.DEFINITELY_CREATED, self._resource(builder)
                ) from error
            if builder is None and (
                creation_disposition(error) is CreationDisposition.DEFINITELY_NOT_CREATED
            ):
                raise ProvisioningFailure(
                    failure,
                    CreationDisposition.DEFINITELY_NOT_CREATED,
                    refused=create_refused(error),
                ) from error
            # Another process is building the same content-addressed image; its image
            # is as good as ours, and its builder is not ours to delete. The caller's
            # deadline bounds the wait.
            while not await self._image_built(name):
                await self._sleep(IMAGE_POLL_SECONDS)
            return ProviderResource(
                provider_id=resource.provider_id,
                name=name,
                labels=resource.labels,
                metadata={"status": "BUILT", "built_by": OTHER_BUILDER},
            )
        return resource

    async def _image_built(self, name: str) -> bool:
        """``image_built``, asked again through a transient failure; the caller's deadline
        bounds it."""
        while True:
            try:
                return await self._api.image_built(name)
            except Exception as error:
                if not reconcile_retryable(error):
                    raise
            await self._sleep(IMAGE_POLL_SECONDS)

    async def _builder(self, name: str) -> Observed | None:
        """The builder sandbox, or ``None`` when Blaxel says there is none."""
        try:
            return await self._api.get(name)
        except Exception as error:
            if status_code(error) == 404:
                return None
            raise ProvisioningFailure(
                f"reading Blaxel builder {name} back failed, so it may be ours: {error}",
                CreationDisposition.AMBIGUOUS,
            ) from error

    # ------------------------------------------------------------ reconcile

    async def reconcile_create(
        self, intent: ResourceIntent, *, settled: bool = False
    ) -> ReconcileResult:
        """Read the uniquely named sandbox back; never issue a second create.

        A just-created sandbox may not be readable yet, so ``404``, ``429``, ``502``
        to ``504``, and transport errors are retried until the registry's reconcile
        deadline; a lookup proves absence only when the create is ``settled``, when a
        ``404`` for the name means no such sandbox exists. A sandbox is waited for until it is
        ``DEPLOYED``; one that is ``FAILED``, ``TERMINATED``, or ``DELETING`` is found
        but unusable, so it is owned and released and its provision fails. A builder
        is usable only once its image is built.
        """
        while True:
            try:
                observed = await self._api.get(intent.desired_name)
            except Exception as error:
                if settled and status_code(error) == 404:
                    return ReconcileResult(
                        ReconcileStatus.PROVEN_ABSENT,
                        evidence=f"GET /sandboxes/{intent.desired_name} answered 404 for a "
                        "settled create: none can still be in flight",
                    )
                if not reconcile_retryable(error):
                    raise
                await self._sleep(POLL_SECONDS)
                continue
            found = self._resource(observed)
            if intent.request.kind == IMAGE_BUILD:
                if await self._image_built(intent.desired_name):
                    return ReconcileResult(ReconcileStatus.FOUND, found)
                return ReconcileResult(
                    ReconcileStatus.FOUND,
                    found,
                    unusable=f"its builder exists but image {intent.desired_name} is not built",
                )
            if usage.deployed(observed.status):
                return ReconcileResult(ReconcileStatus.FOUND, found)
            if observed.status in DEAD:
                return ReconcileResult(
                    ReconcileStatus.FOUND, found, unusable=f"it entered {observed.status}"
                )
            await self._sleep(POLL_SECONDS)

    # -------------------------------------------------------------- release

    async def release(self, resource: ProviderResource) -> None:
        if resource.metadata.get("built_by") == OTHER_BUILDER:
            return  # no builder of ours was left; the other owner's is theirs
        try:
            await self._api.delete(resource.name)
        except Exception as error:
            if status_code(error) != 404:
                raise
        self._deleted.add(resource.name)

    async def confirm_absent(self, resource: ProviderResource) -> ReconcileResult:
        """Absent once the name answers ``404``, or shows ours ``TERMINATED`` after our delete.

        The ``TERMINATED`` rule rests on observed Blaxel behaviour, not documented
        behaviour: a deleted sandbox stays readable as ``TERMINATED`` for many
        minutes, yet a create may take its name at once, so it collides with
        nothing. It counts only for a name this adapter's delete was accepted for.
        The registry's deadline bounds the wait.
        """
        while True:
            try:
                observed = await self._api.get(resource.name)
            except Exception as error:
                if status_code(error) == 404:
                    return ReconcileResult(
                        ReconcileStatus.PROVEN_ABSENT,
                        evidence=f"GET /sandboxes/{resource.name} answered 404 after its delete",
                    )
                if not reconcile_retryable(error):
                    raise
            else:
                owner = observed.labels.get("loops.resource_id")
                if owner != resource.labels.get("loops.resource_id"):
                    # The name now belongs to a different sandbox; ours is not proven
                    # gone by it, and nothing here may touch it.
                    return ReconcileResult(
                        ReconcileStatus.UNKNOWN,
                        evidence=f"{resource.name} is held by another sandbox",
                    )
                if observed.status == TERMINATED and resource.name in self._deleted:
                    return ReconcileResult(
                        ReconcileStatus.PROVEN_ABSENT,
                        evidence=f"GET /sandboxes/{resource.name} shows it TERMINATED after "
                        "its delete; observed, not documented, Blaxel behaviour: a "
                        "terminated name is free for a new create",
                    )
            await self._sleep(POLL_SECONDS)

    async def list_owned(
        self, labels: Mapping[str, str]
    ) -> AsyncGenerator[Sequence[ProviderResource], None]:
        # Blaxel filters no labels server-side, so every page of the workspace is read.
        pages = self._api.list()
        try:
            async for page in pages:
                yield [
                    self._resource(observed)
                    for observed in page
                    if all(observed.labels.get(key) == value for key, value in labels.items())
                ]
        finally:
            await pages.aclose()

    # ---------------------------------------------------------------- usage

    def usage_environment(self, resource: ProviderResource) -> Mapping[str, str | SecretStr]:
        """What a separate process needs to reach this sandbox. The key is workspace-wide."""
        return {
            "BL_WORKSPACE": self._credentials.workspace,
            "BL_REGION": self._settings.region,
            "BL_API_KEY": self._credentials.api_key,
            "BLAXEL_SANDBOX_NAME": resource.name,
        }

    async def client(self, resource: ProviderResource) -> SandboxInstance:
        """The SDK's usage client for the sandbox, once it is ``DEPLOYED``; see ``usage.attach``."""
        return await usage.attach(
            resource,
            timeout_seconds=self.deadlines.attach_seconds,
            sleep=self._sleep,
        )

    async def close_client(self, client: SandboxInstance) -> None:
        await close_clients(client)

    async def alive(self, resource: ProviderResource) -> bool:
        """Whether the sandbox is still ours and ``DEPLOYED``; a ``404`` for its name is not.

        A name held by another sandbox means ours was deleted and the name reused.
        """
        try:
            observed = await self._api.get(resource.name)
        except Exception as error:
            if status_code(error) == 404:
                return False
            raise
        owner = observed.labels.get("loops.resource_id")
        return owner == resource.labels.get("loops.resource_id") and usage.deployed(observed.status)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("the Blaxel adapter is closed")

    async def close(self) -> None:
        """The SDK's control-plane client is process-wide and outlives this adapter."""
        self._closed = True
