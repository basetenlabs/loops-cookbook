"""Provide Blaxel sandboxes and image builds as resources that the registry owns.

Two kinds of request reach this adapter.

A ``sandbox`` request asks for one Blaxel sandbox. Its specification is::

    {"sandbox": <the SandboxInstance.create configuration: name, image, memory,
                 ttl, labels, and optionally region, network, extra_args>,
     "image": {"name": <the image the sandbox runs>, "force_build": bool,
               "build": <an image_build specification without its name>}}

``image`` is optional. When it is present and the image is not yet built, the
sandbox has one ``image_build`` prerequisite, keyed by image name, so concurrent
sandboxes of one image wait for one build. An image that already exists is adopted
and owned by no one, because the cookbook never deletes an image.

An ``image_build`` request asks for one build. Its provider identity is the builder
sandbox, which Blaxel names after the image. Its specification is ``{"name",
"base_image", "instructions", "local_files", "has_entrypoint", "memory",
"sandbox_version"}``. It is always ``release_after_create``, which means the builder is
deleted once the image is registered.

Every create carries the cookbook's ownership labels. A sandbox's name is the one its
caller chose, and this adapter validates it. ``creation_disposition`` classifies a
failed create. Reconciliation finds the sandbox by name. Absence counts only after
Blaxel accepted the sandbox's delete, and then shows as either a ``404`` for the name
or the sandbox still shown as ours but ``TERMINATED``. A ``TERMINATED`` sandbox's name
is free for a new create.
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
# Harbor's own length limit and alphabet for the names it derives.
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
    """Hold ``BL_WORKSPACE`` and ``BL_API_KEY``, as the SDK reads them from the environment."""

    workspace: str
    api_key: SecretStr


class BlaxelContractError(ValueError):
    """Raised for a request that this adapter cannot turn into a Blaxel resource."""


def _mapping(value: JsonValue | None, where: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise BlaxelContractError(f"{where} must be an object")
    return value


def valid_name(name: object) -> str:
    """Return ``name`` if it is a Blaxel resource name in the form Harbor derives.

    Any other value raises ``BlaxelContractError``.
    """
    if not isinstance(name, str) or len(name) > MAX_NAME or not NAME.match(name):
        raise BlaxelContractError(
            f"{name!r} is not a Blaxel name, which has at most {MAX_NAME} lowercase letters, "
            "digits and inner hyphens"
        )
    return name


class BlaxelAdapter:
    """Implement ``ResourceAdapter`` and ``SandboxProvider`` for Blaxel sandboxes."""

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
        # The names whose delete Blaxel accepted.
        self._deleted: set[str] = set()

    @property
    def deadlines(self) -> ProviderDeadlines:
        return self._settings.deadlines

    @property
    def region(self) -> str:
        """Return ``[providers.blaxel] region``, where every sandbox and image build runs."""
        return self._settings.region

    async def image_tags(self, name: str) -> frozenset[str] | None:
        """Return the tags of the built image ``name``, for ``ImageProvider``.

        It returns ``None`` when the workspace holds no built image by that name. The
        method makes one lookup within the reconcile deadline. A failure raises, and the
        caller reports that the image went unchecked.
        """
        async with asyncio.timeout(self.deadlines.reconcile_seconds):
            return await self._api.image_tags(name)

    # ------------------------------------------------------------- identity

    def desired_name(self, request: ResourceRequest, resource_id: str) -> str:
        """Return the caller's name for the sandbox, or a name unique to the resource.

        Harbor names each trial's sandbox. A custom rollout runtime that asks for
        sandboxes by specification alone gets ``loops-<resource id>``, so no two names
        collide.
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
            return []  # An existing image is adopted, and no one owns it.
        build = _mapping(image.get("build"), "image.build")
        return [
            ResourceRequest(
                provider=request.provider,
                kind=IMAGE_BUILD,
                logical_key=f"image-build:{name}",
                specification={**build, "name": name},
                # The identity is the name alone. An image is reused by name, so builds of
                # it that differ only in source paths or builder memory are one build.
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
        """Poll a created sandbox until it is ``DEPLOYED``, and fail if it dies.

        The caller's deadline bounds the wait. A sandbox that dies is known to exist, so
        the failure carries it, and the registry owns and releases it.
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
                # Our own build failed. Its builder exists, and it is ours to release.
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
            # Another process is building the same content-addressed image. Its image
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
        """Ask ``image_built``, and ask again after a transient failure.

        The caller's deadline bounds the retries.
        """
        while True:
            try:
                return await self._api.image_built(name)
            except Exception as error:
                if not reconcile_retryable(error):
                    raise
            await self._sleep(IMAGE_POLL_SECONDS)

    async def _builder(self, name: str) -> Observed | None:
        """Return the builder sandbox, or ``None`` when Blaxel says there is none."""
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
        """Read the uniquely named sandbox back, and never issue a second create.

        A just-created sandbox may not be readable yet. A ``404``, a ``429``, a ``502``
        to ``504`` and transport errors are therefore retried until the registry's
        reconcile deadline. A lookup proves absence only when the create is ``settled``.
        Then a ``404`` for the name means no such sandbox exists. The method waits for a
        sandbox to become ``DEPLOYED``. A sandbox that is ``FAILED``, ``TERMINATED`` or
        ``DELETING`` is found but unusable, so the registry owns and releases it, and its
        provision fails. A builder is usable only once its image is built.
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
            # No builder of ours was left, and the other owner's builder is theirs.
            return
        try:
            await self._api.delete(resource.name)
        except Exception as error:
            if status_code(error) != 404:
                raise
        self._deleted.add(resource.name)

    async def confirm_absent(self, resource: ProviderResource) -> ReconcileResult:
        """Wait until the sandbox is proven absent.

        It is absent once its name answers ``404``, or shows our sandbox ``TERMINATED``
        after our delete. The ``TERMINATED`` rule rests on observed Blaxel behaviour, not
        documented behaviour. A deleted sandbox stays readable as ``TERMINATED`` for many
        minutes, yet a create may take its name at once, so the old sandbox collides with
        nothing. The rule counts only for a name whose delete Blaxel accepted from this
        adapter. The registry's deadline bounds the wait.
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
                    # The name now belongs to a different sandbox. That does not prove
                    # ours is gone, and nothing here may touch the other sandbox.
                    return ReconcileResult(
                        ReconcileStatus.UNKNOWN,
                        evidence=f"{resource.name} is held by another sandbox",
                    )
                if observed.status == TERMINATED and resource.name in self._deleted:
                    return ReconcileResult(
                        ReconcileStatus.PROVEN_ABSENT,
                        evidence=f"GET /sandboxes/{resource.name} shows it TERMINATED after "
                        "its delete; by observed, not documented, Blaxel behaviour, a "
                        "terminated name is free for a new create",
                    )
            await self._sleep(POLL_SECONDS)

    async def list_owned(
        self, labels: Mapping[str, str]
    ) -> AsyncGenerator[Sequence[ProviderResource], None]:
        # Blaxel cannot filter by label on the server, so every page of the workspace is read.
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
        """Return what a separate process needs to reach this sandbox.

        The API key is workspace-wide.
        """
        return {
            "BL_WORKSPACE": self._credentials.workspace,
            "BL_REGION": self._settings.region,
            "BL_API_KEY": self._credentials.api_key,
            "BLAXEL_SANDBOX_NAME": resource.name,
        }

    async def client(self, resource: ProviderResource) -> SandboxInstance:
        """Return the SDK's usage client for the sandbox, once it is ``DEPLOYED``.

        ``usage.attach`` has the details.
        """
        return await usage.attach(
            resource,
            timeout_seconds=self.deadlines.attach_seconds,
            sleep=self._sleep,
        )

    async def close_client(self, client: SandboxInstance) -> None:
        await close_clients(client)

    async def alive(self, resource: ProviderResource) -> bool:
        """Return whether the sandbox is still ours and ``DEPLOYED``.

        A ``404`` for its name means it is not. A name held by another sandbox means ours
        was deleted and the name reused.
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
        """Mark the adapter closed.

        The SDK's control-plane client is process-wide and outlives this adapter, so
        nothing else is closed.
        """
        self._closed = True
