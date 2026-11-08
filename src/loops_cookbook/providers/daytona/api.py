"""Make the adapter's Daytona calls through one narrow facade over the pinned async SDK.

The adapter's ownership logic is tested against a fake of this class. Everything that
touches the SDK's own types lives here. The module reaches two private members of an
image. The first is the build context, ``Image._context_list``, which a declarative
image carries and its JSON description must carry too. The second is the Dockerfile,
``Image._dockerfile``, which the module sets when it rebuilds the image from that
description.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daytona import (
    AsyncDaytona,
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    DaytonaConfig,
    Image,
    ListSandboxesQuery,
    Resources,
)
from daytona.common.errors import DaytonaBadRequestError
from daytona.common.image import Context

from loops_cookbook.evidence import JsonValue
from loops_cookbook.providers.daytona.usage import state_of

LIST_PAGE = 100
# The client's connection cap. With the SDK's default cap of 250, calls queued behind a
# run's ~400 live sandboxes. This cap leaves room for any run near that size and still
# bounds a runaway.
CONNECTIONS = 1024
# The keys of a sandbox's ``image``, which is a declarative build or a registry image.
DOCKERFILE = "dockerfile"
CONTEXT = "context"
BASE = "base"


@dataclass(frozen=True, slots=True)
class Observed:
    """A sandbox as the control plane reported it."""

    id: str
    name: str
    labels: Mapping[str, str]
    state: str


def observed(sandbox: Any) -> Observed:
    """Return the facts the adapter reads from an SDK sandbox.

    The state is Daytona's wire value.
    """
    return Observed(
        id=str(sandbox.id),
        name=str(sandbox.name or sandbox.id),
        labels={str(k): str(v) for k, v in (sandbox.labels or {}).items()},
        state=state_of(sandbox),
    )


def describe_image(image: Image) -> dict[str, JsonValue]:
    """Return a declarative image as JSON, which holds its Dockerfile text and build context."""
    contexts = image._context_list  # pyright: ignore[reportPrivateUsage]
    return {
        DOCKERFILE: image.dockerfile(),
        CONTEXT: [
            {"source_path": item.source_path, "archive_path": item.archive_path}
            for item in contexts
        ],
    }


def dockerfile_image(dockerfile: Path) -> dict[str, JsonValue]:
    """Return a declarative build of ``dockerfile``, with its directory as the build context."""
    return describe_image(Image.from_dockerfile(dockerfile))


def image_from(description: Mapping[str, JsonValue]) -> Image | str:
    """Return the SDK image that a ``describe_image`` result or ``{"base": ref}`` describes."""
    if BASE in description:
        return str(description[BASE])
    image = Image()
    image._dockerfile = str(description[DOCKERFILE])  # pyright: ignore[reportPrivateUsage]
    items = description.get(CONTEXT) or []
    assert isinstance(items, list)
    contexts: list[Context] = []
    for item in items:
        assert isinstance(item, dict)
        archive = item.get("archive_path")
        contexts.append(
            Context(
                source_path=str(item["source_path"]),
                archive_path=None if archive is None else str(archive),
            )
        )
    image._context_list = contexts  # pyright: ignore[reportPrivateUsage]
    return image


def create_params(
    config: Mapping[str, Any],
) -> CreateSandboxFromImageParams | CreateSandboxFromSnapshotParams:
    """Return the SDK's create parameters for a sandbox ``config`` that the adapter completed."""
    fields = dict(config)
    image = fields.pop("image", None)
    resources = fields.pop("resources", None)
    if image is not None:
        sized = Resources(**resources) if resources else None
        return CreateSandboxFromImageParams(image=image_from(image), resources=sized, **fields)
    if resources:
        raise ValueError("Daytona sizes a snapshot's sandboxes by the snapshot; name no resources")
    return CreateSandboxFromSnapshotParams(**fields)


class DaytonaApi:
    """Make the control-plane calls. Each call raises the SDK's own errors.

    The class owns the provider's one ``AsyncDaytona`` client, configured from
    ``[providers.daytona]``. Harbor's environment runs commands and file transfers
    through the sandboxes that this client returns.
    """

    def __init__(self, *, api_key: str, api_url: str | None, target: str | None) -> None:
        self._config = DaytonaConfig(
            api_key=api_key,
            api_url=api_url,
            target=target,
            connection_pool_maxsize=CONNECTIONS,
        )
        self._client: AsyncDaytona | None = None

    def client(self) -> AsyncDaytona:
        if self._client is None:
            self._client = AsyncDaytona(self._config)
        return self._client

    async def create(self, config: Mapping[str, Any], timeout: float) -> Observed:
        """Create a sandbox and wait until Daytona reports it started, within ``timeout``.

        A configuration that the SDK's parameters cannot take fails before anything is
        sent. It raises the SDK's own pre-send ``DaytonaBadRequestError``.
        """
        try:
            params = create_params(config)
        except Exception as error:
            raise DaytonaBadRequestError(f"Failed to create sandbox: {error}") from error
        return observed(await self.client().create(params, timeout=timeout))

    async def get(self, id_or_name: str) -> Observed:
        return observed(await self.client().get(id_or_name))

    async def delete(self, sandbox_id: str) -> None:
        """Ask Daytona to delete the sandbox, and return once Daytona accepts the delete."""
        await (await self.client().get(sandbox_id)).delete()

    async def list(self, labels: Mapping[str, str]) -> AsyncGenerator[Sequence[Observed], None]:
        """Yield every sandbox that carries ``labels``, one fetched page at a time.

        Daytona does the filtering.
        """
        query = ListSandboxesQuery(labels=dict(labels), limit=LIST_PAGE)
        page: list[Observed] = []
        async for sandbox in self.client().list(query):
            page.append(observed(sandbox))
            if len(page) == LIST_PAGE:
                yield page
                page = []
        yield page

    async def close(self) -> None:
        if self._client is not None:
            client, self._client = self._client, None
            await client.close()
