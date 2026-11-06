"""The Blaxel calls the adapter makes, as one narrow facade over the pinned SDK.

The adapter's ownership logic is tested against a fake of this class. Everything
that touches the SDK's own types, private members, or workarounds lives here.
"""

from __future__ import annotations

import dataclasses
import tempfile
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from blaxel.core import ImageBuildContext, ImageInstance, LocalFile, SandboxInstance
from blaxel.core.client import errors
from blaxel.core.client.client import client
from blaxel.core.client.models import MetadataLabels, Sandbox, SandboxSpec
from blaxel.core.client.types import Response

from loops_cookbook.evidence import JsonValue
from loops_cookbook.providers.blaxel.build_file import (
    BUILD_TOML_NAME,
    build_toml,
    check_task_build_region,
    with_region,
)
from loops_cookbook.providers.blaxel.incidents import close_clients

BUILDER_TTL = "5m"  # Harbor's: a builder the cookbook cannot delete reaps itself
LIST_PAGE = 100


@dataclass(frozen=True, slots=True)
class Observed:
    """A sandbox as the control plane reported it."""

    name: str
    labels: Mapping[str, str]
    status: str


def _observed(sandbox: SandboxInstance) -> Observed:
    metadata = sandbox.metadata.to_dict()
    labels = metadata.get("labels") or {}
    return Observed(
        name=str(metadata.get("name") or ""),
        labels={str(k): str(v) for k, v in labels.items()},
        status=str(sandbox.status),
    )


class _LabeledImage(ImageInstance):
    """An image whose builder sandbox carries the cookbook's ownership labels and runs
    in the provider's region.

    ``ImageInstance.build`` creates or updates a builder sandbox from a payload
    that labels it only as auto-generated and names no region; this is the one seam
    to set both at creation, so a crash mid-build still leaves a sandbox the journal
    can find, and Blaxel never places the builder elsewhere.
    """

    def __init__(self, context: ImageBuildContext, labels: Mapping[str, str], region: str) -> None:
        super().__init__(context)
        self._loops_labels = dict(labels)
        self._loops_region = region

    async def _create_sandbox_with_upload(
        self, sandbox: Sandbox
    ) -> tuple[Response[Sandbox], str | None]:
        """The builder's create, failing with the SDK's own typed error for its status.

        ``build`` turns a failed create into a bare ``RuntimeError`` that names the
        status only in its text, so a rate limit would read as an unknown failure.
        """
        response, upload_url = await super()._create_sandbox_with_upload(sandbox)
        if response.status_code >= 400:
            raise errors.from_response(response.status_code, response.content, response.headers)
        return response, upload_url

    def _create_sandbox_payload(
        self, name: str, memory: int = 4096, build_labels: dict[str, str] | None = None
    ) -> Sandbox:
        payload = super()._create_sandbox_payload(name, memory, build_labels)
        labels = payload.metadata.labels
        assert isinstance(labels, MetadataLabels)  # the SDK always labels a builder
        for key, value in self._loops_labels.items():
            labels[key] = value
        assert isinstance(payload.spec, SandboxSpec)  # the SDK always sets a spec
        payload.spec.region = self._loops_region
        return payload


def image_context(build: Mapping[str, JsonValue]) -> ImageBuildContext:
    """The SDK build context an ``image_build`` specification describes."""
    files = build.get("local_files") or []
    assert isinstance(files, list)
    local_files: list[LocalFile] = []
    for item in files:
        assert isinstance(item, dict)
        local_files.append(
            LocalFile(
                source_path=Path(str(item["source_path"])),
                destination_path=str(item["destination_path"]),
                context_name=str(item["context_name"]),
            )
        )
    instructions = build.get("instructions") or []
    assert isinstance(instructions, list)
    return ImageBuildContext(
        base_image=str(build["base_image"]),
        instructions=[str(line) for line in instructions],
        local_files=local_files,
        has_entrypoint=bool(build.get("has_entrypoint", False)),
    )


def image_build_specification(
    image: ImageInstance, *, memory: int, sandbox_version: str, slim: bool = True
) -> dict[str, JsonValue]:
    """The ``image_build`` specification an image describes; ``image_context`` inverts it.

    ``slim=False`` asks for Blaxel's full image, as Harbor always does. It is carried
    as data: the build writes the ``blaxel.toml`` that says so into its own temporary
    directory, so describing an image writes nothing.
    """
    context = image._context  # pyright: ignore[reportPrivateUsage]
    return {
        "base_image": context.base_image,
        "instructions": list(context.instructions),
        "local_files": [
            {
                "source_path": str(Path(item.source_path).resolve()),
                "destination_path": item.destination_path,
                "context_name": item.context_name,
            }
            for item in context.local_files
        ],
        "has_entrypoint": context.has_entrypoint,
        "memory": memory,
        "sandbox_version": sandbox_version,
        "slim": slim,
    }


def _with_build_toml(
    context: ImageBuildContext, directory: Path, region: str, *, slim: bool
) -> ImageBuildContext:
    """``context`` with a ``blaxel.toml`` naming ``region``, written in ``directory``.

    A task that ships its own ``blaxel.toml`` keeps it, as in Harbor: one that names
    no region gets the configured one added in the build's copy (the task's file is
    never written), and one that names another region is refused.
    """
    path = directory / BUILD_TOML_NAME
    files = list(context.local_files)
    for index, item in enumerate(files):
        if item.context_name == BUILD_TOML_NAME:
            source = Path(item.source_path)
            check_task_build_region(source, region)
            path.write_text(with_region(source.read_text(), region))
            files[index] = dataclasses.replace(item, source_path=path)
            return dataclasses.replace(context, local_files=files)
    path.write_text(build_toml(region, slim=slim))
    file = LocalFile(
        source_path=path, destination_path=BUILD_TOML_NAME, context_name=BUILD_TOML_NAME
    )
    return dataclasses.replace(context, local_files=[*files, file])


class BlaxelApi:
    """The control-plane calls, each raising the SDK's own errors."""

    async def create(self, config: Mapping[str, JsonValue]) -> Observed:
        sandbox = await SandboxInstance.create(dict(config), safe=False)
        try:
            return _observed(sandbox)
        finally:
            await close_clients(sandbox)

    async def get(self, name: str) -> Observed:
        sandbox = await SandboxInstance.get(name)
        try:
            return _observed(sandbox)
        finally:
            await close_clients(sandbox)

    async def delete(self, name: str) -> None:
        # ``delete`` is a descriptor the SDK's stubs type as taking no name.
        await SandboxInstance.delete(name)  # pyright: ignore[reportCallIssue]

    async def list(self) -> AsyncGenerator[Sequence[Observed], None]:
        """The workspace's sandboxes, one page at a time: the SDK lists no labels."""
        page = await SandboxInstance.list(limit=LIST_PAGE)
        while True:
            yield [_observed(sandbox) for sandbox in page.data]
            if not page.has_more:
                return
            page = await page.next_page()

    async def image_tags(self, name: str) -> frozenset[str] | None:
        """The tags of the workspace's sandbox image ``name``, or ``None`` when it has no
        image by that name (404). ``GET /images/sandbox/<name>/tags`` names every tag,
        where the image's own answer may only count them (since 22 Oct 2026). Raises when
        Blaxel answers otherwise, or holds more tags than one page."""
        response = await client.get_async_httpx_client().get(f"/images/sandbox/{name}/tags")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        listed = payload.get("data") if isinstance(payload, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if not isinstance(listed, list):
            raise TypeError(f"GET /images/sandbox/{name}/tags answered no list of tags")
        if (payload.get("meta") or {}).get("hasMore"):  # pyright: ignore[reportUnknownMemberType]
            raise ValueError(f"image {name} has more tags than one page holds")
        tags: set[str] = set()
        for entry in listed:  # pyright: ignore[reportUnknownVariableType]
            tag = entry.get("name") if isinstance(entry, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if not isinstance(tag, str) or not tag:
                # A tag unread would read as absent: the whole answer is unreadable.
                raise TypeError(f"GET /images/sandbox/{name}/tags answered a tag {entry!r}")
            tags.add(tag)
        return frozenset(tags)

    async def image_built(self, name: str) -> bool:
        """Whether the content-addressed image is built and consumable.

        Looked up by name: ``GET /images`` answers 502 on a workspace with many images
        (19k on 3 Oct 2026), and its first page never holds most names, so every start
        rebuilt an image that existed. ``GET /images/sandbox/<name>`` answers 404 for a
        name never built. Built means ``BUILT`` with a tag, listed in ``tags`` or, since
        22 Oct 2026, counted in ``tagCount`` instead: a sandbox starts from
        ``:latest``, so an image whose last build FAILED (or is still BUILDING) does not
        start even though an earlier tag is listed (3 Oct 2026).

        Only a 404 proves an image unbuilt, because a build in its place can leave
        ``:latest`` unstartable. Any other failed or unreadable answer raises.
        """
        response = await client.get_async_httpx_client().get(f"/images/sandbox/{name}")
        if response.status_code == 404:
            return False
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError(f"GET /images/sandbox/{name} answered {type(payload).__name__}")
        metadata = payload.get("metadata") or {}  # pyright: ignore[reportUnknownMemberType]
        spec = payload.get("spec") or {}  # pyright: ignore[reportUnknownMemberType]
        tagged = bool(spec.get("tags")) or (spec.get("tagCount") or 0) > 0  # pyright: ignore[reportUnknownMemberType]
        return metadata.get("status") == "BUILT" and tagged  # pyright: ignore[reportUnknownMemberType]

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
    ) -> None:
        """Build ``name``; ``build_memory`` and ``build_volume`` (MB) size the environment
        it is built in, and Blaxel's defaults apply where they are ``None``."""
        memory, version = build.get("memory", 4096), build.get("sandbox_version", "latest")
        try:
            with tempfile.TemporaryDirectory(prefix="loops-blaxel-build-") as directory:
                context = _with_build_toml(
                    image_context(build),
                    Path(directory),
                    region,
                    slim=build.get("slim") is not False,
                )
                await _LabeledImage(context, labels, region).build(
                    name=name,
                    memory=int(memory) if isinstance(memory, int | float) else 4096,
                    timeout=timeout,
                    sandbox_version=str(version),
                    build_memory=build_memory,
                    build_volume=build_volume,
                )
        finally:
            try:
                await SandboxInstance.update_ttl(name, BUILDER_TTL)
            except Exception:  # noqa: BLE001, S110 - a safety net; the release deletes it
                pass
