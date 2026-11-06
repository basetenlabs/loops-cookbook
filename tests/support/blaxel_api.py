"""A fake Blaxel control plane behind the adapter's ``Api`` facade, and SDK-shaped errors."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from pydantic import SecretStr

from loops_cookbook.providers.blaxel.adapter import Credentials
from loops_cookbook.providers.blaxel.api import Observed
from loops_cookbook.providers.blaxel.settings import BlaxelSettings

SETTINGS = BlaxelSettings(
    region="us-pdx-1",
    create_timeout_seconds=2,
    reconcile_timeout_seconds=1,
    image_build_timeout_seconds=3,
    release_timeout_seconds=1,
    confirm_absent_timeout_seconds=1,
    list_timeout_seconds=1,
    attach_timeout_seconds=1,
    labels={"owner": "team-a"},
)
CREDENTIALS = Credentials("team", SecretStr("bl-key-123"))
ENVIRON = {"BL_WORKSPACE": "team", "BL_API_KEY": "bl-key-123"}
TABLE = {
    "region": "us-pdx-1",
    "create_timeout_seconds": 300,
    "reconcile_timeout_seconds": 120,
    "image_build_timeout_seconds": 1200,
    "release_timeout_seconds": 120,
    "confirm_absent_timeout_seconds": 60,
    "list_timeout_seconds": 60,
    "attach_timeout_seconds": 60,
}


class StatusError(Exception):
    """``SandboxAPIError`` as the SDK raises it: a status and a code, no body."""

    def __init__(self, status: int | None, message: str = "blaxel said no") -> None:
        super().__init__(message)
        self.status_code = status
        self.code = None


class BodyError(Exception):
    """A sandbox-API ``ResponseError`` carrying Blaxel's structured error body."""

    def __init__(
        self, code: str, safe_to_retry: bool, status: int, dispatch_state: str = "not_dispatched"
    ) -> None:
        super().__init__(code)
        self.data = {
            "error": {
                "code": code,
                "dispatch_state": dispatch_state,
                "safe_to_retry_request": safe_to_retry,
                "retryable": safe_to_retry,
            },
            "status": status,
        }


def not_dispatched() -> BodyError:
    return BodyError("WORKLOAD_UNAVAILABLE", True, 404)


def workload_failed() -> BodyError:
    return BodyError("WORKLOAD_FAILED", False, 503)


@dataclass
class Sandbox:
    name: str
    labels: dict[str, str]
    statuses: list[str]  # the status each later read reports; the last one sticks
    config: dict[str, Any] = field(default_factory=dict)

    def read(self) -> Observed:
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return Observed(self.name, dict(self.labels), status)


@dataclass
class FakeApi:
    sandboxes: dict[str, Sandbox] = field(default_factory=dict)
    images: set[str] = field(default_factory=set)
    image_tag_names: dict[str, frozenset[str]] = field(default_factory=dict)  # imported
    calls: list[tuple[str, str]] = field(default_factory=list)
    deploy: list[str] = field(default_factory=lambda: ["DEPLOYED"])  # statuses after create
    create_errors: list[BaseException] = field(default_factory=list)
    land_before_error: bool = False  # a create error arrives after the sandbox exists
    get_errors: list[BaseException] = field(default_factory=list)
    delete_errors: list[BaseException] = field(default_factory=list)
    linger: int = 0  # reads that still see a deleted sandbox, as DELETING
    terminated: bool = False  # as Blaxel: a deleted sandbox stays readable as TERMINATED
    build_errors: list[BaseException] = field(default_factory=list)
    build_refusals: list[BaseException] = field(default_factory=list)  # before any builder
    built_elsewhere: bool = False  # a failed build's image appears anyway
    image_errors: list[BaseException | None] = field(default_factory=list)  # None answers
    build_gate: asyncio.Event | None = None
    build_labels: list[dict[str, str]] = field(default_factory=list)  # each build's builder
    build_sizes: list[tuple[int | None, int | None]] = field(default_factory=list)  # memory, volume
    create_gate: asyncio.Event | None = None
    creating: int = 0  # creates in flight now
    created: list[dict[str, Any]] = field(default_factory=list)  # each landed create's config
    most_creating: int = 0
    list_page_size: int = 0  # sandboxes per listed page; 0 lists them in one page

    async def create(self, config):
        self.calls.append(("create", config["name"]))
        self.creating += 1
        self.most_creating = max(self.most_creating, self.creating)
        try:
            if self.create_gate is not None:
                await self.create_gate.wait()
        finally:
            self.creating -= 1
        taken = self.sandboxes.get(config["name"])
        if taken is not None and taken.statuses != ["TERMINATED"]:
            # As Blaxel: a name is taken until it is gone or terminated.
            raise StatusError(409, f"Sandbox '{config['name']}' already exists")
        if self.create_errors:
            error = self.create_errors.pop(0)
            if self.land_before_error:
                self._land(config)
            raise error
        return self._land(config).read()

    def _land(self, config):
        self.created.append(dict(config))
        sandbox = Sandbox(config["name"], dict(config["labels"]), list(self.deploy), dict(config))
        self.sandboxes[sandbox.name] = sandbox
        return sandbox

    async def get(self, name):
        self.calls.append(("get", name))
        if self.get_errors:
            raise self.get_errors.pop(0)
        if name not in self.sandboxes:
            raise StatusError(404, f"Sandbox '{name}' not found")
        return self.sandboxes[name].read()

    async def delete(self, name):
        self.calls.append(("delete", name))
        if self.delete_errors:
            raise self.delete_errors.pop(0)
        if name not in self.sandboxes:
            raise StatusError(404)
        if self.terminated:
            self.sandboxes[name].statuses = ["DELETING", "TERMINATED"]
        elif self.linger:
            self.sandboxes[name].statuses = ["DELETING"] * self.linger + ["GONE"]
        else:
            del self.sandboxes[name]

    async def list(self):
        """The workspace in pages of ``list_page_size``, as the SDK pages it."""
        listed = [sandbox.read() for sandbox in self.sandboxes.values()]
        size = self.list_page_size or max(len(listed), 1)
        for start in range(0, max(len(listed), 1), size):
            self.calls.append(("list", "*"))
            yield listed[start : start + size]

    async def image_tags(self, name):
        self.calls.append(("image_tags", name))
        if self.image_errors and (error := self.image_errors.pop(0)) is not None:
            raise error
        return self.image_tag_names.get(name)

    async def image_built(self, name):
        self.calls.append(("image_built", name))
        if self.image_errors and (error := self.image_errors.pop(0)) is not None:
            raise error
        return name in self.images

    async def build_image(
        self, name, build, labels, timeout, *, region, build_memory=None, build_volume=None
    ):
        self.calls.append(("build", name))
        self.build_labels.append(dict(labels))
        self.build_sizes.append((build_memory, build_volume))
        if self.build_refusals:  # Blaxel turned the builder's create away: nothing exists
            raise self.build_refusals.pop(0)
        if self.build_gate is not None:
            if not self.built_elsewhere:  # Blaxel creates the builder, then builds
                self.sandboxes[name] = Sandbox(name, dict(labels), ["DEPLOYING"])
            await self.build_gate.wait()
        if self.build_errors:
            if self.built_elsewhere:  # another process's build, under its own labels
                self.sandboxes[name] = Sandbox(name, {"loops.resource_id": "other"}, ["DEPLOYED"])
                self.images.add(name)
            else:  # our builder was created, then the build failed
                self.sandboxes[name] = Sandbox(name, dict(labels), ["FAILED"])
            raise self.build_errors.pop(0)
        self.sandboxes[name] = Sandbox(name, dict(labels), ["DEPLOYED"], {"build": dict(build)})
        self.images.add(name)

    def tick(self):
        """Advance lingering deletes: a sandbox whose last status is GONE disappears."""
        for name, sandbox in list(self.sandboxes.items()):
            if sandbox.statuses == ["GONE"]:
                del self.sandboxes[name]


def sandbox_request(key="trial__env", name="harbor-env-trial-env-abc123", **image):
    from loops_cookbook.resources.api import ResourceRequest

    specification: dict[str, Any] = {
        "sandbox": {
            "name": name,
            "image": "sandbox/harbor-img-env-abc123:latest",
            "memory": 4096,
            "ttl": "24h",
            "labels": {"created-by": "harbor", "session-id": key},
        }
    }
    if image:
        specification["image"] = {
            "name": image.get("image_name", "harbor-img-env-abc123"),
            "force_build": image.get("force_build", False),
            "build": {
                "base_image": "python:3.12-slim",
                "instructions": ["RUN pip install pytest"],
                "local_files": [],
                "has_entrypoint": False,
                "memory": 4096,
                "sandbox_version": "latest",
            },
        }
    return ResourceRequest(
        provider="blaxel", kind="sandbox", logical_key=key, specification=specification
    )
