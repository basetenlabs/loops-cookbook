"""A sandbox provider with no cloud behind it: each sandbox is a directory.

It stands in for a provider such as Blaxel so the agent example runs anywhere, and it
shows what a provider owes the run's resource registry. The registry decides when a
sandbox exists; this adapter only creates, finds, and deletes directories, and
proves a sandbox is gone by looking for its directory, which is authoritative on a
local disk. It is also a ``SandboxProvider``, so an Episode can name ``Directory`` as
its sandbox kind: the client is the directory's ``Path``, and the sandbox is alive
while the directory exists.

    [providers.local_sandbox]
    adapter = "local_sandbox:create"
    root = "/tmp/loops-local-sandboxes"
    create_timeout_seconds = 30  # and the other five deadlines, as train.toml sets them
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from collections.abc import AsyncGenerator, Mapping, Sequence
from pathlib import Path
from typing import ClassVar

from pydantic import SecretStr

from loops_cookbook.resources import (
    ProviderDeadlines,
    ProviderResource,
    ReconcileResult,
    ReconcileStatus,
    ResourceIntent,
    ResourceRequest,
    SandboxKind,
)
from loops_cookbook.runtime import JsonValue

LABELS = ".labels.json"


class Directory(SandboxKind[Path]):
    """A directory sandbox, as an Episode's third type argument: its client is its ``Path``."""

    provider = "local_sandbox"


class LocalSandboxes:
    """``ResourceAdapter`` and ``SandboxProvider`` for directory sandboxes under ``root``."""

    sandbox_kind: ClassVar[type[Directory]] = Directory

    def __init__(self, root: Path, deadlines: ProviderDeadlines) -> None:
        self._root = root
        self._deadlines = deadlines

    @property
    def deadlines(self) -> ProviderDeadlines:
        return self._deadlines

    async def prerequisites(self, request: ResourceRequest) -> Sequence[ResourceRequest]:
        return ()

    def desired_name(self, request: ResourceRequest, resource_id: str) -> str:
        return f"sandbox-{resource_id}"

    async def create(self, intent: ResourceIntent) -> ProviderResource:
        """The directory appears whole, labels included, or not at all."""
        final = self._root / intent.desired_name
        if final.exists():
            raise FileExistsError(f"sandbox {intent.desired_name} already exists")
        self._root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=self._root, prefix=".creating-"))
        (staging / LABELS).write_text(json.dumps(dict(intent.ownership_labels)))
        os.rename(staging, final)
        return self._resource(final)

    async def reconcile_create(
        self, intent: ResourceIntent, *, settled: bool = False
    ) -> ReconcileResult:
        # A directory lookup is authoritative whether or not the create is settled.
        return self._lookup(self._root / intent.desired_name)

    async def validate_ownership(self, intent: ResourceIntent, resource: ProviderResource) -> bool:
        return all(
            resource.labels.get(key) == value for key, value in intent.ownership_labels.items()
        )

    async def release(self, resource: ProviderResource) -> None:
        await asyncio.to_thread(shutil.rmtree, self._root / resource.name, ignore_errors=False)

    async def confirm_absent(self, resource: ProviderResource) -> ReconcileResult:
        return self._lookup(self._root / resource.name)

    async def list_owned(
        self, labels: Mapping[str, str]
    ) -> AsyncGenerator[Sequence[ProviderResource], None]:
        if not self._root.is_dir():
            return
        found = [self._resource(path) for path in self._root.iterdir() if (path / LABELS).is_file()]
        yield [r for r in found if all(r.labels.get(k) == v for k, v in labels.items())]

    def usage_environment(self, resource: ProviderResource) -> Mapping[str, str | SecretStr]:
        return {"SANDBOX_DIR": str(self._root / resource.name)}

    async def client(self, resource: ProviderResource) -> Path:
        return self._root / resource.name

    async def close_client(self, client: Path) -> None:
        return None  # a path holds nothing open

    async def alive(self, resource: ProviderResource) -> bool:
        return (self._root / resource.name).is_dir()

    async def close(self) -> None:
        return None

    def _resource(self, path: Path) -> ProviderResource:
        labels = json.loads((path / LABELS).read_text())
        return ProviderResource(provider_id=path.name, name=path.name, labels=labels)

    def _lookup(self, path: Path) -> ReconcileResult:
        if (path / LABELS).is_file():
            return ReconcileResult(ReconcileStatus.FOUND, self._resource(path))
        if path.exists():
            return ReconcileResult(ReconcileStatus.UNKNOWN, evidence=f"{path} has no labels")
        return ReconcileResult(ReconcileStatus.PROVEN_ABSENT, evidence=f"no directory at {path}")


DEADLINES = {
    "create_timeout_seconds": "create_seconds",
    "reconcile_timeout_seconds": "reconcile_seconds",
    "release_timeout_seconds": "release_seconds",
    "confirm_absent_timeout_seconds": "confirm_absent_seconds",
    "list_timeout_seconds": "list_seconds",
    "attach_timeout_seconds": "attach_seconds",
}


def create(settings: Mapping[str, JsonValue]) -> LocalSandboxes:
    """The ``[providers.local_sandbox]`` factory: a root, and every deadline, explicitly."""
    root = settings.get("root")
    if not isinstance(root, str) or not root:
        raise ValueError("[providers.local_sandbox] needs root, the directory sandboxes live in")
    deadlines: dict[str, float] = {}
    for key, field in DEADLINES.items():
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
            raise ValueError(f"[providers.local_sandbox] needs {key}, a positive number")
        deadlines[field] = float(value)
    return LocalSandboxes(Path(root).expanduser(), ProviderDeadlines(**deadlines))
