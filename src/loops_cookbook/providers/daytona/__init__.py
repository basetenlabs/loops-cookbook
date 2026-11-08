"""The built-in Daytona provider: `[providers.daytona]` and its adapter.

Daytona error classification, ownership labels, lifecycle defaults, and discovery
rules live in this package; Harbor's hooks stay in ``harbor/daytona.py``. The
credential is ``DAYTONA_API_KEY`` (and ``DAYTONA_API_URL`` for a self-hosted
Daytona), from the environment, loaded from ``.env``. The target is
``[providers.daytona] target`` alone.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import Field, SecretStr, ValidationError, model_validator

from loops_cookbook.evidence import JsonValue
from loops_cookbook.resources.api import SandboxKind, SandboxSpec
from loops_cookbook.runtime.context import RuntimeConfigurationError

if TYPE_CHECKING:
    from loops_cookbook.providers.daytona.adapter import DaytonaAdapter

__all__ = ["Daytona", "DaytonaSandbox", "create_adapter"]


class DaytonaSandbox(SandboxSpec):
    """``[rollout_runtime.config.sandbox]`` for a Daytona sandbox, beside ``provider = "daytona"``."""

    image: str | None = Field(
        default=None,
        min_length=1,
        description="The registry image it runs, such as python:3.12-slim. Unset, "
        "Daytona's default sandbox image.",
    )
    snapshot: str | None = Field(
        default=None,
        min_length=1,
        description="A Daytona snapshot it starts from, by name, instead of an image. The "
        "snapshot fixes both its image and its size.",
    )
    cpu: int | None = Field(
        default=None, gt=0, description="Its vCPUs. Unset, [providers.daytona] cpu."
    )
    memory_gib: int | None = Field(
        default=None, gt=0, description="Its memory in GiB. Unset, [providers.daytona] memory_gib."
    )
    disk_gib: int | None = Field(
        default=None,
        gt=0,
        description="Its disk in GiB, the image not counted. Unset, [providers.daytona] disk_gib.",
    )

    @model_validator(mode="after")
    def _sized_only_with_an_image(self) -> DaytonaSandbox:
        if self.image is not None and self.snapshot is not None:
            raise ValueError("a Daytona sandbox starts from an image or a snapshot, not both")
        if self.image is None and (self.cpu or self.memory_gib or self.disk_gib):
            raise ValueError(
                "cpu, memory_gib and disk_gib size a sandbox built from an image; a snapshot, "
                "and Daytona's default sandbox, fix their own size, so name an image to size it"
            )
        return self

    def specification(self) -> dict[str, JsonValue]:
        sandbox: dict[str, JsonValue] = {}
        if self.image is not None:
            sandbox["image"] = {"base": self.image}
        if self.snapshot is not None:
            sandbox["snapshot"] = self.snapshot
        resources: dict[str, JsonValue] = {
            key: value
            for key, value in (
                ("cpu", self.cpu),
                ("memory", self.memory_gib),
                ("disk", self.disk_gib),
            )
            if value is not None
        }
        if resources:
            sandbox["resources"] = resources
        return {"sandbox": sandbox}


class Daytona(SandboxKind[Any]):
    """An Episode's concrete Daytona sandbox kind; its client is the SDK's ``AsyncSandbox``.

    Named without importing the SDK, which only a run that uses it needs.
    """

    provider = "daytona"
    spec = DaytonaSandbox

    async def write_file(self, path: str, content: str) -> None:
        """Write UTF-8 text through the attached client; SDK failures propagate."""
        box = await self.client()
        await box.fs.upload_file(content.encode("utf-8"), path)

    async def read_file(self, path: str) -> str | None:
        """The text of the file at ``path``, or ``None`` when the sandbox is alive without it.

        Daytona's 404 does not say whether the file or the sandbox is missing, so the
        sandbox's liveness, asked after the read, tells them apart: one that is gone
        raises ``SandboxGone``. Any other failed read raises as the SDK raised it.
        """
        from loops_cookbook.providers.daytona.errors import not_found  # needs the SDK

        box = await self.client()
        try:
            data: bytes = await box.fs.download_file(path)
        except Exception as error:
            if not not_found(error):
                raise
        else:
            return data.decode()
        await self.ensure_alive()
        return None


def create_adapter(
    settings: Mapping[str, JsonValue], *, environ: Mapping[str, str] | None = None
) -> DaytonaAdapter:
    """The `[providers.daytona]` factory; fails before any Daytona call when a setting or
    the credential is missing."""
    environ = os.environ if environ is None else environ
    if not environ.get("DAYTONA_API_KEY"):
        raise RuntimeConfigurationError(
            "Daytona needs DAYTONA_API_KEY in the environment or in a .env beside the "
            "training config"
        )
    try:
        from loops_cookbook.providers.daytona.adapter import Credentials, DaytonaAdapter
        from loops_cookbook.providers.daytona.api import DaytonaApi
        from loops_cookbook.providers.daytona.settings import DaytonaSettings
    except ImportError as error:
        raise RuntimeConfigurationError(
            "[providers.daytona] needs the Daytona SDK; install the `daytona` extra"
        ) from error
    try:
        daytona = DaytonaSettings.model_validate(dict(settings))
    except ValidationError as error:
        raise RuntimeConfigurationError(f"[providers.daytona] is invalid: {error}") from error
    # The SDK reads DAYTONA_TARGET when its configuration names no target, so one that
    # differs from the table's would decide where sandboxes go behind the table.
    if (variable := environ.get("DAYTONA_TARGET")) and variable != daytona.target:
        raise RuntimeConfigurationError(
            f"DAYTONA_TARGET is {variable!r} in the environment, but [providers.daytona] "
            f"target is {daytona.target!r}; the target is the table's, so set it there and "
            "unset DAYTONA_TARGET"
        )
    credentials = Credentials(
        SecretStr(environ["DAYTONA_API_KEY"]), environ.get("DAYTONA_API_URL") or None
    )
    api = DaytonaApi(
        api_key=credentials.api_key.get_secret_value(),
        api_url=credentials.api_url,
        target=daytona.target,
    )
    return DaytonaAdapter(daytona, credentials, api)
