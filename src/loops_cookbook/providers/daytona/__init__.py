"""The built-in Daytona provider, which is `[providers.daytona]` and its adapter.

This package holds Daytona's error classification, ownership labels, lifecycle
defaults and discovery rules. Harbor's hooks stay in ``harbor/daytona.py``. The
credential is ``DAYTONA_API_KEY``, read from the environment, loaded from ``.env``. A
self-hosted Daytona also needs ``DAYTONA_API_URL`` there. Only
``[providers.daytona] target`` sets the target.
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
    """The ``[rollout_runtime.config.sandbox]`` table for a Daytona sandbox.

    The table sits beside ``provider = "daytona"``.
    """

    image: str | None = Field(
        default=None,
        min_length=1,
        description="The registry image the sandbox runs, such as python:3.12-slim. When "
        "it is unset, the sandbox runs Daytona's default sandbox image.",
    )
    snapshot: str | None = Field(
        default=None,
        min_length=1,
        description="The name of a Daytona snapshot that the sandbox starts from instead "
        "of an image. The snapshot fixes both the sandbox's image and its size.",
    )
    cpu: int | None = Field(
        default=None,
        gt=0,
        description="The sandbox's vCPUs. When it is unset, [providers.daytona] cpu applies.",
    )
    memory_gib: int | None = Field(
        default=None,
        gt=0,
        description="The sandbox's memory, in GiB. When it is unset, "
        "[providers.daytona] memory_gib applies.",
    )
    disk_gib: int | None = Field(
        default=None,
        gt=0,
        description="The sandbox's disk, in GiB, not counting the image. When it is unset, "
        "[providers.daytona] disk_gib applies.",
    )

    @model_validator(mode="after")
    def _sized_only_with_an_image(self) -> DaytonaSandbox:
        if self.image is not None and self.snapshot is not None:
            raise ValueError("a Daytona sandbox starts from an image or a snapshot, not both")
        if self.image is None and (self.cpu or self.memory_gib or self.disk_gib):
            raise ValueError(
                "cpu, memory_gib and disk_gib size only a sandbox built from an image; a "
                "snapshot and Daytona's default sandbox fix their own size, so name an image "
                "to size the sandbox"
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
    """The Daytona sandbox kind that an Episode declares. Its client is the SDK's ``AsyncSandbox``.

    The class avoids importing the SDK, because only a run that uses Daytona needs it.
    """

    provider = "daytona"
    spec = DaytonaSandbox

    async def write_file(self, path: str, content: str) -> None:
        """Write UTF-8 text to ``path`` through the attached client.

        SDK failures propagate to the caller.
        """
        box = await self.client()
        await box.fs.upload_file(content.encode("utf-8"), path)

    async def read_file(self, path: str) -> str | None:
        """Return the text of the file at ``path``, or ``None`` when the file is missing.

        Daytona's 404 does not say whether the file or the sandbox is missing. A liveness
        check after the read tells them apart, and raises ``SandboxGone`` when the sandbox
        is gone. Any other failed read raises the SDK's own exception.
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
    """Build the `[providers.daytona]` adapter.

    It fails before any Daytona call when a setting or the credential is missing.
    """
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
    # The SDK reads DAYTONA_TARGET when its configuration names no target. A variable
    # that differs from the table would then decide where sandboxes go, unseen.
    if (variable := environ.get("DAYTONA_TARGET")) and variable != daytona.target:
        raise RuntimeConfigurationError(
            f"DAYTONA_TARGET is {variable!r} in the environment, but [providers.daytona] "
            f"target is {daytona.target!r}; set the target in [providers.daytona] and unset "
            "DAYTONA_TARGET"
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
