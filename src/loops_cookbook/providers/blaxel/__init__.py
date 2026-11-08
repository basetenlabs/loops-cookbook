"""The built-in Blaxel provider: `[providers.blaxel]` and its adapter.

Blaxel status, exception matching, SDK workarounds, ownership labels, and discovery
rules live in this package; Harbor's private hooks stay in ``harbor/blaxel.py``.
Credentials are ``BL_WORKSPACE`` and ``BL_API_KEY``: the SDK reads them from the
environment, loaded from ``.env``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import Field, SecretStr, ValidationError

from loops_cookbook.evidence import JsonValue
from loops_cookbook.resources.api import SandboxKind, SandboxSpec
from loops_cookbook.runtime.context import RuntimeConfigurationError

if TYPE_CHECKING:
    from blaxel.core.sandbox import SandboxInstance  # noqa: F401 - Blaxel names it as a string

    from loops_cookbook.providers.blaxel.adapter import BlaxelAdapter

__all__ = ["Blaxel", "BlaxelSandbox", "create_adapter"]


# A sandbox's ttl as Blaxel's SandboxRuntime documents it: "Units s, m, h, d, w (e.g.
# '30m', '24h', '7d', '2w')"; a zero would have it deleted as it is created.
TTL = r"^[1-9][0-9]*[smhdw]$"


class BlaxelSandbox(SandboxSpec):
    """``[rollout_runtime.config.sandbox]`` for a Blaxel sandbox, beside ``provider = "blaxel"``."""

    image: str = Field(min_length=1, description="The image it runs.")
    memory_mb: int = Field(gt=0, description="Its memory, as Harbor's memory_mb.")
    ttl: str = Field(
        min_length=1,
        pattern=TTL,
        description="How long Blaxel keeps it, such as 1h, required as Harbor's Blaxel ttl is: "
        "a backstop, as the run releases it. A whole number and one unit: s, m, h, d or w.",
    )

    def specification(self) -> dict[str, JsonValue]:
        return {"sandbox": {"image": self.image, "memory": self.memory_mb, "ttl": self.ttl}}


class Blaxel(SandboxKind["SandboxInstance"]):
    """An Episode's concrete Blaxel sandbox kind; its client is ``SandboxInstance``.

    Named without importing the SDK, which only a run that uses it needs.
    """

    provider = "blaxel"
    spec = BlaxelSandbox

    async def write_file(self, path: str, content: str) -> None:
        """Write text through the attached client; SDK failures propagate to the caller."""
        box = await self.client()
        await box.fs.write(path, content)

    async def read_file(self, path: str) -> str | None:
        """The text of the file at ``path``, or ``None`` when the sandbox is alive without it.

        Blaxel's 404 does not say whether the file or the sandbox is missing, so the
        sandbox's liveness, asked after the read, tells them apart: one that is gone
        raises ``SandboxGone``. Any other failed read raises as the SDK raised it.
        """
        from blaxel.core.sandbox.types import ResponseError

        box = await self.client()
        try:
            return await box.fs.read(path)
        except ResponseError as error:
            if error.response.status_code != 404:
                raise
        await self.ensure_alive()
        return None


_CREDENTIALS = ("BL_WORKSPACE", "BL_API_KEY")


def create_adapter(
    settings: Mapping[str, JsonValue], *, environ: Mapping[str, str] | None = None
) -> BlaxelAdapter:
    """The `[providers.blaxel]` factory; fails before any Blaxel call when a setting,
    a deadline, or a credential is missing."""
    environ = os.environ if environ is None else environ
    missing = [name for name in _CREDENTIALS if not environ.get(name)]
    if missing:
        raise RuntimeConfigurationError(
            f"Blaxel needs {' and '.join(missing)} in the environment or in a .env beside "
            "the training config"
        )
    try:
        from loops_cookbook.providers.blaxel.adapter import BlaxelAdapter, Credentials
        from loops_cookbook.providers.blaxel.api import BlaxelApi
        from loops_cookbook.providers.blaxel.settings import BlaxelSettings
    except ImportError as error:
        raise RuntimeConfigurationError(
            "[providers.blaxel] needs the Blaxel SDK; install the `blaxel` extra"
        ) from error
    try:
        blaxel = BlaxelSettings.model_validate(dict(settings))
    except ValidationError as error:
        raise RuntimeConfigurationError(f"[providers.blaxel] is invalid: {error}") from error
    credentials = Credentials(environ["BL_WORKSPACE"], SecretStr(environ["BL_API_KEY"]))
    return BlaxelAdapter(blaxel, credentials, BlaxelApi())
