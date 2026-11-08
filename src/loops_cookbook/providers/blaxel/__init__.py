"""The built-in Blaxel provider, which is `[providers.blaxel]` and its adapter.

This package holds Blaxel's status rules, exception matching, SDK workarounds,
ownership labels and discovery rules. Harbor's private hooks stay in
``harbor/blaxel.py``. The credentials are ``BL_WORKSPACE`` and ``BL_API_KEY``. The SDK
reads them from the environment, loaded from ``.env``.
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


# A sandbox's ttl, in the form Blaxel's SandboxRuntime documents: "Units s, m, h, d, w
# (e.g. '30m', '24h', '7d', '2w')". A zero is refused, because Blaxel would delete the
# sandbox as it is created.
TTL = r"^[1-9][0-9]*[smhdw]$"


class BlaxelSandbox(SandboxSpec):
    """The ``[rollout_runtime.config.sandbox]`` table for a Blaxel sandbox.

    The table sits beside ``provider = "blaxel"``.
    """

    image: str = Field(min_length=1, description="The image the sandbox runs.")
    memory_mb: int = Field(
        gt=0, description="The sandbox's memory, in MB, like Harbor's memory_mb."
    )
    ttl: str = Field(
        min_length=1,
        pattern=TTL,
        description="How long Blaxel keeps the sandbox, such as 1h. It is a backstop, "
        "because the run releases the sandbox itself. It is required, as Harbor's Blaxel ttl "
        "is. Write a whole number followed by one unit, which is s, m, h, d or w.",
    )

    def specification(self) -> dict[str, JsonValue]:
        return {"sandbox": {"image": self.image, "memory": self.memory_mb, "ttl": self.ttl}}


class Blaxel(SandboxKind["SandboxInstance"]):
    """The Blaxel sandbox kind that an Episode declares. Its client is ``SandboxInstance``.

    The class names the client type without importing the SDK, because only a run that
    uses Blaxel needs the SDK.
    """

    provider = "blaxel"
    spec = BlaxelSandbox

    async def write_file(self, path: str, content: str) -> None:
        """Write text to ``path`` through the attached client.

        SDK failures propagate to the caller.
        """
        box = await self.client()
        await box.fs.write(path, content)

    async def read_file(self, path: str) -> str | None:
        """Return the text of the file at ``path``, or ``None`` when the file is missing.

        Blaxel's 404 does not say whether the file or the sandbox is missing. A liveness
        check after the read tells them apart, and raises ``SandboxGone`` when the sandbox
        is gone. Any other failed read raises the SDK's own exception.
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
    """Build the `[providers.blaxel]` adapter.

    It fails before any Blaxel call when a setting, a deadline or a credential is missing.
    """
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
