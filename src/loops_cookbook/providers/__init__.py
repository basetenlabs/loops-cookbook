"""Resource providers, and the one place adapters are built from configuration.

The run and ``loops cleanup`` both construct their adapters here, from the
training configuration's ``[providers]`` tables, so a run and its later cleanup act
through the same provider code. A built-in provider is known by name; any other
names its factory with ``adapter = "module:callable"``, and the registry uses it
with no change to core code. No table carries a credential: a factory reads the
variables it needs from the environment, loaded from ``.env``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping

from loops_cookbook.evidence import JsonValue
from loops_cookbook.resources.api import ProviderDeadlines, ResourceAdapter
from loops_cookbook.resources.registry import close_adapters
from loops_cookbook.runtime.context import RuntimeConfigurationError, import_attribute

# A provider factory receives its own table (without ``adapter``) and reads its
# credentials from the environment when called, so a missing one fails composition.
AdapterFactory = Callable[[Mapping[str, JsonValue]], ResourceAdapter]

# Built-in providers, by name, as their factory import paths.
BUILT_IN: Mapping[str, str] = {
    "blaxel": "loops_cookbook.providers.blaxel:create_adapter",
    "daytona": "loops_cookbook.providers.daytona:create_adapter",
}


async def create_adapters(
    providers: Mapping[str, Mapping[str, JsonValue]],
) -> dict[str, ResourceAdapter]:
    """Every configured provider's adapter, keyed by provider name.

    When one cannot be built, the ones already built are closed before the failure
    propagates. The caller holds the rest until a run's registry or a cleanup takes
    them, and closes them with ``close_adapters`` if that never happens.
    """
    adapters: dict[str, ResourceAdapter] = {}
    try:
        for name, table in sorted(providers.items()):
            adapters[name] = _create(name, table)
    except BaseException:
        await close_adapters(adapters)
        raise
    return adapters


def _create(name: str, table: Mapping[str, JsonValue]) -> ResourceAdapter:
    declared = table.get("adapter")
    path = declared if isinstance(declared, str) else BUILT_IN.get(name)
    if path is None:
        raise RuntimeConfigurationError(
            f"[providers.{name}] is not a built-in provider; name its factory with "
            'adapter = "module:callable"'
        )
    settings = {key: value for key, value in table.items() if key != "adapter"}
    # Named for its table, which is where the fix goes; this is not a runtime factory.
    what = f"[providers.{name}] adapter factory"
    found = import_attribute(path, what)
    if not callable(found):
        raise RuntimeConfigurationError(f"{what} {path!r} is not callable")
    factory: AdapterFactory = found  # pyright: ignore[reportAssignmentType]
    adapter = factory(settings)
    if not isinstance(getattr(adapter, "deadlines", None), ProviderDeadlines):
        raise RuntimeConfigurationError(
            f"the [providers.{name}] adapter declares no ProviderDeadlines"
        )
    # Cleanup bounds each page of a listing, so it reads list_owned page by page.
    if not inspect.isasyncgenfunction(getattr(type(adapter), "list_owned", None)):
        raise RuntimeConfigurationError(
            f"the [providers.{name}] adapter's list_owned must be an async generator "
            "that yields one page of resources at a time"
        )
    return adapter
