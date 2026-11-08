"""Resource providers, and the one place that builds adapters from configuration.

The run and ``loops cleanup`` both build their adapters here, from the training
configuration's ``[providers]`` tables. A run and its later cleanup therefore act
through the same provider code. The cookbook knows a built-in provider by name. Any
other provider names its factory with ``adapter = "module:callable"``, and the
registry uses it with no change to core code. No table carries a credential. A
factory reads the variables it needs from the environment, loaded from ``.env``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping

from loops_cookbook.evidence import JsonValue
from loops_cookbook.resources.api import ProviderDeadlines, ResourceAdapter
from loops_cookbook.resources.registry import close_adapters
from loops_cookbook.runtime.context import RuntimeConfigurationError, import_attribute

# A provider factory receives its own table, without ``adapter``. It reads its
# credentials from the environment when called, so a missing credential fails composition.
AdapterFactory = Callable[[Mapping[str, JsonValue]], ResourceAdapter]

# The import path of each built-in provider's factory, keyed by provider name.
BUILT_IN: Mapping[str, str] = {
    "blaxel": "loops_cookbook.providers.blaxel:create_adapter",
    "daytona": "loops_cookbook.providers.daytona:create_adapter",
}


async def create_adapters(
    providers: Mapping[str, Mapping[str, JsonValue]],
) -> dict[str, ResourceAdapter]:
    """Build every configured provider's adapter, keyed by provider name.

    When one adapter cannot be built, the adapters already built are closed before the
    failure propagates. On success the caller holds the adapters until a run's registry
    or a cleanup takes them. If that never happens, the caller closes them with
    ``close_adapters``.
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
    # The message names the table, because that is where the fix goes. This factory
    # builds a provider adapter, not a rollout runtime.
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
    # Cleanup puts a deadline on each page of a listing, so it reads list_owned page by page.
    if not inspect.isasyncgenfunction(getattr(type(adapter), "list_owned", None)):
        raise RuntimeConfigurationError(
            f"the [providers.{name}] adapter's list_owned must be an async generator "
            "that yields one page of resources at a time"
        )
    return adapter
