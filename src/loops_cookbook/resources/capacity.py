"""Check, before the run, that each sandbox a run declares fits in one of the provider's.

A rollout runtime knows how big each of its sandboxes is. A ``CapacityProvider`` knows
its default size and the most one sandbox may have, as its table declares. Nothing
here calls a provider, so a limit that the table does not declare is not checked.
"""

from __future__ import annotations

from collections.abc import Mapping

from loops_cookbook.resources.api import SandboxCapacity, SandboxResources

UNITS = ("vCPU", "GiB memory", "GiB disk")


def oversized(capacity: SandboxCapacity, sizes: Mapping[str, SandboxResources]) -> list[str]:
    """Return one line for each named size in ``sizes`` that exceeds ``capacity.per_sandbox``.

    Each line names the dimensions that are over the limit. A dimension that a size
    leaves unknown takes the provider's default.
    """
    found: list[str] = []
    for name, size in sizes.items():
        over = [
            f"{value:g} {unit} > {limit:g}"
            for stated, default, limit, unit in zip(
                _dimensions(size),
                _dimensions(capacity.defaults),
                _dimensions(capacity.per_sandbox),
                UNITS,
                strict=True,
            )
            if (value := default if stated is None else stated) is not None
            and limit is not None
            and value > limit
        ]
        if over:
            found.append(f"task {name}: {', '.join(over)}")
    return found


def _dimensions(size: SandboxResources) -> tuple[float | None, float | None, float | None]:
    return size.cpu, size.memory_gib, size.disk_gib
