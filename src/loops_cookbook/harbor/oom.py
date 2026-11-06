"""A sandbox's OOM-kill count, read through the environment's own ``exec``.

Where it is read depends on what the sandbox is. A micro-VM (Blaxel) has a kernel
of its own, so the kernel's ``oom_kill`` in ``/proc/vmstat`` is the sandbox's. A
container (Daytona) shares its host's kernel, so ``/proc/vmstat`` counts every OOM
kill on the host; its own count is its cgroup's ``oom_kill``, in cgroup v2's
``memory.events``. Observed in a Daytona sandbox on 4 Oct 2026: an allocation past
its 1 GiB limit was killed (exit 137), its cgroup's ``oom_kill`` went from 0 to 1, and
``/proc/vmstat``'s from 31559 to 31560. Under cgroup v1 the mounted memory root may be
a parent's, so a container without v2 reports no count rather than one that may not
be its own.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal, Protocol, runtime_checkable

# Well inside a command's replay budget, so a slow read never adds a whole budget to a trial.
OOM_READ_SECONDS = 10.0

# What to read, by the kind of sandbox: its own kernel's, or its cgroup's.
READS: dict[str, str] = {
    "kernel": "cat /proc/vmstat",
    "cgroup": "cat /sys/fs/cgroup/memory.events",
}


@runtime_checkable
class ReportsOomKills(Protocol):
    """An environment that reads its sandbox's OOM-kill count, or None when it cannot."""

    async def oom_kills(self) -> int | None: ...


async def oom_kills(environment: Any, source: Literal["kernel", "cgroup"]) -> int | None:
    """The sandbox's OOM-kill count (an ``oom_kill`` line of ``source``), or None.

    ``kernel`` only for a sandbox with a kernel of its own; ``cgroup`` for a
    container. A report only: it goes through ``exec`` like any command, is bounded
    by ``OOM_READ_SECONDS``, and never raises; a read that fails returns None.
    """
    try:
        result = await asyncio.wait_for(environment.exec(READS[source]), OOM_READ_SECONDS)
    except Exception as error:  # noqa: BLE001 - a report never fails its trial
        environment.logger.debug(
            "Sandbox OOM-kill count not read: %s: %s", type(error).__name__, error
        )
        return None
    if result.return_code != 0:
        return None
    for line in (result.stdout or "").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == "oom_kill" and fields[1].isdigit():
            return int(fields[1])
    return None
