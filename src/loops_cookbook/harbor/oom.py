"""Read a sandbox's out-of-memory (OOM) kill count through the environment's own ``exec``.

Where the count is read depends on what the sandbox is. A Blaxel sandbox is a micro-VM
with a kernel of its own, so the kernel's ``oom_kill`` in ``/proc/vmstat`` is the
sandbox's count. A Daytona sandbox is a container that shares its host's kernel, so
``/proc/vmstat`` counts every OOM kill on the host. The container's own count is its
cgroup's ``oom_kill``, in cgroup v2's ``memory.events``.

A test in a Daytona sandbox showed this. An allocation past the sandbox's 1 GiB limit
was killed with exit code 137. The cgroup's ``oom_kill`` went from 0 to 1, and
``/proc/vmstat``'s went from 31559 to 31560. Under cgroup v1 the mounted memory root
may be a parent's. So a container without cgroup v2 reports no count, rather than one
that may not be its own.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal, Protocol, runtime_checkable

# This is well inside a command's replay budget, so a slow read never adds a whole
# budget to a trial.
OOM_READ_SECONDS = 10.0

# The command that reads the count for each kind of sandbox.
READS: dict[str, str] = {
    "kernel": "cat /proc/vmstat",
    "cgroup": "cat /sys/fs/cgroup/memory.events",
}


@runtime_checkable
class ReportsOomKills(Protocol):
    """An environment that can read its sandbox's OOM-kill count.

    ``oom_kills`` returns None when the environment cannot read it.
    """

    async def oom_kills(self) -> int | None: ...


async def oom_kills(environment: Any, source: Literal["kernel", "cgroup"]) -> int | None:
    """Return the sandbox's OOM-kill count from the ``oom_kill`` line of ``source``, or None.

    Use ``kernel`` only for a sandbox with a kernel of its own, and ``cgroup`` for a
    container. The count is only a report. The read goes through ``exec`` like any
    command and is bounded by ``OOM_READ_SECONDS``. It never raises, and a read that
    fails returns None.
    """
    try:
        result = await asyncio.wait_for(environment.exec(READS[source]), OOM_READ_SECONDS)
    except Exception as error:  # noqa: BLE001 - A report never fails its trial.
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
