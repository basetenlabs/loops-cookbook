"""Run renderer calls off the event loop, on a few worker threads."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

MAX_DEFAULT_WORKERS = 4


def default_workers() -> int:
    """Return ``MAX_DEFAULT_WORKERS``, or fewer when this process may use fewer CPUs.

    On Python 3.13 and later, ``os.process_cpu_count`` counts the CPUs this process
    may run on. That respects the affinity mask on Linux, and the ``-X cpu_count`` or
    ``PYTHON_CPU_COUNT`` override. On 3.12, ``os.cpu_count`` counts the machine's CPUs.
    Either may return ``None``. Neither sees a container's CPU quota, the cgroup's
    ``cpu.max``, only the cores the container can be scheduled on. So a quota under
    four cores gets more threads than cores. That is harmless here, because the GIL
    keeps the threads mostly to one core's work anyway.
    """
    count = getattr(os, "process_cpu_count", os.cpu_count)()
    return max(1, min(MAX_DEFAULT_WORKERS, count or 1))


class RenderPoolClosed(RuntimeError):
    """Raised when a call is made on a render pool after ``close()``."""


class RenderPool:
    """Runs thread-safe renderers' calls on a bounded set of threads.

    Rendering a turn is CPU work. The template runs in Python, and the tokenizer's
    encode runs in Rust, which releases the GIL. Then the IDs and offsets come back
    into Python lists. On the event loop, a long render, such as a tool result of
    megabytes, holds up every other rollout's I/O until it is done. With the pool, the
    loop keeps running while a render runs. The GIL still takes turns between the loop
    and the Python parts of a render. So the pool buys lower latency on the loop, and
    parallelism only for the encode itself.

    A renderer that says it is ``thread_safe`` is called from up to ``workers``
    threads at once. ``workers`` defaults to ``default_workers()``. Any other renderer
    is called on the loop, as without a pool. Calls on one conversation run one at a
    time whatever the renderer, because a conversation serves one generation at a time.

    Threads start on first use, so a pool that is never used costs nothing.
    """

    def __init__(self, workers: int | None = None) -> None:
        workers = default_workers() if workers is None else workers
        if workers < 1:
            raise ValueError("a render pool needs at least one worker")
        self._executor = ThreadPoolExecutor(workers, thread_name_prefix="render")
        self._closed = False

    async def run[T](self, renderer: object, call: Callable[[], T]) -> T:
        """Run ``call``, a call on ``renderer``, on a thread if the renderer is thread-safe."""
        if self._closed:
            raise RenderPoolClosed("this render pool is closed")
        if getattr(renderer, "thread_safe", False) is not True:
            return call()
        return await asyncio.get_running_loop().run_in_executor(self._executor, call)

    def close(self) -> None:
        """Let the threads exit once their running calls return, and drop queued calls.

        The pool refuses any call after this.
        """
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
