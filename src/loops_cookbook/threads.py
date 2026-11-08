"""Run the cookbook's blocking work in threads of its own.

The event loop's default executor is shared. The Loops SDK, the providers' clients
and the host lookup behind every new connection also use it. Work waiting there
waits on whatever else fills it, and a rollout that waits there to record a sample
stops sampling. So the cookbook never hands its own work to the default executor.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

# Use as many threads as the default executor that this pool stands in for would have.
_MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)

_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None


def executor() -> ThreadPoolExecutor:
    """Return the cookbook's thread pool, which starts on first use."""
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=_MAX_WORKERS, thread_name_prefix="loops-cookbook"
            )
        return _executor


def in_thread[T](function: Callable[..., T], /, *args, **kwargs) -> asyncio.Future[T]:
    """Run ``function`` as ``asyncio.to_thread`` does, but in the cookbook's threads.

    Return a future, as ``run_in_executor`` does, which is done only once the call has
    returned.
    """
    call = functools.partial(contextvars.copy_context().run, function, *args, **kwargs)
    return asyncio.get_running_loop().run_in_executor(executor(), call)


def finish() -> None:
    """Wait for every call already handed to the cookbook's threads.

    This waits as a runner waits for its default executor. Work handed over later
    starts new threads.
    """
    global _executor
    with _lock:
        finishing, _executor = _executor, None
    if finishing is not None:
        finishing.shutdown(wait=True)


def _forget_after_fork() -> None:
    """Drop the parent's executor in a forked child.

    A forked child has none of the parent's threads. It would reuse an executor that
    counts them as idle and so never starts a thread, and its work would wait forever.
    The child starts its own executor on first use instead.
    """
    global _executor, _lock
    _executor = None
    _lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_after_fork)
