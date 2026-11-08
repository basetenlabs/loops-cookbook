"""Reward gates for the Harbor runtime tests, named by ``[rollout_runtime.config].reward_gate``."""

import asyncio
import threading
import time

SEEN: list[str] = []
CALLS: list[str] = []  # each call of the gates that record it, by gate
NOT_CALLABLE = 1


async def drop(trial, result):
    """Reads the answer from the sandbox, which must still be up, then drops the reward."""
    SEEN.append((await trial.agent_environment.exec("cat /app/answer.txt")).stdout.strip())
    return "flagged"


async def keep(trial, result):
    return None


async def broken(trial, result):
    raise RuntimeError("the gate failed")


async def truthy(trial, result):
    return True  # a reason must be a string


async def empty(trial, result):
    return ""


async def spaced(trial, result):
    return "tests modified"  # a reason names a metric: lowercase, digits and underscores


async def capitalised(trial, result):
    return "Flagged"


async def hangs(trial, result):
    await asyncio.Event().wait()


async def counted_keep(trial, result):
    CALLS.append("keep")


async def counted_drop(trial, result):
    CALLS.append("drop")
    return "flagged"


async def fails_once(trial, result):
    CALLS.append("fails_once")
    if CALLS.count("fails_once") == 1:
        raise RuntimeError("the gate failed once")


async def swallows(trial, result):
    """Swallows the deadline's cancellation and keeps the reward anyway."""
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass


async def digit(trial, result):
    return "0_flagged"


IGNORING = threading.Event()  # set: ``ignores`` stops waiting


async def ignores(trial, result):
    """Ignores every cancellation, and waits on until the test lets it go."""
    while not IGNORING.is_set():
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            continue


def not_async(trial, result):
    """A plain function: it returns no awaitable."""


async def blocks(trial, result):
    """Blocks the event loop past a 2s deadline, so no timer runs, then keeps the reward."""
    time.sleep(2.5)  # noqa: ASYNC251 - the point: no await, so no timer runs
