"""CPU-bound work in short slices, so the event loop keeps turning while it runs.

Preparing a step's batch takes seconds of CPU. A worker thread doesn't help, because
it still holds the GIL. The event loop gives the GIL up at every I/O call and then
waits to get it back, for up to a switch interval of 5 ms, or for the whole of a long
C call. Hundreds of rollouts' HTTP traffic then falls to tens of round trips a second
while the batch is prepared. When the work runs on the loop in slices instead, it holds
the loop up for at most one slice. Between slices, the loop serves everything that is
ready.

Work is written as a generator that yields between units, such as a rollout or a row,
and returns its result. ``whole`` runs it to completion. ``in_slices`` runs it on the
loop, and gives the loop a turn whenever a slice has used up its budget. Both run the
same code, so they produce the same result.
"""

import asyncio
from collections.abc import Callable, Generator
from time import perf_counter

type Sliced[T] = Generator[None, None, T]
"""Work that yields between units and returns its result."""

SLICE_SECONDS = 0.002
"""How long a slice runs before the loop gets a turn.

The loop waits at most this long plus one unit, which is a few milliseconds for a
100k-token rollout. A turn costs only what the loop has ready to do, so a short slice
is cheap.
"""


def whole[T](work: Sliced[T]) -> T:
    """Run ``work`` to completion without pausing."""
    while True:
        try:
            next(work)
        except StopIteration as done:
            return done.value


async def in_slices[T](
    work: Sliced[T],
    *,
    budget: float = SLICE_SECONDS,
    clock: Callable[[], float] = perf_counter,
) -> T:
    """Run ``work`` on the event loop, letting other tasks run between its slices.

    A cancellation lands between two slices, and it closes ``work``.
    """
    started = clock()
    try:
        while True:
            try:
                next(work)
            except StopIteration as done:
                return done.value
            if clock() - started >= budget:
                await asyncio.sleep(0)
                started = clock()
    finally:
        # The cancellation's traceback holds this frame, and with it ``work`` and all
        # it has built, until the run has torn down. Closing and dropping ``work``
        # frees its rows now. CPython 3.12.3's close keeps the locals of a generator
        # paused at a plain yield, so only dropping it frees them there.
        work.close()
        del work
