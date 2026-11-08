"""Wait out work that a cancellation must not cut short.

Stopping a hook, and keeping evidence when a rollout is sealed, each end in a wait
whose result decides what the caller may release. Work that has stopped gives back
what it held, and work still running keeps it. A cancellation of the waiter, as on
shutdown, must not end that wait early, or the caller would release what running work
still uses. So the wait is shielded. The cancellation is kept and handed back, to be
raised once the caller has acted on what the wait showed.

``outlast()`` is that wait. ``stop()`` cancels a task first and bounds the wait by its
grace. Both return a ``Waited``. ``see_through()`` waits out work whose result the
caller needs, then raises the cancellation the wait kept. ``run_until()`` runs a task
until its expiry, and stops it then or when the waiter is cancelled.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Waited:
    """What a shielded wait saw.

    ``settled`` says the work is done, so nothing it held is in use. ``interrupted``
    says the waiter itself was cancelled, once or more, while it waited.
    """

    settled: bool
    interrupted: bool

    def propagate(self, cause: BaseException | None = None) -> None:
        """Raise the waiter's cancellation, if it had one, with ``cause`` chained to it.

        A cancellation that arrived while the work failed goes on as a cancellation,
        with the failure as its cause, so a failure never turns a shutdown into an
        ordinary error.
        """
        if self.interrupted:
            raise asyncio.CancelledError from cause


async def outlast(work: asyncio.Future[Any], timeout: float | None = None) -> Waited:
    """Wait until ``work`` is done or ``timeout`` has passed, whatever cancels the waiter.

    Cancellations of the waiter, however many, neither end the wait nor reach
    ``work``. They are returned as ``interrupted``. ``work``'s own result or failure
    stays on it for the caller to read.
    """
    loop = asyncio.get_running_loop()
    deadline = None if timeout is None else loop.time() + timeout
    interrupted = False
    while not work.done():
        remaining = None if deadline is None else deadline - loop.time()
        if remaining is not None and remaining <= 0:
            break
        try:
            await asyncio.wait({work}, timeout=remaining)
        except asyncio.CancelledError:
            interrupted = True
    return Waited(settled=work.done(), interrupted=interrupted)


async def stop(task: asyncio.Task[Any], grace: float | None) -> Waited:
    """Cancel ``task`` and give it ``grace`` seconds to finish, whatever cancels the waiter.

    A ``grace`` of None waits the task out however long it takes. Use it for work whose
    own bound lies elsewhere, so that it always settles.

    - If the result is settled, the task is done. Its result or failure is observed and
      dropped, because the stop is the outcome, not what the task did while it stopped.
    - If the result is not settled, the task ignored its cancellation and is still
      running. The caller keeps whatever the task holds and reports it. The task is not
      cancelled again.
    - If the result is interrupted, the waiter was cancelled during the grace, and it
      still waited the grace out. The caller raises the cancellation with
      ``propagate()`` once it has released or kept what the task held.
    """
    task.cancel()
    waited = await outlast(task, grace)
    if waited.settled and not task.cancelled():
        task.exception()
    return waited


async def see_through[T](work: asyncio.Future[T]) -> T:
    """Return ``work``'s result once it is done, whatever cancels the waiter meanwhile.

    A cancellation of the waiter is raised once ``work`` is done, in place of its
    result or with its failure as the cause.
    """
    waited = await outlast(work)
    try:
        result = work.result()
    except BaseException as failure:
        waited.propagate(failure)
        raise
    waited.propagate()
    return result


@dataclass(frozen=True, slots=True)
class Stopped[E](Waited):
    """The result when ``run_until()`` stopped its task before it finished.

    ``expiry`` is what the expiry returned, or None when the waiter was cancelled first.
    ``interrupted`` counts every cancellation of the waiter, including the one that
    started the stop, so ``propagate()`` raises it.
    """

    expiry: E | None


async def run_until[T, E](
    task: asyncio.Task[T],
    expiry: Coroutine[Any, Any, E],
    grace: float | None,
    *,
    own_failure: Callable[[asyncio.CancelledError], BaseException],
    before_stop: Callable[[], Awaitable[None]] | None = None,
) -> T | Stopped[E]:
    """Return ``task``'s result, or stop it when ``expiry`` returns or the waiter is cancelled.

    A stop runs ``before_stop`` first, then gives the task its ``grace``. Neither wait
    ends early for a cancellation of the waiter. A failure raised by a task that
    finished is the task's own. A CancelledError it raised is its own too, because
    nothing here cancels it before a stop, so it is raised as
    ``own_failure(cancellation)``. ``before_stop`` must not fail. An ``expiry`` that
    fails is also a bug in the caller's code. The task is stopped all the same, and
    then the failure is raised, as the cause of the waiter's cancellation if it had
    one.
    """
    expiring = asyncio.ensure_future(expiry)
    interrupted = False
    try:
        await asyncio.wait({task, expiring}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        interrupted = True
    finally:
        expiring.cancel()
    if task.done() and not interrupted:
        try:
            return task.result()
        except asyncio.CancelledError as leaked:
            raise own_failure(leaked) from leaked
    reason: E | None = None
    failure: Exception | None = None
    if not interrupted:
        try:
            reason = expiring.result()
        except Exception as error:  # noqa: BLE001 - It is raised once the task has stopped.
            failure = error
    preparing = None if before_stop is None else asyncio.ensure_future(before_stop())
    if preparing is not None:
        interrupted = (await outlast(preparing)).interrupted or interrupted
    waited = await stop(task, grace)
    if preparing is not None:
        preparing.result()  # A failure here is a bug in the caller's code, so it is raised.
    stopped = Stopped(waited.settled, waited.interrupted or interrupted, reason)
    if failure is not None:
        stopped.propagate(failure)
        raise failure
    return stopped
