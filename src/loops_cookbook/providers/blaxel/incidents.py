"""Blaxel 0.4.19 behaviour learned from incidents, each kept by a fixture test.

Every rule here is a gap in the pinned SDK or API, with the change that would retire
it; none is specific to a task, a dataset, or Harbor:

- ``never_dispatched`` and ``replay``: replay a command only when Blaxel provably
  never received it. Retires when callers honour ``safe_to_retry_request``.
- ``network_deadlines``: put back the connect, pool, and write bounds that
  ``process.exec`` clears by passing ``timeout=None``. Retires when that call takes
  a transport deadline separately from the read one.
- ``close_clients``: close a sandbox's own HTTP pools. Retires when the SDK exposes
  ``SandboxInstance.aclose()``.
- ``own_client_reads``: ``process.get``, which ``process.wait`` polls, hands the
  sandbox's own client to an API client that drops it inside a running loop and
  opens a new one, never closed and without ``network_deadlines``, on every poll.
  Retires when ``Client.set_async_httpx_client`` records the loop it is set in.
- ``complete_output``: a finished process's record keeps only the last 64 KB of
  each stream behind a truncation marker, and Harbor returns that record as the
  command's output; fetch the whole stream from the process's logs instead, read
  as it arrives and no further than a cap. Retires when the record keeps full
  output or Harbor reads the logs itself.
- ``creation_disposition``: which failed creates leave nothing behind.
- ``create_refused``: which of those to send again. Blaxel answers a burst of creates
  with ``429 WORKSPACE_RATE_LIMIT_EXCEEDED``, a limit every client of the workspace
  shares, and sends no ``Retry-After``. Retires when the SDK retries it.
- ``reconcile_retryable``: which failed lookups of a just-created sandbox to repeat.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

import httpx

from loops_cookbook.resources.api import CreationDisposition

# Typed against the SDK, so pyright sees every member used here change. Imported for
# type checks only: the module loads without the blaxel extra.
if TYPE_CHECKING:
    from blaxel.core.sandbox import SandboxInstance, SandboxProcess
    from blaxel.core.sandbox.client.client import Client
    from blaxel.core.sandbox.client.models import ProcessResponse

logger = logging.getLogger(__name__)

# A connect or pool timeout means no connection was established, so no request was
# sent: safe to replay, writes included. Nothing else in the transport hierarchy is
# -- a read or write timeout, a reset mid-response, a bare 502 with no body, may all
# have executed -- so the exact types are named rather than their `TransportError`
# parent. ProxyError is a sibling of ConnectError, raised only when the CONNECT
# tunnel itself answers non-2xx, before any of the request is written.
UNDISPATCHED: tuple[type[BaseException], ...] = (
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ConnectError,
    httpx.ProxyError,
)
# Create was rejected -- bad request, no permission, or the workspace's rate limit --
# so no sandbox exists. 409 is deliberately absent: "already exists" means one does.
CREATE_LEFT_NOTHING = frozenset({400, 401, 403, 429})
# A just-created sandbox may not be readable yet, or the API may be shedding load.
RECONCILE_RETRY = frozenset({404, 429, 502, 503, 504})
# Statuses a sandbox never leaves for DEPLOYED.
DEAD = frozenset({"FAILED", "TERMINATED", "DELETING"})

REPLAY_BUDGET_SECONDS = 60.0
REPLAY_INITIAL_WAIT_SECONDS = 0.5
REPLAY_MAX_WAIT_SECONDS = 30.0
NETWORK_DEADLINES = (("connect", 10.0), ("pool", 10.0), ("write", 30.0))
CLIENT_CLOSE_SECONDS = 5.0
# How Blaxel's process record starts a stream it has cut to its last 64 KB.
TRUNCATED = "[... truncated, see the process log file"
FULL_OUTPUT_SECONDS = 30.0
# How far a process's logs are read: the body holds each stream twice, alone and in
# "logs", so about 8 MiB of output each. A runaway command's are read no further.
FULL_OUTPUT_MAX_BYTES = 32 * 1024 * 1024
# Set on a SandboxProcess once its methods are wrapped, so an attach wraps them once.
_COMPLETE_OUTPUT = "_loops_complete_output"
_OWN_CLIENT_READS = "_loops_own_client_reads"


class OutputTooLarge(Exception):
    """A process's logs run past ``FULL_OUTPUT_MAX_BYTES``; reading stopped there."""


def status_code(error: BaseException) -> int | None:
    """The HTTP status a Blaxel SDK error carries, whichever error type it is."""
    code = getattr(error, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(error, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def _error_body(error: BaseException) -> dict[str, Any] | None:
    data = getattr(error, "data", None)
    body = data.get("error") if isinstance(data, dict) else None
    return body if isinstance(body, dict) else None  # pyright: ignore[reportUnknownVariableType]


def workload_failed(error: BaseException) -> bool:
    """Blaxel reporting this sandbox as permanently dead, whatever it says next."""
    body = _error_body(error)
    return body is not None and body.get("code") == "WORKLOAD_FAILED"


def never_dispatched(error: BaseException) -> bool:
    """True only when the request provably did not reach the workload.

    Blaxel answers this itself in ``safe_to_retry_request``, which is the field to
    read -- not ``dispatch_state``. A sandbox that is still coming up and one that
    has died permanently both report ``not_dispatched``, because in both cases the
    request went nowhere, but only the first will ever accept it.
    """
    if isinstance(error, UNDISPATCHED):
        return True
    body = _error_body(error)
    return body is not None and body.get("safe_to_retry_request") is True


def creation_disposition(error: BaseException) -> CreationDisposition:
    """Whether a failed create can have left a sandbox behind.

    ``400``, ``401``, ``403``, ``429``, and a request that never dispatched leave
    nothing; everything else, including ``409`` and an unparseable ``500``, may have.
    """
    if status_code(error) in CREATE_LEFT_NOTHING or never_dispatched(error):
        return CreationDisposition.DEFINITELY_NOT_CREATED
    return CreationDisposition.AMBIGUOUS


def create_refused(error: BaseException) -> bool:
    """Whether a failed create is worth sending again: Blaxel's rate limit, or a
    request that never reached it. Both left nothing (``creation_disposition``)."""
    return status_code(error) == 429 or never_dispatched(error)


def reconcile_retryable(error: BaseException) -> bool:
    """Whether a lookup of a just-created sandbox should be repeated."""
    return isinstance(error, (httpx.TransportError, TimeoutError)) or (
        status_code(error) in RECONCILE_RETRY
    )


async def network_deadlines(request: httpx.Request) -> None:
    """Bound connect, pool, and write on every request; leave read to the caller.

    Blaxel builds its sandbox clients with ``httpx.Timeout(300.0, connect=10.0)``,
    then ``process.exec`` passes ``timeout=None`` per request so a long command is
    not cut off mid-output. In httpx that disables every phase, so a connection that
    never establishes hangs until the caller's own deadline. ``exec`` takes no
    timeout argument, so a request hook is the only seam.
    """
    timeouts = request.extensions.setdefault("timeout", {})
    for phase, limit in NETWORK_DEADLINES:
        existing = timeouts.get(phase)
        timeouts[phase] = min(existing, limit) if existing is not None else limit


def configure_clients(sandbox: SandboxInstance) -> None:
    """Install ``network_deadlines``, ``own_client_reads`` and ``complete_output`` on a
    sandbox's clients."""
    for action in (sandbox.fs, sandbox.process):
        hooks = action.get_client().event_hooks["request"]
        if network_deadlines not in hooks:
            hooks.append(network_deadlines)
    process = sandbox.process
    own_client_reads(process)
    if not getattr(process, _COMPLETE_OUTPUT, False):
        waited = process.wait

        async def wait(
            identifier: str, max_wait: int = 60000, interval: int = 1000
        ) -> ProcessResponse:
            record = await waited(identifier, max_wait, interval)
            await complete_output(process, identifier, record)
            return record

        process.wait = wait
        setattr(process, _COMPLETE_OUTPUT, True)


def own_client_reads(process: SandboxProcess) -> None:
    """Make the API clients ``process`` builds keep the sandbox client set on them.

    ``Client.get_async_httpx_client`` replaces its client whenever the running loop is
    not the one it recorded, and ``set_async_httpx_client`` records none, so the
    sandbox's client was always replaced. Recording the loop the API client is built
    in keeps it: each is built and used within one call.

    This also skips the SDK's check that the sandbox's client belongs to the running
    loop. That holds because a run attaches and uses every sandbox in its one loop.
    """
    if getattr(process, _OWN_CLIENT_READS, False):
        return
    build = process.get_api_client

    def get_api_client() -> Client:
        api_client = build()
        try:
            api_client._async_client_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass  # no loop: the SDK builds a client of its own, as without this
        return api_client

    process.get_api_client = get_api_client
    setattr(process, _OWN_CLIENT_READS, True)


async def complete_output(
    process: SandboxProcess, identifier: str, record: ProcessResponse
) -> None:
    """Put each stream Blaxel cut back in full, from the process's logs.

    Only when a stream starts with Blaxel's truncation marker are the logs fetched,
    once for both streams. A failed fetch, or logs past ``FULL_OUTPUT_MAX_BYTES``,
    keeps each cut stream, marker included, so the caller still sees that output is
    missing. Deciding how much of the output a model sees stays with the caller.
    """
    cut = [
        stream
        for stream in ("stdout", "stderr")
        if isinstance(text := getattr(record, stream, None), str) and text.startswith(TRUNCATED)
    ]
    if not cut:
        return
    # The SDK's own retry for a read, as its logs() has.
    from blaxel.core.sandbox.transient_retry import retry_on_transient_reset_async

    try:
        async with asyncio.timeout(FULL_OUTPUT_SECONDS):
            logs = await retry_on_transient_reset_async(lambda: read_logs(process, identifier))
    except Exception as error:  # noqa: BLE001 - the cut output is still an answer
        logger.warning(
            "Blaxel kept only the end of process %s's %s, and fetching the rest "
            "failed (%s); the command's output stays cut",
            identifier,
            " and ".join(cut),
            type(error).__name__,
        )
        return
    for stream in cut:
        full = logs.get(stream)
        if isinstance(full, str):
            setattr(record, stream, full)


async def read_logs(
    process: SandboxProcess, identifier: str, limit: int = FULL_OUTPUT_MAX_BYTES
) -> dict[str, Any]:
    """The process's logs, as the SDK's ``logs`` reads them, but streamed: reading
    stops, and ``OutputTooLarge`` is raised, once the body passes ``limit`` bytes."""
    body = bytearray()
    client = process.get_client()
    async with client.stream("GET", f"/process/{identifier}/logs") as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > limit:
                raise OutputTooLarge(f"process {identifier}'s logs pass {limit} bytes")
    logs = json.loads(body)
    if not isinstance(logs, dict):
        raise TypeError(f"process {identifier}'s logs are not an object")
    return cast(dict[str, Any], logs)


async def close_clients(sandbox: SandboxInstance) -> None:
    """Close a sandbox's own HTTP pools, never the SDK's shared API client.

    Bounded, and never raised: a pool that will not close in ``CLIENT_CLOSE_SECONDS``
    is let go (its connections close when it is collected). A slow close leaks a pool,
    never a sandbox, so it must not fail the attempt that already finished its work.
    """
    clients = {
        action._client for action in (sandbox.fs, sandbox.process) if action._client is not None
    }
    try:
        async with asyncio.timeout(CLIENT_CLOSE_SECONDS):
            await asyncio.gather(*(pool.aclose() for pool in clients))
    except TimeoutError:
        logger.warning(
            "a sandbox's HTTP pools did not close in %gs; left to close on collection",
            CLIENT_CLOSE_SECONDS,
        )
    for action in (sandbox.fs, sandbox.process):
        action._client = None


class ReplayState:
    """One sandbox's replay memory: once Blaxel calls it dead, it is never asked again."""

    def __init__(self) -> None:
        self.workload_failed = False


async def replay[T](
    call: Callable[[], Awaitable[T]],
    *,
    state: ReplayState,
    logger: logging.Logger | logging.LoggerAdapter[Any],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run ``call``, repeating it only while its failures provably never dispatched.

    The budget is wall clock from the first attempt and sits inside whatever deadline
    the caller already has. Backoff doubles from half a second to thirty.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + REPLAY_BUDGET_SECONDS
    wait = REPLAY_INITIAL_WAIT_SECONDS
    attempt = 0
    while True:
        attempt += 1
        try:
            return await call()
        except Exception as error:
            # Blaxel reports a dead sandbox as UNAVAILABLE again once it has been
            # reaped, which reads as retryable; having seen it fail, stop asking.
            state.workload_failed = state.workload_failed or workload_failed(error)
            remaining = deadline - loop.time()
            if remaining <= 0 or state.workload_failed or not never_dispatched(error):
                if attempt > 1:
                    reason = (
                        "workload failed"
                        if state.workload_failed
                        else "budget exhausted"
                        if remaining <= 0
                        else "not safe to replay"
                    )
                    logger.warning(
                        "Blaxel readiness replay: giving up after %d attempts in %.1fs (%s); "
                        "the caller sees %s",
                        attempt,
                        REPLAY_BUDGET_SECONDS - max(remaining, 0.0),
                        reason,
                        type(error).__name__,
                    )
                raise
            wait = min(wait, remaining)
            logger.warning(
                "Blaxel readiness replay: sandbox did not receive the command (%s); "
                "attempt %d, retrying in %.1fs, %.0fs of budget left",
                type(error).__name__,
                attempt,
                wait,
                remaining,
            )
            await sleep(wait)
            wait = min(wait * 2, REPLAY_MAX_WAIT_SECONDS)
