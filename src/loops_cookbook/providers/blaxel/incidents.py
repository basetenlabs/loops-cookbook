"""Work around Blaxel 0.4.19 behaviour learned from incidents.

A fixture test keeps each rule. Every rule covers a gap in the pinned SDK or API, and
the list says which change would retire it. No rule is specific to a task, a dataset
or Harbor.

- ``never_dispatched`` and ``replay`` replay a command only when Blaxel provably never
  received it. They retire when callers honour ``safe_to_retry_request``.
- ``network_deadlines`` puts back the connect, pool and write deadlines that
  ``process.exec`` clears by passing ``timeout=None``. It retires when that call takes
  a transport deadline separately from the read one.
- ``close_clients`` closes a sandbox's own HTTP pools. It retires when the SDK exposes
  ``SandboxInstance.aclose()``.
- ``own_client_reads`` fixes ``process.get``, which ``process.wait`` polls. On every
  poll, ``process.get`` hands the sandbox's own client to an API client that drops it
  inside a running loop. The API client then opens a new client, which is never closed
  and has no ``network_deadlines``. The rule retires when
  ``Client.set_async_httpx_client`` records the loop it is set in.
- ``complete_output`` restores full command output. A finished process's record keeps
  only the last 64 KB of each stream behind a truncation marker, and Harbor returns
  that record as the command's output. The rule fetches the whole stream from the
  process's logs instead, reading it as it arrives and stopping at a cap. It retires
  when the record keeps full output or Harbor reads the logs itself.
- ``creation_disposition`` says which failed creates leave nothing behind.
- ``create_refused`` says which of those creates to send again. Blaxel answers a burst
  of creates with ``429 WORKSPACE_RATE_LIMIT_EXCEEDED``, a limit that every client of
  the workspace shares, and sends no ``Retry-After``. It retires when the SDK retries
  the create.
- ``reconcile_retryable`` says which failed lookups of a just-created sandbox to repeat.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

import httpx

from loops_cookbook.resources.api import CreationDisposition

# These imports type the code against the SDK, so pyright sees any change to a member
# used here. They run for type checks only, so the module loads without the blaxel extra.
if TYPE_CHECKING:
    from blaxel.core.sandbox import SandboxInstance, SandboxProcess
    from blaxel.core.sandbox.client.client import Client
    from blaxel.core.sandbox.client.models import ProcessResponse

logger = logging.getLogger(__name__)

# A connect or pool timeout means no connection was established, so no request was
# sent. Such a request is safe to replay, even a write. Nothing else in the transport
# hierarchy is safe, because a read or write timeout, a reset mid-response or a bare 502
# with no body may each have executed. The exact types are therefore named, rather than
# their `TransportError` parent. ProxyError is a sibling of ConnectError. It is raised
# only when the CONNECT tunnel itself answers non-2xx, before any of the request is
# written.
UNDISPATCHED: tuple[type[BaseException], ...] = (
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ConnectError,
    httpx.ProxyError,
)
# Statuses of a rejected create, which means no sandbox exists. They mean a bad request,
# no permission or the workspace's rate limit. 409 is absent on purpose, because
# "already exists" means a sandbox does exist.
CREATE_LEFT_NOTHING = frozenset({400, 401, 403, 429})
# Statuses of a lookup worth repeating. A just-created sandbox may not be readable yet,
# or the API may be shedding load.
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
# How far the cookbook reads a process's logs. The body holds each stream twice, alone
# and in "logs", so the cap allows about 8 MiB of output per stream. A runaway command's
# logs are read no further.
FULL_OUTPUT_MAX_BYTES = 32 * 1024 * 1024
# Markers set on a SandboxProcess once its methods are wrapped, so repeated attaches
# wrap them only once.
_COMPLETE_OUTPUT = "_loops_complete_output"
_OWN_CLIENT_READS = "_loops_own_client_reads"


class OutputTooLarge(Exception):
    """A process's logs ran past ``FULL_OUTPUT_MAX_BYTES``, so reading stopped there."""


def status_code(error: BaseException) -> int | None:
    """Return the HTTP status that a Blaxel SDK error carries, whatever its type."""
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
    """Return whether Blaxel reports this sandbox as permanently dead.

    The answer holds whatever Blaxel says next.
    """
    body = _error_body(error)
    return body is not None and body.get("code") == "WORKLOAD_FAILED"


def never_dispatched(error: BaseException) -> bool:
    """Return true only when the request provably did not reach the workload.

    Blaxel answers this itself in ``safe_to_retry_request``, which is the field to read.
    ``dispatch_state`` is the wrong field. A sandbox that is still coming up and one
    that has died permanently both report ``not_dispatched``, because in both cases the
    request went nowhere. Only the first will ever accept it.
    """
    if isinstance(error, UNDISPATCHED):
        return True
    body = _error_body(error)
    return body is not None and body.get("safe_to_retry_request") is True


def creation_disposition(error: BaseException) -> CreationDisposition:
    """Return whether a failed create can have left a sandbox behind.

    A ``400``, ``401``, ``403`` or ``429``, and a request that never dispatched, leave
    nothing. Anything else, including a ``409`` and an unparseable ``500``, may have left
    a sandbox.
    """
    if status_code(error) in CREATE_LEFT_NOTHING or never_dispatched(error):
        return CreationDisposition.DEFINITELY_NOT_CREATED
    return CreationDisposition.AMBIGUOUS


def create_refused(error: BaseException) -> bool:
    """Return whether a failed create is worth sending again.

    Only Blaxel's rate limit and a request that never reached Blaxel qualify. By
    ``creation_disposition``, both left nothing.
    """
    return status_code(error) == 429 or never_dispatched(error)


def reconcile_retryable(error: BaseException) -> bool:
    """Return whether a lookup of a just-created sandbox should be repeated."""
    return isinstance(error, (httpx.TransportError, TimeoutError)) or (
        status_code(error) in RECONCILE_RETRY
    )


async def network_deadlines(request: httpx.Request) -> None:
    """Put deadlines on connect, pool and write for every request, and leave read to the caller.

    Blaxel builds its sandbox clients with ``httpx.Timeout(300.0, connect=10.0)``. Then
    ``process.exec`` passes ``timeout=None`` per request, so that a long command is not
    cut off mid-output. In httpx that disables every phase, so a connection that never
    establishes hangs until the caller's own deadline. ``exec`` takes no timeout
    argument, so a request hook is the only place to put the deadlines back.
    """
    timeouts = request.extensions.setdefault("timeout", {})
    for phase, limit in NETWORK_DEADLINES:
        existing = timeouts.get(phase)
        timeouts[phase] = min(existing, limit) if existing is not None else limit


def configure_clients(sandbox: SandboxInstance) -> None:
    """Install ``network_deadlines``, ``own_client_reads`` and ``complete_output`` on a sandbox."""
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
    """Make the API clients that ``process`` builds keep the sandbox client set on them.

    ``Client.get_async_httpx_client`` replaces its client whenever the running loop is
    not the one it recorded. ``set_async_httpx_client`` records no loop, so without this
    fix the sandbox's client is always replaced. Recording the loop that the API client
    is built in keeps the sandbox client, because each API client is built and used
    within one call.

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
            # With no running loop, the SDK builds a client of its own, as it would
            # without this fix.
            pass
        return api_client

    process.get_api_client = get_api_client
    setattr(process, _OWN_CLIENT_READS, True)


async def complete_output(
    process: SandboxProcess, identifier: str, record: ProcessResponse
) -> None:
    """Restore each stream that Blaxel cut, in full, from the process's logs.

    The logs are fetched only when a stream starts with Blaxel's truncation marker, and
    then once for both streams. After a failed fetch, or logs past
    ``FULL_OUTPUT_MAX_BYTES``, each cut stream stays as it was, marker included. The
    caller therefore still sees that output is missing. The caller decides how much of
    the output a model sees.
    """
    cut = [
        stream
        for stream in ("stdout", "stderr")
        if isinstance(text := getattr(record, stream, None), str) and text.startswith(TRUNCATED)
    ]
    if not cut:
        return
    # Use the SDK's own retry for a read, as its logs() does.
    from blaxel.core.sandbox.transient_retry import retry_on_transient_reset_async

    try:
        async with asyncio.timeout(FULL_OUTPUT_SECONDS):
            logs = await retry_on_transient_reset_async(lambda: read_logs(process, identifier))
    except Exception as error:  # noqa: BLE001 - the cut output is still a usable answer
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
    """Return the process's logs, as the SDK's ``logs`` reads them, but streamed.

    Reading stops, and ``OutputTooLarge`` is raised, once the body passes ``limit`` bytes.
    """
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
    """Close a sandbox's own HTTP pools, but never the SDK's shared API client.

    The close has a deadline and never raises. A pool that will not close within
    ``CLIENT_CLOSE_SECONDS`` is let go, and its connections close when it is garbage
    collected. A slow close leaks a pool, never a sandbox, so it must not fail an
    attempt that already finished its work.
    """
    clients = {
        action._client for action in (sandbox.fs, sandbox.process) if action._client is not None
    }
    try:
        async with asyncio.timeout(CLIENT_CLOSE_SECONDS):
            await asyncio.gather(*(pool.aclose() for pool in clients))
    except TimeoutError:
        logger.warning(
            "a sandbox's HTTP pools did not close in %gs; they will close when garbage collected",
            CLIENT_CLOSE_SECONDS,
        )
    for action in (sandbox.fs, sandbox.process):
        action._client = None


class ReplayState:
    """Remember whether Blaxel has called one sandbox dead.

    Once Blaxel calls the sandbox dead, ``replay`` never sends it a command again.
    """

    def __init__(self) -> None:
        self.workload_failed = False


async def replay[T](
    call: Callable[[], Awaitable[T]],
    *,
    state: ReplayState,
    logger: logging.Logger | logging.LoggerAdapter[Any],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run ``call``, and repeat it only while its failures provably never dispatched.

    The budget is wall-clock time from the first attempt, and it sits inside whatever
    deadline the caller already has. The wait between tries doubles from half a second
    up to thirty seconds.
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
            # deleted, which looks retryable. Once the sandbox has failed, stop asking.
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
                        "Blaxel readiness replay gave up after %d attempts in %.1fs (%s); "
                        "the caller sees %s",
                        attempt,
                        REPLAY_BUDGET_SECONDS - max(remaining, 0.0),
                        reason,
                        type(error).__name__,
                    )
                raise
            wait = min(wait, remaining)
            logger.warning(
                "Blaxel readiness replay: the sandbox did not receive the command (%s). "
                "Attempt %d failed; retrying in %.1fs with %.0fs of budget left",
                type(error).__name__,
                attempt,
                wait,
                remaining,
            )
            await sleep(wait)
            wait = min(wait * 2, REPLAY_MAX_WAIT_SECONDS)
