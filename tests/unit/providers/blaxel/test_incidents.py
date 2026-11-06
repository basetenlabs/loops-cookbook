"""Fixture tests for each Blaxel incident behaviour carried from ``develop``."""

import asyncio
import logging
from types import SimpleNamespace as NS

import httpx
import pytest

pytest.importorskip("blaxel")

from support.blaxel_api import BodyError, StatusError, not_dispatched, workload_failed

from loops_cookbook.providers.blaxel import incidents
from loops_cookbook.providers.blaxel.incidents import (
    ReplayState,
    close_clients,
    configure_clients,
    creation_disposition,
    reconcile_retryable,
    replay,
)
from loops_cookbook.resources.api import CreationDisposition

NOT_CREATED = CreationDisposition.DEFINITELY_NOT_CREATED
AMBIGUOUS = CreationDisposition.AMBIGUOUS


@pytest.fixture
async def clock(monkeypatch):
    """Loop time that advances only when ``replay`` sleeps; the sleeps it asked for."""
    now, waits = [0.0], []
    monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: now[0])

    async def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds

    return NS(sleep=sleep, waits=waits)


class Process:
    """A sandbox's process client: ``wait`` returns ``record``, and its pool answers
    ``GET /process/{id}/logs`` with ``full``, each stream whole, or ``logs_status``."""

    def __init__(self, record=None, logs=None, logs_status=None):
        self.record, self.full = record, logs or {}
        self.logs_status, self.fetched = logs_status, []
        self._client = httpx.AsyncClient(
            base_url="http://sandbox.test", transport=httpx.MockTransport(self._answer)
        )

    def get_client(self):
        return self._client

    def get_api_client(self):
        return NS()

    async def wait(self, identifier, max_wait=60000, interval=1000):
        return self.record

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.fetched.append(request.url.path)
        if self.logs_status is not None:
            return httpx.Response(self.logs_status, json={"error": "logs unavailable"})
        streams = {"stdout": "", "stderr": "", **self.full}
        return httpx.Response(200, json={"logs": streams["stdout"] + streams["stderr"], **streams})


@pytest.fixture
def sandbox():
    fs, process = httpx.AsyncClient(), Process()
    return NS(
        fs=NS(_client=fs, get_client=lambda: fs),
        process=process,
        pools=(fs, process.get_client()),
    )


def command(*outcomes):
    """A command whose attempts raise or return ``outcomes`` in order, counting them."""
    remaining, attempts = list(outcomes), []

    async def call():
        attempts.append(None)
        outcome = remaining.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return call, attempts


# ------------------------------------------------------------- exec replay


@pytest.mark.parametrize(
    ("error", "replayed"),
    [
        (not_dispatched(), True),
        (httpx.ConnectTimeout("no connection"), True),
        (httpx.PoolTimeout("no connection free"), True),
        (httpx.ConnectError("refused"), True),
        (httpx.ProxyError("CONNECT answered 502"), True),
        (httpx.ReadTimeout("response never arrived"), False),
        (httpx.WriteTimeout("request never finished"), False),
        (httpx.RemoteProtocolError("reset mid-response"), False),
        (RuntimeError("anything else"), False),
        (workload_failed(), False),  # also "not_dispatched", but Blaxel says it is dead
        # safe_to_retry_request decides, whatever dispatch_state says.
        (BodyError("WORKLOAD_UNAVAILABLE", False, 503, "not_dispatched"), False),
        (BodyError("WORKLOAD_UNAVAILABLE", True, 503, "dispatched"), True),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, BaseException) else str(value),
)
async def test_only_a_command_that_provably_never_dispatched_is_replayed(clock, error, replayed):
    call, attempts = command(error, "done")
    if replayed:
        assert (
            await replay(call, state=ReplayState(), logger=logging.getLogger(), sleep=clock.sleep)
            == "done"
        )
    else:
        with pytest.raises(type(error)):
            await replay(call, state=ReplayState(), logger=logging.getLogger(), sleep=clock.sleep)
    assert len(attempts) == (2 if replayed else 1)


async def test_replay_backs_off_from_half_a_second_to_thirty_within_sixty(clock):
    call, attempts = command(*[not_dispatched()] * 50)
    with pytest.raises(type(not_dispatched())):
        await replay(call, state=ReplayState(), logger=logging.getLogger(), sleep=clock.sleep)
    assert clock.waits[:6] == [0.5, 1, 2, 4, 8, 16]
    assert max(clock.waits) <= 30 and sum(clock.waits) <= 60
    assert len(attempts) == len(clock.waits) + 1


async def test_a_sandbox_blaxel_called_dead_is_never_asked_again(clock):
    state = ReplayState()
    dead, _ = command(workload_failed())
    with pytest.raises(type(workload_failed())):
        await replay(dead, state=state, logger=logging.getLogger(), sleep=clock.sleep)
    later, attempts = command(not_dispatched(), "done")
    with pytest.raises(type(not_dispatched())):
        await replay(later, state=state, logger=logging.getLogger(), sleep=clock.sleep)
    assert len(attempts) == 1


# ------------------------------------------------------- network deadlines


async def test_every_sandbox_request_gets_connect_pool_and_write_deadlines(sandbox):
    configure_clients(sandbox)
    configure_clients(sandbox)  # repeated attaches install nothing twice
    for pool in sandbox.pools:
        request = pool.build_request("POST", "https://sandbox/process")
        request.extensions["timeout"] = {"connect": None, "read": None, "write": None, "pool": 2.0}
        for hook in pool.event_hooks["request"]:
            await hook(request)
        assert request.extensions["timeout"] == {
            "connect": 10.0,
            "read": None,  # a long command keeps its output
            "write": 30.0,
            "pool": 2.0,  # a tighter bound stays
        }
        await pool.aclose()


# -------------------------------------------------------- full output

CUT = "[... truncated, see the process log file for the full output ...]\n"


@pytest.mark.parametrize(
    ("stdout", "stderr", "whole"),
    [
        (CUT + "tail", "fine", ["stdout"]),
        ("fine", CUT + "tail", ["stderr"]),
        (CUT + "tail", CUT + "tail", ["stdout", "stderr"]),
        ("fine", "", []),  # output under 64 KB is never fetched again
    ],
    ids=["stdout_cut", "stderr_cut", "both_cut", "nothing_cut"],
)
async def test_a_stream_blaxel_cut_comes_back_whole_from_the_process_logs(
    sandbox, stdout, stderr, whole
):
    full = {"stdout": "head\n" + "x" * 70_000, "stderr": "whole stderr"}
    sandbox.process.record = NS(stdout=stdout, stderr=stderr, exit_code=0)
    sandbox.process.full = full
    configure_clients(sandbox)
    configure_clients(sandbox)  # a repeated attach never fetches the logs twice
    record = await sandbox.process.wait("proc-1", max_wait=5000, interval=1000)
    # Once, for however many streams were cut.
    assert sandbox.process.fetched == (["/process/proc-1/logs"] if whole else [])
    expected = {"stdout": stdout, "stderr": stderr} | {stream: full[stream] for stream in whole}
    assert (record.stdout, record.stderr) == (expected["stdout"], expected["stderr"])
    for pool in sandbox.pools:
        await pool.aclose()


async def test_a_failed_fetch_keeps_the_cut_output_and_its_marker(sandbox, caplog):
    sandbox.process.record = NS(stdout=CUT + "tail", stderr="", exit_code=0)
    sandbox.process.logs_status = 500
    configure_clients(sandbox)
    with caplog.at_level(logging.WARNING):
        record = await sandbox.process.wait("proc-1")
    assert record.stdout == CUT + "tail"
    assert "fetching the rest failed (HTTPStatusError)" in caplog.text
    for pool in sandbox.pools:
        await pool.aclose()


def sdk_sandbox(handler):
    """The SDK's own sandbox client, whose process API ``handler`` answers."""
    from blaxel.core.client.models import Sandbox as SandboxModel
    from blaxel.core.sandbox import SandboxInstance

    model = {"metadata": {"name": "box", "labels": {}}, "status": "DEPLOYED", "spec": {}}
    sandbox = SandboxInstance(SandboxModel.from_dict(model), force_url="http://sandbox.test")
    transport = httpx.MockTransport(handler)
    sandbox.fs._client, sandbox.process._client = (
        httpx.AsyncClient(base_url="http://sandbox.test", transport=transport) for _ in "ab"
    )
    return sandbox


def finished(stdout: str) -> dict[str, object]:
    """A finished process's record, as Blaxel's API answers it."""
    return {
        "command": "flood",
        "completedAt": "",
        "exitCode": 0,
        "logs": "",
        "name": "p",
        "pid": "p",
        "startedAt": "",
        "status": "completed",
        "stdout": stdout,
        "stderr": "",
        "workingDir": "/",
        "keepAlive": True,
    }


async def test_a_runaway_output_is_read_to_the_cap_and_no_further(caplog):
    chunk, cap = 65_536, incidents.FULL_OUTPUT_MAX_BYTES
    pulled = [0]

    async def flood():
        """A logs body three times the cap, counted as it is read."""
        for part in (b'{"stdout": "', *[b"x" * chunk] * (3 * cap // chunk), b'"}'):
            pulled[0] += len(part)
            yield part

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/process/p/logs":
            return httpx.Response(200, content=flood())
        return httpx.Response(200, json=finished(CUT + "tail"))

    sandbox = sdk_sandbox(handler)
    configure_clients(sandbox)
    with caplog.at_level(logging.WARNING):
        record = await sandbox.process.wait("p", max_wait=5000, interval=1)
    assert pulled[0] <= cap + chunk  # the read stopped at the cap
    assert record.stdout == CUT + "tail"  # and the cut output stays, marker and all
    assert "fetching the rest failed (OutputTooLarge)" in caplog.text
    await close_clients(sandbox)


async def test_a_waited_process_is_polled_on_the_sandboxs_own_client(monkeypatch):
    """Every poll goes out on the sandbox's client, which carries ``network_deadlines``
    and is the only one answered here; no other client is opened."""
    polls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        polls.append(request.url.path)
        record = finished("done")
        return httpx.Response(
            200, json=record | {"status": "running"} if len(polls) < 3 else record
        )

    sandbox = sdk_sandbox(handler)
    configure_clients(sandbox)
    opened = []
    monkeypatch.setattr(httpx.AsyncClient, "__init__", lambda self, *a, **k: opened.append(k))
    record = await sandbox.process.wait("p", max_wait=5000, interval=1)
    assert record.status == "completed" and polls == ["/process/p"] * 3 and opened == []
    await close_clients(sandbox)


# --------------------------------------------------- per-sandbox clients


async def test_a_sandboxs_own_pools_close_once_and_the_shared_client_never(sandbox):
    shared = httpx.AsyncClient()
    await close_clients(sandbox)
    await close_clients(sandbox)
    assert all(pool.is_closed for pool in sandbox.pools) and not shared.is_closed
    await shared.aclose()


async def test_a_pool_that_will_not_close_is_let_go_not_raised(monkeypatch, caplog):
    """Closing is bounded; past the bound the pools are dropped and the caller, whose
    sandbox work is done, carries on: a slow close never fails a finished attempt."""
    monkeypatch.setattr(incidents, "CLIENT_CLOSE_SECONDS", 0.05)

    class Stuck:
        async def aclose(self):
            await asyncio.Event().wait()

    sandbox = NS(fs=NS(_client=Stuck()), process=NS(_client=None))
    await close_clients(sandbox)
    assert sandbox.fs._client is None and "did not close" in caplog.text


# ------------------------------------------------ creation and reconcile


@pytest.mark.parametrize(
    ("error", "disposition"),
    [
        *[(StatusError(status), NOT_CREATED) for status in (400, 401, 403, 429)],
        (httpx.ConnectError("refused"), NOT_CREATED),
        (not_dispatched(), NOT_CREATED),
        (StatusError(409), AMBIGUOUS),  # "already exists" means one does
        (StatusError(500), AMBIGUOUS),
        (StatusError(None, "unparseable 500 body"), AMBIGUOUS),
        (httpx.ReadTimeout("lost after sending"), AMBIGUOUS),
        (TimeoutError(), AMBIGUOUS),
    ],
    ids=lambda value: str(getattr(value, "status_code", None) or type(value).__name__),
)
def test_the_creation_disposition_table(error, disposition):
    assert creation_disposition(error) is disposition


@pytest.mark.parametrize(
    ("error", "retried"),
    [
        *[(StatusError(status), True) for status in (404, 429, 502, 503, 504)],
        (httpx.ConnectError("refused"), True),
        (TimeoutError(), True),
        *[(StatusError(status), False) for status in (400, 401, 403, 500)],
        (RuntimeError("no status"), False),
    ],
    ids=lambda value: str(getattr(value, "status_code", None) or type(value).__name__),
)
def test_reconciliation_retries_only_a_just_created_sandboxs_transient_lookups(error, retried):
    assert reconcile_retryable(error) is retried
