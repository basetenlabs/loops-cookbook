"""The facade over Daytona's SDK and the error classification, with no Daytona call.

The SDK's real error types and image builder are used; the control plane is not.
"""

import asyncio
import contextlib
import socket
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("daytona")

import aiohttp
from aiohttp import web
from daytona import CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams, Image
from daytona.common.errors import (
    DaytonaBadRequestError,
    DaytonaConflictError,
    DaytonaConnectionError,
    DaytonaConnectionTimeoutError,
    DaytonaForbiddenError,
    DaytonaInternalServerError,
    DaytonaNotFoundError,
    DaytonaRateLimitError,
    DaytonaTimeoutError,
)
from daytona_api_client_async.models.sandbox_state import SandboxState

from loops_cookbook.providers.daytona import api as facade
from loops_cookbook.providers.daytona.errors import (
    create_refused,
    creation_disposition,
    reconcile_retryable,
)
from loops_cookbook.resources.api import CreationDisposition

NOT_CREATED = CreationDisposition.DEFINITELY_NOT_CREATED
AMBIGUOUS = CreationDisposition.AMBIGUOUS


def transport_failure(
    cause: BaseException, wrapper: type[DaytonaConnectionError] = DaytonaConnectionError
) -> BaseException:
    """The SDK's transport error, raised from the aiohttp error beneath it."""
    error = wrapper(f"Failed to create sandbox: {cause}")
    error.__cause__ = cause
    return error


KEY = NS(host="app.daytona.io", port=443, ssl=True)
OS_ERROR = OSError("no route to host")


@pytest.mark.parametrize(
    ("error", "disposition", "refused"),
    [
        (
            DaytonaBadRequestError("CPU request 5 exceeds maximum allowed per sandbox (4)", 400),
            NOT_CREATED,
            False,
        ),
        (DaytonaBadRequestError("Disk quota exceeded: limit exceeded", 400), NOT_CREATED, False),
        (DaytonaForbiddenError("forbidden", 403), NOT_CREATED, False),
        (DaytonaRateLimitError("Too many requests", 429), NOT_CREATED, True),
        (
            DaytonaBadRequestError("auto_stop_interval must be a non-negative integer"),
            NOT_CREATED,
            False,
        ),
        (DaytonaConflictError("Sandbox with name x already exists", 409), AMBIGUOUS, False),
        (DaytonaInternalServerError("boom", 500), AMBIGUOUS, False),
        (
            transport_failure(aiohttp.SocketTimeoutError(), DaytonaConnectionTimeoutError),
            AMBIGUOUS,
            False,
        ),
        (transport_failure(aiohttp.ServerDisconnectedError()), AMBIGUOUS, False),
        (
            DaytonaTimeoutError("Sandbox failed to become ready within the timeout"),
            AMBIGUOUS,
            False,
        ),
        (transport_failure(aiohttp.ClientConnectorError(KEY, OS_ERROR)), NOT_CREATED, True),  # pyright: ignore[reportArgumentType]
        (transport_failure(aiohttp.ClientConnectorDNSError(KEY, OS_ERROR)), NOT_CREATED, True),  # pyright: ignore[reportArgumentType]
        (
            transport_failure(aiohttp.ClientProxyConnectionError(KEY, OS_ERROR)),  # pyright: ignore[reportArgumentType]
            NOT_CREATED,
            True,
        ),
        (
            transport_failure(aiohttp.ConnectionTimeoutError(), DaytonaConnectionTimeoutError),
            NOT_CREATED,
            True,
        ),
    ],
    ids=[
        "per_sandbox_limit",
        "bad_request_whatever_it_says",
        "forbidden",
        "rate_limited",
        "validated_before_sending",
        "name_taken",
        "server_error",
        "read_timeout",
        "disconnected",
        "start_timeout",
        "never_connected",
        "dns_failure",
        "proxy_failure",
        "connect_timeout",
    ],
)
def test_a_failed_create_is_classified_by_what_it_may_have_left(error, disposition, refused):
    assert creation_disposition(error) is disposition
    assert create_refused(error) is refused


@pytest.mark.parametrize(
    ("error", "retry"),
    [
        (DaytonaNotFoundError("not found", 404), True),
        (DaytonaRateLimitError("slow down", 429), True),
        (DaytonaInternalServerError("boom", 503), True),
        (DaytonaConnectionError("reset"), True),
        (DaytonaConnectionTimeoutError("read timed out"), True),
        (DaytonaForbiddenError("forbidden", 403), False),
        (DaytonaBadRequestError("bad", 400), False),
    ],
    ids=["404", "429", "503", "transport", "transport_timeout", "403", "400"],
)
def test_a_lookup_is_repeated_only_while_it_may_yet_succeed(error, retry):
    assert reconcile_retryable(error) is retry


def test_a_declarative_image_round_trips_through_json(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.txt").write_text("x")
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\nCOPY data /data\nRUN true\n")
    image = Image.from_dockerfile(tmp_path / "Dockerfile")
    described = facade.describe_image(image)
    assert described["dockerfile"] == image.dockerfile()
    rebuilt = facade.image_from(described)
    assert isinstance(rebuilt, Image)
    assert rebuilt.dockerfile() == image.dockerfile()
    assert facade.describe_image(rebuilt) == described  # the context comes back too


def test_create_parameters_follow_the_sandbox_configuration():
    built = facade.create_params(
        {
            "name": "loops-1",
            "image": {"dockerfile": "FROM debian:12\n", "context": []},
            "resources": {"cpu": 2, "memory": 4, "disk": 3},
            "labels": {"loops.run_id": "r"},
            "auto_stop_interval": 60,
            "auto_delete_interval": 0,
            "ttl_minutes": 1440,
        }
    )
    assert isinstance(built, CreateSandboxFromImageParams)
    assert built.resources is not None and (built.resources.cpu, built.resources.disk) == (2, 3)
    assert (built.name, built.ttl_minutes, built.auto_stop_interval) == ("loops-1", 1440, 60)
    base = facade.create_params({"name": "b", "image": {"base": "python:3.12"}})
    assert isinstance(base, CreateSandboxFromImageParams) and base.image == "python:3.12"
    snapshot = facade.create_params({"name": "s", "snapshot": "harbor__abc__snapshot"})
    assert isinstance(snapshot, CreateSandboxFromSnapshotParams)


def test_an_sdk_sandbox_is_observed_with_its_state_as_daytonas_wire_value():
    sandbox = NS(
        id="id-1",
        name="loops-1",
        labels={"a": "b"},
        state=SandboxState.STARTED,
    )
    observed = facade.observed(sandbox)
    assert (observed.id, observed.name, observed.state) == ("id-1", "loops-1", "started")


@pytest.mark.parametrize(
    "config",
    [
        {"name": "x", "image": {"base": "python:3.12"}, "ttl_minutes": "soon"},
        {"name": "x", "snapshot": "s", "resources": {"cpu": 2}},
        {"name": "x", "image": {"context": []}},
    ],
    ids=["bad_ttl", "sized_snapshot", "no_dockerfile"],
)
async def test_a_configuration_the_sdk_cannot_take_fails_as_not_created_before_sending(
    monkeypatch, config
):
    api = facade.DaytonaApi(api_key="k", api_url=None, target=None)
    monkeypatch.setattr(api, "client", lambda: pytest.fail("Daytona was called"))
    with pytest.raises(Exception) as failure:
        await api.create(config, timeout=1)
    assert creation_disposition(failure.value) is NOT_CREATED
    assert not create_refused(failure.value)


async def test_the_listing_is_read_one_fetched_page_at_a_time(monkeypatch):
    monkeypatch.setattr(facade, "LIST_PAGE", 2)
    queries = []

    class Client:
        async def list(self, query):
            queries.append(query)
            for n in range(5):
                yield NS(id=f"id-{n}", name=f"s{n}", labels={}, state=SandboxState.STARTED)

    api = facade.DaytonaApi(api_key="k", api_url=None, target=None)
    monkeypatch.setattr(api, "client", lambda: Client())
    pages = [[o.name for o in page] async for page in api.list({"loops.run_id": "r"})]
    assert pages == [["s0", "s1"], ["s2", "s3"], ["s4"]]
    assert queries[0].labels == {"loops.run_id": "r"} and queries[0].limit == 2


async def test_more_calls_reach_daytona_at_once_than_the_sdks_default_pool_allows():
    # The SDK's default pool (250) queues the rest client-side; past it, a run's deletes
    # and output copies waited behind its live sandboxes' calls.
    calls = 300
    in_flight = 0
    peak = 0
    all_in = asyncio.Event()

    async def held(request: web.Request) -> web.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        if in_flight == calls:
            all_in.set()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(all_in.wait(), timeout=5)
        in_flight -= 1
        return web.json_response({"message": "not found"}, status=404)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", held)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    await web.SockSite(runner, sock).start()
    api = facade.DaytonaApi(api_key="k", api_url=f"http://127.0.0.1:{port}/api", target=None)
    try:
        results = await asyncio.gather(
            *(api.get(f"sandbox-{n}") for n in range(calls)), return_exceptions=True
        )
    finally:
        await api.close()
        await runner.cleanup()
    assert all(isinstance(r, DaytonaNotFoundError) for r in results), results[:3]
    assert peak == calls
