"""The Chat Completions gateway delivers: attribution to the attempt, idempotency keys,
clients that go away, revocation and quiescing.

Every test runs the real gateway on loopback, a real attempt lease, and a scripted
sampler; nothing reaches the network.
"""

import asyncio
import json

import aiohttp
import openai
import pytest
from support.cancellation import Landing, Response, Trigger, Work, cancellation_matrix, settle_loop
from support.fakes import ScriptedSampler
from support.leases import lease

from loops_cookbook.llm.client import Renderers
from loops_cookbook.model_access.api import (
    ModelAccessError,
    UndeclaredProducer,
)
from loops_cookbook.model_access.lease import DirectModelAccess
from loops_cookbook.model_access.openai.access import AccessEnded
from loops_cookbook.model_access.producers import ProducerSettings

from .rig import (
    MODEL,
    OPENED_CLIENTS,
    TOOLS,
    Renderer,
    _hang_up_mid_stream,
    _huge_reply,
    _same_tool_call,
    attempt,
    connect,
    hello,
    identity,
    raw,
)


def _conversation_turns(first_reply):
    return [
        [{"role": "user", "content": "start"}],
        [
            {"role": "user", "content": "start"},
            first_reply,
            {"role": "user", "content": "more"},
        ],
    ]


async def _disconnect_during_generation(connection, sampler, headers=None):
    """A client that gives up while its generation runs, as a timeout does."""
    session = aiohttp.ClientSession()
    request = asyncio.create_task(
        session.post(
            connection.base_url + "/chat/completions",
            json=hello(),
            headers={
                "Authorization": f"Bearer {connection.api_key.get_secret_value()}",
                **(headers or {}),
            },
        )
    )
    await sampler.entered.wait()
    request.cancel()
    await session.close()
    await asyncio.sleep(0.05)
    sampler.release.set()
    await asyncio.sleep(0.05)


async def _models_status(connection):
    try:
        return (await raw(connection, "/models", method="GET"))[0]
    except aiohttp.ClientConnectionError:
        return None


async def test_a_credentials_producer_cannot_be_changed_by_the_request(gateway):
    access, held, _ = attempt(gateway, producers=ProducerSettings({"main": True, "critic": False}))
    connection = await connect(access, "critic")
    status, _ = await raw(connection, body=hello(), headers={"X-Loops-Producer": "main"})
    assert status == 200
    status, body = await raw(connection, body=hello(user="main"))
    assert status == 400 and body["error"]["param"] == "user"
    (sample,) = (await held.seal()).samples
    assert (sample.producer, sample.trainable) == ("critic", False)


async def test_http_access_refuses_an_undeclared_producer_before_any_credential(gateway):
    access, held, sampler = attempt(
        gateway, producers=ProducerSettings({"main": True, "critic": False})
    )
    with pytest.raises(ModelAccessError, match=r"'critc' is not declared.*\['critic', 'main'\]"):
        await access.http_connection("critc")
    assert isinstance(held.fatal, UndeclaredProducer) and sampler.prompts == []


async def test_identical_simultaneous_requests_across_attempts_never_cross_attribute(gateway):
    release = asyncio.Event()
    attempts = []
    for name in ("a1", "a2", "a3"):
        sampler = ScriptedSampler(stop_reason="stop")
        sampler.release = release
        renderer = Renderer()
        renderer.reply = _same_tool_call
        attempts.append(attempt(gateway, sampler=sampler, renderer=renderer, name=name))
    history = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_same",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_same", "content": "same result"},
    ]
    connections = [await access.http_connection() for access, _, _ in attempts]
    requests = [
        asyncio.create_task(raw(connection, body=hello(messages=history, tools=TOOLS)))
        for connection in connections
    ]
    while not all(sampler.prompts for _, _, sampler in attempts):
        await asyncio.sleep(0.01)
    release.set()
    bodies = [body for _, body in await asyncio.gather(*requests)]
    # Every response carries the same tool-call ID, and each still belongs to the
    # attempt whose credential asked for it.
    calls = {body["choices"][0]["message"]["tool_calls"][0]["id"] for body in bodies}
    assert calls == {"call_same"}
    for (_, held, _), body in zip(attempts, bodies, strict=True):
        (sample,) = (await held.seal()).samples
        assert sample.sample_id == body["id"].removeprefix("chatcmpl-")


async def test_without_the_header_each_request_is_its_own_conversation(gateway):
    access, held, _ = attempt(gateway)
    connection = await connect(access)
    await raw(connection, body=hello())
    await raw(connection, body=hello())
    first, second = (await held.seal()).samples
    assert first.conversation_id != second.conversation_id


async def test_http_and_direct_produce_equivalent_evidence_for_one_conversation(gateway):
    renderer, direct_renderer = Renderer(), Renderer()
    access, held, _ = attempt(gateway, renderer=renderer)
    connection = await connect(access)
    headers = {"X-Loops-Conversation-Id": "session"}
    _status, first = await raw(
        connection, body=hello(messages=_conversation_turns(None)[0]), headers=headers
    )
    echo = dict(first["choices"][0]["message"])
    echo["content"] = echo.get("content") or None
    echo.pop("reasoning_content", None)  # an omitted reasoning_content is the returned one
    await raw(connection, body=hello(messages=_conversation_turns(echo)[1]), headers=headers)
    over_http = (await held.seal()).samples

    direct_lease = lease(ScriptedSampler(stop_reason="stop"), identity=identity("direct"))
    chat = (
        DirectModelAccess(direct_lease, Renderers(direct_renderer))
        .llm()
        .conversation(conversation_id="session")
    )
    reply = await chat.generate(messages=_conversation_turns(None)[0])
    await chat.generate(messages=_conversation_turns(reply.native_message)[1])
    direct = (await direct_lease.seal()).samples
    assert [(s.prompt_ids, s.completion_ids, s.conversation_id) for s in over_http] == [
        (s.prompt_ids, s.completion_ids, s.conversation_id) for s in direct
    ]
    assert renderer.bridges and len(renderer.bridges) == len(direct_renderer.bridges) == 1


async def test_the_bridge_restarts_when_the_tools_change(gateway):
    renderer = Renderer()
    access, _, _ = attempt(gateway, renderer=renderer)
    connection = await connect(access)
    headers = {"X-Loops-Conversation-Id": "session"}
    _, first = await raw(connection, body=hello(), headers=headers)
    echo = first["choices"][0]["message"]
    tools = [{"type": "function", "function": {"name": "added"}}]
    await raw(
        connection,
        body=hello(
            messages=[{"role": "user", "content": "hi"}, echo, {"role": "user", "content": "x"}],
            tools=tools,
        ),
        headers=headers,
    )
    assert renderer.bridges == [] and len(renderer.renders) == 2


async def test_a_second_request_on_a_generating_conversation_is_busy(gateway):
    sampler = ScriptedSampler(stop_reason="stop")
    sampler.release = asyncio.Event()
    access, _, _ = attempt(gateway, sampler=sampler)
    connection = await connect(access)
    headers = {"X-Loops-Conversation-Id": "session"}
    first = asyncio.create_task(raw(connection, body=hello(), headers=headers))
    await sampler.entered.wait()
    status, body = await raw(connection, body=hello(), headers=headers)
    assert status == 409 and body["error"]["code"] == "conversation_busy"
    sampler.release.set()
    assert (await first)[0] == 200


async def test_a_stale_attempts_requests_return_rollout_stale(gateway):
    access, held, _ = attempt(gateway, policy_version=lambda: 5, max_steps_off_policy=0)
    connection = await connect(access)
    for _ in range(2):  # the one that went stale at capture, and every one after
        status, body = await raw(connection, body=hello())
        assert status == 409 and body["error"]["code"] == "rollout_stale"
    assert held.stale is not None


async def test_a_keyed_retry_replays_the_same_response_without_sampling_again(gateway):
    access, held, sampler = attempt(gateway)
    connection = await connect(access)
    headers = {"Idempotency-Key": "k1"}
    first = await raw(connection, body=hello(), headers=headers)
    second = await raw(connection, body=hello(), headers=headers)
    assert first == second and len(sampler.prompts) == 1
    assert len((await held.seal()).samples) == 1


async def test_concurrent_calls_with_one_key_share_one_generation(gateway):
    sampler = ScriptedSampler(stop_reason="stop")
    sampler.release = asyncio.Event()
    access, held, _ = attempt(gateway, sampler=sampler)
    connection = await connect(access)
    headers = {"Idempotency-Key": "k1"}
    calls = [asyncio.create_task(raw(connection, body=hello(), headers=headers)) for _ in range(3)]
    await sampler.entered.wait()
    sampler.release.set()
    results = await asyncio.gather(*calls)
    assert len({json.dumps(body) for _, body in results}) == 1 and len(sampler.prompts) == 1
    assert len((await held.seal()).samples) == 1


async def test_reusing_a_key_with_a_different_body_is_a_conflict_without_sampling(gateway):
    access, _, sampler = attempt(gateway)
    connection = await connect(access)
    headers = {"Idempotency-Key": "k1"}
    await raw(connection, body=hello(), headers=headers)
    status, body = await raw(connection, body=hello(temperature=0.1), headers=headers)
    assert status == 409 and body["error"]["code"] == "idempotency_key_reused"
    assert len(sampler.prompts) == 1


async def test_a_response_whose_client_went_away_is_dropped_not_a_failure(gateway):
    sampler = ScriptedSampler(stop_reason="stop")
    sampler.release = asyncio.Event()
    access, held, _ = attempt(gateway, sampler=sampler)
    connection = await connect(access)
    await _disconnect_during_generation(connection, sampler)
    assert held.failure is None  # the agent never read it: a branch it never took
    assert (await held.seal()).samples == ()


async def test_a_keyed_retry_settles_a_response_its_client_never_read(gateway):
    sampler = ScriptedSampler(stop_reason="stop")
    sampler.release = asyncio.Event()
    access, held, _ = attempt(gateway, sampler=sampler)
    connection = await connect(access)
    headers = {"Idempotency-Key": "k1"}
    await _disconnect_during_generation(connection, sampler, headers)
    assert held.failure is None
    status, _ = await raw(connection, body=hello(), headers=headers)
    assert status == 200 and len(sampler.prompts) == 1
    assert len((await held.seal()).samples) == 1


async def test_the_official_client_that_times_out_and_retries_trains_only_its_retry(gateway):
    """A stock client, default retries and no Idempotency-Key: the abandoned first
    generation is dropped, the retry the agent received trains."""

    first_done = asyncio.Event()

    class SlowFirst(ScriptedSampler):
        async def sample_async(self, prompt, **kwargs):
            self.release = first_done if not self.prompts else None
            return await super().sample_async(prompt, **kwargs)

    sampler = SlowFirst(stop_reason="stop")
    access, held, _ = attempt(gateway, sampler=sampler)
    connection = await access.http_connection()
    stock = openai.AsyncOpenAI(
        base_url=connection.base_url, api_key=connection.api_key.get_secret_value(), timeout=0.3
    )
    OPENED_CLIENTS.append(stock)
    reply = await stock.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "hi"}]
    )
    assert reply.choices[0].message.content is not None and len(sampler.prompts) == 2
    first_done.set()  # the abandoned first generation finishes, unread
    await access.quiesce(1.0)
    assert held.failure is None
    [trained] = (await held.seal()).samples
    assert reply.id == f"chatcmpl-{trained.sample_id}"


async def test_abort_revokes_the_credential_and_cancels_in_flight_work(gateway):
    sampler = ScriptedSampler(stop_reason="stop")
    sampler.release = asyncio.Event()
    access, held, _ = attempt(gateway, sampler=sampler)
    connection = await connect(access)
    call = asyncio.create_task(raw(connection, body=hello()))
    await sampler.entered.wait()
    held.revoke()  # abort and rollout timeout revoke the lease; no clock of its own
    # The attempt has ended, so its handler is cancelled: the client sees the
    # connection close, or an error if the response was already under way.
    try:
        status, _ = await call
        assert status >= 500
    except aiohttp.ServerDisconnectedError:
        pass
    assert (await raw(connection, body=hello()))[0] == 401


async def test_a_credential_admits_nothing_after_seal_abort_or_gateway_close(gateway):
    sealed, sealed_lease, _ = attempt(gateway, name="sealed")
    sealed_connection = await connect(sealed)
    assert await _models_status(sealed_connection) == 200
    await sealed.quiesce(0.5)
    await sealed_lease.seal()
    assert await _models_status(sealed_connection) == 401
    with pytest.raises(ModelAccessError):
        await sealed.http_connection()

    aborted, aborted_lease, _ = attempt(gateway, name="aborted")
    aborted_connection = await connect(aborted)
    aborted_lease.revoke()
    assert await _models_status(aborted_connection) == 401

    live, _, _ = attempt(gateway, name="live")
    live_connection = await connect(live)
    assert await _models_status(live_connection) == 200
    await gateway.close()
    assert await _models_status(live_connection) is None


async def test_quiescing_waits_for_a_response_being_written_then_cancels_the_rest(gateway):
    sampler = ScriptedSampler(stop_reason="stop")
    sampler.release = asyncio.Event()
    access, held, _ = attempt(gateway, sampler=sampler)
    connection = await connect(access)
    call = asyncio.create_task(raw(connection, body=hello()))
    await sampler.entered.wait()
    loop = asyncio.get_running_loop()
    loop.call_later(0.05, sampler.release.set)  # finishes inside the grace
    await access.quiesce(2.0)
    assert (await call)[0] == 200
    assert len((await held.seal()).samples) == 1

    hanging = ScriptedSampler(stop_reason="stop")
    hanging.release = asyncio.Event()  # never set
    access, held, _ = attempt(gateway, sampler=hanging, name="a2")
    connection = await connect(access)
    call = asyncio.create_task(raw(connection, body=hello()))
    await hanging.entered.wait()
    await access.quiesce(0.1)  # cancelled after the grace: never delivered, not evidence
    assert (await held.seal()).samples == ()
    call.cancel()


@cancellation_matrix(
    triggers=tuple(Trigger),
    points={"grace": (Landing.NOTHING, Landing.CANCEL)},
    responses=tuple(Response),
)
async def test_quiescing_cancels_owned_work_once_and_waits_its_grace_whatever_cancels_it(
    gateway, case
):
    """Work still running when the grace passes, or when quiescing is itself cancelled, is
    cancelled once and given its grace; a cancellation of the waiter never cuts it short."""
    work = Work(case.response)
    access, _, _ = attempt(gateway)
    access.own(asyncio.create_task(work()))
    grace = 0.2 if case.trigger is Trigger.DEADLINE or case.still_running else 30
    caller = asyncio.create_task(access.quiesce(grace))
    try:
        async with asyncio.timeout(5):
            await work.running.wait()
            if case.trigger is Trigger.CANCEL:
                caller.cancel()
            await work.cancelled.wait()
            if case.at("grace") is Landing.CANCEL:
                caller.cancel()
                await settle_loop()
            assert not caller.done()
            work.let_go.set()
            outcome = (await asyncio.gather(caller, return_exceptions=True))[0]
        assert work.cancellations == 1
        assert work.finished is not case.still_running
        assert isinstance(outcome, asyncio.CancelledError) if case.cancelled else outcome is None
    finally:
        work.end()


async def test_an_unkeyed_retry_after_a_disconnect_is_sampled_and_trains(gateway):
    sampler = ScriptedSampler(stop_reason="stop")
    sampler.release = asyncio.Event()
    access, held, _ = attempt(gateway, sampler=sampler)
    connection = await connect(access)
    await _disconnect_during_generation(connection, sampler)
    status, _ = await raw(connection, body=hello())  # what a generic client's retry sends
    assert status == 200
    assert len(sampler.prompts) == 2  # a new request, sampled afresh
    assert len((await held.seal()).samples) == 1  # only the one the agent received


async def test_a_keyed_request_refused_before_sampling_can_be_retried_under_its_key(gateway):
    sampler = ScriptedSampler(stop_reason="stop")
    sampler.release = asyncio.Event()
    access, held, _ = attempt(gateway, sampler=sampler)
    connection = await connect(access)
    conversation = {"X-Loops-Conversation-Id": "session"}
    first = asyncio.create_task(raw(connection, body=hello(), headers=conversation))
    await sampler.entered.wait()
    keyed = {**conversation, "Idempotency-Key": "k1"}
    status, body = await raw(connection, body=hello(), headers=keyed)
    assert status == 409 and body["error"]["code"] == "conversation_busy"
    sampler.release.set()
    await first
    status, _ = await raw(connection, body=hello(), headers=keyed)  # the key named nothing
    assert status == 200 and len((await held.seal()).samples) == 2


async def test_a_revoked_attempt_admits_nothing_even_from_a_handler_already_inside(gateway):
    access, held, sampler = attempt(gateway)
    connection = await connect(access)
    await access.quiesce(0.1)
    assert (await raw(connection, body=hello()))[0] == 401
    from loops_cookbook.model_access.openai.policy import admit
    from loops_cookbook.model_access.openai.schema.models import CreateChatCompletionRequest

    request = admit(CreateChatCompletionRequest.model_validate(hello()), model=MODEL)
    with pytest.raises(AccessEnded):
        access.complete("main", request, sent=hello(), conversation_id=None, key=None, digest="d")
    with pytest.raises(AccessEnded):
        await access.http_connection()
    assert sampler.prompts == [] and (await held.seal()).samples == ()


async def test_the_same_request_spelled_differently_is_not_a_key_conflict(gateway):
    access, _, sampler = attempt(gateway)
    connection = await connect(access)
    headers = {"Idempotency-Key": "k1"}
    assert (await raw(connection, body=hello(temperature=1), headers=headers))[0] == 200
    assert (await raw(connection, body=hello(temperature=1.0), headers=headers))[0] == 200
    assert (await raw(connection, body=hello(temperature=1, n=None), headers=headers))[0] == 200
    assert len(sampler.prompts) == 1


async def test_an_unkeyed_stream_cut_after_sampling_is_dropped(gateway):
    renderer = Renderer(reply=_huge_reply)
    access, held, _ = attempt(gateway, renderer=renderer)
    connection = await connect(access)
    await _hang_up_mid_stream(connection)
    await access.quiesce(1.0)
    assert held.failure is None and (await held.seal()).samples == ()


async def test_a_keyed_retry_settles_a_stream_cut_after_sampling(gateway):
    renderer = Renderer(reply=_huge_reply)
    access, held, sampler = attempt(gateway, renderer=renderer)
    connection = await connect(access)
    headers = {"Idempotency-Key": "k1"}
    await _hang_up_mid_stream(connection, headers)
    assert held.failure is None
    async with (
        aiohttp.ClientSession() as session,
        session.post(
            connection.base_url + "/chat/completions",
            json=hello(stream=True),
            headers={
                "Authorization": f"Bearer {connection.api_key.get_secret_value()}",
                **headers,
            },
        ) as response,
    ):
        assert (await response.read()).endswith(b"data: [DONE]\n\n")
    # The client can read the end before the handler resumes to settle the delivery.
    await access.quiesce(1.0)
    assert len(sampler.prompts) == 1 and len((await held.seal()).samples) == 1
