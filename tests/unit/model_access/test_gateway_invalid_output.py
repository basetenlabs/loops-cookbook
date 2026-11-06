"""The Chat Completions gateway's 422: a tool call the model got wrong reaches the agent
with its reply, trains, and can be continued from its sampled tokens.

Every test runs the real gateway on loopback, a real attempt lease, and a scripted
sampler; nothing reaches the network.
"""

import aiohttp
import openai
import pytest
from renderers.base import ToolCallParseStatus
from support.fakes import ScriptedSampler
from support.leases import lease

from loops_cookbook.llm.client import Renderers
from loops_cookbook.llm.transcript import CompletionTurnReply, thaw
from loops_cookbook.model_access.lease import DirectModelAccess

from .rig import (
    MODEL,
    OPENED_CLIENTS,
    TOOLS,
    Renderer,
    _broken_call,
    _same_tool_call,
    _sealed,
    attempt,
    client,
    connect,
    hello,
    identity,
    raw,
)


def _echo(reply):
    """The assistant message an agent echoes for a 422's ``error.reply``."""
    calls = [{key: call[key] for key in ("id", "type", "function")} for call in reply["tool_calls"]]
    return {"role": "assistant", "content": reply["content"], "tool_calls": calls}


def _answers(reply):
    return [
        {"role": "tool", "tool_call_id": call["id"], "content": "error"}
        for call in reply["tool_calls"]
    ]


@pytest.mark.parametrize(
    ("status", "stop_reason", "code"),
    [
        (ToolCallParseStatus.INVALID_JSON, "stop", "tool_call_parse_error"),
        (ToolCallParseStatus.UNKNOWN_TOOL, "stop", "tool_call_parse_error"),
        (ToolCallParseStatus.INVALID_JSON, "length", "tool_call_truncated"),
        (ToolCallParseStatus.UNCLOSED_BLOCK, "stop", "tool_call_truncated"),
    ],
    ids=["invalid_json", "unknown_tool", "cut_by_max_tokens", "unclosed_block"],
)
@pytest.mark.parametrize("stream", [False, True], ids=["whole", "streamed"])
async def test_a_malformed_tool_call_is_a_422_carrying_the_models_reply_and_trains(
    gateway, status, stop_reason, code, stream
):
    renderer = Renderer(reply=_broken_call(status))
    access, held, _ = attempt(
        gateway, sampler=ScriptedSampler(stop_reason=stop_reason), renderer=renderer
    )
    connection = await connect(access)
    answer, body = await raw(connection, body=hello(tools=TOOLS, stream=stream))
    (sample,) = (await held.seal()).samples  # delivered: the agent received the output
    assert answer == 422
    assert body == {
        "error": {
            "message": body["error"]["message"],
            "type": "model_output_invalid",
            "param": None,
            "code": code,
            "reply": {
                "id": f"chatcmpl-{sample.sample_id}",
                "content": "let me look",
                "reasoning_content": "hmm",
                "tool_calls": [
                    {
                        "id": f"call_{sample.sample_id}_0",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                        "raw": '{"name": "f"}',
                        "status": "ok",
                    },
                    {  # nothing read: no name, and the raw call as its arguments
                        "id": f"call_{sample.sample_id}_1",
                        "type": "function",
                        "function": {"name": "", "arguments": '{"name": "f", "argu'},
                        "raw": '{"name": "f", "argu',
                        "status": status.value,
                    },
                ],
                "usage": body["error"]["reply"]["usage"],
            },
        }
    }
    assert body["error"]["message"]
    usage = body["error"]["reply"]["usage"]
    assert usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


async def test_a_422_continues_from_its_sampled_tokens_as_a_direct_conversation_does(gateway):
    renderer, direct_renderer = Renderer(), Renderer()
    for scripted in (renderer, direct_renderer):
        first = _broken_call()
        scripted.reply = lambda n, first=first: first(n) if n == 1 else Renderer().reply(n)
    access, held, _ = attempt(gateway, renderer=renderer)
    connection = await connect(access)
    headers = {"X-Loops-Conversation-Id": "session"}
    start = [{"role": "user", "content": "start"}]
    status, body = await raw(connection, body=hello(messages=start, tools=TOOLS), headers=headers)
    assert status == 422
    reply = body["error"]["reply"]
    followup = [*start, _echo(reply), *_answers(reply)]
    status, _ = await raw(connection, body=hello(messages=followup, tools=TOOLS), headers=headers)
    assert status == 200
    over_http = (await held.seal()).samples
    assert len(renderer.renders) == 1 and len(renderer.bridges) == 1

    direct_lease = lease(ScriptedSampler(stop_reason="stop"), identity=identity("direct"))
    chat = (
        DirectModelAccess(direct_lease, Renderers(direct_renderer))
        .llm()
        .conversation(conversation_id="session")
    )
    declared = [tool["function"] for tool in TOOLS]
    unparsed = await chat.generate(messages=start, tools=declared)
    calls = unparsed.history_message["tool_calls"]
    answers = [{"role": "tool", "tool_call_id": c["id"], "content": "error"} for c in calls]
    await chat.generate(messages=[*start, unparsed.history_message, *answers], tools=declared)
    direct = (await direct_lease.seal()).samples
    assert [(s.prompt_ids, s.completion_ids) for s in over_http] == [
        (s.prompt_ids, s.completion_ids) for s in direct
    ]
    first, second = over_http
    assert second.prompt_ids[: len(first.prompt_ids) + len(first.completion_ids)] == (
        first.prompt_ids + first.completion_ids
    )


async def test_a_422_that_cannot_be_continued_from_its_tokens_is_refused_not_rendered(gateway):
    renderer = Renderer(reply=_broken_call())
    access, held, sampler = attempt(gateway, renderer=renderer)
    connection = await connect(access)
    headers = {"X-Loops-Conversation-Id": "session"}
    start = [{"role": "user", "content": "start"}]
    _, body = await raw(connection, body=hello(messages=start, tools=TOOLS), headers=headers)
    reply = body["error"]["reply"]
    changed = [{"type": "function", "function": {"name": "other"}}]
    status, refused = await raw(
        connection,
        body=hello(messages=[*start, _echo(reply), *_answers(reply)], tools=changed),
        headers=headers,
    )
    assert status == 400 and refused["error"]["code"] == "rendering_failed"
    assert len(sampler.prompts) == 1 and renderer.bridges == []
    assert held.failure is None  # the attempt goes on


async def test_a_discarded_422_echoed_back_is_refused_not_rendered(gateway):
    renderer = Renderer(reply=_broken_call())
    access, held, sampler = attempt(gateway, renderer=renderer)
    connection = await connect(access)
    headers = {"X-Loops-Conversation-Id": "session"}
    start = [{"role": "user", "content": "start"}]
    _, body = await raw(connection, body=hello(messages=start, tools=TOOLS), headers=headers)
    reply = body["error"]["reply"]
    status, _ = await raw(connection, "/loops/discard", body={"id": reply["id"]})
    assert status == 200
    status, refused = await raw(
        connection,
        body=hello(messages=[*start, _echo(reply), *_answers(reply)], tools=TOOLS),
        headers=headers,
    )
    assert status == 400 and refused["error"]["code"] == "rendering_failed"
    assert "discarded" in refused["error"]["message"]
    assert len(sampler.prompts) == 1 and len(renderer.renders) == 1
    assert held.failure is None  # the attempt goes on


async def test_without_the_header_a_422_echo_is_the_agents_own_history(gateway):
    """Unrecognised, it renders from what the agent sent: unparsed arguments cannot."""
    renderer = Renderer(reply=_broken_call())
    access, _, sampler = attempt(gateway, renderer=renderer)
    connection = await connect(access)
    start = [{"role": "user", "content": "start"}]
    _, body = await raw(connection, body=hello(messages=start, tools=TOOLS))
    reply = body["error"]["reply"]
    status, refused = await raw(
        connection, body=hello(messages=[*start, _echo(reply), *_answers(reply)], tools=TOOLS)
    )
    assert status == 400 and refused["error"]["param"].endswith("function.arguments")
    assert len(sampler.prompts) == 1


async def test_a_stock_client_raises_the_422_without_retrying_it(gateway):
    renderer = Renderer(reply=_broken_call())
    access, held, sampler = attempt(gateway, renderer=renderer)
    connection = await access.http_connection()
    stock = openai.AsyncOpenAI(  # the SDK's default retries: 408, 409, 429 and 5xx
        base_url=connection.base_url, api_key=connection.api_key.get_secret_value()
    )
    OPENED_CLIENTS.append(stock)
    with pytest.raises(openai.UnprocessableEntityError) as refused:
        await stock.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "hi"}], tools=TOOLS
        )
    assert refused.value.status_code == 422 and refused.value.code == "tool_call_parse_error"
    assert refused.value.type == "model_output_invalid"
    assert isinstance(refused.value.body, dict) and "reply" in refused.value.body
    assert len(sampler.prompts) == 1  # not retried
    assert len((await held.seal()).samples) == 1


async def test_a_422_whose_write_fails_does_not_train(gateway):
    renderer = Renderer(reply=_broken_call(size=8 * 1024 * 1024))  # more than any socket buffers
    access, held, _ = attempt(gateway, renderer=renderer)
    connection = await connect(access)
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            connection.base_url + "/chat/completions",
            json=hello(tools=TOOLS),
            headers={"Authorization": f"Bearer {connection.api_key.get_secret_value()}"},
        )
        assert response.status == 422
        await response.content.read(1024)
        response.close()  # gone before the body was all written
    await access.quiesce(1.0)
    assert held.failure is None
    assert (await held.seal()).samples == ()


async def test_the_transcript_keeps_the_422_body_as_the_reply_the_agent_received(gateway):
    renderer = Renderer(reply=_broken_call())
    access, held, _ = attempt(gateway, renderer=renderer, transcripts=[])
    connection = await connect(access)
    _, body = await raw(connection, body=hello(tools=TOOLS))
    await access.quiesce(1.0)
    _, (turn,) = await _sealed(held)
    assert isinstance(turn.reply, CompletionTurnReply) and thaw(turn.reply.body) == body


async def test_a_422_reply_can_be_discarded_by_its_id(gateway):
    renderer = Renderer(reply=_broken_call())
    access, held, _ = attempt(gateway, renderer=renderer)
    connection = await connect(access)
    _, body = await raw(connection, body=hello(tools=TOOLS))
    completion_id = body["error"]["reply"]["id"]
    status, _ = await raw(connection, "/loops/discard", body={"id": completion_id})
    assert status == 200 and (await held.seal()).samples == ()


@pytest.mark.parametrize(
    "offered", [{"tool_choice": "none"}, {}], ids=["tool_choice_none", "no_tools"]
)
async def test_a_tool_call_is_not_returned_when_no_tools_were_offered(gateway, offered):
    renderer = Renderer(reply=_same_tool_call)
    access, held, _ = attempt(gateway, renderer=renderer)
    openai_client, _ = await client(access)
    reply = await openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        **({"tools": TOOLS, **offered} if offered else {}),
    )
    assert reply.choices[0].message.tool_calls is None
    assert reply.choices[0].finish_reason == "stop"
    assert len((await held.seal()).samples) == 1  # still captured
