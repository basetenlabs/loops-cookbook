"""The gateway rig: a real gateway on loopback, a real attempt lease, a scripted sampler."""

from functools import partial

import aiohttp
import openai
from renderers.base import ParsedToolCall, ToolCallParseStatus
from support.fakes import ScriptedRenderer, ScriptedSampler
from support.images import png
from support.leases import lease

from loops_cookbook.evidence import AttemptIdentity
from loops_cookbook.llm.client import Renderers
from loops_cookbook.llm.rendering import ParsedReply
from loops_cookbook.model_access.openai.access import HttpModelAccess

MODEL = "policy"


PNG = png((2, 2))


# Bridges each next turn and takes images, as a chat template the gateway serves does.
Renderer = partial(ScriptedRenderer, bridging=True, images=True)


def identity(attempt="a1"):
    return AttemptIdentity("run", "g", f"r-{attempt}", attempt)


OPENED_CLIENTS: list[openai.AsyncOpenAI] = []


def attempt(gateway, *, sampler=None, renderer=None, name="a1", **options):
    sampler = sampler or ScriptedSampler(stop_reason="stop")
    renderers = options.pop("renderers", None) or Renderers(renderer or Renderer())
    held = lease(sampler, identity=identity(name), **options)
    return HttpModelAccess(held, renderers, gateway), held, sampler


async def client(access, producer=None):
    connection = await access.http_connection(producer)
    opened = openai.AsyncOpenAI(
        base_url=connection.base_url,
        api_key=connection.api_key.get_secret_value(),
        max_retries=0,
    )
    OPENED_CLIENTS.append(opened)
    return opened, connection


async def connect(access, producer=None):
    """The attempt's connection, for a test that speaks raw HTTP."""
    return await access.http_connection(producer)


async def raw(connection, path="/chat/completions", *, method="POST", body=None, headers=None):
    async with aiohttp.ClientSession() as session:
        request = session.request(
            method,
            connection.base_url + path,
            json=body,
            headers={
                "Authorization": f"Bearer {connection.api_key.get_secret_value()}",
                **(headers or {}),
            },
        )
        async with request as response:
            return response.status, await response.json()


def hello(**fields):
    return {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], **fields}


def _broken_call(status=ToolCallParseStatus.INVALID_JSON, *, content="let me look", size=0):
    """A reply with one good call and one the renderer could not parse, so no message."""

    def reply(_n):
        good = ParsedToolCall(raw='{"name": "f"}', name="f", arguments={})
        bad = ParsedToolCall(raw='{"name": "f", "argu' + "x" * size, status=status)
        return ParsedReply(content, "hmm", (good, bad), True, None)

    return reply


TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]


def _same_tool_call(n):
    from renderers.base import ParsedToolCall

    call = ParsedToolCall(raw="", name="f", arguments={}, id="call_same")
    native = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_same", "type": "function", "function": {"name": "f", "arguments": {}}}
        ],
    }
    return ParsedReply("", None, (call,), True, native)


def _huge_reply(n):
    text = "x" * (8 * 1024 * 1024)  # more than any socket buffers before a hang-up
    return ParsedReply(text, None, (), True, {"role": "assistant", "content": text})


async def _hang_up_mid_stream(connection, headers=None):
    """A streaming client that reads the first event, then goes away.

    The generation is sampled before the first event is written; the handler may
    still be writing the rest. A test waits it out with ``quiesce`` before it seals.
    """
    session = aiohttp.ClientSession()
    response = await session.post(
        connection.base_url + "/chat/completions",
        json=hello(stream=True),
        headers={
            "Authorization": f"Bearer {connection.api_key.get_secret_value()}",
            **(headers or {}),
        },
    )
    await response.content.readline()
    response.close()
    await session.close()


async def _discard(connection, completion_id):
    return await raw(connection, "/loops/discard", body={"id": completion_id})


async def _sealed(access_held):
    sealed = await access_held.seal()
    assert sealed.transcript is not None
    return sealed.transcript, [turn.turn for turn in sealed.transcript.turns]
