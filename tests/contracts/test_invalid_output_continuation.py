"""After a tool call the renderer could not parse, the conversation continues exactly.

Real tokenizers and renderers, through the gateway: the model writes a call cut short
by its token limit, or a call to a tool that does not exist. The first is a 422 whose
reply the agent echoes and answers; the second is a response like any other, the call's
status saying what the parser made of it. Either way the next prompt is the previous
prompt plus the exact sampled completion, so the episode trains as one chain and the
model sees what it wrote, never a stand-in.
"""

from types import SimpleNamespace

import aiohttp
import pytest
from support.leases import lease
from support.unit_rest import BASH_TOOLS as TOOLS

from loops_cookbook.evidence import AttemptIdentity
from loops_cookbook.llm.client import Renderers
from loops_cookbook.llm.factory import create_renderer
from loops_cookbook.model_access.lease import DirectModelAccess
from loops_cookbook.model_access.openai.access import HttpModelAccess
from loops_cookbook.model_access.openai.gateway import Gateway, GatewaySettings

# A call to a tool that was never declared is read whole, so its message stays in history.
UNKNOWN_CALL = (
    "<tool_call>\n<function=bsah>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
)
HEADERS = {"X-Loops-Conversation-Id": "agent"}


class Sampler:
    """Replies with the scripted texts in turn, encoded as the model would emit them.

    A text ending without the turn's end is a generation cut at ``max_tokens``.
    """

    def __init__(self, tokenizer, texts, stop_reasons=None):
        self.replies = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
        self.stop_reasons = stop_reasons or ["stop"] * len(texts)
        self.prompts = []

    async def sample_async(self, prompt, **kwargs):
        self.prompts.append(prompt.to_ints())
        tokens = self.replies[len(self.prompts) - 1]
        stop_reason = self.stop_reasons[len(self.prompts) - 1]
        return SimpleNamespace(
            prompt_token_ids=prompt.to_ints(),
            policy_version=0,
            sequences=[
                SimpleNamespace(
                    tokens=tokens,
                    logprobs=[-0.1] * len(tokens),
                    stop_reason=stop_reason,
                    sequence_id=None,
                )
            ],
        )


async def _post(connection, body):
    async with (
        aiohttp.ClientSession() as session,
        session.post(
            connection.base_url + "/chat/completions",
            json=body,
            headers={
                "Authorization": f"Bearer {connection.api_key.get_secret_value()}",
                **HEADERS,
            },
        ) as response,
    ):
        return response.status, await response.json()


async def test_an_unknown_tool_call_is_answered_and_continues_from_its_sampled_tokens(
    tool_call_model,
):
    tokenizer, renderer, prefix = tool_call_model
    end = "<|im_end|>"
    sampler = Sampler(tokenizer, [prefix + UNKNOWN_CALL + end, prefix + "Done." + end])
    gateway = Gateway(GatewaySettings("127.0.0.1:0", None, "policy", shutdown_seconds=1.0))
    try:
        held = lease(sampler, identity=AttemptIdentity("run", "g", "r", "a"))
        access = HttpModelAccess(held, Renderers(renderer), gateway)
        connection = await access.http_connection()
        history = [{"role": "user", "content": "List the files."}]

        status, first = await _post(
            connection, {"model": "policy", "messages": history, "tools": TOOLS}
        )
        assert status == 200, first
        message = first["choices"][0]["message"]
        [call] = message["tool_calls"]
        assert call["function"] == {"name": "bsah", "arguments": '{"command": "ls"}'}

        # The agent's tool layer finds no such tool and says so in its answer to the call.
        echo = {"role": "assistant", "content": message["content"], "tool_calls": [call]}
        answer = {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": "Error: unknown tool: bsah",
        }
        status, second = await _post(
            connection,
            {"model": "policy", "messages": [*history, echo, answer], "tools": TOOLS},
        )
        assert status == 200, second
        assert second["choices"][0]["message"]["content"] == "Done."

        first_sample, second_sample = (await held.seal()).samples
        sampled = first_sample.prompt_ids + first_sample.completion_ids
        assert second_sample.prompt_ids[: len(sampled)] == sampled
        rest = tokenizer.decode(list(second_sample.prompt_ids[len(sampled) :]))
        assert "Error: unknown tool: bsah" in rest
        assert "bsah" not in rest.replace("Error: unknown tool: bsah", "")  # never re-rendered
    finally:
        await gateway.close()


GLM_RENDERER = "loops_cookbook.llm.renderer_extensions.glm53:renderer"
TIMED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
                "required": ["command"],
            },
        },
    }
]
# A model gets two other things wrong. It writes an integer as "30s", which reaches the
# tool, and its call gets cut off at max_tokens, which doesn't parse. Each family writes
# them in its own syntax.
BAD_CALLS = {
    ("xml", "invalid_json"): (
        "<tool_call>\n<function=bash>\n<parameter=command>\nsleep 5 && ls\n</parameter>\n"
        "<parameter=timeout>\n30s\n</parameter>\n</function>\n</tool_call>"
    ),
    ("xml", "truncated"): (
        "<tool_call>\n<function=bash>\n<parameter=command>\nfind . -name '*.py' -exec grep -l"
    ),
    ("glm", "truncated"): (
        "<tool_call>bash<arg_key>command</arg_key><arg_value>find . -name '*.py' -exec grep -l"
    ),
}
EXPECTED = {"truncated": ("tool_call_truncated", "unclosed_block")}
RESULT = "Error: could not run the call"
ASK = [{"role": "user", "content": "List the files, give up after 30 seconds."}]


@pytest.fixture
def xml_model(tool_call_model):
    """Qwen and Nemotron: tokenizer, renderer, syntax, reply prefix and turn end."""
    tokenizer, renderer, prefix = tool_call_model
    return tokenizer, renderer, "xml", prefix, "<|im_end|>"


@pytest.fixture
def glm_model(glm53_tokenizer):
    """GLM-5.3: tokenizer, renderer, syntax, reply prefix and turn end."""
    renderer = create_renderer(
        glm53_tokenizer.name_or_path, tokenizer=glm53_tokenizer, renderer=GLM_RENDERER
    )
    return glm53_tokenizer, renderer, "glm", "I should list the files.</think>", "<|observation|>"


def _bad_call_sampler(model, shape):
    tokenizer, _, syntax, prefix, end = model
    call = prefix + BAD_CALLS[(syntax, shape)]
    if shape == "invalid_json":
        return Sampler(tokenizer, [call + end, prefix + "Done." + end])
    return Sampler(tokenizer, [call, prefix + "Done." + end], ["length", "stop"])


def _continues_from_sampled_tokens(tokenizer, samples):
    first_sample, second_sample = samples
    sampled = first_sample.prompt_ids + first_sample.completion_ids
    assert second_sample.prompt_ids[: len(sampled)] == sampled
    rest = tokenizer.decode(list(second_sample.prompt_ids[len(sampled) :]))
    assert RESULT in rest
    assert "30s" not in rest and "find ." not in rest  # never re-rendered


async def _continues_through_the_gateway(bad_call_model, shape):
    tokenizer, renderer, *_ = bad_call_model
    code, status_value = EXPECTED[shape]
    gateway = Gateway(GatewaySettings("127.0.0.1:0", None, "policy", shutdown_seconds=1.0))
    try:
        held = lease(
            _bad_call_sampler(bad_call_model, shape),
            identity=AttemptIdentity("run", "g", "r", "a"),
        )
        access = HttpModelAccess(held, Renderers(renderer), gateway)
        connection = await access.http_connection()
        body = {"model": "policy", "messages": ASK, "tools": TIMED_TOOLS}
        status, first = await _post(connection, body)
        assert status == 422 and first["error"]["code"] == code
        reply = first["error"]["reply"]
        [call] = reply["tool_calls"]
        assert call["status"] == status_value

        echo = {
            "role": "assistant",
            "content": reply["content"],
            "tool_calls": [{key: call[key] for key in ("id", "type", "function")}],
        }
        answer = {"role": "tool", "tool_call_id": call["id"], "content": RESULT}
        status, second = await _post(connection, {**body, "messages": [*ASK, echo, answer]})
        assert status == 200, second
        assert second["choices"][0]["message"]["content"] == "Done."
        _continues_from_sampled_tokens(tokenizer, (await held.seal()).samples)
    finally:
        await gateway.close()


async def _continues_directly(bad_call_model, shape):
    tokenizer, renderer, *_ = bad_call_model
    held = lease(
        _bad_call_sampler(bad_call_model, shape), identity=AttemptIdentity("run", "g", "r", "a")
    )
    chat = DirectModelAccess(held, Renderers(renderer)).llm().conversation()

    reply = await chat.generate(messages=ASK, tools=TIMED_TOOLS, max_tokens=64)
    [call] = reply.tool_calls
    assert reply.native_message is None and call.status == EXPECTED[shape][1]
    assert reply.truncated is (shape == "truncated")
    written = reply.history_message
    results = [
        {"role": "tool", "tool_call_id": each["id"], "content": RESULT}
        for each in written["tool_calls"]
    ]
    second = await chat.generate(
        messages=[*ASK, written, *results], tools=TIMED_TOOLS, max_tokens=64
    )
    assert second.content == "Done."
    _continues_from_sampled_tokens(tokenizer, (await held.seal()).samples)


@pytest.mark.parametrize("shape", sorted(EXPECTED))
@pytest.mark.parametrize("path", [_continues_through_the_gateway, _continues_directly])
async def test_an_xml_call_that_does_not_parse_continues_from_its_sampled_tokens(
    xml_model, path, shape
):
    await path(xml_model, shape)


def test_an_xml_value_its_type_cannot_read_reaches_the_tool(xml_model):
    """Keep the message of a call whose value its type can't read, for the tool to judge.

    The cookbook's Nemotron renderers read ``30s`` as text and report ``ok``. Prime's Qwen
    renderer reports ``invalid_json``. Either way the call was read whole, so its message
    stays in history.
    """
    tokenizer, renderer, syntax, prefix, end = xml_model
    prompt = renderer.render(ASK, TIMED_TOOLS).ids
    completion = tokenizer.encode(
        prefix + BAD_CALLS[(syntax, "invalid_json")] + end, add_special_tokens=False
    )
    reply = renderer.parse(prompt, completion, TIMED_TOOLS)
    [parsed] = reply.tool_calls
    assert parsed.status.value in ("ok", "invalid_json")
    assert parsed.arguments == {"command": "sleep 5 && ls", "timeout": "30s"}
    assert reply.native_message["tool_calls"][0]["function"]["arguments"] == parsed.arguments


@pytest.mark.parametrize("shape", sorted(EXPECTED))
@pytest.mark.parametrize("path", [_continues_through_the_gateway, _continues_directly])
async def test_a_glm_call_that_does_not_parse_continues_from_its_sampled_tokens(
    glm_model, path, shape
):
    await path(glm_model, shape)


def test_a_glm_value_its_type_cannot_read_reaches_the_tool(glm_model):
    tokenizer, renderer, _, prefix, end = glm_model
    call = (
        "<tool_call>bash<arg_key>command</arg_key><arg_value>sleep 5 && ls</arg_value>"
        "<arg_key>timeout</arg_key><arg_value>30s</arg_value></tool_call>"
    )
    prompt = renderer.render(ASK, TIMED_TOOLS).ids
    completion = tokenizer.encode(prefix + call + end, add_special_tokens=False)
    reply = renderer.parse(prompt, completion, TIMED_TOOLS)
    [parsed] = reply.tool_calls
    assert parsed.status.value == "ok"
    assert parsed.arguments == {"command": "sleep 5 && ls", "timeout": "30s"}
    assert reply.native_message is not None
