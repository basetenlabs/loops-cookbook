"""Test tool-call arguments against each model's own chat template.

Random calls that the template renders parse back to the same arguments and types. A call
that the agent echoes through the gateway continues from the tokens the model sampled.
"""

from __future__ import annotations

import json
import random
from enum import Enum

import pytest
from pydantic import BaseModel
from support.leases import lease

from loops_cookbook.evidence import AttemptIdentity
from loops_cookbook.llm.client import Renderers
from loops_cookbook.llm.factory import create_renderer
from loops_cookbook.model_access.openai.access import HttpModelAccess
from loops_cookbook.model_access.openai.gateway import Gateway, GatewaySettings

from .conftest import (
    NEMOTRON_SUPER_VL,
    NEMOTRON_SUPER_VL_REVISIONS,
    NEMOTRON_ULTRA,
    NEMOTRON_ULTRA_REVISION,
    pinned_tokenizer,
)
from .test_invalid_output_continuation import Sampler, _post

GLM53 = "loops_cookbook.llm.renderer_extensions.glm53:renderer"


class Mode(str, Enum):
    read = "read"
    write = "write"


class Sub(BaseModel):
    x: int
    label: str = ""


class Declared(BaseModel):
    """Declare tool arguments the way pydantic writes them.

    Pydantic writes ``anyOf`` for an optional value, and a ``$ref`` into ``$defs`` for a
    ``str`` Enum or a nested model.
    """

    path: str | None = None
    mode: Mode = Mode.read
    sub: Sub | None = None
    lines: list[int] = []


PYDANTIC = Declared.model_json_schema()
SCHEMAS = {
    **PYDANTIC["properties"],
    "s": {"type": "string"},
    "i": {"type": "integer"},
    "x": {"type": "number"},
    "b": {"type": "boolean"},
    "os": {"type": ["string", "null"]},
    "oi": {"type": ["integer", "null"]},
    "u": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
    "e": {"enum": ["low", "high", 3]},
    "a": {"type": "array"},
    "o": {"type": "object"},
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "act",
            "parameters": {"type": "object", "properties": SCHEMAS, "$defs": PYDANTIC["$defs"]},
        },
    }
]
# These pieces make up what a string argument holds in practice. They cover code,
# whitespace, quotes, escapes, text that looks like JSON or Python, and text outside ASCII.
PIECES = [
    " ", "  ", "\n", "\n\n", "\t", "\r\n", "'", '"', '"""', "\\", "\\n", "{", "}", "[", "]",
    ":", ",", "null", "None", "True", "true", "false", "7", "-3.5", "1e3", "é", "合同", "🦀",
    "<", ">", "/", "=", "def f():", "    return 1", "#", "`", "x", "pending",
]  # fmt: skip
# A string that holds GLM's own markup cannot be told apart from the call's structure.
GLM_MARKUP = ("<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "tool_call>", "<|")


def _text(rng: random.Random, markup: tuple[str, ...]) -> str:
    while True:
        text = "".join(rng.choice(PIECES) for _ in range(rng.randint(0, 12)))
        if not any(marker in text for marker in markup):
            return text


def _reads_as_json(text: str) -> object:
    try:
        return json.loads(text)
    except ValueError:
        return text


def _json(rng: random.Random, depth: int = 0) -> object:
    kind = rng.choice(
        ["int", "float", "bool", "null", "str"] + (["list", "dict"] if depth < 2 else [])
    )
    if kind == "int":
        return rng.randint(-(10**12), 10**12)
    if kind == "float":
        return rng.choice([0.5, -1e-07, 3.25, 1e21, 2.0, -0.0])
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "null":
        return None
    if kind == "str":
        return rng.choice(PIECES)
    if kind == "list":
        return [_json(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return {f"k{n}": _json(rng, depth + 1) for n in range(rng.randint(0, 3))}


def random_arguments(
    rng: random.Random, markup: tuple[str, ...], *, python_null: bool = False
) -> dict[str, object]:
    """Return random arguments that the template writes unambiguously.

    A string that reads as another type the schema admits is left out. The template writes
    it exactly as that type, so it cannot round-trip. Examples are ``7`` in an
    integer-or-string union, ``null`` in an optional string, and ``None`` where the
    template writes null the way Python does.
    """

    def string_not(*kinds: type) -> str:
        while True:
            text = _text(rng, markup)
            value = _reads_as_json(text)
            if python_null and text.strip() == "None":
                continue
            if not (value is None or isinstance(value, kinds)) or isinstance(value, str):
                return text

    makers = {
        "s": lambda: _text(rng, markup),
        "i": lambda: rng.randint(-(10**12), 10**12),
        "x": lambda: rng.choice([0.5, -1e-07, 3.25, 1e21, 2.0, 7]),
        "b": lambda: rng.random() < 0.5,
        "os": lambda: rng.choice([None, string_not()]),
        "oi": lambda: rng.choice([None, rng.randint(-5, 5)]),
        "u": lambda: rng.choice([rng.randint(-5, 5), string_not(int)]),
        "e": lambda: rng.choice(["low", "high", 3]),
        "a": lambda: [_json(rng, 1) for _ in range(rng.randint(0, 4))],
        "o": lambda: {f"k{n}": _json(rng, 1) for n in range(rng.randint(0, 4))},
        "path": lambda: rng.choice([None, string_not()]),
        "mode": lambda: rng.choice(["read", "write"]),
        "sub": lambda: rng.choice([None, {"x": rng.randint(-5, 5), "label": rng.choice(PIECES)}]),
        "lines": lambda: [rng.randint(0, 99) for _ in range(rng.randint(0, 3))],
    }
    keys = [key for key in makers if rng.random() < 0.7]
    return {key: makers[key]() for key in keys}


def same(left: object, right: object) -> bool:
    """Return whether two values are equal and of the same JSON types.

    Under this test ``True`` differs from ``1``, and ``2.0`` differs from ``2``.
    """
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def parsed_back(renderer, tokenizer, arguments, user, end_token, prefix=None):
    """Render ``arguments`` in an assistant turn with the template, then parse it as a reply.

    With ``prefix``, the reply is that text followed by the turn's call as the template
    wrote it. That suits templates that render past reasoning differently from a
    generation prompt. Without ``prefix``, the reply is the turn exactly as rendered
    after the prompt.
    """
    messages = [{"role": "user", "content": user}]
    prompt = renderer.render(messages, TOOLS).ids
    call = {"id": "c1", "type": "function", "function": {"name": "act", "arguments": arguments}}
    history = [
        *messages,
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    ids = list(renderer.render(history, TOOLS).ids)
    end_id = tokenizer.convert_tokens_to_ids(end_token)
    if prefix is None:
        assert ids[: len(prompt)] == list(prompt)
        completion = ids[len(prompt) : ids.index(end_id, len(prompt)) + 1]
    else:
        start_id = tokenizer.convert_tokens_to_ids("<tool_call>")
        start = max(i for i, token in enumerate(ids) if token == start_id)
        completion = (
            tokenizer.encode(prefix, add_special_tokens=False)
            + ids[start : ids.index(end_id, start) + 1]
        )
    reply = renderer.parse(prompt, completion, TOOLS)
    [result] = reply.tool_calls
    return result, reply


@pytest.mark.parametrize("seed", range(300))
def test_glm_random_calls_parse_back_to_themselves(glm_tokenizer, seed):
    renderer = create_renderer("zai-org/GLM-5.3", renderer=GLM53, tokenizer=glm_tokenizer)
    arguments = random_arguments(random.Random(seed), GLM_MARKUP)
    result, reply = parsed_back(renderer, glm_tokenizer, arguments, "Act.", "<|observation|>")
    assert result.status.value == "ok"
    assert same(result.arguments, arguments), (result.arguments, arguments)
    assert reply.native_message is not None
    assert same(reply.native_message["tool_calls"][0]["function"]["arguments"], arguments)


EDIT = {"s": "    x = 1\n", "os": '"""Doc."""', "u": "'pending'", "i": 3, "oi": None}


async def _echo_continues(tokenizer, renderer, call_text, prefix, end):
    """Check that a call comes back exact through the gateway.

    After the agent echoes the call and answers it, the next prompt continues from the
    sampled tokens.
    """
    sampler = Sampler(tokenizer, [prefix + call_text + end, prefix + "Done." + end])
    gateway = Gateway(GatewaySettings("127.0.0.1:0", None, "policy", shutdown_seconds=1.0))
    try:
        held = lease(sampler, identity=AttemptIdentity("run", "g", "r", "a"))
        access = HttpModelAccess(held, Renderers(renderer), gateway)
        connection = await access.http_connection()
        history = [{"role": "user", "content": "Edit the file."}]
        status, first = await _post(
            connection, {"model": "policy", "messages": history, "tools": TOOLS}
        )
        assert status == 200, first
        message = first["choices"][0]["message"]
        [call] = message["tool_calls"]
        assert same(json.loads(call["function"]["arguments"]), EDIT)
        echo = {"role": "assistant", "content": message["content"], "tool_calls": [call]}
        answer = {"role": "tool", "tool_call_id": call["id"], "content": "Edited."}
        status, second = await _post(
            connection,
            {"model": "policy", "messages": [*history, echo, answer], "tools": TOOLS},
        )
        assert status == 200, second
        first_sample, second_sample = (await held.seal()).samples
        sampled = first_sample.prompt_ids + first_sample.completion_ids
        assert second_sample.prompt_ids[: len(sampled)] == sampled
        assert "Edited." in tokenizer.decode(list(second_sample.prompt_ids[len(sampled) :]))
    finally:
        await gateway.close()


async def test_glm_echoed_call_continues_from_its_sampled_tokens(glm_tokenizer):
    renderer = create_renderer("zai-org/GLM-5.3", renderer=GLM53, tokenizer=glm_tokenizer)
    body = "".join(
        f"<arg_key>{key}</arg_key><arg_value>"
        f"{value if isinstance(value, str) else json.dumps(value)}</arg_value>"
        for key, value in EDIT.items()
    )
    await _echo_continues(
        glm_tokenizer,
        renderer,
        f"<tool_call>act{body}</tool_call>",
        "I will edit it.</think>",
        "<|observation|>",
    )


QWEN35 = "loops_cookbook.llm.renderer_extensions.qwen35:renderer"
# A string that holds Qwen3.5 and Nemotron 3 markup cannot be told apart from the call.
XML_MARKUP = ("<parameter=", "</parameter>", "<function=", "</function>", "tool_call>", "<|")
# Each model name maps to its revision, cache variable, renderer setting, chat options and
# reply prefix.
XML_MODELS = {
    "Qwen/Qwen3.5-9B": (
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "LOOPS_QWEN_TOKENIZER_CACHE",
        QWEN35,
        {"enable_thinking": False},
        "",
    ),
    NEMOTRON_ULTRA: (
        NEMOTRON_ULTRA_REVISION,
        "LOOPS_NEMOTRON_TOKENIZER_CACHE",
        None,
        {},
        "</think>\n",
    ),
    NEMOTRON_SUPER_VL: (
        NEMOTRON_SUPER_VL_REVISIONS[NEMOTRON_SUPER_VL],
        "LOOPS_NEMOTRON_TOKENIZER_CACHE",
        None,
        {},
        "</think>\n",
    ),
}


@pytest.fixture(scope="module", params=sorted(XML_MODELS))
def xml_model(request):
    """Return the tokenizer, renderer and reply prefix for each XML-format model.

    The models are Qwen3.5 through the cookbook's extension, and both Nemotron 3
    extensions as the cookbook selects them.
    """
    revision, variable, setting, options, prefix = XML_MODELS[request.param]
    tokenizer = pinned_tokenizer(request.param, revision, variable)
    renderer = create_renderer(
        request.param, tokenizer=tokenizer, renderer=setting, chat_template_kwargs=options
    )
    return tokenizer, renderer, prefix


@pytest.mark.parametrize("seed", range(300))
def test_xml_random_calls_parse_back_to_themselves(xml_model, seed):
    """Parse random calls back, including booleans and null written as ``True`` and ``None``."""
    tokenizer, renderer, prefix = xml_model
    arguments = random_arguments(random.Random(seed), XML_MARKUP, python_null=True)
    result, reply = parsed_back(renderer, tokenizer, arguments, "Act.", "<|im_end|>", prefix)
    assert result.status.value == "ok"
    assert same(result.arguments, arguments), (result.arguments, arguments)
    assert reply.native_message is not None
    assert same(reply.native_message["tool_calls"][0]["function"]["arguments"], arguments)


async def test_xml_echoed_call_continues_from_its_sampled_tokens(xml_model):
    tokenizer, renderer, prefix = xml_model
    written = {"i": "3", "oi": "None"}  # as these templates write a non-string
    body = "".join(
        f"<parameter={key}>\n{written.get(key, value)}\n</parameter>\n"
        for key, value in EDIT.items()
    )
    await _echo_continues(
        tokenizer,
        renderer,
        f"<tool_call>\n<function=act>\n{body}</function>\n</tool_call>",
        prefix,
        "<|im_end|>",
    )
