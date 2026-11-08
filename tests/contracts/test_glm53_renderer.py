"""GLM-5.3 renderer extension: HF template parity and sampled-token continuation."""

import random
from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from support.glm53_histories import ASSISTANT, BASE, HISTORIES, TOOLS

from loops_cookbook.llm.client import Renderers
from loops_cookbook.llm.factory import create_renderer
from loops_cookbook.llm.renderer_extensions.glm53 import GLM53Renderer, GLM53RendererConfig

GLM53 = "loops_cookbook.llm.renderer_extensions.glm53:renderer"


@pytest.mark.parametrize("effort", ["low", "high", "max"])
@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("history", HISTORIES)
@pytest.mark.parametrize("tools", [None, TOOLS, [TOOLS[1]]])
def test_hf_parity(glm_tokenizer, effort, clear, history, tools):
    renderer = GLM53Renderer(
        glm_tokenizer, GLM53RendererConfig(reasoning_effort=effort, clear_thinking=clear)
    )
    original = deepcopy(history)
    if tools is None and history == HISTORIES[-1]:
        from jinja2 import UndefinedError

        with pytest.raises(UndefinedError):
            glm_tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        with pytest.raises(ValueError, match="Tool references"):
            renderer.render(history, add_generation_prompt=True)
        return
    text = glm_tokenizer.apply_chat_template(
        history,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=effort,
        clear_thinking=clear,
    )
    expected = glm_tokenizer.encode(text, add_special_tokens=False)
    result = renderer.render(history, tools=tools, add_generation_prompt=True)
    assert result.token_ids == expected
    assert history == original
    assert len(result.message_indices) == len(expected)
    assert len(result.sampled_mask) == len(expected)


@pytest.mark.parametrize(
    "kwargs", [{"reasoning_effort": "medium"}, {"enable_thinking": False}, {"typo": True}]
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValidationError):
        GLM53RendererConfig(**kwargs)


@pytest.mark.parametrize("history", [HISTORIES[4], HISTORIES[8], HISTORIES[9]])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_native_tool_context_through_model_client(glm_tokenizer, history, wrapped):
    from support.leases import lease

    from loops_cookbook.model_access.lease import DirectModelAccess

    tools = [{"type": "function", "function": tool} for tool in TOOLS] if wrapped else TOOLS
    expected = glm_tokenizer.encode(
        glm_tokenizer.apply_chat_template(
            history,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        ),
        add_special_tokens=False,
    )

    class Sampler:
        async def sample_async(self, prompt, **kwargs):
            assert prompt.to_ints() == expected
            tokens = glm_tokenizer.encode("done</think>answer<|user|>", add_special_tokens=False)
            return SimpleNamespace(
                prompt_token_ids=expected,
                policy_version=0,
                sequences=[
                    SimpleNamespace(
                        tokens=tokens,
                        logprobs=[-0.1] * len(tokens),
                        sequence_id=None,
                        stop_reason="stop",
                    )
                ],
            )

    original = deepcopy(history)
    backend = lease(Sampler())
    client = DirectModelAccess(
        backend,
        Renderers(create_renderer("zai-org/GLM-5.3", renderer=GLM53, tokenizer=glm_tokenizer)),
    ).llm()
    response = await client.conversation().generate(messages=history, tools=tools, max_tokens=20)
    assert response.sample.prompt_ids == tuple(expected)
    assert response.parse_error is None
    assert history == original


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("clear", [False, True])
def test_long_randomized_histories_match_hf(glm_tokenizer, seed, clear):
    rng = random.Random(seed)
    effort = ("low", "high", "max")[seed % 3]
    renderer = GLM53Renderer(
        glm_tokenizer, GLM53RendererConfig(reasoning_effort=effort, clear_thinking=clear)
    )
    history = deepcopy(BASE)
    tools = deepcopy(TOOLS)
    for turn in range(40):
        calls = [
            {
                "id": f"{turn}-{i}",
                "function": {
                    "name": "f",
                    "arguments": {
                        "value": rng.choice(
                            [None, True, 42, "中文 café 🦀", {"z": 1, "a": [2, 3]}]
                        ),
                    },
                },
            }
            for i in range(rng.randint(1, 4))
        ]
        history.append(
            {
                "role": "assistant",
                "content": rng.choice(["", " answer \n"]),
                "reasoning_content": rng.choice(["", " plan\n", "\n 考虑 🦀 \t"]),
                "tool_calls": calls,
            }
        )
        observations = [
            {"role": "tool", "tool_call_id": call["id"], "content": f"result {i}\n"}
            for i, call in enumerate(calls)
        ]
        rng.shuffle(observations)
        history.extend(observations)
        if turn % 7 == 0:
            history.append({"role": "user", "content": f"next {turn}"})
        if turn % 9 == 0:
            tools[0]["description"] = f"tool revision {turn}"
        if turn == 23:
            history = deepcopy(BASE) + [{"role": "user", "content": "compacted context"}]
        original = deepcopy(history)
        expected = glm_tokenizer.apply_chat_template(
            history,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=effort,
            clear_thinking=clear,
        )
        rendered = renderer.render(history, tools=tools, add_generation_prompt=True)
        assert rendered.token_ids == glm_tokenizer.encode(expected, add_special_tokens=False), (
            seed,
            turn,
        )
        assert history == original
        assert len(rendered.message_indices) == len(rendered.token_ids)


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_bridge_keeps_sampled_whitespace_and_tool_bytes(glm_tokenizer, effort):
    renderer = create_renderer(
        "zai-org/GLM-5.3",
        renderer=GLM53,
        tokenizer=glm_tokenizer,
        chat_template_kwargs={"reasoning_effort": effort},
    )
    rendered = renderer.render(BASE, TOOLS)
    prompt = rendered.ids
    completion = tuple(
        glm_tokenizer.encode(
            "  plan\n</think>  answer  <tool_call>f<arg_key>value</arg_key><arg_value>1</arg_value></tool_call><|observation|>",
            add_special_tokens=False,
        )
    )
    parsed = renderer.parse(prompt, completion, TOOLS)
    assert parsed.tool_calls[0].name == "f"
    observation = [{"role": "tool", "content": "  result\n"}]
    bridged = renderer.bridge(
        prompt, completion, observation, TOOLS, previous_prompt_input=rendered.prompt_input
    ).ids
    assert bridged[: len(prompt + completion)] == prompt + completion
    assert (
        glm_tokenizer.decode(bridged[len(prompt + completion) :])
        == "<tool_response>  result\n</tool_response><|assistant|><think>"
    )
    full = renderer.render(BASE + [parsed.native_message] + observation, TOOLS).ids
    assert full != bridged  # Full-template whitespace normalization is not used on continuation.


def test_retention_change_requires_full_history(glm_tokenizer):
    renderer = create_renderer(
        "zai-org/GLM-5.3",
        renderer=GLM53,
        tokenizer=glm_tokenizer,
        chat_template_kwargs={"clear_thinking": True},
    )
    rendered = renderer.render(BASE)
    completion = tuple(glm_tokenizer.encode("plan</think>answer<|user|>", add_special_tokens=False))
    assert (
        renderer.bridge(
            rendered.ids,
            completion,
            [{"role": "user", "content": "Next"}],
            previous_prompt_input=rendered.prompt_input,
        )
        is None
    )


@pytest.mark.parametrize("stop", ["<|user|>", "<|observation|>", "<|endoftext|>"])
def test_inherited_prime_parser_and_stops(glm_tokenizer, stop):
    renderer = GLM53Renderer(glm_tokenizer)
    prompt = renderer.render(BASE, add_generation_prompt=True).token_ids
    completion = glm_tokenizer.encode(
        "plan</think><tool_call>f<arg_key>value</arg_key><arg_value>42</arg_value></tool_call>"
        + stop,
        add_special_tokens=False,
    )
    parsed = renderer.parse_response(
        completion,
        prompt_ids=prompt,
        tools=[
            {
                "name": "f",
                "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}},
            }
        ],
    )
    assert parsed.reasoning_content == "plan"
    assert parsed.tool_calls[0].arguments == {"value": 42}
    assert parsed.tool_calls[0].status.value == "ok"
    assert glm_tokenizer.convert_tokens_to_ids(stop) in renderer.get_stop_token_ids()


def test_inherited_parser_reports_truncated_reasoning_and_tool_calls(glm_tokenizer):
    renderer = GLM53Renderer(glm_tokenizer)
    prompt = renderer.render(BASE, add_generation_prompt=True).token_ids
    reasoning = renderer.parse_response(
        glm_tokenizer.encode("unfinished", add_special_tokens=False), prompt_ids=prompt
    )
    assert reasoning.reasoning_complete is False
    action = renderer.parse_response(
        glm_tokenizer.encode("plan</think><tool_call>f<arg_key>x", add_special_tokens=False),
        prompt_ids=prompt,
    )
    assert action.tool_calls[0].status.value == "unclosed_block"


EDIT_TOOL = {
    "name": "edit",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old": {"type": "string"},
            "new": {"type": "string"},
            "line": {"type": "integer"},
        },
    },
}
# A file edit needs each of these back byte for byte. They cover a final newline, a first
# line's indentation, a value of only whitespace, an empty value, inner and trailing
# spaces, tabs, and text outside ASCII.
EXACT_VALUES = [
    "def f():\n    return 1\n",
    "    indented()\n",
    "\n",
    "",
    " a  b  ",
    "\tcol\t\n",
    "  naïve café — ✓\n",
]


def _call(arguments: dict[str, str]) -> str:
    body = "".join(
        f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
        for key, value in arguments.items()
    )
    return f"<tool_call>edit{body}</tool_call>"


def _completion(tokenizer, *calls: str, after_stop: str = "") -> list[int]:
    text = "plan</think>" + "".join(calls) + "<|observation|>" + after_stop
    return tokenizer.encode(text, add_special_tokens=False)


@pytest.mark.parametrize("value", EXACT_VALUES)
def test_string_arguments_parse_exactly_as_sampled(glm_tokenizer, value):
    renderer = GLM53Renderer(glm_tokenizer)
    arguments = {"path": " a.py", "old": value, "new": value}
    [call] = renderer.parse_response(
        _completion(glm_tokenizer, _call(arguments)), tools=[EDIT_TOOL]
    ).tool_calls
    assert call.status.value == "ok"
    assert call.name == "edit"
    assert call.arguments == arguments
    assert all(type(v) is str for v in call.arguments.values())


WEATHER_TOOL = {
    "name": "get_weather",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "when": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
            "note": {"description": "a schema that types nothing"},
            "range": {"type": "array", "items": {"type": "integer"}},
        },
    },
}


@pytest.mark.parametrize(
    "arguments,expected",
    [
        # A string is the text as written. Another declared type is read as JSON.
        ({"city": "  Paris \n", "days": " 2 "}, {"city": "  Paris \n", "days": 2}),
        # A union tries each type. Text that is not an integer stays a string.
        ({"when": " 3 "}, {"when": 3}),
        ({"when": "  next week\n"}, {"when": "  next week\n"}),
        # An untyped argument, declared or not, may have been a string or a number, as
        # "600" shows. It stays the text as written and is never decoded or refused.
        ({"timeout": "600"}, {"timeout": "600"}),
        ({"timeout": " 30s\n"}, {"timeout": " 30s\n"}),
        ({"timeout": '"abc"'}, {"timeout": '"abc"'}),
        ({"note": " 5 "}, {"note": " 5 "}),
        # A typed argument reads null as None. An untyped or string argument keeps the text.
        ({"days": " null "}, {"days": None}),
        ({"timeout": " null "}, {"timeout": " null "}),
        ({"city": "null"}, {"city": "null"}),
        ({"when": "NULL"}, {"when": "NULL"}),
        # When the declared type cannot read the text, the text reaches the tool to refuse.
        ({"range": " [1, 20]\n"}, {"range": [1, 20]}),
        ({"range": "1, 20"}, {"range": "1, 20"}),
        ({"days": " abc "}, {"days": " abc "}),
        ({"days": "None"}, {"days": "None"}),
    ],
)
def test_values_parse_back_as_the_template_wrote_them(glm_tokenizer, arguments, expected):
    body = "".join(
        f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
        for key, value in arguments.items()
    )
    completion = glm_tokenizer.encode(
        f"plan</think><tool_call>get_weather{body}</tool_call><|observation|>",
        add_special_tokens=False,
    )
    [call] = (
        GLM53Renderer(glm_tokenizer).parse_response(completion, tools=[WEATHER_TOOL]).tool_calls
    )
    assert call.status.value == "ok"
    assert call.arguments == expected
    assert {k: type(v) for k, v in call.arguments.items()} == {
        k: type(v) for k, v in expected.items()
    }


def test_without_tools_every_value_is_its_text(glm_tokenizer):
    """Keep every value as its text when no tools are given, where Prime refuses the call.

    Without schemas, nothing says which values were strings.
    """
    arguments = {"path": " a.py", "new": "x\n", "count": "3", "flags": "[1, 2]"}
    completion = _completion(glm_tokenizer, _call(arguments))
    [call] = GLM53Renderer(glm_tokenizer).parse_response(completion).tool_calls
    assert call.status.value == "ok"
    assert call.arguments == arguments


def test_typed_arguments_keep_their_parsed_value(glm_tokenizer):
    renderer = GLM53Renderer(glm_tokenizer)
    completion = _completion(glm_tokenizer, _call({"line": " 42\n", "new": "x\n"}))
    [call] = renderer.parse_response(completion, tools=[EDIT_TOOL]).tool_calls
    assert call.arguments == {"line": 42, "new": "x\n"}
    assert type(call.arguments["line"]) is int


def test_parallel_calls_only_change_ok_string_values(glm_tokenizer):
    """Give each OK call its own values back and leave everything else as Prime parsed it.

    That includes a malformed call and the reply's text.
    """
    from renderers.glm5 import GLM51Renderer

    first = {"path": "a.py", "new": "one\n"}
    third = {"path": "b.py", "new": "    three\n"}
    malformed = "<tool_call>edit<arg_key>new</arg_key></tool_call>"
    completion = _completion(
        glm_tokenizer,
        _call(first),
        malformed,
        _call(third),
        after_stop="<tool_call>edit<arg_key>new</arg_key><arg_value> x </arg_value></tool_call>",
    )
    ours = GLM53Renderer(glm_tokenizer).parse_response(completion, tools=[EDIT_TOOL])
    prime = GLM51Renderer(glm_tokenizer).parse_response(completion, tools=[EDIT_TOOL])
    assert [c.status.value for c in ours.tool_calls] == ["ok", "malformed_structure", "ok"]
    assert [c.arguments for c in ours.tool_calls] == [first, prime.tool_calls[1].arguments, third]
    assert ours.tool_calls[1] == prime.tool_calls[1]
    for mine, theirs in zip(ours.tool_calls, prime.tool_calls, strict=True):
        assert (mine.raw, mine.name, mine.token_span, mine.status, mine.id) == (
            theirs.raw,
            theirs.name,
            theirs.token_span,
            theirs.status,
            theirs.id,
        )
    assert (ours.content, ours.reasoning_content, ours.reasoning_complete) == (
        prime.content,
        prime.reasoning_content,
        prime.reasoning_complete,
    )


def test_a_rendered_call_parses_back_to_its_arguments(glm_tokenizer):
    renderer = GLM53Renderer(glm_tokenizer)
    arguments = dict(zip(["path", "old", "new"], EXACT_VALUES, strict=False))
    message = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "plan",
        "tool_calls": [{"function": {"name": "edit", "arguments": arguments}}],
    }
    prompt = renderer.render(BASE, tools=[EDIT_TOOL], add_generation_prompt=True).token_ids
    full = renderer.render([*BASE, message], tools=[EDIT_TOOL]).token_ids
    assert full[: len(prompt)] == prompt
    [call] = renderer.parse_response(full[len(prompt) :], tools=[EDIT_TOOL]).tool_calls
    assert call.arguments == arguments


@pytest.mark.parametrize(
    "arguments",
    [
        {"city": "  Paris \n", "days": 3, "range": [1, 20], "when": "next week"},
        {"city": "", "days": None, "range": [], "when": 3},
        {"city": '"quoted"', "when": "  later\n"},
    ],
)
def test_every_typed_value_the_template_writes_parses_back(glm_tokenizer, arguments):
    """Parse a call rendered by the chat template back to the same arguments and types.

    A value the template writes the same way for two types cannot round-trip. Examples
    are an untyped ``5`` or ``"5"``, and ``"7"`` in an integer-or-string union, which reads
    back as the integer.
    """
    renderer = GLM53Renderer(glm_tokenizer)
    message = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "plan",
        "tool_calls": [{"function": {"name": "get_weather", "arguments": arguments}}],
    }
    tools = [WEATHER_TOOL]
    prompt = renderer.render(BASE, tools=tools, add_generation_prompt=True).token_ids
    full = renderer.render([*BASE, message], tools=tools).token_ids
    [call] = renderer.parse_response(full[len(prompt) :], tools=tools).tool_calls
    assert call.status.value == "ok"
    assert call.arguments == arguments
    assert {k: type(v) for k, v in call.arguments.items()} == {
        k: type(v) for k, v in arguments.items()
    }


def test_the_cookbook_client_gets_exact_values(glm_tokenizer):
    """Give exact values in the reply and its history message from the cookbook's renderer."""
    renderer = create_renderer("zai-org/GLM-5.3", renderer=GLM53, tokenizer=glm_tokenizer)
    arguments = {"path": "a.py", "new": "    body()\n"}
    prompt = renderer.render(BASE, [EDIT_TOOL]).ids
    reply = renderer.parse(prompt, _completion(glm_tokenizer, _call(arguments)), [EDIT_TOOL])
    [call] = reply.tool_calls
    assert call.arguments == arguments
    assert reply.native_message is not None
    assert reply.native_message["tool_calls"][0]["function"]["arguments"] == arguments


def test_prime_still_strips_argument_values(glm_tokenizer):
    """Check that Prime still strips argument values.

    When this fails, Prime keeps string values as the template wrote them, so revisit
    ``GLM53Renderer.parse_response``.
    """
    from renderers.glm5 import GLM51Renderer

    completion = _completion(glm_tokenizer, _call({"new": "  x\n"}))
    [call] = GLM51Renderer(glm_tokenizer).parse_response(completion, tools=[EDIT_TOOL]).tool_calls
    assert call.arguments == {"new": "x"}


@pytest.mark.parametrize(
    "ids,expected",
    [
        (("b", "a"), ("ONE", "TWO")),
        (("a", "a"), ("TWO", "ONE")),
        ((None, "a"), ("TWO", "ONE")),
        (("unknown", "a"), ("TWO", "ONE")),
    ],
)
def test_id_ordering_preserves_sampled_prefix(glm_tokenizer, ids, expected):
    renderer = create_renderer("zai-org/GLM-5.3", renderer=GLM53, tokenizer=glm_tokenizer)
    rendered = renderer.render(BASE, TOOLS)
    prompt = rendered.ids
    completion = glm_tokenizer.encode(
        " plan \n</think> answer  <|observation|>", add_special_tokens=False
    )
    history = BASE + [ASSISTANT]
    observations = [
        {"role": "tool", "tool_call_id": key, "content": value}
        for key, value in zip(ids, ("TWO", "ONE"), strict=True)
    ]
    original = deepcopy((history, observations))
    bridged = renderer.bridge(
        prompt,
        completion,
        observations,
        TOOLS,
        previous_messages=history,
        previous_prompt_input=rendered.prompt_input,
    )
    prefix = tuple(prompt) + tuple(completion)
    bridged = bridged.ids
    assert bridged[: len(prefix)] == prefix
    suffix = glm_tokenizer.decode(bridged[len(prefix) :])
    assert suffix.index(expected[0]) < suffix.index(expected[1])
    assert (history, observations) == original


async def test_tito_packing_and_context_edit(glm_tokenizer, monkeypatch):
    from support.builders import make_rollout
    from support.leases import lease

    from loops_cookbook.model_access.lease import DirectModelAccess
    from loops_cookbook.rl.grpo import build_rows

    class Sampler:
        async def sample_async(self, prompt, **kwargs):
            return SimpleNamespace(
                prompt_token_ids=prompt.to_ints(),
                policy_version=0,
                sequences=[
                    SimpleNamespace(
                        tokens=glm_tokenizer.encode(
                            " plan </think> answer <|observation|>", add_special_tokens=False
                        ),
                        logprobs=[-0.25]
                        * len(
                            glm_tokenizer.encode(
                                " plan </think> answer <|observation|>", add_special_tokens=False
                            )
                        ),
                        sequence_id=None,
                        stop_reason="stop",
                    )
                ],
            )

    backend = lease(Sampler())
    renderer = create_renderer("zai-org/GLM-5.3", renderer=GLM53, tokenizer=glm_tokenizer)
    render_calls = []
    render = renderer.render

    def counted_render(messages, tools):
        render_calls.append(deepcopy(messages))
        return render(messages, tools)

    monkeypatch.setattr(renderer, "render", counted_render)
    conversation = DirectModelAccess(backend, Renderers(renderer)).llm().conversation()
    history = deepcopy(BASE)
    first = await conversation.generate(messages=history, tools=TOOLS, max_tokens=100)
    history += [first.native_message, {"role": "tool", "tool_call_id": "a", "content": "result"}]
    second = await conversation.generate(messages=history, tools=TOOLS, max_tokens=100)
    assert len(render_calls) == 1
    assert len(build_rows(make_rollout("t", "g", "r", backend.captured(), reward=1), 1)) == 1
    assert (
        second.sample.prompt_ids[: len(first.sample.prompt_ids + first.sample.completion_ids)]
        == first.sample.prompt_ids + first.sample.completion_ids
    )
    history += [second.native_message, {"role": "tool", "content": "next"}]
    await conversation.generate(
        messages=history, tools=[{"name": "new", "parameters": {}}], max_tokens=100
    )
    rows = build_rows(make_rollout("t", "g", "r", backend.captured(), reward=1), 1)
    assert len(rows) == 2
    assert sum(row.trained_positions for row in rows) == sum(
        len(s.completion_ids) for s in backend.captured()
    )
    for sample in backend.captured():
        matching = []
        for row in rows:
            targets = row.datum.loss_fn_inputs["target_tokens"].data
            sequence = row.datum.model_input.to_ints() + [targets[-1]]
            expected = list(sample.prompt_ids + sample.completion_ids)
            start = len(sample.prompt_ids) - 1
            end = start + len(sample.completion_ids)
            if sequence[: len(expected)] == expected and (start, end) in row.actions:
                matching.append(row)
                assert targets[start:end] == list(sample.completion_ids)
                assert row.datum.loss_fn_inputs["logprobs"].data[start:end] == list(sample.logprobs)
        assert len(matching) == 1


@pytest.mark.parametrize("clear", [False, True])
async def test_long_token_capture_survives_context_changes(glm_tokenizer, clear):
    from support.builders import make_rollout
    from support.leases import lease

    from loops_cookbook.model_access.lease import DirectModelAccess
    from loops_cookbook.rl.grpo import build_rows

    class Sampler:
        async def sample_async(self, prompt, **kwargs):
            turn = len(backend.captured())
            tokens = glm_tokenizer.encode(
                f"  thought {turn}\n</think>  action {turn}  "
                "<tool_call>f<arg_key>value</arg_key><arg_value>42</arg_value>"
                "</tool_call><|observation|>",
                add_special_tokens=False,
            )
            return SimpleNamespace(
                prompt_token_ids=prompt.to_ints(),
                policy_version=0,
                sequences=[
                    SimpleNamespace(
                        tokens=tokens,
                        logprobs=[-0.25] * len(tokens),
                        sequence_id=None,
                        stop_reason="stop",
                    )
                ],
            )

    backend = lease(Sampler())
    renderer = create_renderer(
        "zai-org/GLM-5.3",
        renderer=GLM53,
        tokenizer=glm_tokenizer,
        chat_template_kwargs={"clear_thinking": clear},
    )
    chat = (
        DirectModelAccess(backend, Renderers(renderer))
        .llm()
        .conversation(conversation_id="stable-session")
    )
    history, tools = deepcopy(BASE), deepcopy(TOOLS)
    expected_chains = 1
    previous = None
    for turn in range(60):
        if turn and turn % 13 == 0:
            tools[0]["description"] = f"new schema {turn}"
        if turn and turn % 17 == 0:
            history = deepcopy(BASE) + [{"role": "user", "content": f"summary {turn}"}]
        elif turn and turn % 11 == 0:
            history.append({"role": "user", "content": f"next question {turn}"})
        response = await chat.generate(messages=history, tools=tools, max_tokens=100)
        assert response.sample.conversation_id == chat.conversation_id == "stable-session"
        if previous is not None:
            prefix = previous.prompt_ids + previous.completion_ids
            if turn < 11:
                assert response.sample.prompt_ids[: len(prefix)] == prefix
            if response.sample.prompt_ids[: len(prefix)] != prefix:
                expected_chains += 1
        previous = response.sample
        assert response.native_message is not None
        history += [response.native_message, {"role": "tool", "content": f"output {turn}"}]

    rows = build_rows(make_rollout("t", "g", "r", backend.captured(), reward=1), 1)
    assert len(rows) == expected_chains
    assert expected_chains > 1
    assert sum(row.trained_positions for row in rows) == sum(
        len(sample.completion_ids) for sample in backend.captured()
    )
    for sample in backend.captured():
        hits = 0
        expected = list(sample.prompt_ids + sample.completion_ids)
        start = len(sample.prompt_ids) - 1
        end = start + len(sample.completion_ids)
        for row in rows:
            targets = row.datum.loss_fn_inputs["target_tokens"].data
            sequence = row.datum.model_input.to_ints() + [targets[-1]]
            if sequence[: len(expected)] == expected and (start, end) in row.actions:
                assert targets[start:end] == list(sample.completion_ids)
                assert all(row.datum.loss_fn_inputs["advantages"].data[start:end])
                assert row.datum.loss_fn_inputs["logprobs"].data[start:end] == list(sample.logprobs)
                hits += 1
        assert hits == 1


@pytest.mark.parametrize("effort", ["low", "max"])
def test_tool_reference_inside_bundled_output_matches_hf(glm_tokenizer, effort):
    """The bundled tool shape may carry a tool_reference in an output entry; the HF
    template expands it to the deferred tool's spec, and so must we."""
    renderer = GLM53Renderer(glm_tokenizer, GLM53RendererConfig(reasoning_effort=effort))
    history = BASE + [
        ASSISTANT,
        {
            "role": "tool",
            "content": [
                {"tool_call_id": "a", "output": [{"type": "tool_reference", "name": "hidden"}]},
                {"tool_call_id": "b", "output": "TWO"},
            ],
        },
    ]
    expected = glm_tokenizer.apply_chat_template(
        history, tools=TOOLS, tokenize=False, add_generation_prompt=True, reasoning_effort=effort
    )
    rendered = renderer.render(history, tools=TOOLS, add_generation_prompt=True)
    assert rendered.token_ids == glm_tokenizer.encode(expected, add_special_tokens=False)
    # And through the cookbook adapter, which validates content before rendering.
    adapter = create_renderer(
        "zai-org/GLM-5.3",
        renderer=GLM53,
        tokenizer=glm_tokenizer,
        chat_template_kwargs={"reasoning_effort": effort},
    )
    assert list(adapter.render(history, TOOLS).ids) == rendered.token_ids
