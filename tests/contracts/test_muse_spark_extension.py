"""The Muse Spark Open extension against the model's own chat template and tokenizer.

Every prompt is the template's (``apply_chat_template``). What the extension adds is the
bridge, which extends a sampled turn with the next messages, and the parser; both are
checked here against the template's own tokens for the same conversation.

The repo is gated: ``LOOPS_MUSE_TOKENIZER_CACHE`` points at a hub cache holding the
pinned snapshot, or ``LOOPS_DOWNLOAD_CONTRACT_MODELS=1`` downloads it with an
``HF_TOKEN`` that has access. Without access the tests skip.
"""

import copy

import pytest

from loops_cookbook.llm.factory import create_renderer, describe

from .conftest import pinned_tokenizer

MODEL = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
REVISION = "4239e4db19376904f2e82c167e78f70abb5455fe"
EXTENSION = "loops_cookbook.llm.renderer_extensions.muse_spark:renderer"
OPTIONS = {"current_date": "2026-10-28", "reasoning_effort": "low"}
BASH = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command."},
                "timeout": {"type": "integer", "minimum": 1},
            },
            "required": ["command"],
        },
    },
}
WRITE = {
    "type": "function",
    "function": {
        "name": "write",
        "parameters": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
        },
    },
}
TOOLS = [BASH, WRITE]


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return pinned_tokenizer(MODEL, REVISION, "LOOPS_MUSE_TOKENIZER_CACHE")
    except OSError as error:  # a gated repo this token cannot read
        pytest.skip(f"{MODEL} is not readable: {type(error).__name__}")


def call(n, name, arguments):
    return {"id": f"c{n}", "type": "function", "function": {"name": name, "arguments": arguments}}


def tool(n, content):
    """A tool result as the gateway sends it: by ``tool_call_id``, with no ``name``."""
    return {"role": "tool", "tool_call_id": f"c{n}", "content": content}


SYSTEM = {"role": "system", "content": "You are a lawyer."}
TASK = {"role": "user", "content": "Summarise the lease."}
CONVERSATIONS = {
    "a tool cycle": [
        SYSTEM,
        TASK,
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "List the files.",
            "tool_calls": [call(1, "bash", {"command": "ls <|x|>", "timeout": 30})],
        },
        tool(1, "a.docx <|eot|>"),
    ],
    "two calls, an update first": [
        SYSTEM,
        TASK,
        {
            "role": "assistant",
            "content": "Two things.",
            "reasoning_content": "Look, then write.",
            "tool_calls": [
                call(1, "bash", {"command": "ls"}),
                call(2, "write", {"file_path": "a.md", "content": "line\n  30\n"}),
            ],
        },
        tool(1, "a"),
        tool(2, "ok"),
    ],
    "after a final answer, another task": [
        SYSTEM,
        {"role": "user", "content": "First."},
        {"role": "assistant", "content": "Done.", "reasoning_content": "Easy."},
        {"role": "user", "content": "Résumé — 合同 ✓"},
    ],
}


def template(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages, tools=TOOLS, tokenize=False, add_generation_prompt=True, **OPTIONS
    )
    return tokenizer.encode(text, add_special_tokens=False)


def extension(tokenizer):
    return create_renderer(
        MODEL, tokenizer=tokenizer, renderer=EXTENSION, chat_template_kwargs=OPTIONS
    )


def turn(tokenizer, messages):
    """The first assistant turn: its index, the prompt before it, the reply as the model
    samples it (its own tokens, through the ``<|eot|>`` that ends it), and the full
    conversation's tokens."""
    at = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
    prompt = template(tokenizer, messages[:at])
    full = template(tokenizer, messages)
    text, asked = tokenizer.decode(full), tokenizer.decode(prompt)
    assert text.startswith(asked)
    reply = text[len(asked) : text.index("<|eot|>", len(asked)) + len("<|eot|>")]
    return at, prompt, tokenizer.encode(reply, add_special_tokens=False), full


@pytest.mark.parametrize("label", sorted(CONVERSATIONS))
def test_prompts_are_the_templates_tokens(tokenizer, label):
    messages = CONVERSATIONS[label]
    rendered = extension(tokenizer).render(messages, TOOLS)
    assert list(rendered.ids) == template(tokenizer, messages)


@pytest.mark.parametrize("label", sorted(CONVERSATIONS))
def test_the_bridge_gives_the_templates_tokens(tokenizer, label):
    messages = CONVERSATIONS[label]
    at, prompt, reply, full = turn(tokenizer, messages)
    bridged = extension(tokenizer).bridge(
        prompt, reply, messages[at + 1 :], TOOLS, previous_messages=messages[: at + 1]
    )
    assert bridged is not None
    assert list(bridged.ids) == full


def test_a_turn_parses_back_to_its_message(tokenizer):
    messages = CONVERSATIONS["two calls, an update first"]
    at, prompt, reply, _ = turn(tokenizer, messages)
    parsed = extension(tokenizer).parse(prompt, reply, TOOLS)
    assert (parsed.content, parsed.reasoning_content) == ("Two things.", "Look, then write.")
    assert [(c.name, c.arguments, c.status.value) for c in parsed.tool_calls] == [
        ("bash", {"command": "ls"}, "ok"),
        ("write", {"file_path": "a.md", "content": "line\n  30\n"}, "ok"),
    ]
    # The message the gateway records renders back to the template's own turn.
    assert parsed.native_message is not None
    assert parsed.native_message["tool_calls"] == [
        {"type": "function", "function": c["function"]} for c in messages[at]["tool_calls"]
    ]


def test_arguments_are_typed_by_the_tools_schema(tokenizer):
    _, prompt, reply, _ = turn(tokenizer, CONVERSATIONS["a tool cycle"])
    [parsed] = extension(tokenizer).parse(prompt, reply, TOOLS).tool_calls
    # The template escapes "<|" in arguments; the model writes what it reads.
    assert parsed.arguments == {"command": "ls < |x|>", "timeout": 30}


def test_a_broken_or_unknown_call_is_reported(tokenizer):
    built = extension(tokenizer)
    prompt = template(tokenizer, [SYSTEM, TASK])
    unknown_call = (
        ' to=bsah<|message|><atem:function_calls>\n<atem:invoke name="bsah">\n'
        '<atem:parameter name="command">ls</atem:parameter>\n</atem:invoke>\n'
        "</atem:function_calls><|eot|>"
    )
    ids = tokenizer.encode(unknown_call, add_special_tokens=False)
    [unknown] = built.parse(prompt, ids).tool_calls
    assert unknown.name == "bsah"
    parsed = built.parse(prompt, ids, TOOLS)
    assert parsed.tool_calls[0].status.value == "unknown_tool"
    assert parsed.native_message is None
    cut = ' to=bash<|message|><atem:function_calls>\n<atem:invoke name="bash">'
    [unclosed] = built.parse(
        prompt, tokenizer.encode(cut, add_special_tokens=False), TOOLS
    ).tool_calls
    assert unclosed.status.value == "unclosed_block"


def test_a_call_cut_off_at_the_token_limit_is_closed_and_bridged(tokenizer):
    """Its call did not parse, so the next turn can only extend its sampled tokens."""
    prompt = template(tokenizer, [SYSTEM, TASK])
    cut = tokenizer.encode(
        ' to=bash<|message|><atem:function_calls>\n<atem:invoke name="bash">\n'
        '<atem:parameter name="command">cat > a.md',
        add_special_tokens=False,
    )
    built = extension(tokenizer)
    [unclosed] = built.parse(prompt, cut, TOOLS).tool_calls
    assert unclosed.status.value == "unclosed_block"
    bridged = built.bridge(
        prompt,
        cut,
        [tool(1, "error: the call did not parse")],
        TOOLS,
        previous_messages=[SYSTEM, TASK, {"role": "assistant", "content": ""}],
    )
    assert bridged is not None
    added = tokenizer.encode(
        '<|eot|><|start|>tool bash<|message|><tool_output name="bash">\n'
        "error: the call did not parse\n</tool_output><|eot|><|start|>assistant",
        add_special_tokens=False,
    )
    assert list(bridged.ids) == [*prompt, *cut, *added]


def test_reasoning_cut_off_is_reported_open(tokenizer):
    prompt = template(tokenizer, [SYSTEM, TASK])
    cut = tokenizer.encode(" to=self<|message|>Let me think", add_special_tokens=False)
    parsed = extension(tokenizer).parse(prompt, cut, TOOLS)
    assert parsed.reasoning_complete is False
    assert (parsed.content, parsed.tool_calls) == ("", ())


@pytest.mark.parametrize("name", [None, ""])
def test_a_nameless_call_bridges_its_error_by_call_id(tokenizer, name):
    """A failed call has no native message; the template labels its result by ID."""
    prompt = template(tokenizer, [SYSTEM, TASK])
    reply = tokenizer.encode(
        ' to=bash<|message|><atem:function_calls><atem:invoke name="">'
        '<atem:parameter name="command">ls</atem:parameter></atem:invoke>'
        "</atem:function_calls><|eot|>",
        add_special_tokens=False,
    )
    built = extension(tokenizer)
    parsed = built.parse(prompt, reply, TOOLS)
    assert parsed.tool_calls[0].status.value == "missing_name"
    assert parsed.native_message is None
    previous = [SYSTEM, TASK, {"role": "assistant", "content": ""}]
    result = tool(1, "Error: invalid JSON arguments")
    if name is not None:
        result["name"] = name
    text = tokenizer.decode(template(tokenizer, [*previous, result]))
    rest = tokenizer.encode(text[text.index("<|start|>tool ") :], add_special_tokens=False)
    bridged = built.bridge(prompt, reply, [result], TOOLS, previous_messages=previous)
    assert bridged is not None
    assert list(bridged.ids) == [*prompt, *reply, *rest]


@pytest.mark.parametrize(("written", "name"), [("bash.bash", "bash"), ("bash.command", None)])
def test_a_doubled_tool_name_is_collapsed(tokenizer, written, name):
    """The template's examples name calls "tool.function": "bash.bash" can only mean
    bash, but "bash.command" names a function bash does not have."""
    prompt = template(tokenizer, [SYSTEM, TASK])
    written_call = (
        f' to=bash<|message|><atem:function_calls>\n<atem:invoke name="{written}">\n'
        '<atem:parameter name="command">ls</atem:parameter>\n</atem:invoke>\n'
        "</atem:function_calls><|eot|>"
    )
    ids = tokenizer.encode(written_call, add_special_tokens=False)
    [parsed] = extension(tokenizer).parse(prompt, ids, TOOLS).tool_calls
    assert (parsed.name, parsed.status.value) == (
        (name, "ok") if name else (written, "unknown_tool")
    )


# Below, the parser and the bridge on a stand-in: a public tokenizer given Muse's
# control tokens. Neither reads the template, so these run without access to the repo.


@pytest.fixture(scope="module")
def stand_in(qwen_tokenizer):
    tokenizer = copy.deepcopy(qwen_tokenizer)
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|start|>",
                "<|message|>",
                "<|eom|>",
                "<|eot|>",
                "<|end_of_text|>",
            ]
        }
    )
    return tokenizer


def invoke(name, body):
    return f'<atem:invoke name="{name}">\n{body}</atem:invoke>'


def message(recipient, calls, end="<|eom|>"):
    return f"<|start|>assistant to={recipient}<|message|><atem:function_calls>\n{calls}\n</atem:function_calls>{end}"


def parameter(name, value):
    return f'<atem:parameter name="{name}">{value}</atem:parameter>\n'


def parsed_reply(tokenizer, reply):
    """``reply``, a whole turn from its ``<|start|>assistant``, parsed after that prompt."""
    prompt = tokenizer.encode("<|start|>assistant", add_special_tokens=False)
    ids = tokenizer.encode(reply[len("<|start|>assistant") :], add_special_tokens=False)
    return extension(tokenizer).parse(prompt, ids, TOOLS)


@pytest.mark.parametrize(
    "calls",
    [
        invoke("bash", parameter("command", "ls"))
        + "\n"
        + invoke("bash", parameter("command", "rm a")),
        invoke("bash", parameter("command", "ls") + '<atem:parameter name="timeout">30'),
        invoke("bash", parameter("command", "ls") + parameter("command", "rm a")),
        invoke("bash", parameter("command", "ls") + "and more"),
    ],
    ids=["two invocations", "an unclosed parameter", "a repeated parameter", "stray text"],
)
def test_a_call_that_is_not_one_well_formed_invocation_is_malformed(stand_in, calls):
    [parsed] = parsed_reply(stand_in, message("bash", calls, "<|eot|>")).tool_calls
    assert parsed.status.value == "malformed_structure"


ONE_CALL = (
    f"<atem:function_calls>\n{invoke('bash', parameter('command', 'ls'))}\n</atem:function_calls>"
)


@pytest.mark.parametrize(
    "body",
    [ONE_CALL + "\n" + ONE_CALL, "First, " + ONE_CALL, ONE_CALL + " and then"],
    ids=["two blocks", "text before the block", "text after the block"],
)
def test_a_call_message_with_more_than_its_block_is_malformed(stand_in, body):
    """The template writes a call's message as its block and nothing else."""
    reply = f"<|start|>assistant to=bash<|message|>{body}<|eot|>"
    [parsed] = parsed_reply(stand_in, reply).tool_calls
    assert parsed.status.value == "malformed_structure"


@pytest.mark.parametrize(
    "command",
    [
        "grep '<atem:function_calls>' t.txt && grep '</atem:function_calls>' t.txt",
        "grep '</atem:parameter>' t.txt",
        "grep '</atem:invoke>' t.txt",
    ],
    ids=["the block's tags", "a parameter's closing tag", "an invocation's closing tag"],
)
def test_a_value_may_hold_the_call_markup(stand_in, command):
    """The template writes a value as it is, so the model may quote the markup in one."""
    calls = invoke("bash", parameter("command", command) + parameter("timeout", "30"))
    [parsed] = parsed_reply(stand_in, message("bash", calls, "<|eot|>")).tool_calls
    assert (parsed.status.value, parsed.arguments) == ("ok", {"command": command, "timeout": 30})


def test_a_value_that_reads_as_a_second_parameter_is_malformed(stand_in):
    """The one quoting the template cannot tell apart from two parameters."""
    value = 'a</atem:parameter>\n<atem:parameter name="timeout">30'
    calls = invoke("bash", parameter("command", value) + parameter("timeout", "30"))
    [parsed] = parsed_reply(stand_in, message("bash", calls, "<|eot|>")).tool_calls
    assert parsed.status.value == "malformed_structure"


def test_an_offered_doubled_name_is_kept(stand_in):
    doubled = {"type": "function", "function": {"name": "bash.bash", "parameters": {}}}
    calls = invoke("bash.bash", parameter("command", "ls"))
    prompt = stand_in.encode("<|start|>assistant", add_special_tokens=False)
    reply = message("bash.bash", calls, "<|eot|>")[len("<|start|>assistant") :]
    ids = stand_in.encode(reply, add_special_tokens=False)
    [parsed] = extension(stand_in).parse(prompt, ids, [*TOOLS, doubled]).tool_calls
    assert parsed.name == "bash.bash"


def test_a_value_that_is_not_its_schemas_json_is_invalid(stand_in):
    """``timeout`` is an integer: "thirty" is reported, not passed on as a string."""
    calls = invoke("bash", parameter("command", "ls") + parameter("timeout", "thirty"))
    [parsed] = parsed_reply(stand_in, message("bash", calls, "<|eot|>")).tool_calls
    assert parsed.status.value == "invalid_json"


def test_a_turn_can_end_at_end_of_text(stand_in):
    reply = "<|start|>assistant to=commentary<|message|>Done.<|end_of_text|>"
    parsed = parsed_reply(stand_in, reply)
    assert (parsed.content, parsed.reasoning_complete) == ("Done.", True)
    ids = stand_in.encode(reply[len("<|start|>assistant") :], add_special_tokens=False)
    bridged = extension(stand_in).bridge(
        [], ids, [{"role": "user", "content": "More."}], TOOLS, previous_messages=[TASK]
    )
    assert bridged is not None
    # The sampled stop ends the turn; no <|eot|> is added after it.
    assert stand_in.decode(list(bridged.ids)).startswith(
        " to=commentary<|message|>Done.<|end_of_text|><|start|>user"
    )


def test_a_turns_text_messages_are_joined(stand_in):
    reply = (
        "<|start|>assistant to=commentary<|message|>First.<|eom|>"
        "<|start|>assistant to=user<|message|>Second.<|eot|>"
    )
    assert parsed_reply(stand_in, reply).content == "First.\n\nSecond."


def bridged_tool_names(tokenizer, reply, results, previous):
    prompt = tokenizer.encode(
        "<|start|>user<|message|>Go.<|eot|><|start|>assistant", add_special_tokens=False
    )
    ids = tokenizer.encode(reply[len("<|start|>assistant") :], add_special_tokens=False)
    bridged = extension(tokenizer).bridge(prompt, ids, results, TOOLS, previous_messages=previous)
    assert bridged is not None
    added = tokenizer.decode(list(bridged.ids)[len(prompt) + len(ids) :])
    return [piece.split("<|message|>")[0] for piece in added.split("<|start|>tool ")[1:]]


TWO_CALLS = message("bash", invoke("bash", parameter("command", "ls"))) + message(
    "write", invoke("write", parameter("file_path", "a.md")), "<|eot|>"
)
WITH_IDS = {
    "role": "assistant",
    "content": "",
    "tool_calls": [call(1, "bash", {"command": "ls"}), call(2, "write", {"file_path": "a.md"})],
}


@pytest.mark.parametrize(
    ("results", "names"),
    [([tool(2, "ok"), tool(1, "a")], ["write", "bash"]), ([tool(2, "ok")], ["write"])],
    ids=["in reverse", "only the second"],
)
def test_results_are_named_by_their_call_id_not_their_position(stand_in, results, names):
    assert bridged_tool_names(stand_in, TWO_CALLS, results, [SYSTEM, TASK, WITH_IDS]) == names


def test_a_result_without_a_matching_id_is_named_by_the_parsed_call(stand_in):
    """With no IDs to go by, the sampled call names it, parsed as the gateway parses it."""
    reply = message("bash", invoke("bash.bash", parameter("command", "ls")), "<|eot|>")
    previous = [SYSTEM, TASK, {"role": "assistant", "content": ""}]
    assert bridged_tool_names(stand_in, reply, [tool(1, "a")], previous) == ["bash"]


def test_the_cookbook_renderer_is_selected_without_a_renderer_setting(tokenizer):
    renderer = create_renderer(MODEL, tokenizer=tokenizer, chat_template_kwargs=OPTIONS)
    assert describe(renderer) == describe(extension(tokenizer))
