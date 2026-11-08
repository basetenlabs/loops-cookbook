"""Pinned real Qwen3.5 tokenizer, opt-in cookbook parsing and exact-prefix contracts."""

from collections.abc import Mapping

import pytest
from renderers import Qwen35RendererConfig
from renderers.qwen35 import Qwen35Renderer
from support.unit_rest import BASH_TOOLS as TOOLS

from loops_cookbook.llm.factory import create_renderer
from loops_cookbook.llm.prime import PrimeRenderer

from .conftest import pinned_tokenizer

MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
EXTENSION = "loops_cookbook.llm.renderer_extensions.qwen35:renderer"


@pytest.fixture(scope="module")
def tokenizer():
    return pinned_tokenizer(MODEL, REVISION, "LOOPS_QWEN_TOKENIZER_CACHE")


def extension(tokenizer):
    return create_renderer(
        MODEL,
        tokenizer=tokenizer,
        renderer=EXTENSION,
        chat_template_kwargs={"enable_thinking": False},
    )


def completion(tokenizer, key="command", value="echo ok"):
    text = f"<tool_call>\n<function=bash>\n<parameter={key}>\n{value}\n</parameter>\n</function>\n</tool_call><|im_end|>"
    return tuple(tokenizer.encode(text, add_special_tokens=False))


def test_prompt_and_valid_parse_are_unchanged(tokenizer):
    new = extension(tokenizer)
    old = PrimeRenderer(Qwen35Renderer(tokenizer, Qwen35RendererConfig(enable_thinking=False)))
    messages = [{"role": "user", "content": "Run a command."}]
    prompt = new.render(messages, TOOLS)
    assert prompt == old.render(messages, TOOLS)
    expected = tokenizer.apply_chat_template(
        messages, tools=TOOLS, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    if isinstance(expected, Mapping):
        expected = expected["input_ids"]
    assert prompt.ids == tuple(expected)
    ids = completion(tokenizer)
    assert new.parse(prompt.ids, ids, TOOLS) == old.parse(prompt.ids, ids, TOOLS)
    assert new.stop_sequences == old.stop_sequences


def test_whitespace_changes_interpretation_not_sampled_prefix(tokenizer):
    new = extension(tokenizer)
    messages = [{"role": "user", "content": "Run a command."}]
    prompt = new.render(messages, TOOLS)
    ids = completion(tokenizer, "\ncommand")
    parsed = new.parse(prompt.ids, ids, TOOLS)
    assert parsed.native_message["tool_calls"][0]["function"]["arguments"] == {"command": "echo ok"}
    assert "<parameter=\ncommand>" in parsed.tool_calls[0].raw
    bridged = new.bridge(
        prompt.ids,
        ids,
        [{"role": "tool", "content": "ok", "tool_call_id": "a"}],
        TOOLS,
        previous_messages=[*messages, parsed.native_message],
        previous_prompt_input=prompt.prompt_input,
    )
    assert bridged.ids[: len(prompt.ids + ids)] == prompt.ids + ids
    assert "<parameter=\ncommand>" in tokenizer.decode(bridged.ids)


@pytest.mark.parametrize("key", ["parameter=command", "\ncommand"])
async def test_tool_feedback_then_recovery_keeps_exact_tokens(tokenizer, key):
    from copy import deepcopy

    from baseten.loops.models import SamplingParams
    from support.builders import make_rollout
    from support.fakes import ScriptedSampler
    from support.leases import lease

    from loops_cookbook.llm.client import Renderers
    from loops_cookbook.model_access.lease import DirectModelAccess
    from loops_cookbook.rl.grpo import build_rows

    scripted = [completion(tokenizer, key), completion(tokenizer)]
    sampler = ScriptedSampler(stop_reason="stop")

    def next_tokens(index):
        ids = scripted[index - 1]
        sampler.logprobs = [-0.1] * len(ids)
        return ids

    sampler.tokens = next_tokens
    backend = lease(sampler)
    chat = DirectModelAccess(backend, Renderers(extension(tokenizer))).llm().conversation()
    params = SamplingParams(max_tokens=256)
    messages = [{"role": "user", "content": "Run a command."}]
    first = await chat.generate(params, messages=messages, tools=TOOLS)
    (call,) = first.native_message["tool_calls"]
    args = call["function"]["arguments"]
    expected_key = "command" if key == "\ncommand" else key
    assert args == {expected_key: "echo ok"}
    # The tool owns validation, not this parser. No command is executed.
    result = "Error: command is required" if "command" not in args else "ok"
    assert (result.startswith("Error:")) == (key == "parameter=command")
    first_sample = deepcopy(backend.captured()[0])
    messages += [
        first.native_message,
        {"role": "tool", "tool_call_id": call.get("id", "a"), "content": result},
    ]
    last = await chat.generate(params, messages=messages, tools=TOOLS)
    assert last.native_message["tool_calls"][0]["function"]["arguments"] == {"command": "echo ok"}
    samples = backend.captured()
    assert len(samples) == 2 and samples[0] == first_sample
    prefix = first_sample.prompt_ids + first_sample.completion_ids
    assert samples[1].prompt_ids[: len(prefix)] == prefix
    assert list(samples[0].completion_ids) == list(scripted[0])
    assert len(samples[0].completion_ids) == len(samples[0].logprobs)
    rows = build_rows(make_rollout("t", "g", "r", samples, reward=1), 1)
    assert sum(row.trained_positions for row in rows) == sum(map(len, scripted))
    assert rows[-1].datum.loss_fn_inputs["target_tokens"].data[-len(scripted[-1]) :] == list(
        scripted[-1]
    )


def test_multiple_calls_and_unrepresentable_call_are_not_silently_dropped(tokenizer):
    new = extension(tokenizer)
    prompt = new.render([{"role": "user", "content": "Run commands."}], TOOLS).ids
    text = tokenizer.decode(completion(tokenizer, "parameter=command")).removesuffix("<|im_end|>")
    text += tokenizer.decode(completion(tokenizer, "\ncommand"))
    result = new.parse(prompt, tokenizer.encode(text, add_special_tokens=False), TOOLS)
    assert [c.arguments for c in result.tool_calls] == [
        {"parameter=command": "echo ok"},
        {"command": "echo ok"},
    ]
    assert len(result.native_message["tool_calls"]) == 2
    broken = text.removesuffix("<|im_end|>") + "<tool_call>bad<|im_end|>"
    result = new.parse(prompt, tokenizer.encode(broken, add_special_tokens=False), TOOLS)
    assert len(result.tool_calls) == 3 and result.native_message is None
    assert result.tool_calls[-1].status == "unclosed_block"


def test_default_prime_reports_the_bad_argument_and_withholds_the_message(tokenizer):
    """Prime's parser may read a block it marks ``invalid_json`` in part, so the message
    is withheld and the turn continues from the sampled tokens. The extension reads the
    block whole and keeps the message."""
    old = PrimeRenderer(Qwen35Renderer(tokenizer, Qwen35RendererConfig(enable_thinking=False)))
    prompt = old.render([{"role": "user", "content": "Run a command."}], TOOLS).ids
    parsed = old.parse(prompt, completion(tokenizer, "parameter=command"), TOOLS)
    assert parsed.tool_calls[0].status == "invalid_json"
    assert parsed.native_message is None


def xml_call(key="command", value="ls", *, closed=True, function="bash"):
    text = f"<tool_call>\n<function={function}>\n<parameter={key}>\n{value}\n"
    return text + ("</parameter>\n</function>\n</tool_call>" if closed else "")


@pytest.mark.parametrize(
    ("text", "calls"),
    [
        # Stock Prime accepts this as one call whose command holds the second call.
        (xml_call(closed=False) + xml_call(value="pwd") + "<|im_end|>", None),
        (xml_call(value="ls\n<tool_call>") + "<|im_end|>", None),
        (xml_call(closed=False) + "<|im_end|>", None),
        # Cut by the token limit: no closing tags and no end-of-turn token.
        (xml_call(value="cat <<'EOF' > notes.txt\nfirst line", closed=False), None),
        (
            xml_call() + "\n" + xml_call(value="pwd") + "<|im_end|>",
            [("bash", {"command": "ls"}), ("bash", {"command": "pwd"})],
        ),
        (xml_call("file_path", "notes.txt") + "<|im_end|>", [("bash", {"file_path": "notes.txt"})]),
        # Names are kept as written. The tool reports an undeclared argument, and an
        # undeclared tool is ``unknown_tool``. Both calls keep their messages.
        (
            xml_call("parameter=command") + "<|im_end|>",
            [("bash", {"parameter=command": "ls"})],
        ),
        (xml_call(function="x=y") + "<|im_end|>", [("x=y", {"command": "ls"})]),
    ],
)
def test_tool_call_markup_boundary(tokenizer, text, calls):
    new = extension(tokenizer)
    prompt = new.render([{"role": "user", "content": "Run a command."}], TOOLS).ids
    parsed = new.parse(prompt, tokenizer.encode(text, add_special_tokens=False), TOOLS)
    if calls is None:
        assert parsed.native_message is None
        assert any(call.status != "ok" for call in parsed.tool_calls)
    else:
        assert [(c.name, c.arguments) for c in parsed.tool_calls] == calls
        native = parsed.native_message["tool_calls"]
        assert [(c["function"]["name"], c["function"]["arguments"]) for c in native] == calls


def test_argument_values_keep_whitespace_inside_template_separators(tokenizer):
    new = extension(tokenizer)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "edit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "count": {"type": "integer"},
                        "options": {"type": "object"},
                    },
                },
            },
        }
    ]
    arguments = {
        "file_path": "  a.py ",
        "old_string": "    x = 1\n",
        "new_string": "\n\tx = 2  \n\n",
        "count": 3,
        "options": {"a": [1]},
    }
    messages = [{"role": "user", "content": "Edit a file."}]
    prompt = new.render(messages, tools).ids
    call = {"id": "a", "type": "function", "function": {"name": "edit", "arguments": arguments}}
    history = [
        *messages,
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "a", "content": "ok"},
    ]
    ids = new.render(history, tools).ids
    assert ids[: len(prompt)] == prompt
    end = ids.index(tokenizer.convert_tokens_to_ids("<|im_end|>"), len(prompt)) + 1
    parsed = new.parse(prompt, ids[len(prompt) : end], tools)
    assert parsed.native_message["tool_calls"][0]["function"]["arguments"] == arguments
    # The model may omit the newlines the template writes around a value.
    text = "<tool_call>\n<function=edit>\n<parameter=old_string>abc</parameter>\n</function>\n</tool_call><|im_end|>"
    parsed = new.parse(prompt, tokenizer.encode(text, add_special_tokens=False), tools)
    assert parsed.tool_calls[0].arguments == {"old_string": "abc"}


TYPED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "edit",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "count": {"type": "integer"},
                    "all": {"type": "boolean"},
                    "limit": {"type": ["integer", "null"]},
                    "lines": {"type": "array", "items": {"type": "integer"}},
                },
            },
        },
    }
]


@pytest.mark.parametrize(
    "tools,key,text",
    [
        # A declared type that can't read the text.
        (TYPED_TOOLS, "count", "30s"),
        # An undeclared argument.
        (TYPED_TOOLS, "timeout", " 600"),
        # With no tools, nothing says what any value was.
        ((), "count", " 3 "),
    ],
)
def test_text_no_type_reads_reaches_the_tool_as_text(tokenizer, tools, key, text):
    new = extension(tokenizer)
    prompt = new.render([{"role": "user", "content": "Edit a file."}], tools).ids
    raw = (
        f"<tool_call>\n<function=edit>\n<parameter={key}>\n{text}\n</parameter>\n</function>\n"
        "</tool_call><|im_end|>"
    )
    parsed = new.parse(prompt, tokenizer.encode(raw, add_special_tokens=False), tools)
    [result] = parsed.tool_calls
    assert result.status == "ok"
    assert result.arguments == {key: text}
    assert parsed.native_message is not None


def test_prime_still_strips_string_arguments(tokenizer):
    """Pass text that no type reads on to the tool, where Prime strips or refuses it.

    When this fails, Prime keeps string values as the template wrote them, so revisit
    the extension's parse.
    """
    old = PrimeRenderer(Qwen35Renderer(tokenizer, Qwen35RendererConfig(enable_thinking=False)))
    prompt = old.render([{"role": "user", "content": "Edit a file."}], TYPED_TOOLS).ids
    raw = (
        "<tool_call>\n<function=edit>\n<parameter=old_string>\n    x\n\n</parameter>\n"
        "</function>\n</tool_call><|im_end|>"
    )
    parsed = old.parse(prompt, tokenizer.encode(raw, add_special_tokens=False), TYPED_TOOLS)
    assert parsed.tool_calls[0].arguments == {"old_string": "x"}
