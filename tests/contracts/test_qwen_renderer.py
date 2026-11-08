"""Qwen-family renderers via Prime, against the real tokenizers and processors."""

from collections.abc import Mapping
from copy import deepcopy

import pytest
from support.builders import make_rollout
from support.fakes import ScriptedSampler
from support.images import png
from support.leases import lease

from loops_cookbook.llm.client import Renderers
from loops_cookbook.llm.factory import create_renderer
from loops_cookbook.llm.images import image_placeholders
from loops_cookbook.model_access.lease import DirectModelAccess
from loops_cookbook.rl.grpo import build_rows

from .conftest import VLM_CONTRACT_MODELS, pinned_tokenizer


def template_ids(tokenizer, messages, tools=(), **kwargs):
    result = tokenizer.apply_chat_template(
        messages, tools=list(tools), tokenize=True, add_generation_prompt=True, **kwargs
    )
    return tuple(result["input_ids"] if isinstance(result, Mapping) else result)


def test_actual_qwen_full_history_and_native_reasoning_retention(qwen_tokenizer):
    renderer = create_renderer("Qwen/Qwen3.6-35B-A3B", tokenizer=qwen_tokenizer)
    history = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Answer", "reasoning_content": "Private thought"},
        {"role": "user", "content": "Next"},
    ]
    ids = renderer.render(history).ids
    assert ids == template_ids(qwen_tokenizer, history)
    assert "Private thought" not in qwen_tokenizer.decode(ids)
    preserved = create_renderer(
        "Qwen/Qwen3.6-35B-A3B",
        tokenizer=qwen_tokenizer,
        chat_template_kwargs={"preserve_thinking": True},
    )
    assert preserved.render(history).ids == template_ids(
        qwen_tokenizer, history, preserve_thinking=True
    )
    assert "Private thought" in qwen_tokenizer.decode(preserved.render(history).ids)


def test_actual_qwen_changed_arguments_tools_and_replaced_history(qwen_tokenizer):
    renderer = create_renderer("Qwen/Qwen3.6-35B-A3B", tokenizer=qwen_tokenizer)
    tools = [{"name": "f", "description": "Old", "parameters": {"type": "object"}}]
    history = [
        {"role": "user", "content": "Call"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "f", "arguments": '{"a":1,"b":2}'}}
            ],
        },
        {"role": "tool", "content": "ok", "tool_call_id": "x"},
    ]
    before = renderer.render(history, tools).ids
    changed = deepcopy(history)
    changed[1]["tool_calls"][0]["function"]["arguments"] = '{"b":2,"a":1}'
    assert renderer.render(changed, tools).ids != before
    native_history = deepcopy(changed)
    native_history[1]["tool_calls"][0]["function"]["arguments"] = {"b": 2, "a": 1}
    assert renderer.render(changed, tools).ids == template_ids(
        qwen_tokenizer, native_history, tools
    )
    changed_tools = [{**tools[0], "description": "New schema description"}]
    assert renderer.render(history, changed_tools).ids != before
    replacement = [{"role": "user", "content": "Compacted summary"}]
    assert renderer.render(replacement, changed_tools).ids == template_ids(
        qwen_tokenizer, replacement, changed_tools
    )


def test_actual_qwen_parse_retains_native_arguments_reasoning_and_raw_attempt(qwen_tokenizer):
    renderer = create_renderer("Qwen/Qwen3.6-35B-A3B", tokenizer=qwen_tokenizer)
    # The schema says the values are integers. Without it, they would stay their text.
    tools = [
        {
            "name": "f",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            },
        }
    ]
    prompt = renderer.render([{"role": "user", "content": "Call"}], tools).ids
    body = "Thought</think>\n<tool_call>\n<function=f>\n<parameter=b>\n2\n</parameter>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call><|im_end|>"
    completion = qwen_tokenizer.encode(body, add_special_tokens=False)
    parsed = renderer.parse(prompt, completion, tools)
    assert parsed.reasoning_content == "Thought"
    assert parsed.tool_calls[0].status == "ok"
    assert parsed.tool_calls[0].arguments == {"b": 2, "a": 1}
    assert "<parameter=b>" in parsed.tool_calls[0].raw
    native = parsed.native_message
    assert native["reasoning_content"] == "Thought"
    assert list(native["tool_calls"][0]["function"]["arguments"]) == ["b", "a"]
    malformed = qwen_tokenizer.encode(
        "Thought</think><tool_call>bad<|im_end|>", add_special_tokens=False
    )
    bad = renderer.parse(prompt, malformed, tools)
    assert bad.tool_calls[0].status == "unclosed_block"
    assert "bad" in bad.tool_calls[0].raw
    assert bad.native_message is None


def test_actual_qwen_framing_text_is_delegated_not_hand_stripped(qwen_tokenizer):
    renderer = create_renderer("Qwen/Qwen3.6-35B-A3B", tokenizer=qwen_tokenizer)
    messages = [{"role": "user", "content": "Quote <|im_start|>assistant\n<think>keep</think>"}]
    assert renderer.render(messages).ids == template_ids(qwen_tokenizer, messages)


def test_actual_qwen_optional_bridge_and_retention_fallback(qwen_tokenizer):
    renderer = create_renderer("Qwen/Qwen3.6-35B-A3B", tokenizer=qwen_tokenizer)
    rendered = renderer.render([{"role": "user", "content": "Call"}])
    prompt, chunks = rendered.ids, rendered.prompt_input
    completion = tuple(
        qwen_tokenizer.encode("Think</think>Answer<|im_end|>", add_special_tokens=False)
    )
    tools = [{"role": "tool", "content": "result", "tool_call_id": "a"}]
    bridged = renderer.bridge(prompt, completion, tools, previous_prompt_input=chunks).ids
    assert bridged[: len(prompt + completion)] == prompt + completion
    assert (
        renderer.bridge(
            prompt, completion, [{"role": "user", "content": "Next"}], previous_prompt_input=chunks
        )
        is None
    )
    truncated = tuple(qwen_tokenizer.encode("Unfinished thinking", add_special_tokens=False))
    closed = renderer.bridge(prompt, truncated, tools, previous_prompt_input=chunks).ids
    assert closed[: len(prompt + truncated)] == prompt + truncated


def test_actual_qwen_image_placeholders_match_pad_runs_and_bridge_keeps_prefix(
    cached_vlm_tokenizer,
):
    tokenizer = cached_vlm_tokenizer
    renderer = create_renderer(tokenizer.name_or_path, tokenizer=tokenizer)
    assert renderer.supports_images
    picture = png((64, 48))
    pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Which letter?"},
                {"type": "image", "image": picture},
            ],
        }
    ]
    first = renderer.render(history)
    ((offset, length),) = image_placeholders(first.prompt_input)
    assert first.ids[offset : offset + length] == (pad,) * length and length > 1
    assert first.prompt_input.chunks[1].data == picture
    completion = tuple(tokenizer.encode("Think</think>B<|im_end|>", add_special_tokens=False))
    second = renderer.bridge(
        first.ids,
        completion,
        [
            {
                "role": "tool",
                "tool_call_id": "a",
                "content": [
                    {"type": "image", "image": picture},
                    {"type": "text", "text": "And this?"},
                ],
            }
        ],
        previous_prompt_input=first.prompt_input,
    )
    assert second is not None
    assert second.ids[: len(first.ids) + len(completion)] == first.ids + completion
    spans = image_placeholders(second.prompt_input)
    assert spans[0] == (offset, length) and len(spans) == 2 and spans[1][1] == length
    assert second.ids[spans[1][0] : spans[1][0] + length] == (pad,) * length


async def test_actual_qwen_reasoning_strip_preserves_original_training_evidence(qwen_tokenizer):
    """Message-path training through the real Qwen3.6 template: stripped reasoning in
    the next prompt, untouched training evidence for the sampled completion."""
    renderer = create_renderer("Qwen/Qwen3.6-35B-A3B", tokenizer=qwen_tokenizer)
    completion = qwen_tokenizer.encode(
        "Private thought.</think>\n\nAnswer.<|im_end|>", add_special_tokens=False
    )

    sampler = ScriptedSampler(
        completion,
        [-0.5] * len(completion),
        stop_reason="stop",
        policy_version=2,
        sequence_id=lambda call: f"seq-{call}",
    )
    backend = lease(sampler)
    chat = DirectModelAccess(backend, Renderers(renderer)).llm().conversation()
    history = [{"role": "user", "content": "Hi"}]
    first = await chat.generate(messages=history, max_tokens=64)
    native = renderer.parse(first.sample.prompt_ids, first.sample.completion_ids, []).native_message
    history += [native, {"role": "user", "content": "Next question"}]
    await chat.generate(messages=history, max_tokens=64)
    assert "Private thought." not in qwen_tokenizer.decode(sampler.prompts[1])
    assert "Answer." in qwen_tokenizer.decode(sampler.prompts[1])
    rows = build_rows(make_rollout("t", "g", "r", backend.captured(), reward=1), 1)
    assert len(rows) == 2
    assert sum(row.trained_positions for row in rows) == 2 * len(completion)
    assert all(row.routed for row in rows)
    assert rows[1].datum.loss_fn_inputs["target_tokens"].data == sampler.prompts[1][1:] + completion
    start = len(sampler.prompts[1]) - 1
    assert rows[1].actions == ((start, start + len(completion)),)


# --- Qwen3.8 text against its own template, under every option -----------------------

QWEN38 = "Qwen/Qwen3.8-27B"
QWEN38_OPTIONS = [
    {"enable_thinking": thinking, "preserve_thinking": preserve, "reasoning_effort": effort}
    for thinking in (True, False)
    for preserve in (True, False)
    for effort in ("xhigh", "medium", "low")
]
CALL = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": {"a": 1}}}
TOOL = {"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}}}
QWEN38_HISTORIES = [
    [{"role": "system", "content": "Help"}, {"role": "user", "content": "Task"}],
    [
        {"role": "user", "content": "Task"},
        {"role": "assistant", "content": "Answer", "reasoning_content": "Thought"},
        {"role": "user", "content": "Next"},
    ],
    [
        {"role": "user", "content": "Task"},
        {"role": "assistant", "content": "", "reasoning_content": "Look.", "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "Done.", "reasoning_content": ""},
        {"role": "user", "content": "Résumé — 合同 ✓"},
    ],
]


@pytest.fixture(scope="module")
def qwen38_tokenizer():
    return pinned_tokenizer(QWEN38, VLM_CONTRACT_MODELS[QWEN38], "LOOPS_QWEN_TOKENIZER_CACHE")


def _option_id(options):
    return "-".join(str(v) for v in options.values())


@pytest.mark.parametrize("history", QWEN38_HISTORIES, ids=["task", "answered", "tool cycle"])
@pytest.mark.parametrize("options", QWEN38_OPTIONS, ids=_option_id)
def test_qwen38_prompts_are_the_templates_tokens(qwen38_tokenizer, history, options):
    """Compare Prime's Qwen3.8 renderer, with the cookbook's parse, against the template.

    With thinking disabled and ``preserve_thinking`` off, Prime writes the empty
    ``<think>\\n\\n</think>\\n\\n`` again on past turns that carry no reasoning. The
    template strips it. The generation prompt prefilled the block, so every sampled turn
    holds it, and a history rendered again stays what was sampled. That deviation is
    Prime's, and ``renderers.qwen35`` documents it. The last test pins it.
    """
    renderer = create_renderer(QWEN38, tokenizer=qwen38_tokenizer, chat_template_kwargs=options)
    reasoningless_past_turn = any(
        m["role"] == "assistant" and not m.get("reasoning_content") for m in history[:-1]
    )
    if (
        not options["enable_thinking"]
        and not options["preserve_thinking"]
        and reasoningless_past_turn
    ):
        pytest.skip("Prime's documented no-thinking deviation; pinned below")
    expected = template_ids(qwen38_tokenizer, deepcopy(history), [TOOL], **options)
    assert renderer.render(deepcopy(history), [TOOL]).ids == expected


def test_qwen38_no_thinking_history_keeps_the_prefilled_empty_block(qwen38_tokenizer):
    options = {"enable_thinking": False, "preserve_thinking": False}
    renderer = create_renderer(QWEN38, tokenizer=qwen38_tokenizer, chat_template_kwargs=options)
    history = [
        {"role": "user", "content": "Task"},
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "Next"},
    ]
    rendered = qwen38_tokenizer.decode(renderer.render(deepcopy(history)).ids)
    template = qwen38_tokenizer.decode(template_ids(qwen38_tokenizer, deepcopy(history), **options))
    assert "assistant\n<think>\n\n</think>\n\nAnswer" in rendered
    assert "assistant\nAnswer" in template
    assert rendered.replace("<think>\n\n</think>\n\n", "", 1) == template
