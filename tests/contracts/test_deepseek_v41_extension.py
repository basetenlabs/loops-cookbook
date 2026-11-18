"""DeepSeek-V4.1 renderer: template parity, images, the bridge and the parser."""

import io
import itertools
from copy import deepcopy

import pytest

from loops_cookbook.llm.factory import create_renderer
from loops_cookbook.llm.renderer_extensions import deepseek_v41 as extension
from loops_cookbook.llm.renderer_extensions.deepseek_v41 import (
    DeepSeekV41Renderer,
    DeepSeekV41RendererConfig,
    image_tokens,
)
from loops_cookbook.llm.rendering import RenderingError

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            },
        },
    },
    {"name": "bash", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}},
]


def call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


ASSISTANT_WITH_CALLS = {
    "role": "assistant",
    "content": "",
    "reasoning_content": "I need two tools.",
    "tool_calls": [
        call("a", "get_weather", {"city": "Paris", "days": 2}),
        call("b", "bash", {"cmd": "ls\n  -la "}),
    ],
}
HISTORIES = [
    [{"role": "user", "content": "Hi"}],
    [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "reasoning_content": "greet"},
        {"role": "user", "content": "Again"},
    ],
    [
        {"role": "user", "content": "Weather?"},
        ASSISTANT_WITH_CALLS,
        {"role": "tool", "tool_call_id": "b", "content": "files"},
        {"role": "tool", "tool_call_id": "a", "content": "sunny"},
    ],
    [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": "a1",
            "reasoning_content": "r1",
            "tool_calls": [call("a", "bash", {"cmd": "x"})],
        },
        {"role": "tool", "tool_call_id": "a", "content": "out"},
        {"role": "assistant", "content": "done", "reasoning_content": "r2"},
        {"role": "user", "content": "q2"},
    ],
    [
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "x"},
        {"role": "user", "content": "d"},
    ],
    [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "x", "reasoning_content": "r"},
    ],
]
OPTIONS = [
    {},
    {"enable_thinking": False},
    {"reasoning_effort": "max", "drop_thinking": False},
    {"reasoning_effort": 30},
]


def template_ids(tokenizer, messages, tools=None, *, add_generation_prompt=True, **options):
    text = tokenizer.apply_chat_template(
        deepcopy(messages),
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **options,
    )
    return tokenizer.encode(text, add_special_tokens=False)


@pytest.mark.parametrize(
    ("history", "tools", "options", "generation"),
    list(itertools.product(HISTORIES, [None, TOOLS], OPTIONS, [True, False])),
)
def test_prompts_are_the_templates_tokens(deepseek_v41, history, tools, options, generation):
    variant, tokenizer, _ = deepseek_v41
    original = deepcopy(history)
    rendered = DeepSeekV41Renderer(tokenizer, variant, DeepSeekV41RendererConfig(**options)).render(
        history, tools=tools, add_generation_prompt=generation
    )
    expected = template_ids(tokenizer, history, tools, add_generation_prompt=generation, **options)
    assert rendered.token_ids == expected
    assert len(rendered.message_indices) == len(rendered.sampled_mask) == len(expected)
    assert history == original


def test_the_cookbook_selects_it_by_name(deepseek_v41):
    variant, tokenizer, _ = deepseek_v41
    renderer = create_renderer(tokenizer.name_or_path, tokenizer=tokenizer)
    assert renderer._renderer.variant == variant
    assert renderer.supports_images


def test_the_vision_config_is_the_checkpoints(deepseek_v41):
    _, tokenizer, config = deepseek_v41
    vision = config["vision_config"]
    assert config["image_token_id"] == tokenizer.convert_tokens_to_ids(extension.IMAGE)
    assert (
        vision["patch_size"],
        vision["downsample_ratio"],
        vision["max_image_tokens"],
        vision["min_pixels"],
        vision["max_wh_ratio"],
    ) == (
        extension.PATCH_SIZE,
        extension.DOWNSAMPLE_RATIO,
        extension.MAX_IMAGE_TOKENS,
        extension.MIN_PIXELS,
        None,
    )


def png(width, height):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def image(width, height):
    return {"type": "image", "image": png(width, height)}


IMAGE_HISTORIES = [
    [{"role": "user", "content": [image(640, 480), {"type": "text", "text": "What is this?"}]}],
    [
        {"role": "system", "content": "Look closely."},
        {"role": "user", "content": [image(100, 100)]},
        {"role": "user", "content": [image(64, 900), image(4000, 3000)]},
    ],
    [
        {"role": "user", "content": "Screens?"},
        ASSISTANT_WITH_CALLS,
        {"role": "tool", "tool_call_id": "b", "content": [image(300, 200)]},
        {
            "role": "tool",
            "tool_call_id": "a",
            "content": [{"type": "text", "text": "x"}, image(50, 70)],
        },
    ],
]


@pytest.mark.parametrize("history", IMAGE_HISTORIES)
def test_images_are_the_templates_with_each_image_expanded(deepseek_v41, history):
    from baseten.loops.models import ImageChunk
    from PIL import Image

    _, tokenizer, _ = deepseek_v41
    renderer = create_renderer(tokenizer.name_or_path, tokenizer=tokenizer)
    prompt = renderer.render(history, TOOLS)
    chunks = [chunk for chunk in prompt.prompt_input.chunks if isinstance(chunk, ImageChunk)]
    sizes = [Image.open(io.BytesIO(chunk.data)).size for chunk in chunks]
    assert [chunk.expected_tokens for chunk in chunks] == [image_tokens(*s) for s in sizes]
    image_id = tokenizer.convert_tokens_to_ids(extension.IMAGE)
    expected, counts = [], iter(chunk.expected_tokens for chunk in chunks)
    for token in template_ids(tokenizer, history, TOOLS):
        expected.extend([token] * next(counts) if token == image_id else [token])
    assert list(prompt.ids) == expected
    assert next(counts, None) is None


def test_tool_result_images_follow_the_call_order(deepseek_v41):
    from baseten.loops.models import ImageChunk
    from PIL import Image

    _, tokenizer, _ = deepseek_v41
    renderer = create_renderer(tokenizer.name_or_path, tokenizer=tokenizer)
    prompt = renderer.render(IMAGE_HISTORIES[2], TOOLS)
    chunks = [chunk for chunk in prompt.prompt_input.chunks if isinstance(chunk, ImageChunk)]
    # Call "a" comes first, so its result's 50x70 image renders first.
    assert [Image.open(io.BytesIO(chunk.data)).size for chunk in chunks] == [(50, 70), (300, 200)]


@pytest.mark.parametrize("where", ["user", "reasoning", "tool"])
def test_text_holding_the_image_token_is_refused(deepseek_v41, where):
    _, tokenizer, _ = deepseek_v41
    renderer = create_renderer(tokenizer.name_or_path, tokenizer=tokenizer)
    text = f"see {extension.IMAGE}"
    history = {
        "user": [{"role": "user", "content": text}],
        "reasoning": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "x", "reasoning_content": text},
            {"role": "user", "content": "q"},
        ],
        "tool": [{"role": "user", "content": "q"}, {"role": "tool", "content": text}],
    }[where]
    with pytest.raises(RenderingError, match="image parts"):
        renderer.render(history)


def flat(model_input):
    """Return the chunks with adjacent text merged, since the bridge keeps the completion apart."""
    from baseten.loops.models import EncodedTextChunk

    merged = []
    for chunk in model_input.chunks:
        if isinstance(chunk, EncodedTextChunk) and merged and isinstance(merged[-1], list):
            merged[-1].extend(chunk.tokens)
        elif isinstance(chunk, EncodedTextChunk):
            merged.append(list(chunk.tokens))
        else:
            merged.append(chunk)
    return merged


def sampled(renderer, history, assistant, tools):
    """Return the prompt for ``history`` and the completion the model sampled as ``assistant``."""
    prompt = renderer.render(history, tools)
    full = renderer.render_conversation([*history, assistant], tools)
    assert full.ids[: len(prompt.ids)] == prompt.ids
    return prompt, full.ids[len(prompt.ids) :]


BRIDGES = [
    # Declared tools keep all reasoning, so the bridge always extends the sampled turn.
    pytest.param({}, TOOLS, HISTORIES[2][2:], True, id="tools"),
    # Without tools, V4.1 starts a turn at a tool result and drops the reasoning. 1017 keeps it.
    pytest.param({}, (), HISTORIES[2][2:], "1017", id="tool results without tools"),
    pytest.param({}, (), [{"role": "user", "content": "more"}], False, id="new user turn"),
    pytest.param(
        {"drop_thinking": False}, (), [{"role": "user", "content": "more"}], True, id="kept"
    ),
    pytest.param({"enable_thinking": False}, (), [{"role": "user", "content": "more"}], True),
    pytest.param({}, TOOLS, [IMAGE_HISTORIES[0][0]], True, id="image"),
]


@pytest.mark.parametrize(("options", "tools", "new", "bridges"), BRIDGES)
def test_the_bridge_extends_the_sampled_turn_as_a_full_render(
    deepseek_v41, options, tools, new, bridges
):
    variant, tokenizer, _ = deepseek_v41
    renderer = create_renderer(
        tokenizer.name_or_path, tokenizer=tokenizer, chat_template_kwargs=options
    )
    history, assistant = HISTORIES[2][:1], HISTORIES[2][1]
    prompt, completion = sampled(renderer, history, assistant, tools)
    result = renderer.bridge(
        prompt.ids,
        completion,
        new,
        tools,
        previous_messages=[*history, assistant],
        previous_prompt_input=prompt.prompt_input,
    )
    if bridges is True or bridges == variant:
        full = renderer.render([*history, assistant, *new], tools)
        assert result is not None
        assert result.ids == full.ids
        assert flat(result.prompt_input) == flat(full.prompt_input)
    else:
        assert result is None


def test_the_bridge_closes_a_turn_cut_off_in_its_reasoning(deepseek_v41):
    _, tokenizer, _ = deepseek_v41
    renderer = create_renderer(tokenizer.name_or_path, tokenizer=tokenizer)
    history = HISTORIES[2][:1]
    prompt = renderer.render(history, TOOLS)
    cut = renderer.tokenize("I am still thinking")
    result = renderer.bridge(
        prompt.ids,
        cut,
        [{"role": "user", "content": "go on"}],
        TOOLS,
        previous_messages=history,
        previous_prompt_input=prompt.prompt_input,
    )
    closed = {"role": "assistant", "content": "", "reasoning_content": "I am still thinking"}
    full = renderer.render([*history, closed, {"role": "user", "content": "go on"}], TOOLS)
    assert result is not None and result.ids == full.ids


def test_the_parser_reads_back_what_the_template_wrote(deepseek_v41):
    _, tokenizer, _ = deepseek_v41
    renderer = create_renderer(tokenizer.name_or_path, tokenizer=tokenizer)
    history = HISTORIES[2][:1]
    assistant = {
        "role": "assistant",
        "content": "Checking.",
        "reasoning_content": "  plan\n",
        "tool_calls": [
            call("a", "get_weather", {"city": "7", "days": 2}),
            call("b", "bash", {"cmd": "cat <<EOF\n  x\nEOF\n"}),
            call("c", "bash", {"cmd": {"nested": [1, None]}}),
        ],
    }
    prompt, completion = sampled(renderer, history, assistant, TOOLS)
    parsed = renderer.parse(prompt.ids, completion, TOOLS)
    assert parsed.native_message == {
        "role": "assistant",
        "content": "Checking.",
        "reasoning_content": "  plan\n",
        "tool_calls": [
            {"type": "function", "function": call["function"]} for call in assistant["tool_calls"]
        ],
    }
    assert [call.status for call in parsed.tool_calls] == ["ok", "ok", "ok"]


def test_the_parser_reports_a_call_it_cannot_read_whole(deepseek_v41):
    _, tokenizer, _ = deepseek_v41
    renderer = create_renderer(tokenizer.name_or_path, tokenizer=tokenizer)
    history = HISTORIES[2][:1]
    prompt = renderer.render(history, TOOLS)
    unknown = renderer.tokenize(
        'r</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="rm">\n'
        '<｜DSML｜ parameter name="x" string="false">not json</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>"
    )
    parsed = renderer.parse(prompt.ids, unknown, TOOLS)
    assert [call.status for call in parsed.tool_calls] == ["unknown_tool"]
    assert parsed.tool_calls[0].arguments == {"x": "not json"}
    assert parsed.native_message is not None
    cut = renderer.tokenize('r</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="bash">\n')
    parsed = renderer.parse(prompt.ids, cut, TOOLS)
    assert [call.status for call in parsed.tool_calls] == ["unclosed_block"]
    assert parsed.native_message is None


REFUSED = [
    pytest.param(
        [{"role": "user", "content": "q"}, {"role": "system", "content": "mid"}], id="late system"
    ),
    pytest.param(
        [{"role": "system", "content": [image(10, 10)]}, {"role": "user", "content": "q"}],
        id="system image",
    ),
    pytest.param(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [call("a", "bash", '{"cmd": "x"}')]},
        ],
        id="string arguments",
    ),
    pytest.param(
        [{"role": "user", "content": "q"}, {"role": "tool", "tool_call_id": "a", "content": None}],
        id="no tool content",
    ),
]


@pytest.mark.parametrize("history", REFUSED)
def test_what_agents_do_not_send_is_refused(deepseek_v41, history):
    _, tokenizer, _ = deepseek_v41
    renderer = create_renderer(tokenizer.name_or_path, tokenizer=tokenizer)
    with pytest.raises(RenderingError):
        renderer.render(history)
