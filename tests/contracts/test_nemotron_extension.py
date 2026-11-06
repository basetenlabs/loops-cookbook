"""Nemotron 3 Ultra and 3.5 Super VL against each model's own template and processor.

Every conversation is rendered by the template (``apply_chat_template``, then the image
processor's expansion for images) and by the cookbook's renderer, and the token IDs must
be identical, under every combination of the template's options. A bridged turn must
give the tokens of the whole conversation rendered by the template.
"""

import copy
import itertools

import pytest
from baseten.loops.models import ImageChunk
from support.images import lettered_png, pil
from support.unit_rest import BASH_TOOLS

from loops_cookbook.llm.factory import create_renderer, describe
from loops_cookbook.llm.images import image_placeholders
from loops_cookbook.llm.renderer_extensions.nemotron import super_vl
from loops_cookbook.llm.rendering import RenderingError

from .conftest import (
    NEMOTRON_SUPER_VL,
    NEMOTRON_SUPER_VL_REVISIONS,
    NEMOTRON_ULTRA,
    NEMOTRON_ULTRA_REVISION,
    pinned_tokenizer,
)

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
TOOLS = [*BASH_TOOLS, WRITE]
TEXT_MODELS = {
    NEMOTRON_ULTRA: (NEMOTRON_ULTRA_REVISION, "UltraRenderer"),
    NEMOTRON_SUPER_VL: (NEMOTRON_SUPER_VL_REVISIONS[NEMOTRON_SUPER_VL], "SuperVLRenderer"),
}


@pytest.fixture(scope="module", params=sorted(TEXT_MODELS))
def model(request):
    name = request.param
    revision, _ = TEXT_MODELS[name]
    return name, pinned_tokenizer(name, revision, "LOOPS_NEMOTRON_TOKENIZER_CACHE")


def call(n, name, arguments):
    return {"id": f"c{n}", "type": "function", "function": {"name": name, "arguments": arguments}}


def tool(n, name, content):
    return {"role": "tool", "tool_call_id": f"c{n}", "name": name, "content": content}


def text(value):
    return {"type": "text", "text": value}


def image(raw):
    return {"type": "image", "image": raw}


def user(*parts):
    return {"role": "user", "content": list(parts)}


SYSTEM = {"role": "system", "content": "You are a lawyer."}
TASK = {"role": "user", "content": "Summarise the lease."}
LOOK = {
    "role": "assistant",
    "content": "Let me look.",
    "reasoning_content": "List the files.",
    "tool_calls": [call(1, "bash", {"command": "ls"})],
}
DONE = {"role": "assistant", "content": "Done.", "reasoning_content": "Easy."}
CONVERSATIONS = {
    "the task": [SYSTEM, TASK],
    "a tool cycle": [SYSTEM, TASK, LOOK, tool(1, "bash", "a.docx")],
    "two calls, nested arguments": [
        SYSTEM,
        TASK,
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Two things.",
            "tool_calls": [
                call(1, "bash", {"command": "ls"}),
                call(2, "write", {"file_path": "a.md", "content": "line\nline"}),
            ],
        },
        tool(1, "bash", "a"),
        tool(2, "write", "ok"),
    ],
    "after a final answer, another task": [
        SYSTEM,
        {"role": "user", "content": "First."},
        DONE,
        {"role": "user", "content": "Résumé — 合同 ✓"},
    ],
}
OPTIONS = [
    {"enable_thinking": thinking, "truncate_history_thinking": truncate, "medium_effort": effort}
    for thinking, truncate, effort in itertools.product((True, False), repeat=3)
]


def option_id(options):
    return "-".join(k for k, v in options.items() if v) or "none"


def template_text(tokenizer, messages, options):
    return tokenizer.apply_chat_template(
        messages, tools=TOOLS, tokenize=False, add_generation_prompt=True, **options
    )


def template_ids(tokenizer, messages, options):
    return tokenizer.encode(template_text(tokenizer, messages, options), add_special_tokens=False)


def cookbook(model, options):
    """The renderer the cookbook selects for the model, with no ``[model] renderer``."""
    name, tokenizer = model
    return create_renderer(name, tokenizer=tokenizer, chat_template_kwargs=options)


def test_the_cookbook_renderer_is_selected_without_a_renderer_setting(model):
    assert describe(cookbook(model, {})) == TEXT_MODELS[model[0]][1]


@pytest.mark.parametrize("label", sorted(CONVERSATIONS))
@pytest.mark.parametrize("options", OPTIONS, ids=option_id)
def test_prompts_are_the_templates_tokens(model, label, options):
    messages = CONVERSATIONS[label]
    rendered = cookbook(model, options).render(messages, TOOLS)
    assert list(rendered.ids) == template_ids(model[1], messages, options)


def test_a_tool_call_parses_and_an_unknown_tool_is_reported(model):
    tokenizer = model[1]
    built = cookbook(model, {"medium_effort": True, "truncate_history_thinking": False})
    prompt = built.render([SYSTEM, TASK], TOOLS).ids
    good = (
        "Let me look.\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n"
        "</function>\n</tool_call><|im_end|>"
    )
    bad = good.replace("function=bash", "function=bsah")
    parsed = built.parse(prompt, tokenizer.encode(good, add_special_tokens=False), TOOLS)
    [ok] = parsed.tool_calls
    assert (ok.name, ok.arguments, ok.status.value) == ("bash", {"command": "ls"}, "ok")
    assert parsed.native_message is not None
    parsed = built.parse(prompt, tokenizer.encode(bad, add_special_tokens=False), TOOLS)
    [unknown] = parsed.tool_calls
    assert unknown.name == "bsah" and unknown.status.value != "ok"
    assert parsed.native_message is None


def test_reasoning_tokens_are_counted_on_the_sampled_tokens(model):
    """The tokens before ``</think>``, and all of them for a reply cut while reasoning."""
    tokenizer = model[1]
    built = cookbook(model, {"medium_effort": True, "truncate_history_thinking": False})
    prompt = built.render([SYSTEM, TASK], TOOLS).ids
    reasoning = tokenizer.encode(
        "The memo needs the indemnity terms first.", add_special_tokens=False
    )
    closed = [*reasoning, *tokenizer.encode("</think>\nDone.<|im_end|>", add_special_tokens=False)]
    assert built.reasoning_tokens(closed, built.parse(prompt, closed, TOOLS)) == len(reasoning)
    cut = built.parse(prompt, reasoning, TOOLS)
    assert built.reasoning_tokens(reasoning, cut) == len(reasoning)


def sampled_reply(tokenizer, full, prompt):
    """The reply after ``prompt`` as the model samples it: its own tokens, which need not
    be how a re-render tokenizes the same text."""
    whole, asked = tokenizer.decode(full), tokenizer.decode(prompt)
    assert whole.startswith(asked)
    reply = whole[len(asked) : whole.index("<|im_end|>", len(asked)) + len("<|im_end|>")]
    return tokenizer.encode(reply, add_special_tokens=False)


@pytest.mark.parametrize("label", ["a tool cycle", "two calls, nested arguments"])
@pytest.mark.parametrize("effort", [False, True], ids=["no-effort", "medium-effort"])
@pytest.mark.parametrize("truncate", [True, False], ids=["truncate", "keep-thinking"])
def test_tool_results_bridge_to_the_templates_tokens(model, label, effort, truncate):
    """Prime's Ultra renderer declines this bridge under ``medium_effort``."""
    tokenizer = model[1]
    options = {
        "enable_thinking": True,
        "truncate_history_thinking": truncate,
        "medium_effort": effort,
    }
    messages = CONVERSATIONS[label]
    turn = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
    renderer = cookbook(model, options)
    first = renderer.render(messages[:turn], TOOLS)
    full = template_ids(tokenizer, messages, options)
    completion = sampled_reply(tokenizer, full, first.ids)
    bridged = renderer.bridge(
        first.ids,
        completion,
        messages[turn + 1 :],
        TOOLS,
        previous_messages=messages[: turn + 1],
        previous_prompt_input=first.prompt_input,
    )
    assert bridged is not None
    [im_end] = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    rest = full[full.index(im_end, len(first.ids)) + 1 :]
    assert list(bridged.ids) == [*first.ids, *completion, *rest]


# --- Super VL images ---------------------------------------------------------------

WIDE = lettered_png((300, 200), label="W")
SMALL = lettered_png((64, 48), label="S")
IMAGE_CONVERSATIONS = {
    "text then an image": [SYSTEM, user(text("Read this:"), image(WIDE))],
    "text ending in a newline, then an image": [SYSTEM, user(text("Read this:\n"), image(WIDE))],
    "an image then text": [SYSTEM, user(image(SMALL), text("What is it?"))],
    "two images among text": [
        SYSTEM,
        user(text("Compare"), image(WIDE), text("with"), image(SMALL)),
    ],
    "text that places its image": [SYSTEM, user(text("Here: <image> what is it?"), image(SMALL))],
    "an image in a tool result": [
        SYSTEM,
        TASK,
        LOOK,
        tool(1, "bash", [text("screenshot"), image(WIDE)]),
    ],
    "images in two tasks": [
        SYSTEM,
        user(image(SMALL), text("First.")),
        DONE,
        user(text("Then"), image(WIDE)),
    ],
}


def images_of(messages):
    return [
        part["image"]
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image"
    ]


def processor_ids(tokenizer, processor, messages, options):
    """The template's text, expanded by the checkpoint's own processor."""
    images = images_of(messages)
    if not images:
        return template_ids(tokenizer, messages, options)
    encoded = processor(
        text=[template_text(tokenizer, messages, options)],
        images=[pil(raw) for raw in images],
    )
    return list(encoded["input_ids"][0])


def image_runs(tokenizer, ids):
    image_id = tokenizer.convert_tokens_to_ids("<image>")
    runs, i = [], 0
    while i < len(ids):
        if ids[i] == image_id:
            start = i
            while i < len(ids) and ids[i] == image_id:
                i += 1
            runs.append((start, i - start))
        else:
            i += 1
    return runs


def assert_rendered_as_processor(tokenizer, rendered, expected, messages):
    """Same tokens, an image chunk on each placeholder run, the images in order."""
    assert list(rendered.ids) == expected
    assert image_placeholders(rendered.prompt_input) == image_runs(tokenizer, expected)
    chunks = [c for c in rendered.prompt_input.chunks if isinstance(c, ImageChunk)]
    assert [c.data for c in chunks] == images_of(messages)


@pytest.mark.parametrize("label", sorted(IMAGE_CONVERSATIONS))
@pytest.mark.parametrize("options", OPTIONS, ids=option_id)
def test_image_prompts_are_the_processors_tokens(
    super_vl_tokenizer, super_vl_processor, label, options
):
    messages = IMAGE_CONVERSATIONS[label]
    renderer = super_vl(super_vl_tokenizer, options, processor=super_vl_processor)
    rendered = renderer.render(messages, TOOLS)
    expected = processor_ids(super_vl_tokenizer, super_vl_processor, messages, options)
    assert_rendered_as_processor(super_vl_tokenizer, rendered, expected, messages)


BRIDGES = {
    "a text tool result after a text task": CONVERSATIONS["a tool cycle"],
    "a text tool result after an image": [
        SYSTEM,
        user(text("Read this:"), image(WIDE)),
        LOOK,
        tool(1, "bash", "a.docx"),
    ],
    "an image in a tool result": IMAGE_CONVERSATIONS["an image in a tool result"],
    "a second task with an image": IMAGE_CONVERSATIONS["images in two tasks"],
}


def adds_user(messages):
    turn = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
    return any(m["role"] == "user" for m in messages[turn + 1 :])


def declines(messages, options):
    """A new user message moves the effort hint and, with ``truncate_history_thinking``,
    drops earlier reasoning: either way the rendered prompt is no longer a prefix."""
    return adds_user(messages) and (
        options["medium_effort"] or options["truncate_history_thinking"]
    )


def bridge_options(effort, truncate):
    return {"enable_thinking": True, "truncate_history_thinking": truncate, "medium_effort": effort}


BRIDGE_CASES = [
    pytest.param(
        label,
        bridge_options(effort, truncate),
        id=f"{label}-{'medium-effort' if effort else 'no-effort'}-"
        f"{'truncate' if truncate else 'keep-thinking'}",
    )
    for label in sorted(BRIDGES)
    for effort in (False, True)
    for truncate in (True, False)
]


def first_task(messages):
    """The messages before the first reply."""
    return messages[: next(i for i, m in enumerate(messages) if m["role"] == "assistant")]


def bridge_after_first_reply(renderer, messages, reply):
    turn = len(first_task(messages))
    first = renderer.render(messages[:turn], TOOLS)
    return first, renderer.bridge(
        first.ids,
        reply(first),
        messages[turn + 1 :],
        TOOLS,
        previous_messages=messages[: turn + 1],
        previous_prompt_input=first.prompt_input,
    )


@pytest.mark.parametrize(
    "label,options", [c for c in BRIDGE_CASES if not declines(BRIDGES[c.values[0]], c.values[1])]
)
def test_image_conversations_bridge_to_the_processors_tokens(
    super_vl_tokenizer, super_vl_processor, label, options
):
    """Prime's Nemotron 3.5 renderer declines all of these once images are possible."""
    tokenizer = super_vl_tokenizer
    messages = BRIDGES[label]
    renderer = super_vl(tokenizer, options, processor=super_vl_processor)
    full = processor_ids(tokenizer, super_vl_processor, messages, options)
    first, bridged = bridge_after_first_reply(
        renderer, messages, lambda first: sampled_reply(tokenizer, full, first.ids)
    )
    assert bridged is not None
    completion = sampled_reply(tokenizer, full, first.ids)
    [im_end] = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    rest = full[full.index(im_end, len(first.ids)) + 1 :]
    assert_rendered_as_processor(tokenizer, bridged, [*first.ids, *completion, *rest], messages)


@pytest.mark.parametrize(
    "label,options", [c for c in BRIDGE_CASES if declines(BRIDGES[c.values[0]], c.values[1])]
)
def test_a_new_task_that_changes_the_history_renders_in_full(
    super_vl_tokenizer, super_vl_processor, label, options
):
    tokenizer = super_vl_tokenizer
    renderer = super_vl(tokenizer, options, processor=super_vl_processor)
    reply = tokenizer.encode("Easy.</think>Done.<|im_end|>", add_special_tokens=False)
    _, bridged = bridge_after_first_reply(renderer, BRIDGES[label], lambda first: reply)
    assert bridged is None


def test_a_reply_prime_trims_declines_rather_than_misplacing_images(
    super_vl_tokenizer, super_vl_processor
):
    """Prime cuts a reply's tokens after its turn closes, so the bridged prompt does not
    extend the sampled tokens; image placeholders must not be searched for in it."""
    tokenizer = super_vl_tokenizer
    renderer = super_vl(tokenizer, bridge_options(False, False), processor=super_vl_processor)
    trailing = "Plan.</think>Let me look.<|im_end|>" + "<image>" * 40
    reply = tokenizer.encode(trailing, add_special_tokens=False)
    messages = BRIDGES["an image in a tool result"]
    _, bridged = bridge_after_first_reply(renderer, messages, lambda first: reply)
    assert bridged is None


@pytest.mark.parametrize(
    "content", [["look", image(SMALL)], ["look"]], ids=["beside-an-image", "text-only"]
)
def test_a_plain_string_part_is_refused_not_dropped(super_vl_tokenizer, content):
    """The template renders a plain string part in a content list as nothing."""
    renderer = super_vl(super_vl_tokenizer, {})
    with pytest.raises(RenderingError, match="plain string"):
        renderer.render([{"role": "user", "content": content}], TOOLS)


PAGE = 'An <image> tag, an <img> tag and </img>, beside <img src="a.png">.'
WRITTEN_MARKERS = {
    "in a text conversation": [
        SYSTEM,
        TASK,
        LOOK,
        tool(1, "bash", PAGE),
    ],
    "before an image": [
        SYSTEM,
        TASK,
        {
            "role": "assistant",
            "content": "Let me look.",
            "reasoning_content": "Read the page, then take a screenshot.",
            "tool_calls": [
                call(1, "bash", {"command": "cat page.html"}),
                call(2, "bash", {"command": "shot"}),
            ],
        },
        tool(1, "bash", PAGE),
        tool(2, "bash", [text("screenshot"), image(WIDE)]),
    ],
}


def assert_markers_only_frame_images(tokenizer, ids, runs):
    """Every ``<image>``, ``<img>`` and ``</img>`` token belongs to an image's frame."""
    markers = set(tokenizer.convert_tokens_to_ids(["<image>", "<img>", "</img>"]))
    framed = {k for start, length in runs for k in range(start - 1, start + length + 1)}
    assert [k for k, token in enumerate(ids) if token in markers and k not in framed] == []


@pytest.mark.parametrize("label", sorted(WRITTEN_MARKERS))
def test_image_markers_written_in_text_stay_text(super_vl_tokenizer, super_vl_processor, label):
    """The tokenizer reads ``<image>``, ``<img>`` and ``</img>`` in text as image markers,
    and the sampler takes image positions only from images. Written in a tool result,
    they are kept as the characters written, in a full render and in a bridge."""
    tokenizer = super_vl_tokenizer
    messages = WRITTEN_MARKERS[label]
    options = bridge_options(False, False)
    renderer = super_vl(tokenizer, options, processor=super_vl_processor)
    full = renderer.render(messages, TOOLS)
    runs = image_placeholders(full.prompt_input)
    assert len(runs) == len(images_of(messages))
    assert_markers_only_frame_images(tokenizer, full.ids, runs)
    if not runs:
        assert tokenizer.decode(full.ids) == template_text(tokenizer, messages, options)
    _, bridged = bridge_after_first_reply(
        renderer, messages, lambda first: sampled_reply(tokenizer, full.ids, first.ids)
    )
    assert bridged is not None
    assert list(bridged.ids) == list(full.ids)
    assert image_placeholders(bridged.prompt_input) == runs


def test_text_written_as_an_image_frame_is_not_taken_for_the_image(
    super_vl_tokenizer, super_vl_processor
):
    """A tool result that writes ``<img>``, the next image's count of ``<image>`` and
    ``</img>`` matches the image's frame token for token. The image is still the one in
    the message that holds it, and the written frame stays text."""
    tokenizer = super_vl_tokenizer
    options = bridge_options(False, False)
    renderer = super_vl(tokenizer, options, processor=super_vl_processor)
    [(_, count)] = image_placeholders(
        renderer.render(WRITTEN_MARKERS["before an image"], TOOLS).prompt_input
    )
    messages = [
        *WRITTEN_MARKERS["before an image"][:3],
        tool(1, "bash", "<img>" + "<image>" * count + "</img>"),
        WRITTEN_MARKERS["before an image"][4],
    ]
    full = renderer.render(messages, TOOLS)
    [(start, length)] = image_placeholders(full.prompt_input)
    assert length == count
    assert tokenizer.decode(full.ids[:start]).endswith("screenshot\n<img>")
    assert_markers_only_frame_images(tokenizer, full.ids, [(start, length)])
    _, bridged = bridge_after_first_reply(
        renderer, messages, lambda first: sampled_reply(tokenizer, full.ids, first.ids)
    )
    assert bridged is not None
    assert list(bridged.ids) == list(full.ids)
    assert image_placeholders(bridged.prompt_input) == [(start, length)]


class CountingProcessor:
    """A processor that counts the images it is asked to process."""

    def __init__(self, processor):
        self.processor, self.images = processor, 0

    def image_processor(self, **kwargs):
        self.images += len(kwargs["images"])
        return self.processor.image_processor(**kwargs)


@pytest.mark.parametrize(
    "label,options",
    [
        c
        for c in BRIDGE_CASES
        if c.values[1]["medium_effort"] and declines(BRIDGES[c.values[0]], c.values[1])
    ],
)
def test_a_bridge_that_must_decline_processes_no_images(
    super_vl_tokenizer, super_vl_processor, label, options
):
    """A new task under ``medium_effort`` renders in full, which processes its images."""
    tokenizer = super_vl_tokenizer
    processor = CountingProcessor(super_vl_processor)
    renderer = super_vl(tokenizer, options, processor=processor)
    reply = tokenizer.encode("Easy.</think>Done.<|im_end|>", add_special_tokens=False)
    messages = BRIDGES[label]
    _, bridged = bridge_after_first_reply(renderer, messages, lambda first: reply)
    assert bridged is None
    assert processor.images == len(images_of(first_task(messages)))


def test_a_model_name_in_another_case_loads_its_processor(
    super_vl_tokenizer, super_vl_processor, monkeypatch
):
    """The cookbook selects the renderer whatever the name's case, as the Hub resolves it."""
    import transformers

    loaded = []

    def from_pretrained(name, **kwargs):
        loaded.append((name, kwargs))
        return super_vl_processor

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", from_pretrained)
    tokenizer = copy.copy(super_vl_tokenizer)
    tokenizer.name_or_path = NEMOTRON_SUPER_VL.lower()
    messages = IMAGE_CONVERSATIONS["text then an image"]
    rendered = super_vl(tokenizer, {}).render(messages, TOOLS)
    expected = super_vl(super_vl_tokenizer, {}, processor=super_vl_processor).render(
        messages, TOOLS
    )
    assert loaded == [(NEMOTRON_SUPER_VL.lower(), {"trust_remote_code": True})]
    assert list(rendered.ids) == list(expected.ids)
    assert image_placeholders(rendered.prompt_input) == image_placeholders(expected.prompt_input)


@pytest.mark.parametrize("name", sorted(NEMOTRON_SUPER_VL_REVISIONS))
def test_each_super_vl_checkpoint_gets_the_cookbook_renderer(name):
    tokenizer = pinned_tokenizer(
        name, NEMOTRON_SUPER_VL_REVISIONS[name], "LOOPS_NEMOTRON_TOKENIZER_CACHE"
    )
    assert describe(create_renderer(name, tokenizer=tokenizer)) == "SuperVLRenderer"
