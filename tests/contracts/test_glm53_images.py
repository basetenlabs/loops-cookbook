"""GLM-5.3-Flash image rendering against the real Glm5Next processor.

Renderer-side contract only: ids, placeholder spans and chunk bytes. What the sampler
and trainer do with those chunks is exercised in the unit suite with fakes.
"""

import random
from copy import deepcopy

import pytest
from support.glm53_histories import ASSISTANT, BASE, TOOLS
from support.images import letter_png, lettered_png, pil
from support.unit_rest import jpeg

from loops_cookbook.llm.images import image_placeholders

GOLDEN_LETTER_DIGEST = "8575a10c50731b6680cbd8016409fadeebd43de2b05139495422b096b03f5500"


def _hf_ids(processor, messages, images, *, tools=None, **template_kwargs):
    """The processor's own expansion: chat template text, then image placeholders."""
    text = processor.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    encoded = processor(text=[text], images=images or None, return_tensors="np")
    return encoded["input_ids"][0].tolist()


def _image_id(flash_tokenizer):
    return flash_tokenizer.convert_tokens_to_ids("<|image|>")


def _spans_from_ids(ids, image_id):
    spans, i = [], 0
    while i < len(ids):
        if ids[i] == image_id:
            start = i
            while i < len(ids) and ids[i] == image_id:
                i += 1
            spans.append((start, i - start))
        else:
            i += 1
    return spans


def _user(*parts):
    return {"role": "user", "content": list(parts)}


def _text(text):
    return {"type": "text", "text": text}


def _image(raw):
    return {"type": "image", "image": raw}


def _content_parts(message):
    content = message.get("content")
    return content if isinstance(content, list) else []


def _hf_view(message):
    """The same message with image bytes swapped for PIL images, as HF's template wants."""
    content = message.get("content")
    if not isinstance(content, list):
        return message
    return {
        **message,
        "content": [_image(pil(p["image"])) if p.get("type") == "image" else p for p in content],
    }


def _hf_images_in_text_order(processor, text, history):
    """Images in the order the HF template emitted their messages. Messages here carry
    unique text markers, so their position in ``text`` gives the rendered order."""
    placed = []
    for message in history:
        images = [pil(p["image"]) for p in _content_parts(message) if p.get("type") == "image"]
        if not images:
            continue
        marker = next(p["text"] for p in _content_parts(message) if p.get("type") == "text")
        assert text.count(marker) == 1, marker
        placed.append((text.index(marker), images))
    return [image for _, images in sorted(placed, key=lambda item: item[0]) for image in images]


def test_flash_bridge_carries_earlier_image_and_adds_a_tool_image(flash_tokenizer, flash_renderer):
    picture = letter_png()
    renderer = flash_renderer()
    first = renderer.render(
        [
            {
                "role": "user",
                "content": [{"type": "image", "image": picture}, {"type": "text", "text": "Look"}],
            }
        ],
        TOOLS,
    )
    completion = tuple(
        flash_tokenizer.encode(
            "plan</think>ok<tool_call>f<arg_key>value</arg_key><arg_value>1</arg_value></tool_call><|observation|>",
            add_special_tokens=False,
        )
    )
    second = renderer.bridge(
        first.ids,
        completion,
        [
            {
                "role": "tool",
                "content": [{"type": "image", "image": picture}, {"type": "text", "text": "seen"}],
            }
        ],
        TOOLS,
        previous_prompt_input=first.prompt_input,
    )
    assert second is not None
    assert second.ids[: len(first.ids) + len(completion)] == first.ids + completion
    spans = image_placeholders(second.prompt_input)
    assert len(spans) == 2 and spans[0] == image_placeholders(first.prompt_input)[0]
    assert spans[0][1] == spans[1][1] == 36
    assert second.prompt_input.length == len(second.ids)


@pytest.mark.parametrize(
    "size",
    [
        (28, 28),  # one patch-merge cell
        (160, 160),  # the letter.png test asset's geometry
        (640, 48),  # wide strip
        (48, 640),  # tall strip
        (333, 217),  # not divisible by patch * merge
        (1000, 1000),
        (2048, 1536),  # above the processor's max pixels: exercises the downscale
    ],
)
def test_flash_image_geometry_matches_processor(
    flash_tokenizer, flash_processor, size, flash_renderer
):
    raw = lettered_png(size)
    renderer = flash_renderer(reasoning_effort="low")
    ours = renderer.render([_user(_text("Describe"), _image(raw))])
    expected = _hf_ids(
        flash_processor,
        [_user(_text("Describe"), _image(pil(raw)))],
        [pil(raw)],
        reasoning_effort="low",
    )
    assert list(ours.ids) == expected
    ((offset, length),) = image_placeholders(ours.prompt_input)
    assert (offset, length) == _spans_from_ids(expected, _image_id(flash_tokenizer))[0]
    assert ours.prompt_input.chunks[1].expected_tokens == length


def test_flash_image_input_shapes_agree(flash_renderer):
    """Every accepted part shape renders the same ids and carries decodable bytes."""
    import base64
    import io

    from PIL import Image

    png = lettered_png((160, 160))
    as_jpeg = jpeg(png)
    shapes = {
        "png bytes": _image(png),
        "jpeg bytes": _image(as_jpeg),
        "rgba png bytes": _image(lettered_png((160, 160), mode="RGBA")),
        "palette png bytes": _image(lettered_png((160, 160), mode="P")),
        "bytearray": _image(bytearray(png)),
        "png data uri": _image("data:image/png;base64," + base64.b64encode(png).decode()),
        "image_url dict": {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(as_jpeg).decode()},
        },
        "pil rgb": _image(pil(png)),
        "pil rgba": _image(Image.open(io.BytesIO(png)).convert("RGBA")),
    }
    renderer = flash_renderer()
    reference = renderer.render([_user(_text("Look"), _image(png))])
    for name, part in shapes.items():
        rendered = renderer.render([_user(_text("Look"), part)])
        assert rendered.ids == reference.ids, name
        assert image_placeholders(rendered.prompt_input) == image_placeholders(
            reference.prompt_input
        ), name
        chunk = rendered.prompt_input.chunks[1]
        with Image.open(io.BytesIO(chunk.data)) as decoded:
            assert decoded.size == (160, 160), name
            assert decoded.format.lower() in {"png", "jpeg"}, name
        assert chunk.format == (decoded.format or "").lower(), name
    # Encoded payloads pass through byte-for-byte; the trainer sees the user's file.
    assert renderer.render([_user(_image(png))]).prompt_input.chunks[1].data == png
    assert renderer.render([_user(_image(as_jpeg))]).prompt_input.chunks[1].data == as_jpeg


@pytest.mark.parametrize("size", [(160, 160), (333, 217), (2048, 1536)])
@pytest.mark.parametrize("encoding", ["png", "jpeg", "pil"])
def test_flash_expected_tokens_survive_the_sampler_decode(
    flash_processor, size, encoding, flash_renderer
):
    """The trainer recounts tokens from the bytes in the chunk after the sampler's
    decode + RGB convert + PNG re-encode round trip. That count must be ours."""
    import io

    from PIL import Image

    raw = lettered_png(size)
    if encoding == "jpeg":
        part = _image(jpeg(raw))
    elif encoding == "pil":
        part = _image(pil(raw))
    else:
        part = _image(raw)
    renderer = flash_renderer()
    chunk = renderer.render([_user(_text("Look"), part)]).prompt_input.chunks[1]
    # sampler/vlm_prompt.py: Image.open(...).convert("RGB"), then .save(format="PNG")
    sampler_side = io.BytesIO()
    Image.open(io.BytesIO(chunk.data)).convert("RGB").save(sampler_side, format="PNG")
    trainer_view = Image.open(io.BytesIO(sampler_side.getvalue())).convert("RGB")
    grid = flash_processor.image_processor(images=[trainer_view], return_tensors="np")[
        "image_grid_thw"
    ][0]
    merge = flash_processor.image_processor.merge_size
    assert int(grid.prod()) // (merge * merge) == chunk.expected_tokens


MULTI_IMAGE_CASES = {
    "two images, text between": lambda a, b: [
        _user(_text("First"), _image(a), _text("second"), _image(b), _text("?"))
    ],
    "adjacent images": lambda a, b: [_user(_image(a), _image(b))],
    "same image twice": lambda a, b: [_user(_image(a), _text("and again"), _image(a))],
    "images across two user turns": lambda a, b: [
        _user(_text("One"), _image(a)),
        {"role": "assistant", "content": "seen"},
        _user(_text("Two"), _image(b)),
    ],
    "image returned by a tool": lambda a, b: [
        *BASE,
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "look",
            "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": {"value": 1}}}],
        },
        {"role": "tool", "tool_call_id": "a", "content": [_image(a), _text("screenshot")]},
    ],
    "images in user and tool turns": lambda a, b: [
        _user(_text("Start"), _image(a)),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": {"value": 1}}}],
        },
        {"role": "tool", "tool_call_id": "a", "content": [_text("got"), _image(b)]},
    ],
}


@pytest.mark.parametrize("case", sorted(MULTI_IMAGE_CASES))
@pytest.mark.parametrize("effort", ["low", "max"])
@pytest.mark.parametrize("with_tools", [False, True])
def test_flash_multi_image_parity(
    flash_tokenizer, flash_processor, flash_renderer, case, effort, with_tools
):
    a, b = lettered_png((160, 160), label="A"), lettered_png((333, 217), label="B")
    history = MULTI_IMAGE_CASES[case](a, b)
    tools = TOOLS if with_tools else ()
    renderer = flash_renderer(reasoning_effort=effort)
    ours = renderer.render(history, tools)

    hf_history = [_hf_view(m) for m in history]
    images = [
        pil(p["image"]) for m in history for p in _content_parts(m) if p.get("type") == "image"
    ]
    expected = _hf_ids(
        flash_processor, hf_history, images, tools=list(tools) or None, reasoning_effort=effort
    )
    assert list(ours.ids) == expected, case
    spans = image_placeholders(ours.prompt_input)
    assert spans == _spans_from_ids(expected, _image_id(flash_tokenizer)), case
    assert len(spans) == len(images), case
    assert all(spans[i][0] + spans[i][1] < spans[i + 1][0] for i in range(len(spans) - 1)), case
    image_chunks = [c for c in ours.prompt_input.chunks if c.type == "image"]
    assert [c.expected_tokens for c in image_chunks] == [length for _, length in spans]
    assert ours.prompt_input.length == len(ours.ids)


@pytest.mark.parametrize("seed", range(4))
def test_flash_bridge_matches_full_render_with_images(flash_tokenizer, flash_renderer, seed):
    """Multi-turn agentic history with images in user and tool turns: every bridged
    continuation must equal the full render, ids and image spans alike."""
    rng = random.Random(seed)
    pictures = [
        lettered_png((160, 160), label="A"),
        lettered_png((333, 217), label="B"),
        lettered_png((48, 640), label="C"),
    ]
    renderer = flash_renderer(reasoning_effort="low")
    history = deepcopy(BASE) + [_user(_text("Begin"), _image(rng.choice(pictures)))]
    current = renderer.render(history, TOOLS)
    for turn in range(8):
        if rng.random() < 0.6:
            completion = flash_tokenizer.encode(
                f"plan {turn}</think>ok<tool_call>f<arg_key>value</arg_key>"
                f"<arg_value>{turn}</arg_value></tool_call><|observation|>",
                add_special_tokens=False,
            )
            parsed = renderer.parse(current.ids, completion, TOOLS)
            new_messages = [
                {
                    "role": "tool",
                    "tool_call_id": parsed.tool_calls[0].id,
                    "content": rng.choice(
                        [
                            [_image(rng.choice(pictures)), _text("screenshot")],
                            [_text("no image this time")],
                            [_image(rng.choice(pictures)), _image(rng.choice(pictures))],
                        ]
                    ),
                }
            ]
        else:
            completion = flash_tokenizer.encode(
                f"plan {turn}</think>answer {turn}<|user|>", add_special_tokens=False
            )
            parsed = renderer.parse(current.ids, completion, TOOLS)
            new_messages = [_user(_text(f"next {turn}"), _image(rng.choice(pictures)))]
        history = history + [parsed.native_message] + new_messages
        bridged = renderer.bridge(
            current.ids,
            tuple(completion),
            new_messages,
            TOOLS,
            previous_prompt_input=current.prompt_input,
        )
        assert bridged is not None, (seed, turn)
        assert bridged.ids[: len(current.ids) + len(completion)] == current.ids + tuple(completion)
        full = renderer.render(history, TOOLS)
        # Whitespace normalisation aside (see the text bridge test), the image spans must
        # be identical in count and length, and the bridged tail must equal the full tail.
        tail_len = len(bridged.ids) - len(current.ids) - len(completion)
        assert bridged.ids[-tail_len:] == full.ids[-tail_len:], (seed, turn)
        assert [n for _, n in image_placeholders(bridged.prompt_input)] == [
            n for _, n in image_placeholders(full.prompt_input)
        ], (seed, turn)
        assert bridged.prompt_input.length == len(bridged.ids)
        expected_images = sum(
            1 for m in history for p in _content_parts(m) if p.get("type") == "image"
        )
        assert len(image_placeholders(bridged.prompt_input)) == expected_images
        current = bridged


def test_flash_stray_image_token_in_completion_fails_loudly(flash_tokenizer, flash_renderer):
    """A sampled ``<|image|>`` must not be mistaken for an image the caller supplied."""
    from loops_cookbook.llm.rendering import RenderingError

    renderer = flash_renderer()
    first = renderer.render([_user(_text("Look"), _image(lettered_png((160, 160))))])
    completion = tuple(
        flash_tokenizer.encode("plan</think>answer <|image|><|user|>", add_special_tokens=False)
    )
    with pytest.raises(RenderingError):
        renderer.bridge(
            first.ids,
            completion,
            [_user(_text("next"))],
            previous_prompt_input=first.prompt_input,
        )


@pytest.mark.parametrize(
    ("part", "error"),
    [
        ({"type": "video", "video": b"\x00"}, "UnsupportedContextError"),
        (
            {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}},
            "UnsupportedContextError",
        ),
        ({"type": "image", "image": b""}, "RenderingError"),
        ({"type": "image", "image": b"\x89PNG\r\n\x1a\njunk"}, "RenderingError"),
        ({"type": "image", "image": "data:image/png;base64,@@@"}, "RenderingError"),
        ({"type": "image"}, "RenderingError"),
    ],
)
def test_flash_rejects_unusable_parts(flash_renderer, part, error):
    from loops_cookbook.llm import rendering

    renderer = flash_renderer()
    with pytest.raises(getattr(rendering, error)):
        renderer.render([_user(_text("Look"), part)])


def test_flash_bundled_tool_outputs_with_images_match_hf(
    flash_tokenizer, flash_processor, flash_renderer
):
    """GLM's template renders images inside bundled tool ``output`` entries; so do we."""
    small, wide = lettered_png((28, 28), label="S"), lettered_png((333, 217), label="W")
    history = [
        *BASE,
        ASSISTANT,
        {
            "role": "tool",
            "content": [
                {"tool_call_id": "a", "output": [_image(small), _text("one")]},
                {"tool_call_id": "b", "output": "two"},
                {"tool_call_id": "c", "output": [_text("three"), _image(wide)]},
            ],
        },
    ]
    renderer = flash_renderer(reasoning_effort="low")
    ours = renderer.render(history, TOOLS)
    hf_history = [
        *BASE,
        ASSISTANT,
        {
            "role": "tool",
            "content": [
                {"tool_call_id": "a", "output": [_image(pil(small)), _text("one")]},
                {"tool_call_id": "b", "output": "two"},
                {"tool_call_id": "c", "output": [_text("three"), _image(pil(wide))]},
            ],
        },
    ]
    expected = _hf_ids(
        flash_processor, hf_history, [pil(small), pil(wide)], tools=TOOLS, reasoning_effort="low"
    )
    assert list(ours.ids) == expected
    spans = image_placeholders(ours.prompt_input)
    assert spans == _spans_from_ids(expected, _image_id(flash_tokenizer))
    chunks = [c for c in ours.prompt_input.chunks if c.type == "image"]
    assert [(c.expected_tokens, c.data) for c in chunks] == [(16, small), (96, wide)]

    # Same bundled shape arriving through the bridge after a sampled tool-call turn.
    first = renderer.render(BASE, TOOLS)
    completion = tuple(
        flash_tokenizer.encode(
            "plan</think><tool_call>f<arg_key>value</arg_key><arg_value>1</arg_value></tool_call>"
            "<tool_call>f<arg_key>value</arg_key><arg_value>2</arg_value></tool_call><|observation|>",
            add_special_tokens=False,
        )
    )
    parsed = renderer.parse(first.ids, completion, TOOLS)
    bundled = {
        "role": "tool",
        "content": [
            {"output": [_image(wide), _text("first")]},
            {"output": [_text("second"), _image(small)]},
        ],
    }
    bridged = renderer.bridge(
        first.ids,
        completion,
        [bundled],
        TOOLS,
        previous_messages=[*BASE, parsed.native_message],
        previous_prompt_input=first.prompt_input,
    )
    assert bridged is not None
    full = renderer.render([*BASE, parsed.native_message, bundled], TOOLS)
    tail = len(bridged.ids) - len(first.ids) - len(completion)
    assert bridged.ids[-tail:] == full.ids[-tail:]
    assert [
        (c.expected_tokens, c.data) for c in bridged.prompt_input.chunks if c.type == "image"
    ] == [
        (96, wide),
        (16, small),
    ]


def test_flash_multimodal_contract_surface(flash_tokenizer, flash_renderer):
    import renderers.base

    renderer = flash_renderer()
    inner = renderer._renderer
    assert renderers.base.is_multimodal(inner)
    assert renderer.supports_images
    assert inner.mm_token_type_id_map == {_image_id(flash_tokenizer): 1}
    text_only = renderer.render(BASE, TOOLS)
    assert [c.type for c in text_only.prompt_input.chunks] == ["encoded_text"]
    assert text_only.prompt_input.to_ints() == list(text_only.ids)


def test_flash_letter_png_golden(flash_tokenizer, flash_renderer):
    """Pinned ids for the letter.png test asset. A transformers or hub-revision change
    that alters the prompt the model sees fails here with a diff, not silently."""
    import hashlib
    import json

    renderer = flash_renderer(reasoning_effort="low")
    rendered = renderer.render([_user(_text("Which letter comes next?"), _image(letter_png()))])
    digest = hashlib.sha256(json.dumps(list(rendered.ids)).encode()).hexdigest()
    assert (len(rendered.ids), image_placeholders(rendered.prompt_input)) == (55, [(16, 36)])
    assert digest == GOLDEN_LETTER_DIGEST, flash_tokenizer.decode(rendered.ids)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("clear", [False, True])
def test_flash_randomized_image_histories_match_hf(
    flash_tokenizer, flash_processor, flash_renderer, seed, clear
):
    """The text fuzz, with images: random tool loops where user and tool turns carry
    zero to two images of varying geometry, compared with the HF processor every turn."""
    rng = random.Random(seed)
    effort = ("low", "high", "max")[seed % 3]
    renderer = flash_renderer(reasoning_effort=effort, clear_thinking=clear)
    pictures = [
        lettered_png((160, 160), label="A"),
        lettered_png((333, 217), label="B"),
        lettered_png((48, 640), label="C"),
        lettered_png((28, 28), label="D"),
    ]

    def parts(text):
        out = [_text(text)]
        for _ in range(rng.choice([0, 0, 1, 1, 2])):
            out.insert(rng.randint(0, len(out)), _image(rng.choice(pictures)))
        return out

    history = deepcopy(BASE) + [_user(*parts("Begin"))]
    tools = deepcopy(TOOLS)
    for turn in range(12):
        calls = [
            {
                "id": f"{turn}-{i}",
                "function": {"name": "f", "arguments": {"value": rng.choice([1, "x", None])}},
            }
            for i in range(rng.randint(1, 3))
        ]
        history.append(
            {
                "role": "assistant",
                "content": rng.choice(["", " answer \n"]),
                "reasoning_content": rng.choice(["", " plan\n"]),
                "tool_calls": calls,
            }
        )
        observations = [
            {"role": "tool", "tool_call_id": call["id"], "content": parts(f"result {turn}-{i}\n")}
            for i, call in enumerate(calls)
        ]
        rng.shuffle(observations)  # the template sorts these back into call order
        history.extend(observations)
        if turn % 4 == 3:
            history.append(_user(*parts(f"next {turn}")))
        ours = renderer.render(history, tools)
        text = flash_processor.apply_chat_template(
            [_hf_view(m) for m in history],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=effort,
            clear_thinking=clear,
        )
        images = _hf_images_in_text_order(flash_processor, text, history)
        expected = flash_processor(text=[text], images=images, return_tensors="np")["input_ids"][
            0
        ].tolist()
        assert list(ours.ids) == expected, (seed, clear, turn)
        spans = image_placeholders(ours.prompt_input)
        assert spans == _spans_from_ids(expected, _image_id(flash_tokenizer)), (seed, turn)
        assert len(spans) == len(images) and ours.prompt_input.length == len(ours.ids)
        # Each chunk carries the bytes of the image that produced its span, not the
        # image at the same position in history order.
        for chunk in ours.prompt_input.chunks:
            if chunk.type == "image":
                grid = flash_processor.image_processor(
                    images=[pil(chunk.data)], return_tensors="np"
                )
                merge = flash_processor.image_processor.merge_size
                assert (
                    int(grid["image_grid_thw"][0].prod()) // (merge * merge)
                    == chunk.expected_tokens
                )


def test_flash_shuffled_tool_results_keep_image_bytes_on_their_own_spans(
    flash_tokenizer, flash_processor, flash_renderer
):
    """GLM sorts tool results into call order. The image bytes must follow their span."""
    small, wide = lettered_png((28, 28), label="S"), lettered_png((333, 217), label="W")
    calls = [
        {"id": "first", "function": {"name": "f", "arguments": {"value": 1}}},
        {"id": "second", "function": {"name": "f", "arguments": {"value": 2}}},
    ]
    assistant = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": calls}
    # Results arrive in reverse call order; "second" (wide image) is listed first.
    results = [
        {"role": "tool", "tool_call_id": "second", "content": [_image(wide), _text("second")]},
        {"role": "tool", "tool_call_id": "first", "content": [_image(small), _text("first")]},
    ]
    renderer = flash_renderer(reasoning_effort="low")
    rendered = renderer.render([*BASE, assistant, *results], TOOLS)
    chunks = [c for c in rendered.prompt_input.chunks if c.type == "image"]
    assert [c.expected_tokens for c in chunks] == [16, 96]  # rendered order: first, second
    assert [c.data for c in chunks] == [small, wide]
    text = flash_tokenizer.decode(rendered.ids)
    assert text.index("first</tool_response>") < text.index("second</tool_response>")

    # Through the bridge, GLM tool calls carry no ids, so results cannot be joined to
    # calls and keep the order given (the HF template does the same). Bytes must still
    # follow their own spans, and the bridge must equal a full render.
    first = renderer.render(BASE, TOOLS)
    completion = tuple(
        flash_tokenizer.encode(
            "plan</think><tool_call>f<arg_key>value</arg_key><arg_value>1</arg_value></tool_call>"
            "<tool_call>f<arg_key>value</arg_key><arg_value>2</arg_value></tool_call><|observation|>",
            add_special_tokens=False,
        )
    )
    parsed = renderer.parse(first.ids, completion, TOOLS)
    assert [call.id for call in parsed.tool_calls] == [None, None]
    observations = [
        {"role": "tool", "tool_call_id": "", "content": [_image(wide), _text("second")]},
        {"role": "tool", "tool_call_id": "", "content": [_image(small), _text("first")]},
    ]
    bridged = renderer.bridge(
        first.ids,
        completion,
        observations,
        TOOLS,
        previous_messages=[*BASE, parsed.native_message],
        previous_prompt_input=first.prompt_input,
    )
    assert bridged is not None
    chunks = [c for c in bridged.prompt_input.chunks if c.type == "image"]
    assert [c.expected_tokens for c in chunks] == [96, 16]
    assert [c.data for c in chunks] == [wide, small]
    full = renderer.render([*BASE, parsed.native_message, *observations], TOOLS)
    tail = len(bridged.ids) - len(first.ids) - len(completion)
    assert bridged.ids[-tail:] == full.ids[-tail:]
    assert [(c.expected_tokens, c.data) for c in chunks] == [
        (c.expected_tokens, c.data) for c in full.prompt_input.chunks if c.type == "image"
    ]


async def test_flash_transcript_names_each_tool_image_by_its_own_bytes(flash_renderer):
    """GLM renders tool results in call order; the transcript keeps the agent's order,
    and each image part is named by its own bytes, not by the chunk at its index."""
    import hashlib
    from types import SimpleNamespace

    from support.leases import lease

    from loops_cookbook.llm.client import Renderers
    from loops_cookbook.llm.transcript import thaw
    from loops_cookbook.model_access.lease import DirectModelAccess

    small, wide = lettered_png((28, 28), label="S"), lettered_png((333, 217), label="W")
    calls = [
        {"id": "first", "function": {"name": "f", "arguments": {"value": 1}}},
        {"id": "second", "function": {"name": "f", "arguments": {"value": 2}}},
    ]
    assistant = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": calls}
    results = [
        {"role": "tool", "tool_call_id": "second", "content": [_image(wide), _text("second")]},
        {"role": "tool", "tool_call_id": "first", "content": [_image(small), _text("first")]},
    ]

    class Sampler:
        """Answers without echoing: the real image token is the renderer's, not a pad."""

        def __init__(self):
            self.prompts = []

        async def sample_async(self, prompt, **kwargs):
            self.prompts.append(prompt)
            reply = SimpleNamespace(
                tokens=[5], logprobs=[-0.1], stop_reason="stop", sequence_id=None
            )
            return SimpleNamespace(prompt_token_ids=None, policy_version=0, sequences=[reply])

    sampler = Sampler()
    held = lease(sampler, transcripts=[])
    model = DirectModelAccess(held, Renderers(flash_renderer(reasoning_effort="low")))
    await model.llm().conversation().generate(messages=[*BASE, assistant, *results], tools=TOOLS)
    chunks = [c.data for c in sampler.prompts[0].chunks if c.type == "image"]
    assert chunks == [small, wide]  # rendered in call order

    transcript = (await held.seal()).transcript
    assert transcript is not None
    *_, second, first = thaw(transcript.turns[0].turn.messages)
    assert second["tool_call_id"] == "second" and first["tool_call_id"] == "first"
    digest = {raw: "sha256:" + hashlib.sha256(raw).hexdigest() for raw in (small, wide)}
    assert second["content"][0]["image"] == digest[wide]
    assert first["content"][0]["image"] == digest[small]
    assert {name: image.data for name, image in transcript.images.items()} == {
        digest[small]: small,
        digest[wide]: wide,
    }


def test_flash_renders_interleaved_across_threads_keep_their_own_images(
    flash_renderer, flash_processor
):
    """One renderer serves every rollout; a user runtime may render from worker threads.
    Two renders that overlap mid-render must each pair their placeholders with their
    own image, never the other thread's."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    renderer = flash_renderer()
    both_inside = threading.Barrier(2, timeout=30)
    image_processor = flash_processor.image_processor

    class Overlapping:
        """The real image processor, held until both renders are inside it."""

        merge_size = image_processor.merge_size

        def __call__(self, *args, **kwargs):
            both_inside.wait()
            return image_processor(*args, **kwargs)

    renderer._renderer._processor = SimpleNamespace(image_processor=Overlapping())
    first, second = lettered_png((64, 64), label="A"), lettered_png((64, 64), label="B")

    def render(picture):
        return renderer.render([{"role": "user", "content": [_image(picture), _text("Look")]}])

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(render, [first, second]))
    for picture, rendered in zip([first, second], results, strict=True):
        assert [c.data for c in rendered.prompt_input.chunks if c.type == "image"] == [picture]
