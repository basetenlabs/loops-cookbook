"""Image parts: Prime placeholders become Loops chunks; ids stay the training record."""

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
from baseten.loops.models import EncodedTextChunk, ImageChunk, ModelInput, SamplingParams
from PIL import Image
from support.audit import audit_run
from support.builders import make_rollout, make_sample
from support.fakes import PAD, EchoSampler, FakePrime, MemoryTrainer, ScriptedRenderer
from support.images import png
from support.leases import direct_llm, lease
from support.unit_rest import jpeg

from loops_cookbook.evidence import TrajectoryError
from loops_cookbook.llm.images import image_bytes, image_placeholders, model_input_from_placeholders
from loops_cookbook.llm.prime import PrimeRenderer
from loops_cookbook.llm.rendering import ParsedReply, RenderedPrompt, RenderingError
from loops_cookbook.model_access.api import ModelAccessMode, SamplingFailure
from loops_cookbook.rl.grpo import build_rows
from loops_cookbook.rl.run import Run, datum_record, redact_model_input, sample_record
from loops_cookbook.runtime import EpisodeResult, GradeResult, RuntimeCapabilities

PNG = png()


# --- chunk helpers ----------------------------------------------------------------


def test_placeholders_split_ids_into_text_and_image_chunks_covering_every_position():
    ids = (1, 2, PAD, PAD, PAD, 3)
    built = model_input_from_placeholders(ids, [(2, 3)], [(PNG, "png")])
    assert [type(c).__name__ for c in built.chunks] == [
        "EncodedTextChunk",
        "ImageChunk",
        "EncodedTextChunk",
    ]
    assert built.length == len(ids)
    assert built.chunks[1].expected_tokens == 3 and built.chunks[1].format == "png"
    assert image_placeholders(built) == [(2, 3)]
    with pytest.raises(TypeError):
        built.to_ints()


def test_adjacent_and_edge_images_have_no_empty_text_chunks():
    ids = (PAD, PAD, PAD, PAD)
    built = model_input_from_placeholders(ids, [(0, 2), (2, 2)], [(PNG, "png"), (PNG, "jpeg")])
    assert all(isinstance(c, ImageChunk) for c in built.chunks) and built.length == 4
    assert model_input_from_placeholders((1,), [], []).chunks[0].tokens == [1]


@pytest.mark.parametrize(
    "placeholders,images,match",
    [
        ([(0, 2)], [], "1 image placeholder"),
        ([], [(PNG, "png")], "0 image placeholder"),
        ([(3, 2)], [(PNG, "png")], "within the prompt"),
        ([(1, 2), (2, 1)], [(PNG, "png")] * 2, "ordered"),
    ],
)
def test_placeholder_mismatches_are_rendering_errors(placeholders, images, match):
    with pytest.raises(RenderingError, match=match):
        model_input_from_placeholders((1, PAD, PAD, 2), placeholders, images)


def test_image_bytes_pass_png_and_jpeg_through_and_reencode_everything_else():
    assert image_bytes({"type": "image", "image": PNG}) == (PNG, "png")
    uri = "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert image_bytes({"type": "image_url", "image_url": {"url": uri}}) == (PNG, "png")
    data, fmt = image_bytes({"type": "image", "image": Image.new("RGB", (8, 8))})
    assert fmt == "jpeg" and data.startswith(b"\xff\xd8")


def test_data_uri_with_parameters_and_jpeg_bytes_pass_through_verbatim():
    picture = jpeg(png((8, 8), "red"))
    uri = "data:image/jpeg;charset=utf-8;base64," + base64.b64encode(picture).decode()
    assert image_bytes({"type": "image_url", "image_url": {"url": uri}}) == (picture, "jpeg")
    assert image_bytes({"type": "image", "image": bytearray(picture)}) == (picture, "jpeg")
    png_uri = "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert image_bytes({"type": "image", "image": png_uri}) == (PNG, "png")


@pytest.mark.parametrize(
    "part,match",
    [
        ({"type": "image", "image": b"\x89PNG\r\n\x1a\njunk"}, "not a decodable image"),
        ({"type": "image", "image": "data:image/png;base64,@@@"}, "not valid base64"),
        ({"type": "image", "image": b"definitely not an image"}, "could not load"),
    ],
)
def test_bad_image_payloads_are_rendering_errors(part, match):
    with pytest.raises(RenderingError, match=match):
        image_bytes(part)


@pytest.mark.parametrize(
    "payload",
    [
        PNG,
        b"\xff\xd8\xff\xe0" + b"\0" * 16,
        b"\xff\xd8\x00\x00" + b"\0" * 16,
        b"\x89PNG" + b"\0" * 16,
    ],
    ids=["png", "jpeg", "jpeg_soi_without_marker", "truncated_png_signature"],
)
def test_direct_and_http_access_agree_on_what_is_png_or_jpeg(payload):
    """One signature check: bytes one access passes through as an image, the other
    does not refuse as the wrong type."""
    from loops_cookbook.llm.images import untouched_image
    from loops_cookbook.model_access.openai.policy import InvalidRequestError
    from loops_cookbook.model_access.openai.translate import image_bytes as gateway_image

    direct = untouched_image({"type": "image", "image": payload})
    encoded = base64.b64encode(payload).decode()
    accepted = set()
    for kind in ("png", "jpeg"):
        try:
            gateway_image(f"data:image/{kind};base64,{encoded}", "messages[0].content[0]")
        except InvalidRequestError:
            continue
        accepted.add(kind)
    assert accepted == ({direct[1]} if direct is not None else set())


# --- Prime adapter with a fake multimodal renderer ---------------------------------


def test_render_zips_prime_placeholders_with_image_parts_in_order():
    renderer = PrimeRenderer(FakePrime())
    rendered = renderer.render(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "ab"}, {"type": "image", "image": PNG}],
            },
            {"role": "tool", "content": [{"type": "image", "image": png(color="black")}]},
        ]
    )
    assert renderer.supports_images
    assert image_placeholders(rendered.prompt_input) == [(3, 3), (7, 3)]
    assert rendered.prompt_input.length == len(rendered.ids)
    assert rendered.ids[3:6] == (PAD,) * 3
    assert rendered.prompt_input.chunks[1].data == PNG


def test_bridge_uses_only_new_placeholders_and_extends_the_previous_chunks():
    renderer = PrimeRenderer(FakePrime())
    first = renderer.render([{"role": "user", "content": [{"type": "image", "image": PNG}]}])
    completion = (5, 6)
    second = renderer.bridge(
        first.ids,
        completion,
        [
            {
                "role": "tool",
                "content": [{"type": "text", "text": "x"}, {"type": "image", "image": PNG}],
            }
        ],
        previous_prompt_input=first.prompt_input,
    )
    assert second.ids[: len(first.ids) + 2] == first.ids + completion
    assert image_placeholders(second.prompt_input) == [(1, 3), (len(first.ids) + 2 + 2, 3)]
    assert second.prompt_input.length == len(second.ids)
    previous = renderer._renderer.bridge_calls[0]["previous_multi_modal_data"]
    assert [(p.offset, p.length) for p in previous.mm_placeholders["image"]] == [(1, 3)]
    assert previous.mm_items["image"] == [{}]


def test_bridge_without_images_stays_text_only_and_matches_legacy_shape():
    renderer = PrimeRenderer(FakePrime())
    first = renderer.render([{"role": "user", "content": "hi"}])
    second = renderer.bridge(
        first.ids,
        (5,),
        [{"role": "user", "content": "yo"}],
        previous_prompt_input=first.prompt_input,
    )
    assert second.prompt_input.to_ints() == list(second.ids)


def test_kimi_images_are_rejected_until_expansion_is_implemented():
    from renderers.kimi_k25 import KimiK25Renderer

    from loops_cookbook.llm.rendering import UnsupportedContextError

    class FakeKimi(KimiK25Renderer):
        def __init__(self):
            pass

        def get_stop_token_ids(self):
            return []

        def render(self, messages, **kwargs):
            return SimpleNamespace(token_ids=[1], multi_modal_data=None)

    renderer = PrimeRenderer(FakeKimi())
    assert not renderer.supports_images
    assert renderer.render([{"role": "user", "content": "text only is fine"}]).ids == (1,)
    with pytest.raises(UnsupportedContextError, match="does not support image"):
        renderer.render([{"role": "user", "content": [{"type": "image", "image": PNG}]}])


def test_bridge_declines_without_previous_chunks_when_images_are_possible():
    renderer = PrimeRenderer(FakePrime())
    first = renderer.render([{"role": "user", "content": [{"type": "image", "image": PNG}]}])
    assert renderer.bridge(first.ids, (5,), [{"role": "tool", "content": "x"}]) is None
    assert (
        renderer.bridge(
            first.ids,
            (5,),
            [{"role": "tool", "content": "x"}],
            previous_prompt_input=first.prompt_input,
        )
        is not None
    )


class EarlierImagesDecline(FakePrime):
    """Declines once it is handed earlier image data, as Prime's Nemotron 3.5 does."""

    def bridge_to_next_turn(self, prompt_ids, completion_ids, new_messages, **kwargs):
        previous = kwargs.get("previous_multi_modal_data")
        if previous is not None and not previous.is_empty():
            self.bridge_calls.append(kwargs)
            return None
        return super().bridge_to_next_turn(prompt_ids, completion_ids, new_messages, **kwargs)


@pytest.mark.parametrize(
    "first_content,bridged",
    [
        pytest.param("hi", True, id="text-only-history-bridges"),
        pytest.param([{"type": "image", "image": PNG}], False, id="earlier-image-declines"),
    ],
)
def test_a_bridge_hands_earlier_image_data_only_when_there_were_earlier_images(
    first_content, bridged
):
    prime = EarlierImagesDecline()
    renderer = PrimeRenderer(prime)
    first = renderer.render([{"role": "user", "content": first_content}])
    second = renderer.bridge(
        first.ids,
        (5,),
        [{"role": "user", "content": "yo"}],
        previous_prompt_input=first.prompt_input,
    )
    assert (second is not None) is bridged
    [call] = prime.bridge_calls
    handed = call.get("previous_multi_modal_data")
    spans = (
        [] if handed is None else [(p.offset, p.length) for p in handed.mm_placeholders["image"]]
    )
    assert spans == image_placeholders(first.prompt_input)


def test_images_nested_in_tool_output_are_rejected_not_dropped():
    from loops_cookbook.llm.rendering import UnsupportedContextError

    with pytest.raises(UnsupportedContextError, match="output"):
        PrimeRenderer(FakePrime()).render(
            [{"role": "tool", "content": [{"output": [{"type": "image", "image": PNG}]}]}]
        )


def test_prime_failures_surface_as_rendering_errors_without_dumping_payloads():
    class Exploding:
        is_multimodal = True

        def render(self, messages, **kwargs):
            raise ValueError("Unexpected content item: " + "x" * 10_000)

        def get_stop_token_ids(self):
            return []

    with pytest.raises(RenderingError) as captured:
        PrimeRenderer(Exploding()).render([{"role": "user", "content": "hi"}])
    assert len(str(captured.value)) < 400 and "ValueError" in str(captured.value)


def test_adapter_follows_the_renderer_reported_image_order(fake_prime):
    """A renderer that reorders messages reports its image parts in mm_items; the
    adapter attaches bytes by that order. Non-part items (Prime's PIL images) are
    ignored and history order is kept."""
    from renderers.base import MultiModalData, PlaceholderRange

    a, b = png((8, 8), "white"), png((8, 8), "black")
    history = [
        {"role": "user", "content": [{"type": "image", "image": a}]},
        {"role": "user", "content": [{"type": "image", "image": b}]},
    ]

    class Reordering:
        is_multimodal = True

        def get_stop_token_ids(self):
            return [99]

        def render(self, messages, **kwargs):
            parts = [p for m in messages for p in m["content"]]
            ids = [1, PAD, PAD, 2, PAD, PAD]
            return SimpleNamespace(
                token_ids=ids,
                multi_modal_data=MultiModalData(
                    mm_hashes={"image": ["x", "y"]},
                    mm_placeholders={"image": [PlaceholderRange(1, 2), PlaceholderRange(4, 2)]},
                    mm_items={"image": list(reversed(parts))},
                ),
            )

    rendered = PrimeRenderer(Reordering()).render(history)
    assert [c.data for c in rendered.prompt_input.chunks if c.type == "image"] == [b, a]

    plain = PrimeRenderer(fake_prime).render(history)  # FakePrime reports {} items
    assert [c.data for c in plain.prompt_input.chunks if c.type == "image"] == [a, b]


# --- client -----------------------------------------------------------------------


async def test_conversation_sends_image_chunks_and_bridges_after_an_image():
    sampler, prime = EchoSampler(), FakePrime()
    client = direct_llm(sampler, PrimeRenderer(prime))
    conversation = client.conversation()
    history = [{"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (4, 4))}]}]
    first = await conversation.generate(messages=history, max_tokens=2)
    assert isinstance(sampler.prompts[0].chunks[1], ImageChunk)
    assert first.sample.prompt_input.length == len(first.sample.prompt_ids)
    history.append(first.native_message)
    history.append({"role": "tool", "content": [{"type": "image", "image": PNG}]})
    second = await conversation.generate(messages=history, max_tokens=2)
    assert prime.bridge_calls, "bridge eligibility must survive image parts in history"
    assert second.sample.prompt_ids[: len(first.sample.prompt_ids) + 2] == (
        first.sample.prompt_ids + first.sample.completion_ids
    )
    assert len(image_placeholders(second.sample.prompt_input)) == 2


@pytest.mark.parametrize(
    ("resent", "bridged"),
    [(bytes(PNG), True), (png(color="black"), False)],
    ids=["same", "changed"],
)
async def test_a_history_image_bridges_only_while_its_bytes_are_unchanged(resent, bridged):
    sampler, prime = EchoSampler(), FakePrime()
    conversation = direct_llm(sampler, PrimeRenderer(prime)).conversation()
    first = await conversation.generate(
        messages=[{"role": "user", "content": [{"type": "image", "image": PNG}]}], max_tokens=2
    )
    history = [
        {"role": "user", "content": [{"type": "image", "image": resent}]},
        first.native_message,
        {"role": "tool", "content": "ok"},
    ]
    await conversation.generate(messages=history, max_tokens=2)
    assert bool(prime.bridge_calls) is bridged


async def test_the_renderer_gets_every_history_image_as_bytes_with_its_other_keys():
    """Encoded once, so a deep copy of the history is cheap bytes, never a PIL image."""
    renderer = ScriptedRenderer(images=True)
    chat = direct_llm(EchoSampler(), renderer).conversation()
    uri = "data:image/png;base64," + base64.b64encode(PNG).decode()
    pil = Image.new("RGB", (4, 4))
    parts = [
        {"type": "image", "image": pil},
        {"type": "image_url", "image_url": {"url": uri}, "detail": "high"},
    ]
    await chat.generate(messages=[{"role": "user", "content": parts}], max_tokens=2)
    [(messages, _)] = renderer.renders
    first, second = messages[0]["content"]
    assert first == {"type": "image", "image": image_bytes(parts[0])[0]}
    assert second == {"type": "image", "image": PNG, "detail": "high"}


async def test_legacy_renderer_returning_plain_ids_still_works():
    class Legacy:
        stop_sequences = ()

        def render(self, messages, tools):
            return (1, 2, 3)

        def parse(self, prompt, completion, tools):
            return ParsedReply("ok", None, (), True, {"role": "assistant", "content": "ok"})

    sampler = EchoSampler()
    reply = (
        await direct_llm(sampler, Legacy())
        .conversation()
        .generate(messages=[{"role": "user", "content": "hi"}], max_tokens=2)
    )
    assert reply.sample.prompt_input.to_ints() == [1, 2, 3]


# --- capture ----------------------------------------------------------------------


def image_prompt():
    ids = (1, PAD, PAD, 2)
    return RenderedPrompt(ids, model_input_from_placeholders(ids, [(1, 2)], [(PNG, "png")]))


async def capture(sampler, prompt):
    return await lease(sampler).generate(
        "main", prompt, SamplingParams(max_tokens=2), conversation_id="c"
    )


async def test_backend_rejects_a_sampler_expansion_that_differs_from_the_renderer():
    class WrongSampler(EchoSampler):
        async def sample_async(self, prompt, **kwargs):
            response = await super().sample_async(prompt, **kwargs)
            response.prompt_token_ids = response.prompt_token_ids + [PAD]
            return response

    with pytest.raises(SamplingFailure, match="differs"):
        await capture(WrongSampler(), image_prompt())


async def test_an_image_echo_is_compared_token_by_token():
    class MiddleSampler(EchoSampler):
        async def sample_async(self, prompt, **kwargs):
            response = await super().sample_async(prompt, **kwargs)
            response.prompt_token_ids = [1, PAD, 99, 2]  # same length and ends
            return response

    with pytest.raises(SamplingFailure, match="differs"):
        await capture(MiddleSampler(), image_prompt())


async def test_captured_image_sample_keeps_its_chunks():
    sample = await capture(EchoSampler(), image_prompt())
    assert sample.prompt_ids == (1, PAD, PAD, 2)
    assert image_placeholders(sample.prompt_input) == [(1, 2)]
    assert sample.prompt_input.length == len(sample.prompt_ids)


def test_prompts_and_samples_validate_chunk_coverage():
    with pytest.raises(RenderingError, match="cover"):
        RenderedPrompt((1, 2), ModelInput.from_ints([1]))
    with pytest.raises(TrajectoryError, match="cover exactly"):
        make_sample("s", (1, 2), (3,), (-0.1,), 0, prompt_input=ModelInput.from_ints([1]))
    assert make_sample("s", (1, 2), (3,), (-0.1,), 0).prompt_input.to_ints() == [1, 2]


async def test_missing_echo_on_an_image_prompt_names_the_real_cause():
    class SilentSampler(EchoSampler):
        async def sample_async(self, prompt, **kwargs):
            response = await super().sample_async(prompt, **kwargs)
            response.prompt_token_ids = []
            return response

    with pytest.raises(SamplingFailure, match="did not echo"):
        await capture(SilentSampler(), image_prompt())


# --- datums -----------------------------------------------------------------------


def test_two_turn_image_rollout_builds_one_datum_with_image_chunks_and_shifted_length():
    prompt = (1, PAD, PAD, 2)
    first = make_sample(
        "a",
        prompt,
        (5, 6),
        (-0.1, -0.2),
        0,
        prompt_input=model_input_from_placeholders(prompt, [(1, 2)], [(PNG, "png")]),
        sequence_id=None,
    )
    second_prompt = prompt + (5, 6, 9)
    second = make_sample(
        "b",
        second_prompt,
        (7,),
        (-0.3,),
        0,
        prompt_input=first.prompt_input.append(EncodedTextChunk(tokens=[5, 6, 9])),
    )
    (row,) = build_rows(make_rollout("t", "g", "r", (first, second), 1.0), 0.5)
    datum = row.datum
    assert datum.model_input.length == len(second_prompt) + 1 - 1
    assert [type(c).__name__ for c in datum.model_input.chunks] == [
        "EncodedTextChunk",
        "ImageChunk",
        "EncodedTextChunk",
        "EncodedTextChunk",
    ]
    assert datum.loss_fn_inputs["target_tokens"].data[len(prompt) - 1 : len(prompt) + 1] == [5, 6]


def test_single_token_completion_keeps_the_prompt_chunks_unchanged():
    prompt = (1, PAD, PAD, 2)
    sample = make_sample(
        "a",
        prompt,
        (5,),
        (-0.1,),
        0,
        prompt_input=model_input_from_placeholders(prompt, [(1, 2)], [(PNG, "png")]),
    )
    (row,) = build_rows(make_rollout("t", "g", "r", (sample,), 1.0), 0.5)
    assert row.datum.model_input == sample.prompt_input


def test_fork_with_image_in_shared_prefix_yields_two_image_datums():
    prompt = (1, PAD, PAD, 2)
    shared = model_input_from_placeholders(prompt, [(1, 2)], [(PNG, "png")])
    first = make_sample("a", prompt, (5, 6), (-0.1, -0.2), 0, prompt_input=shared)
    branch_a = make_sample(
        "b",
        prompt + (5, 6, 7),
        (8,),
        (-0.3,),
        0,
        prompt_input=shared.append(EncodedTextChunk(tokens=[5, 6, 7])),
    )
    branch_b = make_sample(
        "c",
        prompt + (9,),
        (8,),
        (-0.3,),
        0,
        prompt_input=shared.append(EncodedTextChunk(tokens=[9])),
    )
    datums = build_rows(make_rollout("t", "g", "r", (first, branch_a, branch_b), 1.0), 0.5)
    assert len(datums) == 2
    for datum in (row.datum for row in datums):
        assert isinstance(datum.model_input.chunks[1], ImageChunk)
        assert datum.model_input.length == len(datum.loss_fn_inputs["target_tokens"].data)


# --- recording and audit ----------------------------------------------------------


def test_records_replace_image_bytes_with_digests():
    sample = make_sample(
        "a",
        (1, PAD, 2),
        (5,),
        (-0.1,),
        0,
        prompt_input=model_input_from_placeholders((1, PAD, 2), [(1, 1)], [(PNG, "png")]),
    )
    record = sample_record(sample, tokens=True)
    chunk = record["prompt_input"]["chunks"][1]
    assert set(chunk["data"]) == {"sha256", "bytes"} and chunk["data"]["bytes"] == len(PNG)
    assert base64.b64encode(PNG).decode() not in json.dumps(record)
    (row,) = build_rows(make_rollout("t", "g", "r", (sample,), 1.0), 0.5)
    assert datum_record(row.datum)["model_input"]["chunks"][1]["data"] == chunk["data"]
    assert redact_model_input({"chunks": [{"type": "encoded_text", "tokens": [1]}]}) == {
        "chunks": [{"type": "encoded_text", "tokens": [1]}]
    }


class ImageRuntime:
    """Each rollout samples once with an image prompt; grading alternates rewards."""

    capabilities = RuntimeCapabilities(model_access_modes=frozenset({ModelAccessMode.DIRECT}))

    def __init__(self):
        self.closed = False

    async def prepare(self):
        from support.builders import task

        return [task("task")]

    async def sample(self, attempt):
        history = [{"role": "user", "content": [{"type": "image", "image": PNG}]}]
        conversation = attempt.services.model_access.llm().conversation()
        await conversation.generate(messages=history, max_tokens=2)
        return EpisodeResult()

    async def grade(self, attempt, rollout):
        return GradeResult(reward=float(rollout.identity.rollout_id.endswith("r1")))

    async def finish(self, attempt, rollout, disposition):
        pass

    async def abort(self, attempt, reason):
        pass

    async def close(self):
        self.closed = True


class ImageTrainer(MemoryTrainer):
    async def sample_async(self, prompt, **kwargs):
        response = await super().sample_async(prompt, **kwargs)
        ids = []
        for chunk in prompt.chunks:
            ids.extend(
                chunk.tokens if isinstance(chunk, EncodedTextChunk) else [PAD] * chunk.length
            )
        response.prompt_token_ids = ids
        return response


async def test_run_with_images_records_digests_and_passes_the_audit(config, tmp_path):
    trainer, runtime = ImageTrainer(), ImageRuntime()
    run = Run(
        config(
            output_dir=str(tmp_path),
            max_steps=1,
            group_size=2,
            max_group_losses=0,
            trace="full",
            has_images=True,
            router_replay=False,
        ),
        trainer,
        runtime_factory=lambda _config, context: runtime,
        renderer_factory=lambda _config: PrimeRenderer(FakePrime()),
    )
    outcome = await asyncio.wait_for(run.execute(), timeout=5)
    assert outcome.ok
    assert all(isinstance(row.datum.model_input.chunks[1], ImageChunk) for row in trainer.rows[0])
    text = "".join(p.read_text() for p in (run.out / "samples").glob("*.jsonl"))
    text += (run.out / "batches" / "step-0.json").read_text()
    assert base64.b64encode(PNG).decode() not in text and "sha256" in text
    report = audit_run(run.out)
    assert report.steps == 1 and report.datums == 2


# --- check ------------------------------------------------------------------------


class _Stub:
    """A renderer that says whether it takes images and renders with ``render``."""

    stop_sequences = ()

    def __init__(self, supports_images, render=None):
        self.supports_images = supports_images
        self._render = render
        self.rendered = 0

    def render(self, messages, tools):
        self.rendered += 1
        return self._render(messages, tools)

    def parse(self, prompt_ids, completion_ids, tools):
        return ParsedReply("ok", None, (), None, {"role": "assistant", "content": "ok"})


def _needs_torch(messages, tools):
    raise ImportError("Qwen3VLVideoProcessor requires the PyTorch library")


@pytest.mark.parametrize(
    ("has_images", "renderer", "outcome"),
    [
        (True, PrimeRenderer(FakePrime()), "declared (3 tokens for a 32x32 probe)"),
        (True, _Stub(True, lambda m, t: (1, 2, 3)), "declared (0 tokens for a 32x32 probe)"),
        (True, _Stub(False), ValueError(r"has_images = true, but _Stub takes no images")),
        (
            True,
            _Stub(True, _needs_torch),
            ValueError(
                r"has_images = true, but the renderer can't render an image \(ImportError: "
                r"Qwen3VLVideoProcessor requires the PyTorch library\)\. If the vision extra "
                r"is missing, run `uv sync --extra vision`\."
            ),
        ),
        (
            False,
            _Stub(True, _needs_torch),
            (
                "not declared, because [rollout_runtime] has_images = false. "
                "A request with images is refused."
            ),
        ),
    ],
    ids=["placeholders", "legacy ids", "text only", "processor missing", "not declared"],
)
async def test_preflight_probes_images_only_when_the_config_declares_them(
    config, has_images, renderer, outcome
):
    """Preflight probes images only when the config declares them, and reports the result."""
    from loops_cookbook.rl.recipe import ImageSupport

    run = Run(
        config(max_steps=1, has_images=has_images, router_replay=False),
        MemoryTrainer(),
        runtime_factory=lambda _config, context: ImageRuntime(),
        renderer_factory=lambda _config: renderer,
    )
    if isinstance(outcome, ValueError):
        with pytest.raises(ValueError, match=outcome.args[0]):
            await run.preflight()
        return
    _runtime, _tasks, renderers = await run.preflight()
    assert renderers.images is has_images
    assert ImageSupport(run.image_tokens).describe() == outcome
    if not has_images:
        assert renderer.rendered == 0  # never probed


@pytest.mark.parametrize("declared", [True, False])
async def test_a_request_with_images_is_refused_unless_the_run_declares_them(declared):
    from loops_cookbook.llm.client import Renderers
    from loops_cookbook.llm.rendering import UnsupportedContextError
    from loops_cookbook.model_access.lease import DirectModelAccess

    renderer = _Stub(True, lambda m, t: RenderedPrompt.text([1, 2, 3]))
    access = DirectModelAccess(lease(EchoSampler()), Renderers(renderer, images=declared))
    conversation = access.llm().conversation()
    history = [{"role": "user", "content": [{"type": "image", "image": png()}]}]
    if declared:
        await conversation.generate(messages=history, max_tokens=2)
        assert renderer.rendered == 1
    else:
        with pytest.raises(
            UnsupportedContextError, match=r"\[rollout_runtime\] has_images = false"
        ):
            await conversation.generate(messages=history, max_tokens=2)
        assert renderer.rendered == 0


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image", "path": "/no/such/image.png"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,not base64!"}},
        {"type": "tool_result", "output": [{"type": "image", "image": b"not loaded"}]},
    ],
    ids=["missing path", "malformed data URI", "inside a tool's output"],
)
async def test_an_undeclared_image_is_refused_before_it_is_loaded(part):
    from loops_cookbook.llm.client import Renderers
    from loops_cookbook.llm.rendering import UnsupportedContextError
    from loops_cookbook.model_access.lease import DirectModelAccess

    renderer = _Stub(True, lambda m, t: RenderedPrompt.text([1, 2, 3]))
    access = DirectModelAccess(lease(EchoSampler()), Renderers(renderer, images=False))
    conversation = access.llm().conversation()
    history = [{"role": "user", "content": [part]}]
    with pytest.raises(UnsupportedContextError, match=r"\[rollout_runtime\] has_images = false"):
        await conversation.generate(messages=history, max_tokens=2)
