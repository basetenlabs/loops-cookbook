"""A sample keeps its prompt as chunk structure over its token span, not as a ModelInput."""

import copy
import gc
import pickle
import tracemalloc
from dataclasses import fields
from types import SimpleNamespace

import pytest
from baseten.loops.models import EncodedTextChunk, ImageChunk, ModelInput
from support.builders import make_rollout, make_sample
from support.fakes import PAD, EchoSampler, FakePrime
from support.images import lettered_png, png
from support.leases import direct_llm

from loops_cookbook.evidence import PromptChunks, Sample, TokenBuffers, TokenSpan, TrajectoryError
from loops_cookbook.llm.prime import PrimeRenderer
from loops_cookbook.llm.rendering import ParsedReply
from loops_cookbook.rl.grpo import build_rows
from loops_cookbook.rl.run import redact_model_input, sample_record

PNG = png()

# --- the compact form builds back exactly what it was made from -------------------------


def test_text_chunks_are_kept_as_lengths_and_built_back_chunk_for_chunk():
    ids = TokenSpan(range(70_000, 70_010))
    model_input = ModelInput(
        chunks=[
            EncodedTextChunk(tokens=list(ids[:4])),
            EncodedTextChunk(tokens=[]),
            EncodedTextChunk(tokens=list(ids[4:])),
        ]
    )
    chunks = PromptChunks.of(model_input, ids)
    assert repr(chunks) == "PromptChunks(text[4], text[0], text[6])"
    assert chunks.length == 10 and chunks.text_only
    built = chunks.model_input(ids)
    assert built == model_input and built.model_dump() == model_input.model_dump()
    assert all(type(token) is int for chunk in built.chunks for token in chunk.tokens)
    # New lists every time: the caller owns what it is given.
    assert built.chunks[0].tokens is not chunks.model_input(ids).chunks[0].tokens
    assert chunks.model_input(tuple(ids)) == model_input


def test_images_are_kept_by_reference_and_handed_out_as_copies():
    ids = TokenSpan((1, PAD, PAD, 2, 3))
    image = ImageChunk(data=PNG, format="png", expected_tokens=2)
    model_input = ModelInput(
        chunks=[EncodedTextChunk(tokens=[1]), image, EncodedTextChunk(tokens=[2, 3])]
    )
    chunks = PromptChunks.of(model_input, ids)
    assert repr(chunks) == "PromptChunks(text[1], image[2], text[2])"
    assert not chunks.text_only and chunks.length == 5
    built = chunks.model_input(ids)
    assert built == model_input
    assert built.chunks[1] is not image and built.chunks[1].data is image.data
    built.chunks[1].expected_tokens = 7  # changing what was handed out changes nothing kept
    assert chunks.model_input(ids) == model_input


def test_an_attempt_holds_one_chunk_for_an_image_however_often_it_is_rendered():
    buffers, raw = TokenBuffers(), lettered_png((16, 16), label="Z")

    def rendered(data):
        ids = buffers.span("c", (1, PAD, PAD, 2))
        image = ImageChunk(data=data, format="png", expected_tokens=2)
        chunks = [EncodedTextChunk(tokens=[1]), image, EncodedTextChunk(tokens=[2])]
        return ids, buffers.chunks(ids, ModelInput(chunks=chunks))

    first_ids, first = rendered(raw)
    again_ids, again = rendered(bytes(bytearray(raw)))  # equal bytes, another object
    assert again == first
    assert (
        again.model_input(again_ids).chunks[1].data is first.model_input(first_ids).chunks[1].data
    )
    _, other = rendered(lettered_png((16, 16), label="Y"))
    assert other != first


def test_a_text_chunk_that_is_not_the_prompt_tokens_is_kept_as_it_is():
    model_input = ModelInput(chunks=[EncodedTextChunk(tokens=[1, 9, 3])])
    chunks = PromptChunks.of(model_input, (1, 2, 3))
    assert repr(chunks) == "PromptChunks(encoded_text[3])"
    assert chunks.model_input((1, 2, 3)) == model_input
    # Past the end of the prompt too: kept, and the sample then refuses the length.
    longer = PromptChunks.of(ModelInput.from_ints([1, 2, 3, 4]), (1, 2, 3))
    assert longer.length == 4 and longer.model_input((1, 2, 3)).to_ints() == [1, 2, 3, 4]


@pytest.mark.parametrize("entry", [-1, True, 1.0, "text", None, (1,)])
def test_entries_are_run_lengths_or_chunks(entry):
    with pytest.raises(ValueError):
        PromptChunks([entry])


def test_a_sample_refuses_chunks_that_do_not_cover_its_prompt():
    good = make_sample("s", (1, 2), (3,), (-0.1,))
    with pytest.raises(TrajectoryError, match="cover exactly"):
        Sample(**{**_init(good), "prompt_chunks": PromptChunks([1])})
    with pytest.raises(TrajectoryError, match="must be PromptChunks"):
        Sample(**{**_init(good), "prompt_chunks": ModelInput.from_ints([1, 2])})


def _init(sample):
    return {field.name: getattr(sample, field.name) for field in fields(Sample)}


def test_samples_compare_copy_and_pickle_with_their_prompt():
    prompt = (1, PAD, PAD, 2)
    image = ModelInput(
        chunks=[
            EncodedTextChunk(tokens=[1]),
            ImageChunk(data=PNG, format="png", expected_tokens=2),
            EncodedTextChunk(tokens=[2]),
        ]
    )
    sample = make_sample("s", prompt, (3,), (-0.1,), prompt_input=image)
    assert sample.prompt_input == image
    for restored in (pickle.loads(pickle.dumps(sample)), copy.deepcopy(sample)):
        assert restored == sample and restored.prompt_input == image
    assert copy.deepcopy(sample).prompt_chunks is sample.prompt_chunks
    assert sample != make_sample("s", prompt, (3,), (-0.1,))
    # As unhashable as the ModelInput it replaces.
    with pytest.raises(TypeError):
        hash(sample.prompt_chunks)


# --- through real conversations: identical to the ModelInput the sampler was sent -------


class _Recorded:
    """Each generation's evidence beside the ModelInput the sampler was actually sent."""

    def __init__(self):
        self.sampler, self.prime = EchoSampler(), FakePrime()
        self.llm = direct_llm(self.sampler, PrimeRenderer(self.prime))
        self.sent: dict[str, ModelInput] = {}

    async def turn(self, chat, history):
        response = await chat.generate(messages=history, max_tokens=2)
        # A deep copy: the evidence must equal the request, not merely share its objects.
        self.sent[response.sample.sample_id] = self.sampler.prompts[-1].model_copy(deep=True)
        return response


async def _image_conversation():
    """Bridged turns over images, a discarded turn and its retry, then a forked re-render."""
    recorded = _Recorded()
    chat = recorded.llm.conversation()
    image = {"type": "image", "image": PNG}
    history = [{"role": "user", "content": [{"type": "text", "text": "look"}, image]}]
    first = await recorded.turn(chat, history)
    history += [first.native_message, {"role": "tool", "content": "text only"}]
    second = await recorded.turn(chat, history)
    history += [second.native_message, {"role": "tool", "content": [image]}]
    third = await recorded.turn(chat, history)
    chat.discard(third)
    retried = await recorded.turn(chat, history)
    forked_history = [{"role": "user", "content": [{"type": "text", "text": "edit"}, image]}]
    forked_history += history[1:]
    forked = await recorded.turn(chat, forked_history)
    forked_history += [forked.native_message, {"role": "tool", "content": "more"}]
    after_fork = await recorded.turn(chat, forked_history)
    assert len(recorded.prime.bridge_calls) == 4  # every turn but the first and the fork
    responses = (first, second, third, retried, forked, after_fork)
    return recorded, [response.sample for response in responses]


async def _text_conversation():
    """Bridged text turns, then a re-render: the chunk structure differs between them."""
    recorded = _Recorded()
    chat = recorded.llm.conversation()
    history = [{"role": "user", "content": "start"}]
    samples = []
    for turn in range(4):
        response = await recorded.turn(chat, history)
        samples.append(response.sample)
        history += [response.native_message, {"role": "tool", "content": f"r{turn}"}]
    history[0] = {"role": "user", "content": "restart"}
    samples.append((await recorded.turn(chat, history)).sample)
    return recorded, samples


@pytest.mark.parametrize("conversation", [_image_conversation, _text_conversation])
async def test_evidence_rebuilds_exactly_the_model_input_each_generation_was_sent(conversation):
    recorded, samples = await conversation()
    for sample in samples:
        sent = recorded.sent[sample.sample_id]
        assert sample.prompt_input == sent
        assert sample.prompt_input.model_dump(mode="json") == sent.model_dump(mode="json")
        assert sample.prompt_chunks.text_only == all(c.type == "encoded_text" for c in sent.chunks)
        # Kept compactly: a few entries over the span, never the token lists.
        assert isinstance(sample.prompt_ids, TokenSpan)
        assert len(sample.prompt_chunks.__reduce__()[1][0]) == len(sent.chunks)


@pytest.mark.parametrize("conversation", [_image_conversation, _text_conversation])
async def test_datums_and_records_are_what_the_sent_model_inputs_made(conversation):
    recorded, _ = await conversation()
    kept = list(recorded.llm.samples)
    rollout = make_rollout("task", "group", "r", kept, 1.0)
    datums = build_rows(rollout, 0.5)
    assert len(datums) == len(rollout.chains) == 2  # the chain, and the fork
    for chain, datum in zip(rollout.chains, datums, strict=True):
        last = chain[-1]
        # What build_rows made before: a deep copy of the request, plus the completion.
        expected = (
            recorded.sent[last.sample_id]
            .model_copy(deep=True)
            .append(EncodedTextChunk(tokens=list(last.completion_ids[:-1])))
        )
        assert datum.datum.model_input.model_dump(mode="json") == expected.model_dump(mode="json")
    for sample in kept:
        sent = recorded.sent[sample.sample_id]
        record = sample_record(sample, tokens=True)
        # The on-disk row is unchanged: the same keys, and chunks only for images.
        keys = [
            "sample_id",
            "producer",
            "conversation_id",
            "trainable",
            "prompt_ids",
            "completion_ids",
            "logprobs",
            "policy_version",
            "stop_reason",
            "sequence_id",
            "max_tokens",
            "clamped_from",
        ]
        if any(chunk.type != "encoded_text" for chunk in sent.chunks):
            keys.append("prompt_input")
            assert record["prompt_input"] == redact_model_input(sent.model_dump(mode="json"))
        else:
            assert list(record["prompt_ids"]) == sent.to_ints()
        assert list(record) == keys
        assert "prompt_chunks" not in sample_record(sample, tokens=False)


async def test_a_forked_re_render_refers_to_the_image_the_chain_already_holds():
    _, samples = await _image_conversation()
    first, *_, forked, after_fork = samples
    data = first.prompt_input.chunks[1].data
    for sample in (forked, after_fork):
        assert sample.prompt_input.chunks[1].data is data


# --- memory: a re-rendered conversation holds its prompt once ---------------------------


class _LongRenderer:
    """One token per character at IDs past the small-int cache, fresh ints every call.

    Without a ``bridge`` every turn re-renders the whole history, the way an agent that
    edits history, a changed reasoning effort or a declining renderer makes it.
    """

    stop_sequences = (99,)

    def render(self, messages, tools):
        ids = []
        for index, message in enumerate(messages):
            ids.append(1_000 + index)
            text = str(message.get("content") or "")
            ids.extend(20_000 + (index * 131 + j) % 50_000 for j in range(len(text)))
        return ids

    def parse(self, prompt, completion, tools):
        return ParsedReply(
            "a", None, (), True, native_message={"role": "assistant", "content": "a"}
        )


class _ForgetfulEcho:
    """Echoes the prompt as the Loops sampler does, and keeps nothing of it."""

    policy_version = 0

    async def sample_async(self, prompt, **kwargs):
        completion = list(range(30_000, 30_050))
        sequence = SimpleNamespace(
            tokens=completion, logprobs=[-0.1] * 50, stop_reason="stop", sequence_id=None
        )
        return SimpleNamespace(
            prompt_token_ids=prompt.to_ints(), policy_version=0, sequences=[sequence]
        )


async def test_a_re_rendered_conversation_holds_each_prompt_in_a_few_bytes_a_token():
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        llm = direct_llm(_ForgetfulEcho(), _LongRenderer())
        chat = llm.conversation()
        history = [{"role": "user", "content": "u" * 4_000}]
        for _ in range(12):
            response = await chat.generate(messages=history, max_tokens=50)
            history += [response.native_message, {"role": "tool", "content": "r" * 4_000}]
        del response
        gc.collect()
        held = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    samples = llm.samples
    last = samples[-1]
    assert len(samples) == 12 and len(last.prompt_ids) > 45_000
    total = sum(len(sample.prompt_ids) for sample in samples)
    assert total > 300_000
    # Every turn re-rendered, and every prompt is still a view of one buffer.
    buffers = {id(sample.prompt_ids._buffer) for sample in samples}  # pyright: ignore[reportPrivateUsage]
    assert len(buffers) == 1
    # One compact copy of the last prompt is 4 bytes a token. The conversation, its
    # history text and all 12 samples stay within three of those. A single kept
    # ModelInput of the last prompt alone is 36 bytes a token, nine compact prompts,
    # and every sample's would be about 11 MB here.
    compact = 4 * len(last.prompt_ids)
    assert held < 3 * compact, f"held {held} bytes for a {compact}-byte compact prompt"
