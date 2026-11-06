"""Captured prompts share one token buffer per chain, and read exactly as tuples."""

import asyncio
import copy
import gc
import json
import pickle
import tracemalloc
from array import array
from dataclasses import replace
from hashlib import sha256

import pytest
from support.builders import make_rollout, make_sample
from support.fakes import ScriptedSampler
from support.leases import direct_llm
from support.unit_rest import ChainRenderer, captured_chain

from loops_cookbook.evidence import TokenBuffers, TokenSpan, TrajectoryError
from loops_cookbook.llm.rendering import ParsedReply, RenderedPrompt, RenderingError
from loops_cookbook.rl.grpo import build_rows
from loops_cookbook.rl.run import sample_record

# --- the span reads as the tuple of its tokens -----------------------------------------


def test_a_span_compares_hashes_and_prints_as_its_tuple():
    span = TokenSpan((5, 6, 7))
    assert span == (5, 6, 7) and (5, 6, 7) == span
    assert span != (5, 6) and span != (5, 6, 8)
    assert not (span != (5, 6, 7))  # noqa: SIM202 -- __ne__ itself
    # Like a tuple, never equal to a list, and equal to a tuple of equal numbers.
    assert span != [5, 6, 7] and [5, 6, 7] != span
    assert span == (5.0, 6, 7) and span != (-5, 6, 7) and span != ("5", 6, 7)
    assert span != (5, 6, 2**40) and span != (5, 6, True) and TokenSpan((1,)) == (True,)
    assert hash(span) == hash((5, 6, 7))
    assert {span: 1}[(5, 6, 7)] == 1
    assert repr(span) == repr((5, 6, 7)) and str(span) == str((5, 6, 7))
    assert list(span) == [5, 6, 7] and len(span) == 3 and bool(span)
    assert not TokenSpan(()) and TokenSpan(()) == ()


class _Index:
    """An object ints accept through ``__index__`` but tuples compare by ``__eq__``."""

    def __init__(self, value, *, fails=False):
        self.value, self.fails = value, fails

    def __index__(self):
        if self.fails:
            raise RuntimeError("__index__ must not be called")
        return self.value


def test_a_span_compares_with_a_tuple_element_by_element_never_through_index():
    span = TokenSpan((5, 6))
    assert (5, _Index(6)) != (5, 6) and span != (5, _Index(6)) and (5, _Index(6)) != span
    assert span != (5, _Index(6, fails=True))


@pytest.mark.parametrize("bad", [(True,), (5, _Index(6)), (-1,), (2**40,), (1.0,)])
def test_only_exact_nonnegative_ints_become_a_span(bad):
    with pytest.raises(ValueError, match="nonnegative integers"):
        TokenSpan(bad)
    kept = TokenBuffers().span("c", bad)
    assert type(kept) is tuple and kept == bad
    sample = make_sample("s", (1,), (2,), (-0.1,), 0)
    if bad == (2**40,):  # a valid token too large for a buffer: kept, as a tuple
        assert type(replace(sample, prompt_ids=kept).prompt_ids) is tuple
        return
    with pytest.raises(TrajectoryError, match="nonnegative integers"):
        replace(sample, prompt_ids=kept)


def test_a_span_owns_its_tokens_whatever_it_was_built_from():
    source = array("I", (5, 6, 7))
    span = TokenSpan(source)
    key = {span: "kept"}
    source[0] = 9
    source.append(8)
    assert span == (5, 6, 7) and key[(5, 6, 7)] == "kept" and hash(span) == hash((5, 6, 7))


def test_a_span_indexes_slices_and_concatenates_as_a_tuple_does():
    values = (5, 6, 7, 8)
    span = TokenSpan(values)
    for index in range(-4, 4):
        assert span[index] == values[index]
    for index in (4, -5):
        with pytest.raises(IndexError):
            span[index]
    for start in (None, -9, -2, 0, 1, 3, 9):
        for stop in (None, -9, -1, 0, 2, 4, 9):
            for step in (None, 1, 2, -1):
                assert span[start:stop:step] == values[start:stop:step]
                assert tuple(span[start:stop:step]) == values[start:stop:step]
    assert span[1:3][1:] == (7,)
    assert span + (9,) == values + (9,) and type(span + (9,)) is tuple
    assert (4,) + span == (4, *values) and type((4,) + span) is tuple
    assert span + span == values + values
    with pytest.raises(TypeError):
        span + [9]  # pyright: ignore[reportOperatorIssue]
    assert 7 in span and 9 not in span and span.index(7) == 2 and span.count(5) == 1
    assert list(reversed(span)) == list(reversed(values))


def test_a_span_copies_and_pickles_as_the_same_tokens():
    span = TokenSpan((5, 6, 7))[1:]
    assert copy.copy(span) is span and copy.deepcopy(span) is span
    restored = pickle.loads(pickle.dumps(span))
    assert isinstance(restored, TokenSpan) and restored == (6, 7)


# --- buffers ------------------------------------------------------------------------


def _buffer(span):
    return span._buffer  # pyright: ignore[reportPrivateUsage]


def test_a_chain_shares_one_buffer_and_a_fork_never_rewrites_it():
    buffers = TokenBuffers()
    first = buffers.span("c", (1, 2))
    second = buffers.span("c", (1, 2, 8, 9, 3))
    again = buffers.span("c", (1, 2, 8))  # a retry of a shorter prefix reads in place
    fork = buffers.span("c", (1, 2, 8, 7))  # diverges below the buffer's end
    after_fork = buffers.span("c", (1, 2, 8, 7, 4))
    assert _buffer(first) is _buffer(second) is _buffer(again)
    assert _buffer(fork) is not _buffer(first) and _buffer(after_fork) is _buffer(fork)
    assert (first, second, again, fork, after_fork) == (
        (1, 2),
        (1, 2, 8, 9, 3),
        (1, 2, 8),
        (1, 2, 8, 7),
        (1, 2, 8, 7, 4),
    )
    # Another conversation extending the attempt's latest chain shares it too.
    other = buffers.span("d", (1, 2, 8, 7, 4, 5))
    assert _buffer(other) is _buffer(fork) and other == (1, 2, 8, 7, 4, 5)


def test_ids_a_buffer_cannot_hold_stay_a_tuple():
    huge = (1, 2**40)
    assert type(TokenBuffers().span("c", huge)) is tuple
    assert TokenBuffers().span("c", huge) == huge


# --- through a real attempt: identical evidence to plain tuples ----------------------


def _as_tuples(sample):
    """The sample as it was captured before spans: its prompt a private tuple."""
    return replace(sample, prompt_ids=tuple(sample.prompt_ids))


async def test_captured_evidence_is_identical_to_plain_tuples():
    llm, samples, prompts = await captured_chain()
    first, second, truncated, retried, forked, after_fork = samples
    # The prompts each generation saw, from the renderer's rules alone.
    assert prompts == [
        (1, 15),
        (1, 15, 41, 51, 3, 32),
        (1, 15, 41, 51, 3, 32, 42, 52, 3, 33),
        (1, 15, 41, 51, 3, 32, 42, 52, 3, 33),
        (1, 22, 11, 12, 11, 13),
        (1, 22, 11, 12, 11, 13, 45, 55, 3, 31),
    ]
    assert all(isinstance(sample.prompt_ids, TokenSpan) for sample in samples)
    # Later appends never moved an earlier view, the discarded one included.
    assert [tuple(sample.prompt_ids) for sample in samples] == prompts
    assert _buffer(first.prompt_ids) is _buffer(second.prompt_ids) is _buffer(retried.prompt_ids)
    assert _buffer(truncated.prompt_ids) is _buffer(first.prompt_ids)
    assert _buffer(forked.prompt_ids) is _buffer(after_fork.prompt_ids)
    assert _buffer(forked.prompt_ids) is not _buffer(first.prompt_ids)

    kept = list(llm.samples)
    assert [sample.sample_id for sample in kept] == [
        s.sample_id for s in (first, second, retried, forked, after_fork)
    ]
    rollout = make_rollout("task", "group", "r", kept, 1.0)
    reference = make_rollout("task", "group", "r", [_as_tuples(s) for s in kept], 1.0)
    assert rollout == reference

    def chains(of):
        return [[sample.sample_id for sample in chain] for chain in of.chains]

    assert (
        chains(rollout)
        == chains(reference)
        == [
            [first.sample_id, second.sample_id, retried.sample_id],
            [forked.sample_id, after_fork.sample_id],
        ]
    )

    def datums(of):
        return [row.datum.model_dump(mode="json") for row in build_rows(of, 0.5)]

    assert datums(rollout) == datums(reference) and len(datums(rollout)) == 2

    def records(of):
        return "".join(
            json.dumps(sample_record(sample, tokens=tokens), allow_nan=False) + "\n"
            for sample in of.samples
            for tokens in (True, False)
        )

    assert records(rollout) == records(reference)
    assert (
        sha256(records(rollout).encode()).digest() == sha256(records(reference).encode()).digest()
    )
    assert json.loads(records(rollout).splitlines()[0])["prompt_ids"] == list(prompts[0])


class _YieldingSampler(ScriptedSampler):
    """Yields to the loop mid-generation, so concurrent attempts interleave."""

    async def sample_async(self, prompt, **kwargs):
        await asyncio.sleep(0)
        return await super().sample_async(prompt, **kwargs)


async def test_concurrent_attempts_never_share_or_corrupt_a_buffer():
    async def one(index):
        renderer = ChainRenderer()
        rendered = []
        render, bridge = renderer.render, renderer.bridge
        renderer.render = lambda *a: rendered.append(render(*a)) or rendered[-1]
        renderer.bridge = lambda *a, **k: rendered.append(bridge(*a, **k)) or rendered[-1]
        sampler = _YieldingSampler(tokens=lambda call: [40 + index, 50 + call])
        llm = direct_llm(sampler, renderer)
        chat = llm.conversation()
        history = [{"role": "user", "content": "s" * (index + 1)}]
        seen = []
        for turn in range(6):
            response = await chat.generate(messages=history, max_tokens=2)
            seen.append((response.sample, rendered[-1]))
            if turn == 2:
                chat.discard(response)
                response = await chat.generate(messages=history, max_tokens=2)
                seen.append((response.sample, rendered[-1]))
            if turn == 4:  # fork: an earlier message changes, so the history re-renders
                history = [{"role": "user", "content": "f" * (index + 2)}, *history[1:]]
            history = [
                *history,
                {"role": "assistant", "content": "a"},
                {"role": "tool", "content": "r"},
            ]
        return seen

    attempts = await asyncio.gather(*(one(index) for index in range(8)))
    owner = {}
    for index, seen in enumerate(attempts):
        for sample, prompt in seen:
            assert sample.prompt_ids == prompt and tuple(sample.prompt_ids) == prompt
            assert owner.setdefault(id(_buffer(sample.prompt_ids)), index) == index
    assert len(owner) == 16  # one chain and one fork per attempt


# --- compared as buffers, as tuples compare ---------------------------------------------


_PARTS = [(), (5,), (5, 6), (6, 7), (5, 6, 7), (5, 6, 7, 8), (5.0, 6), (5, True), (-5,), (2**40,)]


def _fits(part):
    return all(type(token) is int and 0 <= token < 2**32 for token in part)


@pytest.mark.parametrize("part", [*_PARTS, *map(list, _PARTS)])
def test_same_tokens_says_what_tuple_equality_says(part):
    for span in (TokenSpan((5, 6, 7)), TokenSpan((4, 5, 6, 7, 8))[1:4], TokenSpan((5, 6))):
        assert span.same_tokens(part) == (span == tuple(part))
        if _fits(part):
            assert span.same_tokens(TokenSpan(part)) == (span == tuple(part))


@pytest.mark.parametrize("first", _PARTS)
@pytest.mark.parametrize("second", [(), (7,), [7, 8], (7.0,), (8,)])
def test_startswith_says_what_a_tuple_prefix_says(first, second):
    for span in (TokenSpan((5, 6, 7, 8)), TokenSpan((1, 5, 6, 7, 8, 9))[1:5]):
        prefix = (*first, *second)
        assert span.startswith(first, second) == (tuple(span[: len(prefix)]) == prefix)
        if _fits(first):
            assert span.startswith(TokenSpan(first), second) == span.startswith(first, second)


# --- a rendered prompt holds its tokens compactly ---------------------------------------


def test_a_rendered_prompt_holds_its_ids_at_four_bytes_a_token():
    values = tuple(range(70_000, 120_000))  # past the small-int cache, as real IDs are
    rendered = RenderedPrompt.text(list(values))
    assert isinstance(rendered.ids, TokenSpan)
    assert rendered.ids == values and hash(rendered.ids) == hash(values)
    assert rendered.ids[10:20] == values[10:20] and list(rendered.ids) == list(values)
    assert rendered.prompt_input.to_ints() == list(values)
    buffer = _buffer(rendered.ids)
    assert buffer.itemsize == 4 and len(buffer) == len(values)
    assert RenderedPrompt(rendered.ids, rendered.prompt_input).ids is rendered.ids


@pytest.mark.parametrize("bad", [(), (-1,), (True,), (1.0,), ("1",), (2**32,)])
def test_a_rendered_prompt_refuses_what_it_always_refused(bad):
    with pytest.raises(RenderingError, match="nonempty nonnegative integers"):
        RenderedPrompt.text(bad)


# --- the bridge holds the evidence, not a copy of it ------------------------------------


class LongRenderer:
    """A long first prompt; each bridge appends the sampled turn and a long tool result."""

    stop_sequences = (99,)
    first, result = 40_000, 8_000

    def render(self, messages, tools):
        return list(range(1_000, 1_000 + self.first))

    def bridge(self, prompt, completion, messages, tools, *, previous_messages=()):
        start = 100_000 + len(prompt)
        return [*prompt, *completion, *range(start, start + self.result)]

    def parse(self, prompt, completion, tools):
        return ParsedReply(
            "a", None, (), True, native_message={"role": "assistant", "content": "a"}
        )


async def test_a_long_conversation_holds_its_prompt_once_in_the_evidence():
    # Traced from the start: only what was allocated while tracing is counted freed.
    tracemalloc.start()
    try:
        await _holds_its_prompt_once()
    finally:
        tracemalloc.stop()


async def _holds_its_prompt_once():
    completion = list(range(500_000, 500_300))
    sampler = ScriptedSampler(tokens=lambda call: completion, logprobs=[-0.1] * len(completion))
    llm = direct_llm(sampler, LongRenderer())
    chat = llm.conversation()
    history = [{"role": "user", "content": "start"}]
    for _ in range(5):
        response = await chat.generate(messages=history, max_tokens=300)
        history += [response.native_message, {"role": "tool", "content": "result"}]
    samples = llm.samples
    last = samples[-1]
    assert len(samples) == 5 and len(last.prompt_ids) > 70_000
    # Every turn's prompt is a view of the one buffer: the last prompt, stored once.
    assert all(_buffer(sample.prompt_ids) is _buffer(last.prompt_ids) for sample in samples)
    assert len(_buffer(last.prompt_ids)) == len(last.prompt_ids)

    sampler.prompts.clear()  # the fake's own record of every request
    del response
    gc.collect()
    held = tracemalloc.get_traced_memory()[0]
    del chat
    gc.collect()
    freed = held - tracemalloc.get_traced_memory()[0]
    # Dropping the conversation frees its message history and nothing of the prompt:
    # not a tuple (8 bytes a token and more), not even a packed copy (4 bytes a token).
    assert freed < len(last.prompt_ids)
