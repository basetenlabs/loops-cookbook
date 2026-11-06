"""The target rule on a template whose every token is known; real templates are
checked in tests/contracts/test_sft_sequences.py."""

import pickle

import pytest
from baseten.loops.models import EncodedTextChunk, ImageChunk, ModelInput
from support.rl_sft import chat_row

from loops_cookbook.llm.rendering import RenderedConversation
from loops_cookbook.sft.data import ChatExample, Row, TokenExample
from loops_cookbook.sft.sequences import (
    IGNORE_INDEX,
    Refused,
    SupervisedSequence,
    check_renderer,
    digest,
    supervise,
)

from .conftest import END, HEADERS, ChatRenderer, Merging, text


def row(example):
    return Row("train.jsonl", 1, example)


def test_a_turn_trains_its_content_and_stop_but_never_its_header():
    supervised = supervise(
        row(ChatExample.model_validate(chat_row(("hi", "yo")))), ChatRenderer(), max_seq_len=64
    )
    (sequence,) = supervised.sequences
    trained = [token for token, flag in zip(sequence.ids, sequence.trained, strict=True) if flag]
    assert trained == [*text("yo"), END]
    assert HEADERS["assistant"] in sequence.ids


def test_a_sequence_ends_at_its_last_target():
    example = ChatExample.model_validate(
        {"messages": [*chat_row(("hi", "yo"))["messages"], {"role": "user", "content": "bye"}]}
    )
    (sequence,) = supervise(row(example), ChatRenderer(), max_seq_len=64).sequences
    assert sequence.trained[-1] and sequence.ids[-1] == END


def test_a_tokens_row_trains_exactly_its_weights_with_no_renderer():
    example = TokenExample(input_ids=[7, 8, 9, 10], weights=[0, 1, 0, 1])
    (sequence,) = supervise(row(example), None, max_seq_len=64).sequences
    datum = sequence.datum()
    assert datum.model_input.to_ints() == [7, 8, 9]
    assert datum.loss_fn_inputs["target_tokens"].data == [8, IGNORE_INDEX, 10]
    assert datum.loss_fn_inputs["weights"].data == [1.0, 0.0, 1.0]


@pytest.mark.parametrize(
    "example",
    [
        ChatExample.model_validate(chat_row(("hi", "a long answer"))),
        TokenExample(input_ids=list(range(40)), weights=[0] + [1] * 39),
    ],
    ids=["chat", "tokens"],
)
def test_a_row_over_the_budget_is_refused(example):
    with pytest.raises(Refused, match="over max_seq_len 16"):
        supervise(row(example), ChatRenderer(), max_seq_len=16)


class HiddenStop(ChatRenderer):
    """A template that never has the model produce its stop token."""

    def render_conversation(self, messages, tools=()):
        rendered = super().render_conversation(messages, tools)
        generated = tuple(
            flag and token != END
            for token, flag in zip(rendered.ids, rendered.generated, strict=True)
        )
        return type(rendered)(rendered.ids, rendered.model_input, rendered.owners, generated)


def test_a_turn_that_could_never_learn_to_end_is_refused():
    with pytest.raises(Refused, match="no stop token"):
        supervise(
            row(ChatExample.model_validate(chat_row(("hi", "yo")))), HiddenStop(), max_seq_len=64
        )


class EndsWithNextMarker(ChatRenderer):
    """GLM's shape: a turn has no stop inside it; the next message's role marker ends it."""

    stop_sequences = (HEADERS["user"],)

    def __init__(self, marker_owner):
        self.marker_owner = marker_owner

    def render_conversation(self, messages, tools=()):
        ids, owners, generated = [], [], []
        for index, message in enumerate(messages):
            closes = index > 0 and messages[index - 1]["role"] == "assistant"
            ids.append(HEADERS[message["role"]])
            owners.append(index - 1 if closes and self.marker_owner == "turn" else index)
            generated.append(closes)
            for token in text(message.get("content")):
                ids.append(token)
                owners.append(index)
                generated.append(message["role"] == "assistant")
        return RenderedConversation(
            tuple(ids), ModelInput.from_ints(ids), tuple(owners), tuple(generated)
        )


def test_a_turn_ending_marker_trains_with_its_turn_and_only_when_the_turn_owns_it():
    example = ChatExample.model_validate(chat_row(("hi", "ab"), ("yo", "cd")))
    with pytest.raises(Refused, match="must belong to the assistant turn"):
        supervise(row(example), EndsWithNextMarker("next"), max_seq_len=64)
    (sequence,) = supervise(row(example), EndsWithNextMarker("turn"), max_seq_len=64).sequences
    trained = [token for token, flag in zip(sequence.ids, sequence.trained, strict=True) if flag]
    assert trained == [*text("ab"), HEADERS["user"], *text("cd"), HEADERS["user"]]


def test_the_digest_tells_apart_inputs_with_the_same_tokens():
    """Two images of the same size render to the same placeholder tokens."""
    (sequence,) = supervise(
        row(ChatExample.model_validate(chat_row(("hi", "yo")))), ChatRenderer(), max_seq_len=64
    ).sequences
    image = ImageChunk(data=b"one", format="png", expected_tokens=1)
    other = ImageChunk(data=b"two", format="png", expected_tokens=1)
    assert digest([with_input(sequence, ModelInput(chunks=[image]))], []) != digest(
        [with_input(sequence, ModelInput(chunks=[other]))], []
    )


def with_input(sequence, model_input):
    return SupervisedSequence(sequence.row, sequence.ids, model_input, sequence.trained)


@pytest.mark.parametrize(
    "chunks",
    [
        [EncodedTextChunk(tokens=[1, 2]), EncodedTextChunk(tokens=[3, 2**40, 5])],
        [ImageChunk(data=b"one", format="png", expected_tokens=2), EncodedTextChunk(tokens=[3])],
        # Text that is not the ids at its positions is kept as it is.
        [EncodedTextChunk(tokens=[9, 9]), EncodedTextChunk(tokens=[3, 2**40, 5])],
    ],
    ids=["text", "image", "text other than the ids"],
)
def test_a_sequence_gives_back_exactly_what_it_was_given(chunks):
    ids, trained = (1, 2, 3, 2**40, 5), (False, False, True, False, True)
    model_input = ModelInput(chunks=chunks)
    sequence = SupervisedSequence("train.jsonl:1", ids, model_input, trained)
    for kept in (sequence, pickle.loads(pickle.dumps(sequence))):
        assert (kept.ids, kept.trained, kept.length, kept.trained_tokens) == (ids, trained, 5, 2)
        assert kept.model_input.model_dump_json() == model_input.model_dump_json()
        assert kept == sequence


def test_the_digest_tells_apart_a_held_out_row_moved_into_training():
    first, second = (
        supervise(
            row(ChatExample.model_validate(chat_row((q, "yo")))), ChatRenderer(), max_seq_len=64
        ).sequences[0]
        for q in ("a", "b")
    )
    assert digest([first], [second]) != digest([first, second], [])


class RendersCallIds(ChatRenderer):
    """A template that writes each tool call's id, as Kimi K2's does."""

    def render_conversation(self, messages, tools=()):
        messages = [
            {**message, "content": "".join(call["id"] for call in message["tool_calls"])}
            if message.get("tool_calls")
            else message
            for message in messages
        ]
        return super().render_conversation(messages, tools)


def test_a_template_that_writes_tool_call_ids_is_refused():
    with pytest.raises(ValueError, match="renders tool call ids"):
        check_renderer(RendersCallIds())


def test_a_renderer_missing_what_training_needs_is_refused():
    class Partial:
        stop_sequences = (END,)

        def render(self, messages, tools=()): ...
        def render_conversation(self, messages, tools=()): ...

    with pytest.raises(ValueError, match="lacks tokenize, detokenize"):
        check_renderer(Partial())


def test_a_turn_merged_across_its_prompt_trains_what_the_model_writes():
    example = ChatExample.model_validate(chat_row(("hi", "ab")))
    supervised = supervise(row(example), Merging(), max_seq_len=64)
    (sequence,) = supervised.sequences
    assert supervised.per_turn
    trained = [token for token, flag in zip(sequence.ids, sequence.trained, strict=True) if flag]
    assert trained == [*text("ab"), END]


def test_template_text_the_model_does_not_write_is_never_trained_when_merged():
    example = ChatExample.model_validate(chat_row(("hi", "abc")))
    with pytest.raises(Refused, match="does not write"):
        supervise(row(example), Merging(unwritten=True), max_seq_len=64)


def test_template_text_is_never_trained_when_the_turn_ends_with_the_next_message():
    """GLM ends a turn with the next message's marker, so the turn holds no stop."""
    renderer = Merging(unwritten=True)
    renderer.stop_sequences = (HEADERS["user"],)
    example = ChatExample.model_validate(chat_row(("hi", "abc")))
    with pytest.raises(Refused, match="does not write"):
        supervise(row(example), renderer, max_seq_len=64)
