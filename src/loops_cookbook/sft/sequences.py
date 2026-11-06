"""Conversations to supervised sequences: the tokens a model sees, and which it learns.

One rule decides the targets for every model: **a token is trained when the trained
model's chat template has it produce that token, in a trained turn, after the turn's
own generation prompt.** It needs only what the renderer reports per token -- which
message owns it and whether the template has the model generate it -- so no model has
code of its own here. Its consequences (every turn learns to end, each turn trains in
its own context, a prefill is context, a turn is tokenized as the model writes it) are
set out in docs/reference/sft.md. A conversation the rule cannot train is refused with
the reason, never trained some other way.
"""

from __future__ import annotations

import hashlib
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from baseten.loops.models import (
    Datum,
    EncodedTextChunk,
    ModelInput,
    ModelInputChunk,
    TensorData,
)

from loops_cookbook.llm.rendering import (
    Message,
    RenderedConversation,
    RenderedPrompt,
    RenderingError,
    Tool,
)
from loops_cookbook.sft.data import Row, TokenExample

IGNORE_INDEX = -100


class TrainingRenderer(Protocol):
    """What supervised training needs from a renderer; ``PrimeRenderer`` is one."""

    @property
    def stop_sequences(self) -> tuple[str, ...] | tuple[int, ...]: ...

    def render(self, messages: Sequence[Message], tools: Sequence[Tool]) -> RenderedPrompt: ...

    def render_conversation(
        self, messages: Sequence[Message], tools: Sequence[Tool]
    ) -> RenderedConversation: ...

    def tokenize(self, text: str) -> tuple[int, ...]: ...

    def detokenize(self, ids: Sequence[int]) -> str: ...


class Refused(ValueError):
    """A conversation the rule cannot train, and why."""


@dataclass(frozen=True, init=False)
class SupervisedSequence:
    """One training sequence: every token the model reads, and the ones it learns.

    A run holds every sequence until it ends, so each is kept compact: its ids as
    machine integers, a byte per token for whether it trains, and its model input as
    the runs of ``ids`` its text chunks hold. ``ids``, ``trained``, ``model_input``
    and ``datum()`` are built from these when asked for, equal to what was given.
    """

    row: str
    """The name of the row it trains, ``file:line``."""
    _ids: array[int]
    _trained: bytes
    _chunks: tuple[int | ModelInputChunk, ...]
    """The model input's chunks: a text chunk holding the next ``n`` of ``ids`` as
    ``n``, and any other chunk (an image) as itself."""

    def __init__(
        self, row: str, ids: Sequence[int], model_input: ModelInput, trained: Sequence[bool]
    ) -> None:
        object.__setattr__(self, "row", row)
        object.__setattr__(self, "_ids", _packed(ids))
        object.__setattr__(self, "_trained", bytes(trained))
        chunks: list[int | ModelInputChunk] = []
        position = 0
        for chunk in model_input.chunks:
            end = position + chunk.length
            spans_ids = isinstance(chunk, EncodedTextChunk) and chunk.tokens == list(
                ids[position:end]
            )
            chunks.append(chunk.length if spans_ids else chunk)
            position = end
        object.__setattr__(self, "_chunks", tuple(chunks))

    @property
    def length(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(self._ids)

    @property
    def trained(self) -> tuple[bool, ...]:
        return tuple(map(bool, self._trained))

    @property
    def trained_tokens(self) -> int:
        return self._trained.count(1)

    @property
    def model_input(self) -> ModelInput:
        chunks: list[ModelInputChunk] = []
        position = 0
        for chunk in self._chunks:
            if isinstance(chunk, int):
                chunks.append(EncodedTextChunk(tokens=list(self._ids[position : position + chunk])))
                position += chunk
            else:
                chunks.append(chunk)
                position += chunk.length
        return ModelInput(chunks=chunks)

    def datum(self) -> Datum:
        """The Loops row: position ``i`` reads token ``i`` and predicts token ``i + 1``."""
        ids, trained = self.ids, self.trained
        length = len(ids) - 1
        targets = [
            token if flag else IGNORE_INDEX
            for token, flag in zip(ids[1:], trained[1:], strict=True)
        ]
        weights = [float(flag) for flag in trained[1:]]
        return Datum(
            model_input=_without_last_token(self.model_input),
            loss_fn_inputs={
                "target_tokens": TensorData(data=targets, dtype="int64", shape=[length]),
                "weights": TensorData(data=weights, dtype="float32", shape=[length]),
            },
        )


def _packed(ids: Sequence[int]) -> array[int]:
    """``ids`` in four bytes each, or eight for an id that does not fit (a tokens row
    is only held to non-negative integers)."""
    try:
        return array("I", ids)
    except OverflowError:
        return array("q", ids)


@dataclass(frozen=True)
class Supervised:
    """What one conversation became."""

    sequences: tuple[SupervisedSequence, ...]
    per_turn: bool
    """Whether the template's history differs from a turn's context, so each trained
    turn is its own sequence."""


def supervise(row: Row, renderer: TrainingRenderer | None, *, max_seq_len: int) -> Supervised:
    """The sequences that train ``row``'s trained turns, or ``Refused``.

    ``renderer`` is needed for a chat row only; a tokens row is never rendered.
    """
    example = row.example
    if isinstance(example, TokenExample):
        # Taken as given: the user owns these targets, stop tokens included.
        sequence = SupervisedSequence(
            row.name,
            example.input_ids,
            ModelInput.from_ints(list(example.input_ids)),
            tuple(bool(weight) for weight in example.weights),
        )
        if sequence.length > max_seq_len:
            raise Refused(f"{sequence.length} tokens, over max_seq_len {max_seq_len}")
        return Supervised((sequence,), per_turn=False)
    if renderer is None:
        raise ValueError("a chat row needs the model's renderer")
    messages = example.chat_messages()
    tools = list(example.tools)
    turns = example.trained_turns
    stops = _stop_ids(renderer)
    try:
        whole = _conversation(renderer, messages, tools)
        prompts = {turn: _prompt(renderer.render(messages[:turn], tools)) for turn in turns}
        split = not all(_starts_with(whole.ids, prompts[turn].ids) for turn in turns)
        if not split:
            sequences = (_sequence(row, renderer, whole, messages, turns, prompts, tools, stops),)
        else:
            sequences = tuple(
                _turn_sequence(row, renderer, messages[: turn + 1], prompts[turn], tools, stops)
                for turn in turns
            )
    except RenderingError as error:
        raise Refused(f"the renderer refused it: {error}") from error
    for sequence in sequences:
        if sequence.length > max_seq_len:
            raise Refused(f"{sequence.length} tokens, over max_seq_len {max_seq_len}")
    return Supervised(sequences, per_turn=split)


def _turn_sequence(
    row: Row,
    renderer: TrainingRenderer,
    messages: list[dict[str, object]],
    prompt: RenderedPrompt,
    tools: Sequence[Tool],
    stops: frozenset[int],
) -> SupervisedSequence:
    """The sequence that trains the last of ``messages`` alone, after its own prompt."""
    turn = len(messages) - 1
    conversation = _conversation(renderer, messages, tools)
    if _starts_with(conversation.ids, prompt.ids):
        return _sequence(
            row, renderer, conversation, messages, (turn,), {turn: prompt}, tools, stops
        )
    return _written_after(row, renderer, conversation, messages, prompt, tools, stops)


def _written_after(
    row: Row,
    renderer: TrainingRenderer,
    conversation: RenderedConversation,
    messages: list[dict[str, object]],
    prompt: RenderedPrompt,
    tools: Sequence[Tool],
    stops: frozenset[int],
) -> SupervisedSequence:
    """The latest turn as the model writes it: the prompt's own tokens, then the turn's
    text tokenized from where the prompt ends.

    For a template whose tokenizer merges a token across the end of the generation
    prompt -- Qwen and Nemotron prefill ``<think>\\n``, and an empty reasoning's ``\\n``
    renders with it as one ``\\n\\n`` -- so the full render is not the prompt plus
    anything. The model never reads that merged token: it is given the prompt and
    writes the rest. The texts must still agree exactly, or the row is refused.
    """
    shared = next(
        (k for k, (a, b) in enumerate(zip(prompt.ids, conversation.ids)) if a != b),
        min(len(prompt.ids), len(conversation.ids)),
    )
    given = renderer.detokenize(prompt.ids[shared:])
    rendered = renderer.detokenize(conversation.ids[shared:])
    if not rendered.startswith(given):
        raise Refused(
            f"messages[{len(messages) - 1}] renders differently as the latest turn than "
            "after its own generation prompt, so no sequence shows it its context; for "
            "turns without reasoning_content, a template's thinking-off option (such as "
            "enable_thinking = false in [model] renderer_chat_template_kwargs) renders "
            "them the way the model writes them"
        )
    turn = len(messages) - 1
    # What trains here is the turn up to its stop; the template may add text after it.
    after_prompt = [
        (token, generated)
        for position, (token, owner, generated) in enumerate(
            zip(conversation.ids, conversation.owners, conversation.generated, strict=True)
        )
        if owner == turn and position >= len(prompt.ids)
    ]
    stop = next((k for k, (token, _) in enumerate(after_prompt) if token in stops), None)
    # Without a stop (GLM's turn ends with the marker of a message after it), all of it trains.
    span = after_prompt if stop is None else after_prompt[: stop + 1]
    if not all(generated for _, generated in span):
        raise Refused(
            f"messages[{turn}] has template text the model does not write after its "
            "generation prompt, which this turn's re-tokenized span cannot leave untrained"
        )
    written = renderer.tokenize(rendered[len(given) :])
    end = next((k + 1 for k, token in enumerate(written) if token in stops), None)
    if end is None:
        written = (*written, *_terminator(renderer, messages, tools, stops))
        end = len(written)
    ids = (*prompt.ids, *written[:end])
    model_input = prompt.prompt_input.append(EncodedTextChunk(tokens=list(written[:end])))
    trained = (*(False for _ in prompt.ids), *(True for _ in written[:end]))
    return SupervisedSequence(row.name, ids, model_input, trained)


def _sequence(
    row: Row,
    renderer: TrainingRenderer,
    conversation: RenderedConversation,
    messages: list[dict[str, object]],
    turns: Sequence[int],
    prompts: dict[int, RenderedPrompt],
    tools: Sequence[Tool],
    stops: frozenset[int],
) -> SupervisedSequence:
    owners = list(conversation.owners)

    def trains(position: int, turn: int, generated: bool) -> bool:
        # Past the turn's own generation prompt: a prefilled token is context.
        return generated and turn in turns and position >= len(prompts[turn].ids)

    trained = [
        trains(position, turn, generated)
        for position, (turn, generated) in enumerate(
            zip(owners, conversation.generated, strict=True)
        )
    ]
    ids, model_input = conversation.ids, conversation.model_input
    last = len(messages) - 1
    if last in turns and not _ends(ids, trained, owners, last, stops):
        terminator = _terminator(renderer, messages, tools, stops)
        ids = (*ids, *terminator)
        model_input = model_input.append(EncodedTextChunk(tokens=list(terminator)))
        trained = [*trained, *(True for _ in terminator)]
        owners = [*owners, *(last for _ in terminator)]
    for turn in turns:
        if not _ends(ids, trained, owners, turn, stops):
            raise Refused(
                f"messages[{turn}] has no trained stop token: the template does not have "
                "the model end this turn, so it would never learn to"
            )
    # Nothing after the last target is ever read: a template's trailing newline
    # would only cost a position.
    end = max(position for position, flag in enumerate(trained) if flag) + 1
    return SupervisedSequence(row.name, ids[:end], _first(model_input, end), trained[:end])


def _conversation(
    renderer: TrainingRenderer, messages: list[dict[str, object]], tools: Sequence[Tool]
) -> RenderedConversation:
    """``messages`` rendered, refused unless every token the model writes belongs to the
    assistant turn that writes it -- the role marker that ends a turn included, though
    GLM renders it with the next message. A turn's targets are then its own tokens."""
    conversation = renderer.render_conversation(messages, tools)
    for position, (owner, generated) in enumerate(
        zip(conversation.owners, conversation.generated, strict=True)
    ):
        if generated and (owner < 0 or messages[owner]["role"] != "assistant"):
            whose = f"messages[{owner}], a {messages[owner]['role']}" if owner >= 0 else "none"
            raise Refused(
                f"the renderer has the model write token {position} but gives it to "
                f"{whose}: a token the model writes must belong to the assistant turn "
                "that writes it, the role marker that ends the turn included"
            )
    return conversation


def _ends(
    ids: Sequence[int],
    trained: Sequence[bool],
    owners: Sequence[int],
    turn: int,
    stops: frozenset[int],
) -> bool:
    """Whether ``turn``'s trained tokens include a stop token."""
    return any(
        trained[position] and owners[position] == turn and token in stops
        for position, token in enumerate(ids)
    )


def _terminator(
    renderer: TrainingRenderer,
    messages: list[dict[str, object]],
    tools: Sequence[Tool],
    stops: frozenset[int],
) -> tuple[int, ...]:
    """The tokens the template would have the model produce to end the final turn.

    Rendered rather than assumed: the message that naturally follows the turn -- its
    tool results when it called tools, the user's reply otherwise -- is appended. The
    turn had no stop token, or it would not need one, so the first it now owns is where
    its terminator starts.
    """
    final = messages[-1]
    calls = final.get("tool_calls")
    follower: dict[str, object] = (
        {"role": "tool", "content": "", **_answering(calls)}
        if isinstance(calls, list) and calls
        else {"role": "user", "content": ""}
    )
    probe = _conversation(renderer, [*messages, follower], tools)
    last = len(messages) - 1
    owned = [
        token
        for token, owner, generated in zip(probe.ids, probe.owners, probe.generated, strict=True)
        if owner == last and generated
    ]
    start = next((k for k, token in enumerate(owned) if token in stops), None)
    if start is None:
        raise Refused(
            "the template gives the final turn no stop token, so the model would never "
            "learn to end it"
        )
    return tuple(owned[start:])


def _answering(calls: list[object]) -> dict[str, object]:
    first = calls[0]
    identity = first.get("id") if isinstance(first, dict) else None
    return {"tool_call_id": identity} if isinstance(identity, str) else {}


_PROBE_CALL_ID = "sft-probe-call-7f3a"


def check_renderer(renderer: object) -> None:
    """Refuse a renderer SFT cannot train with, before any row is rendered.

    It must have the whole ``TrainingRenderer`` surface and stop sequences that are
    token IDs. And it must not render tool call ids: a template that writes them has
    the model generate them in its own format (Kimi K2's ``functions.{name}:{index}``),
    while ids in data from elsewhere are only how results name their calls -- training
    them as written would teach the model ids its own parser cannot read.
    """
    missing = [
        name
        for name in ("stop_sequences", "render", "render_conversation", "tokenize", "detokenize")
        if not hasattr(renderer, name)
    ]
    if missing:
        raise ValueError(
            f"the renderer lacks {', '.join(missing)}, which supervised training needs; "
            "use a Prime renderer"
        )
    training: TrainingRenderer = renderer  # pyright: ignore[reportAssignmentType]
    _stop_ids(training)
    call = {"name": "probe", "arguments": {}}
    rendered = training.render_conversation(
        [
            {"role": "user", "content": "probe"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": _PROBE_CALL_ID, "type": "function", "function": call}],
            },
        ],
        [{"type": "function", "function": {"name": "probe", "parameters": {"type": "object"}}}],
    )
    if _PROBE_CALL_ID in training.detokenize(rendered.ids):
        raise ValueError(
            "this model's template renders tool call ids, so the model writes them in a "
            "format of its own; SFT does not yet rewrite a dataset's ids into it"
        )


def _stop_ids(renderer: TrainingRenderer) -> frozenset[int]:
    stops = renderer.stop_sequences
    if not all(isinstance(stop, int) for stop in stops):
        raise RenderingError(
            "supervised training needs a renderer whose stop sequences are token IDs"
        )
    return frozenset(int(stop) for stop in stops)


def _prompt(rendered: RenderedPrompt | Sequence[int]) -> RenderedPrompt:
    return rendered if isinstance(rendered, RenderedPrompt) else RenderedPrompt.text(rendered)


def _starts_with(ids: Sequence[int], prefix: Sequence[int]) -> bool:
    return len(prefix) <= len(ids) and tuple(ids[: len(prefix)]) == tuple(prefix)


def _without_last_token(model_input: ModelInput) -> ModelInput:
    """The input positions: every token but the last, which is only ever a target."""
    return _first(model_input, model_input.length - 1)


def _first(model_input: ModelInput, length: int) -> ModelInput:
    """The first ``length`` positions; the cut must fall in text, never inside an image."""
    chunks, remaining = [], length
    for chunk in model_input.chunks:
        if remaining >= chunk.length:
            chunks.append(chunk)
            remaining -= chunk.length
            continue
        if remaining:
            if not isinstance(chunk, EncodedTextChunk):
                raise RenderingError("a supervised sequence cannot end inside an image")
            chunks.append(EncodedTextChunk(tokens=chunk.tokens[:remaining]))
        break
    return ModelInput(chunks=chunks)


def digest(train: Sequence[SupervisedSequence], validation: Sequence[SupervisedSequence]) -> str:
    """Identifies exactly what trains and what is held out -- inputs, targets, order and
    which split each sequence is in -- for resume.

    The model input is hashed as well as the ids: an image is a run of placeholder
    tokens in the ids, so two images of one size differ only there. Each split is
    hashed with its length, so a held-out row moved into training is a change.
    """
    hasher = hashlib.sha256()
    for name, sequences in (("train", train), ("validation", validation)):
        hasher.update(f"{name}:{len(sequences)}".encode())
        for sequence in sequences:
            hasher.update(repr((sequence.row, sequence.ids, sequence.trained)).encode())
            hasher.update(sequence.model_input.model_dump_json().encode())
    return hasher.hexdigest()
