"""Immutable identities and sampling evidence that every cookbook layer shares.

A rollout runtime can read these values but never author them. The cookbook captures
``Sample`` evidence, and only the attempt driver constructs a ``Rollout``.
"""

from __future__ import annotations

# baseten-loops ships no stubs, so Sample validates every value it takes from it.
# pyright: reportMissingTypeStubs=false
import re
from array import array
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import islice
from math import isfinite
from operator import index as as_index
from types import MappingProxyType
from typing import cast, final, overload

from baseten.loops.models import (
    EncodedTextChunk,
    ImageAssetPointerChunk,
    ImageChunk,
    ModelInput,
)

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type PromptChunk = EncodedTextChunk | ImageChunk | ImageAssetPointerChunk

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class TrajectoryError(ValueError):
    """Raised when sampling evidence cannot safely be trained."""


class StaleRolloutError(TrajectoryError):
    """Raised when the oldest contributing generation exceeds the freshness budget."""


class InternalPolicyDeltaError(TrajectoryError):
    """Raised when the contributing generations span too many policy versions."""


def json_value(value: object, *, where: str = "value") -> JsonValue:
    """Return a private, plain-JSON copy of ``value``, and reject anything that is not JSON.

    Tuples become lists, as they would through any JSON encoder. Numbers must be finite,
    because no JSON document can carry the others.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not isfinite(value):
            raise TrajectoryError(f"{where} contains a non-finite number")
        return float(value)
    if isinstance(value, (list, tuple)):
        items: list[object] = list(value)  # pyright: ignore[reportUnknownArgumentType]
        return [json_value(item, where=where) for item in items]
    if isinstance(value, Mapping):
        entries: Mapping[object, object] = value  # pyright: ignore[reportUnknownVariableType]
        result: dict[str, JsonValue] = {}
        for key, item in entries.items():
            if not isinstance(key, str):
                raise TrajectoryError(f"{where} has a non-string key {key!r}")
            result[key] = json_value(item, where=where)
        return result
    raise TrajectoryError(f"{where} contains a {type(value).__name__}, which is not JSON")


def json_mapping(value: Mapping[str, object], *, where: str) -> Mapping[str, JsonValue]:
    """Return a read-only JSON copy of a string-keyed mapping."""
    copied = json_value(value, where=where)
    if not isinstance(copied, dict):
        raise TrajectoryError(f"{where} must be a mapping")
    return MappingProxyType(copied)


def _identity(value: object, name: str) -> None:
    if type(value) is not str or not value:
        raise TrajectoryError(f"{name} must be a nonempty string")


# The type code is unsigned and at least 32 bits, so every token ID a renderer
# produces fits.
_TOKEN_CODE = "I" if array("I").itemsize >= 4 else "L"


@final
class TokenSpan(Sequence[int]):
    """A read-only run of an append-only token buffer that behaves as a tuple of ints.

    A conversation's prompts extend one another, so one buffer holds every token of the
    chain, and each generation's prompt is a span of it. The attempt keeps its tokens
    once, at four bytes each, instead of once per generation that saw them. The buffer
    only ever grows, so a span never changes after it is made.

    A span compares, hashes, slices, concatenates and prints exactly as the tuple of its
    tokens does. It equals that tuple and any span with the same tokens, but never a
    list. Concatenation materializes a tuple. JSON encoders do not know the type, so
    writers convert it with ``tuple()``.
    """

    __slots__ = ("_buffer", "_start", "_stop")

    _buffer: array[int]
    _start: int
    _stop: int

    def __init__(self, ids: Sequence[int] = ()) -> None:
        """Make a span over a private copy of ``ids``, which no caller can change afterwards.

        Only exact nonnegative ints are accepted. Otherwise ``array`` would coerce a
        ``bool`` or any ``__index__`` object to a token that the tuple path rejects.
        """
        buffer = _packed(ids)
        if buffer is None:
            raise ValueError("token IDs must be nonnegative integers below 2**32")
        self._buffer, self._start, self._stop = buffer, 0, len(buffer)

    @classmethod
    def _view(cls, buffer: array[int], start: int, stop: int) -> TokenSpan:
        """Make a zero-copy view, only over a buffer that this module owns and only appends to."""
        if not 0 <= start <= stop <= len(buffer):
            raise ValueError("a token span must lie inside its buffer")
        span = cls.__new__(cls)
        span._buffer, span._start, span._stop = buffer, start, stop
        return span

    def _tokens(self) -> array[int]:
        """Return these tokens for comparison, as the buffer itself when the span covers all of it.

        The result is read only. A caller that keeps or changes it copies it first.
        """
        if self._start == 0 and self._stop == len(self._buffer):
            return self._buffer
        return self._buffer[self._start : self._stop]

    def same_tokens(self, values: Sequence[int]) -> bool:
        """Return whether ``values`` are these tokens, as ``self == tuple(values)`` says.

        This avoids building that tuple. A list of exact ints, such as a sampler's echo
        of the prompt, is packed and compared as a buffer.
        """
        if isinstance(values, TokenSpan):
            return self == values
        if len(values) != len(self):
            return False
        packed = _packed(values)
        if packed is None:
            return self == tuple(values)
        return self._tokens() == packed

    def startswith(self, *parts: Sequence[int]) -> bool:
        """Return whether these tokens begin with ``parts``, one after another.

        The answer is what ``self[:n] == tuple(chain(*parts))`` says. The parts are
        compared one by one as buffers, instead of through a tuple of every token. This
        is the check that a prompt extends the sampled prefix, which runs on every turn
        of a long conversation.
        """
        offset = 0
        for part in parts:
            stop = offset + len(part)
            if stop > len(self):
                return False
            head = TokenSpan._view(self._buffer, self._start + offset, self._start + stop)
            if not head.same_tokens(part):
                return False
            offset = stop
        return True

    def tolist(self) -> list[int]:
        """Return these tokens as a new list, copied in one step from the buffer."""
        return self._buffer[self._start : self._stop].tolist()

    def __len__(self) -> int:
        return self._stop - self._start

    @overload
    def __getitem__(self, index: int) -> int: ...
    @overload
    def __getitem__(self, index: slice) -> TokenSpan | tuple[int, ...]: ...
    def __getitem__(self, index: int | slice) -> int | TokenSpan | tuple[int, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return tuple(self)[index]
            return TokenSpan._view(
                self._buffer, self._start + start, self._start + max(start, stop)
            )
        position = as_index(index)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError("tuple index out of range")
        return self._buffer[self._start + position]

    def __iter__(self) -> Iterator[int]:
        return islice(self._buffer, self._start, self._stop)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TokenSpan):
            if len(self) != len(other):
                return False
            if self._buffer is other._buffer and self._start == other._start:
                return True
            return self._tokens() == other._tokens()
        if isinstance(other, tuple):
            items: tuple[object, ...] = other  # pyright: ignore[reportUnknownVariableType]
            if len(self) != len(items):
                return False
            if not _exact_ints(items):
                # Compare element by element, as tuple equality does, and never coerce
                # through __index__.
                return tuple(self) == items
            try:
                packed = array(_TOKEN_CODE, cast("tuple[int, ...]", items))
            except OverflowError:  # A negative or oversized int is never a span's token.
                return False
            return self._tokens() == packed
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __add__(self, other: object) -> tuple[int, ...]:
        if isinstance(other, TokenSpan):
            return tuple(self) + tuple(other)
        if isinstance(other, tuple):
            items: tuple[int, ...] = other  # pyright: ignore[reportUnknownVariableType]
            return tuple(self) + items
        return NotImplemented

    def __radd__(self, other: object) -> tuple[int, ...]:
        if isinstance(other, tuple):
            items: tuple[int, ...] = other  # pyright: ignore[reportUnknownVariableType]
            return items + tuple(self)
        return NotImplemented

    def __repr__(self) -> str:
        return repr(tuple(self))

    def __copy__(self) -> TokenSpan:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> TokenSpan:
        return self

    def __reduce__(self) -> tuple[type[TokenSpan], tuple[tuple[int, ...]]]:
        return (TokenSpan, (tuple(self),))


@final
class Logprobs(Sequence[float]):
    """Behavior logprobs, packed at eight bytes each, that behave as a tuple of floats.

    A tuple holds a pointer and a float object for each completion token, which is 32
    bytes, for every graded rollout waiting to train. Packed as C doubles, they are the
    same values bit for bit, and they read back as equal floats, because a Python float
    is a C double.

    ``Logprobs`` compares, hashes, indexes and prints exactly as the tuple of its values
    does. It equals that tuple and any ``Logprobs`` with the same values, but never a
    list. Slicing and concatenation materialize tuples. JSON encoders do not know the
    type, so writers convert it with ``tuple()``.
    """

    __slots__ = ("_values",)

    _values: array[float]

    def __init__(self, values: Sequence[float] = ()) -> None:
        """Make a private packed copy of ``values``. Only exact floats are accepted.

        Otherwise ``array`` would turn an int, a bool or any ``__float__`` object into a
        float that the tuple it stands for would not hold.
        """
        packed = _packed_floats(values)
        if packed is None:
            raise ValueError("logprobs must be floats")
        self._values = packed

    @classmethod
    def _of(cls, packed: array[float]) -> Logprobs:
        """Wrap ``packed`` itself, a buffer of doubles that no one else holds."""
        logprobs = cls.__new__(cls)
        logprobs._values = packed
        return logprobs

    def tolist(self) -> list[float]:
        """Return these values as a new list, copied in one step."""
        return self._values.tolist()

    def __len__(self) -> int:
        return len(self._values)

    @overload
    def __getitem__(self, index: int) -> float: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[float, ...]: ...
    def __getitem__(self, index: int | slice) -> float | tuple[float, ...]:
        if isinstance(index, slice):
            return tuple(self._values[index])
        position = as_index(index)
        if not -len(self) <= position < len(self):
            raise IndexError("tuple index out of range")
        return self._values[position]

    def __iter__(self) -> Iterator[float]:
        return iter(self._values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Logprobs):
            return self._values == other._values
        if isinstance(other, tuple):
            items: tuple[object, ...] = other  # pyright: ignore[reportUnknownVariableType]
            return len(self) == len(items) and tuple(self) == items
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __add__(self, other: object) -> tuple[float, ...]:
        if isinstance(other, Logprobs):
            return tuple(self) + tuple(other)
        if isinstance(other, tuple):
            items: tuple[float, ...] = other  # pyright: ignore[reportUnknownVariableType]
            return tuple(self) + items
        return NotImplemented

    def __radd__(self, other: object) -> tuple[float, ...]:
        if isinstance(other, tuple):
            items: tuple[float, ...] = other  # pyright: ignore[reportUnknownVariableType]
            return items + tuple(self)
        return NotImplemented

    def __repr__(self) -> str:
        return repr(tuple(self))

    def __copy__(self) -> Logprobs:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Logprobs:
        return self

    def __reduce__(self) -> tuple[type[Logprobs], tuple[tuple[float, ...]]]:
        return (Logprobs, (tuple(self),))


def _packed_floats(values: Sequence[float]) -> array[float] | None:
    """Return ``values`` as a new buffer of doubles, or ``None`` unless each is exactly a ``float``.

    A ``Logprobs`` is copied as it is.
    """
    if isinstance(values, Logprobs):
        return values._values[:]  # pyright: ignore[reportPrivateUsage]
    items = values if isinstance(values, (tuple, list)) else tuple(values)
    if not set(map(type, items)) <= {float}:
        return None
    return array("d", items)


@final
class PromptChunks:
    """A prompt's chunk structure without its tokens, which is a compact form of a sample's ``ModelInput``.

    A text chunk whose tokens are the prompt's own tokens at its positions is kept as
    its length alone, and read back from the prompt's ``TokenSpan``. Every text chunk a
    renderer makes is like that. Any other chunk, such as an image with its bytes, is
    kept by reference and never copied. So a sample's prompt costs a few entries on top
    of the span it already holds, while a ``ModelInput`` holds a Python int per token.

    ``model_input(ids)`` builds the ``ModelInput`` that the chunks were made from, chunk
    for chunk. It is built when a boundary needs one, and it is not kept.
    """

    __slots__ = ("_entries", "_length")

    _entries: tuple[int | PromptChunk, ...]
    _length: int

    def __init__(self, entries: Sequence[int | PromptChunk]) -> None:
        """Take the entries in order, each a text run's length over the prompt or a chunk itself."""
        length = 0
        for entry in entries:
            if type(entry) is int:
                if entry < 0:
                    raise ValueError("a text run cannot be negative")
                length += entry
            elif isinstance(entry, (EncodedTextChunk, ImageChunk, ImageAssetPointerChunk)):
                length += entry.length
            else:
                raise ValueError(f"a prompt chunk cannot be a {type(entry).__name__}")
        self._entries, self._length = tuple(entries), length

    @classmethod
    def of(
        cls,
        model_input: ModelInput,
        ids: Sequence[int],
        shared: dict[tuple[str, int, bytes], ImageChunk] | None = None,
    ) -> PromptChunks:
        """Return the chunk structure of ``model_input``, a prompt whose tokens are ``ids``.

        ``shared`` maps images already held to their chunk. So a prompt rendered again
        with the same image refers to the chunk it held before.
        """
        entries: list[int | PromptChunk] = []
        offset = 0
        for chunk in model_input.chunks:
            if isinstance(chunk, EncodedTextChunk) and _same_run(ids, offset, chunk.tokens):
                entries.append(len(chunk.tokens))
            elif isinstance(chunk, ImageChunk) and shared is not None:
                key = (chunk.format, chunk.expected_tokens, chunk.data)
                entries.append(shared.setdefault(key, chunk))
            else:
                entries.append(chunk)
            offset += chunk.length
        return cls(entries)

    @property
    def length(self) -> int:
        """The number of positions these chunks cover, as ``ModelInput.length`` counts them."""
        return self._length

    @property
    def text_only(self) -> bool:
        """Whether every chunk is text, so ``prompt_ids`` alone rebuilds the prompt."""
        return all(isinstance(entry, (int, EncodedTextChunk)) for entry in self._entries)

    def model_input(self, ids: Sequence[int]) -> ModelInput:
        """Build the ``ModelInput`` that these chunks describe over ``ids``, from new lists and chunks.

        A chunk kept by reference is shallow-copied, so a caller that changes the result
        changes nothing kept. Image bytes are immutable and shared.
        """
        chunks: list[PromptChunk] = []
        offset = 0
        for entry in self._entries:
            if isinstance(entry, int):
                run = ids[offset : offset + entry]
                if isinstance(run, TokenSpan):
                    # A span holds only exact nonnegative ints, so there is nothing to
                    # validate again.
                    chunks.append(EncodedTextChunk.model_construct(tokens=run.tolist()))
                else:
                    chunks.append(EncodedTextChunk(tokens=list(run)))
                offset += entry
            else:
                chunks.append(entry.model_copy())
                offset += entry.length
        return ModelInput(chunks=chunks)

    # Defining __eq__ leaves the class unhashable, like the ModelInput it stands for.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PromptChunks):
            return NotImplemented
        return self._entries == other._entries

    def __repr__(self) -> str:
        kinds = ", ".join(
            f"text[{entry}]" if isinstance(entry, int) else f"{entry.type}[{entry.length}]"
            for entry in self._entries
        )
        return f"PromptChunks({kinds})"

    def __copy__(self) -> PromptChunks:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> PromptChunks:
        return self

    def __reduce__(self) -> tuple[type[PromptChunks], tuple[tuple[int | PromptChunk, ...]]]:
        return (PromptChunks, (self._entries,))


def _same_run(ids: Sequence[int], start: int, tokens: Sequence[int]) -> bool:
    """Return whether ``tokens`` are ``ids`` from ``start`` on, compared as buffers for a span."""
    stop = start + len(tokens)
    if stop > len(ids):
        return False
    run = ids[start:stop]
    if isinstance(run, TokenSpan):
        return run.same_tokens(tokens)
    return tuple(run) == tuple(tokens)


class TokenBuffers:
    """One attempt's append-only token buffers, so its prompts share their prefixes.

    ``span`` returns the prompt as a span of a buffer that already holds it, or of a
    buffer that it extends, appending only the new tokens. A prompt that diverges from
    every buffer it is tried against starts a buffer of its own. A fork, a render that
    tokenizes history differently, or a retry after a discard can each diverge. A
    buffer is never written below its end, so no earlier span can change.

    There is a single writer. ``span`` runs to completion without awaiting, on the
    attempt's event loop, so no two calls interleave.
    """

    def __init__(self) -> None:
        self._latest: dict[str, array[int]] = {}
        self._recent: array[int] | None = None
        self._images: dict[tuple[str, int, bytes], ImageChunk] = {}

    def span(self, key: str, ids: Sequence[int]) -> TokenSpan | tuple[int, ...]:
        """Store ``ids`` once per chain, or return a plain tuple if they do not fit a buffer.

        Anything other than exact nonnegative ints stays the tuple it was, so ``Sample``
        validates it in the usual way.
        """
        fresh = _packed(ids)
        if fresh is None:
            return tuple(ids)
        chosen = None
        for buffer in (self._latest.get(key), self._recent):
            if buffer is not None and _holds_or_extends(buffer, fresh):
                chosen = buffer
                break
        if chosen is None:
            chosen = fresh
        self._latest[key] = self._recent = chosen
        return TokenSpan._view(chosen, 0, len(fresh))  # pyright: ignore[reportPrivateUsage]

    def chunks(self, ids: Sequence[int], model_input: ModelInput) -> PromptChunks:
        """Return ``model_input``'s chunks over ``ids``, a span that ``span`` returned.

        The text is read back from ``ids``. An image that this attempt already holds,
        matched by bytes, format and placeholder count, reuses the chunk already held.
        """
        return PromptChunks.of(model_input, ids, self._images)


def _exact_ints(values: Sequence[object]) -> bool:
    """Return whether every value is exactly an ``int``, and not a ``bool`` or an ``__index__`` object."""
    return set(map(type, values)) <= {int}


def _packed(ids: Sequence[int]) -> array[int] | None:
    """Return ``ids`` as a new token buffer, or ``None`` unless each is an exact nonnegative int that fits.

    ``array`` alone would coerce ``bool`` and ``__index__`` objects. A span is copied as
    it is, because it is already checked. A list or tuple is read in place.
    """
    if isinstance(ids, TokenSpan):
        return ids._buffer[ids._start : ids._stop]  # pyright: ignore[reportPrivateUsage]
    values = ids if isinstance(ids, (tuple, list)) else tuple(ids)
    if not _exact_ints(values):
        return None
    try:
        return array(_TOKEN_CODE, values)
    except OverflowError:
        return None


def compact_ids(ids: Sequence[int]) -> TokenSpan | tuple[int, ...]:
    """Return ``ids`` packed in a span of their own, or as a tuple if they do not pack."""
    if isinstance(ids, TokenSpan):
        return ids
    values = ids if isinstance(ids, (tuple, list)) else tuple(ids)
    packed = _packed(values)
    if packed is None:
        return tuple(values)
    return TokenSpan._view(packed, 0, len(packed))  # pyright: ignore[reportPrivateUsage]


def compact_logprobs(values: Sequence[float]) -> Logprobs | tuple[float, ...]:
    """Return ``values`` packed, or as a tuple unless every one is exactly a float."""
    if isinstance(values, Logprobs):
        return values
    items = values if isinstance(values, (tuple, list)) else tuple(values)
    packed = _packed_floats(items)
    if packed is None:
        return tuple(items)
    return Logprobs._of(packed)  # pyright: ignore[reportPrivateUsage]


def _holds_or_extends(buffer: array[int], ids: array[int]) -> bool:
    """Return whether ``buffer`` begins with ``ids``, or ``ids`` with ``buffer``.

    In the second case, the rest of ``ids`` is appended to ``buffer``.
    """
    if len(buffer) >= len(ids):
        return buffer[: len(ids)] == ids
    if buffer != ids[: len(buffer)]:
        return False
    buffer.extend(ids[len(buffer) :])
    return True


def _continues(ids: tuple[int, ...] | TokenSpan, previous: Sample) -> bool:
    """Return whether ``ids`` begins with ``previous``'s prompt and then its completion.

    The length is checked first, then the completion, then the prompt. The completion
    is short, so a fork is rejected cheaply. The parts are compared in place rather than
    joined, and a span compares as a buffer. A prompt captured into the same buffer as
    ``previous``'s compares in constant time, and nothing the length of the
    conversation is copied for each candidate.
    """
    split = len(previous.prompt_ids)
    end = split + len(previous.completion_ids)
    if len(ids) < end:
        return False
    if isinstance(ids, TokenSpan):
        # Both comparisons use packed buffers, because the completion's slice is a view.
        return ids[split:end] == previous.completion_ids and ids.startswith(previous.prompt_ids)
    return ids[split:end] == previous.completion_ids and ids[:split] == previous.prompt_ids


@dataclass(frozen=True, slots=True)
class TaskRef:
    """A task that the rollout runtime prepared, named by its ID.

    What the task held when an attempt ran is part of that attempt's evidence, in
    ``Rollout.task_digest``.
    """

    id: str

    def __post_init__(self) -> None:
        _identity(self.id, "task id")


def check_task_digest(digest: object, *, where: str) -> str | None:
    """Return ``digest`` after checking it.

    It must be None, or exactly ``sha256:`` followed by 64 lowercase hex characters.
    """
    if digest is not None and (type(digest) is not str or not _DIGEST.fullmatch(digest)):
        raise TrajectoryError(
            f"{where} task digest must be 'sha256:' followed by 64 lowercase hex characters"
        )
    return digest


@dataclass(frozen=True, slots=True)
class AttemptIdentity:
    """The IDs that name one attempt: its run, its group, its rollout and the attempt itself.

    A rollout tried again keeps its ``rollout_id`` and gets a new ``attempt_id``.
    """

    run_id: str
    group_id: str
    rollout_id: str
    attempt_id: str

    def __post_init__(self) -> None:
        for name in ("run_id", "group_id", "rollout_id", "attempt_id"):
            _identity(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class Sample:
    """One generation that the cookbook captured at the model boundary.

    It holds the generation's tokens, behavior logprobs and policy version.
    ``trainable`` says whether its producer's generations receive gradient.
    """

    sample_id: str
    producer: str
    conversation_id: str
    trainable: bool
    prompt_ids: tuple[int, ...] | TokenSpan  # A span when captured, as TokenSpan explains.
    # These are packed whenever they can be, which is always from a sampler. TokenSpan and
    # Logprobs explain the packing.
    completion_ids: tuple[int, ...] | TokenSpan
    logprobs: tuple[float, ...] | Logprobs
    policy_version: int
    prompt_chunks: PromptChunks  # The text and image chunks that cover exactly prompt_ids.
    stop_reason: str | None = None
    sequence_id: str | None = None
    """The sampler's record of this generation, when it recorded one.

    Token provenance names this record to replay the generation's routing."""
    max_tokens: int | None = None
    """The completion limit the sampler was given, or ``None`` when it was given none."""
    clamped_from: int | None = None
    """The ``max_tokens`` the caller asked for, when the trajectory budget lowered it to ``max_tokens``.

    A generation that then stops on ``"length"`` reached the budget."""

    @property
    def prompt_input(self) -> ModelInput:
        """Build the prompt as the Loops SDK takes it, as text and image chunks.

        The prompt is built from ``prompt_ids`` and ``prompt_chunks`` on every read and
        never kept, because a ``ModelInput`` holds a Python int per token. Read it at the
        boundary that needs it, such as a training row or a run record, and do not hold
        it.
        """
        return self.prompt_chunks.model_input(self.prompt_ids)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_ids, TokenSpan):
            object.__setattr__(self, "prompt_ids", tuple(self.prompt_ids))
        # A graded rollout holds these while it waits for training. Packed, they cost 4
        # and 8 bytes a token instead of 36 and 32. Anything that does not pack stays the
        # tuple it was, and it is validated or refused in the usual way.
        object.__setattr__(self, "completion_ids", compact_ids(self.completion_ids))
        object.__setattr__(self, "logprobs", compact_logprobs(self.logprobs))
        for name in ("sample_id", "producer", "conversation_id"):
            _identity(getattr(self, name), name)
        if type(self.trainable) is not bool:
            raise TrajectoryError("trainable must be a boolean")
        if not self.prompt_ids or not self.completion_ids:
            raise TrajectoryError("sampling evidence requires prompt and completion tokens")
        # A span holds only exact nonnegative ints by construction, so only a tuple is
        # checked.
        tokens = [
            ids for ids in (self.prompt_ids, self.completion_ids) if not isinstance(ids, TokenSpan)
        ]
        if not all(type(token) is int and token >= 0 for ids in tokens for token in ids):
            raise TrajectoryError("token IDs must be nonnegative integers")
        if len(self.completion_ids) != len(self.logprobs):
            raise TrajectoryError("completion tokens and behavior logprobs must align")
        if not all(isfinite(p) for p in self.logprobs):
            raise TrajectoryError("behavior logprobs must be finite")
        if type(self.policy_version) is not int or self.policy_version < 0:
            raise TrajectoryError("the sampler must report a nonnegative policy version")
        if not isinstance(self.prompt_chunks, PromptChunks):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TrajectoryError("prompt_chunks must be PromptChunks")
        if self.prompt_chunks.length != len(self.prompt_ids):
            raise TrajectoryError("prompt chunks must cover exactly the prompt tokens")
        if self.sequence_id is not None and (
            type(self.sequence_id) is not str or not self.sequence_id
        ):
            raise TrajectoryError("a sequence_id must be a nonempty string")
        if self.max_tokens is not None:
            if type(self.max_tokens) is not int or self.max_tokens < 1:
                raise TrajectoryError("max_tokens must be a positive integer")
            # The trajectory budget holds only if the sampler stops at its limit. A
            # longer completion could make a chain that the trainer refuses.
            if len(self.completion_ids) > self.max_tokens:
                raise TrajectoryError(
                    f"{len(self.completion_ids)} completion tokens, over the max_tokens "
                    f"({self.max_tokens}) the sampler was given"
                )
        if self.clamped_from is not None and (
            self.max_tokens is None
            or type(self.clamped_from) is not int
            or self.clamped_from <= self.max_tokens
        ):
            raise TrajectoryError("clamped_from must be a max_tokens above the one sampled with")


@dataclass(frozen=True, slots=True)
class Rollout:
    """One attempt's sealed sampling evidence, with its reward once it is graded.

    Only the attempt driver builds one. A rollout runtime reads it in ``grade()`` and
    ``finish()``, and a group grader in each member's evidence.
    """

    identity: AttemptIdentity
    task: TaskRef
    samples: tuple[Sample, ...]
    reward: float | None  # None until grade() returns, then the committed reward.
    metadata: Mapping[str, JsonValue] = field(compare=False, hash=False)
    """The rollout runtime's own report about its episode, kept verbatim and never interpreted."""
    discarded_versions: tuple[int, ...] = ()
    """The policy versions of the trainable generations that the agent discarded.

    They train nothing, but they date a rollout whose every generation was discarded.
    Such a rollout is still graded, and it still counts in its group's baseline."""
    task_digest: str | None = None
    """What the task held when this attempt ran, as its rollout runtime reported it.

    It is None when the rollout runtime reports no digest."""

    def __post_init__(self) -> None:
        check_task_digest(self.task_digest, where="rollout")
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(
            self, "metadata", json_mapping(deepcopy(dict(self.metadata)), where="metadata")
        )
        identities = [sample.sample_id for sample in self.samples]
        if len(identities) != len(set(identities)):
            raise TrajectoryError("a generation must occur only once in a rollout")
        if self.reward is not None and (
            type(self.reward) not in (int, float) or not isfinite(self.reward)
        ):
            raise TrajectoryError("reward must be a finite number")

    @property
    def policy_versions(self) -> list[int]:
        """The policy versions of the rollout's trainable generations, in order."""
        return [sample.policy_version for sample in self.samples if sample.trainable]

    @property
    def minimum_policy_version(self) -> int | None:
        """The oldest policy the rollout trains on.

        When it trains on none, this is the oldest policy it sampled from before
        discarding.
        """
        return min(self.policy_versions or self.discarded_versions, default=None)

    @property
    def trains(self) -> bool:
        """Whether any of the rollout's generations trains."""
        return bool(self.policy_versions)

    @property
    def internal_delta(self) -> int:
        """How many policy versions apart the trainable generations are, or 0 when there are none.

        A rollout whose delta is over ``max_policy_delta_in_trajectory`` is discarded
        before grading.
        """
        versions = self.policy_versions
        return max(versions) - min(versions) if versions else 0

    @property
    def chains(self) -> tuple[tuple[Sample, ...], ...]:
        """Cover chronological samples with exact-prefix chains, independent of labels."""
        result: list[list[Sample]] = []
        for sample in self.samples:
            current = None
            longest = -1
            for candidate in result:
                previous = candidate[-1]
                length = len(previous.prompt_ids) + len(previous.completion_ids)
                if length > longest and _continues(sample.prompt_ids, previous):
                    current, longest = candidate, length
            if current is not None:
                current.append(sample)
            else:
                result.append([sample])
        return tuple(tuple(chain) for chain in result)
