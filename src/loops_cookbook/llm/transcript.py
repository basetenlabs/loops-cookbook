"""Record the conversation at the model boundary, one turn per generation, as the agent saw it.

A ``Rollout`` holds token IDs, which cannot be rendered back into messages, and it
holds no tools. So when a run asks for it, model access also records each generation
as a ``TranscriptTurn``. The turn holds the request as the agent sent it and the reply
as the agent received it. It is built from each call's arguments as the call passes.
The direct ``Conversation`` is a sampling cache, not a history, so nothing could be
read back from it afterwards.

A turn holds only what is known when it is written. Whether its generation is
evidence is not known then, because over HTTP, delivery is settled only after the
gateway finishes writing. So the record never says. Sealing joins the turns with its
own delivery facts in ``seal_transcript``.

Each turn stores only its messages beyond an earlier turn's whole request. When there
is no such turn, it stores the whole request. A request is frozen once, with image
payloads replaced by the digests of the bytes the sampler received. It begins with an
earlier request exactly when their messages' canonical spellings match. Images and
tool definitions are stored once per attempt, by digest.

Recording never raises into the agent, and it never fails an attempt. The transcript
is a record of what happened, not a condition of training. A request or reply that
can't be recorded is kept as a turn marked ``unrecordable``, beside every other turn.
For example, a direct agent's history may hold a ``bytes`` or a ``set``. A request
whose generation never came back is kept too, as a turn with no reply.

Everything recorded is deeply immutable, made of tuples and ``FrozenMap``s all the way
down. It also deep-copies and pickles, because it is handed to user code.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256

from pydantic import BaseModel

from loops_cookbook.evidence import JsonValue, Sample, TrajectoryError
from loops_cookbook.llm.images import image_bytes, is_image_part, untouched_image
from loops_cookbook.llm.rendering import ParsedReply

type Digest = str
"""A ``sha256:<64 lowercase hex>`` digest of an image's bytes or a tool's canonical JSON."""


class FrozenMap[K, V](Mapping[K, V]):
    """A read-only mapping that, unlike ``MappingProxyType``, deep-copies and pickles."""

    __slots__ = ("_items",)
    _items: dict[K, V]

    def __init__(self, items: Mapping[K, V] | Iterable[tuple[K, V]] = ()) -> None:
        object.__setattr__(self, "_items", dict(items))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __hash__(self) -> int:
        return hash(frozenset(self._items.items()))

    def __getitem__(self, key: K) -> V:
        return self._items[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._items!r})"

    def __reduce__(self) -> tuple[type[FrozenMap[K, V]], tuple[dict[K, V]]]:
        return (type(self), (self._items,))


type FrozenJson = (
    None | bool | int | float | str | tuple[FrozenJson, ...] | FrozenMap[str, FrozenJson]
)
"""A JSON value with lists as tuples and objects as ``FrozenMap``s."""


def thaw(value: FrozenJson) -> JsonValue:
    """Return the plain JSON that a recorded value stands for."""
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    if isinstance(value, FrozenMap):
        return {key: thaw(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class TranscriptImage:
    """An image exactly as the sampler received it."""

    data: bytes
    format: str  # "png" or "jpeg".


@dataclass(frozen=True, slots=True)
class ParsedTurnReply:
    """The reply that ``generate`` returned, recorded for direct model access.

    ``parsed`` is ``None`` when the completion did not parse, and ``parse_error`` says
    why, exactly as the agent's ``LLMResponse`` did. A generation whose reply raised
    instead, as a renderer bug does, is recorded this way under either kind of model
    access. Then ``parsed`` is ``None``, and ``parse_error`` names what was raised. A
    direct agent got that error itself, and an agent over HTTP got a server error.
    """

    parsed: FrozenMap[str, FrozenJson] | None
    parse_error: str | None

    @classmethod
    def of(cls, parsed: ParsedReply | None, error: Exception | None) -> ParsedTurnReply:
        recorded = None
        if parsed is not None:
            calls = [
                {
                    "raw": call.raw,
                    "name": call.name,
                    "arguments": call.arguments,
                    "status": call.status.value,
                    "id": call.id,
                }
                for call in parsed.tool_calls
            ]
            recorded = _object(
                {
                    "content": parsed.content,
                    "reasoning_content": parsed.reasoning_content,
                    "reasoning_complete": parsed.reasoning_complete,
                    "tool_calls": calls,
                    "native_message": parsed.native_message,
                },
                None,
            )
        return cls(recorded, None if error is None else f"{type(error).__name__}: {error}")


@dataclass(frozen=True, slots=True)
class CompletionTurnReply:
    """The Chat Completions body that the gateway built for the agent, recorded for HTTP access."""

    body: FrozenMap[str, FrozenJson]


type TurnReply = ParsedTurnReply | CompletionTurnReply


@dataclass(frozen=True, slots=True)
class TranscriptTurn:
    """One generation, holding what the agent asked, with which tools, and what it got back."""

    sample_id: str | None
    """The generation's sample ID, or ``None`` when no generation came back.

    That happens when the request was sent and then the attempt ended, or sampling
    failed, before a reply.
    """
    producer: str
    conversation_id: str
    extends: str | None
    """The turn whose whole request begins this one's, or ``None`` for a full request."""
    messages: tuple[FrozenJson, ...]
    """The request beyond the ``extends`` turn's request, with image payloads replaced by digests.

    It is empty when the request is ``unrecordable``.
    """
    tools: tuple[Digest, ...]
    reply: TurnReply | None
    """``None`` when no generation came back, or the reply is ``unrecordable``."""
    unrecordable: str | None = None
    """Why the request or the reply is not recorded as JSON, if it is not."""


@dataclass(frozen=True, slots=True)
class Transcript:
    """The turns an attempt recorded, in the order their generations returned."""

    turns: tuple[TranscriptTurn, ...]
    images: FrozenMap[Digest, TranscriptImage]
    tools: FrozenMap[Digest, FrozenJson]


@dataclass(frozen=True, slots=True)
class _Recorded:
    """A recorded turn's whole request, for later turns to extend.

    Each message is kept in its canonical spelling, with image payloads as digests. So
    a later request begins with this one exactly when its spellings begin with these.
    """

    sample_id: str
    producer: str
    conversation_id: str
    spelled: tuple[str, ...]


class TranscriptRecorder:
    """One attempt's transcript, written turn by turn as generations return."""

    def __init__(self) -> None:
        self._turns: list[TranscriptTurn] = []
        self._images: dict[Digest, TranscriptImage] = {}
        self._tools: dict[Digest, FrozenJson] = {}
        self._recorded: list[_Recorded] = []
        # Requests sent and not yet answered, keyed by identity. Until a reply is
        # recorded, each one is the turn of a generation that never came back.
        self._pending: dict[int, PendingTurn] = {}

    def begin(
        self,
        messages: Sequence[object],
        tools: Sequence[object],
        *,
        rendered: Sequence[object],
        named: bool,
        producer: str,
        conversation_id: str,
    ) -> PendingTurn:
        """Keep a request as it is sent. The turn is written once its reply is known.

        ``rendered`` is the same history as the renderer was given it. Its image parts
        hold the bytes the sampler receives. ``named`` says the request belongs to a
        conversation that its caller names, so it extends that conversation's latest
        turn. An unnamed request extends the latest turn of its producer whose whole
        request it begins with.
        """
        try:
            pending = PendingTurn(
                self,
                _snapshot(messages),
                _snapshot(tools),
                _image_parts(rendered),
                named,
                producer,
                conversation_id,
            )
        except Exception as error:  # noqa: BLE001 - A history that loops back on itself.
            # Keep the error to raise where the turn is recorded, which marks the turn
            # unrecordable.
            pending = PendingTurn(self, (), (), (), named, producer, conversation_id, error)
        self._pending[id(pending)] = pending
        return pending

    def record(self, pending: PendingTurn, sample: Sample, reply: Callable[[], TurnReply]) -> None:
        """Write ``pending``'s turn for ``sample``, marking whatever cannot be recorded."""
        self._pending.pop(id(pending), None)
        problems: list[str] = []
        extends, added, tools = None, (), ()
        try:
            if pending.unsnapshotted is not None:
                raise pending.unsnapshotted
            images = _SampledImages(pending.rendered_images)
            request = tuple(_frozen(message, images) for message in pending.messages)
            images.finish()
            spelled = tuple(_canonical(message) for message in request)
            base = self._base(pending, spelled)
            if base is not None:
                # Reuse the base's own spellings, so every turn of a history shares one copy.
                spelled = base.spelled + spelled[len(base.spelled) :]
            definitions = tuple(_frozen(tool, None) for tool in pending.tools)
            digests = tuple(_digest(_canonical(definition).encode()) for definition in definitions)
        except Exception as error:  # noqa: BLE001 - Recording never raises into the agent.
            problems.append(f"request: {_named(error)}")
        else:
            extends = None if base is None else base.sample_id
            added = request[0 if base is None else len(base.spelled) :]
            tools = digests
            self._images.update(images.found)
            self._tools.update(zip(tools, definitions, strict=True))
            self._recorded.append(
                _Recorded(sample.sample_id, sample.producer, sample.conversation_id, spelled)
            )
        try:
            turn_reply = reply()
            _encodes(
                turn_reply.parsed if isinstance(turn_reply, ParsedTurnReply) else turn_reply.body
            )
        except Exception as error:  # noqa: BLE001 - Recording never raises into the agent.
            turn_reply = None
            problems.append(f"reply: {_named(error)}")
        self._turns.append(
            TranscriptTurn(
                sample_id=sample.sample_id,
                producer=sample.producer,
                conversation_id=sample.conversation_id,
                extends=extends,
                messages=added,
                tools=tools,
                reply=turn_reply,
                unrecordable="; ".join(problems) or None,
            )
        )

    def _base(self, pending: PendingTurn, spelled: tuple[str, ...]) -> _Recorded | None:
        """Return the earlier turn whose whole request ``spelled`` begins with exactly."""
        for earlier in reversed(self._recorded):
            if earlier.producer != pending.producer:
                continue
            if pending.named and earlier.conversation_id != pending.conversation_id:
                continue
            if spelled[: len(earlier.spelled)] == earlier.spelled:
                return earlier
            if pending.named:
                return None  # Only the conversation's latest turn is its history.
        return None

    def recorded(self) -> Transcript:
        """Return the turns so far, then each request still waiting for its generation."""
        images, tools = dict(self._images), dict(self._tools)
        unanswered = tuple(
            _unanswered(pending, images, tools) for pending in self._pending.values()
        )
        return Transcript(tuple(self._turns) + unanswered, FrozenMap(images), FrozenMap(tools))


def _unanswered(
    pending: PendingTurn,
    images: dict[Digest, TranscriptImage],
    tools: dict[Digest, FrozenJson],
) -> TranscriptTurn:
    """Record a request that no generation answered, whole, because no later turn extends it."""
    try:
        if pending.unsnapshotted is not None:
            raise pending.unsnapshotted
        found = _SampledImages(pending.rendered_images)
        request = tuple(_frozen(message, found) for message in pending.messages)
        found.finish()
        for message in request:
            _encodes(message)
        definitions = tuple(_frozen(tool, None) for tool in pending.tools)
        digests = tuple(_digest(_canonical(definition).encode()) for definition in definitions)
    except Exception as error:  # noqa: BLE001 - Recording never raises.
        request, digests, unrecordable = (), (), f"request: {_named(error)}"
    else:
        images.update(found.found)
        tools.update(zip(digests, definitions, strict=True))
        unrecordable = None
    return TranscriptTurn(
        sample_id=None,
        producer=pending.producer,
        conversation_id=pending.conversation_id,
        extends=None,
        messages=request,
        tools=digests,
        reply=None,
        unrecordable=unrecordable,
    )


def _named(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


@dataclass(frozen=True, slots=True)
class PendingTurn:
    """A request as it was sent, waiting for the reply the agent receives."""

    recorder: TranscriptRecorder
    messages: tuple[object, ...]
    tools: tuple[object, ...]
    rendered_images: tuple[Mapping[str, object], ...]
    named: bool
    producer: str
    conversation_id: str
    unsnapshotted: Exception | None = None
    """Why the request could not even be copied as it was sent, or None if it was copied."""

    def parsed(self, sample: Sample, parsed: ParsedReply | None, error: Exception | None) -> None:
        self.recorder.record(self, sample, lambda: ParsedTurnReply.of(parsed, error))

    def failed(self, sample: Sample, error: Exception) -> None:
        """Record that the reply to ``sample`` raised ``error`` instead of reaching the agent."""
        self.parsed(sample, None, error)

    def completion(self, sample: Sample, body: bytes) -> None:
        self.recorder.record(
            self, sample, lambda: CompletionTurnReply(_object(json.loads(body), None))
        )


class TurnDisposition(str, Enum):
    DELIVERED = "delivered"  # The generation is in the sealed evidence.
    DISCARDED = "discarded"  # The agent took the generation back.
    UNDELIVERED = "undelivered"  # Sealing dropped it, because it was never wholly delivered.
    UNANSWERED = "unanswered"  # No generation came back for the request.


@dataclass(frozen=True, slots=True)
class SealedTurn:
    turn: TranscriptTurn
    disposition: TurnDisposition
    trainable: bool
    """Whether the turn was delivered by a sealed attempt, from a trainable producer.

    The rollout itself can still be dropped after the seal. Invalid metadata, a failed
    grade or an unusable group can drop it.
    """


@dataclass(frozen=True, slots=True)
class SealedTranscript:
    """An attempt's transcript, beside what its seal kept.

    When the attempt failed instead of sealing, the transcript sits beside how far the
    attempt got. Then ``sealed`` is false and no turn trains.
    """

    turns: tuple[SealedTurn, ...]
    images: FrozenMap[Digest, TranscriptImage]
    tools: FrozenMap[Digest, FrozenJson]
    sealed: bool = True

    def request(self, sample_id: str) -> tuple[FrozenJson, ...]:
        """Return the whole request of the turn that produced ``sample_id``, rebuilt from deltas."""
        turns = {
            sealed.turn.sample_id: sealed.turn
            for sealed in self.turns
            if sealed.turn.sample_id is not None
        }
        turn = turns[sample_id]
        messages = turn.messages
        while turn.extends is not None:
            turn = turns[turn.extends]
            messages = turn.messages + messages
        return messages


def seal_transcript(
    transcript: Transcript,
    *,
    evidence: Sequence[Sample],
    discarded: Collection[str],
    undelivered: Collection[str],
    sealed: bool = True,
) -> SealedTranscript:
    """Mark each turn with what sealing decided for its generation.

    ``evidence`` is what the seal returned, ``discarded`` is what the agent took back,
    and ``undelivered`` is what sealing dropped. Every recorded turn is in exactly one
    of them. Every generation in ``evidence`` must have a turn, because a transcript
    missing one would show a judge a conversation that is not the one that trains. A
    turn that no generation answered is in none of them. When the attempt failed
    instead of sealing, ``sealed`` is false. The turns are then marked by the same facts
    as of the attempt's end, and none of them is trainable.
    """
    recorded = {turn.sample_id for turn in transcript.turns}
    for sample in evidence:
        if sample.sample_id not in recorded:
            raise ValueError(f"generation {sample.sample_id!r} has no recorded turn")
    kept = {sample.sample_id: sample for sample in evidence}
    marked: list[SealedTurn] = []
    for turn in transcript.turns:
        if turn.sample_id is None:
            marked.append(SealedTurn(turn, TurnDisposition.UNANSWERED, False))
            continue
        sample = kept.get(turn.sample_id)
        if sample is not None:
            marked.append(SealedTurn(turn, TurnDisposition.DELIVERED, sealed and sample.trainable))
        elif turn.sample_id in discarded:
            marked.append(SealedTurn(turn, TurnDisposition.DISCARDED, False))
        elif turn.sample_id in undelivered:
            marked.append(SealedTurn(turn, TurnDisposition.UNDELIVERED, False))
        else:
            raise ValueError(f"turn {turn.sample_id!r} has no generation in this seal")
    return SealedTranscript(tuple(marked), transcript.images, transcript.tools, sealed)


def _snapshot(values: Sequence[object]) -> tuple[object, ...]:
    """Return ``values`` with every list and mapping copied.

    So an agent that edits its history while the generation runs changes nothing
    recorded. Nothing is converted yet, and what is not JSON fails when the turn is
    recorded. Only a history that contains itself fails here, and ``begin`` keeps that
    error for the turn.
    """

    def copied(value: object) -> object:
        if isinstance(value, Mapping):
            entries: Mapping[object, object] = value  # pyright: ignore[reportUnknownVariableType]
            return {key: copied(item) for key, item in entries.items()}
        if isinstance(value, (list, tuple)):
            items: list[object] = list(value)  # pyright: ignore[reportUnknownArgumentType]
            return [copied(item) for item in items]
        return value

    return tuple(copied(value) for value in values)


class _SampledImages:
    """The images the sampler received, handed out in the order the request holds them.

    They are taken from the history as the renderer was given it. That history holds
    the client's normalized parts or the gateway's decoded ones. It has the request's
    shape, so the n-th image part of the request is the n-th there. Pairing them with
    the sampled prompt's chunks instead would be wrong for a renderer that reorders
    messages. For example, GLM-5.3 puts tool results in call order.
    """

    def __init__(self, parts: Sequence[Mapping[str, object]]) -> None:
        self._parts = tuple(parts)
        self._next = 0
        self.found: dict[Digest, TranscriptImage] = {}

    def take(self) -> Digest:
        if self._next >= len(self._parts):
            raise TrajectoryError("the request has more images than the rendered history")
        part = self._parts[self._next]
        self._next += 1
        # Take the bytes as the sampler got them. Verifying them is rendering's job.
        data, fmt = untouched_image(part) or image_bytes(part)
        digest = _digest(data)
        self.found.setdefault(digest, TranscriptImage(data, fmt))
        return digest

    def finish(self) -> None:
        """Refuse a rendered history that holds images the request does not."""
        if self._next != len(self._parts):
            raise TrajectoryError("the rendered history has more images than the request")


def _image_parts(values: Sequence[object]) -> tuple[Mapping[str, object], ...]:
    """Return the image parts in the order that ``_frozen`` meets them."""
    found: list[Mapping[str, object]] = []

    def visit(value: object) -> None:
        if is_image_part(value):
            found.append(value)  # pyright: ignore[reportArgumentType]
        elif isinstance(value, Mapping):
            entries: Mapping[object, object] = value  # pyright: ignore[reportUnknownVariableType]
            for item in entries.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            items: list[object] = list(value)  # pyright: ignore[reportUnknownArgumentType]
            for item in items:
                visit(item)

    for value in values:
        visit(value)
    return tuple(found)


def _frozen(value: object, images: _SampledImages | None) -> FrozenJson:
    """Return a frozen JSON copy of ``value``, with image parts as digests given ``images``."""
    if isinstance(value, BaseModel):
        return _frozen(value.model_dump(mode="json"), images)
    if images is not None and is_image_part(value):
        return _image_reference(value, images)  # pyright: ignore[reportArgumentType]
    if isinstance(value, Mapping):
        return _object(value, images)  # pyright: ignore[reportUnknownArgumentType]
    if isinstance(value, (list, tuple)):
        items: list[object] = list(value)  # pyright: ignore[reportUnknownArgumentType]
        return tuple(_frozen(item, images) for item in items)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str.__str__(value)  # A str enum is recorded as its value.
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        # This keeps NaN and infinities too, because they are the policy's output as it was.
        return float(value)
    raise TrajectoryError(f"a transcript contains a {type(value).__name__}, which is not JSON")


def _object(
    value: Mapping[object, object], images: _SampledImages | None
) -> FrozenMap[str, FrozenJson]:
    entries: dict[str, FrozenJson] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TrajectoryError(f"a transcript has a non-string key {key!r}")
        entries[key] = _frozen(item, images)
    return FrozenMap(entries)


# The keys where an image part carries its payload, in the order the image loader reads them.
_PAYLOAD_KEYS = ("image", "image_url", "url", "path")


def _image_reference(part: Mapping[str, object], images: _SampledImages) -> FrozenJson:
    """Return the part with its payload replaced by the digest of the bytes the sampler got."""
    digest = images.take()
    carrier = next(key for key in _PAYLOAD_KEYS if part.get(key) is not None)
    recorded = {key: _frozen(item, None) for key, item in part.items() if key != carrier}
    payload = part[carrier]
    if carrier == "image_url" and isinstance(payload, Mapping):
        url: Mapping[object, object] = payload  # pyright: ignore[reportUnknownVariableType]
        kept = {key: item for key, item in _object(url, None).items() if key != "url"}
        recorded[carrier] = FrozenMap({**kept, "url": digest})
    else:
        recorded[carrier] = digest
    return FrozenMap(recorded)


def _encodes(value: FrozenJson | None) -> None:
    """Raise whatever writing ``value`` would raise.

    A value that freezes need not encode. For example, an integer past Python's digit
    limit does not.
    """
    if value is not None:
        _canonical(value)


def _canonical(value: FrozenJson) -> str:
    """Return one spelling for each JSON value, so ``1``, ``1.0`` and ``true`` stay apart.

    Keys keep their order, because a request is rebuilt from the turns it extends. So a
    request that only reorders an earlier message's keys is not that history, and it is
    recorded whole.
    """
    return json.dumps(thaw(value), separators=(",", ":"))


def _digest(raw: bytes) -> Digest:
    return "sha256:" + sha256(raw).hexdigest()
