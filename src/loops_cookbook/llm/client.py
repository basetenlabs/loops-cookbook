"""Agent-owned conversation state over immutable token-level sampling evidence."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Protocol
from uuid import uuid4

from baseten.loops.models import SamplingParams
from pydantic import BaseModel
from renderers.base import ParsedToolCall, ToolCallParseStatus

from loops_cookbook.evidence import JsonValue, Sample, TrajectoryError
from loops_cookbook.llm.images import image_bytes, image_payload, is_image_part
from loops_cookbook.llm.render_pool import RenderPool
from loops_cookbook.llm.rendering import (
    ParsedReply,
    RenderedPrompt,
    Renderer,
    RendererFailure,
    RenderingError,
)
from loops_cookbook.llm.transcript import TranscriptRecorder


def _same_render_input(left: Any, right: Any) -> bool:
    """Conservative structural equality, retaining mapping order and native types.

    This only authorizes an incremental optimization, never training-token reuse.
    Ordinary dict equality would miss reordered tool arguments or declarations.
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, BaseModel):
        return _same_render_input(left.model_dump(), right.model_dump())
    if isinstance(left, Mapping):
        return len(left) == len(right) and all(
            _same_render_input(lk, rk) and _same_render_input(lv, rv)
            for (lk, lv), (rk, rv) in zip(left.items(), right.items(), strict=True)
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same_render_input(lv, rv) for lv, rv in zip(left, right, strict=True)
        )
    if left is None or type(left) in (str, int, float, bool, bytes):
        return left == right
    return False


class _HistoryImages:
    """A conversation's history images, as ``{"type": "image", "image": <bytes>}`` parts.

    Every ``generate`` normalizes the caller's full history. A payload given as bytes
    or a data URI is the same image whichever turn resends it, so it is decoded once
    and reused while the history holds it. PIL objects, paths and URLs can change
    between calls, so they are loaded again each call; agents that care should pass
    bytes. Other keys on a part are kept.
    """

    def __init__(self) -> None:
        self._decoded: dict[bytes | str, bytes] = {}

    def canonical(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        earlier, self._decoded = self._decoded, {}
        result = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list) and any(is_image_part(part) for part in content):
                parts = []
                for part in content:
                    if is_image_part(part):
                        kept = {
                            k: v
                            for k, v in part.items()
                            if k not in {"image", "image_url", "url", "path"}
                        }
                        parts.append({**kept, "type": "image", "image": self._bytes(part, earlier)})
                    else:
                        parts.append(part)
                message = {**message, "content": parts}
            result.append(dict(message))
        return result

    def _bytes(self, part: Mapping[str, Any], earlier: Mapping[bytes | str, bytes]) -> bytes:
        payload = image_payload(part)
        fixed = type(payload) is bytes or (type(payload) is str and payload.startswith("data:"))
        if not fixed:
            return image_bytes(part)[0]
        data = self._decoded.get(payload) or earlier.get(payload) or image_bytes(part)[0]
        self._decoded[payload] = data
        return data


def _reasoning_tokens(renderer: Renderer, sample: Sample, parsed: ParsedReply) -> int | None:
    """The renderer's count of the reply's reasoning tokens, when it has one."""
    count = getattr(renderer, "reasoning_tokens", None)
    return count(sample.completion_ids, parsed) if count is not None else None


def _as_rendered(value: Any) -> RenderedPrompt:
    """Accept a ``RenderedPrompt`` or, from text-only renderers, plain token IDs."""
    if isinstance(value, RenderedPrompt):
        return value
    return RenderedPrompt.text(value)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """One generation's reply: its exact sample, and what the renderer read from it.

    A tool call the renderer could not parse is the model's output, not an error:
    ``tool_calls`` carries every call the model wrote, each with its ``status``;
    ``native_message`` is ``None`` and ``parse_error`` says why. The generation was
    delivered by returning, so it trains; what to do about the call is the agent's.
    To answer it, append ``history_message``: the conversation continues from the
    reply's sampled tokens, exactly as the model wrote them.
    """

    sample: Sample
    parsed: ParsedReply | None
    parse_error: Exception | None = None
    # The completion's tokens before its reasoning closed, from a renderer that can find
    # them (``reasoning_tokens``); ``None`` from one that cannot.
    reasoning_tokens: int | None = None

    @property
    def usage(self) -> TokenUsage:
        return TokenUsage(len(self.sample.prompt_ids), len(self.sample.completion_ids))

    @property
    def response_id(self) -> str:
        return self.sample.sample_id

    @property
    def reasoning_complete(self) -> bool | None:
        return self.parsed.reasoning_complete if self.parsed is not None else None

    @property
    def content(self) -> str | None:
        return self.parsed.content if self.parsed is not None else None

    @property
    def reasoning_content(self) -> str | None:
        return self.parsed.reasoning_content if self.parsed is not None else None

    @property
    def tool_calls(self):
        return self.parsed.tool_calls if self.parsed is not None else ()

    @property
    def native_message(self) -> Mapping[str, object] | None:
        """Provider-native assistant history, separate from immutable sampling evidence."""
        return deepcopy(self.parsed.native_message) if self.parsed is not None else None

    @property
    def history_message(self) -> Mapping[str, object] | None:
        """The assistant turn to append to history before answering this reply.

        ``native_message`` when the renderer built one. For a reply with a tool call the
        renderer could not parse there is none, since no message renders back to what
        the model wrote; this is that turn as written instead: its content, its
        reasoning, and every call by its id (``call_<sample id>_<n>`` when the model gave
        none), its name (``""`` when it wrote none) and its arguments as text (the raw
        call when none could be read). The conversation never renders it: the next turn
        bridges from the reply's sampled tokens, and a conversation that cannot bridge
        refuses to continue rather than put words in the model's mouth. Append it as it
        is: a copy stored differently is still this turn, but one with other content or
        call IDs is history the agent wrote.
        ``None`` for a completion with nothing parsed.
        """
        native = self.native_message
        if native is not None or self.parsed is None:
            return native
        message: dict[str, object] = {"role": "assistant", "content": self.parsed.content}
        if self.parsed.reasoning_content is not None:
            message["reasoning_content"] = self.parsed.reasoning_content
        message["tool_calls"] = [
            {
                "id": call.id or f"call_{self.sample.sample_id}_{index}",
                "type": "function",
                "function": {"name": call.name or "", "arguments": _arguments_text(call)},
            }
            for index, call in enumerate(self.parsed.tool_calls)
        ]
        return message

    @property
    def stop_reason(self) -> str | None:
        return self.sample.stop_reason

    @property
    def reasoning(self) -> str | None:
        return self.reasoning_content

    @property
    def truncated(self) -> bool:
        """The generation stopped before the model finished its turn.

        Three signals say the generation is structurally incomplete: the sampler
        reached its limit, reasoning never closed, or a tool-call block never closed
        (the renderer reports that as an opening delimiter reaching a stop). ``content`` and
        ``tool_calls`` separate none of them from a finished turn -- a generation
        severed inside reasoning leaves an empty message and no tool calls, which is
        byte for byte what a model that chose to say nothing looks like -- so an
        agent asks once here rather than rediscovering all three. A cut turn
        recorded as an answer is graded as an answer.

        Whether every tool call parsed is a separate question, answered by
        ``native_message``: it is ``None`` when the renderer withheld the message
        because a tool call did not parse. Either way the reply is the model's turn,
        and ``history_message`` is what to append: the native message, or the turn as
        written, which continues only from its sampled tokens. (A completion the
        renderer cannot read at all never becomes a reply: ``generate`` raises
        ``RendererFailure``.) A cut generation can parse and a parsed one can be cut,
        so the two are worth asking separately.

        What to do about either is the caller's. ``tool_calls[i].status`` says what
        the renderer made of each attempt, which is information the model itself
        does not have.
        """
        return (
            self.stop_reason == "length"
            # False, not None: None means the renderer reports no reasoning state.
            or self.reasoning_complete is False
            or any(call.status == ToolCallParseStatus.UNCLOSED_BLOCK for call in self.tool_calls)
        )


def _arguments_text(call: ParsedToolCall) -> str:
    """A call's arguments as the model wrote them: JSON when they were read, else raw."""
    if isinstance(call.arguments, str):
        return call.arguments
    if call.arguments is not None:
        return json.dumps(call.arguments, ensure_ascii=False)
    return call.raw


def canonical_arguments(arguments: object) -> str:
    """Tool-call arguments as one string however they were serialized: JSON is normalized."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True, slots=True)
class _Turn:
    """An assistant turn as the gateway recognises an echo: content, calls, reasoning.

    ``content: null`` equals an empty string, and each call compares by ID, name and
    canonical arguments, so a copy stored or serialized differently is the same turn.
    """

    content: object
    calls: tuple[tuple[object, object, str], ...]
    reasoning: object

    @classmethod
    def of(cls, message: Mapping[str, Any]) -> _Turn:
        calls: list[tuple[object, object, str]] = []
        for call in message.get("tool_calls") or ():
            if isinstance(call, BaseModel):
                call = call.model_dump()
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            if isinstance(function, BaseModel):
                function = function.model_dump()
            if isinstance(function, Mapping):
                name, arguments = function.get("name"), function.get("arguments")
            else:
                name, arguments = call.get("name"), call.get("arguments")
            calls.append((call.get("id"), name, canonical_arguments(arguments)))
        return cls(message.get("content") or "", tuple(calls), message.get("reasoning_content"))

    def echoed_by(self, message: Mapping[str, Any]) -> bool:
        """An omitted ``reasoning_content`` is taken to be this turn's."""
        echo = _Turn.of(message)
        return (
            echo.content == self.content
            and echo.calls == self.calls
            and echo.reasoning in (None, self.reasoning)
        )


@dataclass(frozen=True, slots=True)
class SamplingPolicy:
    """What the run decided the policy may produce, for agents that do not care.

    These are run-level choices, not agent implementation details, so an agent gets
    them without being configured with them -- and an agent with a real reason to
    differ (a cooler summarising sub-call, say) still passes its own.
    """

    temperature: float | None = None
    max_generation_tokens: int | None = None
    max_trajectory_tokens: int | None = None


class CaptureChannel(Protocol):
    """One producer's path into an attempt's capture ledger.

    The channel owns request identity, capture, and the attempt binding; the
    client only renders, samples through it, and parses.
    """

    @property
    def producer(self) -> str: ...

    @property
    def sampling_policy(self) -> SamplingPolicy: ...

    @property
    def samples(self) -> tuple[Sample, ...]: ...

    @property
    def transcript(self) -> TranscriptRecorder | None:
        """Where ``generate`` records its turns; ``None`` when nothing is recorded here.

        A transport that translates requests records the agent's own view itself.
        """
        ...

    async def generate(
        self,
        prompt: RenderedPrompt,
        sampling_params: SamplingParams,
        *,
        conversation_id: str,
        clamped_from: int | None = None,
        on_failed_reply: Callable[[Sample, Exception], None] | None = None,
    ) -> Sample:
        """Sample ``prompt``; ``clamped_from`` is the ``max_tokens`` the caller asked for
        when the trajectory budget lowered it to ``sampling_params.max_tokens``, and
        ``on_failed_reply`` is told of a sample found stale once sampled."""
        ...

    def discard(self, sample_id: str) -> None: ...

    def reply_failed(self, sample_id: str, failure: RendererFailure) -> None:
        """The renderer could not read ``sample_id``'s completion: the attempt has failed."""
        ...


class ContextWindowExceeded(ValueError):
    """The rendered prompt alone fills the trajectory budget: no room is left to generate."""


class ConversationBusy(RuntimeError):
    """A conversation serves one generation at a time, and one is still running."""


class UnsupportedParameterError(ValueError):
    """A well-formed request asked for something this run does not support."""

    def __init__(self, param: str, message: str) -> None:
        super().__init__(message)
        self.param = param


@dataclass(frozen=True, slots=True)
class Renderers:
    """The run's renderer, and one more per declared ``reasoning_effort`` value.

    Each is built once, at startup, through the run's renderer factory; a request
    selects one and never changes its template options. With a ``pool``, a
    conversation renders a thread-safe renderer's prompts on the pool's threads
    instead of the event loop.
    """

    base: Renderer
    by_effort: Mapping[str, Renderer] = field(default_factory=dict[str, Renderer])
    pool: RenderPool | None = None

    async def call[T](self, renderer: Renderer, call: Callable[[], T]) -> T:
        """``call()``, a call on ``renderer``: through the pool when there is one, else here."""
        if self.pool is None:
            return call()
        return await self.pool.run(renderer, call)

    def select(self, reasoning_effort: str | None) -> Renderer:
        if reasoning_effort is None:
            return self.base
        renderer = self.by_effort.get(reasoning_effort)
        if renderer is None:
            declared = ", ".join(sorted(self.by_effort)) or "none"
            raise UnsupportedParameterError(
                "reasoning_effort",
                f"reasoning_effort {reasoning_effort!r} is not declared in "
                f"[model.reasoning_efforts] (declared: {declared})",
            )
        return renderer


class LoopsLLM:
    """A producer-locked model client: every conversation samples as ``producer``."""

    def __init__(self, channel: CaptureChannel, renderers: Renderers):
        self._channel = channel
        self._renderers = renderers

    @property
    def producer(self) -> str:
        return self._channel.producer

    @property
    def renderer(self) -> Renderer:
        """The run's renderer, used when a generation names no ``reasoning_effort``."""
        return self._renderers.base

    @property
    def renderers(self) -> Renderers:
        return self._renderers

    @property
    def samples(self) -> tuple[Sample, ...]:
        """The attempt's captured evidence so far; raises once capture is broken."""
        return self._channel.samples

    def conversation(self, *, conversation_id: str | None = None) -> Conversation:
        """Create a generation session; the caller supplies history on every call."""
        return Conversation(self, conversation_id)


@dataclass(frozen=True, slots=True)
class _BridgeCache:
    """The sampled prefix a following turn can extend, plus the inputs it rendered from,
    and the turns kept unrendered.

    The prefix is the last ``Sample`` itself: its prompt is a span of the attempt's
    token buffer and its chunks are the ones captured, so the bridge holds no copy
    of either.

    Held as one value so a rejected generation can be wound back in a single step;
    restoring these piecemeal is how a conversation ends up bridging onto a sample
    that no longer exists, or rendering a turn a discarded reply had released.
    """

    history: list[Mapping[str, Any]] | None
    tools: list[Mapping[str, Any]]
    sample: Sample | None
    renderer: Renderer | None
    unrendered: dict[str, _Turn]
    discarded: frozenset[str]


class _NotGiven(Enum):
    """Omission inherits run defaults; explicit None remains a sampling option."""

    VALUE = 0


class Conversation:
    """Per-conversation sampling cache, not an agent history manager."""

    def __init__(self, llm: LoopsLLM, conversation_id: str | None):
        self._llm = llm
        self._conversation_id = conversation_id or str(uuid4())
        self._previous_history: list[Mapping[str, Any]] | None = None
        self._previous_tools: list[Mapping[str, Any]] = []
        self._last_sample: Sample | None = None
        self._last_renderer: Renderer | None = None
        self._undo: _BridgeCache | None = None
        # Each reply whose tool call did not parse, by sample: history that only ever
        # continues from its sampled tokens, never re-rendered. Kept as the gateway
        # recognises an echo, so a copy the agent stored differently is still that turn.
        self._unrendered: dict[str, _Turn] = {}
        # Those of them discarded: the turn did not happen, so nothing continues from it.
        self._discarded: set[str] = set()
        self._images = _HistoryImages()
        self._busy = False

    @property
    def _cache(self) -> _BridgeCache:
        return _BridgeCache(
            self._previous_history,
            self._previous_tools,
            self._last_sample,
            self._last_renderer,
            dict(self._unrendered),
            frozenset(self._discarded),
        )

    def _restore(self, cache: _BridgeCache) -> None:
        self._previous_history = cache.history
        self._previous_tools = cache.tools
        self._last_sample = cache.sample
        self._last_renderer = cache.renderer
        self._unrendered = dict(cache.unrendered)
        self._discarded = set(cache.discarded)

    def discard(self, response: LLMResponse) -> None:
        """Reject the generation just sampled, as if the turn had not happened.

        An agent that samples a turn and then declines it -- a cut generation, a
        losing candidate in best-of-n, a speculative probe -- must say so, or the
        rejected completion is still trained with the rollout's advantage. Removing
        it also restores the sampled prefix the next turn bridges from, so re-asking
        the same question renders one clean chain instead of forking.

        A discarded reply whose tool call did not parse cannot be sent back in history:
        its ``history_message`` only continues from tokens that are no longer there.

        Only the most recent generation can be rejected: earlier ones are already
        built on. The sampling record is unaffected; this decides what trains.
        """
        if self._busy:
            raise ConversationBusy("conversation is generating; wait before discarding")
        if self._last_sample is None or self._undo is None:
            raise ValueError("no generation to discard")
        if response.response_id != self._last_sample.sample_id:
            raise ValueError(
                f"only the most recent generation can be discarded; "
                f"{response.response_id!r} is not {self._last_sample.sample_id!r}"
            )
        self._llm._channel.discard(response.response_id)  # pyright: ignore[reportPrivateUsage]
        kept = self._unrendered.get(response.response_id)
        self._restore(self._undo)
        self._undo = None
        if kept is not None:  # still recognised, so sending it back is refused
            self._unrendered[response.response_id] = kept
            self._discarded.add(response.response_id)

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    @property
    def producer(self) -> str:
        return self._llm.producer

    async def generate(
        self,
        sampling_params: SamplingParams | None = None,
        *,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]] | None = None,
        temperature: float | _NotGiven = _NotGiven.VALUE,
        max_tokens: int | None | _NotGiven = _NotGiven.VALUE,
        top_p: float | _NotGiven = _NotGiven.VALUE,
        top_k: int | _NotGiven = _NotGiven.VALUE,
        seed: int | None | _NotGiven = _NotGiven.VALUE,
        stop: str | list[str] | list[int] | None | _NotGiven = _NotGiven.VALUE,
        structured_outputs: Mapping[str, JsonValue] | None | _NotGiven = _NotGiven.VALUE,
        max_trajectory_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """Render the current observation; capture exact evidence before parsing it.

        ``temperature``, ``max_tokens`` and ``max_trajectory_tokens`` fall back to the
        run's sampling policy when the caller leaves them out, so an agent only states
        what it needs to differ on. Passing ``SamplingParams`` opts out of the first
        two entirely -- an explicit object is taken as given. ``reasoning_effort``
        selects one of the run's declared reasoning renderers for this generation.

        A generation never runs past the trajectory budget: ``max_tokens`` is lowered
        to the room the prompt leaves (and is that room when none was asked for), and a
        generation that reaches it stops with ``stop_reason`` ``"length"``. Only a
        prompt that leaves no room at all raises ``ContextWindowExceeded``.
        """
        options = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "stop": stop,
            "structured_outputs": structured_outputs,
        }
        kwargs = {key: value for key, value in options.items() if value is not _NotGiven.VALUE}
        return await self._generate(
            sampling_params,
            messages=messages,
            tools=tools,
            max_trajectory_tokens=max_trajectory_tokens,
            reasoning_effort=reasoning_effort,
            begin_turn=None,
            **kwargs,
        )

    async def _generate(
        self,
        sampling_params: SamplingParams | None,
        *,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]] | None,
        max_trajectory_tokens: int | None,
        reasoning_effort: str | None,
        begin_turn: Callable[[], Callable[[Sample, Exception], None]] | None,
        **kwargs,
    ) -> LLMResponse:
        """``generate``, for a transport that records its own turns with ``begin_turn``.

        It is called where a direct turn begins, once the request is rendered and fits,
        so a request refused before then leaves no turn; it returns what to tell of a
        sample whose reply then raised. Otherwise the channel's transcript records.
        """
        if self._busy:
            raise ConversationBusy("conversation is generating; wait before sampling again")
        if sampling_params is not None and kwargs:
            raise ValueError("pass SamplingParams or keyword sampling options, not both")
        renderer = self._llm.renderers.select(reasoning_effort)
        channel = self._llm._channel  # pyright: ignore[reportPrivateUsage]
        policy = channel.sampling_policy
        if sampling_params is None:
            if policy.temperature is not None:
                kwargs.setdefault("temperature", policy.temperature)
            if policy.max_generation_tokens is not None:
                kwargs.setdefault("max_tokens", policy.max_generation_tokens)
        if max_trajectory_tokens is None:
            max_trajectory_tokens = policy.max_trajectory_tokens
        params = (
            sampling_params.model_copy(deep=True)
            if sampling_params is not None
            else SamplingParams(**kwargs)
        )
        params.stop = _effective_stop(params.stop, renderer.stop_sequences)
        self._busy = True
        try:
            history = deepcopy(self._images.canonical(messages))
            declarations = deepcopy(list(tools or ()))
            rendered: RenderedPrompt | None = None
            bridge = getattr(renderer, "bridge", None)
            previous = self._previous_history
            can_bridge = (
                previous is not None
                # A different renderer tokenizes the same history differently.
                and renderer is self._last_renderer
                and len(history) >= len(previous)
                and _same_render_input(history[: len(previous)], previous)
                and _same_render_input(declarations, self._previous_tools)
                and all(message.get("role") != "assistant" for message in history[len(previous) :])
            )
            last = self._last_sample
            # Rendering is the only CPU-heavy step here, and reads nothing the loop
            # changes meanwhile: this conversation's state is only written after
            # sampling, and every input is this call's own copy or immutable.
            renders = self._llm.renderers
            if can_bridge and last is not None and bridge is not None and previous is not None:
                rendered = await renders.call(
                    renderer, partial(_bridged, bridge, last, previous, history, declarations)
                )
            if rendered is None:
                self._refuse_unrendered(history)
                rendered = await renders.call(
                    renderer, partial(_rendered, renderer, history, declarations)
                )
            prompt = rendered.ids
            clamped_from = None
            if max_trajectory_tokens is not None:
                if max_trajectory_tokens < 1:
                    raise ValueError("max_trajectory_tokens must be positive")
                # The budget keeps every chain within what the trainer can take, so a
                # generation gets the room the prompt leaves, as a model API clamps
                # max_tokens to its context. One token is the least a generation is.
                room = max_trajectory_tokens - len(prompt)
                if room < 1:
                    raise ContextWindowExceeded(
                        f"Prompt ({len(prompt)}) leaves no room to generate within the "
                        f"trajectory budget ({max_trajectory_tokens})"
                    )
                if params.max_tokens is None:
                    params.max_tokens = room  # no limit asked for: the room is the limit
                elif params.max_tokens > room:
                    clamped_from, params.max_tokens = params.max_tokens, room
            # As the agent sent them, with images as ``history`` normalized them.
            recorder = channel.transcript
            if recorder is not None:
                pending = recorder.begin(
                    messages,
                    declarations,
                    rendered=history,
                    named=True,
                    producer=channel.producer,
                    conversation_id=self.conversation_id,
                )
                on_failed_reply = pending.failed
            else:
                pending = None
                on_failed_reply = begin_turn() if begin_turn is not None else None
            sample = await channel.generate(
                rendered,
                params,
                conversation_id=self.conversation_id,
                clamped_from=clamped_from,
                on_failed_reply=on_failed_reply,
            )
            if sample.prompt_ids != prompt:
                raise TrajectoryError("generation backend returned a different prompt")
            # Captured before the first assignment below, so `discard` can put the
            # bridge back exactly as this call found it.
            self._undo = self._cache
            self._last_sample = sample
            self._last_renderer = renderer
            self._previous_history = None
            self._previous_tools = declarations
            try:
                response = self._parsed(renderer, sample, history, declarations)
            except RendererFailure as failure:
                # Told to the lease, not only raised: the attempt fails whether or not
                # the caller lets this propagate. The turn records what the agent got.
                channel.reply_failed(sample.sample_id, failure)
                if on_failed_reply is not None:
                    on_failed_reply(sample, failure)
                raise
            if pending is not None:
                pending.parsed(sample, response.parsed, response.parse_error)
            return response
        finally:
            self._busy = False

    def _parsed(
        self,
        renderer: Renderer,
        sample: Sample,
        history: list[dict[str, Any]],
        declarations: list[Mapping[str, object]],
    ) -> LLMResponse:
        """The reply to ``sample``, and the history the next turn bridges from.

        A reply with a native message bridges from it. A reply without one is the
        model's output when the renderer reports a tool call it could not parse: it is
        returned with each call's status, and bridges from its ``history_message``,
        which is never rendered. Anything else the renderer cannot read -- no native
        message with every call parsed, or one with a call that did not parse -- is its
        own failure, never evidence about the model: ``RendererFailure``.
        """
        try:
            parsed = renderer.parse(
                sample.prompt_ids, sample.completion_ids, deepcopy(declarations)
            )
        except Exception as error:  # whatever a renderer raises is its own fault
            raise RendererFailure(
                f"the renderer could not parse the completion: {type(error).__name__}: {error}"
            ) from error
        try:
            reasoning_tokens = _reasoning_tokens(renderer, sample, parsed)
        except Exception as error:
            raise RendererFailure(
                f"the renderer could not count the reasoning tokens: {type(error).__name__}: {error}"
            ) from error
        if parsed.native_message is None:
            if all(call.status == ToolCallParseStatus.OK for call in parsed.tool_calls):
                raise RendererFailure(
                    "the renderer returned no native assistant message and reported no "
                    "tool call it could not parse"
                )
            response = LLMResponse(
                sample=sample,
                parsed=parsed,
                parse_error=RenderingError("a tool call did not parse; see tool_calls[i].status"),
                reasoning_tokens=reasoning_tokens,
            )
            # The next turn bridges from this reply's tokens, when the agent appends
            # ``history_message``; nothing ever renders it.
            standin = response.history_message
            if standin is not None:
                self._unrendered[sample.sample_id] = _Turn.of(standin)
                self._previous_history = [*history, deepcopy(standin)]
            return response
        if any(call.status != ToolCallParseStatus.OK for call in parsed.tool_calls):
            raise RendererFailure(
                "the renderer returned a native assistant message for a tool call it could "
                "not parse; no message renders back to what the model wrote"
            )
        # The model has now written, and parsed, a turn equal to one kept unrendered: a
        # model that names its calls can retry word for word. Rendering it puts no
        # words in the model's mouth, whichever of the two the agent appends -- unless
        # the history already holds the kept one, written before the model wrote it whole.
        written = _Turn.of(parsed.native_message)
        released = [
            key
            for key, kept in self._unrendered.items()
            if kept == written
            and not any(
                message.get("role") == "assistant" and kept.echoed_by(message)
                for message in history
            )
        ]
        for sample_id in released:
            del self._unrendered[sample_id]
            self._discarded.discard(sample_id)
        self._previous_history = [*history, deepcopy(parsed.native_message)]
        return LLMResponse(
            sample=sample,
            parsed=parsed,
            reasoning_tokens=reasoning_tokens,
        )

    def _refuse_unrendered(self, history: Sequence[Mapping[str, Any]]) -> None:
        """A full render must not render a turn the model did not write as written."""
        if not self._unrendered:
            return
        for message in history:
            if message.get("role") != "assistant":
                continue
            for sample_id, kept in self._unrendered.items():
                if not kept.echoed_by(message):
                    continue
                if sample_id in self._discarded:
                    raise RenderingError(
                        "the history holds a reply whose tool call did not parse and that "
                        "was discarded; the turn did not happen, so there are no sampled "
                        "tokens to continue from"
                    )
                raise RenderingError(
                    "the history holds a reply whose tool call did not parse; it continues "
                    "only from that reply's sampled tokens, which this request does not "
                    "extend (the history before it, the tools or the renderer changed, or the "
                    "renderer cannot bridge)"
                )


def probe_image_tokens(renderer) -> int | None:
    """Render a 32x32 PNG through ``renderer``; ``None`` for text-only renderers.

    Image-capable renderers need their model's image processor to count tokens, and
    that dependency (the ``vision`` extra) is only exercised when an image arrives.
    Rendering a probe up front turns a missing processor into an error before any
    Loops resources are provisioned, instead of a rollout failure after.
    """
    if not getattr(renderer, "supports_images", False):
        return None
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    probe = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "probe"},
                {"type": "image", "image": buffer.getvalue()},
            ],
        }
    ]
    rendered = _as_rendered(renderer.render(probe, []))
    return rendered.prompt_input.length - sum(
        chunk.length for chunk in rendered.prompt_input.chunks if chunk.type == "encoded_text"
    )


def _rendered(
    renderer: Renderer,
    history: Sequence[Mapping[str, Any]],
    declarations: Sequence[Mapping[str, object]],
) -> RenderedPrompt:
    """``history`` rendered in full, from copies the renderer may keep or change."""
    return _as_rendered(renderer.render(deepcopy(history), deepcopy(declarations)))


def _bridged(
    bridge: Callable[..., Any],
    last: Sample,
    previous: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    declarations: Sequence[Mapping[str, object]],
) -> RenderedPrompt | None:
    """The renderer's bridge from ``last`` to ``history``, or ``None`` if it declines.

    ``previous_prompt_input`` is built from the compact evidence for this call alone:
    a bridged prompt's prefix chunks are made from it, and a declined bridge drops it
    here, before the history is rendered in full and sampled.
    """
    options: dict[str, Any] = {"previous_messages": previous}
    if "previous_prompt_input" in inspect.signature(bridge).parameters:
        options["previous_prompt_input"] = last.prompt_input
    bridged = bridge(
        last.prompt_ids,
        last.completion_ids,
        deepcopy(history[len(previous) :]),
        deepcopy(declarations),
        **options,
    )
    if bridged is None:
        return None
    rendered = _as_rendered(bridged)
    # Part by part, as buffers: never a tuple of the whole prefix.
    if not rendered.ids.startswith(last.prompt_ids, last.completion_ids):
        raise RenderingError("Renderer bridge changed the sampled prefix")
    return rendered


def _effective_stop(
    caller: str | list[str] | list[int] | None, renderer: tuple[int, ...] | tuple[str, ...]
) -> str | list[str] | list[int] | None:
    """The caller's stops, with the renderer's end-of-turn stops where they can join.

    One request carries token ids or strings, not both, so a caller's stops of the
    other kind go as given. The sampler still ends the turn at the model's own
    end-of-sequence tokens either way.
    """
    own: list[int | str] = [caller] if isinstance(caller, str) else [*(caller or ())]
    stops = [*own, *renderer]
    if not stops:
        return None
    ids = [stop for stop in stops if isinstance(stop, int)]
    if len(ids) == len(stops):
        return list(dict.fromkeys(ids))
    texts = [stop for stop in stops if isinstance(stop, str)]
    if len(texts) == len(stops):
        return list(dict.fromkeys(texts))
    return own
