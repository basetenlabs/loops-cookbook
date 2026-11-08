"""The model client that an in-process agent calls.

The agent owns its conversation history. The client renders that history, samples
through the attempt's capture channel, and keeps the exact sampled tokens as evidence.
"""

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
from loops_cookbook.llm.images import image_bytes, image_parts, image_payload, is_image_part
from loops_cookbook.llm.render_pool import RenderPool
from loops_cookbook.llm.rendering import (
    ParsedReply,
    RenderedPrompt,
    Renderer,
    RendererFailure,
    RenderingError,
    UnsupportedContextError,
    renders_back,
)
from loops_cookbook.llm.transcript import TranscriptRecorder


def _same_render_input(left: Any, right: Any) -> bool:
    """Compare two render inputs structurally, keeping mapping order and native types.

    A match only permits an optimization, the incremental render. It never permits
    reusing tokens for training. Ordinary dict equality would treat reordered tool
    arguments or declarations as equal, so this comparison also checks order.
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
    """Normalize a conversation's history images to ``{"type": "image", "image": <bytes>}`` parts.

    Every ``generate`` normalizes the caller's full history. An image given as bytes
    or a data URI is the same image whichever turn resends it. So the client decodes it
    once and reuses it while the history holds it. PIL objects, paths and URLs can
    change between calls, so the client loads them again on each call. An agent that
    cares about that cost should pass bytes. The client keeps any other keys on a part.
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
    """Return the renderer's count of the reply's reasoning tokens, or ``None`` without one."""
    count = getattr(renderer, "reasoning_tokens", None)
    return count(sample.completion_ids, parsed) if count is not None else None


def _as_rendered(value: Any) -> RenderedPrompt:
    """Return a ``RenderedPrompt``, accepting plain token IDs from text-only renderers."""
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
    """One generation's reply, holding its exact sample and what the renderer read from it.

    A tool call that the renderer could not parse is the model's output, not an error.
    ``tool_calls`` carries every call the model wrote, each with its ``status``. In
    that case ``native_message`` is ``None`` and ``parse_error`` says why. The
    generation was delivered when ``generate`` returned, so it trains. The agent
    decides what to do about the call. To answer it, append ``history_message``. The
    conversation then continues from the reply's sampled tokens, exactly as the model
    wrote them.
    """

    sample: Sample
    parsed: ParsedReply | None
    parse_error: Exception | None = None
    # The number of completion tokens before the reasoning closed. It is ``None`` when
    # the renderer has no ``reasoning_tokens`` method to count them.
    reasoning_tokens: int | None = None

    @property
    def usage(self) -> TokenUsage:
        """The generation's prompt and completion token counts."""
        return TokenUsage(len(self.sample.prompt_ids), len(self.sample.completion_ids))

    @property
    def response_id(self) -> str:
        """The generation's sample ID, which ``Conversation.discard`` matches against."""
        return self.sample.sample_id

    @property
    def reasoning_complete(self) -> bool | None:
        """Whether the reasoning closed, or ``None`` if the renderer reports no reasoning state."""
        return self.parsed.reasoning_complete if self.parsed is not None else None

    @property
    def content(self) -> str | None:
        """The reply's text, outside its reasoning and tool calls."""
        return self.parsed.content if self.parsed is not None else None

    @property
    def reasoning_content(self) -> str | None:
        """The reply's reasoning text, or ``None`` when it has none."""
        return self.parsed.reasoning_content if self.parsed is not None else None

    @property
    def tool_calls(self):
        """Every tool call the model wrote, each with the ``status`` the renderer gave it."""
        return self.parsed.tool_calls if self.parsed is not None else ()

    @property
    def native_message(self) -> Mapping[str, object] | None:
        """Return a copy of the provider-native assistant message, kept apart from the evidence."""
        return deepcopy(self.parsed.native_message) if self.parsed is not None else None

    @property
    def history_message(self) -> Mapping[str, object] | None:
        """Return the assistant turn to append to history before answering this reply.

        It is ``native_message`` when the renderer built one. A reply with a tool call
        that the renderer could not parse has no native message, because no message
        renders back to what the model wrote. For that reply, this property returns the
        turn as the model wrote it. The turn holds the content, the reasoning and every
        call. Each call carries its ID, its name and its arguments as text. A call
        without an ID gets ``call_<sample id>_<n>``. A call without a name gets ``""``.
        A call whose arguments could not be read carries the raw call text.

        The conversation never renders this message. The next turn bridges from the
        reply's sampled tokens. A conversation that cannot bridge refuses to continue
        rather than put words in the model's mouth. Append the message as it is. A copy
        stored differently still counts as this turn, but a copy with other content or
        call IDs is history the agent wrote.

        It is ``None`` for a completion with nothing parsed.
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
        """Why the sampler stopped, such as ``"length"`` at its token limit."""
        return self.sample.stop_reason

    @property
    def reasoning(self) -> str | None:
        """The same text as ``reasoning_content``."""
        return self.reasoning_content

    @property
    def truncated(self) -> bool:
        """Return whether the generation stopped before the model finished its turn.

        Three signals say a generation is structurally incomplete. The sampler reached
        its token limit, the reasoning never closed, or a tool-call block never closed.
        The renderer reports an unclosed tool-call block as an opening delimiter that
        reached a stop. ``content`` and ``tool_calls`` can't tell any of these apart
        from a finished turn. For example, a generation cut inside its reasoning leaves
        an empty message and no tool calls, which is exactly what a model that chose to
        say nothing produces. So an agent asks here once instead of checking all three
        itself. A cut turn recorded as an answer is graded as an answer.

        Whether every tool call parsed is a separate question, and ``native_message``
        answers it. It is ``None`` when the renderer withheld the message because a
        tool call did not parse. Either way the reply is the model's turn, and
        ``history_message`` is what to append. That is the native message, or the turn
        as written, which continues only from its sampled tokens. A completion that the
        renderer cannot read at all never becomes a reply, because ``generate`` raises
        ``RendererFailure``. A cut generation can parse, and a parsed one can be cut, so
        ask both questions.

        The caller decides what to do about either. ``tool_calls[i].status`` says what
        the renderer made of each call, which is information the model itself does not
        have.
        """
        return (
            self.stop_reason == "length"
            # Compare with False, because None means the renderer reports no reasoning state.
            or self.reasoning_complete is False
            or any(call.status == ToolCallParseStatus.UNCLOSED_BLOCK for call in self.tool_calls)
        )


def _arguments_text(call: ParsedToolCall) -> str:
    """Return a call's arguments as text, as JSON when they were read and raw otherwise."""
    if isinstance(call.arguments, str):
        return call.arguments
    if call.arguments is not None:
        return json.dumps(call.arguments, ensure_ascii=False)
    return call.raw


def canonical_arguments(arguments: object) -> str:
    """Return tool-call arguments as one canonical string, however they were serialized.

    JSON arguments are normalized with sorted keys and no spaces. Text that is not JSON
    is returned unchanged.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True, slots=True)
class _Turn:
    """An assistant turn's content, calls and reasoning, compared the way an echo is recognised.

    ``content: null`` equals an empty string. Each call compares by ID, name and
    canonical arguments. So a copy stored or serialized differently is the same turn.
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
        """Return whether ``message`` repeats this turn.

        A message that omits ``reasoning_content`` is taken to carry this turn's reasoning.
        """
        echo = _Turn.of(message)
        return (
            echo.content == self.content
            and echo.calls == self.calls
            and echo.reasoning in (None, self.reasoning)
        )


@dataclass(frozen=True, slots=True)
class SamplingPolicy:
    """The run's sampling defaults, which an agent gets without configuring them.

    These are run-level choices, not details of the agent's implementation. An agent
    with a real reason to differ, such as a cooler summarising sub-call, still passes
    its own values.
    """

    temperature: float | None = None
    max_generation_tokens: int | None = None
    max_trajectory_tokens: int | None = None


class CaptureChannel(Protocol):
    """One producer's path into an attempt's capture ledger.

    The channel owns request identity, capture and the attempt binding. The client
    only renders, samples through the channel and parses.
    """

    @property
    def producer(self) -> str: ...

    @property
    def sampling_policy(self) -> SamplingPolicy: ...

    @property
    def samples(self) -> tuple[Sample, ...]: ...

    @property
    def transcript(self) -> TranscriptRecorder | None:
        """Return where ``generate`` records its turns, or ``None`` when nothing is recorded here.

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
        """Sample ``prompt`` and capture the result.

        ``clamped_from`` is the ``max_tokens`` the caller asked for, when the trajectory
        budget lowered it to ``sampling_params.max_tokens``. ``on_failed_reply`` is
        called with a sample that turns out to be stale once sampled.
        """
        ...

    def discard(self, sample_id: str) -> None: ...

    def reply_failed(self, sample_id: str, failure: RendererFailure) -> None:
        """Record that the renderer could not read ``sample_id``'s completion.

        That fails the attempt.
        """
        ...


class ContextWindowExceeded(ValueError):
    """Raised when the rendered prompt leaves no room to generate within the trajectory budget."""


class ConversationBusy(RuntimeError):
    """Raised when a conversation is used while one of its generations is still running.

    A conversation serves one generation at a time.
    """


class UnsupportedParameterError(ValueError):
    """Raised when a well-formed request asks for something this run does not support."""

    def __init__(self, param: str, message: str) -> None:
        super().__init__(message)
        self.param = param


@dataclass(frozen=True, slots=True)
class Renderers:
    """The run's renderer, plus one more for each declared ``reasoning_effort`` value.

    The run's renderer factory builds each one once, at startup. A request selects a
    renderer and never changes its template options. When ``pool`` is set, a
    conversation renders a thread-safe renderer's prompts on the pool's threads
    instead of on the event loop. The run's ``[rollout_runtime] has_images`` sets
    ``images``. When it is false, a conversation refuses a request that has an image part
    before it renders the request.
    """

    base: Renderer
    by_effort: Mapping[str, Renderer] = field(default_factory=dict[str, Renderer])
    pool: RenderPool | None = None
    images: bool = True

    async def call[T](self, renderer: Renderer, call: Callable[[], T]) -> T:
        """Run ``call``, a call on ``renderer``, through the pool when there is one."""
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
    """A model client locked to one producer.

    Every conversation it creates samples as ``producer``.
    """

    def __init__(self, channel: CaptureChannel, renderers: Renderers):
        self._channel = channel
        self._renderers = renderers

    @property
    def producer(self) -> str:
        """The producer every conversation from this client samples as."""
        return self._channel.producer

    @property
    def renderer(self) -> Renderer:
        """Return the run's renderer, used when a generation names no ``reasoning_effort``."""
        return self._renderers.base

    @property
    def renderers(self) -> Renderers:
        """The run's renderer and the one for each declared ``reasoning_effort``."""
        return self._renderers

    @property
    def samples(self) -> tuple[Sample, ...]:
        """Return the attempt's captured samples so far.

        Reading this raises once capture is broken.
        """
        return self._channel.samples

    def conversation(self, *, conversation_id: str | None = None) -> Conversation:
        """Create a conversation, which caches sampling state between generations.

        The caller passes the full history on every call.
        """
        return Conversation(self, conversation_id)


@dataclass(frozen=True, slots=True)
class _BridgeCache:
    """The state that the next turn bridges from, held as one value.

    It holds the sampled prefix the next turn can extend, the inputs that prefix was
    rendered from, and the turns kept unrendered. The prefix is the last ``Sample``
    itself. Its prompt is a span of the attempt's token buffer and its chunks are the
    captured ones, so the cache holds no copy of either.

    The state is one value so that a rejected generation can be wound back in a single
    step. Restoring the parts one by one can leave a conversation bridging onto a
    sample that no longer exists, or rendering a turn that a discarded reply had
    released.
    """

    history: list[Mapping[str, Any]] | None
    tools: list[Mapping[str, Any]]
    sample: Sample | None
    renderer: Renderer | None
    unrendered: dict[str, _Turn]
    discarded: frozenset[str]


class _NotGiven(Enum):
    """Marks a sampling option the caller left out.

    A left-out option takes the run's default. An explicit ``None`` still goes to the
    sampler as a value.
    """

    VALUE = 0


class Conversation:
    """One conversation's sampling cache, which lets each turn extend the last sample.

    It is not a history manager. The agent keeps its own history and passes all of it
    to every ``generate`` call.
    """

    def __init__(self, llm: LoopsLLM, conversation_id: str | None):
        self._llm = llm
        self._conversation_id = conversation_id or str(uuid4())
        self._previous_history: list[Mapping[str, Any]] | None = None
        self._previous_tools: list[Mapping[str, Any]] = []
        self._last_sample: Sample | None = None
        self._last_renderer: Renderer | None = None
        self._undo: _BridgeCache | None = None
        # The replies whose tool call did not parse, by sample ID. Each continues only
        # from its sampled tokens and is never rendered again. They are kept in the form
        # used to recognise an echo, so a copy the agent stored differently is that turn.
        self._unrendered: dict[str, _Turn] = {}
        # The kept replies that were discarded. Those turns did not happen, so nothing
        # continues from them.
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

        An agent that samples a turn and then declines it must call this. Examples are
        a cut generation, a losing candidate in best-of-n, or a speculative probe.
        Otherwise the rejected completion still trains with the rollout's advantage.
        Discarding also restores the sampled prefix that the next turn bridges from. So
        asking the same question again renders one clean chain instead of a fork.

        A discarded reply whose tool call did not parse can't be sent back in history.
        Its ``history_message`` continues only from tokens that are no longer there.

        Only the most recent generation can be discarded, because later turns already
        build on the earlier ones. The sampling record does not change. Discarding
        decides only what trains.
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
        # Keep recognising the reply, so that sending it back is refused.
        if kept is not None:
            self._unrendered[response.response_id] = kept
            self._discarded.add(response.response_id)

    @property
    def conversation_id(self) -> str:
        """The conversation's ID, which is a new UUID when the caller gave none."""
        return self._conversation_id

    @property
    def producer(self) -> str:
        """The producer this conversation samples as."""
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
        """Render ``messages``, sample one reply, and capture its exact tokens before parsing.

        When you leave out ``temperature``, ``max_tokens`` or ``max_trajectory_tokens``,
        the client uses the run's sampling policy. So an agent states only what it needs
        to change. Passing ``SamplingParams`` opts out of the policy's ``temperature``
        and ``max_tokens``, because the client takes an explicit object as given.
        ``reasoning_effort`` selects one of the run's declared reasoning renderers for
        this generation.

        A generation never runs past the trajectory budget. The client lowers
        ``max_tokens`` to the room the prompt leaves. When no ``max_tokens`` was asked
        for, that room becomes the limit. A generation that reaches the limit stops with
        ``stop_reason`` ``"length"``. Only a prompt that leaves no room at all raises
        ``ContextWindowExceeded``.
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
        """Run ``generate`` for a transport that records its own turns through ``begin_turn``.

        The client calls ``begin_turn`` where a direct turn begins, once the request is
        rendered and fits the budget. So a request refused before that point leaves no
        turn. ``begin_turn`` returns the callback to tell about a sample whose reply then
        raised. When the channel has a transcript, the transcript records the turn
        instead and ``begin_turn`` is not called.
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
            # Refuse before canonical() decodes data URIs or loads paths and URLs.
            if not self._llm.renderers.images and image_parts(messages):
                raise UnsupportedContextError(
                    "this request has images, but [rollout_runtime] has_images = false. Set "
                    "has_images = true if this rollout runtime's prompts carry images."
                )
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
            # Rendering is the only CPU-heavy step here, so it may run on the pool. It
            # reads nothing the event loop changes meanwhile. This conversation's state
            # is written only after sampling, and every input is a copy or immutable.
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
                # The budget keeps every chain within what the trainer can take. A
                # generation gets the room the prompt leaves, the way a model API clamps
                # max_tokens to its context. A generation needs at least one token.
                room = max_trajectory_tokens - len(prompt)
                if room < 1:
                    raise ContextWindowExceeded(
                        f"the prompt of {len(prompt)} tokens leaves no room to generate "
                        f"within the trajectory budget of {max_trajectory_tokens} tokens"
                    )
                if params.max_tokens is None:
                    # No limit was asked for, so the room is the limit.
                    params.max_tokens = room
                elif params.max_tokens > room:
                    clamped_from, params.max_tokens = params.max_tokens, room
            # The transcript records the messages as the agent sent them, and the
            # rendered history with its images normalized.
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
            # Capture the cache before the assignments below, so that `discard` can put
            # the bridge back exactly as this call found it.
            self._undo = self._cache
            self._last_sample = sample
            self._last_renderer = renderer
            self._previous_history = None
            self._previous_tools = declarations
            try:
                response = self._parsed(renderer, sample, history, declarations)
            except RendererFailure as failure:
                # Tell the lease as well as raising, so the attempt fails whether or not
                # the caller lets this error propagate. The turn records what the agent got.
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
        """Return the reply to ``sample``, and set the history the next turn bridges from.

        A reply with a native message bridges from that message. A reply without one is
        the model's output when the renderer reports a tool call it could not parse. The
        client returns that reply with each call's status, and the next turn bridges
        from its ``history_message``, which is never rendered. Anything else the
        renderer cannot read is the renderer's own failure, never evidence about the
        model, and raises ``RendererFailure``. That covers a missing native message when
        every call parsed, and a native message with a call that did not parse.
        """
        try:
            parsed = renderer.parse(
                sample.prompt_ids, sample.completion_ids, deepcopy(declarations)
            )
        except Exception as error:
            # Whatever a renderer raises is the renderer's own fault.
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
            # When the agent appends ``history_message``, the next turn bridges from
            # this reply's tokens. Nothing ever renders that message.
            standin = response.history_message
            if standin is not None:
                self._unrendered[sample.sample_id] = _Turn.of(standin)
                self._previous_history = [*history, deepcopy(standin)]
            return response
        if not all(renders_back(call) for call in parsed.tool_calls):
            raise RendererFailure(
                "the renderer returned a native assistant message for a tool call it could "
                "not parse; no message renders back to what the model wrote"
            )
        # The model may now have written a parsed turn equal to one kept unrendered,
        # because a model that names its calls can retry word for word. Rendering that
        # turn puts no words in the model's mouth, whichever copy the agent appends. The
        # exception is a history that already holds the kept copy, written before the
        # model wrote the turn whole.
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
        """Refuse a full render that would render a turn kept unrendered.

        The kept turn's message is not exactly what the model wrote, so rendering it
        would put words in the model's mouth.
        """
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
                        "the history holds a discarded reply whose tool call did not parse. "
                        "That turn did not happen, so there are no sampled tokens to "
                        "continue from."
                    )
                raise RenderingError(
                    "the history holds a reply whose tool call did not parse. That reply "
                    "continues only from its sampled tokens, and this request does not "
                    "extend them. The history before it, the tools or the renderer "
                    "changed, or the renderer cannot bridge."
                )


def probe_image_tokens(renderer) -> int | None:
    """Count the image tokens in a 32x32 PNG rendered through ``renderer``.

    It returns ``None`` for a text-only renderer. An image-capable renderer needs its
    model's image processor to count tokens. That dependency, the ``vision`` extra, is
    otherwise used only when an image arrives. Rendering a probe up front turns a
    missing processor into an error before any Loops resources are provisioned,
    instead of a rollout failure later.
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
    """Render ``history`` in full, from copies the renderer may keep or change."""
    return _as_rendered(renderer.render(deepcopy(history), deepcopy(declarations)))


def _bridged(
    bridge: Callable[..., Any],
    last: Sample,
    previous: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    declarations: Sequence[Mapping[str, object]],
) -> RenderedPrompt | None:
    """Return the renderer's bridge from ``last`` to ``history``, or ``None`` if it declines.

    ``previous_prompt_input`` is built from the compact evidence for this call alone. A
    bridged prompt's prefix chunks are made from it. A declined bridge drops it here,
    before the history is rendered in full and sampled.
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
    # Compare part by part, as buffers, never as a tuple of the whole prefix.
    if not rendered.ids.startswith(last.prompt_ids, last.completion_ids):
        raise RenderingError("the renderer's bridge changed the sampled prefix")
    return rendered


def _effective_stop(
    caller: str | list[str] | list[int] | None, renderer: tuple[int, ...] | tuple[str, ...]
) -> str | list[str] | list[int] | None:
    """Return the caller's stops, plus the renderer's end-of-turn stops where they can join.

    One request carries token IDs or strings, not both. So when the caller's stops are
    of the other kind, they go as given. Either way, the sampler still ends the turn at
    the model's own end-of-sequence tokens.
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
