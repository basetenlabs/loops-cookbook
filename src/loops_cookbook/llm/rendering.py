"""The renderer contract: what the client needs from any message-to-token provider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict

from baseten.loops.models import ModelInput
from renderers.base import ParsedToolCall

from loops_cookbook.evidence import JsonValue, TokenSpan

# Message values stay open: content parts carry image bytes or PIL images as well
# as JSON, and the selected renderer is what checks them.
Message = Mapping[str, object]
Tool = Mapping[str, JsonValue]

Content = str | Sequence[str | Mapping[str, object]] | None
"""Message content: plain text, or parts whose shapes belong to the renderer.

Open on purpose. Which parts render -- text, images, tool references, the bundled
output of a tool part -- is the selected renderer's to decide, and ``check_messages``
rejects whatever it cannot represent.
"""


class SystemMessage(TypedDict):
    role: Literal["system"]
    content: Content


class UserMessage(TypedDict):
    role: Literal["user"]
    content: Content


class ToolMessage(TypedDict):
    """One tool result, answering a call by the id the model gave it."""

    role: Literal["tool"]
    tool_call_id: str
    name: str
    content: Content


AssistantMessage = Mapping[str, object]
"""The renderer's own assistant history, taken from ``LLMResponse.native_message``.

Deliberately not a closed shape: what a model family puts here differs -- GLM-5.3
carries ``reasoning_content``, others do not -- so an agent appends the message it
was handed instead of building one, and this type promises no fields.
"""


class _OptionalToolFields(TypedDict, total=False):
    description: str
    parameters: Mapping[str, JsonValue]


class ToolSchema(_OptionalToolFields):
    """A function tool: a name, and optionally how to call it.

    The name is the only part the renderer requires. Prime also accepts the wrapped
    ``{"type": "function", "function": {...}}`` envelope; ``Tool`` covers both.
    """

    name: str


class RenderingError(ValueError):
    """A message or renderer result violates the rendering contract."""


class UnsupportedContextError(RenderingError):
    """The selected provider cannot represent this context."""


class RendererFailure(RuntimeError):
    """The renderer could not read a sampled completion: the cookbook's fault, not the model's.

    Raised after sampling, when parsing raises or returns no native message without
    reporting a tool call it could not parse. The attempt fails whatever its caller
    does with this, and is retried as a fresh one; nothing of it trains. A tool call
    the renderer reports as broken is not this: that is the model's output.
    """


def token_ids(values: Sequence[int]) -> TokenSpan:
    """Validate a rendered token sequence, kept at four bytes a token.

    A span is returned as it is: it was checked when it was made, and never changes.
    """
    refused = "Renderer token IDs must be nonempty nonnegative integers below 2**32"
    try:
        result = values if isinstance(values, TokenSpan) else TokenSpan(values)
    except ValueError as error:
        raise RenderingError(refused) from error
    if not result:
        raise RenderingError(refused)
    return result


@dataclass(frozen=True, slots=True)
class ParsedReply:
    content: str
    reasoning_content: str | None
    tool_calls: tuple[ParsedToolCall, ...]
    reasoning_complete: bool | None
    native_message: Message | None = None


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """Expanded prompt token IDs plus the chunk structure the Loops wire needs.

    ``ids`` is what training sees and compares; ``prompt_input`` carries the same
    positions as text and image chunks, so ``prompt_input.length == len(ids)``. ``ids``
    is a ``TokenSpan``: it reads, compares and hashes as the tuple of its tokens, at
    four bytes a token instead of a Python int each.
    """

    ids: TokenSpan
    prompt_input: ModelInput

    def __post_init__(self):
        object.__setattr__(self, "ids", token_ids(self.ids))
        if not isinstance(self.prompt_input, ModelInput):
            raise RenderingError("prompt_input must be a Loops ModelInput")
        if self.prompt_input.length != len(self.ids):
            raise RenderingError(
                f"prompt chunks cover {self.prompt_input.length} positions, "
                f"but the prompt has {len(self.ids)} tokens"
            )

    @classmethod
    def text(cls, ids: Sequence[int]) -> RenderedPrompt:
        ids = token_ids(ids)
        return cls(ids, ModelInput.from_ints(list(ids)))


@dataclass(frozen=True, slots=True)
class RenderedConversation:
    """A whole conversation's tokens, with the renderer's attribution of each one.

    For supervised training, where the targets come from the conversation rather than
    from a sampler. ``owners[k]`` is the index of the message token ``k`` belongs to, or
    -1 for template text no message owns (a preamble, the tool declarations).
    ``generated[k]`` says whether the template has the model produce token ``k`` rather
    than receive it: reasoning, content, tool calls, and the token that ends the turn.
    A generated token belongs to the assistant message whose turn writes it, even where
    the template renders it as the next message's role marker (GLM's ``<|user|>``).
    """

    ids: tuple[int, ...]
    model_input: ModelInput
    owners: tuple[int, ...]
    generated: tuple[bool, ...]

    def __post_init__(self):
        object.__setattr__(self, "ids", tuple(token_ids(self.ids)))
        if not (
            self.model_input.length == len(self.ids) == len(self.owners) == len(self.generated)
        ):
            raise RenderingError(
                "a rendered conversation needs one owner and one generated flag per token"
            )


class Renderer(Protocol):
    """Stateless message formatting and parsing, with provider-native policy.

    ``render`` may return a ``RenderedPrompt`` or, for text-only renderers, a plain
    sequence of token IDs; the client accepts both. An optional ``bridge`` with the
    same return type extends a sampled prefix instead of re-rendering. An optional
    ``reasoning_tokens(completion_ids, parsed)`` counts a reply's reasoning tokens, which
    the Chat Completions gateway reports as ``usage.completion_tokens_details``.
    """

    @property
    def stop_sequences(self) -> tuple[str, ...] | tuple[int, ...]: ...

    def render(
        self, messages: Sequence[Message], tools: Sequence[Tool]
    ) -> RenderedPrompt | Sequence[int]: ...

    def parse(
        self,
        prompt_ids: Sequence[int],
        completion_ids: Sequence[int],
        tools: Sequence[Tool],
    ) -> ParsedReply: ...
