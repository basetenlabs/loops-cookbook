"""The renderer contract, which says what the client needs from any renderer.

A renderer turns messages into tokens and parses sampled tokens back into a reply.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict

from baseten.loops.models import ModelInput
from renderers.base import ParsedToolCall

from loops_cookbook.evidence import JsonValue, TokenSpan

# Message values stay open, because content parts carry image bytes or PIL images as
# well as JSON. The selected renderer checks them.
Message = Mapping[str, object]
Tool = Mapping[str, JsonValue]

Content = str | Sequence[str | Mapping[str, object]] | None
"""Message content, which is plain text or a list of parts whose shapes the renderer defines.

The type is open on purpose. The selected renderer decides which parts render, such as
text, images, tool references, or the bundled output of a tool part.
``check_messages`` rejects whatever the renderer cannot represent.
"""


class SystemMessage(TypedDict):
    role: Literal["system"]
    content: Content


class UserMessage(TypedDict):
    role: Literal["user"]
    content: Content


class ToolMessage(TypedDict):
    """One tool result, answering a call by the ID the model gave it."""

    role: Literal["tool"]
    tool_call_id: str
    name: str
    content: Content


AssistantMessage = Mapping[str, object]
"""The renderer's own assistant message for history, taken from ``LLMResponse.native_message``.

This type is deliberately not a closed shape, because model families put different
fields here. For example, GLM-5.3 carries ``reasoning_content`` and others do not. So
an agent appends the message it was handed instead of building one, and this type
promises no fields.
"""


class _OptionalToolFields(TypedDict, total=False):
    description: str
    parameters: Mapping[str, JsonValue]


class ToolSchema(_OptionalToolFields):
    """A function tool, with a name and optionally a description of how to call it.

    The name is the only part the renderer requires. Prime also accepts the wrapped
    ``{"type": "function", "function": {...}}`` envelope, and ``Tool`` covers both.
    """

    name: str


class RenderingError(ValueError):
    """Raised when a message or a renderer's result violates the rendering contract."""


class UnsupportedContextError(RenderingError):
    """Raised when the selected renderer cannot represent this context."""


class RendererFailure(RuntimeError):
    """Raised when the renderer could not read a sampled completion.

    This is the cookbook's fault, not the model's. It is raised after sampling, when
    parsing raises, or when it returns no native message without reporting a tool call
    it could not parse. The attempt fails whatever its caller does with this error, and
    it is retried as a fresh attempt. Nothing of it trains. A tool call that the
    renderer reports as broken is not this failure, because that call is the model's
    output.
    """


def token_ids(values: Sequence[int]) -> TokenSpan:
    """Validate a rendered token sequence, and keep it at four bytes a token.

    A ``TokenSpan`` is returned as it is, because it was checked when it was made and
    never changes.
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
    """A prompt's expanded token IDs, plus the chunk structure that Loops requests need.

    ``ids`` is what training sees and compares. ``prompt_input`` carries the same
    positions as text and image chunks, so ``prompt_input.length == len(ids)``. ``ids``
    is a ``TokenSpan``. It reads, compares and hashes as the tuple of its tokens, but it
    takes four bytes a token instead of a Python int each.
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

    Supervised training uses it, because its targets come from the conversation
    rather than from a sampler. ``owners[k]`` is the index of the message that token
    ``k`` belongs to. It is -1 for template text that no message owns, such as a
    preamble or the tool declarations. ``generated[k]`` says whether the template has
    the model produce token ``k`` rather than receive it. Generated tokens are the
    reasoning, the content, the tool calls, and the token that ends the turn. A
    generated token belongs to the assistant message whose turn writes it. That holds
    even where the template renders the token as the next message's role marker, such
    as GLM's ``<|user|>``.
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
    """Formats messages into tokens and parses sampled tokens, without keeping state.

    ``render`` may return a ``RenderedPrompt``. A text-only renderer may instead return
    a plain sequence of token IDs, and the client accepts both. An optional ``bridge``
    with the same return type extends a sampled prefix instead of rendering again. An
    optional ``reasoning_tokens(completion_ids, parsed)`` counts a reply's reasoning
    tokens. The Chat Completions gateway reports that count in
    ``usage.completion_tokens_details``.
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
