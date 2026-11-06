"""Model client for agents: renders messages, samples through Loops, keeps exact tokens."""

from loops_cookbook.llm.client import (
    ContextWindowExceeded,
    Conversation,
    LLMResponse,
    LoopsLLM,
)
from loops_cookbook.llm.factory import create_renderer, describe
from loops_cookbook.llm.prime import PrimeRenderer
from loops_cookbook.llm.rendering import (
    AssistantMessage,
    Content,
    Message,
    ParsedReply,
    RenderedPrompt,
    Renderer,
    RenderingError,
    SystemMessage,
    Tool,
    ToolMessage,
    ToolSchema,
    UserMessage,
)
from loops_cookbook.llm.transcript import (
    CompletionTurnReply,
    ParsedTurnReply,
    SealedTranscript,
    SealedTurn,
    TranscriptTurn,
    TurnDisposition,
)

__all__ = [
    "AssistantMessage",
    "CompletionTurnReply",
    "Content",
    "ContextWindowExceeded",
    "Conversation",
    "LLMResponse",
    "LoopsLLM",
    "Message",
    "ParsedReply",
    "ParsedTurnReply",
    "PrimeRenderer",
    "RenderedPrompt",
    "Renderer",
    "RenderingError",
    "SealedTranscript",
    "SealedTurn",
    "SystemMessage",
    "Tool",
    "ToolMessage",
    "ToolSchema",
    "TranscriptTurn",
    "TurnDisposition",
    "UserMessage",
    "create_renderer",
    "describe",
]
