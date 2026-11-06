"""What the gateway serves of Chat Completions: the capability matrix and its policy.

Schema validation and capability are separate. The generated types reject malformed
values; this layer then decides, field by field, what a well-formed request may
ask for. Every field the schema can carry appears in the matrix below as
supported, accepted only at a value with no observable effect, or rejected, and a
rejected or unknown field fails before admission with the canonical
``unsupported_parameter`` error naming it. Nothing is accepted and ignored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, RootModel

from loops_cookbook.evidence import JsonValue
from loops_cookbook.llm.client import UnsupportedParameterError
from loops_cookbook.model_access.openai.schema.models import (
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestDeveloperMessage,
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestToolMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionTool,
    CreateChatCompletionRequest,
    ResponseFormatJsonObject,
    ResponseFormatJsonSchema,
)


class Support(Enum):
    SUPPORTED = "supported"
    DEFAULT_ONLY = "default_only"  # accepted only where it changes nothing
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Capability:
    support: Support
    meaning: str


def supported(meaning: str) -> Capability:
    return Capability(Support.SUPPORTED, meaning)


def default_only(meaning: str) -> Capability:
    return Capability(Support.DEFAULT_ONLY, meaning)


def rejected(meaning: str) -> Capability:
    return Capability(Support.REJECTED, meaning)


# Keyed by the generated model and field. Extensions outside the pinned schema are
# listed under the model that carries them.
MATRIX: Mapping[str, Mapping[str, Capability]] = {
    "CreateChatCompletionRequest": {
        "model": supported("must be the advertised model"),
        "messages": supported("rendered through the run renderer"),
        "temperature": supported("overrides [sampling].temperature"),
        "top_p": supported("passed to the sampler"),
        "seed": supported("passed to the sampler"),
        "max_tokens": supported("overrides [sampling].max_generation_tokens"),
        "max_completion_tokens": supported("alias of max_tokens; the two are exclusive"),
        "stop": supported("strings; the model's own end-of-sequence tokens still end the turn"),
        "stream": supported("server-sent events from the captured response"),
        "stream_options": supported("see ChatCompletionStreamOptions1"),
        "tools": supported("function tools, rendered as declarations"),
        "tool_choice": supported("omitted or auto offers the tools; none renders without them"),
        "reasoning_effort": supported("selects a [model.reasoning_efforts] renderer"),
        "n": default_only("1"),
        "parallel_tool_calls": default_only("true: permission, not a guarantee"),
        "modalities": default_only('["text"]'),
        "response_format": supported("text, json_schema or json_object; enforced by the sampler"),
        "store": default_only("false"),
        "service_tier": default_only("auto"),
        "frequency_penalty": default_only("0"),
        "presence_penalty": default_only("0"),
        "logit_bias": default_only("empty"),
        "logprobs": rejected("the gateway has no token codec"),
        "top_logprobs": rejected("Loops returns no alternatives"),
        "audio": rejected("text output only"),
        "function_call": rejected("deprecated; use tools"),
        "functions": rejected("deprecated; use tools"),
        "metadata": rejected("no downstream consumer"),
        "user": rejected("no downstream consumer"),
        "safety_identifier": rejected("no downstream consumer"),
        "prompt_cache_key": rejected("the sampler has no prompt cache control"),
        "prompt_cache_retention": rejected("the sampler has no prompt cache control"),
        "prompt_cache_options": rejected("the sampler has no prompt cache control"),
        "moderation": rejected("no moderation"),
        "prediction": rejected("no predicted outputs"),
        "verbosity": rejected("the run renderer decides"),
        "web_search_options": rejected("no hosted tools"),
        # Extension: accepted empty; run-level options belong in
        # [model].renderer_chat_template_kwargs, reasoning in reasoning_effort.
        "chat_template_kwargs": default_only("empty"),
    },
    "ChatCompletionStreamOptions1": {
        "include_usage": supported("a final usage chunk"),
        "include_obfuscation": default_only("false"),
    },
    "ChatCompletionRequestDeveloperMessage": {
        "role": supported("rendered as a system message"),
        "content": supported("text"),
        "name": rejected("participant names are not rendered"),
    },
    "ChatCompletionRequestSystemMessage": {
        "role": supported("system"),
        "content": supported("text"),
        "name": rejected("participant names are not rendered"),
    },
    "ChatCompletionRequestUserMessage": {
        "role": supported("user"),
        "content": supported("text and base64 PNG or JPEG images"),
        "name": rejected("participant names are not rendered"),
    },
    "ChatCompletionRequestAssistantMessage": {
        "role": supported("assistant history"),
        "content": supported("text; null equals empty"),
        "tool_calls": supported("function calls"),
        "refusal": default_only("null: the gateway returns none"),
        "audio": rejected("text output only"),
        "function_call": rejected("deprecated; use tool_calls"),
        "name": rejected("participant names are not rendered"),
        # Extension (vLLM convention): prior reasoning, kept or dropped by the renderer.
        "reasoning_content": supported("passed to the renderer"),
    },
    "ChatCompletionRequestToolMessage": {
        "role": supported("tool result"),
        "content": supported("text"),
        "tool_call_id": supported("names the call it answers"),
    },
    "ChatCompletionRequestFunctionMessage": {
        "role": rejected("deprecated; use tool messages"),
        "content": rejected("deprecated; use tool messages"),
        "name": rejected("deprecated; use tool messages"),
    },
    "ChatCompletionRequestMessageContentPartText": {
        "type": supported("text"),
        "text": supported("rendered"),
        "prompt_cache_breakpoint": rejected("the sampler has no prompt cache control"),
    },
    "ChatCompletionRequestMessageContentPartImage": {
        "type": supported("image_url"),
        "image_url": supported("base64 data URI only; the gateway never fetches"),
        "prompt_cache_breakpoint": rejected("the sampler has no prompt cache control"),
    },
    "ImageUrl": {
        "url": supported("data:image/png or data:image/jpeg, base64"),
        "detail": default_only("auto"),
    },
    "ChatCompletionRequestMessageContentPartAudio": {
        "type": rejected("no audio input"),
        "input_audio": rejected("no audio input"),
        "prompt_cache_breakpoint": rejected("no audio input"),
    },
    "ChatCompletionRequestMessageContentPartFile": {
        "type": rejected("no file input"),
        "file": rejected("no file input"),
        "prompt_cache_breakpoint": rejected("no file input"),
    },
    "ChatCompletionRequestMessageContentPartRefusal": {
        "type": rejected("the gateway returns no refusals"),
        "refusal": rejected("the gateway returns no refusals"),
    },
    "ChatCompletionMessageToolCall": {
        "id": supported("the call's identity"),
        "type": supported("function"),
        "function": supported("name and JSON arguments"),
    },
    "Function": {
        "name": supported("the called function"),
        "arguments": supported("a JSON object, as a string"),
    },
    "ResponseFormatText": {
        "type": supported("text"),
    },
    "ChatCompletionMessageCustomToolCall": {
        "id": rejected("custom tools are not supported"),
        "type": rejected("custom tools are not supported"),
        "custom": rejected("custom tools are not supported"),
    },
    "ChatCompletionTool": {
        "type": supported("function"),
        "function": supported("see FunctionObject"),
    },
    "FunctionObject": {
        "name": supported("declared to the renderer"),
        "description": supported("declared to the renderer"),
        "parameters": supported("declared to the renderer"),
        "strict": default_only("false: tool arguments are not constrained"),
    },
    # tool_choice objects: required and named choices constrain generation.
    "ChatCompletionAllowedToolsChoice": {
        "type": rejected("constrained tool choice is not supported"),
        "allowed_tools": rejected("constrained tool choice is not supported"),
    },
    "ChatCompletionAllowedTools": {
        "mode": rejected("constrained tool choice is not supported"),
        "tools": rejected("constrained tool choice is not supported"),
    },
    "ChatCompletionNamedToolChoice": {
        "type": rejected("constrained tool choice is not supported"),
        "function": rejected("constrained tool choice is not supported"),
    },
    "ChatCompletionNamedToolChoiceCustom": {
        "type": rejected("constrained tool choice is not supported"),
        "custom": rejected("constrained tool choice is not supported"),
    },
    "Function2": {"name": rejected("constrained tool choice is not supported")},
    "Custom1": {"name": rejected("constrained tool choice is not supported")},
    # response_format objects other than plain text constrain generation in the sampler.
    "ResponseFormatJsonObject": {"type": supported("any JSON object")},
    "ResponseFormatJsonSchema": {
        "type": supported("json_schema"),
        "json_schema": supported("see JsonSchema"),
    },
    "JsonSchema": {
        "name": supported("required by the schema; not rendered"),
        "description": rejected("not rendered: state the format in the messages"),
        "schema_": supported("the JSON Schema the completion must match; required"),
        "strict": supported("either value: the sampler always enforces the schema"),
    },
    "CustomToolChatCompletions": {
        "type": rejected("custom tools are not supported"),
        "custom": rejected("custom tools are not supported"),
    },
}


# Free-form by design: a function's parameters are a JSON Schema the renderer declares.
OPAQUE = frozenset({"FunctionParameters", "ResponseFormatJsonSchemaSchema"})


class ModelNotFoundError(LookupError):
    """The request names a model this connection does not serve."""


class InvalidRequestError(ValueError):
    """A well-formed request the gateway cannot interpret, such as a dangling tool result."""

    def __init__(self, param: str, message: str) -> None:
        super().__init__(message)
        self.param = param


@dataclass(frozen=True, slots=True)
class SamplingOptions:
    """Only what the request supplied; the run's defaults apply to the rest."""

    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    stop: tuple[str, ...] | None = None
    # vLLM structured_outputs, passed through the Loops SDK unchanged.
    structured_outputs: dict[str, JsonValue] | None = None


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """A request the run can serve, in schema terms; translation renders it."""

    messages: tuple[BaseModel, ...]
    tools: tuple[ChatCompletionTool, ...]
    offer_tools: bool
    sampling: SamplingOptions
    reasoning_effort: str | None
    stream: bool
    include_usage: bool


_MESSAGE_TYPES = (
    ChatCompletionRequestDeveloperMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestToolMessage,
)


def admit(request: CreateChatCompletionRequest, *, model: str) -> ChatRequest:
    """Apply the matrix to a schema-valid request, or raise naming the first violation."""
    choice = request.tool_choice
    if choice not in (None, "auto", "none"):
        raise UnsupportedParameterError("tool_choice", "tool_choice must be omitted, auto, or none")
    _check_fields(request, "")
    if request.model != model:
        raise ModelNotFoundError(f"The model {request.model!r} does not exist; use {model!r}")
    if request.max_tokens is not None and request.max_completion_tokens is not None:
        raise UnsupportedParameterError(
            "max_completion_tokens", "max_tokens and max_completion_tokens are exclusive"
        )
    max_tokens = request.max_completion_tokens or request.max_tokens
    for name in ("max_tokens", "max_completion_tokens"):
        value = getattr(request, name)
        if value is not None and value < 1:
            raise InvalidRequestError(name, f"{name} must be positive")
    options = request.stream_options
    if options is not None and not request.stream:
        raise InvalidRequestError("stream_options", "stream_options requires stream")
    tools: list[ChatCompletionTool] = []
    for index, tool in enumerate(request.tools or ()):
        if not isinstance(tool, ChatCompletionTool):
            raise UnsupportedParameterError(f"tools[{index}]", "only function tools are supported")
        tools.append(tool)
    structured_outputs = _structured_outputs(request.response_format)
    if structured_outputs is not None and tools and choice != "none":
        # The sampler holds the constraint from the first token: no tool call could start.
        raise UnsupportedParameterError(
            "response_format", "a JSON response_format cannot be combined with offered tools"
        )
    for index, message in enumerate(request.messages):
        if not isinstance(message, _MESSAGE_TYPES):
            raise UnsupportedParameterError(
                f"messages[{index}].role", "function messages are deprecated; use tool messages"
            )
    return ChatRequest(
        messages=tuple(request.messages),
        tools=tuple(tools),
        offer_tools=choice != "none",
        sampling=SamplingOptions(
            temperature=_root(request.temperature),
            top_p=_root(request.top_p),
            seed=request.seed,
            max_tokens=max_tokens,
            stop=_stop(request.stop),
            structured_outputs=structured_outputs,
        ),
        reasoning_effort=request.reasoning_effort,
        stream=bool(request.stream),
        include_usage=bool(options is not None and options.include_usage),
    )


def _structured_outputs(value: object) -> dict[str, JsonValue] | None:
    """The sampler's constraint for a response format; plain text has none."""
    if isinstance(value, ResponseFormatJsonObject):
        return {"json_object": True}
    if isinstance(value, ResponseFormatJsonSchema):
        schema = value.json_schema.schema_
        if schema is None:
            raise InvalidRequestError(
                "response_format.json_schema.schema", "json_schema requires a schema"
            )
        return {"json": schema.model_dump(by_alias=True)}
    return None


def _root(value: RootModel[float] | None) -> float | None:
    return None if value is None else value.root


def _stop(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, RootModel):
        value = value.root  # pyright: ignore[reportUnknownMemberType]
        if value is None:
            return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    raise InvalidRequestError("stop", "stop must be a string or a list of strings")


# Values that have no observable effect, for DEFAULT_ONLY fields.
_DEFAULTS: Mapping[str, Mapping[str, object]] = {
    "CreateChatCompletionRequest": {
        "n": 1,
        "parallel_tool_calls": True,
        "modalities": ["text"],
        "store": False,
        "service_tier": "auto",
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "logit_bias": {},
        "chat_template_kwargs": {},
    },
    "ChatCompletionStreamOptions1": {"include_obfuscation": False},
    "ChatCompletionRequestAssistantMessage": {"refusal": None},
    "ImageUrl": {"detail": "auto"},
    "FunctionObject": {"strict": False},
}


def _check_fields(value: BaseModel, path: str) -> None:
    """Every supplied field, recursively, must be supported or at its no-op default.

    An explicit ``null`` is treated as omitted, as the schema defines it.
    """
    kind = type(value).__name__
    if kind in OPAQUE:
        return
    # A model the matrix does not list accepts nothing, so a schema upgrade that
    # adds one fails loudly until someone decides what it means.
    table = MATRIX.get(kind, {})
    extra = value.model_extra or {}
    supplied = {name: getattr(value, name) for name in value.model_fields_set} | extra
    # A part or message is named by its kind first: what it is decides the rest.
    for name, field in sorted(supplied.items(), key=lambda item: item[0] not in ("type", "role")):
        if field is None:
            continue
        param = f"{path}.{name}" if path else name
        capability = table.get(name)
        if capability is None or capability.support is Support.REJECTED:
            reason = capability.meaning if capability is not None else "not a supported field"
            raise UnsupportedParameterError(param, f"Unsupported parameter: {param} ({reason})")
        if capability.support is Support.DEFAULT_ONLY and _plain(field) != _DEFAULTS[kind][name]:
            raise UnsupportedParameterError(
                param, f"Unsupported value for {param}; only {capability.meaning} is accepted"
            )
    for name in value.model_fields_set:
        _check_nested(getattr(value, name), f"{path}.{name}" if path else name)


def _plain(value: object) -> object:
    if isinstance(value, RootModel):
        return _plain(value.root)  # pyright: ignore[reportUnknownMemberType]
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_unset=True)
    return value


def _check_nested(value: object, path: str) -> None:
    if isinstance(value, RootModel):
        _check_nested(value.root, path)  # pyright: ignore[reportUnknownMemberType]
    elif isinstance(value, BaseModel):
        _check_fields(value, path)
    elif isinstance(value, Sequence) and not isinstance(value, str):
        for index, item in enumerate(value):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            _check_nested(item, f"{path}[{index}]")


def text_parts(content: object, param: str) -> list[ChatCompletionRequestMessageContentPartText]:
    """A text-only content value as parts; ``param`` names it in errors."""
    if isinstance(content, RootModel):
        content = content.root  # pyright: ignore[reportUnknownMemberType]
    if isinstance(content, str):
        return [ChatCompletionRequestMessageContentPartText(type="text", text=content)]
    parts: list[ChatCompletionRequestMessageContentPartText] = []
    for index, part in enumerate(content if isinstance(content, list) else []):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        if isinstance(part, RootModel):  # system and tool parts wrap the text part
            part = part.root  # pyright: ignore[reportUnknownMemberType]
        if not isinstance(part, ChatCompletionRequestMessageContentPartText):
            raise UnsupportedParameterError(
                f"{param}[{index}].type", "only text parts are supported here"
            )
        parts.append(part)
    return parts
