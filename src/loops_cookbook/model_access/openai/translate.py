"""Chat Completions messages to renderer input, and the parsed reply back to the wire.

Translation never decides attribution and never changes evidence: it only says what
the renderer should render and how the captured completion reads as a Chat
Completions response.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, RootModel
from renderers.base import ToolCallParseStatus

from loops_cookbook.llm.client import LLMResponse, UnsupportedParameterError, canonical_arguments
from loops_cookbook.llm.images import SIGNATURES, image_format
from loops_cookbook.model_access.openai.policy import ChatRequest, InvalidRequestError, text_parts
from loops_cookbook.model_access.openai.schema.models import (
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk,
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestDeveloperMessage,
    ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestToolMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponseDelta,
    Choice,
    Choice1,
    CompletionTokensDetails,
    CompletionUsage,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    Function,
    Function1,
)

RendererMessage = dict[str, object]
FinishReason = Literal["stop", "length", "tool_calls"]

# A completion's ``id`` is this prefix and the sample ID it returned.
COMPLETION_ID_PREFIX = "chatcmpl-"


class AssistantMessage(ChatCompletionResponseMessage):
    """The response message, with reasoning kept apart from the answer (vLLM's field)."""

    reasoning_content: str | None = None


class AssistantDelta(ChatCompletionStreamResponseDelta):
    """A stream delta, with the same reasoning extension."""

    reasoning_content: str | None = None


@dataclass(frozen=True, slots=True)
class _Call:
    id: str
    name: str
    arguments: str  # canonical JSON


@dataclass(frozen=True, slots=True)
class _Returned:
    content: str
    reasoning_content: str | None
    calls: tuple[_Call, ...]
    native: Mapping[str, object]


@dataclass(slots=True)
class Returned:
    """The assistant messages this conversation returned, to recognise their echoes.

    An agent sends back what it was given, as the Chat Completions contract
    expects; the renderer's own native message is substituted for that echo so
    the conversation bridges exactly as a direct one does.
    """

    messages: list[_Returned] = field(default_factory=list[_Returned])

    def remember(self, message: AssistantMessage, native: Mapping[str, object]) -> None:
        calls = tuple(_call(call) for call in message.tool_calls or ())
        entry = _Returned(message.content or "", message.reasoning_content, calls, native)
        self.messages.append(entry)

    def recall(self, message: ChatCompletionRequestAssistantMessage) -> Mapping[str, object] | None:
        """The native message ``message`` echoes, compared after canonicalization.

        ``content: null`` equals an empty string, an omitted ``reasoning_content``
        is taken to be the one returned, and tool calls compare by ID, name, and
        canonical JSON arguments.
        """
        content = "".join(part.text for part in text_parts(message.content or "", "content"))
        calls = tuple(_call(call) for call in message.tool_calls or ())
        reasoning = (message.model_extra or {}).get("reasoning_content")
        for returned in reversed(self.messages):
            if (
                returned.content == content
                and returned.calls == calls
                and reasoning in (None, returned.reasoning_content)
            ):
                return returned.native
        return None


def _call(call: BaseModel) -> _Call:
    if not isinstance(call, ChatCompletionMessageToolCall):
        raise UnsupportedParameterError("tool_calls", "only function tool calls are supported")
    return _Call(call.id, call.function.name, canonical_arguments(call.function.arguments))


def renderer_input(
    request: ChatRequest, returned: Returned
) -> tuple[list[RendererMessage], list[RendererMessage]]:
    """The request's messages and tools in the run renderer's input shape."""
    names = _tool_names(request.messages)
    messages = [
        _message(message, index, returned, names) for index, message in enumerate(request.messages)
    ]
    tools: list[RendererMessage] = []
    if request.offer_tools:
        for tool in request.tools:
            function: dict[str, object] = {"name": tool.function.name}
            if tool.function.description is not None:
                function["description"] = tool.function.description
            if tool.function.parameters is not None:
                function["parameters"] = tool.function.parameters.model_dump(exclude_unset=True)
            tools.append({"type": "function", "function": function})
    return messages, tools


def _tool_names(messages: Sequence[BaseModel]) -> dict[str, str]:
    names: dict[str, str] = {}
    for message in messages:
        if isinstance(message, ChatCompletionRequestAssistantMessage):
            for call in message.tool_calls or ():
                if isinstance(call, ChatCompletionMessageToolCall):
                    names[call.id] = call.function.name
    return names


def _message(
    message: BaseModel, index: int, returned: Returned, names: Mapping[str, str]
) -> RendererMessage:
    where = f"messages[{index}]"
    if isinstance(
        message, (ChatCompletionRequestDeveloperMessage, ChatCompletionRequestSystemMessage)
    ):
        return {"role": "system", "content": _text_content(message.content, f"{where}.content")}
    if isinstance(message, ChatCompletionRequestUserMessage):
        return {"role": "user", "content": _user_content(message.content, f"{where}.content")}
    if isinstance(message, ChatCompletionRequestToolMessage):
        name = names.get(message.tool_call_id)
        if name is None:
            raise InvalidRequestError(
                f"{where}.tool_call_id",
                f"no assistant tool call in this request has id {message.tool_call_id!r}",
            )
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "name": name,
            "content": _text_content(message.content, f"{where}.content"),
        }
    if isinstance(message, ChatCompletionRequestAssistantMessage):
        native = returned.recall(message)
        if native is not None:
            return dict(native)
        return _assistant(message, where)
    raise UnsupportedParameterError(f"{where}.role", "unsupported message role")


def _assistant(message: ChatCompletionRequestAssistantMessage, where: str) -> RendererMessage:
    """History the gateway did not return itself, in the renderer's assistant shape."""
    content = "".join(part.text for part in text_parts(message.content or "", f"{where}.content"))
    result: RendererMessage = {"role": "assistant", "content": content}
    reasoning = (message.model_extra or {}).get("reasoning_content")
    if reasoning is not None:
        if not isinstance(reasoning, str):
            raise InvalidRequestError(
                f"{where}.reasoning_content", "reasoning_content must be text"
            )
        result["reasoning_content"] = reasoning
    calls: list[RendererMessage] = []
    for number, call in enumerate(message.tool_calls or ()):
        param = f"{where}.tool_calls[{number}]"
        if not isinstance(call, ChatCompletionMessageToolCall):
            raise UnsupportedParameterError(param, "only function tool calls are supported")
        try:
            arguments = json.loads(call.function.arguments)
        except json.JSONDecodeError as error:
            raise InvalidRequestError(
                f"{param}.function.arguments", f"arguments are not JSON: {error}"
            ) from None
        if not isinstance(arguments, dict):
            raise InvalidRequestError(f"{param}.function.arguments", "arguments must be an object")
        calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": arguments},
            }
        )
    if calls:
        result["tool_calls"] = calls
    return result


def _text_content(content: object, param: str) -> str | list[RendererMessage]:
    if isinstance(content, str):
        return content
    return [{"type": "text", "text": part.text} for part in text_parts(content, param)]


def _user_content(content: object, param: str) -> str | list[RendererMessage]:
    if isinstance(content, RootModel):
        content = content.root  # pyright: ignore[reportUnknownMemberType]
    if isinstance(content, str):
        return content
    parts: list[RendererMessage] = []
    for index, part in enumerate(content if isinstance(content, list) else []):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        where = f"{param}[{index}]"
        if isinstance(part, ChatCompletionRequestMessageContentPartText):
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ChatCompletionRequestMessageContentPartImage):
            parts.append({"type": "image", "image": image_bytes(str(part.image_url.url), where)})
        else:
            raise UnsupportedParameterError(
                f"{where}.type", "only text and image parts are supported"
            )
    return parts


def image_bytes(url: str, param: str) -> bytes:
    """A base64 PNG or JPEG data URI's bytes, unchanged; nothing else is ever read.

    The gateway never fetches a URL or reads a host file, so every other form of
    image reference, and video, is refused before sampling.
    """
    where = f"{param}.image_url.url"
    header, comma, payload = url.partition(",")
    if not url.startswith("data:") or not comma:
        raise UnsupportedParameterError(where, "images must be base64 data: URIs")
    kind, _, encoding = header.removeprefix("data:").partition(";")
    if encoding != "base64":
        raise UnsupportedParameterError(where, "image data URIs must be base64")
    fmt = kind.removeprefix("image/") if kind.startswith("image/") else None
    if fmt not in SIGNATURES:
        raise UnsupportedParameterError(where, "images must be image/png or image/jpeg")
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise InvalidRequestError(where, "the image payload is not valid base64") from None
    if image_format(data) != fmt:
        raise InvalidRequestError(where, f"the image payload is not {kind}")
    return data


# ------------------------------------------------------------------ responses


@dataclass(frozen=True, slots=True)
class Completion:
    """One captured generation as the Chat Completions response the agent receives."""

    sample_id: str
    response: CreateChatCompletionResponse
    message: AssistantMessage
    finish_reason: FinishReason

    def body(self) -> bytes:
        return self.response.model_dump_json(exclude_none=True, serialize_as_any=True).encode()

    def events(self, *, include_usage: bool) -> list[bytes]:
        """The response as server-sent events, ending with ``[DONE]``."""
        common = {
            "id": self.response.id,
            "created": self.response.created,
            "model": self.response.model,
            "object": "chat.completion.chunk",
        }
        message = self.message
        deltas = [AssistantDelta(role="assistant")]
        if message.reasoning_content:
            deltas.append(AssistantDelta(reasoning_content=message.reasoning_content))
        if message.content:
            deltas.append(AssistantDelta(content=message.content))
        for index, call in enumerate(message.tool_calls or ()):
            if isinstance(call, ChatCompletionMessageToolCall):
                chunk = ChatCompletionMessageToolCallChunk(
                    index=index,
                    id=call.id,
                    type="function",
                    function=Function1(name=call.function.name, arguments=call.function.arguments),
                )
                deltas.append(AssistantDelta(tool_calls=[chunk]))
        chunks = [
            CreateChatCompletionStreamResponse(
                **common,
                choices=[Choice1(index=0, delta=delta, finish_reason=None)],  # pyright: ignore[reportArgumentType]
            )
            for delta in deltas
        ]
        chunks.append(
            CreateChatCompletionStreamResponse(
                **common,  # pyright: ignore[reportArgumentType]
                choices=[
                    Choice1(index=0, delta=AssistantDelta(), finish_reason=self.finish_reason)
                ],
            )
        )
        if include_usage and self.response.usage is not None:
            chunks.append(
                CreateChatCompletionStreamResponse(
                    **common,  # pyright: ignore[reportArgumentType]
                    choices=[],
                    usage=self.response.usage.model_dump(),  # pyright: ignore[reportArgumentType]
                )
            )
        events = [
            b"data: "
            + chunk.model_dump_json(exclude_none=True, serialize_as_any=True).encode()
            + b"\n\n"
            for chunk in chunks
        ]
        return [*events, b"data: [DONE]\n\n"]


def error_body(
    message: str, *, kind: str, code: str | None, param: str | None = None, **extra: object
) -> dict[str, object]:
    """The OpenAI error envelope every refusal and failure is written in."""
    return {"error": {"message": message, "type": kind, "param": param, "code": code, **extra}}


@dataclass(frozen=True, slots=True)
class InvalidOutput:
    """A generation with a tool call the renderer could not parse: the model's output.

    The agent receives it as ``422 model_output_invalid``, whose ``error.reply`` holds
    what the model wrote: its text, its reasoning, and each call as an assistant tool
    call (``id``, ``type``, ``function``: the name and the arguments as written) with
    its ``raw`` text and parse ``status``; and the generation's ``usage``, as a
    completion's, so an agent counting its context counts this call too. It is delivered like any response, so it
    trains; what to do next is the agent's. To answer the calls, the agent echoes
    ``message`` (the reply without ``id``, ``raw`` and ``status``) and its tool
    results: the gateway recognises the echo and continues from the sampled tokens.
    """

    sample_id: str
    code: Literal["tool_call_truncated", "tool_call_parse_error"]
    reply: Mapping[str, object]
    message: AssistantMessage

    def body(self) -> bytes:
        message = (
            "the model's reply was cut short inside a tool call"
            if self.code == "tool_call_truncated"
            else "the model's reply has a tool call that does not parse"
        )
        return json.dumps(
            error_body(
                f"{message}; error.reply holds the reply",
                kind="model_output_invalid",
                code=self.code,
                reply=self.reply,
            )
        ).encode()


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def completion(
    reply: LLMResponse, *, model: str, created: int, tools_offered: bool
) -> Completion | InvalidOutput:
    """The Chat Completions view of a captured generation.

    A reply with a tool call the renderer could not parse is ``InvalidOutput``: the
    call is the model's, and is never silently dropped. A completion whose request
    offered no tools, as with ``tool_choice: "none"``, is still a response: a call the
    model wrote anyway is not returned. A tool call without an ID gets one derived
    from the sample and its position, so tool results and bridging can refer to it.
    """
    sample_id = reply.sample.sample_id
    written = reply.history_message
    if written is not None and any(
        call.status != ToolCallParseStatus.OK for call in reply.tool_calls
    ):
        calls = [
            ChatCompletionMessageToolCall.model_validate(call)
            for call in _sequence(written.get("tool_calls"))
        ]
        return InvalidOutput(
            sample_id,
            "tool_call_truncated" if reply.truncated else "tool_call_parse_error",
            {
                "id": f"{COMPLETION_ID_PREFIX}{sample_id}",
                "content": reply.content,
                "reasoning_content": reply.reasoning_content,
                "tool_calls": [
                    {
                        **call.model_dump(mode="json"),
                        "raw": parsed.raw,
                        "status": parsed.status.value,
                    }
                    for call, parsed in zip(calls, reply.tool_calls, strict=True)
                ],
                "usage": _usage(reply).model_dump(exclude_none=True),
            },
            AssistantMessage(
                role="assistant",
                content=reply.content or "",
                refusal=None,
                reasoning_content=reply.reasoning_content,
                tool_calls=calls,  # pyright: ignore[reportArgumentType]
            ),
        )
    calls: list[ChatCompletionMessageToolCall] = []
    if reply.native_message is not None and tools_offered:
        for index, call in enumerate(reply.tool_calls):
            if call.name is None:
                continue
            arguments = call.arguments
            text = (
                arguments
                if isinstance(arguments, str)
                else json.dumps(arguments or {}, ensure_ascii=False)
            )
            calls.append(
                ChatCompletionMessageToolCall(
                    id=call.id or f"call_{sample_id}_{index}",
                    type="function",
                    function=Function(name=call.name, arguments=text),
                )
            )
    message = AssistantMessage(
        role="assistant",
        content=reply.content or "",
        refusal=None,
        reasoning_content=reply.reasoning_content,
        tool_calls=calls or None,  # pyright: ignore[reportArgumentType]
    )
    finish: FinishReason = "length" if reply.truncated else "tool_calls" if calls else "stop"
    usage = _usage(reply)
    response = CreateChatCompletionResponse(
        id=f"{COMPLETION_ID_PREFIX}{sample_id}",
        object="chat.completion",
        created=created,
        model=model,
        choices=[Choice(index=0, message=message, finish_reason=finish, logprobs=None)],
        usage=usage,
    )
    return Completion(sample_id, response, message, finish)


def _usage(reply: LLMResponse) -> CompletionUsage:
    """The generation's tokens, with its reasoning's when the renderer counted them."""
    details = (
        CompletionTokensDetails(reasoning_tokens=reply.reasoning_tokens)
        if reply.reasoning_tokens is not None
        else None
    )
    return CompletionUsage(
        prompt_tokens=reply.usage.prompt_tokens,
        completion_tokens=reply.usage.completion_tokens,
        total_tokens=reply.usage.total_tokens,
        completion_tokens_details=details,
    )
