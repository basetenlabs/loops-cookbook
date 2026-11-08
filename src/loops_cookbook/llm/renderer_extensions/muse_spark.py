"""The cookbook's renderer for Muse Spark Open, with its own parser and bridge.

Prime has no renderer for this format, so prompts come from the model's chat template
itself, through Prime's ``DefaultRenderer``. An assistant turn is a run of messages,
each written as ``<|start|>assistant to=<recipient><|message|>...``. The recipient
``self`` marks reasoning, and ``commentary`` and ``user`` mark text. Any other
recipient marks a tool call, written as ``<atem:function_calls>``. Messages within a
turn end in ``<|eom|>``, and the turn ends in ``<|eot|>``. The template defaults
``current_date`` to today. So pin it in ``renderer_chat_template_kwargs``, or the
system prompt changes during a run.
"""

import html
import re

from renderers.base import ParsedResponse, ParsedToolCall, RenderedTokens, ToolCallParseStatus
from renderers.configs import DefaultRendererConfig
from renderers.default import DefaultRenderer
from renderers.parsing import _build_param_type_index, _coerce_arg_value, _extract_tool_names

from loops_cookbook.llm.prime import PrimeRenderer

_START, _MESSAGE, _EOM, _EOT, _END = (
    "<|start|>",
    "<|message|>",
    "<|eom|>",
    "<|eot|>",
    "<|end_of_text|>",
)
_HEADER = re.compile(r" to=([^<]*)<\|message\|>")
_OPEN, _CLOSE = "<atem:function_calls>", "</atem:function_calls>"
_CALLS = re.compile(r"\s*<atem:function_calls>(.*?)</atem:function_calls>\s*", re.DOTALL)
_INVOKE = re.compile(r'\s*<atem:invoke name="([^"]*)">(.*?)</atem:invoke>\s*', re.DOTALL)
# The template writes a value as it is. So a value ends only at a closing tag that is
# followed by the next parameter or by the invocation's end, as the template writes them.
_PARAMETER = re.compile(
    r'\s*<atem:parameter name="([^"]*)">(.*?)</atem:parameter>'
    r'(?=\s*(?:<atem:parameter name="|</atem:invoke>|\Z))',
    re.DOTALL,
)


def renderer(tokenizer, chat_template_kwargs):
    """Build the renderer for Muse Spark Open, which the cookbook selects for it."""
    built = MuseSparkRenderer(tokenizer, dict(chat_template_kwargs))
    return PrimeRenderer(built, prepare_extension=built.prepare_extension)


class MuseSparkRenderer(DefaultRenderer):
    """Renders with Muse Spark Open's template, and parses and bridges its format by hand."""

    def __init__(self, tokenizer, kwargs):
        super().__init__(tokenizer, DefaultRendererConfig(**kwargs))
        self._stops = [self._id(_EOT), self._id(_END)]

    def _id(self, token):
        [token_id] = self._tokenizer.encode(token, add_special_tokens=False)
        return token_id

    def get_stop_token_ids(self):
        return list(self._stops)

    def parse_response(self, token_ids, *, tools=None, prompt_ids=None):
        """Parse the turn's messages after the prompt's ``<|start|>assistant``.

        Reasoning is every ``self`` message, and content is every ``commentary`` and
        ``user`` message. Each other message is one tool call. Its arguments are typed by
        the tool's schema, as Prime types them. A string parameter is kept verbatim, and
        any other is read as JSON. A value that is not the JSON its schema asks for makes
        the call ``INVALID_JSON``.
        """
        text = self._tokenizer.decode(token_ids, skip_special_tokens=False)
        reasoning = []
        content = []
        calls = []
        reasoning_open = False
        for message in text.split(_START):
            header = _HEADER.match(message.removeprefix("assistant"))
            if header is None:
                continue
            recipient = header.group(1).strip()
            body = message.removeprefix("assistant")[header.end() :]
            closed = body.endswith((_EOM, _EOT, _END))
            body = body.removesuffix(_EOM).removesuffix(_EOT).removesuffix(_END)
            if recipient == "self":
                reasoning.append(body)
                reasoning_open = not closed
            elif recipient in ("commentary", "user"):
                content.append(body)
            else:
                calls.append(_tool_call(recipient, body, tools))
        return ParsedResponse(
            content="\n\n".join(content).strip(),
            reasoning_content="\n\n".join(reasoning).strip() or None,
            tool_calls=calls,
            reasoning_complete=not reasoning_open,
        )

    def prepare_extension(self, previous_messages, new_messages, tools):
        """Name each nameless tool result by its ``tool_call_id``, as the template does.

        The previous assistant turn supplies the tool name for each ID. A result whose
        ID matches none of that turn's calls is left for the bridge to name. That
        happens when the turn's calls did not parse, because such a turn has no calls.
        """
        calls = (previous_messages[-1].get("tool_calls") if previous_messages else None) or []
        names = {
            call.get("id"): (call.get("function") or {}).get("name")
            for call in calls
            if call.get("id")
        }
        prepared = []
        for message in new_messages:
            name = names.get(message.get("tool_call_id"))
            if message.get("role") == "tool" and not message.get("name") and name:
                message = {**message, "name": name}
            prepared.append(message)
        return prepared

    def bridge_to_next_turn(
        self, previous_prompt_ids, previous_completion_ids, new_messages, *, tools=None
    ):
        """Return the sampled turn as it was, then the new messages as the template writes them.

        A turn cut off at the token limit gets the ``<|eot|>`` that the template ends a
        turn with, as Prime's renderers close one in ``trim_to_turn_close``. Its call
        did not parse, so the gateway only continues from its sampled tokens. The bridge
        declines, and the caller renders in full, unless every new message is a tool
        result or a user, system or developer message.

        A result still without a ``name`` after ``prepare_extension`` belongs to a turn
        whose calls did not parse. It takes the name of the sampled call in the same
        position. If that call has no name either, it takes the result's call ID, as
        the template does. So the error feedback is kept without inventing a tool.
        """
        if not previous_completion_ids:
            return None
        closed = (
            previous_completion_ids
            if previous_completion_ids[-1] in self._stops
            else [*previous_completion_ids, self._stops[0]]
        )
        called = [call.name for call in self.parse_response(closed, tools=tools).tool_calls]
        pieces = []
        results = 0
        for message in new_messages:
            role = message.get("role")
            body = _text(message.get("content"))
            if body is None:
                return None
            if role == "tool":
                name = (
                    message.get("name")
                    or (called[results] if results < len(called) else None)
                    or message.get("tool_call_id")
                )
                results += 1
                if not name:
                    return None
                pieces.append(
                    f'{_START}tool {_escape_header(name)}{_MESSAGE}<tool_output name="'
                    f'{_escape_attribute(name)}">\n{body}\n</tool_output>{_EOT}'
                )
            elif role in ("user", "system", "developer"):
                pieces.append(f"{_START}{role}{_MESSAGE}{body}{_EOT}")
            else:
                return None
        added = self._tokenizer.encode(
            "".join(pieces) + _START + "assistant", add_special_tokens=False
        )
        ids = [*previous_prompt_ids, *closed, *added]
        return RenderedTokens(
            token_ids=ids,
            message_indices=[-1] * len(ids),
            sampled_mask=[False] * len(ids),
        )


def _tool_call(recipient, body, tools):
    calls = _CALLS.fullmatch(body)
    if calls is None:
        # The template writes a call's message as one block and nothing else. A second
        # block fails the invocation's checks below, while markup inside a value is kept.
        unclosed = body.lstrip().startswith(_OPEN) and _CLOSE not in body
        status = (
            ToolCallParseStatus.UNCLOSED_BLOCK
            if unclosed
            else ToolCallParseStatus.MALFORMED_STRUCTURE
        )
        return ParsedToolCall(raw=body, name=recipient, status=status)
    invoke = _INVOKE.fullmatch(calls.group(1))
    parameters = None if invoke is None else _parameters_of(invoke.group(2))
    if invoke is None or parameters is None:
        # The body is not exactly one invocation of complete, uniquely named parameters.
        return ParsedToolCall(
            raw=body, name=recipient, status=ToolCallParseStatus.MALFORMED_STRUCTURE
        )
    name = html.unescape(invoke.group(1))
    if not name:
        return ParsedToolCall(raw=body, status=ToolCallParseStatus.MISSING_NAME)
    names = _extract_tool_names(tools)
    head, dot, tail = name.partition(".")
    if dot and head == tail and names and head in names and name not in names:
        # The template's examples name calls "tool.function", so for a tool without
        # a namespace "bash.bash" can only mean bash. Nothing else is renamed, so
        # "bash.command" stays unknown.
        name = head
    if names is not None and name not in names:
        return ParsedToolCall(raw=body, name=name, status=ToolCallParseStatus.UNKNOWN_TOOL)
    schemas = _build_param_type_index(tools).get(name, {})
    arguments = {}
    invalid = False
    for key, text in parameters:
        schema = schemas.get(key)
        arguments[key], fallback = _coerce_arg_value(text, schema)
        # An undeclared key is for the tool to reject, not invalid JSON.
        invalid |= fallback and schema is not None
    status = ToolCallParseStatus.INVALID_JSON if invalid else ToolCallParseStatus.OK
    return ParsedToolCall(raw=body, name=name, arguments=arguments, status=status)


def _parameters_of(text):
    """Return an invocation's parameters as ``(name, value)`` pairs.

    It returns ``None`` unless the text is nothing but complete parameters with
    distinct names.
    """
    found = []
    at = 0
    while (parameter := _PARAMETER.match(text, at)) is not None:
        found.append((html.unescape(parameter.group(1)), parameter.group(2)))
        at = parameter.end()
    if text[at:].strip() or len({key for key, _ in found}) != len(found):
        return None
    return found


def _text(content):
    """Return message content as the template writes it, or ``None`` for a part it can't write."""
    if content is None:
        return ""
    if isinstance(content, str):
        return _escape(content)
    if not isinstance(content, list):
        return None
    texts = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in ("text", "input_text"):
            return None
        texts.append(_escape(part.get("text", "")))
    return "".join(texts)


def _escape(text):
    """Apply the template's ``esc``, so no control sequence or media placeholder survives."""
    for raw, safe in (
        ("<|", "< |"),
        ("<image>", "< image>"),
        ("<video>", "< video>"),
        ("<audio>", "< audio>"),
    ):
        text = text.replace(raw, safe)
    return text


def _escape_header(text):
    return _escape(text).replace("\r", " ").replace("\n", " ")


def _escape_attribute(text):
    return html.escape(_escape_header(text), quote=True).replace("&#x27;", "&#39;")
