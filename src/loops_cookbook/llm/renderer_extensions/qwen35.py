"""Opt-in Qwen3.5 parsing compatibility; rendering and sampled tokens stay upstream.

The official template emits ``<parameter=NAME>``. The vLLM Qwen3 parser also
accepts whitespace after ``=``; this is compatibility, not a claim that Qwen's
published format requires that tolerance. Prime a3bc9d6 instead includes that
whitespace in the key. Only complete, unambiguous function/parameter blocks are
interpreted here. Source text and token spans are always retained.
"""

import re
from dataclasses import replace

from renderers.base import ToolCallParseStatus
from renderers.configs import Qwen35RendererConfig
from renderers.parsing import _build_param_type_index, _coerce_arg_value
from renderers.qwen35 import Qwen35Renderer

from loops_cookbook.llm.prime import PrimeRenderer

_FUNCTION = re.compile(r"\s*<function=([^>]+)>(.*?)</function>\s*", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=\s*([^>]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def _parse_call(call, param_index):
    if call.status not in (ToolCallParseStatus.OK, ToolCallParseStatus.INVALID_JSON):
        return call
    malformed = replace(call, status=ToolCallParseStatus.MALFORMED_STRUCTURE)
    function = _FUNCTION.fullmatch(call.raw)
    if function is None or not function[1].strip():
        return malformed
    body = function[2]
    params = param_index.get(call.name, {})
    arguments = {}
    offset = 0
    invalid_json = False
    for parameter in _PARAMETER.finditer(body):
        name, text = parameter[1], parameter[2]
        if (
            body[offset : parameter.start()].strip()
            or not name.strip()
            or name in arguments
            or "<tool_call>" in text
            or "<parameter=" in text
            or "<function=" in text
            or "</function>" in text
        ):
            # Duplicate keys and unconsumed/nested markup cannot be faithfully
            # represented by the native argument dict. Never silently drop them.
            return malformed
        schema = params.get(name)
        # _PARAMETER removes only the newline the template writes on each side
        # of a value. Other whitespace belongs to a string value and is kept;
        # json.loads ignores surrounding whitespace for non-string values.
        value, fallback = _coerce_arg_value(text, schema)
        if schema is None:
            # An undeclared key is a tool-validation error, not invalid JSON.
            # Keep Prime's decoded value (including a raw string fallback) and
            # the exact key. The tool can reject it; never guess a replacement.
            fallback = False
        arguments[name] = value
        invalid_json |= fallback
        offset = parameter.end()
    if body[offset:].strip():
        return malformed
    return replace(
        call,
        arguments=arguments,
        status=ToolCallParseStatus.INVALID_JSON if invalid_json else ToolCallParseStatus.OK,
    )


class Qwen35CompatibilityRenderer(Qwen35Renderer):
    """Inherit the pinned template, reasoning parser, stop tokens and exact bridge."""

    def parse_response(self, token_ids, *, tools=None, prompt_ids=None):
        parsed = super().parse_response(token_ids, tools=tools, prompt_ids=prompt_ids)
        params = _build_param_type_index(tools)
        return replace(parsed, tool_calls=[_parse_call(call, params) for call in parsed.tool_calls])


def renderer(tokenizer, chat_template_kwargs):
    """Explicit factory; does not change auto-detection or any other model family."""
    return PrimeRenderer(
        Qwen35CompatibilityRenderer(tokenizer, Qwen35RendererConfig(**dict(chat_template_kwargs)))
    )
