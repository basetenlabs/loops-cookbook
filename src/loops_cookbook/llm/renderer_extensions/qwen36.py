"""Renderers for Qwen3.6 and Qwen3.8 that read tool-call arguments as the template wrote them.

Both templates write a string argument as it is and any other value as JSON, as GLM-5.3
does. Qwen3.5 writes booleans and null the way Python does instead. Prime renders both
models correctly, so the cookbook keeps Prime's rendering, bridge and stop tokens. Only
the parse of a call's arguments changes, using ``arguments.xml_arguments_as_written``.
File text keeps its whitespace, and a value that is not its type's JSON reaches the
tool as text. The cookbook selects these renderers for the Qwen3.6 and Qwen3.8 models
that Loops serves, as listed in ``COOKBOOK_RENDERERS``.
"""

from dataclasses import replace

from renderers.configs import Qwen36RendererConfig, Qwen38RendererConfig
from renderers.qwen36 import Qwen36Renderer
from renderers.qwen38 import Qwen38Renderer

from loops_cookbook.llm.prime import PrimeRenderer
from loops_cookbook.llm.renderer_extensions.arguments import xml_arguments_as_written


class Qwen36ArgumentsRenderer(Qwen36Renderer):
    """Extend Prime's Qwen3.6 renderer to read tool arguments as the template wrote them."""

    reads_calls_whole = True  # a block is read whole or reported, never in part

    def parse_response(self, token_ids, *, tools=None, prompt_ids=None):
        parsed = super().parse_response(token_ids, tools=tools, prompt_ids=prompt_ids)
        calls = xml_arguments_as_written(parsed.tool_calls, tools, python_scalars=False)
        return replace(parsed, tool_calls=calls)


class Qwen38ArgumentsRenderer(Qwen38Renderer):
    """Extend Prime's Qwen3.8 renderer to read tool arguments as the template wrote them."""

    reads_calls_whole = True  # a block is read whole or reported, never in part

    def parse_response(self, token_ids, *, tools=None, prompt_ids=None):
        parsed = super().parse_response(token_ids, tools=tools, prompt_ids=prompt_ids)
        calls = xml_arguments_as_written(parsed.tool_calls, tools, python_scalars=False)
        return replace(parsed, tool_calls=calls)


def qwen36(tokenizer, chat_template_kwargs):
    """Build the renderer for Qwen3.6."""
    config = Qwen36RendererConfig(**dict(chat_template_kwargs))
    return PrimeRenderer(Qwen36ArgumentsRenderer(tokenizer, config))


def qwen38(tokenizer, chat_template_kwargs):
    """Build the renderer for Qwen3.8."""
    config = Qwen38RendererConfig(**dict(chat_template_kwargs))
    return PrimeRenderer(Qwen38ArgumentsRenderer(tokenizer, config))
