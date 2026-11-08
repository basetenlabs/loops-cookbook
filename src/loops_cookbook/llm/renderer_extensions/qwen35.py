"""An opt-in Qwen3.5 renderer that reads tool-call arguments as the template wrote them.

It keeps Prime's rendering, bridge and stop tokens, and reads arguments with
``arguments.xml_arguments_as_written``. Prime strips every parameter value, so file
text loses its indentation and final newline while the call reports success. Prime
also marks a call ``invalid_json`` when a value is not the JSON its schema asks for.
The vLLM Qwen3 parser accepts whitespace after ``<parameter=``, but Prime puts that
whitespace in the parameter name. This renderer always keeps the source text and the
token spans.
"""

from dataclasses import replace

from renderers.configs import Qwen35RendererConfig
from renderers.qwen35 import Qwen35Renderer

from loops_cookbook.llm.prime import PrimeRenderer
from loops_cookbook.llm.renderer_extensions.arguments import xml_arguments_as_written


class Qwen35CompatibilityRenderer(Qwen35Renderer):
    """Prime's Qwen3.5 renderer with vLLM-compatible tool-call parsing.

    It inherits the pinned template, reasoning parser, stop tokens and exact bridge.
    """

    reads_calls_whole = True  # a block is read whole or reported, never in part

    def parse_response(self, token_ids, *, tools=None, prompt_ids=None):
        parsed = super().parse_response(token_ids, tools=tools, prompt_ids=prompt_ids)
        return replace(parsed, tool_calls=xml_arguments_as_written(parsed.tool_calls, tools))


def renderer(tokenizer, chat_template_kwargs):
    """Build the Qwen3.5 compatibility renderer.

    Select it explicitly with ``[model] renderer``. It changes neither auto-detection
    nor any other model family.
    """
    return PrimeRenderer(
        Qwen35CompatibilityRenderer(tokenizer, Qwen35RendererConfig(**dict(chat_template_kwargs)))
    )
