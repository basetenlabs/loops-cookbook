"""An adapter that makes Prime Intellect's renderers meet the cookbook's ``Renderer`` contract."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import renderers
from baseten.loops.models import EncodedTextChunk, ModelInput

from loops_cookbook.llm.images import (
    image_bytes,
    image_parts,
    image_placeholders,
    is_image_part,
    is_video_part,
    model_input_from_placeholders,
)
from loops_cookbook.llm.rendering import (
    Message,
    ParsedReply,
    RenderedConversation,
    RenderedPrompt,
    RenderingError,
    Tool,
    UnsupportedContextError,
    renders_back,
    token_ids,
)


class PrimeRenderer:
    """Wraps a Prime renderer so the cookbook's model client can render and parse with it."""

    def __init__(self, renderer, *, prepare_extension=None):
        self._renderer = renderer
        self._prepare_extension = prepare_extension

    @property
    def stop_sequences(self) -> tuple[int, ...]:
        """The token IDs that end the model's turn, as the Prime renderer gives them."""
        return tuple(self._renderer.get_stop_token_ids())

    @property
    def thread_safe(self) -> bool:
        """Return whether several threads may render and bridge through this renderer at once.

        Prime's text renderers keep nothing between calls. Their tokenizer's encode is
        safe to share while nothing turns on its truncation or padding, and no renderer
        does. Prime's multimodal renderers cache images and load their processor on
        first use, without a lock, so a ``RenderPool`` leaves them on the loop.
        """
        return not renderers.base.is_multimodal(self._renderer)

    @property
    def supports_images(self) -> bool:
        """Return whether the renderer is multimodal with one placeholder per image token.

        Kimi K2.5 renders one ``<|media_pad|>`` per image and lets the engine expand
        it. So its rendered length is not the trainer's ``expected_tokens``, and it
        needs a translation that the cookbook does not implement.
        """
        return renderers.base.is_multimodal(self._renderer) and not _is_kimi(self._renderer)

    def render(self, messages: Sequence[Message], tools: Sequence[Tool] = ()) -> RenderedPrompt:
        """Render ``messages`` and ``tools`` into a prompt that ends where the model's turn starts.

        Messages or images the renderer cannot represent are refused before rendering.
        """
        checked = check_messages(messages, tool_output_images=self._renders_tool_output_images)
        self._require_image_support(checked)
        rendered = _call(
            self._renderer.render, checked, tools=check_tools(tools), add_generation_prompt=True
        )
        ids = token_ids(rendered.token_ids)
        images = [image_bytes(part) for part in _rendered_image_parts(rendered, checked)]
        return RenderedPrompt(
            ids, model_input_from_placeholders(ids, _placeholders(rendered), images)
        )

    def render_conversation(
        self, messages: Sequence[Message], tools: Sequence[Tool] = ()
    ) -> RenderedConversation:
        """Render ``messages`` as history, with each token's owner and generated flag."""
        checked = check_messages(messages, tool_output_images=self._renders_tool_output_images)
        self._require_image_support(checked)
        rendered = _call(
            self._renderer.render, checked, tools=check_tools(tools), add_generation_prompt=False
        )
        ids = token_ids(rendered.token_ids)
        if len(rendered.sampled_mask) != len(ids):
            raise RenderingError(
                f"{type(self._renderer).__name__} does not report which tokens the model "
                "generates, which supervised training needs"
            )
        images = [image_bytes(part) for part in _rendered_image_parts(rendered, checked)]
        return RenderedConversation(
            ids,
            model_input_from_placeholders(ids, _placeholders(rendered), images),
            tuple(rendered.message_indices),
            tuple(bool(flag) for flag in rendered.sampled_mask),
        )

    # ``tokenize`` and ``detokenize`` read Prime's private ``_tokenizer``, which every
    # Prime renderer holds. The Prime pin in pyproject.toml keeps it there.
    def tokenize(self, text: str) -> tuple[int, ...]:
        """Return ``text`` as the model's tokenizer encodes it, special tokens included."""
        return tuple(token_ids(self._renderer._tokenizer.encode(text, add_special_tokens=False)))

    def detokenize(self, ids: Sequence[int]) -> str:
        """Return the exact text of ``ids``, special tokens included."""
        return self._renderer._tokenizer.decode(list(ids), skip_special_tokens=False)

    # Prime's thinking renderers keep their ``</think>`` token ID in a private
    # ``_think_end``, as they keep ``_tokenizer``. The same pin keeps it there.
    def reasoning_tokens(self, completion_ids: Sequence[int], parsed: ParsedReply) -> int | None:
        """Count the completion's tokens before its reasoning closed.

        The count is all of them when the reasoning never closed, and 0 without
        reasoning. It is ``None`` when this renderer has no single-token ``</think>``
        to find.
        """
        end = getattr(self._renderer, "_think_end", None)
        if not isinstance(end, int):
            return None
        # The parser says whether there was reasoning, and the marker only says where it
        # ended. A ``</think>`` written with reasoning off belongs to the answer. Parsers
        # report reasoning of whitespace alone as none, so it counts 0. Telling it apart
        # from a ``</think>`` in an answer needs the prompt, which is too much work for
        # a usage count.
        if parsed.reasoning_complete is False:
            return len(completion_ids)
        if parsed.reasoning_content is None:
            return 0
        return completion_ids.index(end) if end in completion_ids else None

    def bridge(
        self,
        prompt_ids: Sequence[int],
        completion_ids: Sequence[int],
        new_messages: Sequence[Message],
        tools: Sequence[Tool] = (),
        *,
        previous_messages: Sequence[Message] = (),
        previous_prompt_input: ModelInput | None = None,
    ) -> RenderedPrompt | None:
        """Extend the sampled prefix with the new messages, instead of rendering in full.

        This is an optional fast path. The caller must confirm that the earlier history
        and the tools are unchanged. An image-capable renderer requires
        ``previous_prompt_input``. Without the previous prompt's chunk structure, earlier
        images can't be carried into the new prompt. So the bridge declines, and the
        caller renders in full.
        """
        prior = len(prompt_ids) + len(completion_ids)
        messages = check_messages(new_messages, tool_output_images=self._renders_tool_output_images)
        self._require_image_support(messages)
        declarations = check_tools(tools)
        if self._prepare_extension is not None:
            messages = self._prepare_extension(previous_messages, messages, declarations)
        options: dict[str, Any] = {}
        previous_count = 0
        if self.supports_images:
            if previous_prompt_input is None:
                return None
            previous_count = len(image_placeholders(previous_prompt_input))
            # Prime reads any prior data as earlier images, even with no images in it,
            # and its Nemotron 3.5 renderer then declines every bridge.
            if previous_count:
                options["previous_multi_modal_data"] = _slim_multimodal(previous_prompt_input)
        previous_input = previous_prompt_input or ModelInput.from_ints(list(prompt_ids))
        # Prime takes lists, so this call is the one place the bridge expands the prefix.
        result = _call(
            self._renderer.bridge_to_next_turn,
            list(prompt_ids),
            list(completion_ids),
            messages,
            tools=declarations,
            **options,
        )
        if result is None:
            return None
        ids = token_ids(result.token_ids)
        if not ids.startswith(prompt_ids, completion_ids):
            return None
        # Prime merges prior images ahead of the new turn's, so only the tail is new here.
        tail = [
            (offset - prior, length) for offset, length in _placeholders(result)[previous_count:]
        ]
        images = [image_bytes(part) for part in _rendered_image_parts(result, messages)]
        prefix = previous_input.append(EncodedTextChunk(tokens=list(completion_ids)))
        extension = model_input_from_placeholders(ids[prior:], tail, images)
        return RenderedPrompt(ids, ModelInput(chunks=[*prefix.chunks, *extension.chunks]))

    @property
    def _renders_tool_output_images(self) -> bool:
        """Return whether the renderer renders images inside bundled tool ``output`` entries.

        GLM's template renders them, and the cookbook's GLM extension flattens them.
        Prime's own renderers would drop them.
        """
        return bool(getattr(self._renderer, "renders_tool_output_images", False))

    def _require_image_support(self, messages: Sequence[Message]) -> None:
        if image_parts(messages) and not self.supports_images:
            raise UnsupportedContextError(
                f"{type(self._renderer).__name__} does not support image inputs in this cookbook"
            )

    def parse(
        self,
        prompt_ids: Sequence[int],
        completion_ids: Sequence[int],
        tools: Sequence[Tool] = (),
    ) -> ParsedReply:
        """Read the reply from ``completion_ids``.

        The native message is ``None`` when any tool call did not parse, so the caller
        chooses the history to append.
        """
        parsed = self._renderer.parse_response(
            list(token_ids(completion_ids)),
            prompt_ids=list(token_ids(prompt_ids)),
            tools=check_tools(tools),
        )
        calls = tuple(parsed.tool_calls)
        native: dict[str, Any] = {"role": "assistant", "content": parsed.content}
        if parsed.reasoning_content is not None:
            native["reasoning_content"] = parsed.reasoning_content
        if parsed.tool_calls:
            native["tool_calls"] = [
                {
                    "type": "function",
                    **({"id": call.id} if call.id is not None else {}),
                    "function": {"name": call.name, "arguments": deepcopy(call.arguments)},
                }
                for call in parsed.tool_calls
                if call.name is not None
            ]
        # Prime returns every call it attempted. Only a renderer that reads a block whole
        # can vouch for a call it marked unknown_tool or invalid_json; Prime's own parsers
        # may have dropped part of such a block, so for them only an ok call keeps the
        # message. Without one, the caller continues from the sampled tokens.
        whole = bool(getattr(self._renderer, "reads_calls_whole", False))
        if not all(renders_back(call) and (whole or call.status == "ok") for call in calls):
            native = None
        return ParsedReply(
            parsed.content, parsed.reasoning_content, calls, parsed.reasoning_complete, native
        )


# --- message and tool validation ------------------------------------------------------


def check_messages(messages: Sequence[Message], *, tool_output_images: bool = False) -> list[dict]:
    """Copy the history and reject content the renderer would drop or cannot represent.

    ``tool_output_images`` allows image parts inside a tool message's bundled
    ``output`` entries, for renderers that render them.
    """
    result = deepcopy([dict(message) for message in messages])
    for message in result:
        _check_content(
            message.get("content"),
            tool=message.get("role") == "tool",
            tool_output_images=tool_output_images,
        )
    return result


def _check_content(content: Any, *, tool: bool = False, tool_output_images: bool = False) -> None:
    if content is None or isinstance(content, str):
        return
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                continue
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    continue
                if is_image_part(part):
                    continue
                if (
                    tool
                    and part.get("type") == "tool_reference"
                    and isinstance(part.get("name"), str)
                ):
                    continue
                if tool and "output" in part:
                    output = part["output"]
                    if (
                        not tool_output_images
                        and isinstance(output, list)
                        and any(is_image_part(item) for item in output)
                    ):
                        raise UnsupportedContextError(
                            "images inside a tool part's output are not rendered. "
                            "Put them directly in the tool message's content list."
                        )
                    _check_content(output, tool=tool, tool_output_images=tool_output_images)
                    continue
                if part.get("type") == "thinking":
                    raise UnsupportedContextError(
                        "Prime history uses reasoning_content, not thinking content parts"
                    )
                if is_video_part(part) or part.get("type") in {"audio", "audio_url", "input_audio"}:
                    raise UnsupportedContextError("Only text and image content parts are supported")
            raise UnsupportedContextError("Only text and image content parts are supported")
        return
    raise RenderingError("Message content must use the selected renderer's text message format")


def check_tools(tools: Sequence[Tool]) -> list[dict]:
    result = []
    for tool in tools:
        value = deepcopy(dict(tool))
        if "function" in value and value.get("type", "function") != "function":
            raise UnsupportedContextError("Only function tool schemas are supported")
        declaration = value.get("function", value)
        if not isinstance(declaration, dict) or not isinstance(declaration.get("name"), str):
            raise RenderingError("Tool schemas require a function name")
        result.append(value)
    return result


def _call(method, *args, **kwargs):
    """Run a Prime renderer call, turning its failures into RenderingErrors without payloads."""
    try:
        return method(*args, **kwargs)
    except RenderingError:
        raise
    except (ValueError, TypeError, OSError) as error:
        summary = str(error)
        if len(summary) > 200:
            summary = summary[:200] + "..."
        raise RenderingError(
            f"{type(error).__name__} in {type(method.__self__).__name__}: {summary}"
        ) from error


# --- Prime multimodal plumbing --------------------------------------------------------


def _is_kimi(prime_renderer: Any) -> bool:
    try:
        from renderers.kimi_k25 import KimiK25Renderer
    except ImportError:  # pragma: no cover - The renderer set is pinned.
        return False
    return isinstance(prime_renderer, KimiK25Renderer)


def _placeholders(rendered: Any) -> list[tuple[int, int]]:
    data = getattr(rendered, "multi_modal_data", None)
    if data is None:
        return []
    return [(p.offset, p.length) for p in data.mm_placeholders.get("image", [])]


def _rendered_image_parts(rendered: Any, messages: Sequence[Message]) -> list[Message]:
    """Return the image parts in placeholder order.

    A renderer that reorders messages reports the parts it rendered in ``mm_items``.
    For example, the GLM extension sorts tool results into call order, as the Hugging
    Face template does. When a renderer reports its parts, that order wins. Prime's
    own renderers put PIL images there and render messages in the order given, so
    history order is used for them.
    """
    fallback = image_parts(messages)
    data = getattr(rendered, "multi_modal_data", None)
    items = data.mm_items.get("image", []) if data is not None else []
    reported = [item for item in items if is_image_part(item)]
    if reported and len(reported) == len(fallback):
        return reported
    return fallback


def _slim_multimodal(previous_prompt_input: ModelInput):
    """Return the image counts and placeholder ranges that Prime's bridge reads."""
    from renderers.base import MultiModalData, PlaceholderRange

    spans = image_placeholders(previous_prompt_input)
    return MultiModalData(
        mm_hashes={"image": [f"prior-{i}" for i in range(len(spans))]},
        mm_placeholders={"image": [PlaceholderRange(offset=o, length=n) for o, n in spans]},
        mm_items={"image": [{} for _ in spans]},
    )
