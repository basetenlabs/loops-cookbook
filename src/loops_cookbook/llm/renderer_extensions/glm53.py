"""The cookbook's GLM-5.3 renderer, which extends the pinned Prime GLM-5.1 renderer.

It follows zai-org/GLM-5.3's chat_template.jinja at commit
aca966e4e02791568aa6a4ced368624b3d897f42. Training targets still come from captured
samples, never from this renderer's masks over historical assistant turns.

GLM-5.3-Flash takes images. The Hugging Face template writes each image part as the
literal text ``<|begin_of_image|><|image|><|end_of_image|>``, and the processor then
repeats ``<|image|>`` once for each vision token. This extension emits the repeated
form directly in ``_visible_text``. So Prime's render and bridge walks carry the image
tokens without further changes. The token count comes from the model's own image
processor, exactly as the trainer computes it.
"""

import json
from copy import deepcopy
from dataclasses import replace
from typing import Any, Literal

from pydantic import ConfigDict
from renderers.base import MultiModalData, PlaceholderRange, extract_message_tool_names
from renderers.configs import GLM51RendererConfig
from renderers.glm5 import _TOOLS_FOOTER, _TOOLS_HEADER, GLM51Renderer

IMAGE_TOKEN = "<|image|>"
IMAGE_START, IMAGE_END = "<|begin_of_image|>", "<|end_of_image|>"


def _image_parts(messages):
    """Return the image parts in the order a render emits them, by message and then by content.

    Pass the messages that Prime walks, after ``_prepare``. ``_prepare`` reorders tool
    results into call order, as the Hugging Face template does, so this order can
    differ from history order. The cookbook adapter reads it back from ``mm_items``.
    """
    from renderers.qwen3_vl import _is_image_part

    return [
        part
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and _is_image_part(part)
    ]


class GLM53RendererConfig(GLM51RendererConfig):
    model_config = ConfigDict(frozen=True, extra="forbid")
    _template_fields = GLM51RendererConfig._template_fields | {"reasoning_effort"}
    name: Literal["glm-5.3"] = "glm-5.3"
    enable_thinking: Literal[True] = True
    clear_thinking: bool = False
    reasoning_effort: Literal["low", "high", "max"] = "max"


def _identity(value):
    return value.get("tool_call_id") or value.get("id")


def _ordered_results(block, calls):
    """Sort tool results into call order, as the Hugging Face template does.

    The template sorts only when the call and result IDs make the join unambiguous.
    """
    call_ids = [_identity(call) for call in calls]
    result_ids = [_identity(message) for message in block]
    if (
        not call_ids
        or not all(call_ids)
        or len(set(call_ids)) != len(call_ids)
        or not all(result_ids)
        or len(set(result_ids)) != len(result_ids)
        or not set(result_ids).issubset(call_ids)
    ):
        return block
    order = {identity: i for i, identity in enumerate(call_ids)}
    return sorted(block, key=lambda message: order[_identity(message)])


class GLM53Renderer(GLM51Renderer):
    _config_cls = GLM53RendererConfig
    is_multimodal = True  # renderers.base.is_multimodal honours this.
    renders_tool_output_images = True  # ``_prepare`` flattens bundled tool outputs.

    def __init__(self, tokenizer, config=None, *, processor: Any = None):
        super().__init__(tokenizer, config)
        self._processor = processor
        self._image_token = self._token_id(IMAGE_TOKEN)
        self._image_cache: dict[str, int] = {}

    # --- images ----------------------------------------------------------------

    @property
    def mm_token_type_id_map(self) -> dict[int, int]:
        return {self._image_token: 1}

    def _get_image_processor(self):
        if self._processor is None:
            name = getattr(self._tokenizer, "name_or_path", None)
            if not name:
                raise RuntimeError(
                    "GLM53Renderer needs an image processor for image parts. Pass "
                    "processor=AutoProcessor.from_pretrained(...) or a tokenizer with name_or_path."
                )
            from transformers import AutoProcessor

            # GLM-5.3-Flash ships processor_config.json, not preprocessor_config.json,
            # so AutoImageProcessor cannot load it directly. The trainer does the same.
            self._processor = AutoProcessor.from_pretrained(name)
        return getattr(self._processor, "image_processor", self._processor)

    def _image_tokens(self, part) -> int:
        """Return the vision tokens one image occupies, ``grid_thw.prod() // merge_size**2``."""
        from renderers.qwen3_vl import _image_hash, _load_pil_image

        pil = _load_pil_image(dict(part))
        key = _image_hash(pil)
        if key not in self._image_cache:
            processor = self._get_image_processor()
            out = processor(images=[pil], return_tensors="np")
            grid = out["image_grid_thw"][0]
            merge = int(getattr(processor, "merge_size", 2))
            self._image_cache[key] = int(grid.prod()) // (merge * merge)
        return self._image_cache[key]

    def _visible_text(self, content):
        if content is None:
            return ""
        if isinstance(content, list):
            from renderers.qwen3_vl import _is_image_part

            pieces = []
            for item in content:
                if isinstance(item, str):
                    pieces.append(item)
                elif item.get("type") == "text":
                    pieces.append(item.get("text", ""))
                elif _is_image_part(item):
                    pieces.append(IMAGE_START + IMAGE_TOKEN * self._image_tokens(item) + IMAGE_END)
            return "".join(pieces)
        return str(content)

    def _with_image_placeholders(self, rendered, seen):
        """Attach ``mm_placeholders`` for every maximal run of the image token.

        Also attach the image parts that produced the runs, in the same order. Earlier
        turns, whose parts this call never saw, are marked with empty items.
        """
        ranges, ids, i = [], rendered.token_ids, 0
        while i < len(ids):
            if ids[i] == self._image_token:
                start = i
                while i < len(ids) and ids[i] == self._image_token:
                    i += 1
                ranges.append(PlaceholderRange(offset=start, length=i - start))
            else:
                i += 1
        if not ranges:
            return rendered
        if len(seen) > len(ranges):
            raise ValueError(
                f"rendered {len(seen)} image parts but found {len(ranges)} image token runs"
            )
        items = [{} for _ in range(len(ranges) - len(seen))] + seen
        data = MultiModalData(
            mm_hashes={"image": [f"glm-{r.offset}" for r in ranges]},
            mm_placeholders={"image": ranges},
            mm_items={"image": items},
        )
        return replace(rendered, multi_modal_data=data)

    def bridge_to_next_turn(
        self,
        previous_prompt_ids,
        previous_completion_ids,
        new_messages,
        *,
        tools=None,
        previous_multi_modal_data=None,
    ):
        # Prime's GLM bridge is text-based. Images ride along as repeated ``<|image|>``
        # tokens from ``_visible_text``. Placeholders are recomputed over the full
        # sequence, so earlier images are reported ahead of the new turn's.
        result = super().bridge_to_next_turn(
            previous_prompt_ids, previous_completion_ids, new_messages, tools=tools
        )
        if result is None:
            return None
        return self._with_image_placeholders(result, _image_parts(new_messages))

    # --- text ------------------------------------------------------------------

    def _prepare(self, messages, tools):
        prepared, indices = [], []
        for i, original in enumerate(messages):
            message = deepcopy(original)
            content = message.get("content")
            if (
                message["role"] == "tool"
                and isinstance(content, list)
                and content
                and "output" in content[0]
            ):
                for entry in content:
                    prepared.append({**message, **entry, "content": entry["output"]})
                    indices.append(i)
            else:
                prepared.append(message)
                indices.append(i)
        for message in prepared:
            content = message.get("content")
            if (
                message["role"] == "tool"
                and isinstance(content, list)
                and content
                and content[0].get("type") == "tool_reference"
            ):
                if not tools:
                    raise ValueError("Tool references require tool declarations")
                lines = []
                for reference in content:
                    for declaration in tools or []:
                        tool = declaration.get("function", declaration)
                        if tool["name"] == reference["name"]:
                            lines.append(self._format_tool_spec(tool) + "\n")
                message["content"] = "<tools>\n" + "".join(lines) + "</tools>"
        start = 0
        while start < len(prepared):
            if prepared[start]["role"] != "tool":
                start += 1
                continue
            end = start + 1
            while end < len(prepared) and prepared[end]["role"] == "tool":
                end += 1
            calls = prepared[start - 1].get("tool_calls", []) if start else []
            block = prepared[start:end]
            index_by_object = {
                id(message): index for message, index in zip(block, indices[start:end], strict=True)
            }
            ordered = _ordered_results(block, calls)
            prepared[start:end] = ordered
            indices[start:end] = [index_by_object[id(message)] for message in ordered]
            start = end
        return prepared, indices

    def render(self, messages, *, tools=None, add_generation_prompt=False):
        prepared, indices = self._prepare(messages, tools)
        rendered = super().render(prepared, tools=None, add_generation_prompt=add_generation_prompt)
        prefix = [self._system] + self._encode(
            "Reasoning Effort: " + self.config.reasoning_effort.capitalize()
        )
        if tools:
            included = [
                tool for tool in tools if not tool.get("function", tool).get("defer_loading")
            ]
            text = (
                _TOOLS_HEADER
                + "".join(self._format_tool_spec(tool) + "\n" for tool in included)
                + _TOOLS_FOOTER
            )
            prefix += [self._system] + self._encode(text)
        n = len(prefix)
        rendered = replace(
            rendered,
            token_ids=rendered.token_ids[:2] + prefix + rendered.token_ids[2:],
            message_indices=[-1] * (2 + n)
            + [indices[i] if i >= 0 else -1 for i in rendered.message_indices[2:]],
            sampled_mask=rendered.sampled_mask[:2] + [False] * n + rendered.sampled_mask[2:],
            is_content=(
                rendered.is_content[:2] + [False] * n + rendered.is_content[2:]
                if rendered.is_content
                else []
            ),
            message_roles=[message["role"] for message in messages],
            message_tool_names=extract_message_tool_names(messages),
        )
        return self._with_image_placeholders(rendered, _image_parts(prepared))

    def _render_assistant(
        self, msg, msg_idx, content, last_user_index, *, emit_special, emit_text, emit_text_segments
    ):
        reasoning = msg.get("reasoning_content")
        if not isinstance(reasoning, str):
            reasoning = (
                content.split("</think>")[0].split("<think>")[-1] if "</think>" in content else ""
            )
            if "</think>" in content:
                content = content.split("</think>")[-1]
        if self.config.clear_thinking and msg_idx <= last_user_index:
            reasoning = ""
        emit_special(self._assistant, msg_idx, is_sampled=False, is_content=False)
        emit_special(self._think, msg_idx, is_sampled=False, is_content=False)
        emit_text(reasoning, msg_idx, is_sampled=True, is_content=True)
        emit_special(self._think_end, msg_idx, is_sampled=True, is_content=True)
        emit_text(content.strip(), msg_idx, is_sampled=True, is_content=True)
        for call in msg.get("tool_calls") or []:
            function = call.get("function") or call
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            emit_special(self._tool_call_tok, msg_idx, is_sampled=True, is_content=True)
            emit_text(function["name"], msg_idx, is_sampled=True, is_content=True)
            for key, value in arguments.items():
                emit_special(self._arg_key, msg_idx, is_sampled=True, is_content=True)
                emit_text(key, msg_idx, is_sampled=True, is_content=True)
                emit_special(self._arg_key_end, msg_idx, is_sampled=True, is_content=True)
                emit_special(self._arg_value, msg_idx, is_sampled=True, is_content=True)
                emit_text(
                    value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
                    msg_idx,
                    is_sampled=True,
                    is_content=True,
                )
                emit_special(self._arg_value_end, msg_idx, is_sampled=True, is_content=True)
            emit_special(self._tool_call_end_tok, msg_idx, is_sampled=True, is_content=True)

    def prepare_extension(self, previous_messages, new_messages, tools):
        """Prepare only the new observations, using the preceding assistant turn's calls.

        The previous assistant turn supplies only the call IDs to match, never rendered
        tokens. Prime's inherited bridge keeps the actual sampled token stream.
        """
        preceding = list(previous_messages[-1:])
        prepared, indices = self._prepare(preceding + list(new_messages), tools)
        return [
            message
            for message, index in zip(prepared, indices, strict=True)
            if index >= len(preceding)
        ]


def renderer(tokenizer, chat_template_kwargs, *, processor=None):
    """Build the renderer for GLM-5.3 and GLM-5.3-Flash, which the cookbook selects for them.

    Image parts need the model's image processor. Unless ``processor`` is given, it is
    loaded from ``tokenizer.name_or_path`` on first use.
    """
    from loops_cookbook.llm.prime import PrimeRenderer

    built = GLM53Renderer(
        tokenizer, GLM53RendererConfig(**dict(chat_template_kwargs)), processor=processor
    )
    return PrimeRenderer(built, prepare_extension=built.prepare_extension)
