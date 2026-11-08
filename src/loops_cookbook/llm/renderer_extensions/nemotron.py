"""The cookbook's renderers for Nemotron 3 Ultra and Nemotron 3.5 Super VL.

``tests/contracts/test_nemotron_extension.py`` pins the references. They are the
``chat_template.jinja`` and tokenizer of nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
and of nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16, plus Super VL's image
processor.

For text, Super VL's template is Ultra's. It has the same reasoning glue and the same
options, namely ``enable_thinking``, ``truncate_history_thinking`` and
``medium_effort``. ``medium_effort`` appends ``{reasoning effort: efficient}`` to the
last user message. Prime's Nemotron 3.5 renderer was written for 3.5 Lightning and has
no ``medium_effort``. It also writes an image that follows text without the newline
that the template puts between them.

Both renderers bridge in two cases where Prime's renderers decline:

- Under ``medium_effort``, when no user message is added. The hint then stays where
  it was rendered, and the new turns' tokens are those of the same renderer without it.
- For Super VL, when the conversation has images. The template numbers images within
  one message, so the new turns' tokens do not depend on earlier images.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from renderers import Nemotron3RendererConfig, Nemotron3UltraRendererConfig
from renderers.base import MultiModalData, PlaceholderRange
from renderers.nemotron3 import Nemotron3Renderer, Nemotron3UltraRenderer, Nemotron35Renderer

from loops_cookbook.llm.prime import PrimeRenderer
from loops_cookbook.llm.renderer_extensions.arguments import xml_arguments_as_written

_PROCESSOR_OUTPUTS = ("pixel_values", "imgs_sizes", "num_tokens", "num_patches")
_IMAGE_MARKERS = ("<image>", "<img>", "</img>")
_PER_TOKEN = ("token_ids", "message_indices", "sampled_mask", "is_content")


def ultra(tokenizer, chat_template_kwargs):
    """Build the renderer for Nemotron 3 Ultra."""
    config = Nemotron3UltraRendererConfig(**dict(chat_template_kwargs))
    return PrimeRenderer(UltraRenderer(tokenizer, config))


def nano_super(tokenizer, chat_template_kwargs):
    """Build the renderer for Nemotron 3 Nano and Super, which share Prime's renderer."""
    config = Nemotron3RendererConfig(**dict(chat_template_kwargs))
    return PrimeRenderer(SuperRenderer(tokenizer, config))


def super_vl(tokenizer, chat_template_kwargs, *, processor=None):
    """Build the renderer for Nemotron 3.5 Super VL.

    Image parts need the model's processor. Unless ``processor`` is given, it is loaded
    from ``tokenizer.name_or_path`` on first use.
    """
    config = Nemotron3UltraRendererConfig(**dict(chat_template_kwargs))
    return PrimeRenderer(SuperVLRenderer(tokenizer, config, processor=processor))


def _adds_user(messages) -> bool:
    return any(message.get("role") == "user" for message in messages)


def _without_hint(config: Nemotron3UltraRendererConfig) -> Nemotron3UltraRendererConfig:
    return config.model_copy(update={"medium_effort": False})


class SuperRenderer(Nemotron3Renderer):
    """Extend Prime's Nemotron 3 Super and Nano renderer to read tool arguments as written.

    It reads them back as the template wrote them, using
    ``arguments.xml_arguments_as_written``.
    """

    reads_calls_whole = True  # a block is read whole or reported, never in part

    def parse_response(self, token_ids, *, tools=None, prompt_ids=None):
        parsed = super().parse_response(token_ids, tools=tools, prompt_ids=prompt_ids)
        return replace(parsed, tool_calls=xml_arguments_as_written(parsed.tool_calls, tools))


class UltraRenderer(Nemotron3UltraRenderer):
    """Extend Prime's Ultra renderer to bridge under ``medium_effort``.

    It also reads tool arguments back as the template wrote them, using
    ``arguments.xml_arguments_as_written``.
    """

    reads_calls_whole = True  # a block is read whole or reported, never in part

    def parse_response(self, token_ids, *, tools=None, prompt_ids=None):
        parsed = super().parse_response(token_ids, tools=tools, prompt_ids=prompt_ids)
        return replace(parsed, tool_calls=xml_arguments_as_written(parsed.tool_calls, tools))

    def __init__(self, tokenizer, config: Nemotron3UltraRendererConfig):
        super().__init__(tokenizer, config)
        self._unhinted = (
            Nemotron3UltraRenderer(tokenizer, _without_hint(config)) if self._effort_hint else None
        )

    def bridge_to_next_turn(
        self, previous_prompt_ids, previous_completion_ids, new_messages, *, tools=None
    ):
        bridge = super().bridge_to_next_turn
        if self._unhinted is not None and not _adds_user(new_messages):
            bridge = self._unhinted.bridge_to_next_turn
        return bridge(previous_prompt_ids, previous_completion_ids, new_messages, tools=tools)


class SuperVLRenderer(Nemotron35Renderer):
    """Extend Prime's Nemotron 3.5 renderer with Ultra's options and the template's image text.

    It also reads tool arguments back as the template wrote them, using
    ``arguments.xml_arguments_as_written``.
    """

    reads_calls_whole = True  # a block is read whole or reported, never in part

    def parse_response(self, token_ids, *, tools=None, prompt_ids=None):
        parsed = super().parse_response(token_ids, tools=tools, prompt_ids=prompt_ids)
        return replace(parsed, tool_calls=xml_arguments_as_written(parsed.tool_calls, tools))

    _config_cls = Nemotron3UltraRendererConfig
    supports_multimodal_bridge = True

    def __init__(self, tokenizer, config: Nemotron3UltraRendererConfig, *, processor=None):
        super().__init__(tokenizer, config, processor=processor)
        self._image_frame = (self._token_id("<img>"), self._token_id("</img>"))
        self._as_written = {
            self._token_id(marker): tokenizer.encode(
                marker, add_special_tokens=False, split_special_tokens=True
            )
            for marker in _IMAGE_MARKERS
        }
        self._unhinted = None
        if self._effort_hint:
            self._unhinted = SuperVLRenderer(tokenizer, _without_hint(config))
            self._unhinted._get_processor = self._get_processor

    def _get_processor(self):
        """Load the processor as Prime does, but accept the model's name in any casing.

        Prime approves only the exact casing of the model's name. The Hub and the
        cookbook's renderer selection accept any casing.
        """
        name = getattr(self._tokenizer, "name_or_path", None) or ""
        prefix = self._processor_model_prefix
        if (
            self._processor is None
            and not name.startswith(prefix)
            and name.casefold().startswith(prefix.casefold())
            and not Path(name).expanduser().is_dir()
        ):
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        return super()._get_processor()

    @staticmethod
    def _render_content(content: Any) -> str:
        _refuse_plain_strings(content)
        return Nemotron3Renderer._render_content(content)

    @classmethod
    def _logical_multimodal_content(cls, content: list[Any]) -> tuple[str, list[dict[str, Any]]]:
        """Render content as the template's ``render_content`` does.

        Each image part becomes ``<image>``, numbered ``<image N><image>`` when the
        message has several. An image that follows text comes after a newline. Text
        that already holds ``<image>`` places the images itself.
        """
        _refuse_plain_strings(content)
        images = [item for item in content if cls._is_image_part(item)]
        texts = [item for item in content if item.get("type") in ("text", "input_text")]
        place = "<image>" not in "".join(str(item.get("text") or "") for item in texts)
        rendered, after_text, index = "", False, 0
        for item in content:
            if cls._is_image_part(item):
                index += 1
                if place:
                    if after_text and not rendered.endswith("\n"):
                        rendered += "\n"
                    rendered += f"<image {index}><image>" if len(images) > 1 else "<image>"
                    after_text = False
            elif item.get("type") in ("video", "video_url", "input_video"):
                raise NotImplementedError("Nemotron 3.5 Super VL video parts are not supported")
            elif item.get("type") in ("text", "input_text"):
                text = str(item.get("text") or "")
                rendered += text
                after_text = after_text or bool(text)
        return rendered, images

    def render(self, messages, *, tools=None, add_generation_prompt=False, process_multimodal=True):
        """Render as Prime does, but find images by the ``<img>`` frame around them.

        The processor's output carries that frame. Image markers that text writes are
        kept as text, as ``_as_text`` describes.
        """
        transformed, images = self._transform_multimodal_messages(
            messages, process_multimodal=process_multimodal
        )
        rendered = Nemotron3Renderer.render(
            self, transformed, tools=tools, add_generation_prompt=add_generation_prompt
        )
        runs = self._image_runs(rendered, 0, images, _owners(messages, images))
        if runs is None:
            raise ValueError("the rendered Nemotron image placeholders were not found")
        rendered, runs = self._as_text(rendered, 0, runs)
        if not images:
            return rendered
        return replace(rendered, multi_modal_data=_with_images(MultiModalData(), images, runs))

    def _as_text(self, rendered, start: int, runs: list[PlaceholderRange]):
        """Encode image markers that text writes as the characters they were written as.

        ``<image>``, ``<img>`` and ``</img>`` written in text tokenize as the image
        markers. Such text can be an HTML page that a tool read. The sampler takes image
        positions only from images. So from ``start``, each marker outside an image's
        frame is encoded as its characters. This returns the tokens and the moved runs.
        """
        framed = {k for run in runs for k in range(run.offset - 1, run.offset + run.length + 1)}
        tokens = rendered.token_ids
        if not any(
            tokens[k] in self._as_written and k not in framed for k in range(start, len(tokens))
        ):
            return rendered, runs
        names = [name for name in _PER_TOKEN if len(getattr(rendered, name)) == len(tokens)]
        columns = {name: list(getattr(rendered, name)[:start]) for name in names}
        offsets = {}
        for k in range(start, len(tokens)):
            offsets[k] = len(columns["token_ids"])
            written = None if k in framed else self._as_written.get(tokens[k])
            for name in names:
                value = getattr(rendered, name)[k]
                if name == "token_ids":
                    columns[name].extend(written or [value])
                else:
                    columns[name].extend([value] * len(written or [value]))
        moved = [replace(run, offset=offsets.get(run.offset, run.offset)) for run in runs]
        return replace(rendered, **columns), moved

    def _image_runs(self, rendered, start: int, images, owners) -> list[PlaceholderRange] | None:
        """Return the placeholder run of each image, in order from ``start``.

        Each run is the frame the image's expansion wrote. That frame is ``<img>``, then
        as many ``<image>`` tokens as the processor gave the image, then ``</img>``. A
        frame is searched for only among the tokens of the message that holds the
        image. This returns ``None`` if a frame is missing.
        """
        image, (opening, closing) = self._image_token_id(), self._image_frame
        tokens, indices = list(rendered.token_ids), list(rendered.message_indices)
        runs, cursor = [], start
        for (_, length, _), owner in zip(images, owners, strict=True):
            frame = [opening, *[image] * length, closing]
            while tokens[cursor : cursor + len(frame)] != frame or set(
                indices[cursor : cursor + len(frame)]
            ) != {owner}:
                if opening not in tokens[cursor + 1 :]:
                    return None
                cursor = tokens.index(opening, cursor + 1)
            runs.append(PlaceholderRange(offset=cursor + 1, length=length))
            cursor += len(frame)
        return runs

    def bridge_to_next_turn(
        self,
        previous_prompt_ids,
        previous_completion_ids,
        new_messages,
        *,
        tools=None,
        previous_multi_modal_data: MultiModalData | None = None,
        process_multimodal: bool = True,
    ):
        if self._unhinted is not None and not _adds_user(new_messages):
            return self._unhinted.bridge_to_next_turn(
                previous_prompt_ids,
                previous_completion_ids,
                new_messages,
                tools=tools,
                previous_multi_modal_data=previous_multi_modal_data,
                process_multimodal=process_multimodal,
            )
        if self._effort_hint:
            # A new user message moves the hint, so Prime declines. Process no images.
            return None
        messages, images = self._transform_multimodal_messages(
            new_messages, process_multimodal=process_multimodal
        )
        bridged = Nemotron3Renderer.bridge_to_next_turn(
            self, previous_prompt_ids, previous_completion_ids, messages, tools=tools
        )
        if bridged is None:
            return None
        kept = [*previous_prompt_ids, *previous_completion_ids]
        shared = _shared_prefix(bridged.token_ids, kept)
        if not images and previous_multi_modal_data is None:
            # This is text only, and Prime may have trimmed the reply. So the new tokens
            # start where the bridge stops repeating the prompt and the reply.
            return self._as_text(bridged, shared, [])[0]
        if shared != len(kept):
            return None
        runs = self._image_runs(bridged, len(kept), images, _owners(new_messages, images))
        if runs is None:
            return None
        bridged, runs = self._as_text(bridged, len(kept), runs)
        prior = previous_multi_modal_data or MultiModalData()
        return replace(bridged, multi_modal_data=_with_images(prior, images, runs))


def _with_images(prior: MultiModalData, images, runs) -> MultiModalData:
    return MultiModalData(
        mm_hashes={"image": [*prior.mm_hashes.get("image", []), *(h for _, _, h in images)]},
        mm_placeholders={"image": [*prior.mm_placeholders.get("image", []), *runs]},
        mm_items={
            "image": [
                *prior.mm_items.get("image", []),
                *({key: output[key] for key in _PROCESSOR_OUTPUTS} for output, _, _ in images),
            ]
        },
    )


def _shared_prefix(tokens, kept) -> int:
    """Count how many tokens of ``kept`` begin ``tokens``."""
    for k, (token, expected) in enumerate(zip(tokens, kept)):
        if token != expected:
            return k
    return min(len(tokens), len(kept))


def _owners(messages, images) -> list[int]:
    """Return the index in ``messages`` of the message holding each image.

    The indices follow the order that ``_transform_multimodal_messages`` processed the
    images in. The list is empty when it processed none.
    """
    if not images:
        return []
    return [
        index
        for index, message in enumerate(messages)
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if Nemotron35Renderer._is_image_part(part)
    ]


def _refuse_plain_strings(content: Any) -> None:
    """Refuse plain string parts in a content list, which the template renders as nothing."""
    if isinstance(content, list) and any(isinstance(part, str) for part in content):
        raise ValueError(
            "Nemotron 3.5 Super VL's template drops plain string parts in a content list. "
            'Write them as {"type": "text", "text": ...}.'
        )
