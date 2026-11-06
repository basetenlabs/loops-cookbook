"""Nemotron 3 Ultra and Nemotron 3.5 Super VL.

References, pinned in ``tests/contracts/test_nemotron_extension.py``:
nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 and
nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16, each ``chat_template.jinja`` with
its tokenizer, and Super VL's image processor.

For text, Super VL's template is Ultra's: the same reasoning glue and the same options,
``enable_thinking``, ``truncate_history_thinking`` and ``medium_effort`` (which appends
``{reasoning effort: efficient}`` to the last user message). Prime's Nemotron 3.5
renderer, written for 3.5 Lightning, has no ``medium_effort``, and it writes an image
that follows text without the newline the template puts between them.

Both renderers bridge where Prime's decline:

- Under ``medium_effort``, when no user message is added. The hint then stays where
  it was rendered, and the new turns' tokens are those of the same renderer without it.
- For Super VL, when the conversation has images. The template numbers images within
  one message, so the new turns' tokens do not depend on earlier images.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from renderers import Nemotron3UltraRendererConfig
from renderers.base import MultiModalData, PlaceholderRange
from renderers.nemotron3 import Nemotron3Renderer, Nemotron3UltraRenderer, Nemotron35Renderer

from loops_cookbook.llm.prime import PrimeRenderer

_PROCESSOR_OUTPUTS = ("pixel_values", "imgs_sizes", "num_tokens", "num_patches")
_IMAGE_MARKERS = ("<image>", "<img>", "</img>")
_PER_TOKEN = ("token_ids", "message_indices", "sampled_mask", "is_content")


def ultra(tokenizer, chat_template_kwargs):
    """Renderer factory for Nemotron 3 Ultra."""
    config = Nemotron3UltraRendererConfig(**dict(chat_template_kwargs))
    return PrimeRenderer(UltraRenderer(tokenizer, config))


def super_vl(tokenizer, chat_template_kwargs, *, processor=None):
    """Renderer factory for Nemotron 3.5 Super VL.

    Image parts need the model's processor. It is loaded from ``tokenizer.name_or_path``
    on first use unless ``processor`` is given.
    """
    config = Nemotron3UltraRendererConfig(**dict(chat_template_kwargs))
    return PrimeRenderer(SuperVLRenderer(tokenizer, config, processor=processor))


def _adds_user(messages) -> bool:
    return any(message.get("role") == "user" for message in messages)


def _without_hint(config: Nemotron3UltraRendererConfig) -> Nemotron3UltraRendererConfig:
    return config.model_copy(update={"medium_effort": False})


class UltraRenderer(Nemotron3UltraRenderer):
    """Prime's Ultra renderer, which also bridges under ``medium_effort``."""

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
    """Prime's Nemotron 3.5 renderer, with Ultra's options and the template's image text."""

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
        """Prime's, which approves only the exact casing of the model's name; the Hub,
        and the cookbook's selection, accept any casing."""
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
        """The template's ``render_content``: image parts become ``<image>``, numbered
        ``<image N><image>`` when the message has several, after a newline when they
        follow text. Text that already holds ``<image>`` places the images itself."""
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
        """Prime's, with images found inside the ``<img>`` frame their processor output
        carries, and the image markers that text writes kept as text (``_as_text``)."""
        transformed, images = self._transform_multimodal_messages(
            messages, process_multimodal=process_multimodal
        )
        rendered = Nemotron3Renderer.render(
            self, transformed, tools=tools, add_generation_prompt=add_generation_prompt
        )
        runs = self._image_runs(rendered, 0, images, _owners(messages, images))
        if runs is None:
            raise ValueError("Rendered Nemotron image placeholders were not found.")
        rendered, runs = self._as_text(rendered, 0, runs)
        if not images:
            return rendered
        return replace(rendered, multi_modal_data=_with_images(MultiModalData(), images, runs))

    def _as_text(self, rendered, start: int, runs: list[PlaceholderRange]):
        """``<image>``, ``<img>`` and ``</img>`` written in text, such as an HTML page a
        tool read, tokenize as the image markers. The sampler takes image positions only
        from images, so from ``start`` each marker outside an image's frame is encoded
        as the characters it was written as. Returns the tokens and the moved runs."""
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
        """The placeholder run of each image, in order from ``start``: the frame its
        expansion wrote, ``<img>``, as many ``<image>`` as the processor gave it,
        ``</img>``, searched for only among the tokens of the message that holds the
        image. ``None`` if one is missing."""
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
            # A new user message moves the hint, so Prime declines: process no images.
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
            # Text only, where Prime may have trimmed the reply: the new tokens start
            # where the bridge stops repeating the prompt and the reply.
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
    """How many of ``kept`` begin ``tokens``."""
    for k, (token, expected) in enumerate(zip(tokens, kept)):
        if token != expected:
            return k
    return min(len(tokens), len(kept))


def _owners(messages, images) -> list[int]:
    """The index in ``messages`` of the message holding each image, in the order
    ``_transform_multimodal_messages`` processed them; none when it processed none."""
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
    """The template renders a plain string part in a content list as nothing."""
    if isinstance(content, list) and any(isinstance(part, str) for part in content):
        raise ValueError(
            "Nemotron 3.5 Super VL's template drops plain string parts in a content list; "
            'write them as {"type": "text", "text": ...}'
        )
