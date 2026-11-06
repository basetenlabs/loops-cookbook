"""Image content parts and the text/image chunk structure of a prompt.

Prime renders images to placeholder token runs and reports where they are; Loops
wants the same prompt as text chunks interleaved with ``ImageChunk``s. The two
functions at the bottom convert between those views; they slice token IDs but
never re-tokenize anything.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Mapping, Sequence
from typing import Any

from baseten.loops.models import EncodedTextChunk, ImageChunk, ModelInput

from loops_cookbook.llm.rendering import Message, RenderingError

Placeholder = tuple[int, int]
"""``(offset, length)`` of one image's placeholder run inside the expanded prompt."""


# The three ``renderers.qwen3_vl`` helpers below are private to Prime. They are used
# so we accept exactly the part shapes Prime's renderers accept; re-check them on any
# renderers pin bump (tests/unit/llm/test_rendering.py imports them and fails loudly).


def is_image_part(part: Any) -> bool:
    """Prime's own predicate, so we accept exactly the shapes its renderers accept."""
    from renderers.qwen3_vl import _is_image_part

    return _is_image_part(part)


def is_video_part(part: Any) -> bool:
    from renderers.qwen3_vl import _is_video_part

    return _is_video_part(part)


def image_parts(messages: Sequence[Message]) -> list[Mapping[str, Any]]:
    """Image parts in history order: message order, then content order.

    Images inside a tool message's bundled ``output`` entries count too, in entry order,
    for renderers that flatten those entries (see ``check_messages``).
    """
    found = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if is_image_part(part):
                found.append(part)
            elif isinstance(part, Mapping) and isinstance(part.get("output"), list):
                found.extend(item for item in part["output"] if is_image_part(item))
    return found


def image_bytes(part: Mapping[str, Any]) -> tuple[bytes, str]:
    """Raw image bytes and Loops format for a Prime-shaped image part.

    PNG and JPEG payloads, given as bytes or as a base64 data URI, pass through
    untouched, so the sampler and trainer see exactly the user's file. Anything else
    (PIL, path, URL, other encodings) is loaded through Prime's loader and re-encoded
    as JPEG, which keeps the dimensions and so the token count. Malformed payloads are
    ``RenderingError``s.
    """
    raw = image_payload(part)
    if raw is None:
        raise RenderingError("image part has no image, image_url, url or path")
    if isinstance(raw, str) and raw.startswith("data:") and ";base64," in raw:
        try:
            raw = base64.b64decode(raw.partition(",")[2], validate=True)
        except (ValueError, TypeError) as error:
            raise RenderingError("image data URI is not valid base64") from error
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw)
        fmt = image_format(raw)
        if fmt is not None:
            _check_decodable(raw)
            return raw, fmt
    try:
        from renderers.qwen3_vl import _load_pil_image

        pil = _load_pil_image(dict(part))
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG")
    except (OSError, ValueError, TypeError) as error:
        raise RenderingError(f"could not load image part: {type(error).__name__}") from error
    return buffer.getvalue(), "jpeg"


def untouched_image(part: Mapping[str, Any]) -> tuple[bytes, str] | None:
    """What ``image_bytes`` returns for a part it passes through as it is, unverified,
    or ``None`` when it would decode or load the payload first."""
    raw = image_payload(part)
    if isinstance(raw, (bytes, bytearray)) and (fmt := image_format(bytes(raw))) is not None:
        return bytes(raw), fmt
    return None


def image_payload(part: Mapping[str, Any]) -> Any:
    """What an image part carries, as ``image_bytes`` reads it, or ``None``."""
    raw = part.get("image")
    if raw is None:
        raw = part.get("image_url")
        if isinstance(raw, Mapping):
            raw = raw.get("url")
    if raw is None:
        raw = part.get("url") or part.get("path")
    return raw


def _check_decodable(raw: bytes) -> None:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
    except Exception as error:  # PIL raises several unrelated types here
        raise RenderingError(
            f"image bytes are not a decodable image: {type(error).__name__}"
        ) from error


# The one signature check both accesses use: what passes through untouched directly
# is exactly what the gateway accepts. JPEG is its start-of-image marker and the
# next marker's lead byte.
SIGNATURES = {"png": b"\x89PNG\r\n\x1a\n", "jpeg": b"\xff\xd8\xff"}


def image_format(raw: bytes) -> str | None:
    """``png`` or ``jpeg`` when ``raw`` starts with that format's signature."""
    return next((fmt for fmt, magic in SIGNATURES.items() if raw.startswith(magic)), None)


def model_input_from_placeholders(
    ids: Sequence[int],
    placeholders: Sequence[Placeholder],
    images: Sequence[tuple[bytes, str]],
) -> ModelInput:
    """Split expanded ``ids`` into text and image chunks at Prime's placeholder ranges.

    The count check is load-bearing: a text-only renderer drops image parts silently,
    and this is where that becomes an error instead of a prompt missing its evidence.
    """
    if len(placeholders) != len(images):
        hint = "; images are only rendered in user and tool messages" if not placeholders else ""
        raise RenderingError(
            f"renderer emitted {len(placeholders)} image placeholder(s) "
            f"for {len(images)} image part(s){hint}"
        )
    chunks = []
    cursor = 0
    for (offset, length), (data, fmt) in zip(placeholders, images, strict=True):
        if offset < cursor or length < 1 or offset + length > len(ids):
            raise RenderingError("image placeholder ranges must be ordered and within the prompt")
        if offset > cursor:
            chunks.append(EncodedTextChunk(tokens=list(ids[cursor:offset])))
        chunks.append(ImageChunk(data=data, format=fmt, expected_tokens=length))
        cursor = offset + length
    if cursor < len(ids):
        chunks.append(EncodedTextChunk(tokens=list(ids[cursor:])))
    return ModelInput(chunks=chunks)


def image_placeholders(model_input: ModelInput) -> list[Placeholder]:
    """``(offset, length)`` of each image chunk; the inverse of the function above."""
    result, cursor = [], 0
    for chunk in model_input.chunks:
        if isinstance(chunk, ImageChunk):
            result.append((cursor, chunk.expected_tokens))
        cursor += chunk.length
    return result
