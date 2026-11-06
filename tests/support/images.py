"""Small image builders for tests: PNG bytes of a given size, mode and content."""

import io
from pathlib import Path

from PIL import Image, ImageDraw

LETTER_PNG = Path(__file__).resolve().parent / "assets" / "letter.png"


def png(size=(32, 32), color="white") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def lettered_png(size, mode="RGB", label="A"):
    """A PNG of ``size`` with a letter drawn on it, so images differ in content."""
    image = Image.new(mode, size, "white" if mode != "P" else 0)
    ImageDraw.Draw(image).text((2, 2), label, fill="black" if mode != "P" else 1)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def pil(raw):
    return Image.open(io.BytesIO(raw)).convert("RGB")


def letter_png() -> bytes:
    """A 160x160 bold letter A, the fixed image the golden contract test pins."""
    return LETTER_PNG.read_bytes()
