"""Count the circles in a picture: an image prompt, scored on the host.

Each task is a ``CountTask``: how many circles to draw, and the seed that places them.
``run()`` draws the picture, sends it to the model as an image part with the question,
and keeps the number the model replies with. ``grade()`` takes a quarter point off for
each circle miscounted, so a near miss scores above a far one. The circles overlap, which
vision models count badly; that gap is what training closes.
"""

from __future__ import annotations

import io
import random
import re

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from loops_cookbook.episodes import Episode, Grading, Sampling, Task
from loops_cookbook.runtime import JsonValue

SIZE = 448  # pixels on a side; 196 image tokens on Qwen3.6
FEWEST, MOST = 9, 20  # circles in a picture; up to 20 always fit at SEPARATION
# How far apart two circles must be: |dx| + |dy| + |dr| in pixels. One of the three then
# differs by more than 12 px, four stroke widths, so no two outlines nearly coincide.
# At 24, two of 20 circles could still read as one.
SEPARATION = 36
QUESTION = "How many circles are in this picture? Reply with the number only."


class CountTask(BaseModel):
    """One line of the dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str  # for people reading the dataset; the episode does not use it
    count: int = Field(ge=FEWEST, le=MOST)
    seed: int


def place(count: int, seed: int) -> list[tuple[int, int, int]]:
    """``count`` circles as ``(x, y, r)``, placed by ``seed`` inside the picture. Two
    circles may overlap but are always more than ``SEPARATION`` apart, so every circle
    can be told apart."""
    rng = random.Random(seed)
    circles: list[tuple[int, int, int]] = []
    while len(circles) < count:
        r = rng.randint(30, 70)
        x, y = rng.randint(r + 4, SIZE - r - 4), rng.randint(r + 4, SIZE - r - 4)
        if all(abs(x - a) + abs(y - b) + abs(r - c) > SEPARATION for a, b, c in circles):
            circles.append((x, y, r))
    return circles


def draw(count: int, seed: int) -> bytes:
    """A PNG of the circles ``place()`` gives, as outlines. The same task always draws
    the same picture."""
    circles = place(count, seed)
    image = Image.new("RGB", (SIZE, SIZE), "white")
    pen = ImageDraw.Draw(image)
    for x, y, r in circles:
        pen.ellipse((x - r, y - r, x + r, y + r), outline="black", width=3)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def parse_count(reply: str | None) -> int | None:
    """The first number in the reply when it is a whole count, or ``None``: scored as
    wrong. A sign or a fraction is part of the number, so ``-12`` and ``12.9`` are not 12."""
    match = re.search(r"-?\d+(\.\d+)?", reply or "")
    if match is None or match.group().startswith("-") or match.group(1):
        return None
    return int(match.group())


def score(guess: int | None, count: int) -> float:
    """1 for the right count, a quarter point less for each circle off, never below 0."""
    if guess is None:
        return 0.0
    return max(0.0, 1.0 - abs(guess - count) / 4)


class CountEpisode(Episode[CountTask, None]):
    """One picture, one question, scored against the task's count."""

    grading_retries = 2

    def __init__(self, task: Task[CountTask], options: None) -> None:
        super().__init__(task, options)
        self.guess: int | None = None  # None until the model replies a number

    async def run(self, sample: Sampling) -> dict[str, JsonValue]:
        task = self.task.value
        picture = draw(task.count, task.seed)
        message = {
            "role": "user",
            "content": [
                {"type": "image", "image": picture},
                {"type": "text", "text": QUESTION},
            ],
        }
        reply = await sample.model.conversation().generate(messages=[message])
        self.guess = parse_count(reply.content)
        return {"count": task.count, "guess": self.guess}

    async def grade(self, grading: Grading) -> float:
        return score(self.guess, self.task.value.count)
