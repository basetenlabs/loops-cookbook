"""Write tasks.jsonl: ``TASKS`` pictures to count, each with a count the episode accepts.

Deterministic, so the file beside it is what this produces. A task holds only the
count and the seed; the episode draws the picture from them. Run it from anywhere:
``uv run python examples/counting/make_tasks.py``.
"""

import json
import random
from pathlib import Path

from count_episode import FEWEST, MOST

HERE = Path(__file__).parent
TASKS = 512


def tasks() -> list[dict[str, int | str]]:
    rng = random.Random(0)
    lines = []
    for n in range(TASKS):
        count = rng.randint(FEWEST, MOST)
        lines.append({"name": f"circles-{n:03d}", "count": count, "seed": rng.randrange(2**31)})
    return lines


if __name__ == "__main__":
    text = "".join(json.dumps(line) + "\n" for line in tasks())
    (HERE / "tasks.jsonl").write_text(text)
