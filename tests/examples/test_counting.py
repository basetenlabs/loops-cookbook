"""The counting example: the picture it draws, the reply it reads, and the score."""

import io
import json
import runpy
from pathlib import Path

import pytest
from count_episode import (
    MOST,
    QUESTION,
    SEPARATION,
    SIZE,
    CountEpisode,
    CountTask,
    draw,
    parse_count,
    place,
    score,
)
from PIL import Image
from pydantic import ValidationError
from support.harness import Renderer

from loops_cookbook.llm.rendering import ParsedReply

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "counting"


def test_the_shipped_tasks_are_what_make_tasks_writes():
    written = runpy.run_path(str(EXAMPLE / "make_tasks.py"))["tasks"]()
    assert shipped_tasks() == written


def shipped_tasks() -> list[dict]:
    return [json.loads(line) for line in (EXAMPLE / "tasks.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("task", shipped_tasks(), ids=lambda task: task["name"])
def test_every_shipped_picture_holds_count_circles_that_can_be_told_apart(task):
    circles = place(task["count"], task["seed"])
    assert len(circles) == task["count"]
    for x, y, r in circles:
        assert r <= x <= SIZE - r and r <= y <= SIZE - r
    for i, (x, y, r) in enumerate(circles):
        for a, b, c in circles[i + 1 :]:
            assert abs(x - a) + abs(y - b) + abs(r - c) > SEPARATION


def test_a_count_the_picture_cannot_hold_is_refused_when_the_task_loads():
    with pytest.raises(ValidationError):
        CountTask(name="too-many", count=MOST + 1, seed=0)


def test_a_task_always_draws_the_same_png():
    picture = draw(9, seed=7)
    assert picture == draw(9, seed=7) != draw(9, seed=8)
    with Image.open(io.BytesIO(picture)) as image:
        assert (image.format, image.size) == ("PNG", (SIZE, SIZE))


@pytest.mark.parametrize(
    ("reply", "guess"),
    [
        ("12", 12),
        (" 7\n", 7),
        ("There are 9 circles.", 9),
        ("12.", 12),
        ("-12", None),
        ("12.9", None),
        ("nine", None),
        (None, None),
    ],
)
def test_the_first_number_in_the_reply_is_the_guess(reply, guess):
    assert parse_count(reply) == guess


@pytest.mark.parametrize(
    ("guess", "reward"), [(10, 1.0), (11, 0.75), (8, 0.5), (14, 0.0), (30, 0.0), (None, 0.0)]
)
def test_each_circle_miscounted_costs_a_quarter_point(guess, reward):
    assert score(guess, 10) == reward


async def test_the_model_sees_the_picture_and_the_reply_is_scored(make, monkeypatch):
    prompts = []
    render = Renderer.render

    def record_prompt(self, messages, tools):
        prompts.append(messages)
        return render(self, messages, tools)

    def parse(self, prompt_ids, completion_ids, tools):
        return ParsedReply("10", None, (), None, {"role": "assistant", "content": "10"})

    monkeypatch.setattr(Renderer, "render", record_prompt)
    monkeypatch.setattr(Renderer, "parse", parse)
    task = {"id": "circles", "name": "circles", "count": 9, "seed": 3}
    built = await make(episode=CountEpisode, sandbox=False, tasks=(task,))
    result = await built.rig.rollout("g-r1", built.task)
    await built.rig.driver.drain()
    assert result.reward == 0.75
    assert result.metadata == {"count": 9, "guess": 10}
    [[message]] = prompts
    assert message["content"] == [
        {"type": "image", "image": draw(9, seed=3)},
        {"type": "text", "text": QUESTION},
    ]
