"""An episode runtime's tasks: how a dataset line becomes a task, its id and its digest."""

from __future__ import annotations

import json

import pytest
from support.builders import task_file

from loops_cookbook.episodes.tasks import load_tasks, task_digest
from loops_cookbook.runtime.context import RuntimeConfigurationError

from .declared import (
    DECLARED,
    DOCUMENT,
    Documents,
    Everything,
    Question,
    build,
    declaring,
)


async def test_tasks_are_named_by_their_id_or_their_position_and_digested_canonically(tmp_path):
    runtime = build(tmp_path, tasks=({"q": 1, "id": "named"}, {"b": [1, 2], "a": "x"}))
    named, unnamed = await runtime.prepare()
    assert (named.id, unnamed.id) == ("named", "1")
    assert unnamed.digest == task_digest({"a": "x", "b": [1, 2]})  # key order is not content
    assert named.digest != unnamed.digest


@pytest.mark.parametrize(
    ("lines", "problem"),
    [
        (["[1, 2]"], "one JSON object"),
        (["{not json"], "one JSON object"),
        (['{"x": NaN}'], "one JSON object"),
        (['{"id": 7}'], "nonempty string"),
        (['{"id": "a"}', '{"id": "a"}'], "used twice"),
        (['{"id": "1"}', "{}"], "used twice"),  # the second is task 1 by position
        (["", "  "], "has no tasks"),
    ],
)
def test_a_dataset_that_is_not_one_object_per_task_is_refused(tmp_path, lines, problem):
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(RuntimeConfigurationError, match=problem):
        load_tasks(path)


def test_blank_lines_are_skipped_and_named_by_the_line_they_are_on(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps({"a": 1}) + "\n\n" + "{oops\n")
    with pytest.raises(RuntimeConfigurationError, match=r"tasks.jsonl:3:"):
        load_tasks(path)


async def test_a_declared_task_model_makes_the_validated_dump_the_data_and_the_digest(tmp_path):
    raw = {"id": "q1", "prompt": "2+2", "answer": "4"}
    runtime = build(tmp_path, tasks=(raw,), episode=f"{DECLARED}:Declares")
    (task,) = await runtime.prepare()
    dump = {"id": "q1", "prompt": "2+2", "answer": "4", "points": 1}  # the default ran
    assert task.id == "q1"
    assert task.digest == task_digest(dump)
    assert task.digest != task_digest(raw)


def test_the_task_id_is_the_lines_and_stays_in_the_data_the_episode_sees(tmp_path):
    path = task_file(
        tmp_path, [{"id": "a", "prompt": "p", "answer": "x"}, {"prompt": "q", "answer": "y"}]
    )
    named, unnamed = load_tasks(path, Question)
    assert (named.id, unnamed.id) == ("a", "1")
    assert named.data == {"id": "a", "prompt": "p", "answer": "x", "points": 1}
    assert "id" not in unnamed.data  # named by position, as without a model
    assert named.value == Question(prompt="p", answer="x")


@pytest.mark.parametrize(
    "lines",
    [
        [{"id": "a", "prompt": "p", "answer": "a"}, {"id": "b", "prompt": "p"}],  # missing
        [{"id": "a", "prompt": "p", "answer": "a"}, {"id": "b", "prompt": "p", "answer": [1]}],
        [{"id": "a", "prompt": "p", "answer": "a"}, {"id": "b", "prompt": "p", "anwser": "4"}],
    ],
)
def test_a_line_the_task_model_refuses_is_refused_by_file_and_line(tmp_path, lines):
    path = task_file(tmp_path, lines)
    with pytest.raises(RuntimeConfigurationError, match=r"tasks\.jsonl:2: .*not a valid Question"):
        load_tasks(path, Question)


def test_a_task_copy_shares_nothing_with_the_loaded_task(tmp_path):
    (task,) = load_tasks(task_file(tmp_path, [{"prompt": "p", "answer": "a"}]), Question)
    copy = task.copy()
    copy.value.answer = "changed"
    assert task.value.answer == "a"


async def test_two_tasks_that_differ_only_in_what_a_model_would_hide_never_share_a_digest(
    tmp_path, monkeypatch
):
    # Allowed fields dump as validated: tasks that differ anywhere digest apart.
    runtime = build(
        tmp_path,
        tasks=({"text": "t", "tags": ["a"]}, {"text": "t", "tags": ["b"]}),
        episode=declaring(monkeypatch, "task model", Everything),
    )
    first, second = await runtime.prepare()
    assert first.digest != second.digest


def test_a_recursive_json_value_round_trips_as_the_tasks_data(tmp_path):
    line = {"id": "t", "cookbooks": DOCUMENT, "pydantics": DOCUMENT, "owns": DOCUMENT}
    (task,) = load_tasks(task_file(tmp_path, [line]), Documents)
    assert task.data == line
    assert task.value == Documents(cookbooks=DOCUMENT, pydantics=DOCUMENT, owns=DOCUMENT)
