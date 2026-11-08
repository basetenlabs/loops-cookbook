"""Eval sets at preflight: built from training's runtime factory on their own files,
checked as training's runtime is, reported by ``loops check``, and closed with it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loops_cookbook import command
from loops_cookbook.evidence import TaskRef
from loops_cookbook.model_access.api import ModelAccessMode
from loops_cookbook.runtime.api import RuntimeCapabilities

CLOSED: list[str] = []


class FileRuntime:
    """A rollout runtime whose tasks are its file's lines; the file's ``mode`` line, when there
    is one, is the model access it declares, and a ``broken`` file fails prepare()."""

    def __init__(self, path: Path) -> None:
        self.path = path
        lines = [json.loads(line) for line in path.read_text().splitlines() if line]
        modes = [line["mode"] for line in lines if "mode" in line]
        self.tasks = [line for line in lines if "id" in line]
        self.capabilities = RuntimeCapabilities(
            model_access_modes=frozenset({ModelAccessMode(modes[0] if modes else "direct")})
        )

    async def prepare(self) -> list[TaskRef]:
        if "broken" in self.path.name:
            raise RuntimeError("the dataset would not load")
        return [TaskRef(task["id"]) for task in self.tasks]

    async def close(self) -> None:
        CLOSED.append(self.path.name)


def make_runtime(config, *, context):
    return FileRuntime(context.files.path("tasks"))


def tasks_file(directory: Path, name: str, tasks: list[tuple[str, str]], mode: str | None = None):
    lines = [{"id": task_id, "text": text} for task_id, text in tasks]
    if mode is not None:
        lines.append({"mode": mode})
    path = directory / name
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


TRAINING = [("t0", "two plus two"), ("t1", "three plus three")]


def write_config(tmp_path: Path, sets: str, eval_table: str = "", training_table: str = "") -> Path:
    training = tasks_file(tmp_path, "train.jsonl", TRAINING)
    path = tmp_path / "train.toml"
    path.write_text(
        f"""
recipe = "rl"
[rollout_runtime]
factory = "{__name__}:make_runtime"
[rollout_runtime.files]
tasks = "{training}"
[model]
base_model = "m"
lora_rank = 32
[sampling]
max_generation_tokens = 4096
[rollout]
max_concurrent_rollouts = 64
max_concurrent_verifiers = 16
rollout_timeout_seconds = 30
setup_timeout_seconds = 30
[rollout.grading]
timeout_seconds = 10
[training]
learning_rate = 1e-5
group_size = 8
max_group_losses = 2
groups_per_step = 8
loss_fn = "cispo"
{training_table or "max_steps = 0"}
[compute]
scope = "dedicated"
[output]
output_dir = "{tmp_path / "runs"}"
run_name = "evals"
trainer_checkpoint_every = 5
sampler_checkpoint_every = 5
checkpoint_ttl_seconds = 0
[eval]
{eval_table}
{sets}
"""
    )
    return path


@pytest.fixture
def check(capsys, baseline_renderer):
    CLOSED.clear()

    def run(path: Path):
        try:
            code = command.main(["check", str(path)])
        except SystemExit as exit:
            code = exit.code
        captured = capsys.readouterr()
        return code, captured.out + captured.err

    return run


def test_check_reports_each_eval_set_and_closes_its_runtime(check, tmp_path):
    val = tasks_file(tmp_path, "val.jsonl", [("v0", "a"), ("v1", "b"), ("v2", "c")])
    code, out = check(
        write_config(
            tmp_path,
            f"""
[eval.sets.val]
rollout_runtime.files = {{ tasks = "{val}" }}
samples_per_task = 2
max_tasks = 2
max_concurrent_rollouts = 4
""",
            "every_steps = 10",
            "max_steps = 50",
        )
    )
    assert code == 0, out
    assert (
        "Eval set 'val': 2 tasks × 2 samples, 4 at once; every 10 updates, before the "
        "first and after the last" in out
    )
    assert "shared with training" not in out
    assert sorted(CLOSED) == ["train.jsonl", "val.jsonl"]


def test_tasks_shared_with_training_are_warned_about_by_id(check, tmp_path):
    """Shared tasks are matched by ID alone.

    v1 holds a training task's content under another ID, so it isn't a shared task.
    """
    val = tasks_file(
        tmp_path,
        "val.jsonl",
        [("t0", "something new"), ("v1", "three plus three"), ("v2", "held out")],
    )
    code, out = check(
        write_config(tmp_path, f'[eval.sets.val]\nrollout_runtime.files = {{ tasks = "{val}" }}')
    )
    assert code == 0, out
    assert "eval set 'val' shares 1 task(s) with training (t0)" in out
    assert (
        "Eval set 'val': 3 tasks × 1 sample, 16 at once; before the first update and "
        "no final eval without max_steps; 1 task shared with training" in out
    )


def test_only_the_tasks_a_set_evaluates_count_as_shared(check, tmp_path):
    val = tasks_file(tmp_path, "val.jsonl", [("v0", "held out"), ("t0", "x"), ("t1", "y")])
    code, out = check(
        write_config(
            tmp_path,
            f'[eval.sets.val]\nrollout_runtime.files = {{ tasks = "{val}" }}\nmax_tasks = 1',
        )
    )
    assert code == 0, out
    assert "Eval set 'val': 1 task × 1 sample" in out
    assert "shared with training" not in out and "shares" not in out


def test_a_run_from_other_weights_is_not_promised_a_base_eval(check, tmp_path):
    val = tasks_file(tmp_path, "val.jsonl", [("v0", "held out")])
    code, out = check(
        write_config(
            tmp_path,
            f'[eval.sets.val]\nrollout_runtime.files = {{ tasks = "{val}" }}',
            "every_steps = 5",
            'max_steps = 10\ninit_from = "bt://loops:other/state/step-9"',
        )
    )
    assert code == 0, out
    assert "every 5 updates, no base eval (init_from) and after the last" in out


def test_a_resumed_run_is_not_promised_a_base_eval(check, tmp_path, monkeypatch):
    from loops_cookbook.rl.run import Run

    monkeypatch.setattr(Run, "resume_step", property(lambda self: 4))
    val = tasks_file(tmp_path, "val.jsonl", [("v0", "held out")])
    code, out = check(
        write_config(
            tmp_path,
            f'[eval.sets.val]\nrollout_runtime.files = {{ tasks = "{val}" }}',
            "every_steps = 5",
            "max_steps = 10",
        )
    )
    assert code == 0, out
    assert "every 5 updates, no base eval (resume) and after the last" in out


def test_an_eval_set_whose_runtime_reaches_the_model_another_way_is_refused(check, tmp_path):
    val = tasks_file(tmp_path, "val.jsonl", [("v0", "a")], mode="openai_http")
    code, out = check(
        write_config(tmp_path, f'[eval.sets.val]\nrollout_runtime.files = {{ tasks = "{val}" }}')
    )
    assert code != 0
    assert "eval set 'val' resolves model access 'openai_http', and training 'direct'" in out
    assert sorted(CLOSED) == ["train.jsonl", "val.jsonl"]


def test_an_eval_set_that_fails_to_prepare_closes_every_runtime_built(check, tmp_path):
    good = tasks_file(tmp_path, "good.jsonl", [("v0", "a")])
    broken = tasks_file(tmp_path, "broken.jsonl", [("v1", "b")])
    code, out = check(
        write_config(
            tmp_path,
            f"""
[eval.sets.good]
rollout_runtime.files = {{ tasks = "{good}" }}
[eval.sets.bad]
rollout_runtime.files = {{ tasks = "{broken}" }}
""",
        )
    )
    assert code != 0
    assert "the dataset would not load" in out
    assert sorted(CLOSED) == ["broken.jsonl", "good.jsonl", "train.jsonl"]
