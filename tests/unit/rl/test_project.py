"""The public file-backed workflow, using the real runtime and fake remote compute."""

import json
import os
import sys

import pytest

from loops_cookbook.project import load
from loops_cookbook.rl import recipe


async def test_loaded_project_checks_then_trains_from_another_directory(
    tmp_path, monkeypatch, config, baseline_renderer, capsys
):
    from baseline_model import BaselineTrainer

    source = tmp_path / "project"
    source.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("LOOPS_PROJECT_TEST_PROMPT", "shell prompt")
    monkeypatch.delenv("LOOPS_PROJECT_TEST_ANSWER", raising=False)
    # Register the absent variable for restoration after dotenv changes it.
    monkeypatch.setenv("LOOPS_PROJECT_TEST_ANSWER", "")
    monkeypatch.delenv("LOOPS_PROJECT_TEST_ANSWER")
    monkeypatch.delitem(sys.modules, "loaded_project_episode", raising=False)
    (source / ".env").write_text(
        "LOOPS_PROJECT_TEST_PROMPT=file prompt\nLOOPS_PROJECT_TEST_ANSWER=B\n"
    )
    (source / "loaded_project_episode.py").write_text("""
import os
from loops_cookbook.episodes import Episode

PROMPT = os.environ["LOOPS_PROJECT_TEST_PROMPT"]
ANSWER = os.environ["LOOPS_PROJECT_TEST_ANSWER"]

class Answer(Episode[None, None]):
    def __init__(self, task, options):
        super().__init__(task, options)
        self.answer = None

    async def run(self, sample):
        conversation = sample.model.llm(producer="solver").conversation(conversation_id="answer")
        reply = await conversation.generate(messages=[{"role": "user", "content": PROMPT}])
        self.answer = reply.content

    async def grade(self, grading):
        return float(self.answer == ANSWER)
""")
    (source / "tasks.jsonl").write_text('{"id":"question"}\n')
    values = config(
        **{
            "rollout_runtime.factory": "loops_cookbook.episodes:create_runtime",
            "rollout_runtime.config": {
                "episode": "loaded_project_episode:Answer",
                "agent_timeout_seconds": 10,
            },
            "rollout_runtime.files": {"tasks": "tasks.jsonl"},
            "rollout.rollout_timeout_seconds": None,
            "rollout.producers": {"solver": {"trainable": True}},
            "training.max_steps": 1,
            "compute.sampler_replica_affinity": True,
            "output.output_dir": "results",
        }
    )
    (source / "train.json").write_text(values.model_dump_json())
    trainers = []

    def trainer(*args):
        result = BaselineTrainer()
        trainers.append(result)
        return result

    monkeypatch.setattr(recipe, "Trainer", trainer)
    project = load("project/train.json", env_file=".env", overrides={"training.group_size": 8})
    monkeypatch.chdir(elsewhere)
    report = await project.check()
    assert trainers == [], "check must never construct a Loops trainer"
    assert report.config.training.group_size == 8
    assert report.rollout_timeout_seconds > 0
    assert report.images.error is None
    assert report.reasoning_efforts == ()
    assert [task.id for task in report.tasks] == ["question"]
    assert report.output_dir.parent == tmp_path / "results" / "preflight"
    assert capsys.readouterr().out == "", "a library check returns its report"
    assert "tasks: 1" in "\n".join(report.describe())
    assert os.environ["LOOPS_PROJECT_TEST_PROMPT"] == "shell prompt"
    assert os.environ["LOOPS_PROJECT_TEST_ANSWER"] == "B"
    outcome = await project.train()
    assert outcome.ok and outcome.steps == 1
    assert len(trainers) == 1 and len(trainers[0].steps) == 1
    recorded = [
        json.loads(path.read_text()) for path in (tmp_path / "results").glob("*/config.json")
    ]
    assert recorded and all(value["training"]["group_size"] == 8 for value in recorded)


def test_loading_is_offline_and_dotenv_is_explicit(tmp_path, monkeypatch, config):
    path = tmp_path / "train.json"
    path.write_text(config().model_dump_json())
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("LOOPS_EXPLICIT_ENV_TEST", raising=False)
    (tmp_path / ".env").write_text("LOOPS_EXPLICIT_ENV_TEST=loaded\n")
    load(path)
    assert "LOOPS_EXPLICIT_ENV_TEST" not in os.environ
    with pytest.raises(FileNotFoundError):
        load(path, env_file="missing.env")
