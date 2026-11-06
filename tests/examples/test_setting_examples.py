"""Examples by setting: each is the Harbor in-process direct pattern with one story changed."""

from pathlib import Path

import pytest
import yaml
from harbor.models.job.config import JobConfig
from harbor.models.task.task import Task

from loops_cookbook.harbor.runtime import _validate_agent_binding
from loops_cookbook.model_access.api import ModelAccessMode

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def job(path: Path) -> JobConfig:
    return JobConfig.model_validate(yaml.safe_load(path.read_text()))


@pytest.mark.parametrize("example", ["critic", "multi_step"])
def test_each_blaxel_example_runs_its_direct_agent_on_blaxel(example):
    on_blaxel = job(EXAMPLES / example / "job_blaxel.yaml")
    assert on_blaxel.environment.import_path == "loops_cookbook.harbor.blaxel:BlaxelEnvironment"
    assert _validate_agent_binding(on_blaxel.agents[0]) is ModelAccessMode.DIRECT


def test_the_multi_step_example_runs_the_shared_two_step_task():
    on_blaxel = job(EXAMPLES / "multi_step" / "job_blaxel.yaml")
    assert [task.path for task in on_blaxel.tasks] == [Path("tasks/hill_climb_steps")]
    task = Task(ROOT / "tasks" / "hill_climb_steps")
    assert [step.name for step in task.config.steps] == ["first", "second"]
