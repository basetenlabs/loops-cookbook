"""The Harbor in-process direct pattern, and the hill-climb task every Harbor pattern shares."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import yaml
from harbor.models.job.config import JobConfig
from harbor.models.task.task import Task
from support.builders import services_for

from loops_cookbook.harbor.runtime import _validate_agent_binding
from loops_cookbook.model_access.api import ModelAccessMode
from loops_cookbook.rl.cli import load_training_config

ROOT = Path(__file__).resolve().parents[2]
PATTERN = ROOT / "patterns" / "harbor" / "in_process_direct"
TASK = ROOT / "tasks" / "hill_climb"


def job(name: str) -> JobConfig:
    return JobConfig.model_validate(yaml.safe_load((PATTERN / name).read_text()))


def test_the_task_is_graded_separately_from_the_answer_it_leaves():
    task = Task(TASK)
    assert task.config.verifier.environment_mode.value == "separate"
    assert task.config.artifacts[0].destination == "answer.txt"


def test_the_task_rewards_one_fixed_label_never_adapted_to_a_baseline():
    spec = spec_from_file_location("hill_climb_grade", TASK / "tests" / "grade.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert [module.score(value) for value in ("A", "B", "C", "D")] == [0, 1, 0, 0]
    assert module.score(" B\n") == 1
    assert module.score("I choose B") == 0


def test_the_docker_job_runs_the_direct_agent_on_the_shared_task_in_docker():
    docker = job("job.yaml")
    assert _validate_agent_binding(docker.agents[0]) is ModelAccessMode.DIRECT
    assert docker.environment.type.value == "docker"
    assert [task.path for task in docker.tasks] == [Path("tasks/hill_climb")]


def test_the_docker_config_has_a_hard_step_limit():
    config = load_training_config(PATTERN / "train.toml")
    assert config.training.max_steps <= 5
    assert config.rollout.max_consecutive_rollout_failures > 1  # one failure never ends a run
    assert config.runtime.factory == "loops_cookbook.harbor:create_runtime"
    assert config.runtime.files["job"] == str(PATTERN / "job.yaml")


async def test_the_agent_asks_with_llm_and_writes_the_stripped_answer(tmp_path):
    from direct_agent import DirectAgent

    seen = []

    class Conversation:
        async def generate(self, **kwargs):
            seen.append(kwargs)
            return SimpleNamespace(content=" B\n")

    class LLM:
        def conversation(self, *, conversation_id=None):
            return Conversation()

    class Environment:
        async def exec(self, command):
            assert command == "mkdir -p /app && printf %s B > /app/answer.txt"
            return SimpleNamespace(return_code=0)

    agent = DirectAgent(logs_dir=tmp_path)
    agent.bind_loops_services(services_for(LLM()))
    await agent.run("Pick a label", Environment(), None)
    assert seen == [{"messages": [{"role": "user", "content": "Pick a label"}]}]


def test_the_blaxel_job_runs_the_direct_agent_on_the_shared_task_on_blaxel():
    on_blaxel = job("job_blaxel.yaml")
    assert on_blaxel.environment.import_path == "loops_cookbook.harbor.blaxel:BlaxelEnvironment"
    assert [task.path for task in on_blaxel.tasks] == [Path("tasks/hill_climb")]
    assert _validate_agent_binding(on_blaxel.agents[0]) is ModelAccessMode.DIRECT
