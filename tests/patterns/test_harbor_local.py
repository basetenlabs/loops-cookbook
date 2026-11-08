"""The Harbor local pattern: an agent program over HTTP, tools in the sandbox.

Harbor runs the real hill-climb task against a local sandbox that exposes a usage
environment, as the cookbook's Blaxel environment does; the agent program is a
separate Python process that reaches the model through the run's gateway.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml
from support.configs import read_toml, write_toml
from support.offline_run import offline_config, train_offline

pytest.importorskip("harbor")

from loops_cookbook.model_access.api import ModelAccessMode
from loops_cookbook.rl import cli

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "patterns" / "harbor" / "local"
HILL_CLIMB = ROOT / "tasks" / "hill_climb"
SUPPORT = Path(__file__).resolve().parents[1] / "support"
GROUP = 4

import local_agent

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")


@pytest.fixture
def train(tmp_path):
    """Train one step of the hill-climb task with ``LocalAgent`` and ``kwargs``."""

    async def run(task=None, settings=None, **kwargs):
        job = tmp_path / "job.yaml"
        job.write_text(
            yaml.safe_dump(
                {
                    "tasks": [{"path": str(task or HILL_CLIMB)}],
                    "agents": [{"import_path": "local_agent:LocalAgent", "kwargs": kwargs}],
                    "environment": {
                        "import_path": "support.harbor_sandbox:SandboxLocalEnvironment"
                    },
                    "retry": {"max_retries": 0},
                }
            )
        )
        config = offline_config(
            EXAMPLE / "train_blaxel.toml",
            tmp_path,
            {
                "rollout_runtime.files": {"job": str(job)},
                "providers": {},
                "training.max_steps": 1,
                # One wave of four agent programs: each is a Python process and a Harbor trial.
                "training.group_size": GROUP,
                "rollout.max_concurrent_rollouts": GROUP,
                **(settings or {}),
            },
        )
        outcome, trainer, _ = await train_offline(
            config, config_dir=EXAMPLE, adapters={}, timeout=120
        )
        return outcome, trainer, tmp_path / "runs"

    return run


def test_the_example_declares_http_model_access_and_the_blaxel_environment():
    from harbor.models.job.config import JobConfig

    from loops_cookbook.harbor.runtime import _validate_agent_binding

    job = JobConfig.model_validate(yaml.safe_load((EXAMPLE / "job_blaxel.yaml").read_text()))
    assert _validate_agent_binding(job.agents[0]) is ModelAccessMode.OPENAI_HTTP
    assert job.environment.import_path == "loops_cookbook.harbor.blaxel:BlaxelEnvironment"
    config = cli.load_training_config(EXAMPLE / "train_blaxel.toml")
    assert config.rollout_runtime.model_access is None  # the class states it
    assert "blaxel" in config.providers


async def test_the_agent_program_trains_through_the_gateway_with_its_tools_in_the_sandbox(train):
    outcome, trainer, runs = await train(
        command=[
            "python3",
            str(SUPPORT / "local_agent_program.py"),
            str(EXAMPLE / "agent_program.py"),
        ]
    )
    assert outcome.ok and outcome.steps == 1
    (step,) = trainer.steps
    # The verifier read what the program wrote in the sandbox: the baseline sampler
    # answers B in some rollouts, and only B scores.
    assert 1.0 in step["rewards"] and len(step["rewards"]) == GROUP
    # Every trained rollout carried the attempt's samples, attributed by credential.
    assert len(step["sampled_versions"]) >= len(step["rewards"])
    logs = list(runs.rglob("process.log"))
    assert logs and all("wrote" in log.read_text() for log in logs)


async def test_an_http_agent_that_outlives_its_deadline_is_graded_not_an_infrastructure_failure(
    train, tmp_path
):
    """Its answer is in the sandbox; the deadline ends it; the verifier's reward trains."""
    import shutil

    task = tmp_path / "short-deadline"
    shutil.copytree(HILL_CLIMB, task)
    toml = task / "task.toml"
    write_toml(toml, read_toml(toml), {"agent": {"timeout_sec": 2}})
    outcome, trainer, _runs = await train(
        task=task,
        command=[
            "python3",
            str(SUPPORT / "answer_then_sleep.py"),
            str(EXAMPLE / "agent_program.py"),
        ],
    )
    assert outcome.ok and outcome.steps == 1  # graded: never a rollout failure
    (step,) = trainer.steps
    assert 1.0 in step["rewards"] and len(step["rewards"]) == GROUP
    (recorded,) = [json.loads(line) for line in next(_runs.rglob("steps.jsonl")).open()]
    assert recorded["runtime/harbor/trial_exceptions/AgentTimeoutError"] == GROUP  # each timed out


def stubbed(logs_dir, exit_code=0, signal=None):
    """``LocalAgent`` over stub services; returns it and the commands it ran."""
    from pydantic import SecretStr

    from loops_cookbook.model_access.api import ModelConnection
    from loops_cookbook.runtime import ProcessResult

    commands = []

    async def connection(producer=None):
        return ModelConnection("http://127.0.0.1:9/v1", SecretStr("loops-key"), "m")

    async def execute(command):
        commands.append(command)
        return ProcessResult(exit_code, signal, "out", "err", 0.5)

    agent = local_agent.LocalAgent(logs_dir=logs_dir)
    agent.bind_loops_services(
        NS(model_access=NS(http_connection=connection), processes=NS(run=execute))
    )
    return agent, commands


SANDBOX = NS(usage_environment=lambda: {"SANDBOX_ROOT": "/sb"})


async def test_the_program_gets_the_model_connection_and_the_sandboxs_usage_environment(
    tmp_path,
):
    agent, commands = stubbed(tmp_path)
    context = NS(metadata=None)
    await agent.run("Say B", SANDBOX, context)
    (command,) = commands
    assert command.argv == ("python3", str(EXAMPLE / "agent_program.py"))
    assert command.cwd == tmp_path
    env = {
        name: value.get_secret_value() if hasattr(value, "get_secret_value") else value
        for name, value in command.env.items()
    }
    assert env == {
        "SANDBOX_ROOT": "/sb",
        "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
        "OPENAI_API_KEY": "loops-key",
        "OPENAI_MODEL": "m",
        "AGENT_INSTRUCTION": "Say B",
    }
    assert context.metadata == {"agent_exit": "status_0", "seconds": 0.5}
    assert (tmp_path / "process.log").read_text() == "outerr"


@pytest.mark.parametrize(("exit_code", "signal"), [(4, None), (-1, 9)], ids=["status", "signal"])
async def test_a_program_that_fails_is_harbors_agent_exit(tmp_path, exit_code, signal):
    from harbor.agents.installed.base import NonZeroAgentExitCodeError

    agent, _commands = stubbed(tmp_path, exit_code, signal)
    context = NS(metadata=None)
    with pytest.raises(NonZeroAgentExitCodeError):
        await agent.run("x", SANDBOX, context)
    assert context.metadata["agent_exit"] == ("signal_9" if signal else "status_4")


async def test_the_agent_needs_a_sandbox_it_can_reach(tmp_path):
    agent, commands = stubbed(tmp_path)
    with pytest.raises(TypeError, match="usage environment"):
        await agent.run("x", NS(), NS(metadata=None))
    assert commands == []
