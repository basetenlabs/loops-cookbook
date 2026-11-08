"""The custom local pattern's sandbox provider: a directory the run owns, and recovers
after the run is killed."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import local_sandbox
import pytest
from support.builders import task_file
from support.configs import read_toml, write_toml

from loops_cookbook import command

from .custom_local import DEADLINES, EXAMPLE, TASKS

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "offline"


async def test_the_local_provider_is_a_sandbox_alive_until_released_and_proves_it_gone(tmp_path):
    from loops_cookbook.resources import ReconcileStatus, SandboxProvider

    provider = local_sandbox.create({"root": str(tmp_path), **DEADLINES})
    assert isinstance(provider, SandboxProvider)
    intent = NS(desired_name="sandbox-r1", ownership_labels={"loops.resource_id": "r1"})
    resource = await provider.create(intent)
    assert await provider.validate_ownership(intent, resource)
    listed = [page async for page in provider.list_owned({"loops.resource_id": "r1"})]
    assert [[r.name for r in page] for page in listed] == [["sandbox-r1"]]
    with pytest.raises(FileExistsError):
        await provider.create(intent)
    assert await provider.client(resource) == tmp_path / "sandbox-r1"  # the directory's Path
    assert await provider.alive(resource)
    await provider.release(resource)
    assert not await provider.alive(resource)
    assert (await provider.confirm_absent(resource)).status is ReconcileStatus.PROVEN_ABSENT
    assert (await provider.reconcile_create(intent)).status is ReconcileStatus.PROVEN_ABSENT


KILLED_RUN = """
import asyncio, sys
from pathlib import Path
sys.path[:0] = [{here!r}, {example!r}, {fixtures!r}]
from baseline_model import BaselineTrainer, create_renderer
from loops_cookbook.rl import cli
from loops_cookbook.providers import create_adapters
from loops_cookbook.resources import SandboxGone
from loops_cookbook.rl.recipe import assemble

async def main():
    settings = cli.load_training_config(Path({config!r}), {{
        "rollout_runtime.config": {{
            "episode": "local_episode:DirectoryAgentEpisode",
            "agent_timeout_seconds": 90,
            "sandbox": {{"provider": "local_sandbox"}},
            "options": {{
                "agent": [sys.executable, "-c", "import time; time.sleep(60)", {token!r}],
                "verifier": ["python3", "./verifier.py"],
                "grade_against": "files",
            }},
        }},
        "rollout_runtime.files": {{"tasks": {tasks!r}}},
        "compute.sampler_replica_affinity": True,
        "output.output_dir": {runs!r},
        "training.max_steps": 1,
        "training.group_size": 2,
    }})
    run = assemble(settings, trainer=BaselineTrainer(), config_dir=Path({example!r}),
                   renderer_factory=lambda _config: create_renderer(),
                   adapters=await create_adapters(settings.providers))
    await run.execute()

asyncio.run(main())
"""


def test_loops_cleanup_recovers_a_killed_runs_sandboxes(tmp_path, sandboxes):
    """The training process is killed mid-rollout, holding sandboxes; its journal is enough to recover."""
    import signal
    import subprocess
    import time

    token = f"killed-run-{os.getpid()}"
    shipped = read_toml(EXAMPLE / "train.toml")
    assert shipped["providers"]["local_sandbox"]["root"] == "/tmp/loops-local-sandboxes"
    config = write_toml(
        tmp_path / "train.toml", shipped, {"providers": {"local_sandbox": {"root": sandboxes}}}
    )
    script = KILLED_RUN.format(
        token=token,
        here=str(tmp_path),
        example=str(EXAMPLE),
        fixtures=str(FIXTURES),
        config=str(config),
        tasks=str(task_file(tmp_path, TASKS)),
        runs=str(tmp_path / "runs"),
    )
    training = subprocess.Popen(
        [sys.executable, "-c", script], cwd=tmp_path, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 60
        while not (sandboxes.is_dir() and len(list(sandboxes.glob("sandbox-*"))) >= 2):
            assert training.poll() is None, training.stderr.read() if training.stderr else ""
            assert time.monotonic() < deadline, "the training process never held its sandboxes"
            time.sleep(0.1)
        os.kill(training.pid, signal.SIGKILL)  # no shutdown, no release
        training.wait(10)
    finally:
        if training.poll() is None:
            training.kill()
        # Its agent processes are in sessions of their own; they die with the test.
        subprocess.run(["pkill", "-9", "-f", token], check=False)
    left = sorted(path.name for path in sandboxes.glob("sandbox-*"))
    assert len(left) >= 2
    (state,) = (tmp_path / "runs").glob("*/state")
    assert command.main(["cleanup", str(config), "--state-dir", str(state)]) == 0
    assert list(sandboxes.glob("sandbox-*")) == []
