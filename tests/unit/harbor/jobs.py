"""Harbor jobs, tasks and rollout runtimes for the Harbor rollout runtime tests.

Every job runs its trials in the offline directory-backed environment from
tests/fixtures/offline, with an agent from ``harbor_test_agents``. Nothing reaches a
container, the network, or a billable resource.
"""

from __future__ import annotations

import asyncio
import shutil
import weakref
from pathlib import Path
from typing import Any

import yaml
from support.configs import Tables, read_toml, write_toml
from support.harness import rig
from support.resources import run_resources, scratch_registry

from loops_cookbook.harbor.runtime import DAYTONA_ENVIRONMENT, create_runtime
from loops_cookbook.runtime.api import RunLimits, RuntimeContext
from loops_cookbook.runtime.context import EvidenceRecord, MetricsSink, RuntimeFileResolver

HILL_CLIMB = Path(__file__).resolve().parents[3] / "tasks" / "hill_climb"
GRADE = 'if [ "$(cat /app/answer.txt)" = B ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt\n'
LIMITS = {
    "max_concurrent_rollouts": 4,
    "max_concurrent_verifiers": 4,
    "rollout_timeout_seconds": 60,
    "grading_timeout_seconds": 3600,
    "cancellation_grace_seconds": 5,
    "process_kill_grace_seconds": 5,
}

# What each rollout runtime reported to its run's metrics, kept beside it rather than on it.
_METRICS: weakref.WeakKeyDictionary[Any, MetricsSink] = weakref.WeakKeyDictionary()


def shared_task(root: Path, *, agent_timeout: float = 30, steps: int = 0) -> Path:
    """A task verified in its agent's own environment; with ``steps``, a multi-step one."""
    task = root / f"shared-{steps}"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM scratch\n")
    tables: Tables = {
        "version": "1.0",
        "agent": {"timeout_sec": agent_timeout},
        "verifier": {"timeout_sec": 30},
    }
    if steps:
        for index in range(1, steps + 1):
            step = task / "steps" / f"s{index}"
            (step / "tests").mkdir(parents=True)
            (step / "instruction.md").write_text(f"Step {index}.\n")
            (step / "tests" / "test.sh").write_text("#!/bin/sh\n" + GRADE)
        tables["steps"] = [{"name": f"s{index}"} for index in range(1, steps + 1)]
    else:
        (task / "tests").mkdir()
        (task / "instruction.md").write_text("Answer B.\n")
        (task / "tests" / "test.sh").write_text("#!/bin/sh\n" + GRADE)
    write_toml(task / "task.toml", tables)
    return task


def separate_task(root: Path, *, agent_timeout: float, verifier_timeout: float = 30) -> Path:
    """The hill-climb task, graded in a separate verifier environment, with these deadlines."""
    task = root / "separate"
    shutil.copytree(HILL_CLIMB, task)
    deadlines = {
        "agent": {"timeout_sec": agent_timeout},
        "verifier": {"timeout_sec": verifier_timeout},
    }
    write_toml(task / "task.toml", read_toml(task / "task.toml"), deadlines)
    return task


def separate_steps_task(root: Path) -> Path:
    """Two steps, each graded in its own verifier environment."""
    task = root / "separate-steps"
    shutil.copytree(HILL_CLIMB / "environment", task / "environment")
    for index in (1, 2):
        step = task / "steps" / f"s{index}"
        shutil.copytree(HILL_CLIMB / "tests", step / "tests")
        (step / "instruction.md").write_text("Answer B.\n")
    steps = {"steps": [{"name": "s1"}, {"name": "s2"}]}
    write_toml(task / "task.toml", read_toml(HILL_CLIMB / "task.toml"), steps)
    return task


def sized_task(root: Path, verifier: dict | None = None, **environment) -> Path:
    """The hill-climb task, with its [environment] table replaced by ``environment``, and
    its separate verifier's [verifier.environment] by ``verifier`` when given."""
    task = root / "sized"
    shutil.copytree(HILL_CLIMB, task)
    tables = read_toml(task / "task.toml")
    tables["environment"] = environment or {"cpus": 1}
    if verifier is not None:
        tables["verifier"]["environment"] = verifier or {"cpus": 1}
    write_toml(task / "task.toml", tables)
    return task


def job(root: Path, task: Path, agent: str, **extra) -> Path:
    """A Harbor job of one task and one ``harbor_test_agents`` agent; ``extra`` keys win."""
    path = root / "job.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tasks": [{"path": str(task)}],
                "agents": [{"import_path": f"harbor_test_agents:{agent}"}],
                "environment": {"import_path": "baseline_environment:LocalEnvironment"},
                "retry": {"max_retries": 0},
                **extra,
            }
        )
    )
    return path


def runtime_for(
    job_path: Path,
    work: Path,
    *,
    limits=None,
    provider_deadlines=None,
    provider_regions=None,
    provider_capacity=None,
    provider_settings=None,
    provider_images=None,
    **config,
):
    """A Harbor rollout runtime of ``job_path``; ``limits`` changes the run's limits by name."""
    metrics = MetricsSink()
    context = RuntimeContext(
        work_dir=work,
        project_root=work,
        config_dir=work,
        files=RuntimeFileResolver({"job": job_path}),
        run_resources=run_resources(scratch_registry(), run_id="run"),
        limits=RunLimits(
            **{**LIMITS, **(limits or {})}, provider_deadlines=provider_deadlines or {}
        ),
        metrics=metrics,
        evidence=EvidenceRecord(),
        sandbox_providers={},
        provider_regions=provider_regions or {},
        provider_capacity=provider_capacity or {},
        provider_settings=provider_settings or {},
        provider_images=provider_images or {},
    )
    runtime = create_runtime(config, context=context)
    _METRICS[runtime] = metrics
    return runtime


async def harbor(tmp_path, task, agent="AnswerAgent", **options):
    """``(rig, runtime, task)``: a prepared rollout runtime of ``task`` under the real driver.

    ``job`` changes the job, ``config`` the rollout runtime, and every other option the rig."""
    runtime = runtime_for(
        job(tmp_path, task, agent, **options.pop("job", {})),
        tmp_path / "work",
        **options.pop("config", {}),
    )
    tasks = await runtime.prepare()
    r = rig(runtime, **options)
    return r, runtime, tasks[0]


def reported(runtime) -> dict[str, float]:
    """What ``runtime`` reported to its run's metrics."""
    return _METRICS[runtime].snapshot()


def live_trials(runtime) -> float:
    """Trials ``runtime`` still holds: started, parked for grading, or tearing down."""
    return reported(runtime).get("runtime/harbor/live_trials", 0)


def sampled(runtime) -> list:
    """The attempt of every ``runtime.sample()`` call, in order, as the driver makes it."""
    attempts = []
    sample = runtime.sample

    async def recording(attempt):
        attempts.append(attempt)
        return await sample(attempt)

    runtime.sample = recording
    return attempts


def held_stops(monkeypatch):
    """Hold every sandbox stop once ``grading`` is set until ``released`` is, as a slow
    Blaxel delete would; returns both events."""
    import baseline_environment

    grading, released = asyncio.Event(), asyncio.Event()
    stop = baseline_environment.LocalEnvironment.stop

    async def held(self, delete):
        if grading.is_set():
            await released.wait()
        return await stop(self, delete)

    monkeypatch.setattr(baseline_environment.LocalEnvironment, "stop", held)
    return grading, released


def daytona_runtime(tmp_path, *, override=None, verifier=None, step_verifier=None, **table):
    """A Daytona job over the sized hill-climb task; with ``step_verifier``, a one-step
    task whose step is verified in an environment of that size."""
    from support.daytona_api import CREDENTIALS, SETTINGS, FakeApi

    from loops_cookbook.providers.daytona.adapter import DaytonaAdapter

    settings = SETTINGS.model_copy(update=table)
    adapter = DaytonaAdapter(settings, CREDENTIALS, FakeApi())
    environment = {"import_path": DAYTONA_ENVIRONMENT, **(override or {})}
    task = sized_task(
        tmp_path, verifier=verifier or {"cpus": 2, "memory_mb": 4096}, cpus=2, memory_mb=4096
    )
    if step_verifier is not None:
        step = task / "steps" / "s1"
        shutil.copytree(task / "tests", step / "tests")
        (step / "instruction.md").write_text("Answer B.\n")
        sizes = "".join(f"{key} = {value}\n" for key, value in step_verifier.items())
        with (task / "task.toml").open("a") as toml:
            toml.write(f'\n[[steps]]\nname = "s1"\n\n[steps.verifier.environment]\n{sizes}')
    return runtime_for(
        job(tmp_path, task, "AnswerAgent", environment=environment),
        tmp_path / "work",
        provider_deadlines={"daytona": adapter.deadlines},
        provider_capacity={"daytona": adapter.capacity},
        provider_settings={"daytona": adapter.settings},
    )
