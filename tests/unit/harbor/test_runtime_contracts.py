"""The core contracts, held by a Harbor rollout runtime and a non-Harbor rollout runtime alike.

Each test runs against both real runtimes through the same driver: Harbor's, with a
real Harbor trial, and the custom-runtime patterns'. What they assert is what the
driver and its evidence show, never either rollout runtime's internals.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from support.builders import runtime_context, task_file
from support.harness import rig

from loops_cookbook.episodes import EpisodeRuntime
from loops_cookbook.episodes.tasks import load_tasks
from loops_cookbook.runtime.driver import Discard

from .jobs import HILL_CLIMB, harbor, job, runtime_for, shared_task

CUSTOM = Path(__file__).resolve().parents[3] / "patterns" / "custom_runtime"
HELLO = CUSTOM / "in_process_direct" / "direct_episode.py"


def _example(path: Path):
    spec = importlib.util.spec_from_file_location(f"{path.stem}_contracts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ONE_TASK = [{"name": "b", "instruction": "Say B", "answer": "B"}]


def _hello(tmp_path):
    """The direct pattern's episode, run by the cookbook's episode runtime."""
    hello = _example(HELLO)
    return EpisodeRuntime(
        hello.AnswerEpisode,
        load_tasks(task_file(tmp_path, ONE_TASK), hello.AnswerTask),
        agent_timeout_seconds=20,
        rollout_timeout_seconds=30,
        sandbox=None,
        options=None,
        work_dir=tmp_path,
    )


@pytest.fixture(params=["harbor", "custom"])
def runtime(request, tmp_path):
    """``build(**settings)`` gives a driver rig and the task it runs, for either runtime."""

    async def build(**settings):
        if request.param == "harbor":
            r, _runtime, task = await harbor(tmp_path, shared_task(tmp_path), **settings)
            return r, task
        hello = _hello(tmp_path)
        (task,) = await hello.prepare()
        return rig(hello, **settings), task

    build.kind = request.param
    return build


async def test_a_graded_rollout_carries_only_its_attempts_evidence_and_leaves_nothing(runtime):
    r, task = await runtime()
    graded = await r.rollout("g-r1", task)
    assert not isinstance(graded, Discard) and graded.samples
    assert {sample.sample_id is not None for sample in graded.samples} == {True}
    assert graded.identity.rollout_id == "g-r1"
    await r.driver.drain()
    assert r.driver.live_attempts == 0 and not r.systemic
    assert r.capacity.snapshot()["slots/sampling"] == 0
    assert r.capacity.snapshot()["slots/verification"] == 0


async def test_a_rollout_stale_by_grading_is_discarded_before_it_is_graded(runtime):
    r, task = await runtime(max_steps_off_policy=0)
    admitted = await r.driver.sample(task, "g", "g-r1")
    r.version[0] = 1  # the policy moved on while the rollout waited
    assert await r.driver.grade(admitted) is Discard.STALE_BEFORE_GRADING
    await r.driver.drain()
    assert r.driver.live_attempts == 0 and not r.systemic


async def test_cancelling_a_sample_ends_the_attempt_and_frees_its_capacity(runtime):
    r, task = await runtime()
    sampling = asyncio.create_task(r.driver.sample(task, "g", "g-r1"))
    async with asyncio.timeout(10):
        while r.driver.live_attempts == 0:
            await asyncio.sleep(0)
    sampling.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(sampling, 10)
    await r.driver.drain()
    assert r.driver.live_attempts == 0 and not r.systemic
    assert r.capacity.snapshot()["slots/sampling"] == 0


@pytest.mark.parametrize("kind", ["harbor", "hello", "episode"])
def test_a_runtime_instance_exposes_nothing_public_but_the_contract(tmp_path, monkeypatch, kind):
    """What a rollout runtime keeps for itself stays private: only the hooks and capabilities."""

    if kind == "harbor":
        runtime = runtime_for(job(tmp_path, HILL_CLIMB, "AnswerAgent"), tmp_path / "work")
    elif kind == "hello":
        runtime = _hello(tmp_path)
    else:
        from loops_cookbook.episodes import create_runtime
        from loops_cookbook.runtime.context import EvidenceRecord

        monkeypatch.syspath_prepend(str(CUSTOM / "local"))
        from local_sandbox import Directory  # pyright: ignore[reportMissingImports]

        runtime = create_runtime(
            {
                "episode": "local_episode:DirectoryAgentEpisode",
                "agent_timeout_seconds": 10,
                "sandbox": {"provider": "local_sandbox"},
                "options": {"agent": ["true"], "verifier": ["true"], "grade_against": "files"},
            },
            context=runtime_context(
                task_file(tmp_path, ONE_TASK),
                limits=SimpleNamespace(
                    rollout_timeout_seconds=None,
                    cancellation_grace_seconds=1,
                    process_kill_grace_seconds=1,
                    provider_deadlines={"local_sandbox": SimpleNamespace(attach_seconds=10.0)},
                    trajectory_tokens=None,
                ),
                work_dir=tmp_path,
                evidence=EvidenceRecord(),
                sandbox_providers={"local_sandbox": Directory},
            ),
        )
    assert {name for name in vars(runtime) if not name.startswith("_")} == set()
