"""Standalone Episode examples through the real Blaxel registry and driver.

Policy replies are deterministic; sandbox creation, files, retries and deletion
use the real provider. Model training is covered by the hill-climb recipe.
"""

import asyncio
import uuid
from types import SimpleNamespace as NS
from typing import ClassVar

import pytest
from episodes import Options, Question, SeparateVerifierEpisode, SharedSandboxEpisode
from support.builders import task_file
from support.harness import rig
from support.resources import episodes as resource_factory

from loops_cookbook.episodes import Closing, Grading, Sampling, Task, create_runtime
from loops_cookbook.providers import create_adapters
from loops_cookbook.providers.blaxel import Blaxel
from loops_cookbook.resources.registry import ResourceRegistry
from loops_cookbook.runtime import JsonValue
from loops_cookbook.runtime.context import EvidenceRecord

pytestmark = [pytest.mark.live_blaxel, pytest.mark.asyncio]


class SharedProbe(SharedSandboxEpisode):
    instances: ClassVar[list["SharedProbe"]] = []

    def __init__(self, task: Task[Question], options: Options) -> None:
        super().__init__(task, options)
        self.execution: Blaxel | None = None
        self.closed: Closing | None = None
        self.instances.append(self)

    async def run(self, sample: Sampling) -> dict[str, JsonValue]:
        assert isinstance(sample.sandbox, Blaxel)
        self.execution = sample.sandbox
        return await super().run(sample)

    async def grade(self, grading: Grading) -> float:
        assert grading.execution_sandbox is self.execution
        assert grading.grading_sandbox is None
        return await super().grade(grading)

    async def close(self, closing: Closing) -> None:
        assert self.closed is None
        self.closed = closing


class RetryProbe(SeparateVerifierEpisode):
    grading_retries = 1
    instances: ClassVar[list["RetryProbe"]] = []

    def __init__(self, task: Task[Question], options: Options) -> None:
        super().__init__(task, options)
        self.execution: Blaxel | None = None
        self.verifiers: list[Blaxel] = []
        self.clients: list[object] = []
        self.closed: Closing | None = None
        self.instances.append(self)

    async def run(self, sample: Sampling) -> dict[str, JsonValue]:
        assert isinstance(sample.sandbox, Blaxel)
        self.execution = sample.sandbox
        return await super().run(sample)

    async def grade(self, grading: Grading) -> float:
        assert grading.execution_sandbox is None
        verifier = grading.grading_sandbox
        assert isinstance(verifier, Blaxel)
        self.verifiers.append(verifier)
        self.clients.append(await verifier.client())
        reward = await super().grade(grading)
        assert reward == 1
        if len(self.verifiers) == 1:
            raise ConnectionError("deliberate live grading retry")
        return reward

    async def close(self, closing: Closing) -> None:
        assert self.closed is None
        self.closed = closing


class CancelProbe(SharedProbe):
    started: ClassVar[asyncio.Event]

    async def run(self, sample: Sampling) -> dict[str, JsonValue]:
        await super().run(sample)
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("sampling should have been cancelled")


@pytest.fixture
async def build(tmp_path):
    adapters = await create_adapters({"blaxel": {"region": "us-was-1"}})
    run_id = f"live-episode-{uuid.uuid4().hex[:8]}"
    registry = ResourceRegistry.open(tmp_path / "state", adapters, mode="write", run_id=run_id)
    drivers = []

    async def owned():
        return [
            resource
            async for page in adapters["blaxel"].list_owned({"loops.run_id": run_id})
            for resource in page
        ]

    async def make(episode, *, separate=False):
        tasks = task_file(tmp_path, [{"id": "q", "prompt": "say ok", "answer": "ok"}])
        context = NS(
            files=NS(path=lambda alias: tasks),
            limits=NS(
                rollout_timeout_seconds=None,
                cancellation_grace_seconds=2,
                process_kill_grace_seconds=1,
                provider_deadlines={"blaxel": adapters["blaxel"].deadlines},
                trajectory_tokens=None,
            ),
            work_dir=tmp_path / "work",
            evidence=EvidenceRecord(),
            sandbox_providers={"blaxel": Blaxel},
        )
        sandbox = {
            "provider": "blaxel",
            "image": "blaxel/base-image:latest",
            "memory_mb": 2048,
            "ttl": "10m",
        }
        config = {
            "episode": f"{episode.__module__}:{episode.__name__}",
            "agent_timeout_seconds": 60,
            "sandbox": sandbox,
        }
        if separate:
            config["grading_sandbox"] = sandbox
        runtime = create_runtime(config, context=context)
        harness = rig(
            runtime,
            resources=resource_factory(registry),
            run_id=run_id,
            setup_timeout_seconds=150,
            rollout_timeout_seconds=180,
            grading_timeout_seconds=180,
            cleanup_timeout_seconds=180,
            cancellation_grace_seconds=2,
        )
        drivers.append(harness.driver)
        (task,) = await runtime.prepare()
        return NS(rig=harness, task=task, owned=owned)

    try:
        yield make
    finally:
        try:
            for driver in drivers:
                await driver.drain()
        finally:
            try:
                report = await registry.release_all()
                assert report.complete, report
                assert await owned() == []
            finally:
                await registry.close()


async def assert_closed(handle):
    assert isinstance(handle, Blaxel)
    with pytest.raises(RuntimeError):
        await handle.client()


async def test_shared_sandbox_example_releases_after_grading(build):
    built = await build(SharedProbe)
    admitted = await built.rig.driver.sample(built.task, "g", "g-r1")
    assert len(await built.owned()) == 1
    assert (await built.rig.driver.grade(admitted)).reward == 1
    await built.rig.driver.drain()
    instance = SharedProbe.instances[-1]
    assert instance.closed.execution_sandbox is instance.execution
    await assert_closed(instance.execution)
    assert await built.owned() == []


async def test_separate_verifier_example_releases_execution_and_retries(build):
    built = await build(RetryProbe, separate=True)
    admitted = await built.rig.driver.sample(built.task, "g", "g-r1")
    instance = RetryProbe.instances[-1]
    await assert_closed(instance.execution)
    assert await built.owned() == []
    assert (await built.rig.driver.grade(admitted)).reward == 1
    await built.rig.driver.drain()
    assert len(instance.verifiers) == 2
    assert instance.verifiers[0] is instance.verifiers[1]
    assert instance.clients[0] is instance.clients[1]
    assert instance.closed.execution_sandbox is None
    assert instance.closed.grading_sandbox is instance.verifiers[-1]
    await assert_closed(instance.verifiers[-1])
    assert await built.owned() == []


async def test_cancelled_sampling_releases_shared_sandbox(build):
    CancelProbe.started = asyncio.Event()
    built = await build(CancelProbe)
    sampling = asyncio.create_task(built.rig.driver.sample(built.task, "g", "g-r1"))
    try:
        await asyncio.wait_for(CancelProbe.started.wait(), timeout=180)
        assert len(await built.owned()) == 1
    finally:
        sampling.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sampling
    await built.rig.driver.drain()
    instance = SharedProbe.instances[-1]
    assert instance.closed is not None
    await assert_closed(instance.execution)
    assert await built.owned() == []
