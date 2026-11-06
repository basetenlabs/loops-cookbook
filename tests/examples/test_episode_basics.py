"""The public hooks through the existing driver and registered-resource lifecycle."""

import asyncio
from dataclasses import FrozenInstanceError
from types import SimpleNamespace as NS
from typing import ClassVar, assert_type

import pytest
from episodes import (
    HostScoredEpisode,
    Options,
    Question,
    SeparateVerifierEpisode,
    SharedSandboxEpisode,
)
from support.builders import runtime_context, task_file
from support.episode_rig import live_sandboxes
from support.resources import Fake, not_created

from loops_cookbook.episodes import Closing, Grading, Sampling, Setup, Task
from loops_cookbook.providers.blaxel import Blaxel
from loops_cookbook.runtime import JsonValue
from loops_cookbook.runtime.driver import Discard

TASKS = ({"id": "t", "prompt": "q", "answer": "ok"},)


class Tracked(HostScoredEpisode):
    execution_kind = Fake
    grading_kind = Fake
    sandbox_lifetime = "sampling"
    grading_retries = 1
    instances: ClassVar[list["Tracked"]] = []

    def __init__(self, task: Task[Question], options: Options) -> None:
        super().__init__(task, options)
        assert_type(self.task.value, Question)
        assert_type(self.options, Options)
        self.grades: list[Grading] = []
        self.closed: Closing | None = None
        self.clients: list[object] = []
        self.instances.append(self)

    async def run(self, sample: Sampling) -> dict[str, JsonValue]:
        assert isinstance(sample.sandbox, Fake)
        await sample.sandbox.client()
        return await super().run(sample)

    async def grade(self, grading: Grading) -> float:
        self.grades.append(grading)
        verifier = grading.grading_sandbox
        assert isinstance(verifier, Fake)
        self.clients.append(await verifier.client())
        if len(self.grades) == 1:
            raise ConnectionError("retry the verifier")
        return await super().grade(grading)

    async def close(self, closing: Closing) -> None:
        self.closed = closing


class SharedTracked(Tracked):
    grading_kind = None
    sandbox_lifetime = "attempt"

    async def grade(self, grading: Grading) -> float:
        self.grades.append(grading)
        assert isinstance(grading.execution_sandbox, Fake)
        self.clients.append(await grading.execution_sandbox.client())
        return await HostScoredEpisode.grade(self, grading)


class TimeoutTracked(Tracked):
    async def run(self, sample: Sampling) -> dict[str, JsonValue]:
        await super().run(sample)
        await asyncio.Event().wait()
        raise AssertionError("the agent timeout should interrupt this hook")


async def test_host_episode_uses_typed_options_without_a_context(make):
    built = await make(episode=HostScoredEpisode, sandbox=False, tasks=TASKS)
    assert (await built.rig.rollout("g-r1", built.task)).reward == 1
    await built.rig.driver.drain()
    assert built.provider.calls == []


@pytest.fixture
async def on_blaxel(tmp_path, blaxel):
    """``await on_blaxel(episode)``: one rollout of the example ``episode`` through the
    real driver, its sandboxes from the cookbook's Blaxel adapter over the fake Blaxel."""
    from support.fake_blaxel import SETTINGS
    from support.harness import rig
    from support.resources import episodes, local

    from loops_cookbook.episodes import create_runtime
    from loops_cookbook.resources.registry import ResourceRegistry
    from loops_cookbook.runtime.context import EvidenceRecord

    adapters = {"blaxel": blaxel.adapter(SETTINGS)}
    registry = ResourceRegistry.open(
        tmp_path / "state", adapters, mode="write", run_id="run", kind=local
    )

    async def rollout(episode):
        sandbox = {"provider": "blaxel", "image": "img:1", "memory_mb": 2048, "ttl": "1h"}
        config = {"episode": f"episodes:{episode.__name__}", "sandbox": sandbox}
        if episode.grading_kind is not None:
            config["grading_sandbox"] = sandbox
        limits = NS(
            rollout_timeout_seconds=None,
            cancellation_grace_seconds=0.2,
            process_kill_grace_seconds=0.2,
            provider_deadlines={"blaxel": SETTINGS.deadlines},
            trajectory_tokens=None,
        )
        context = runtime_context(
            task_file(tmp_path, list(TASKS)),
            limits=limits,
            work_dir=tmp_path / "work",
            evidence=EvidenceRecord(),
            sandbox_providers={"blaxel": Blaxel},
        )
        config["agent_timeout_seconds"] = 5.0
        runtime = create_runtime(config, context=context)  # pyright: ignore[reportArgumentType]
        driven = rig(
            runtime,
            resources=episodes(registry),
            rollout_timeout_seconds=5.0,
            cancellation_grace_seconds=limits.cancellation_grace_seconds,
            process_kill_grace_seconds=limits.process_kill_grace_seconds,
        )
        (task,) = await runtime.prepare()
        result = await driven.rollout("g-r1", task)
        await driven.driver.drain()
        return result

    yield rollout
    await registry.close()


@pytest.mark.parametrize("episode", [SharedSandboxEpisode, SeparateVerifierEpisode])
async def test_standalone_sandbox_examples_transfer_and_score_the_answer(
    blaxel, on_blaxel, episode
):
    assert (await on_blaxel(episode)).reward == 1
    # The answer was written into each sandbox and read back from it.
    separate = episode is SeparateVerifierEpisode
    assert list(blaxel.files.values()) == [{"/tmp/answer.txt": "ok"}] * (2 if separate else 1)
    assert blaxel.clients and all(client.closed for client in blaxel.clients)
    assert blaxel.api.sandboxes == {}  # every sandbox released


async def test_shared_grading_keeps_the_execution_sandbox_through_close(make):
    built = await make(episode=SharedTracked, tasks=TASKS)
    admitted = await built.rig.driver.sample(built.task, "g", "g-r1")
    assert live_sandboxes(built.provider) != []
    assert (await built.rig.driver.grade(admitted)).reward == 1
    await built.rig.driver.drain()
    instance = Tracked.instances[-1]
    assert instance.closed is not None
    assert instance.closed.execution_sandbox is instance.grades[0].execution_sandbox
    assert instance.closed.grading_sandbox is None
    assert [request.logical_key for request in built.provider.created] == ["sandbox"]
    assert live_sandboxes(built.provider) == []


async def test_agent_timeout_is_recorded_in_the_grading_phase(make):
    built = await make(
        episode=TimeoutTracked,
        tasks=TASKS,
        agent_timeout=0.02,
        grading_sandbox={"provider": "fake"},
    )
    assert (await built.rig.rollout("g-r1", built.task)).reward == 1
    await built.rig.driver.drain()
    assert all(grade.timed_out for grade in Tracked.instances[-1].grades)
    assert live_sandboxes(built.provider) == []


async def test_phase_resources_follow_early_release_and_grading_retries(make):
    built = await make(episode=Tracked, tasks=TASKS, grading_sandbox={"provider": "fake"})
    admitted = await built.rig.driver.sample(built.task, "g", "g-r1")
    assert live_sandboxes(built.provider) == []
    assert [request.logical_key for request in built.provider.created] == ["sandbox"]
    assert (await built.rig.driver.grade(admitted)).reward == 1
    await built.rig.driver.drain()
    instance = Tracked.instances[-1]
    assert len(instance.grades) == 2
    assert all(grade.execution_sandbox is None for grade in instance.grades)
    assert instance.clients[0] is instance.clients[1]
    assert instance.closed is not None and instance.closed.execution_sandbox is None
    assert instance.closed.grading_sandbox is instance.grades[-1].grading_sandbox
    assert all(client.closed == 1 for client in built.provider.clients)
    assert live_sandboxes(built.provider) == []


async def test_discard_before_grading_has_no_verifier_in_close(make):
    built = await make(episode=Tracked, tasks=TASKS, grading_sandbox={"provider": "fake"})
    admitted = await built.rig.driver.sample(built.task, "g", "g-r1")
    built.rig.driver.discard_admitted(admitted)
    await built.rig.driver.drain()
    closing = Tracked.instances[-1].closed
    assert closing is not None
    assert closing.execution_sandbox is closing.grading_sandbox is None
    assert [request.logical_key for request in built.provider.created] == ["sandbox"]


async def test_failed_verifier_provisioning_does_not_invent_a_closing_resource(make):
    built = await make(episode=Tracked, tasks=TASKS, grading_sandbox={"provider": "fake"})
    admitted = await built.rig.driver.sample(built.task, "g", "g-r1")
    built.provider.create_errors = [not_created("quota"), not_created("quota")]
    assert await built.rig.driver.grade(admitted) is Discard.GRADING_FAILED
    await built.rig.driver.drain()
    closing = Tracked.instances[-1].closed
    assert closing is not None
    assert closing.execution_sandbox is closing.grading_sandbox is None
    assert live_sandboxes(built.provider) == []


class ConcurrentHost(HostScoredEpisode):
    active_sampling: ClassVar[int] = 0
    active_grading: ClassVar[int] = 0
    peak_sampling: ClassVar[int] = 0
    peak_grading: ClassVar[int] = 0
    instances: ClassVar[list["ConcurrentHost"]] = []

    def __init__(self, task: Task[Question], options: Options) -> None:
        super().__init__(task, options)
        self.instances.append(self)
        self.phases: list[Setup | Sampling | Grading | Closing] = []

    async def setup(self, setup: Setup) -> None:
        self.phases.append(setup)

    async def run(self, sample: Sampling) -> dict[str, JsonValue]:
        self.phases.append(sample)
        type(self).active_sampling += 1
        type(self).peak_sampling = max(self.peak_sampling, self.active_sampling)
        try:
            await asyncio.sleep(0.02)
            return await super().run(sample)
        finally:
            type(self).active_sampling -= 1

    async def grade(self, grading: Grading) -> float:
        self.phases.append(grading)
        type(self).active_grading += 1
        type(self).peak_grading = max(self.peak_grading, self.active_grading)
        try:
            await asyncio.sleep(0.02)
            return await super().grade(grading)
        finally:
            type(self).active_grading -= 1

    async def close(self, closing: Closing) -> None:
        self.phases.append(closing)


async def test_host_phases_keep_concurrency_limits_and_isolate_attempts(make):
    ConcurrentHost.instances = []
    ConcurrentHost.peak_sampling = ConcurrentHost.peak_grading = 0
    built = await make(
        episode=ConcurrentHost,
        sandbox=False,
        tasks=TASKS,
        sampling_slots=2,
        verification_slots=1,
    )
    admitted = await asyncio.gather(
        *(built.rig.driver.sample(built.task, "g", f"g-r{i}") for i in range(6))
    )
    results = await asyncio.gather(*(built.rig.driver.grade(item) for item in admitted))
    await built.rig.driver.drain()
    assert [result.reward for result in results] == [1] * 6
    assert ConcurrentHost.peak_sampling == 2
    assert ConcurrentHost.peak_grading == 1
    assert built.provider.calls == []
    assert len({id(episode.task.value) for episode in ConcurrentHost.instances}) == 6
    assert len({id(episode.options) for episode in ConcurrentHost.instances}) == 6
    for episode in ConcurrentHost.instances:
        assert not hasattr(episode, "ctx")
        assert [type(phase) for phase in episode.phases] == [Setup, Sampling, Grading, Closing]
        assert len({phase.attempt_id for phase in episode.phases}) == 1
        assert all(
            hasattr(phase, "model") == isinstance(phase, Sampling) for phase in episode.phases
        )
        for phase in episode.phases:
            with pytest.raises(FrozenInstanceError):
                phase.attempt_id = "another-attempt"  # pyright: ignore[reportAttributeAccessIssue]
