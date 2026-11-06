"""Episodes through the real driver, capacity broker, lease and resource registry.

Every episode here is ``Scripted``: its hooks do what the test's ``script`` says, and
every call is recorded, so a test states one hostile behaviour and reads back what
the cookbook did about it. ``make`` builds an ``EpisodeRuntime`` from its factory in a
harness rig over a fake sandbox provider. Registered for every test by the root
conftest.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace as NS
from typing import Any

import pytest

from loops_cookbook.episodes import Closing, Episode, Grading, Sampling, Setup, create_runtime
from loops_cookbook.runtime.context import EvidenceRecord
from support.builders import runtime_context, task_file
from support.harness import rig
from support.resources import FAST, Fake, FakeSandboxProvider, episodes, registry_at


async def answer(episode: Scripted, model: Any) -> dict[str, Any]:
    """The default agent: ask the model once and keep its reply."""
    reply = await model.llm().conversation().generate(messages=[{"role": "user", "content": "q"}])
    episode.answer = reply.content
    return {"answer": reply.content}


async def score(episode: Scripted, rollout: Any) -> float:
    return float(episode.answer == "ok")  # the harness renderer's every reply


async def nothing(episode: Scripted, *args: Any) -> None:
    return None


@dataclass
class Script:
    setup: Any = nothing
    run: Any = answer
    grade: Any = score
    close: Any = nothing
    events: list[tuple[str, Scripted]] = field(default_factory=list)

    def of(self, name: str) -> list[Scripted]:
        return [episode for event, episode in self.events if event == name]


SCRIPT = Script()


class Scripted(Episode[None, None]):
    execution_kind = Fake

    def __init__(self, task, options):
        super().__init__(task, options)
        self.answer: str | None = None
        self.phase: Setup | Sampling | Grading | Closing
        self.sandbox = None
        SCRIPT.events.append(("init", self))

    async def setup(self, setup: Setup):
        self.phase = setup
        self.sandbox = setup.sandbox
        self.sandbox_at_setup = setup.sandbox
        SCRIPT.events.append(("setup", self))
        await SCRIPT.setup(self)

    async def run(self, sample: Sampling):
        self.phase = self.sampling = sample
        model = sample.model
        SCRIPT.events.append(("run", self))
        return await SCRIPT.run(self, model)

    async def grade(self, grading: Grading):
        self.phase = grading
        self.grading_phase = grading
        rollout = grading.rollout
        SCRIPT.events.append(("grade", self))
        return await SCRIPT.grade(self, rollout)

    async def close(self, closing: Closing):
        self.phase = closing
        SCRIPT.events.append(("close", self))
        await SCRIPT.close(self)


class Unsandboxed(Scripted, Episode[None, None]):
    execution_kind = None


@pytest.fixture
def script():
    SCRIPT.__init__()  # pyright: ignore[reportUnknownMemberType]
    yield SCRIPT
    SCRIPT.__init__()  # pyright: ignore[reportUnknownMemberType]


@pytest.fixture
async def make(tmp_path, script):
    """``make(**settings)``: an EpisodeRuntime from its factory, in a rig with a sandbox provider."""
    registries = []

    async def build(
        episode=Scripted,
        *,
        sandbox=True,
        agent_timeout=1.0,
        rollout_timeout=5.0,
        grace=0.2,
        kill_grace=0.2,
        tasks=({"id": "t", "prompt": "q"},),
        options=None,
        deadlines=FAST,
        provides=Fake,
        grading_sandbox=None,
        grading_provider=None,
        grading_kind=Fake,
        trajectory_tokens=None,
        **settings,
    ):
        provider = FakeSandboxProvider(deadlines=deadlines)
        verifier_provider = (
            None
            if grading_provider is None
            else FakeSandboxProvider(name=grading_provider, deadlines=deadlines)
        )
        providers = [provider] + ([] if verifier_provider is None else [verifier_provider])
        registry = registry_at(tmp_path / f"state-{len(registries)}", *providers, run_id="run")
        registries.append(registry)
        config: dict[str, Any] = {
            "episode": f"{episode.__module__}:{episode.__name__}",
            "agent_timeout_seconds": agent_timeout,
        }
        if sandbox:
            config["sandbox"] = {"provider": "fake"}
        if grading_sandbox is not None:
            config["grading_sandbox"] = grading_sandbox
        if options is not None:
            config["options"] = options
        context = runtime_context(
            task_file(tmp_path, list(tasks)),
            limits=NS(
                rollout_timeout_seconds=None,
                cancellation_grace_seconds=grace,
                process_kill_grace_seconds=kill_grace,
                provider_deadlines={item.name: deadlines for item in providers},
                trajectory_tokens=trajectory_tokens,
            ),
            work_dir=tmp_path / "work",
            evidence=EvidenceRecord(),
            sandbox_providers={
                "fake": provides,
                **({} if verifier_provider is None else {verifier_provider.name: grading_kind}),
            },
        )
        runtime = create_runtime(config, context=context)
        r = rig(
            runtime,
            resources=episodes(registry),
            rollout_timeout_seconds=rollout_timeout,
            cancellation_grace_seconds=grace,
            process_kill_grace_seconds=kill_grace,
            **settings,
        )
        (task,) = await runtime.prepare()
        return NS(
            rig=r,
            runtime=runtime,
            task=task,
            provider=provider,
            verifier_provider=verifier_provider,
            evidence=context.evidence,
            registry=registry,
        )

    yield build
    for registry in registries:
        await registry.close()


def live_sandboxes(provider: FakeSandboxProvider) -> list[str]:
    return sorted(resource.name for resource in provider.store.values())


async def until(predicate: Callable[[], bool]) -> None:
    """Wait for a state the test names, rather than for a guess at how long it takes."""
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.001)


def calls(provider: FakeSandboxProvider, operation: str) -> int:
    return [op for op, _ in provider.calls].count(operation)
