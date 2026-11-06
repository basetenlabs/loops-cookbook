"""An RL run assembled from fakes: a memory trainer, a one-task runtime, a scripted template."""

import asyncio
import json

import pytest
from support.fakes import MemoryTrainer
from support.harness import Renderer

from loops_cookbook.evidence import TaskRef
from loops_cookbook.model_access.api import ModelAccessMode
from loops_cookbook.rl import recipe
from loops_cookbook.runtime.api import EpisodeResult, GradeResult, RuntimeCapabilities

TASK = TaskRef("task-agent", "sha256:" + "0" * 64)


def unused(*args, **kwargs):
    raise AssertionError("this run never preflights")


class MemoryRuntime:
    """A native runtime with one task and one generation per episode.

    Rollout ``r1`` of each group is rewarded, the rest are not. ``context`` is the
    runtime context the run built it with.
    """

    capabilities = RuntimeCapabilities(model_access_modes=frozenset({ModelAccessMode.DIRECT}))

    def __init__(self):
        self.closed = False
        self.samples = []
        self.generations = 1
        self.context = None

    async def prepare(self):
        return [TASK]

    async def sample(self, attempt):
        conversation = attempt.services.model_access.llm().conversation()
        for _ in range(self.generations):
            await conversation.generate(messages=[{"role": "user", "content": "x"}])
        return EpisodeResult()

    async def grade(self, attempt, rollout):
        await asyncio.sleep(0)
        self.samples.append(rollout)
        return GradeResult(reward=float(rollout.identity.rollout_id.endswith("r1")))

    async def finish(self, attempt, rollout, disposition):
        pass

    async def abort(self, attempt, reason):
        pass

    async def close(self):
        self.closed = True


class KeepingTrainer(MemoryTrainer):
    """A memory trainer whose updates are kept: each checkpoint names the trainer state
    ``bt://loops:test/state/<step>`` (none when not ``scheduled``), and the way out keeps
    ``state-<step>-v<version>``."""

    def __init__(self, *, scheduled=True):
        super().__init__()
        self.scheduled = scheduled

    async def checkpoint(self, report, *, final=False, keep_sampler_weights=False):
        if self.scheduled:
            report.checkpoint_state_path = f"bt://loops:test/state/{report.step}"
        return report

    async def checkpoint_state_only(self, step):
        return f"state-{step}-v{self.policy_version}"


@pytest.fixture
def rl_run(config):
    """``rl_run(trainer=None, runtime=None, *, adapters=None, config_dir=None, **settings)``:
    the run ``loops run`` would assemble from ``config(**settings)``, with a
    ``MemoryTrainer`` and a ``MemoryRuntime`` unless given; ``(run, trainer, runtime)``."""

    def build(trainer=None, runtime=None, *, adapters=None, config_dir=None, **settings):
        trainer = MemoryTrainer() if trainer is None else trainer
        runtime = MemoryRuntime() if runtime is None else runtime

        def factory(_settings, *, context):
            runtime.context = context
            return runtime

        run = recipe.assemble(
            config(**settings),
            trainer=trainer,
            runtime_factory=factory,
            renderer_factory=lambda _config: Renderer(),
            adapters=adapters,
            config_dir=config_dir,
        )
        return run, trainer, runtime

    return build


def records(run, name="rollouts"):
    """The run's ``<name>.jsonl`` records."""
    return [json.loads(line) for line in (run.out / f"{name}.jsonl").read_text().splitlines()]
