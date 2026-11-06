"""Strict pyright checks for data types, inheritance, and explicit provider narrowing.

Not a test module: pytest collects nothing here. pyproject.toml lists this file among
pyright's strict paths, and pyright is what fails if a type here stops checking.
"""

from __future__ import annotations

from typing import assert_type

from blaxel.core.sandbox import SandboxInstance
from pydantic import BaseModel, ConfigDict

from loops_cookbook.episodes import Closing, Episode, Grading, Model, Sampling, Setup
from loops_cookbook.providers.blaxel import Blaxel
from loops_cookbook.runtime import Rollout


class MathTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str
    answer: str


class JudgeOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str


class Judged(Episode[MathTask, JudgeOptions]):
    async def run(self, sample: Sampling) -> None:
        assert_type(self.task.value, MathTask)
        assert_type(self.task.value.prompt, str)
        assert_type(self.options, JudgeOptions)
        assert_type(self.options.url, str)
        assert_type(sample.model, Model)

    async def grade(self, grading: Grading) -> float:
        assert_type(grading.rollout, Rollout)
        assert_type(grading.timed_out, bool)
        return float(self.task.value.answer == "4")


class Refined(Judged):
    async def run(self, sample: Sampling) -> None:
        assert_type(self.task.value, MathTask)
        assert_type(self.options, JudgeOptions)


class GenericJudged[TaskT: BaseModel | None](Episode[TaskT, JudgeOptions]):
    pass


class BoundJudged(GenericJudged[MathTask]):
    async def run(self, sample: Sampling) -> None:
        assert_type(self.task.value, MathTask)
        assert_type(self.options, JudgeOptions)


class Plain(Episode[None, None]):
    async def run(self, sample: Sampling) -> None:
        assert_type(self.task.value, None)
        assert_type(self.options, None)

    async def grade(self, grading: Grading) -> float:
        return 0.0


class SeparateVerifier(Episode[MathTask, JudgeOptions]):
    execution_kind = Blaxel
    grading_kind = Blaxel
    sandbox_lifetime = "sampling"

    async def setup(self, setup: Setup) -> None:
        sandbox = setup.require_sandbox(Blaxel)
        assert_type(await sandbox.client(), SandboxInstance)

    async def run(self, sample: Sampling) -> None:
        sandbox = sample.require_sandbox(Blaxel)
        assert_type(await sandbox.client(), SandboxInstance)
        assert_type(await sandbox.alive(), bool)

    async def grade(self, grading: Grading) -> float:
        verifier = grading.require_grading_sandbox(Blaxel)
        assert_type(await verifier.client(), SandboxInstance)
        assert_type(self.options, JudgeOptions)
        return 0.0

    async def close(self, closing: Closing) -> None:
        # Failed setup or verifier provisioning may leave either resource absent.
        if isinstance(closing.grading_sandbox, Blaxel):
            assert_type(await closing.grading_sandbox.client(), SandboxInstance)
