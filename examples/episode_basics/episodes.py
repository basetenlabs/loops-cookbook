"""Standalone host, shared-sandbox and separate-verifier Episode definitions."""

from pydantic import BaseModel, ConfigDict

from loops_cookbook.episodes import Episode, Grading, Sampling, Task
from loops_cookbook.providers.blaxel import Blaxel
from loops_cookbook.runtime import JsonValue


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str
    answer: str


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strip_whitespace: bool = True


class HostScoredEpisode(Episode[Question, Options]):
    def __init__(self, task: Task[Question], options: Options) -> None:
        super().__init__(task, options)
        self.answer = ""

    async def run(self, sample: Sampling) -> dict[str, JsonValue]:
        reply = await sample.model.conversation().generate(
            messages=[{"role": "user", "content": self.task.value.prompt}]
        )
        self.answer = reply.content or ""
        if self.options.strip_whitespace:
            self.answer = self.answer.strip()
        return {"answer": self.answer}

    async def grade(self, grading: Grading) -> float:
        return float(self.answer == self.task.value.answer)


class SharedSandboxEpisode(Episode[Question, Options]):
    execution_kind = Blaxel

    def __init__(self, task: Task[Question], options: Options) -> None:
        super().__init__(task, options)
        self.answer = ""

    async def run(self, sample: Sampling) -> dict[str, JsonValue]:
        reply = await sample.model.conversation().generate(
            messages=[{"role": "user", "content": self.task.value.prompt}]
        )
        self.answer = reply.content or ""
        if self.options.strip_whitespace:
            self.answer = self.answer.strip()
        sandbox = sample.require_sandbox(Blaxel)
        await sandbox.write_file("/tmp/answer.txt", self.answer)
        return {"answer": self.answer}

    async def grade(self, grading: Grading) -> float:
        sandbox = grading.require_execution_sandbox(Blaxel)
        saved = await sandbox.read_file("/tmp/answer.txt")
        return float(saved == self.task.value.answer)


class SeparateVerifierEpisode(Episode[Question, Options]):
    execution_kind = Blaxel
    sandbox_lifetime = "sampling"
    grading_kind = Blaxel

    def __init__(self, task: Task[Question], options: Options) -> None:
        super().__init__(task, options)
        self.answer = ""

    async def run(self, sample: Sampling) -> dict[str, JsonValue]:
        reply = await sample.model.conversation().generate(
            messages=[{"role": "user", "content": self.task.value.prompt}]
        )
        self.answer = reply.content or ""
        if self.options.strip_whitespace:
            self.answer = self.answer.strip()
        sandbox = sample.require_sandbox(Blaxel)
        await sandbox.write_file("/tmp/answer.txt", self.answer)
        # Keep the output on the host before the execution sandbox is released.
        self.answer = await sandbox.read_file("/tmp/answer.txt") or ""
        return {"answer": self.answer}

    async def grade(self, grading: Grading) -> float:
        verifier = grading.require_grading_sandbox(Blaxel)
        await verifier.write_file("/tmp/answer.txt", self.answer)
        saved = await verifier.read_file("/tmp/answer.txt")
        return float(saved == self.task.value.answer)
