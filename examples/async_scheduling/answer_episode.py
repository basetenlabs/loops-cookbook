"""The answer episode from patterns/custom_runtime/in_process_direct, trained asynchronously.

The episode is unchanged: async scheduling is a setting of the run, not of the episode.
``AnswerEpisode`` (``train.toml``) keeps its answer and ``grade()`` scores it;
``SandboxAnswerEpisode`` (``train_blaxel.toml``) writes the answer into the attempt's
Blaxel sandbox, and ``grade()`` scores what the sandbox holds.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from loops_cookbook.episodes import Episode, Grading, Model, Sampling, Setup, Task
from loops_cookbook.providers.blaxel import Blaxel

ANSWER = "/tmp/answer.txt"  # where the answer lives in a sandbox


class AnswerTask(BaseModel):
    """One line of the dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str  # for people reading the dataset; the episode does not use it
    instruction: str
    answer: str


async def ask(model: Model, instruction: str) -> str:
    """The model's one reply to ``instruction``; one that won't parse is a wrong answer."""
    conversation = model.conversation()
    reply = await conversation.generate(messages=[{"role": "user", "content": instruction}])
    return (reply.content or "").strip()


class AnswerEpisode(Episode[AnswerTask, None]):
    """One question asked once, scored against the task's answer."""

    grading_retries = 2

    def __init__(self, task: Task[AnswerTask], options: None) -> None:
        super().__init__(task, options)
        self.answer: str | None = None  # None until the model replies; scored as wrong

    async def run(self, sample: Sampling) -> None:
        model = sample.model
        self.answer = await ask(model, self.task.value.instruction)

    async def grade(self, grading: Grading) -> float:
        return float(self.answer == self.task.value.answer)


class SandboxAnswerEpisode(Episode[AnswerTask, None]):
    """The same episode on a Blaxel sandbox: the answer goes into it and is graded from it."""

    execution_kind = Blaxel

    grading_retries = 2

    async def setup(self, setup: Setup) -> None:
        sandbox = setup.sandbox
        assert isinstance(sandbox, Blaxel)
        await sandbox.client()  # attached here, on the setup clock, not the agent's

    async def run(self, sample: Sampling) -> None:
        model = sample.model
        sandbox = sample.sandbox
        assert isinstance(sandbox, Blaxel)
        answer = await ask(model, self.task.value.instruction)
        box = await sandbox.client()
        await box.fs.write(ANSWER, answer)  # the tool call: into the sandbox

    async def grade(self, grading: Grading) -> float:
        sandbox = grading.execution_sandbox
        assert isinstance(sandbox, Blaxel)
        held = await sandbox.read_file(ANSWER)  # None: a run that wrote none
        return float(held == self.task.value.answer)
