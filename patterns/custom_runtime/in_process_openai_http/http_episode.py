"""An in-process agent on a stock OpenAI client, as an ``Episode``.

Each task is an ``AnswerTask``, an instruction with one expected answer, from the
dataset that ``[runtime.files] tasks`` names; ``loops check`` refuses a line that is
not one. ``run()`` is the agent: inside the training process, it asks the model once
through an OpenAI client pointed at the attempt's OpenAI-compatible endpoint, as an
agent built on any OpenAI client would. The run's gateway behind that endpoint records
the exact tokens of every reply; the credential is the attempt's, and it is revoked
when the rollout is sealed.

``AnswerEpisode`` (``train.toml``, no environment) keeps the answer, and ``grade()``
scores it. ``SandboxAnswerEpisode`` (``train_blaxel.toml``, with
``[runtime.config.sandbox]``) writes the answer into the attempt's Blaxel sandbox, and
``grade()`` scores what the sandbox holds. Grading is pure either way, so it is
declared safe to retry.
"""

from __future__ import annotations

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from loops_cookbook.episodes import Episode, Grading, Model, Sampling, Setup, Task
from loops_cookbook.model_access import ModelAccessMode
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
    connection = await model.http_connection()
    async with AsyncOpenAI(
        base_url=connection.base_url, api_key=connection.api_key.get_secret_value()
    ) as client:
        completion = await client.chat.completions.create(
            model=connection.model, messages=[{"role": "user", "content": instruction}]
        )
    return (completion.choices[0].message.content or "").strip()


class AnswerEpisode(Episode[AnswerTask, None]):
    """One question asked once, scored against the task's answer."""

    model_access = ModelAccessMode.OPENAI_HTTP
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

    model_access = ModelAccessMode.OPENAI_HTTP
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
