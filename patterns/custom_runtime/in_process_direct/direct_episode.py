"""An in-process agent with direct model access, as an ``Episode``.

Each task is an ``AnswerTask``, an instruction with one expected answer, from the
dataset that ``[rollout_runtime.files] tasks`` names. ``loops check`` refuses a line
that is not one. ``run()`` is the agent. Inside the training process, it asks the model
once with the cookbook's ``llm()`` client, which records the exact tokens it samples.

``AnswerEpisode``, which ``train.toml`` runs with no environment, keeps the answer, and
``grade()`` scores it. ``SandboxAnswerEpisode``, which ``train_blaxel.toml`` runs with
``[rollout_runtime.config.sandbox]``, writes the answer into the attempt's Blaxel
sandbox, and ``grade()`` scores what the sandbox holds. Grading is pure either way, so
it is declared safe to retry."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from loops_cookbook.episodes import Episode, Grading, Model, Sampling, Setup, Task
from loops_cookbook.providers.blaxel import Blaxel

# The answer's path inside a sandbox.
ANSWER = "/tmp/answer.txt"


class AnswerTask(BaseModel):
    """One line of the dataset."""

    model_config = ConfigDict(extra="forbid")

    # The name is for people reading the dataset. The episode doesn't use it.
    name: str
    instruction: str
    answer: str


async def ask(model: Model, instruction: str) -> str:
    """Return the model's one reply to ``instruction``. An empty reply is a wrong answer."""
    conversation = model.conversation()
    reply = await conversation.generate(messages=[{"role": "user", "content": instruction}])
    return (reply.content or "").strip()


class AnswerEpisode(Episode[AnswerTask, None]):
    """One question asked once, scored against the task's answer."""

    grading_retries = 2

    def __init__(self, task: Task[AnswerTask], options: None) -> None:
        super().__init__(task, options)
        # The answer stays None until the model replies, and None scores as wrong.
        self.answer: str | None = None

    async def run(self, sample: Sampling) -> None:
        model = sample.model
        self.answer = await ask(model, self.task.value.instruction)

    async def grade(self, grading: Grading) -> float:
        return float(self.answer == self.task.value.answer)


class SandboxAnswerEpisode(Episode[AnswerTask, None]):
    """The same episode on a Blaxel sandbox, which holds the answer that ``grade()`` scores."""

    execution_kind = Blaxel

    grading_retries = 2

    async def setup(self, setup: Setup) -> None:
        sandbox = setup.sandbox
        assert isinstance(sandbox, Blaxel)
        # Attaching here counts against the setup deadline, not the agent's.
        await sandbox.client()

    async def run(self, sample: Sampling) -> None:
        model = sample.model
        sandbox = sample.sandbox
        assert isinstance(sandbox, Blaxel)
        answer = await ask(model, self.task.value.instruction)
        box = await sandbox.client()
        # This write is the agent's tool call. It puts the answer in the sandbox.
        await box.fs.write(ANSWER, answer)

    async def grade(self, grading: Grading) -> float:
        sandbox = grading.execution_sandbox
        assert isinstance(sandbox, Blaxel)
        # read_file returns None when the run wrote no answer, which scores as wrong.
        held = await sandbox.read_file(ANSWER)
        return float(held == self.task.value.answer)
