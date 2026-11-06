"""An in-process agent built on LiteLLM, streaming every reply, as an ``Episode``.

The ``AnswerEpisode`` of ``patterns/custom_runtime/in_process_openai_http`` with only
the client changed: ``run()`` reaches the attempt's OpenAI-compatible endpoint through
LiteLLM's ``acompletion`` with ``stream=True``, as an agent built on LiteLLM would,
with the endpoint given as an OpenAI-compatible ``api_base``. The run's gateway samples
the whole reply, records its exact tokens, and then streams it back as server-sent
events, so the chunks arrive together at the end rather than token by token; the agent
assembles them as it would any stream.

Each task is an ``AnswerTask``, an instruction with one expected answer, from the
dataset that ``[runtime.files] tasks`` names; ``loops check`` refuses a line that is
not one. The episode keeps the answer, and ``grade()`` scores it. Grading is pure, so
it is declared safe to retry.
"""

from __future__ import annotations

import litellm
from pydantic import BaseModel, ConfigDict

from loops_cookbook.episodes import Episode, Grading, Sampling, Task
from loops_cookbook.model_access import ModelAccessMode


class AnswerTask(BaseModel):
    """One line of the dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str  # for people reading the dataset; the episode does not use it
    instruction: str
    answer: str


class AnswerEpisode(Episode[AnswerTask, None]):
    """One question asked once, scored against the task's answer."""

    model_access = ModelAccessMode.OPENAI_HTTP
    grading_retries = 2

    def __init__(self, task: Task[AnswerTask], options: None) -> None:
        super().__init__(task, options)
        self.answer: str | None = None  # None until the model replies; scored as wrong

    async def run(self, sample: Sampling) -> None:
        model = sample.model
        connection = await model.http_connection()
        stream = await litellm.acompletion(
            model=f"openai/{connection.model}",  # any OpenAI-compatible endpoint
            api_base=connection.base_url,
            api_key=connection.api_key.get_secret_value(),
            messages=[{"role": "user", "content": self.task.value.instruction}],
            stream=True,
        )
        parts = [chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices]
        self.answer = "".join(parts).strip()

    async def grade(self, grading: Grading) -> float:
        return float(self.answer == self.task.value.answer)
