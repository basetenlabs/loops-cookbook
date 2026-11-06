"""Host-only hill climb: sample one A–D label; reward the hidden target (B by default)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from loops_cookbook.episodes import Episode, Grading, Sampling, Task
from loops_cookbook.runtime import JsonValue


class HillClimbTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = (
        "Choose exactly one of these four labels: A, B, C, D.\n"
        "Reply with the single label only, without explanation or punctuation.\n"
        "Your selection is scored by a hidden deterministic rule."
    )
    target: Literal["A", "B", "C", "D"] = "B"


class HillClimbEpisode(Episode[HillClimbTask, None]):
    def __init__(self, task: Task[HillClimbTask], options: None) -> None:
        super().__init__(task, options)
        # A timeout before a reply leaves an incorrect guess, not missing state.
        self.guess = ""

    async def run(self, sample: Sampling) -> dict[str, JsonValue]:
        reply = await sample.model.conversation().generate(
            messages=[{"role": "user", "content": self.task.value.prompt}]
        )
        self.guess = (reply.content or "").strip()
        return {"guess": self.guess}

    async def grade(self, grading: Grading) -> float:
        return float(self.guess == self.task.value.target)
