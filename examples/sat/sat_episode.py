"""3-SAT, one model call per problem: find values for the variables that make every
clause true.

Each problem is generated from its seed: random clauses of three literals over
``variables`` variables, ``clause_ratio`` clauses per variable, each kept only if a
hidden assignment satisfies it, so every problem has a solution. Near a ratio of 4.26
random 3-SAT is at its hardest. Any assignment that satisfies every clause scores 1;
anything else, or no answer, scores 0. A reply cut off by the token budget scores 0
too, but is discarded, so it counts in its group's baseline and doesn't train.
"""

import random
import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

from loops_cookbook.episodes import Episode, Grading, Sampling, Task
from loops_cookbook.runtime import JsonValue


class Problem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(description="Generates the problem: the same seed, the same clauses.")


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variables: int = Field(default=21, ge=3, description="Variables x1 … xN.")
    clause_ratio: float = Field(default=4.26, gt=0, description="Clauses per variable.")

    @model_validator(mode="after")
    def _has_clauses(self) -> "Options":
        if round(self.clause_ratio * self.variables) < 1:
            raise ValueError("clause_ratio × variables rounds to no clauses")
        return self


INSTRUCTION = (
    "Find values for the boolean variables x1 … x{variables} that make every one of these "
    "{count} clauses true. Each clause is an OR of its literals; ¬ is NOT.\n\n{clauses}\n\n"
    "A satisfying assignment exists. After your reasoning, end your reply with one line "
    "of the form\nTrue: <the numbers of the variables set to true, separated by commas, "
    "or none>"
)
ANSWER = re.compile(r"True:\s*((?:\d+\s*,\s*)*\d+|none)", re.IGNORECASE)

type Clause = tuple[int, int, int]


def clauses(seed: int, variables: int, ratio: float) -> list[Clause]:
    """The problem ``seed`` generates: each clause three distinct variables, each negated
    or not, kept only if a hidden assignment satisfies it."""
    rng = random.Random(seed)
    hidden = [rng.random() < 0.5 for _ in range(variables + 1)]
    kept: list[Clause] = []
    while len(kept) < round(ratio * variables):
        names = rng.sample(range(1, variables + 1), 3)
        clause = tuple(v if rng.random() < 0.5 else -v for v in names)
        if any(hidden[abs(literal)] == (literal > 0) for literal in clause):
            kept.append(clause)  # pyright: ignore[reportArgumentType]
    return kept


def satisfied(problem: list[Clause], true: set[int]) -> int:
    """How many clauses the assignment ``true`` (the variables set to true) satisfies."""
    return sum(any((abs(lit) in true) == (lit > 0) for lit in clause) for clause in problem)


class SatEpisode(Episode[Problem, Options]):
    def __init__(self, task: Task[Problem], options: Options) -> None:
        super().__init__(task, options)
        self.problem = clauses(task.value.seed, options.variables, options.clause_ratio)
        self.answer: set[int] | None = None

    def prompt(self) -> str:
        def literal(value: int) -> str:
            return f"x{value}" if value > 0 else f"¬x{-value}"

        body = "\n".join("(" + " ∨ ".join(map(literal, c)) + ")" for c in self.problem)
        return INSTRUCTION.format(
            variables=self.options.variables, count=len(self.problem), clauses=body
        )

    async def run(self, sample: Sampling) -> dict[str, JsonValue]:
        conversation = sample.model.conversation()
        reply = await conversation.generate(messages=[{"role": "user", "content": self.prompt()}])
        # Cut off by the token budget, not wrong: it scores 0 in its group's baseline but
        # doesn't train, so long reasoning isn't pushed down for its length. A reply the
        # model ended itself, answer or not, trains as it scored.
        cut_off = reply.sample.stop_reason == "length"
        if cut_off:
            conversation.discard(reply)
        # The answer is the reply's last line, exactly; a reply cut off has none.
        lines = [] if cut_off else (reply.content or "").strip().splitlines()
        found = ANSWER.fullmatch(lines[-1].strip()) if lines else None
        if found:
            given = found.group(1)
            self.answer = set() if given.lower() == "none" else {int(n) for n in given.split(",")}
        satisfies = 0 if self.answer is None else satisfied(self.problem, self.answer)
        return {
            "answered": self.answer is not None,
            "satisfied": satisfies / len(self.problem),
            "cut_off": cut_off,
        }

    async def grade(self, grading: Grading) -> float:
        if self.answer is None:
            return 0.0
        return float(satisfied(self.problem, self.answer) == len(self.problem))
