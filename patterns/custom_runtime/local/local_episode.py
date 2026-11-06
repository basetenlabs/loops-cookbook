"""An episode with a local agent: a separate program the cookbook does not own.

For each attempt the cookbook provisions a sandbox from ``[runtime.config.sandbox]``,
then ``run()`` starts the agent program ``[runtime.config.options] agent`` names as a
supervised process on the training machine, with the attempt's model connection and
the sandbox's usage environment. The agent reaches the model only through the run's
gateway, and works in the sandbox without ever creating or deleting it. However the
program ends, ``run()`` returns, recording how it ended as ``agent_exit`` in the
episode's metadata, so the episode is graded.

``grade()`` runs the verifier program over what the agent left: with ``grade_against =
"files"``, the ``result.json`` it wrote in its working directory, the attempt's own;
with ``"sandbox"``, the answer it left in the sandbox, which the episode proves was
still there when the verifier found none, by asking the sandbox's ``alive()`` after
the read. The verifier decides what a failed agent scores; this one scores a missing
answer 0, and says it was missing. It is pure, so grading is declared safe to retry.
A result it cannot parse, or a verifier that prints no reward, is not a measurement,
and is a permanent grading failure.

The episode is written once, over its sandbox kind, and named for each provider:
``DirectoryAgentEpisode`` on ``local_sandbox`` (``train.toml``) and
``BlaxelAgentEpisode`` on Blaxel (``train_blaxel.toml``). Each task is an
``AnswerTask``, and ``loops check`` refuses a dataset line that is not one, options
that are not ``LocalOptions``, or a sandbox table of another provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from local_sandbox import Directory
from pydantic import BaseModel, ConfigDict, Field

from loops_cookbook.episodes import Episode, Grading, Sampling
from loops_cookbook.model_access import ModelAccessMode
from loops_cookbook.providers.blaxel import Blaxel
from loops_cookbook.runtime import GradeResult, PermanentGradingFailure, ProcessResult

HERE = Path(__file__).resolve().parent
INVALID_MEASUREMENT = 3  # the verifier's exit status for "nothing valid to score"


class AnswerTask(BaseModel):
    """One line of the dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str  # for people reading the dataset; the episode does not use it
    instruction: str
    answer: str


class LocalOptions(BaseModel):
    """``[runtime.config.options]``: the two programs, and what the verifier reads.

    A ``./`` or ``../`` part of a command is a path beside this file.
    """

    model_config = ConfigDict(extra="forbid")

    agent: list[str] = Field(min_length=1)
    verifier: list[str] = Field(min_length=1)
    grade_against: Literal["files", "sandbox"]


def command(argv: list[str]) -> tuple[str, ...]:
    """``argv`` with its relative paths resolved beside this file."""
    return tuple(
        str((HERE / part).resolve()) if part.startswith(("./", "../")) else part for part in argv
    )


def ended(result: ProcessResult) -> str:
    """How a program ended, as a label: ``status_0``, ``status_1``, ``signal_9``."""
    if result.signal is not None:
        return f"signal_{result.signal}"
    return f"status_{result.exit_code}"


def tail(result: ProcessResult) -> str:
    lines = result.stderr_tail.strip().splitlines()
    return " | ".join(lines[-3:]) or "no output"


class LocalAgentEpisode(Episode[AnswerTask, LocalOptions]):
    """One run of the agent program on one task, scored by the verifier program.

    The agent works in its sandbox, whichever answer is verified. It reaches it only
    through the sandbox's environment, so the episode never uses the client.
    """

    model_access = ModelAccessMode.OPENAI_HTTP
    grading_retries = 2

    async def run(self, sample: Sampling) -> dict[str, str]:
        model = sample.model
        sandbox = sample.sandbox
        assert sandbox is not None
        connection = await model.http_connection()
        ran = await sample.processes.run(
            command(self.options.agent),
            env={
                **sandbox.environment,
                "OPENAI_BASE_URL": connection.base_url,
                "OPENAI_API_KEY": connection.api_key,
                "OPENAI_MODEL": connection.model,
                "AGENT_INSTRUCTION": self.task.value.instruction,
            },
        )
        return {"agent_exit": ended(ran)}

    async def grade(self, grading: Grading) -> GradeResult:
        sandbox = grading.execution_sandbox
        assert sandbox is not None
        against = self.options.grade_against
        environment = sandbox.environment if against == "sandbox" else {}
        verified = await grading.processes.run(
            command(self.options.verifier),
            env={
                **environment,
                "VERIFY_AGAINST": against,
                "EXPECTED_ANSWER": self.task.value.answer,
            },
        )
        if verified.exit_code == INVALID_MEASUREMENT:
            raise PermanentGradingFailure(f"nothing valid to score: {tail(verified)}")
        if verified.exit_code != 0 or verified.signal is not None:
            raise RuntimeError(f"the verifier failed: {tail(verified)}")  # retried
        try:
            printed = json.loads(verified.stdout_tail)
            reward, answered = printed["reward"], printed["answered"]
        except (ValueError, KeyError, TypeError) as error:
            raise PermanentGradingFailure(f"the verifier printed no reward: {error}") from None
        if isinstance(reward, bool) or not isinstance(reward, int | float):
            raise PermanentGradingFailure(f"the verifier's reward {reward!r} is not a number")
        if not isinstance(answered, bool):
            raise PermanentGradingFailure(f"the verifier's answered {answered!r} is not a bool")
        # A missed read does not say whether the answer or the sandbox is missing.
        if against == "sandbox" and not answered:
            await sandbox.ensure_alive()
        return GradeResult(reward=float(reward))


class DirectoryAgentEpisode(LocalAgentEpisode):
    """The episode on ``local_sandbox``'s directory sandboxes."""

    execution_kind = Directory


class BlaxelAgentEpisode(LocalAgentEpisode):
    """The episode on Blaxel sandboxes."""

    execution_kind = Blaxel
