"""An Episode whose groups an LLM judge ranks.

Each task is a question to answer in a sentence or two, from ``tasks.jsonl``,
validated as a ``Question``.

- ``run()`` is the agent: it asks the policy once with the cookbook's ``llm()``
  client, which records the exact tokens it samples.
- ``grade()`` gives each rollout a small reward of its own: 1 when the answer stays
  within ``max_words``, else 0. ``SandboxedRankedEpisode`` (``train_blaxel.toml``,
  ``Episode[Question, JudgeOptions] with execution_kind = Blaxel``) keeps the answer in the attempt's
  Blaxel sandbox, and ``grade()`` reads it back from there.
- ``grade_group()`` asks a judge model to rank the group's answers, best first,
  through its own OpenAI-compatible client, set up by ``JudgeOptions`` from
  ``[rollout_runtime.config.options]``: its base URL, its model, and the name of the
  environment variable holding its API key, never the key. It never sees the policy,
  a sandbox or a live episode: only each member's task, transcript, reward and
  metadata. The final reward composes the two: ``rank_weight`` of the rank, the rest
  of ``grade()``'s.

The judge reads each answer from the member's transcript: the reply to its last
trainable generation, exactly as the agent received it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, field_validator

from loops_cookbook.episodes import Episode, Grading, GroupMember, Model, Sampling, Setup, Task
from loops_cookbook.llm import ParsedTurnReply
from loops_cookbook.providers.blaxel import Blaxel
from loops_cookbook.runtime import JsonValue, PermanentGradingFailure

ANSWER = "/tmp/answer.txt"  # where the answer lives in a sandbox


class Question(BaseModel):
    """One line of ``tasks.jsonl``, beside its ``id``."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)


class JudgeOptions(BaseModel):
    """``[rollout_runtime.config.options]``: the judge, and how the two rewards compose."""

    model_config = ConfigDict(extra="forbid")

    judge_base_url: str = Field(min_length=1, description="An OpenAI-compatible base URL.")
    judge_model: str = Field(min_length=1)
    judge_api_key_env: str = Field(
        min_length=1, description="The name of the variable holding the judge's key."
    )
    rank_weight: float = Field(default=0.8, ge=0, le=1)
    max_words: int = Field(default=60, gt=0)

    @field_validator("judge_api_key_env")
    @classmethod
    def _key_is_set(cls, name: str) -> str:
        """Refused at check rather than at every group; only the name is kept."""
        if not os.environ.get(name):
            raise ValueError(f"the judge's API key variable {name} is not set")
        return name


class JudgeReplyError(ValueError):
    """The judge did not reply with a ranking of the group; another try may."""


async def ask(model: Model, prompt: str) -> str:
    """The policy's one answer; one that won't parse is an empty one."""
    conversation = model.conversation()
    reply = await conversation.generate(messages=[{"role": "user", "content": prompt}])
    return (reply.content or "").strip()


def own_reward(answer: str, options: JudgeOptions) -> float:
    """``grade()``'s reward: 1 for an answer within ``max_words``, else 0."""
    return float(0 < len(answer.split()) <= options.max_words)


async def judged(members: Sequence[GroupMember[Question]], options: JudgeOptions) -> list[float]:
    """Each member's final reward: ``rank_weight`` of the judge's rank, the rest its own."""
    api_key = os.environ.get(options.judge_api_key_env)
    if not api_key:
        raise PermanentGradingFailure(
            f"the judge's API key variable {options.judge_api_key_env} is not set"
        )
    answers = [answer_of(member) for member in members]
    order = await rank(options, api_key, members[0].task.value.prompt, answers)
    places = {index: place for place, index in enumerate(order)}
    worst = max(1, len(members) - 1)
    return [
        options.rank_weight * (1 - places[index] / worst)
        + (1 - options.rank_weight) * (member.reward or 0.0)
        for index, member in enumerate(members)
    ]


class RankedEpisode(Episode[Question, JudgeOptions]):
    grading_retries = 2  # grade() is pure, and the judge is safe to ask again

    def __init__(self, task: Task[Question], options: JudgeOptions) -> None:
        super().__init__(task, options)
        self.answer = ""

    async def run(self, sample: Sampling) -> Mapping[str, JsonValue]:
        model = sample.model
        self.answer = await ask(model, self.task.value.prompt)
        return {"words": len(self.answer.split())}

    async def grade(self, grading: Grading) -> float:
        return own_reward(self.answer, self.options)

    @classmethod
    async def grade_group(
        cls, members: Sequence[GroupMember[Question]], options: JudgeOptions
    ) -> Sequence[float]:
        return await judged(members, options)


class SandboxedRankedEpisode(Episode[Question, JudgeOptions]):
    """The same episode, its answer written into the attempt's Blaxel sandbox and read back."""

    execution_kind = Blaxel

    grading_retries = 2

    async def setup(self, setup: Setup) -> None:
        sandbox = setup.sandbox
        assert isinstance(sandbox, Blaxel)
        await sandbox.client()  # attached here, on the setup clock, not the agent's

    async def run(self, sample: Sampling) -> Mapping[str, JsonValue]:
        model = sample.model
        sandbox = sample.sandbox
        assert isinstance(sandbox, Blaxel)
        answer = await ask(model, self.task.value.prompt)
        box = await sandbox.client()
        await box.fs.write(ANSWER, answer)  # the tool call: it lands in the sandbox
        return {"words": len(answer.split())}

    async def grade(self, grading: Grading) -> float:
        sandbox = grading.execution_sandbox
        assert isinstance(sandbox, Blaxel)
        answer = await sandbox.read_file(ANSWER)  # None: a run that wrote none
        return own_reward(answer or "", self.options)

    @classmethod
    async def grade_group(
        cls, members: Sequence[GroupMember[Question]], options: JudgeOptions
    ) -> Sequence[float]:
        return await judged(members, options)


def answer_of(member: GroupMember[Question]) -> str:
    """The reply to the member's last trainable generation, as the agent received it."""
    for sealed in reversed(member.transcript.turns):
        reply = sealed.turn.reply
        if sealed.trainable and isinstance(reply, ParsedTurnReply):
            content = None if reply.parsed is None else reply.parsed.get("content")
            return content.strip() if isinstance(content, str) else ""
    return ""


async def rank(
    options: JudgeOptions, api_key: str, question: str, answers: Sequence[str]
) -> list[int]:
    """The answers' indices, best first, as the judge ranked them."""
    listing = "\n".join(
        f"[{index}] {answer or '(no answer)'}" for index, answer in enumerate(answers)
    )
    prompt = (
        f"Question: {question}\n\nAnswers:\n{listing}\n\n"
        "Rank every answer from best to worst: correct, clear, and suited to the question. "
        'Reply with JSON only, in the form {"ranking": [best index, ..., worst index]}.'
    )
    async with (
        aiohttp.ClientSession() as session,
        session.post(
            f"{options.judge_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": options.judge_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
        ) as response,
    ):
        response.raise_for_status()
        body = await response.json()
    return ranking(body["choices"][0]["message"]["content"] or "", len(answers))


def ranking(reply: str, count: int) -> list[int]:
    """Every index from 0 to ``count - 1`` exactly once, as the judge's JSON reply lists them."""
    text = reply.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        order = json.loads(text)["ranking"]
    except (ValueError, KeyError, TypeError) as error:
        raise JudgeReplyError(f"the judge's reply is not a ranking: {reply[:200]!r}") from error
    if not isinstance(order, list) or sorted(order) != list(range(count)):
        raise JudgeReplyError(f"the judge ranked {order!r}, not each of {count} answers once")
    return [int(index) for index in order]
