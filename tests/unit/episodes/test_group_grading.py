"""Group grading through a real ``Run``: an episode class whose ``grade_group()`` scores a group.

The trainer, sampler and renderer are the deterministic baseline stand-ins; the
driver, pipeline, capacity broker and gateway are the run's own. Each judge here is
scripted by the test, and every hostile case asserts what the run did about it.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import ClassVar

import pytest
from pydantic import BaseModel, ConfigDict
from support.offline_run import train_offline

from loops_cookbook.episodes import Episode, Grading, Sampling
from loops_cookbook.llm.transcript import TurnDisposition
from loops_cookbook.runtime import GradeResult, PermanentGradingFailure

from . import runs
from .runs import TASKS


class JudgeOutage(Exception):
    """The judge's endpoint is down."""


def ranks(members, options):
    """A judge with an opinion: the members in order, 0 to n - 1."""
    return [float(index) for index in range(len(members))]


class JudgeOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judge: str = "scripted"


class Judged(Episode[None, JudgeOptions]):
    """No grade(): the group grader gives every reward."""

    judge = staticmethod(ranks)
    calls: ClassVar[list[tuple[list, JudgeOptions]]] = []

    def __init__(self, task, options):
        super().__init__(task, options)
        self.answer = None

    async def run(self, sample: Sampling):
        model = sample.model
        reply = (
            await model.llm()
            .conversation()
            .generate(messages=[{"role": "user", "content": self.task.data["instruction"]}])
        )
        self.answer = (reply.content or "").strip()
        return {"answer": self.answer}

    @classmethod
    async def grade_group(cls, members, options):
        cls.calls.append((list(members), options))
        result = cls.judge(members, options)
        return await result if asyncio.iscoroutine(result) else result


class AnswerTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str
    answer: str


class Meddling(Episode[AnswerTask, JudgeOptions]):
    """A group grader that changes what it is handed, to see whether anyone else sees it."""

    seen: ClassVar[list[tuple[str, str]]] = []

    async def run(self, sample: Sampling):
        model = sample.model
        await model.llm().conversation().generate(messages=[{"role": "user", "content": "x"}])

    @classmethod
    async def grade_group(cls, members, options):
        cls.seen.append((options.judge, members[0].task.value.answer))
        options.judge = "changed"
        for member in members:
            member.task.value.answer = "changed"
        return ranks(members, options)


class Tagging(Judged):
    """A judge that writes into the nested metadata it is handed."""

    calls: ClassVar[list[tuple[list, JudgeOptions]]] = []

    async def run(self, sample: Sampling):
        await super().run(sample)
        return {"answer": self.answer, "tags": ["episode"], "detail": {"by": "run"}}

    @staticmethod
    def judge(members, options):
        for member in members:
            member.metadata["tags"].append("judge")
            member.metadata["detail"]["by"] = "judge"
        return ranks(members, options)


class Composed(Judged):
    """grade() gives a per-rollout reward, and grade_group() composes it with a rank."""

    calls: ClassVar[list[tuple[list, JudgeOptions]]] = []

    async def grade(self, grading: Grading):
        return GradeResult(reward=float(self.answer == self.task.data["answer"]), metadata={})

    @staticmethod
    def judge(members, options):
        return [member.reward + 10 * index for index, member in enumerate(members)]


class Ungraded(Judged):
    """grade() gives no reward: the group grader gives it."""

    calls: ClassVar[list[tuple[list, JudgeOptions]]] = []

    async def grade(self, grading: Grading):
        return None


class NoRewardWithoutAGroupGrader(Episode[None, None]):
    async def run(self, sample: Sampling):
        model = sample.model
        await model.llm().conversation().generate(messages=[{"role": "user", "content": "x"}])

    async def grade(self, grading: Grading):
        return None


@pytest.fixture(autouse=True)
def fresh_calls():
    for episode in (Judged, Composed, Ungraded, Tagging):
        episode.calls = []
    yield


def settings(config, tmp_path, episode, **overrides):
    return runs.settings(config, tmp_path, f"{__name__}:{episode}", **overrides)


def records(run):
    path = run.out / "rollouts.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# ------------------------------------------------------------------ the capability


@pytest.mark.parametrize("trace", ["off", "samples", "transcripts", "full"])
async def test_without_grade_the_group_grader_gives_every_reward_from_recorded_evidence(
    config, tmp_path, trace
):
    outcome, trainer, run = await train_offline(
        settings(config, tmp_path, "Judged", options={"judge": "the test's"}, trace=trace)
    )
    assert outcome.ok and outcome.steps == 1
    (step,) = trainer.steps
    assert step["rewards"] == [0.0, 1.0, 2.0, 3.0]  # the judge's, final
    (members, options), *_ = Judged.calls
    assert options == JudgeOptions(judge="the test's")  # validated, and typed
    for member in members:
        assert member.task.id in {"b", "c"} and member.task.data["instruction"].startswith("Say")
        assert member.reward is None  # no grade(), so no per-rollout reward
        (turn,) = member.transcript.turns  # recorded whatever the run's trace
        assert turn.disposition is TurnDisposition.DELIVERED and turn.trainable
        assert turn.turn.reply.parsed["content"] == member.metadata["answer"]
    trained = [record for record in records(run) if record["disposition"] == "trained"]
    assert sorted(record["reward"] for record in trained) == [0.0, 1.0, 2.0, 3.0]
    # Written only when the trace keeps transcripts: the grader's recording stays in memory.
    written = {path.stem for path in (run.out / "transcripts").glob("*.json")}
    judged = {record["attempt_id"] for record in trained}
    assert (judged <= written) if trace in {"transcripts", "full"} else not written


async def test_grade_gives_the_group_grader_its_per_rollout_rewards_to_compose(config, tmp_path):
    outcome, trainer, _ = await train_offline(settings(config, tmp_path, "Composed"))
    assert outcome.ok
    (members, _), *_ = Composed.calls
    own = [member.reward for member in members]
    assert set(own) <= {0.0, 1.0}
    assert trainer.steps[0]["rewards"] == sorted(r + 10 * i for i, r in enumerate(own))


async def test_grade_may_return_no_reward_when_the_group_grader_gives_it(config, tmp_path):
    outcome, trainer, _ = await train_offline(settings(config, tmp_path, "Ungraded"))
    assert outcome.ok and trainer.steps[0]["rewards"] == [0.0, 1.0, 2.0, 3.0]
    assert all(member.reward is None for members, _ in Ungraded.calls for member in members)


async def test_without_a_group_grader_a_grade_with_no_reward_is_a_grader_bug(config, tmp_path):
    outcome, trainer, run = await train_offline(
        settings(config, tmp_path, "NoRewardWithoutAGroupGrader", max_consecutive_unusable_groups=2)
    )
    assert outcome.status == "aborted" and not trainer.steps
    assert "2 consecutive unusable groups" in (outcome.reason or "")
    assert {record["cause"] for record in records(run)} == {"grading_failed/permanent_grading"}


async def test_a_judge_that_changes_nested_metadata_leaves_the_trained_rollouts_unchanged(
    config, tmp_path, monkeypatch
):
    from loops_cookbook.rl import run as run_module

    trained = []
    score_batch = run_module.score_batch

    def recording(groups):
        trained.extend(rollout for group in groups for rollout in group.members)
        return score_batch(groups)

    monkeypatch.setattr(run_module, "score_batch", recording)
    outcome, _, _ = await train_offline(settings(config, tmp_path, "Tagging"))
    assert outcome.ok and len(trained) == 4
    assert {tuple(rollout.metadata["tags"]) for rollout in trained} == {("episode",)}
    assert {rollout.metadata["detail"]["by"] for rollout in trained} == {"run"}


async def test_a_group_grader_gets_private_copies_of_its_options_and_tasks(config, tmp_path):
    Meddling.seen = []
    outcome, _, _ = await train_offline(
        settings(config, tmp_path, "Meddling", tasks=TASKS[:1], max_steps=2)
    )
    assert outcome.ok
    assert Meddling.seen == [("scripted", "B"), ("scripted", "B")]  # the same task, twice


# --------------------------------------------------------------------- the judge


async def test_a_judge_that_keeps_failing_stops_the_run_as_consecutive_group_grading_failures(
    config, tmp_path, monkeypatch
):
    def down(members, options):
        raise JudgeOutage("connection refused")

    monkeypatch.setattr(Judged, "judge", staticmethod(down))
    outcome, trainer, run = await train_offline(
        settings(
            config,
            tmp_path,
            "Judged",
            max_consecutive_rollout_failures=3,
            max_consecutive_unusable_groups=1,  # a failed judge is not an unusable group
        )
    )
    assert outcome.status == "aborted" and not trainer.steps
    assert (outcome.reason or "").startswith("3 consecutive group-grading failures")
    assert len(Judged.calls) == 3  # never retried: the episode is not retry-safe
    failed = [record for record in records(run) if record["reason"] == "grading_failed"]
    assert len(failed) == 12
    assert {record["cause"] for record in failed} == {"grading_failed/judge_outage"}
    assert {record["reward"] for record in failed} == {None}  # no placeholder rewards


async def test_a_judge_that_raises_its_own_cancellation_is_a_judge_failure(
    config, tmp_path, monkeypatch
):
    """A client library leaking a CancelledError fails its group; the run goes on."""

    async def leaks(members, options):
        raise asyncio.CancelledError  # nothing cancelled it

    monkeypatch.setattr(Judged, "judge", staticmethod(leaks))
    outcome, trainer, run = await train_offline(
        settings(config, tmp_path, "Judged", max_consecutive_rollout_failures=2)
    )
    assert outcome.status == "aborted" and not trainer.steps
    assert (outcome.reason or "").startswith("2 consecutive group-grading failures")
    failed = [record for record in records(run) if record["reason"] == "grading_failed"]
    assert len(failed) == 8
    assert {record["cause"] for record in failed} == {"grading_failed/cancelled"}


async def test_a_judged_group_resets_the_group_grading_count_and_failed_members_are_counted(
    config, tmp_path, monkeypatch
):
    def flaky(members, options):
        if len(Judged.calls) % 2:
            raise JudgeOutage("every other group")
        return ranks(members, options)

    monkeypatch.setattr(Judged, "judge", staticmethod(flaky))
    outcome, _, run = await train_offline(
        settings(config, tmp_path, "Judged", max_consecutive_rollout_failures=2, max_steps=3)
    )
    assert outcome.ok and outcome.steps == 3
    assert len(Judged.calls) == 6  # failed, judged, failed, judged, failed, judged
    *_, step = [json.loads(line) for line in (run.out / "steps.jsonl").read_text().splitlines()]
    assert step["discard/groups/grading_failed"] == 3
    assert step["discard/rollouts/grading_failed/judge_outage"] == 12  # each member, once


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(PermanentGradingFailure("this group cannot be judged"), id="raises-permanent"),
        pytest.param([1.0], id="too-few"),
        pytest.param([0.0, 1.0, 2.0, 3.0, 4.0], id="too-many"),
        pytest.param([0.0, math.nan, 1.0, 2.0], id="nan"),
        pytest.param([0.0, math.inf, 1.0, 2.0], id="infinite"),
        pytest.param([0.0, True, 1.0, 2.0], id="bool"),
        pytest.param(["0", "1", "2", "3"], id="strings"),
        pytest.param("0123", id="a-string"),
        pytest.param(None, id="none"),
    ],
)
async def test_a_permanent_judge_failure_is_never_retried(config, tmp_path, monkeypatch, result):
    """Raised, or a result that is not one finite reward per member: the same every try."""

    def judge(members, options):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(Judged, "judge", staticmethod(judge))
    monkeypatch.setattr(Judged, "grading_retries", 3)
    outcome, _, run = await train_offline(
        settings(
            config,
            tmp_path,
            "Judged",
            max_consecutive_rollout_failures=1,
        )
    )
    assert outcome.status == "aborted"
    assert (outcome.reason or "").startswith("1 consecutive group-grading failures")
    assert len(Judged.calls) == 1
    causes = {record["cause"] for record in records(run) if record["reason"] == "grading_failed"}
    assert causes == {"grading_failed/permanent_grading"}


async def test_a_retry_safe_judge_is_tried_again_with_backoff(config, tmp_path, monkeypatch):
    def once(members, options):
        if len(Judged.calls) == 1:
            raise JudgeOutage("a blip")
        return ranks(members, options)

    monkeypatch.setattr(Judged, "judge", staticmethod(once))
    monkeypatch.setattr(Judged, "grading_retries", 1)
    outcome, trainer, run = await train_offline(
        settings(
            config,
            tmp_path,
            "Judged",
            max_consecutive_rollout_failures=1,
        )
    )
    assert outcome.ok and trainer.steps[0]["rewards"] == [0.0, 1.0, 2.0, 3.0]
    assert len(Judged.calls) == 2
    first, second = (judged for judged, _ in Judged.calls)
    assert [m.metadata for m in first] == [m.metadata for m in second]  # the same group
    (step,) = [json.loads(line) for line in (run.out / "steps.jsonl").read_text().splitlines()]
    assert step["retried/groups/grading_failed/judge_outage"] == 1.0  # a group's, not a rollout's
    assert "retried/attempts/grading_failed/judge_outage" not in step


async def test_a_group_that_goes_stale_during_the_backoff_is_dropped_before_the_next_try(
    config, tmp_path, monkeypatch
):
    trainers = []

    def fails_while_the_policy_moves(members, options):
        if len(Judged.calls) == 1:
            trainers[0].policy_version = 1  # every member is now off policy
            raise JudgeOutage("a blip")
        return ranks(members, options)

    monkeypatch.setattr(Judged, "judge", staticmethod(fails_while_the_policy_moves))
    monkeypatch.setattr(Judged, "grading_retries", 1)
    from baseline_model import BaselineTrainer

    trainer = BaselineTrainer()
    trainers.append(trainer)
    run_settings = settings(config, tmp_path, "Judged")
    outcome, _, run = await train_offline(run_settings, trainer=trainer)
    assert outcome.ok
    assert len(Judged.calls) == 2  # the stale group once, never again; then the next group
    first = [record for record in records(run) if record["group_id"].startswith("g0000000-")]
    assert {record["reason"] for record in first} == {"stale_before_training"}
    assert {record["reward"] for record in first} == {None}


async def test_a_slow_judge_fails_at_the_grading_deadline(config, tmp_path, monkeypatch):
    async def slow(members, options):
        await asyncio.sleep(30)
        return ranks(members, options)

    monkeypatch.setattr(Judged, "judge", staticmethod(slow))
    outcome, _, run = await train_offline(
        settings(
            config,
            tmp_path,
            "Judged",
            max_consecutive_rollout_failures=1,
            **{"rollout.grading.timeout_seconds": 0.3},
        )
    )
    assert outcome.status == "aborted"
    assert "grade_group() exceeded its deadline" in (outcome.reason or "")
    causes = {record["cause"] for record in records(run) if record["reason"] == "grading_failed"}
    assert causes == {"grading_failed/hook_deadline"}


async def test_a_judge_that_ignores_its_deadline_stops_the_run_unsettled(
    config, tmp_path, monkeypatch, caplog
):
    gives_up = asyncio.Event()

    async def stubborn(members, options):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await gives_up.wait()  # past its cancellation grace, until the test ends
        return ranks(members, options)

    monkeypatch.setattr(Judged, "judge", staticmethod(stubborn))
    try:
        outcome, _, _ = await train_offline(
            settings(
                config,
                tmp_path,
                "Judged",
                shutdown_drain_seconds=1,
                **{
                    "rollout.grading.timeout_seconds": 0.2,
                    "rollout.cleanup.cancellation_grace_seconds": 0.2,
                },
            )
        )
    finally:
        gives_up.set()
    assert outcome.status == "aborted"
    assert "grade_group() ignored cancellation" in (outcome.reason or "")
    # Still running, so it keeps its slot, and the run does not end clean.
    assert "Rollout runtime hook grade_group:g0000000-" in caplog.text
    assert "ignored cancellation and still holds its permit" in caplog.text


# ------------------------------------------------------------------ the stage order


async def test_a_group_shrunk_by_infrastructure_failures_is_unusable_and_never_judged(
    config, tmp_path, monkeypatch
):
    runs = 0
    original = Judged.run  # Composed's too

    async def mostly_fails(self, model):
        nonlocal runs
        runs += 1
        if runs % 4:
            raise ConnectionError("the environment went away")
        return await original(self, model)

    monkeypatch.setattr(Judged, "run", mostly_fails)
    outcome, trainer, run = await train_offline(
        settings(
            config,
            tmp_path,
            "Composed",
            max_consecutive_unusable_groups=3,
            max_consecutive_rollout_failures=0,
        )
    )
    assert outcome.status == "aborted" and not trainer.steps
    assert "3 consecutive unusable groups" in (outcome.reason or "")
    assert "3 had too few usable rollouts" in (outcome.reason or "")
    assert Composed.calls == []  # never paid for a group about to be dropped
    small = [record for record in records(run) if record["reason"] == "group_too_small"]
    # grade() gave each survivor a reward, but a member dropped before judging has none.
    assert len(small) == 3 and {record["reward"] for record in small} == {None}


async def test_a_judged_group_without_variance_is_unusable_like_any_other(
    config, tmp_path, monkeypatch
):
    monkeypatch.setattr(Judged, "judge", staticmethod(lambda members, options: [1.0] * 4))
    outcome, trainer, run = await train_offline(
        settings(config, tmp_path, "Judged", max_consecutive_unusable_groups=2)
    )
    assert outcome.status == "aborted" and not trainer.steps
    assert "2 gave every rollout one reward (1 in 2)" in (outcome.reason or "")
    flat = [record for record in records(run) if record["reason"] == "no_variance"]
    assert len(flat) == 8 and {record["reward"] for record in flat} == {1.0}  # judged


async def test_a_member_gone_stale_before_the_judge_is_dropped_and_its_group_never_judged(
    config, tmp_path, monkeypatch
):
    trainers = []

    async def moves_the_policy(self, rollout):
        trainer = trainers[0]
        if trainer.policy_version == 0:
            trainer.policy_version = 1  # the policy moved on while this group was grading
        # No reward: the group grader gives it.

    monkeypatch.setattr(Ungraded, "grade", moves_the_policy)
    from baseline_model import BaselineTrainer

    trainer = BaselineTrainer()
    trainers.append(trainer)
    run_settings = settings(config, tmp_path, "Ungraded")
    outcome, _, run = await train_offline(run_settings, trainer=trainer)
    assert outcome.ok
    first = [record for record in records(run) if record["group_id"].startswith("g0000000-")]
    reasons = {record["reason"] for record in first}
    assert "stale_before_training" in reasons  # graded, then stale at the group stage
    assert reasons <= {"stale_before_training", "stale_before_grading", "group_too_small"}
    assert {record["reward"] for record in first} == {None}
    assert len(Ungraded.calls) == 1  # only the next group, which trained
    trained = [record for record in records(run) if record["disposition"] == "trained"]
    assert trained and not any(record in first for record in trained)


# ------------------------------------------------------------------------- check


def write_config(tmp_path, episode):
    return runs.write_config(tmp_path, f"{__name__}:{episode}")


@pytest.mark.parametrize(
    ("episode", "phases"),
    [
        ("Judged", "sampling, verification"),  # no grade(): it never takes a slot
        ("Composed", "sampling"),  # grade() takes the cookbook's slot
    ],
)
def test_check_reports_the_group_grader_an_episode_class_declares(
    loops_check, tmp_path, episode, phases
):
    code, out, err = loops_check(write_config(tmp_path, episode))
    assert code == 0, err
    assert (
        f"runtime-acquired phases {phases}; grading retries 0 ({episode}.grading_retries); "
        "group grading true" in out
    )


def test_check_reports_no_group_grader_for_an_episode_without_one(loops_check, tmp_path):
    code, out, _ = loops_check(write_config(tmp_path, "NoRewardWithoutAGroupGrader"))
    assert code == 0 and "group grading false" in out
