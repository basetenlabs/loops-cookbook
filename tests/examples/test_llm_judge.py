"""The LLM-judge example: an Episode whose groups a judge model ranks through its own client.

The judge is a scripted OpenAI-compatible server on loopback, so the whole path runs
offline: the episode's own HTTP client, its settings from ``[runtime.config.options]``,
and the group stage of a real ``Run``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import judge_episode
import pytest
from aiohttp import web
from support.offline_run import offline_config, train_offline

from loops_cookbook.resources import SandboxGone
from loops_cookbook.rl import cli
from loops_cookbook.runtime import PermanentGradingFailure

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "llm_judge"
CONFIG = EXAMPLE / "train.toml"
SHIPPED = cli.load_training_config(CONFIG).runtime.config


def options(**changes):
    """The shipped ``[runtime.config.options]`` with ``changes``, as a ``--runtime.config`` override."""
    return json.dumps({"options": {**SHIPPED["options"], **changes}})


def test_the_shipped_judge_names_its_key_by_variable_never_by_value(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "check-only")
    options = judge_episode.JudgeOptions.model_validate(SHIPPED["options"])
    assert options.judge_api_key_env == "BASETEN_API_KEY"


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"rank_weight": 2}, "rank_weight"),
        ({"max_words": 0}, "max_words"),
        ({"judge_modle": "m"}, "judge_modle"),  # misspelt: refused, never dropped
        ({"judge_base_url": ""}, "judge_base_url"),
    ],
)
def test_check_refuses_a_bad_judge_option(loops_check, changes, problem):
    code, _, err = loops_check(CONFIG, "--runtime.config", options(**changes))
    assert code == 2, err
    assert "is not a valid JudgeOptions" in err and problem in err


def test_check_refuses_a_judge_whose_key_variable_is_not_set(loops_check, monkeypatch):
    monkeypatch.delenv("NO_SUCH_JUDGE_KEY", raising=False)
    changed = options(judge_api_key_env="NO_SUCH_JUDGE_KEY")
    code, _, err = loops_check(CONFIG, "--runtime.config", changed)
    assert code == 2, err
    assert "the judge's API key variable NO_SUCH_JUDGE_KEY is not set" in err


def test_check_accepts_the_shipped_judge_options_and_grades_by_group(loops_check):
    code, out, err = loops_check(CONFIG)
    assert code == 0, err
    assert "group grading true" in out


class Judge:
    """A scripted judge: it ranks the answers it is shown, last first."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def complete(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests.append({"authorization": request.headers["Authorization"], **body})
        prompt = body["messages"][0]["content"]
        count = sum(line.startswith("[") for line in prompt.splitlines())
        ranking = list(reversed(range(count)))
        reply = f"```json\n{json.dumps({'ranking': ranking})}\n```"
        return web.json_response({"choices": [{"message": {"content": reply}}]})


@pytest.fixture
async def judge():
    scripted = Judge()
    app = web.Application()
    app.router.add_post("/v1/chat/completions", scripted.complete)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    _, port = runner.addresses[0]
    scripted.base_url = f"http://127.0.0.1:{port}/v1"
    yield scripted
    await runner.cleanup()


async def test_a_judge_ranks_every_group_of_a_real_run_through_the_episodes_own_client(
    judge, tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_JUDGE_KEY", "judge-secret")
    shipped = cli.load_training_config(EXAMPLE / "train.toml")
    options = {
        **shipped.runtime.config["options"],
        "judge_base_url": judge.base_url,
        "judge_api_key_env": "TEST_JUDGE_KEY",
    }
    config = offline_config(
        EXAMPLE / "train.toml",
        tmp_path,
        {
            "runtime.config": {**shipped.runtime.config, "options": options},
            "training.max_steps": 1,
        },
    )
    outcome, trainer, _ = await train_offline(config)
    assert outcome.ok and outcome.steps == 1
    size, groups = shipped.training.group_size, shipped.training.groups_per_step
    assert len(judge.requests) == groups  # one call per group, never one per rollout
    for request in judge.requests:
        assert request["authorization"] == "Bearer judge-secret"
        assert request["model"] == shipped.runtime.config["options"]["judge_model"]
        assert "Question: In one or two sentences" in request["messages"][0]["content"]
    # The baseline policy answers one word, within max_words: grade() gives 1 to all,
    # and the judge's reversed ranking puts member i in place size - 1 - i.
    expected = sorted([0.8 * index / (size - 1) + 0.2 for index in range(size)] * groups)
    assert trainer.steps[0]["rewards"] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("reply", "problem"),
    [
        ("I prefer the second one", "not a ranking"),
        ('{"order": [0, 1]}', "not a ranking"),
        ('{"ranking": [0, 0]}', "not each of 2 answers once"),
        ('{"ranking": [1]}', "not each of 2 answers once"),
    ],
)
def test_a_judge_reply_that_does_not_rank_each_answer_once_can_be_asked_again(reply, problem):
    with pytest.raises(judge_episode.JudgeReplyError, match=problem):
        judge_episode.ranking(reply, 2)


async def test_a_judge_whose_key_is_unset_after_check_fails_permanently(monkeypatch):
    monkeypatch.setenv("SOON_UNSET_JUDGE_KEY", "set at check")
    options = judge_episode.JudgeOptions(
        judge_base_url="http://x", judge_model="m", judge_api_key_env="SOON_UNSET_JUDGE_KEY"
    )
    monkeypatch.delenv("SOON_UNSET_JUDGE_KEY")
    with pytest.raises(PermanentGradingFailure, match="SOON_UNSET_JUDGE_KEY is not set"):
        await judge_episode.RankedEpisode.grade_group([], options)


def sandboxed(on, *, max_words=3):
    options = judge_episode.JudgeOptions(
        judge_base_url="http://x", judge_model="m", judge_api_key_env="K", max_words=max_words
    )
    task = NS(id="t", value=judge_episode.Question(prompt="Why?"))
    return judge_episode.SandboxedRankedEpisode(task, options)


def replying(text):
    """A ``Model`` whose one conversation replies ``text``."""

    async def generate(**kwargs):
        return NS(content=text)

    return NS(conversation=lambda: NS(generate=generate))


async def test_on_blaxel_the_answer_lives_in_the_sandbox_and_is_graded_from_it(
    monkeypatch, blaxel, blaxel_sandbox
):
    monkeypatch.setenv("K", "a key")
    on = await blaxel_sandbox()
    episode = sandboxed(on)
    await episode.setup(NS(sandbox=on))
    client = await on.client()
    assert blaxel.attached() == [client]  # setup() attached, once
    assert await episode.run(NS(model=replying(" Because the moon pulls. "), sandbox=on)) == {
        "words": 4
    }
    assert client.files == {judge_episode.ANSWER: "Because the moon pulls."}
    assert await episode.grade(NS(execution_sandbox=on)) == 0.0  # four words, over max_words
    client.files[judge_episode.ANSWER] = "The moon."  # what is graded is what the sandbox holds
    assert await episode.grade(NS(execution_sandbox=on)) == 1.0
    assert blaxel.attached() == [client]  # every hook used the same client


async def test_on_blaxel_a_missing_answer_scores_zero_while_the_sandbox_is_alive(
    monkeypatch, blaxel_sandbox
):
    monkeypatch.setenv("K", "a key")
    on = await blaxel_sandbox()
    episode = sandboxed(on)
    assert await episode.grade(NS(execution_sandbox=on)) == 0.0  # run() wrote nothing


@pytest.mark.parametrize("fails", ["sandbox_gone", "unavailable"])
async def test_on_blaxel_a_failed_read_is_never_a_zero_but_a_grading_failure_to_retry(
    monkeypatch, blaxel, blaxel_sandbox, fails
):
    from blaxel.core.sandbox.types import ResponseError

    monkeypatch.setenv("K", "a key")
    on = await blaxel_sandbox()
    episode = sandboxed(on)
    await episode.setup(NS(sandbox=on))
    if fails == "sandbox_gone":
        blaxel.lose(on.name, "deleted")  # its file reads answer 404 all the same
    else:
        (await on.client()).failing = 503
    with pytest.raises(SandboxGone if fails == "sandbox_gone" else ResponseError):
        await episode.grade(NS(execution_sandbox=on))
