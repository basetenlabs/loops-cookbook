"""Episodes through a real ``Run``: training, the HTTP gateway's credentials, and ``check``.

The trainer, sampler and renderer are the deterministic baseline stand-ins; the
driver, pipeline, gateway and resource registry are the run's own.
"""

from __future__ import annotations

import asyncio
import json
import os
from functools import partial

import pytest
from pydantic import BaseModel, ConfigDict, Field
from support.offline_run import train_offline
from support.resources import (
    PROVIDERS,
    Fake,
    FakeSandboxProvider,
)
from support.resume import Checkpointing, resumed_offline

from loops_cookbook.episodes import Episode, Grading, Sampling
from loops_cookbook.episodes.tasks import task_digest
from loops_cookbook.model_access import ModelAccessMode
from loops_cookbook.providers.blaxel import Blaxel

from . import runs
from .runs import FACTORY, TASKS


class AnswerEpisode(Episode[None, None]):
    """The spec's direct example: one question, one answer, scored against the task's."""

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

    async def grade(self, grading: Grading):
        return float(self.answer == self.task.data["answer"])


AFTER_SEAL: list[object] = []


class HttpEpisode(Episode[None, None]):
    """Keeps its connection and its Model past run(), and tries both from grade()."""

    model_access = ModelAccessMode.OPENAI_HTTP

    def __init__(self, task, options):
        super().__init__(task, options)
        self.answer = None

    async def run(self, sample: Sampling):
        model = sample.model
        from openai import AsyncOpenAI

        self.model, self.connection = model, await model.http_connection()
        key = self.connection.api_key.get_secret_value()
        async with AsyncOpenAI(base_url=self.connection.base_url, api_key=key) as client:
            completion = await client.chat.completions.create(
                model=self.connection.model,
                messages=[{"role": "user", "content": self.task.data["instruction"]}],
            )
        self.answer = (completion.choices[0].message.content or "").strip()

    async def grade(self, grading: Grading):
        import aiohttp

        key = self.connection.api_key.get_secret_value()
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                self.connection.base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": self.connection.model,
                    "messages": [{"role": "user", "content": "x"}],
                },
            ) as response,
        ):
            AFTER_SEAL.append(response.status)
        try:
            await self.model.http_connection()
        except Exception as error:  # noqa: BLE001 - the refusal is what is recorded
            AFTER_SEAL.append(str(error))
        return float(self.answer == self.task.data["answer"])


def settings(config, tmp_path, episode, **overrides):
    return runs.settings(config, tmp_path, f"{__name__}:{episode}", **overrides)


async def test_an_episode_class_trains_through_the_shared_pipeline_in_its_own_sandboxes(
    config, tmp_path
):
    provider = FakeSandboxProvider()
    run_settings = settings(config, tmp_path, "SandboxedEpisode")
    run_settings = run_settings.model_copy(
        update={
            "rollout_runtime": run_settings.rollout_runtime.model_copy(
                update={
                    "config": {
                        **run_settings.rollout_runtime.config,
                        "sandbox": {"provider": "fake"},
                    }
                }
            )
        }
    )
    outcome, trainer, run = await train_offline(run_settings, adapters={"fake": provider})
    assert outcome.ok and outcome.steps == 1
    (step,) = trainer.steps
    assert len(step["rewards"]) == 4 and set(step["rewards"]) <= {0.0, 1.0}
    records = [json.loads(line) for line in (run.out / "rollouts.jsonl").read_text().splitlines()]
    creates = [operation for operation, _ in provider.calls if operation == "create"]
    assert len(creates) == len(records) == 4  # one sandbox per attempt
    assert provider.store == {}  # every one released
    assert [client.closed for client in provider.clients] == [1] * 4  # and its client closed
    (recorded,) = [json.loads(line) for line in (run.out / "steps.jsonl").read_text().splitlines()]
    assert recorded["runtime/episodes/agent_timed_out/mean"] == 0.0


RUBRIC = ["exact"]  # the options model's default, as its code says when it validates


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    rubric: str = Field(default_factory=lambda: RUBRIC[0])


class RubricEpisode(AnswerEpisode, Episode[None, Rubric]):
    pass


async def test_the_validated_options_are_run_evidence_and_a_resume_compares_them(
    config, tmp_path, caplog
):
    RUBRIC[0] = "exact"
    options = {"url": "http://judge"}
    run_settings = settings(config, tmp_path, "RubricEpisode")
    runtime = {**run_settings.rollout_runtime.config, "options": options}
    first_settings = settings(
        config, tmp_path, "RubricEpisode", **{"rollout_runtime.config": runtime}
    )
    outcome, _, run = await train_offline(first_settings, trainer=Checkpointing())
    assert outcome.ok
    ran = {"url": "http://judge", "rubric": "exact"}  # the default ran
    recorded = {
        "options": task_digest(ran),
        "sandbox_lifetime": task_digest({"sandbox_lifetime": "attempt"}),
    }
    assert json.loads((run.out / "runtime_evidence_digests.json").read_text()) == recorded
    manifest = run.out / "checkpoints" / "step-0.json"
    assert json.loads(manifest.read_text())["contract"]["runtime_evidence_digests"] == recorded

    # The same [rollout_runtime.config], and a default that now says otherwise.
    RUBRIC[0] = "strict"
    resumed_trainer = Checkpointing()
    resumed_trainer.policy_version = 1
    with caplog.at_level("WARNING", logger="loops_cookbook.rl.run"):
        outcome, _, _ = await train_offline(
            settings(
                config,
                tmp_path,
                "RubricEpisode",
                max_steps=2,
                **{"rollout_runtime.config": runtime, "training.resume_from": str(manifest)},
            ),
            trainer=resumed_trainer,
        )
    RUBRIC[0] = "exact"
    assert outcome.ok
    assert "Resuming with changed settings (runtime_evidence_digests)" in caplog.text


async def test_a_resume_from_a_record_made_before_runtime_evidence_reports_no_change(
    config, tmp_path, caplog
):
    def forget(manifest):
        del manifest["contract"]["runtime_evidence_digests"]  # as a run before it existed

    with caplog.at_level("WARNING", logger="loops_cookbook.rl.run"):
        outcome, _, _ = await resumed_offline(
            partial(settings, config, tmp_path, "AnswerEpisode"), forget
        )
    assert outcome.ok
    assert "changed settings" not in caplog.text


class TokenFromEnvironment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    token: str = Field(default_factory=lambda: os.environ["JUDGE_TOKEN"])  # not a SecretStr


class TokenEpisode(AnswerEpisode, Episode[None, TokenFromEnvironment]):
    pass


async def test_an_options_default_read_from_the_environment_never_reaches_the_run_record(
    config, tmp_path, monkeypatch
):
    monkeypatch.setenv("JUDGE_TOKEN", "tok-5e3f9a")
    run_settings = settings(config, tmp_path, "TokenEpisode")
    runtime = {**run_settings.rollout_runtime.config, "options": {"url": "http://judge"}}
    outcome, _, run = await train_offline(
        settings(config, tmp_path, "TokenEpisode", **{"rollout_runtime.config": runtime}),
        trainer=Checkpointing(),
    )
    assert outcome.ok
    written = [path for path in run.out.rglob("*") if path.is_file()]
    assert run.out / "runtime_evidence_digests.json" in written
    assert run.out / "checkpoints" / "step-0.json" in written
    assert [path for path in written if b"tok-5e3f9a" in path.read_bytes()] == []


async def test_an_http_credential_and_the_model_are_refused_once_the_attempt_is_sealed(
    config, tmp_path
):
    AFTER_SEAL.clear()
    outcome, _, _ = await train_offline(settings(config, tmp_path, "HttpEpisode"))
    assert outcome.ok and outcome.steps == 1
    assert AFTER_SEAL and set(AFTER_SEAL) == {401, "model access for this attempt is sealed"}


def write_config(tmp_path, *tables, episode="AnswerEpisode", tasks=TASKS):
    return runs.write_config(tmp_path, f"{__name__}:{episode}", *tables, tasks=tasks)


def test_check_builds_the_episode_runtime_and_reports_its_capabilities(loops_check, tmp_path):
    code, out, err = loops_check(write_config(tmp_path))
    assert code == 0, err
    assert f"Rollout runtime: {FACTORY}; tasks: 2" in out
    assert "model access direct (AnswerEpisode.model_access)" in out
    assert (
        "runtime-acquired phases sampling; grading retries 0 (AnswerEpisode.grading_retries)" in out
    )


@pytest.mark.parametrize(
    ("clock", "shown"),
    [
        (None, "Rollout clock: 29s (derived by the rollout runtime)"),  # 20 + 2 + 5 + 2
        (30, "Rollout clock: 30s ([rollout] rollout_timeout_seconds)"),
    ],
    ids=["derived", "set"],
)
def test_check_prints_the_rollout_clock_the_run_keeps_and_where_it_came_from(
    loops_check, tmp_path, clock, shown
):
    clock = {"rollout": {"rollout_timeout_seconds": clock}}
    code, out, err = loops_check(write_config(tmp_path, clock))
    assert code == 0, err
    assert shown in out


@pytest.mark.parametrize(
    "agent_timeout", [22, 25.9], ids=["below the minimum", "the minimum without the group exit"]
)
def test_check_refuses_a_rollout_clock_below_what_the_agent_deadline_needs(
    loops_check, tmp_path, agent_timeout
):
    deadline = {"rollout_runtime": {"config": {"agent_timeout_seconds": agent_timeout}}}
    code, _, err = loops_check(write_config(tmp_path, deadline))  # the rollout clock: 30
    assert code == 2
    assert "[rollout] rollout_timeout_seconds (30s) is below" in err


class Graded(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str
    answer: str


class Judge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str


class DeclaredEpisode(AnswerEpisode, Episode[Graded, Judge]):
    pass


class SandboxedEpisode(AnswerEpisode, Episode[None, None]):
    """Reaches its sandbox's client before it answers."""

    execution_kind = Fake

    async def run(self, sample: Sampling):
        sandbox = sample.sandbox
        assert isinstance(sandbox, Fake)
        assert sandbox is not None
        await sandbox.client()
        return await super().run(sample)


OPTIONS = {"rollout_runtime": {"config": {"options": {"url": "http://judge"}}}}
FAKE = {"adapter": "support.resources:create_fake", "instance": "check"}
SANDBOX = {
    "rollout_runtime": {"config": {"sandbox": {"provider": "fake"}}},
    "providers": {"fake": FAKE},
}


def test_check_builds_an_episode_that_declares_its_task_and_options_models(loops_check, tmp_path):
    code, _, err = loops_check(write_config(tmp_path, OPTIONS, episode="DeclaredEpisode"))
    assert code == 0, err


def test_check_refuses_a_task_line_the_task_model_refuses_naming_file_and_line(
    loops_check, tmp_path
):
    tasks = [*TASKS, {"id": "d", "instruction": "Say D"}]
    code, _, err = loops_check(
        write_config(tmp_path, OPTIONS, tasks=tasks, episode="DeclaredEpisode")
    )
    assert code == 2
    assert "tasks.jsonl:3: the task is not a valid Graded" in err


def test_check_refuses_a_runtime_configuration_error_with_its_message(loops_check, tmp_path):
    tables = {"rollout_runtime": {"config": {"options": {"url": 3}}}}
    code, _, err = loops_check(write_config(tmp_path, tables, episode="DeclaredEpisode"))
    assert code == 2
    assert "is not a valid Judge" in err and "Traceback" not in err


class BlaxelEpisode(AnswerEpisode, Episode[None, None]):
    execution_kind = Blaxel


BLAXEL_SETTINGS = {
    "region": "us-pdx-1",
    **{
        f"{operation}_timeout_seconds": 60
        for operation in (
            "create",
            "reconcile",
            "image_build",
            "release",
            "confirm_absent",
            "list",
            "attach",
        )
    },
}


@pytest.mark.parametrize(
    ("name", "adapter", "problem"),
    [
        (
            "blaxel",
            FAKE,
            (
                "the [providers.blaxel] adapter provides Fake sandboxes, and episode "
                f"'{__name__}:BlaxelEpisode' is written for Blaxel"
            ),
        ),
        (
            "cloud",
            {"adapter": "loops_cookbook.providers.blaxel:create_adapter", **BLAXEL_SETTINGS},
            None,
        ),
    ],
    ids=["another adapter named blaxel", "blaxel's adapter named otherwise"],
)
def test_check_matches_a_sandbox_to_its_adapters_kind_not_to_the_providers_name(
    loops_check, tmp_path, monkeypatch, name, adapter, problem
):
    pytest.importorskip("blaxel")
    monkeypatch.setitem(PROVIDERS, "check", FakeSandboxProvider())
    tables = {
        "rollout_runtime": {
            "config": {
                "sandbox": {"provider": name, "image": "img:1", "memory_mb": 2048, "ttl": "1h"}
            }
        },
        "providers": {name: adapter},
    }
    code, _, err = loops_check(write_config(tmp_path, tables, episode="BlaxelEpisode"))
    if problem is None:
        assert code == 0, err
    else:
        assert code == 2 and problem in err


def test_check_builds_the_sandbox_a_providers_adapter_provides_for_an_episode_written_for_it(
    loops_check, tmp_path, monkeypatch
):
    monkeypatch.setitem(PROVIDERS, "check", FakeSandboxProvider())
    code, _, err = loops_check(write_config(tmp_path, SANDBOX, episode="SandboxedEpisode"))
    assert code == 0, err


class LingeringEpisode(AnswerEpisode):
    """Answers, then takes half a second to stop once its agent deadline cuts it."""

    async def run(self, sample: Sampling):
        result = await super().run(sample)
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.sleep(0.5)
        return result


@pytest.mark.parametrize(
    ("clock", "cause"),
    [(None, "agent_timeout/rollout_clock"), (10, None)],
    ids=["derived: 0.4s is up first", "set: 10s outlasts the linger"],
)
async def test_an_episode_run_keeps_the_derived_rollout_clock_unless_one_is_set(
    config, tmp_path, monkeypatch, clock, cause
):
    # Only so the derived clock is short enough to watch expire.
    monkeypatch.setattr("loops_cookbook.episodes.runtime.GROUP_EXIT_SECONDS", 0.0)
    run_settings = settings(
        config,
        tmp_path,
        "LingeringEpisode",
        **{
            "rollout_runtime.config": {
                "episode": f"{__name__}:LingeringEpisode",
                "agent_timeout_seconds": 0.2,
            },
            "rollout.rollout_timeout_seconds": clock,
            "rollout.cleanup.cancellation_grace_seconds": 0.1,
            "rollout.cleanup.process_kill_grace_seconds": 0.1,
            "max_consecutive_rollout_failures": 4,
        },
    )
    outcome, _, run = await train_offline(run_settings)
    records = [json.loads(line) for line in (run.out / "rollouts.jsonl").read_text().splitlines()]
    assert outcome.ok is (cause is None)
    assert {record.get("cause") for record in records} == {cause}


async def test_a_checkpoint_before_lifetime_evidence_warns_but_resumes(config, tmp_path, caplog):
    def forget(manifest):
        del manifest["contract"]["runtime_evidence_digests"]["sandbox_lifetime"]

    with caplog.at_level("WARNING", logger="loops_cookbook.rl.run"):
        outcome, _, _ = await resumed_offline(
            partial(settings, config, tmp_path, "AnswerEpisode"), forget
        )
    assert outcome.ok
    assert "Resuming with changed settings (runtime_evidence_digests)" in caplog.text
