"""Evals during a run: when they start, what they sample, what they leave behind, and
that training never waits for them."""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import json
import logging
import threading

import pytest
from baseten.loops.models import EncodedTextChunk, ImageChunk
from support import fakes
from support.fakes import MemoryTrainer
from support.harness import Renderer

from loops_cookbook.evidence import TaskRef
from loops_cookbook.model_access.api import ModelAccessMode
from loops_cookbook.rl import recipe
from loops_cookbook.rl.run import Run
from loops_cookbook.runtime.api import EpisodeResult, GradeResult, RuntimeCapabilities

TRAIN_TASK = TaskRef("train-0", "sha256:" + "0" * 64)
EVAL_TASKS = [TaskRef("e0", "sha256:" + "1" * 64), TaskRef("e1", "sha256:" + "2" * 64)]


class EpisodeRuntime:
    """One generation per episode, directly or over the run's gateway.

    ``hold`` keeps every sample() waiting until set; ``on_sample`` runs first, to make
    an episode fail or ignore its cancellation.
    """

    def __init__(self, tasks, reward, mode=ModelAccessMode.DIRECT):
        self.tasks, self.reward, self.mode = tasks, reward, mode
        self.capabilities = RuntimeCapabilities(model_access_modes=frozenset({mode}))
        self.hold: asyncio.Event | None = None
        self.on_sample = None
        self.graded: list[str] = []
        self.closed = False
        self.context = None
        self.content: object = "x"  # what each episode asks the model

    async def prepare(self):
        return list(self.tasks)

    async def sample(self, attempt):
        if self.on_sample is not None:
            await self.on_sample(attempt)
        if self.hold is not None:
            await self.hold.wait()
        access = attempt.services.model_access
        if self.mode is ModelAccessMode.OPENAI_HTTP:
            from openai import AsyncOpenAI

            connection = await access.http_connection()
            key = connection.api_key.get_secret_value()
            async with AsyncOpenAI(base_url=connection.base_url, api_key=key) as client:
                await client.chat.completions.create(
                    model=connection.model, messages=[{"role": "user", "content": "x"}]
                )
        else:
            conversation = access.llm().conversation()
            await conversation.generate(messages=[{"role": "user", "content": self.content}])
        return EpisodeResult()

    async def grade(self, attempt, rollout):
        self.graded.append(attempt.spec.task.id)
        return GradeResult(reward=self.reward(attempt.spec.task))

    async def finish(self, attempt, rollout, disposition):
        pass

    async def abort(self, attempt, reason):
        pass

    async def close(self):
        self.closed = True


def build(
    config, *, evals=True, mode=ModelAccessMode.DIRECT, renderer=Renderer, paced=True, **overrides
):
    if not evals:
        overrides["eval.sets"] = {}
    run, trainer, runtimes = build_sets(
        config, sets=("val",), mode=mode, renderer=renderer, paced=paced, **overrides
    )
    return run, trainer, runtimes["train"], runtimes["val"]


class PacedTrainer(MemoryTrainer):
    """Each update waits for the evals before it to end, as a real update outlasts a
    small eval; so every due eval runs, none skipped behind its set's last."""

    run = None

    async def accumulate(self, rows):
        await until(lambda: self.run.evals_running == 0)
        return await super().accumulate(rows)


def build_sets(
    config, *, sets, mode=ModelAccessMode.DIRECT, renderer=Renderer, paced=True, **overrides
):
    """A run with an eval set per name in ``sets``, each its own runtime over the same
    tasks; the rollout runtimes by name, training's as ``train``. ``paced``: each update waits
    for the evals before it."""
    # Training rewards alternate, so every group has variance and is usable.
    flips = itertools.cycle([0.0, 1.0])
    runtimes = {"train": EpisodeRuntime([TRAIN_TASK], reward=lambda task: next(flips), mode=mode)}
    for name in sets:
        runtimes[name] = EpisodeRuntime(
            EVAL_TASKS, reward=lambda task: float(task.id == "e0"), mode=mode
        )
    settings = {
        "max_steps": 5,
        "rollout_runtime.files": {"tasks": "train.jsonl"},
        "eval.every_steps": 2,
        "eval.sets": {
            name: {"rollout_runtime": {"files": {"tasks": f"{name}.jsonl"}}} for name in sets
        },
        **overrides,
    }
    trainer = PacedTrainer() if paced else MemoryTrainer()
    run = recipe.assemble(
        config(**settings),
        trainer=trainer,
        runtime_factory=lambda _settings, *, context: _with(
            runtimes[context.files.path("tasks").stem], context
        ),
        renderer_factory=lambda _config: renderer(),
    )
    trainer.run = run
    return run, trainer, runtimes


def _with(runtime, context):
    """``runtime``, keeping the context it was built with."""
    runtime.context = context
    return runtime


def eval_files(run, name="val"):
    """Each eval's results file; its transcripts file sits beside it."""
    directory = run.out / "eval" / name
    if not directory.exists():
        return []
    return sorted(
        p.name for p in directory.glob("*.jsonl") if not p.name.endswith(".transcripts.jsonl")
    )


async def until(condition, seconds=10):
    async with asyncio.timeout(seconds):
        while not condition():
            await asyncio.sleep(0.01)


async def test_evals_run_before_training_on_the_cadence_and_after_the_last_update(config):
    run, trainer, training, evaluation = build(config)
    events = []
    run.add_step_sink(events.append)
    assert (await asyncio.wait_for(run.execute(), 10)).ok

    assert eval_files(run) == ["base.jsonl", "step-1.jsonl", "step-3.jsonl", "step-4.jsonl"]
    # The base model, then each evaluated step's kept weights, each sampler let go.
    assert sorted(
        (s.model_path or "" for s in trainer.eval_samplers), key=lambda p: (len(p), p)
    ) == [
        "",
        "bt://loops:fake/sampler_weights/step-1",
        "bt://loops:fake/sampler_weights/step-3",
        "bt://loops:fake/sampler_weights/step-4",
    ]
    assert all(sampler.deactivated for sampler in trainer.eval_samplers)
    rows = [
        json.loads(line) for line in (run.out / "eval/val/step-3.jsonl").read_text().splitlines()
    ]
    assert [(row["task_id"], row["reward"]) for row in rows] == [("e0", 1.0), ("e1", 0.0)]
    # On the update axis, beside training's own metrics.
    evaluated = sorted(e.step for e in events if "eval/val/reward_mean" in e.gauges)
    assert evaluated == [-1, 1, 3, 4]
    (at_three,) = [e for e in events if e.step == 3 and "eval/val/reward_mean" in e.gauges]
    assert at_three.gauges["eval/val/reward_mean"] == 0.5
    assert training.closed and evaluation.closed


async def test_a_dedicated_eval_sampler_reporting_its_own_version_is_still_graded(config):
    """A dedicated standalone sampler reports its own version counter (0), not the
    step it serves; the eval still grades every rollout at that step."""
    # Sync, so on-policy only, as the live run that found this: any lag at all is stale.
    run, trainer, _, _ = build(config, max_steps=2)
    trainer.eval_sampler_reports = 0
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    rows = [
        json.loads(line) for line in (run.out / "eval/val/step-1.jsonl").read_text().splitlines()
    ]
    assert [(row["reward"], row["discarded"]) for row in rows] == [(1.0, None), (0.0, None)]


async def test_each_eval_keeps_what_every_rollout_was_asked_and_replied(config):
    run, _, _, _ = build(config, max_steps=2)
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    directory = run.out / "eval" / "val"
    for stem in ("base", "step-1"):
        rows = [json.loads(line) for line in (directory / f"{stem}.jsonl").read_text().splitlines()]
        transcripts = [
            json.loads(line)
            for line in (directory / f"{stem}.transcripts.jsonl").read_text().splitlines()
        ]
        assert [t["rollout_id"] for t in transcripts] == [row["rollout_id"] for row in rows]
        for transcript in transcripts:
            [turn] = transcript["turns"]
            assert turn["messages"] == [{"role": "user", "content": "x"}]
            assert turn["reply"]["parsed"]["content"] == "ok"  # what the policy said


async def test_with_no_cadence_only_the_base_model_and_the_last_update_are_evaluated(config):
    run, _, _, _ = build(config, **{"eval.every_steps": 0})
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    assert eval_files(run) == ["base.jsonl", "step-4.jsonl"]


async def test_a_run_that_starts_from_other_weights_has_no_base_eval(config):
    run, _, _, _ = build(config, **{"training.init_from": "bt://loops:other/state/step-9"})
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    assert eval_files(run) == ["step-1.jsonl", "step-3.jsonl", "step-4.jsonl"]


async def test_a_resumed_run_has_no_base_eval(config, tmp_path):
    class Resumable(MemoryTrainer):
        async def checkpoint(self, report, *, final=False, keep_sampler_weights=False):
            report = await super().checkpoint(
                report, final=final, keep_sampler_weights=keep_sampler_weights
            )
            report.checkpoint_state_path = "bt://loops:test/state/checkpoint"
            return report

    first, _, _, _ = build(config, max_steps=1, output_dir=str(tmp_path))
    first.trainer = Resumable()
    assert (await asyncio.wait_for(first.execute(), 10)).ok
    manifest = first.out / "checkpoints" / "step-0.json"

    resumed, trainer, _, _ = build(
        config, max_steps=3, output_dir=str(tmp_path), resume_from=str(manifest)
    )
    trainer.policy_version = 1
    assert (await asyncio.wait_for(resumed.execute(), 10)).ok
    assert resumed.out == first.out
    # The first run evaluated the base model; the resumed one only its own updates.
    assert eval_files(resumed) == ["base.jsonl", "step-0.jsonl", "step-1.jsonl", "step-2.jsonl"]
    assert all(s.model_path for s in trainer.eval_samplers)


async def test_an_eval_set_sends_image_prompts_as_training_does(config, monkeypatch):
    from support.fakes import FakePrime
    from support.images import png

    from loops_cookbook.llm.prime import PrimeRenderer

    monkeypatch.setattr(fakes, "EvalSampler", ImageEvalSampler)
    run, trainer, _, evaluation = build(
        config, max_steps=1, renderer=lambda: PrimeRenderer(FakePrime())
    )
    evaluation.content = [{"type": "image", "image": png()}]
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    rows = [json.loads(line) for line in (run.out / "eval/val/base.jsonl").read_text().splitlines()]
    assert [(row["reward"], row["discarded"]) for row in rows] == [(1.0, None), (0.0, None)]
    prompts = [prompt for sampler in trainer.eval_samplers for prompt in sampler.prompts]
    assert prompts and all(
        any(isinstance(chunk, ImageChunk) for chunk in prompt.chunks) for prompt in prompts
    )


class ImageEvalSampler(fakes.EvalSampler):
    """An eval sampler that echoes the expanded prompt, as Loops does for an image one."""

    def __init__(self, model_path, reports=None):
        super().__init__(model_path, reports)
        self.prompts = []

    async def sample_async(self, prompt, **kwargs):
        self.prompts.append(prompt)
        response = await super().sample_async(prompt, **kwargs)
        response.prompt_token_ids = [
            token
            for chunk in prompt.chunks
            for token in (
                chunk.tokens if isinstance(chunk, EncodedTextChunk) else [fakes.PAD] * chunk.length
            )
        ]
        return response


async def test_eval_samples_never_train(config):
    run, trainer, _, evaluation = build(config)
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    plain, untouched, _, _ = build(config, evals=False)
    assert (await asyncio.wait_for(plain.execute(), 10)).ok

    assert sorted(set(evaluation.graded)) == ["e0", "e1"]  # the evals did run
    # Training took and recorded what a run without evals does, and only its own task.
    assert [len(rows) for rows in trainer.rows] == [len(rows) for rows in untouched.rows]
    records = [json.loads(line) for line in (run.out / "rollouts.jsonl").read_text().splitlines()]
    assert len(records) == len((plain.out / "rollouts.jsonl").read_text().splitlines())
    assert {record["task_id"] for record in records} == {TRAIN_TASK.id}
    samples = list((run.out / "samples").glob("*"))
    assert len(samples) == len(list((plain.out / "samples").glob("*")))


async def test_a_run_without_eval_sets_starts_no_eval_sampler(config):
    run, trainer, _, evaluation = build(config, evals=False)
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    assert trainer.eval_samplers == [] and evaluation.graded == []
    assert not (run.out / "eval").exists()


async def test_evals_reach_the_model_over_the_runs_gateway_as_training_does(config):
    run, trainer, _, _ = build(config, max_steps=2, mode=ModelAccessMode.OPENAI_HTTP)
    assert (await asyncio.wait_for(run.execute(), 20)).ok
    assert eval_files(run) == ["base.jsonl", "step-1.jsonl"]
    rows = [
        json.loads(line) for line in (run.out / "eval/val/step-1.jsonl").read_text().splitlines()
    ]
    assert [row["reward"] for row in rows] == [1.0, 0.0]
    assert sum(s.calls for s in trainer.eval_samplers) == 4  # each eval's own sampler served it


async def test_on_dedicated_training_goes_on_while_a_set_evaluates_and_skips_the_evals_due_meanwhile(
    config, caplog
):
    run, trainer, _, evaluation = build(config, paced=False)
    evaluation.hold = asyncio.Event()
    checkpoint, kept = trainer.checkpoint, []

    async def recording(report, *, final=False, keep_sampler_weights=False):
        kept.append(keep_sampler_weights)
        return await checkpoint(report, final=final, keep_sampler_weights=keep_sampler_weights)

    trainer.checkpoint = recording
    seen = []

    async def release_once_training_is_done():
        await until(lambda: trainer.policy_version == 5 and run.evals_running == 2)
        # Only the base eval is up; the final one waits for it.
        seen.append((len(trainer.eval_samplers), run.evals_running))
        evaluation.hold.set()

    releasing = asyncio.create_task(release_once_training_is_done())
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    await releasing
    assert seen == [(1, 2)]
    assert eval_files(run) == ["base.jsonl", "step-4.jsonl"]
    # The skipped steps kept their weights all the same, to evaluate later.
    assert kept == [False, True, False, True, True]
    for step in (1, 3):
        assert (
            f"eval 'val' at step {step} skipped: its eval before training is still running"
            in caplog.text
        )
        assert f"Run it later with `loops eval <config> --step {step}`" in caplog.text


async def test_on_serverless_a_sets_evals_overlap_none_skipped_each_with_its_own_cap(config):
    run, trainer, runtimes = build_sets(
        config,
        sets=("val",),
        paced=False,
        **{
            "eval.sets": {
                "val": {
                    "rollout_runtime": {"files": {"tasks": "val.jsonl"}},
                    "max_concurrent_rollouts": 2,
                }
            }
        },
    )
    trainer.serverless = True
    evaluation = runtimes["val"]
    evaluation.hold = asyncio.Event()
    held, seen = [0], []

    async def holding(attempt):
        held[0] += 1

    evaluation.on_sample = holding

    async def release_once_training_is_done():
        await until(lambda: trainer.policy_version == 5 and held[0] == 8)
        # Four evals at once, each holding both its rollouts: none waits on another.
        seen.append((len(trainer.eval_samplers), run.evals_running))
        evaluation.hold.set()

    releasing = asyncio.create_task(release_once_training_is_done())
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    await releasing
    assert seen == [(4, 4)]
    assert eval_files(run) == ["base.jsonl", "step-1.jsonl", "step-3.jsonl", "step-4.jsonl"]
    assert all(sampler.deactivated for sampler in trainer.eval_samplers)


async def test_an_eval_whose_sampler_does_not_start_is_skipped_and_training_completes(
    config, caplog
):
    run, trainer, _, _ = build(config)
    start = trainer.eval_sampler

    async def no_capacity_at_step_one(model_path, *, ready_timeout):
        if model_path and model_path.endswith("step-1"):
            raise TimeoutError("Sampler deployment 7wljz0q did not reach ACTIVE in 900.0s.")
        return await start(model_path, ready_timeout=ready_timeout)

    trainer.eval_sampler = no_capacity_at_step_one
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    assert eval_files(run) == ["base.jsonl", "step-3.jsonl", "step-4.jsonl"]
    [skipped] = [r for r in caplog.records if "skipped" in r.getMessage()]
    assert skipped.levelno == logging.ERROR
    assert "eval 'val' at step 1 skipped: its sampler was not ready (TimeoutError" in caplog.text
    assert "needs free allocation for" in caplog.text
    assert "may still be up and BILLING" in skipped.getMessage()


async def test_stopping_while_an_eval_sampler_starts_says_it_may_be_up(config, caplog):
    run, trainer, _, _ = build(config, max_steps=1)
    starting = asyncio.Event()

    async def never_ready(model_path, *, ready_timeout):
        starting.set()
        await asyncio.Event().wait()

    trainer.eval_sampler = never_ready
    execution = asyncio.create_task(run.execute())
    await asyncio.wait_for(starting.wait(), 5)
    execution.cancel()
    await asyncio.wait({execution}, timeout=10)
    [stopped] = [r for r in caplog.records if "while its sampler was starting" in r.getMessage()]
    assert stopped.levelno == logging.ERROR
    assert "may still be up and BILLING" in stopped.getMessage()


async def test_an_eval_that_fails_systemically_is_logged_and_training_completes(config, caplog):
    run, _, _, evaluation = build(config, max_steps=2, max_consecutive_rollout_failures=1)

    async def broken_eval_sandbox(attempt):
        raise RuntimeError("eval sandbox provider down")

    evaluation.on_sample = broken_eval_sandbox
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    assert eval_files(run) == []
    assert "eval 'val' before training failed. Training continues" in caplog.text
    assert "eval 'val' at step 1 failed. Training continues" in caplog.text


async def test_an_eval_whose_episode_ignores_cancellation_leaves_the_run_unclean(config):
    run, _, _, evaluation = build(config, max_steps=1, max_consecutive_rollout_failures=1)

    async def one_fails_one_will_not_stop(attempt):
        if attempt.spec.task.id == "e0":
            raise RuntimeError("eval sandbox provider down")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(1.5)  # past its cancellation grace

    evaluation.on_sample = one_fails_one_will_not_stop
    outcome = await asyncio.wait_for(run.execute(), 20)
    assert outcome.status == "unclean"


async def test_an_eval_whose_sets_slots_kept_hooks_all_hold_is_skipped_before_its_sampler(
    config, caplog
):
    """A hook kept past its grace holds its set's only slot into the set's next eval,
    which could run nothing: it is skipped without starting a sampler."""
    run, trainer, _, evaluation = build(
        config,
        max_steps=2,
        **{
            "rollout.rollout_timeout_seconds": 0.3,
            "eval.sets": {
                "val": {
                    "rollout_runtime": {"files": {"tasks": "val.jsonl"}},
                    "max_concurrent_rollouts": 1,
                }
            },
        },
    )

    async def will_not_stop(attempt):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(3.0)  # past its cancellation grace: kept

    evaluation.on_sample = will_not_stop
    await asyncio.wait_for(run.execute(), 20)
    assert "eval 'val' before training failed. Training continues" in caplog.text
    assert (
        "eval 'val' at step 1 skipped. Hooks of its earlier evals ignored cancellation"
        in caplog.text
    )
    assert len(trainer.eval_samplers) == 1


async def test_an_eval_of_weights_a_sampler_cannot_serve_is_skipped(config, caplog):
    run, trainer, _, _ = build(config, max_steps=2)
    checkpoint = trainer.checkpoint

    async def local_weight_sync(report, *, final=False, keep_sampler_weights=False):
        kept = await checkpoint(report, final=final, keep_sampler_weights=keep_sampler_weights)
        if kept.sampler_weights is None:
            return kept
        return dataclasses.replace(kept, sampler_weights="/tmp/weights/sampler_weights/step-1/")

    trainer.checkpoint = local_weight_sync
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    assert eval_files(run) == ["base.jsonl"]
    assert "eval at step 1 skipped. The trainer returned '/tmp/weights" in caplog.text


async def test_a_stopped_run_lets_every_eval_sampler_go_and_names_the_evals_left(config, caplog):
    caplog.set_level(logging.WARNING)
    run, trainer, training, evaluation = build(config, paced=False)
    evaluation.hold = asyncio.Event()  # never set: the base eval runs, the final one waits
    execution = asyncio.create_task(run.execute())
    await until(lambda: trainer.policy_version == 5 and run.evals_running == 2)
    execution.cancel()
    await asyncio.wait({execution}, timeout=10)
    assert all(s.deactivated for s in trainer.eval_samplers)
    assert eval_files(run) == []
    assert "eval 'val' before training did not finish" in caplog.text
    assert "eval 'val' at step 4 did not finish" in caplog.text
    assert trainer.closed and training.closed and evaluation.closed


async def test_a_completed_run_waits_for_its_final_eval(config):
    run, trainer, _, evaluation = build(config, max_steps=1, paced=False)
    evaluation.hold = asyncio.Event()

    async def release_after_training():
        await until(lambda: trainer.policy_version == 1)
        await asyncio.sleep(0.2)
        evaluation.hold.set()

    releasing = asyncio.create_task(release_after_training())
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    await releasing
    assert eval_files(run) == ["base.jsonl", "step-0.jsonl"]


@pytest.mark.parametrize("presses", [1, 2], ids=["Ctrl-C", "Ctrl-C twice"])
async def test_stopping_while_a_completed_run_waits_for_its_evals_still_closes_everything(
    config, presses
):
    run, trainer, training, evaluation = build(config, max_steps=1, paced=False)
    evaluation.hold = asyncio.Event()  # never set: the final wait never ends alone
    execution = asyncio.create_task(run.execute())
    await until(lambda: trainer.policy_version == 1 and run.evals_running == 2)
    await asyncio.sleep(0.1)  # the run is now waiting for its evals
    for _ in range(presses):
        execution.cancel()
        await asyncio.sleep(0)
    await asyncio.wait({execution}, timeout=10)
    assert execution.done()
    assert all(s.deactivated for s in trainer.eval_samplers)
    assert trainer.closed and training.closed and evaluation.closed
    # Recorded as any interruption, though every update was already made.
    assert json.loads((run.out / "failure.json").read_text())["status"] == "interrupted"


async def test_eval_sets_at_one_step_run_at_once_none_waiting_behind_another(config):
    run, _, runtimes = build_sets(config, sets=("val", "hard"), max_steps=1, paced=False)
    runtimes["val"].hold = asyncio.Event()
    execution = asyncio.create_task(run.execute())
    # Every 'hard' eval finishes while every 'val' eval is still held.
    await until(lambda: eval_files(run, "hard") == ["base.jsonl", "step-0.jsonl"])
    assert eval_files(run, "val") == []
    runtimes["val"].hold.set()
    assert (await asyncio.wait_for(execution, 10)).ok
    assert eval_files(run, "val") == ["base.jsonl", "step-0.jsonl"]


async def test_an_eval_sets_rollout_clock_is_its_own_runtimes(config):
    run, _, runtimes = build_sets(
        config, sets=("val",), max_steps=1, **{"rollout.rollout_timeout_seconds": None}
    )
    training, evaluation = runtimes["train"], runtimes["val"]
    training.capabilities = dataclasses.replace(training.capabilities, rollout_timeout_seconds=0.2)
    evaluation.capabilities = dataclasses.replace(
        evaluation.capabilities, rollout_timeout_seconds=10.0
    )

    async def longer_than_trainings_clock(attempt):
        await asyncio.sleep(0.5)

    evaluation.on_sample = longer_than_trainings_clock
    assert (await asyncio.wait_for(run.execute(), 20)).ok
    rows = [json.loads(line) for line in (run.out / "eval/val/base.jsonl").read_text().splitlines()]
    assert [row["reward"] for row in rows] == [1.0, 0.0]


async def test_what_an_eval_episode_reports_is_the_evals_not_trainings(config):
    run, _, runtimes = build_sets(config, sets=("val",), max_steps=1)
    events = []
    run.add_step_sink(events.append)

    async def report(attempt):
        attempt.services.metrics.observe("probe", 1.0)

    runtimes["val"].on_sample = report
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    training = [e for e in events if "eval/val/reward_mean" not in e.gauges]
    evals = [e for e in events if "eval/val/reward_mean" in e.gauges]
    assert training and not any("runtime/probe/count" in e.gauges for e in training)
    # Each eval its own: its 2 episodes, not training's or another eval's.
    assert [e.gauges["eval/val/runtime/probe/count"] for e in evals] == [2.0, 2.0]


async def test_stopping_while_one_eval_runtime_closes_still_closes_the_others(config):
    run, _, runtimes = build_sets(config, sets=("val", "hard"), max_steps=1)
    closing = asyncio.Event()

    async def close_slowly():
        closing.set()
        await asyncio.sleep(60)

    runtimes["val"].close = close_slowly
    execution = asyncio.create_task(run.execute())
    await until(closing.is_set)
    execution.cancel()
    await asyncio.wait({execution}, timeout=10)
    assert execution.done()
    assert runtimes["hard"].closed and runtimes["train"].closed


async def test_a_sets_rollout_cap_holds_in_each_of_its_evals(config):
    run, _, runtimes = build_sets(
        config,
        sets=("val",),
        **{
            "eval.sets": {
                "val": {
                    "rollout_runtime": {"files": {"tasks": "val.jsonl"}},
                    "max_concurrent_rollouts": 1,
                }
            }
        },
    )
    running, peak = [0], [0]

    async def one_at_a_time(attempt):
        running[0] += 1
        peak[0] = max(peak[0], running[0])
        await asyncio.sleep(0.05)
        running[0] -= 1

    runtimes["val"].on_sample = one_at_a_time
    assert (await asyncio.wait_for(run.execute(), 20)).ok
    assert eval_files(run) == ["base.jsonl", "step-1.jsonl", "step-3.jsonl", "step-4.jsonl"]
    assert peak == [1]


async def test_an_eval_step_whose_weights_have_no_path_skips_that_eval_and_training_completes(
    config,
):
    run, trainer, _, _ = build(config)
    keep = trainer.checkpoint

    async def no_path_at_step_three(report, *, final=False, keep_sampler_weights=False):
        kept = await keep(report, final=final, keep_sampler_weights=keep_sampler_weights)
        return dataclasses.replace(kept, sampler_weights=None) if report.step == 3 else kept

    trainer.checkpoint = no_path_at_step_three
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    assert eval_files(run) == ["base.jsonl", "step-1.jsonl", "step-4.jsonl"]


async def test_what_an_eval_runtime_reports_through_its_context_reaches_the_evals(config):
    run, _, runtimes = build_sets(config, sets=("val",), max_steps=1)
    events = []
    run.add_step_sink(events.append)

    async def report(attempt):
        runtimes["val"].context.metrics.observe("trials", 1.0)

    runtimes["val"].on_sample = report
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    training = [e for e in events if "eval/val/reward_mean" not in e.gauges]
    evals = [e for e in events if "eval/val/reward_mean" in e.gauges]
    assert training and not any("runtime/trials/count" in e.gauges for e in training)
    # The set's, across its evals: 2 episodes each, base and final.
    counts = [e.gauges["eval/val/set/runtime/trials/count"] for e in evals]
    assert len(counts) == 2 and max(counts) == 4.0


async def test_stopping_while_closing_after_an_eval_set_fails_preflight_closes_the_rest(config):
    run, _, runtimes = build_sets(config, sets=("val", "hard", "bad"), max_steps=1)
    closing = asyncio.Event()

    async def close_slowly():
        closing.set()
        await asyncio.sleep(60)

    async def no_tasks():
        raise RuntimeError("its task file is missing")

    runtimes["val"].close = close_slowly
    runtimes["bad"].prepare = no_tasks
    execution = asyncio.create_task(run.execute())
    await until(closing.is_set)
    execution.cancel()
    await asyncio.wait({execution}, timeout=10)
    assert execution.done()
    assert runtimes["hard"].closed and runtimes["bad"].closed and runtimes["train"].closed


async def test_stopping_while_an_eval_is_written_waits_for_the_write(config, monkeypatch):
    from loops_cookbook.rl import run as run_module

    write = run_module._write_eval
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def write_slowly(*args):
        started.set()
        release.wait(5)
        write(*args)
        finished.set()

    monkeypatch.setattr(run_module, "_write_eval", write_slowly)
    run, _, _, _ = build(config, max_steps=1)
    execution = asyncio.create_task(run.execute())
    await until(started.is_set)
    execution.cancel()
    threading.Timer(0.3, release.set).start()
    await asyncio.wait({execution}, timeout=10)
    assert execution.done() and finished.is_set()


async def test_a_stopped_run_closes_only_after_every_sets_eval_has_cleaned_up(config):
    run, trainer, runtimes = build_sets(config, sets=("val", "hard"), max_steps=1, paced=False)
    order = []
    for name in ("val", "hard"):
        runtimes[name].hold = asyncio.Event()  # never set: both base evals run on

    async def slow_abort(attempt, reason):
        await asyncio.sleep(0.3)
        order.append("hard cleaned up")

    runtimes["hard"].abort = slow_abort
    close = trainer.close

    async def closing():
        order.append("trainer closed")
        await close()

    trainer.close = closing
    execution = asyncio.create_task(run.execute())
    await until(lambda: len(trainer.eval_samplers) == 2 and run.evals_running >= 2)
    await asyncio.sleep(0.05)  # both sets' rollouts are sampling
    execution.cancel()
    await asyncio.wait({execution}, timeout=10)
    assert order == ["hard cleaned up"] * len(EVAL_TASKS) + ["trainer closed"]
    assert all(s.deactivated for s in trainer.eval_samplers)


async def evaluate_later(config, directory, step, *, sets=("val",), only=(), **overrides):
    """``loops eval`` of ``step`` in run ``directory``, on fresh fakes: what it returned,
    the trainer and the rollout runtimes by name."""
    runtimes = {
        name: EpisodeRuntime(EVAL_TASKS, reward=lambda task: float(task.id == "e0"))
        for name in ("train", *sets)
    }
    trainer = MemoryTrainer()
    settings = {
        "rollout_runtime.files": {"tasks": "train.jsonl"},
        "eval.sets": {
            name: {"rollout_runtime": {"files": {"tasks": f"{name}.jsonl"}}} for name in sets
        },
        **overrides,
    }
    evaluated = await recipe.evaluate(
        config(**settings),
        directory,
        step,
        sets=only,
        trainer=trainer,
        runtime_factory=lambda _settings, *, context: _with(
            runtimes[context.files.path("tasks").stem], context
        ),
        renderer_factory=lambda _config: Renderer(),
    )
    return evaluated, trainer, runtimes


async def test_a_run_records_the_sampler_weights_it_keeps_for_loops_eval(config):
    run, _, _, _ = build(config)
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    rows = [
        json.loads(line) for line in (run.out / "sampler_weights.jsonl").read_text().splitlines()
    ]
    assert [(row["step"], row["path"], row["policy_version"]) for row in rows] == [
        (step, f"bt://loops:fake/sampler_weights/step-{step}", step + 1) for step in (1, 3, 4)
    ]


async def test_a_skipped_eval_runs_after_the_fact_on_the_weights_the_run_kept(config):
    run, _, _, evaluation = build(config, paced=False)
    evaluation.hold = asyncio.Event()

    async def release_once_training_is_done():
        await until(lambda: run.evals_running == 2)
        evaluation.hold.set()

    releasing = asyncio.create_task(release_once_training_is_done())
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    await releasing
    assert eval_files(run) == ["base.jsonl", "step-4.jsonl"]

    evaluated, trainer, runtimes = await evaluate_later(config, run.out, 3)
    assert evaluated == ["val"]
    assert eval_files(run) == ["base.jsonl", "step-3.jsonl", "step-4.jsonl"]
    rows = [
        json.loads(line) for line in (run.out / "eval/val/step-3.jsonl").read_text().splitlines()
    ]
    assert [(row["task_id"], row["reward"]) for row in rows] == [("e0", 1.0), ("e1", 0.0)]
    assert (run.out / "eval/val/step-3.transcripts.jsonl").is_file()
    [sampler] = trainer.eval_samplers
    assert sampler.model_path == "bt://loops:fake/sampler_weights/step-3" and sampler.deactivated
    # Served as kept: no trainer of its own.
    assert trainer.loaded == [] and not trainer.run_active
    assert trainer.closed and all(r.closed for r in runtimes.values())


async def test_a_step_kept_only_as_trainer_state_is_saved_for_a_sampler_then_its_run_deactivated(
    config, tmp_path
):
    directory = tmp_path / "copied-run"
    (directory / "checkpoints").mkdir(parents=True)
    state = "bt://loops:womx17q/weights/step-29"
    (directory / "checkpoints/step-29.json").write_text(json.dumps({"trainer_state_path": state}))

    evaluated, trainer, _ = await evaluate_later(config, directory, 29)
    assert evaluated == ["val"]
    assert (directory / "eval/val/step-29.jsonl").is_file()
    assert trainer.loaded == [state]
    [sampler] = trainer.eval_samplers
    assert sampler.model_path == "bt://loops:eval/sampler_weights/eval-step-29"
    assert sampler.deactivated and not trainer.run_active


async def test_the_base_model_is_evaluated_after_the_fact_without_weights(config, tmp_path):
    evaluated, trainer, _ = await evaluate_later(config, tmp_path, -1)
    assert evaluated == ["val"]
    assert (tmp_path / "eval/val/base.jsonl").is_file()
    assert [s.model_path for s in trainer.eval_samplers] == [None]


async def test_only_the_sets_named_are_evaluated_after_the_fact(config, tmp_path):
    evaluated, _, runtimes = await evaluate_later(
        config, tmp_path, -1, sets=("val", "hard"), only=("hard",)
    )
    assert evaluated == ["hard"]
    assert (tmp_path / "eval/hard/base.jsonl").is_file()
    assert not (tmp_path / "eval/val").exists()
    # The set left out is never built; every runtime built is closed.
    assert runtimes["val"].context is None
    assert runtimes["hard"].closed and runtimes["train"].closed


async def test_an_eval_after_the_fact_needs_open_files_for_its_eval_rollouts_alone(
    config, tmp_path, monkeypatch
):
    """``loops eval`` trains nothing, so training's rollouts and verifiers do not count
    toward the open-file limit, nor do the sets it leaves out."""
    import resource

    monkeypatch.setattr(resource, "getrlimit", lambda _which: (256, 4096))  # macOS's default
    big = {
        "rollout.max_concurrent_rollouts": 48,
        "rollout.max_concurrent_verifiers": 64,
        "eval.sets": {
            "val": {
                "rollout_runtime": {"files": {"tasks": "val.jsonl"}},
                "max_concurrent_rollouts": 4,
            },
            "hard": {
                "rollout_runtime": {"files": {"tasks": "hard.jsonl"}},
                "max_concurrent_rollouts": 64,
            },
        },
    }
    evaluated, _, _ = await evaluate_later(
        config, tmp_path, -1, sets=("val", "hard"), only=("val",), **big
    )
    assert evaluated == ["val"]


async def test_a_trainer_made_for_an_eval_after_the_fact_stops_whatever_fails_to_close(
    config, tmp_path, monkeypatch
):
    directory = tmp_path / "copied-run"
    (directory / "checkpoints").mkdir(parents=True)
    (directory / "checkpoints/step-29.json").write_text(
        json.dumps({"trainer_state_path": "bt://loops:womx17q/weights/step-29"})
    )

    async def fails_to_close(self):
        raise RuntimeError("an adapter would not close")

    monkeypatch.setattr(Run, "close_resources", fails_to_close)
    made = []
    real = MemoryTrainer.servable_weights

    async def servable_weights(self, *args):
        made.append(self)
        return await real(self, *args)

    monkeypatch.setattr(MemoryTrainer, "servable_weights", servable_weights)
    with pytest.raises(RuntimeError, match="would not close"):
        await evaluate_later(config, directory, 29)
    [trainer] = made
    assert not trainer.run_active and trainer.closed


@pytest.mark.parametrize(
    ("weights_at", "state_at", "served"),
    [
        (2.0, 1.0, "weights"),
        (1.0, 2.0, "state"),
        (None, 2.0, "weights"),
    ],
    ids=["weights kept after the state", "retrained after a resume", "weights not dated"],
)
def test_a_step_retrained_after_a_resume_is_evaluated_from_its_newer_state(
    tmp_path, weights_at, state_at, served
):
    """A resume from an earlier step retrains this one: what it kept last is its."""
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/step-9.json").write_text(
        json.dumps({"trainer_state_path": "state-9", "written_at": state_at})
    )
    row = {"step": 9, "path": "weights-9", "policy_version": 10}
    if weights_at is not None:
        row["written_at"] = weights_at
    (tmp_path / "sampler_weights.jsonl").write_text(json.dumps(row) + "\n")
    saved = recipe.saved_weights(tmp_path, 9)
    assert (saved.sampler_weights, saved.trainer_state) == (
        ("weights-9", None) if served == "weights" else (None, "state-9")
    )


async def test_an_eval_after_the_fact_writes_into_the_run_named_not_the_one_resumed(
    config, tmp_path
):
    resumed = tmp_path / "resumed-run"
    (resumed / "checkpoints").mkdir(parents=True)
    (resumed / "checkpoints/step-2.json").write_text(
        json.dumps(
            {
                "trainer_state_path": "bt://loops:a/weights/step-2",
                "next_step": 3,
                "run_id": "a",
                "run_directory": str(resumed),
            }
        )
    )
    named = tmp_path / "named-run"
    evaluated, _, _ = await evaluate_later(
        config, named, -1, **{"training.resume_from": str(resumed / "checkpoints/step-2.json")}
    )
    assert evaluated == ["val"]
    assert (named / "eval/val/base.jsonl").is_file()
    assert not (resumed / "eval").exists()


@pytest.mark.parametrize(
    ("step", "only", "refusal"),
    [
        (5, (), "kept no weights for step 5. It can evaluate 2, 9"),
        (-1, ("nope",), "no eval set 'nope'. The config's eval sets are 'val'"),
    ],
    ids=["no weights", "unknown set"],
)
async def test_an_eval_after_the_fact_it_cannot_run_is_refused(
    config, tmp_path, step, only, refusal
):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/step-9.json").write_text(json.dumps({"trainer_state_path": "s"}))
    (tmp_path / "sampler_weights.jsonl").write_text(
        json.dumps({"step": 2, "path": "p", "policy_version": 3}) + "\n"
    )
    with pytest.raises(ValueError, match=refusal):
        await evaluate_later(config, tmp_path, step, only=only)
