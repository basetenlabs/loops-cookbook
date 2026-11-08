"""Compose the real scheduler, attempt driver, and lease without network infrastructure."""

import asyncio
import contextlib
import dataclasses
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from baseten.loops.models import SamplingParams
from support.cancellation import (
    Gate,
    Landing,
    Response,
    Trigger,
    Work,
    WorkFailed,
    cancellation_matrix,
    outcome_within,
    settle_loop,
)
from support.fakes import MemoryTrainer
from support.harness import Renderer

from loops_cookbook.evidence import AttemptIdentity
from loops_cookbook.llm.rendering import RenderedPrompt
from loops_cookbook.rl import recipe
from loops_cookbook.rl.run import Manifest, Run, finished_hint, row_record
from loops_cookbook.run_record import finished_message, manifest_path
from loops_cookbook.runtime.api import (
    AttemptAbortReason,
    EpisodeResult,
    GradeResult,
    PermanentGradingFailure,
    PermanentRolloutFailure,
)
from loops_cookbook.runtime.driver import RuntimeExecutionFailure

from .conftest import TASK, KeepingTrainer, MemoryRuntime, records, unused


async def test_the_snapshot_is_one_merged_view_while_running_and_silent_after(rl_run):
    """One definition of current state, so the region and the heartbeat cannot differ."""
    seen = []
    run, _, _ = rl_run(max_steps=1)
    run.add_step_sink(lambda event: seen.append(run.snapshot()))
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok

    (during,) = seen
    assert during["progress/groups_needed"] == float(run.config.training.groups_per_step)
    # The pipeline, the rollout runtime and this run's own discard counters, in one dict.
    assert {"pipeline/sampled", "discard/generations"} <= during.keys()
    # Every count the progress display reads is one this run actually produces
    # (Harbor's own row is checked against the Harbor rollout runtime's metrics).
    from loops_cookbook.console import STAGES, WORKER_SLOTS

    read = {key for key, _ in STAGES} | {k for key, _, cap in WORKER_SLOTS for k in (key, cap)}
    read = {key for key in read if not key.startswith("runtime/harbor/")}
    assert read <= during.keys(), sorted(read - during.keys())
    # Observers keep polling through Harbor's cleanup drain. Once the pipeline has
    # stopped there is nothing true left to say, and its frozen counts beside a
    # climbing clock would read as a run still working.
    assert run.snapshot() is None


async def test_async_run_samples_next_batch_while_trainer_is_busy(rl_run):
    run, trainer, runtime = rl_run(
        mode="async",
        group_size=2,
        groups_per_step=1,
        max_steps=2,
        max_steps_off_policy=1,
        max_concurrent_rollouts=1,
    )
    accumulate = trainer.accumulate

    async def held_training(rows):
        await accumulate(rows)
        if trainer.policy_version:
            return  # the second step: nothing comes after it
        async with asyncio.timeout(3):
            while len(runtime.samples) < 4:
                await asyncio.sleep(0)
        assert trainer.policy_version == 0  # next batch finished before publication
        assert run.snapshot()["pipeline/trained"] == 0
        assert run.snapshot()["queue/trainable_rollouts"] == 2

    trainer.accumulate = held_training
    assert (await asyncio.wait_for(run.execute(), 5)).ok
    assert trainer.policy_version == 2


@pytest.mark.parametrize(
    ("group_size", "max_concurrent_rollouts", "lost"),
    [(2, 4, False), (2, 8, False), (3, 12, True)],
    ids=["slots-fit-window", "slots-exceed-window", "groups-lose-members"],
)
async def test_async_run_starts_no_rollout_that_would_train_stale(
    rl_run, group_size, max_concurrent_rollouts, lost
):
    """Concurrency above the off-policy window must not start groups that train too late.

    A batch takes whole groups, so groups that lost members still use a batch place each.
    The pipeline's own tests cover each admission setting; this is the whole run.
    """
    run, trainer, runtime = rl_run(
        mode="async",
        group_size=group_size,
        max_group_losses=1 if lost else 0,
        groups_per_step=2,
        max_steps=6,
        max_steps_off_policy=1,
        max_concurrent_rollouts=max_concurrent_rollouts,
    )
    sample, accumulate = runtime.sample, trainer.accumulate

    async def slow_sample(attempt):
        await asyncio.sleep(0.01)
        if lost and attempt.spec.identity.rollout_id.endswith("r2"):
            raise PermanentRolloutFailure("lost")
        return await sample(attempt)

    async def slow_training(rows):
        await asyncio.sleep(0.05)
        await accumulate(rows)

    runtime.sample, trainer.accumulate = slow_sample, slow_training
    assert (await asyncio.wait_for(run.execute(), 10)).ok
    stale = {"stale_before_grading", "stale_before_training"}
    assert [r for r in records(run) if r.get("reason") in stale] == []


async def test_a_failed_step_record_cannot_discard_a_finished_optimizer_update(
    rl_run, caplog, monkeypatch
):
    """A full disk is a reason to shout, not to throw away a two-day run."""
    from loops_cookbook.rl import run as run_module

    monkeypatch.setattr(
        run_module,
        "_append_json",
        lambda path, record: (_ for _ in ()).throw(OSError("no space left on device")),
    )
    run, _, _ = rl_run(max_steps=1)
    outcome = await asyncio.wait_for(run.execute(), timeout=5)
    assert outcome.ok and outcome.steps == 1 and outcome.next_step == 1
    assert "Could not record step 0" in caplog.text


async def test_a_broken_observer_cannot_discard_a_finished_optimizer_update(rl_run, caplog):
    """There is no recovery past this point, so a watcher's bug must not end the run."""
    run, _, _ = rl_run(max_steps=1)

    def broken(event):
        raise RuntimeError("observer is broken")

    run.add_step_sink(broken)
    outcome = await asyncio.wait_for(run.execute(), timeout=5)
    assert outcome.ok and outcome.steps == 1
    # The step still counts, so a resume does not repeat work that was committed.
    assert outcome.next_step == 1
    assert "A step observer failed" in caplog.text


async def test_an_observer_can_leave_without_disturbing_the_others(rl_run):
    staying, leaving = [], []
    run, _, _ = rl_run(max_steps=1)
    run.add_step_sink(staying.append)
    run.add_step_sink(leaving.append)()
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok
    assert len(staying) == 1 and leaving == []


@pytest.mark.parametrize("sticky", [False, True])
async def test_affinity_unique_per_run_rollout_but_stable_across_retries_and_forks(config, sticky):
    trainer = MemoryTrainer()
    keys = []
    original = trainer.sample_async

    async def capture(prompt, **kwargs):
        keys.append(kwargs.get("affinity_key"))
        if not sticky:
            assert kwargs.get("affinity_key") is None
        return await original(prompt, **kwargs)

    trainer.sample_async = capture
    cfg = config(sampler_replica_affinity=sticky)
    params = SamplingParams(max_tokens=2)
    prompt = RenderedPrompt.text((1,))

    def attempt(owner, rollout_id):
        lease = owner._lease(owner._lease_wiring())
        return lease(AttemptIdentity("run", "same-group", rollout_id, uuid4().hex))

    run = Run(cfg, trainer=trainer, runtime_factory=unused, renderer_factory=unused)
    first = attempt(run, "r1")
    for conversation in ("main", "fork"):
        await first.generate("main", prompt, params, conversation_id=conversation)
    for owner, rollout in (
        (run, "r1"),
        (run, "r2"),
        (Run(cfg, trainer=trainer, runtime_factory=unused, renderer_factory=unused), "r1"),
    ):
        await attempt(owner, rollout).generate("main", prompt, params, conversation_id="c")
    if sticky:
        assert keys[0] == keys[1] == keys[2]
        assert len({keys[0], keys[3], keys[4]}) == 3
    else:
        assert keys == [None] * 5


def test_network_environment_names_the_proxy_without_leaking_its_password(monkeypatch):
    """A proxy failure is undiagnosable later unless the run wrote down the proxy."""
    from loops_cookbook.rl.run import _network_environment

    monkeypatch.setattr("urllib.request.getproxies", dict)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    assert _network_environment() == {}
    monkeypatch.setenv("HTTPS_PROXY", "http://someone:hunter2@proxy.internal:8080")
    resolved = _network_environment()
    assert resolved == {"https": "http://<redacted>@proxy.internal:8080"}
    assert "hunter2" not in str(resolved)


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_uniform_reward_run_aborts_without_training(rl_run, mode):
    run, trainer, runtime = rl_run(mode=mode, max_consecutive_unusable_groups=3)

    async def uniform(attempt, rollout):
        return GradeResult(reward=0.0)

    runtime.grade = uniform
    outcome = await asyncio.wait_for(run.execute(), timeout=5)
    assert outcome.status == "aborted" and not outcome.ok
    assert outcome.steps == 0 and not trainer.rows
    assert outcome.discards["no_variance"] == 3 * run.config.training.group_size
    assert trainer.closed and runtime.closed
    recorded = records(run)
    unusable = [r for r in recorded if r["reason"] == "no_variance"]
    assert len(unusable) == 3 * run.config.training.group_size  # each discarded one, recorded
    assert {(r["disposition"], r["generations"]) for r in unusable} == {("discarded", 1)}
    # What async mode had in flight when the run aborted is recorded as cancelled.
    assert {r["reason"] for r in recorded} <= {"no_variance", "cancelled"}
    assert len({r["rollout_id"] for r in recorded}) == len(recorded)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("aggregation", ["sum", "token_mean", "fixed_budget", "task_mean"])
async def test_composed_run(rl_run, mode, aggregation):
    from support.audit import audit_run

    run, trainer, runtime = rl_run(
        mode=mode,
        trace="full",
        max_steps=3,
        loss_aggregation=aggregation,
        **({"loss_token_budget": 10} if aggregation == "fixed_budget" else {}),
    )
    outcome = await asyncio.wait_for(run.execute(), timeout=5)
    assert outcome.ok and outcome.steps == 3
    assert trainer.closed and runtime.closed
    assert len(trainer.rows) == 3
    assert len((run.out / "steps.jsonl").read_text().splitlines()) == 3
    captures = list((run.out / "samples").glob("*.jsonl"))
    assert captures
    record = json.loads(captures[0].read_text().splitlines()[0])
    assert record["completion_ids"] == [3, 4]
    assert record["group_id"] and record["rollout_id"] and record["attempt_id"]
    audit = audit_run(run.out)
    assert audit.steps == 3 and audit.rollouts == 12 and audit.datums == 12
    steps = records(run, "steps")
    batch = json.loads((run.out / "batches" / "step-0.json").read_text())
    assert batch["aggregation"]["mode"] == aggregation
    assert batch["aggregation"]["active_tokens"] == 8
    if aggregation in ("sum", "token_mean"):
        assert not [key for step in steps for key in step if key.startswith("aggregation/")]
    if aggregation == "task_mean":
        assert all(
            {key for key in step if key.startswith("aggregation/")}
            == {"aggregation/group_denominator_min", "aggregation/group_denominator_max"}
            for step in steps
        )
    if aggregation == "fixed_budget":
        assert all(step["aggregation/denominator"] == 40 for step in steps)
        weights = [
            value
            for row in trainer.rows[0]
            for value in row.datum.loss_fn_inputs["advantages"].data
        ]
        assert sorted(weights) == pytest.approx([-0.25 / 40] * 6 + [0.75 / 40] * 2)
    if mode == "sync":
        assert [r.minimum_policy_version for r in runtime.samples] == [0] * 4 + [1] * 4 + [2] * 4


async def test_task_mean_preserves_groups_when_one_task_is_admitted_twice(rl_run):
    from support.audit import audit_run

    class VariableLength(MemoryRuntime):
        async def sample(self, attempt):
            model = attempt.services.model_access.llm()
            turns = 1 if attempt.spec.identity.group_id.startswith("g0000000-") else 3
            conversation = model.conversation()
            for _ in range(turns):
                await conversation.generate(messages=[{"role": "user", "content": "x"}])
            return EpisodeResult()

    trainer = MemoryTrainer()
    run, trainer, _ = rl_run(
        trainer,
        VariableLength(),
        groups_per_step=2,
        max_steps=1,
        trace="full",
        loss_aggregation="task_mean",
    )
    outcome = await asyncio.wait_for(run.execute(), 5)
    assert outcome.ok
    batch = json.loads((run.out / "batches" / "step-0.json").read_text())
    assert batch["aggregation"]["group_denominators"] == [16, 48]
    rows = trainer.rows[0]
    assert len(rows) == 16
    first = {value for row in rows[:4] for value in row.datum.loss_fn_inputs["advantages"].data}
    assert sorted(first) == pytest.approx([-0.25 / 16, 0.75 / 16])
    second = {value for row in rows[4:] for value in row.datum.loss_fn_inputs["advantages"].data}
    assert sorted(second) == pytest.approx([-0.25 / 48, 0.75 / 48])
    assert audit_run(run.out).rollouts == 8


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_a_member_that_discarded_every_generation_passes_the_audit(rl_run, mode):
    """It counts in its group's baseline and trains nothing; the audit agrees."""
    from support.audit import audit_run

    class DiscardingR2(MemoryRuntime):
        async def sample(self, attempt):
            conversation = attempt.services.model_access.llm().conversation()
            reply = await conversation.generate(messages=[{"role": "user", "content": "x"}])
            if attempt.spec.identity.rollout_id.endswith("r2"):
                conversation.discard(reply)
            return EpisodeResult()

    run, trainer, _ = rl_run(MemoryTrainer(), DiscardingR2(), mode=mode, trace="full", max_steps=2)
    outcome = await asyncio.wait_for(run.execute(), timeout=5)
    assert outcome.ok and outcome.steps == 2
    trained = [r for r in records(run) if r["disposition"] == "trained"]
    assert sum(not r["sample_ids"] for r in trained) == 2
    audit = audit_run(run.out)
    assert audit.rollouts == 8 and audit.datums == 6 == sum(len(rows) for rows in trainer.rows)


async def test_every_rollout_is_recorded_with_its_disposition_and_length(rl_run):
    """Discarded rollouts too, so dispositions slice by task or length from one file."""
    failed = set()

    class Flaky(MemoryRuntime):
        async def sample(self, attempt):
            rollout_id = attempt.spec.identity.rollout_id
            if rollout_id.endswith("r3"):
                raise OSError("the sandbox died")  # every attempt
            if rollout_id.endswith("r2") and rollout_id not in failed:
                failed.add(rollout_id)
                raise OSError("the sandbox died")  # once: its retry trains
            return await super().sample(attempt)

        async def grade(self, attempt, rollout):
            if rollout.identity.rollout_id.endswith("r0"):
                raise PermanentGradingFailure("the verifier wrote no reward")
            return await super().grade(attempt, rollout)

    run, _, _ = rl_run(runtime=Flaky(), max_steps=1, max_group_losses=2, max_rollout_retries=1)
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok
    by_rollout = {record["rollout_id"].rsplit("-", 1)[1]: record for record in records(run)}
    lost = by_rollout.pop("r3")
    assert (lost["disposition"], lost["cause"], lost["retries"]) == (
        "discarded",
        "rollout_failed/os",
        1,
    )
    assert lost["task_id"] == TASK.id and lost["generations"] is None  # it never sealed
    ungraded = by_rollout.pop("r0")
    assert (ungraded["disposition"], ungraded["cause"]) == (
        "discarded",
        "grading_failed/permanent_grading",
    )
    assert ungraded["generations"] == 1  # it sealed, so its length is known
    assert {r["disposition"] for r in by_rollout.values()} == {"trained"}
    assert {r["generations"] for r in by_rollout.values()} == {1}
    assert {r["sampled_tokens"] for r in by_rollout.values()} == {2}
    assert [by_rollout[k]["retries"] for k in sorted(by_rollout)] == [0, 1]


@pytest.mark.parametrize("trace", ["transcripts", "full"])
async def test_an_attempt_that_fails_after_generating_keeps_its_transcript(rl_run, trace):
    """What failed is what a reader most needs to see, so it is written too."""

    class DiesLate(MemoryRuntime):
        async def sample(self, attempt):
            result = await super().sample(attempt)
            if attempt.spec.identity.rollout_id.endswith("r3"):
                raise OSError("the sandbox died after the model answered")
            return result

    run, _, _ = rl_run(
        runtime=DiesLate(), max_steps=1, max_group_losses=1, max_rollout_retries=0, trace=trace
    )
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok
    written = [json.loads(path.read_text()) for path in (run.out / "transcripts").glob("*.json")]
    by_rollout = {record["rollout_id"].rsplit("-", 1)[1]: record for record in written}
    failed = by_rollout.pop("r3")
    assert failed["sealed"] is False
    (turn,) = failed["turns"]
    assert turn["reply"] is not None and turn["trainable"] is False
    assert {record["sealed"] for record in by_rollout.values()} == {True}


async def test_a_trace_that_cannot_be_written_is_counted_and_everything_still_trains(
    rl_run, caplog
):
    """The trace is for reading the run: losing it never changes what trains."""
    run, _, _ = rl_run(max_steps=1, trace="full")
    run.out.mkdir(parents=True)
    for name in ("samples", "transcripts"):
        (run.out / name).write_text("a file where the directory should be")
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok
    rollouts = records(run)
    assert rollouts and {record["disposition"] for record in rollouts} == {"trained"}
    step = json.loads((run.out / "steps.jsonl").read_text().strip())
    assert step["trace/unwritten_samples"] == len(rollouts)  # one generation each
    assert step["trace/unwritten_transcripts"] == len(rollouts)
    assert "was not written" in caplog.text


async def test_a_step_reports_what_its_groups_lost_and_why(rl_run):
    class LosesOne(MemoryRuntime):
        async def sample(self, attempt):
            if attempt.spec.identity.rollout_id.endswith("r3"):
                raise OSError("the sandbox died")
            return await super().sample(attempt)

    run, _, _ = rl_run(runtime=LosesOne(), max_steps=1, max_group_losses=1, max_rollout_retries=0)
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok
    step = json.loads((run.out / "steps.jsonl").read_text().strip())
    assert step["groups/rollouts_lost"] == step["groups/rollouts_lost_max"] == 1
    assert step["discard/rollouts/rollout_failed/os"] == 1


_WINDOW_GAUGES = {
    "admission/committed_groups",
    "admission/window_groups",
    "admission/group_survival",
    "admission/window_held_seconds",
    "admission/window_held_share",
}


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_only_async_reports_the_window(rl_run, mode):
    """Sync starts every group it needs; a window it never applies would mislead."""
    run, _, _ = rl_run(mode=mode, max_steps=1)
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok
    step = json.loads((run.out / "steps.jsonl").read_text().strip())
    assert (_WINDOW_GAUGES & step.keys()) == (_WINDOW_GAUGES if mode == "async" else set())


@pytest.mark.parametrize(
    ("unrecorded", "fork_some"),
    [(False, False), (False, True), (True, False)],
    ids=["all recorded", "all recorded with forks", "one generation unrecorded"],
)
async def test_batch_replay_normalization_precedes_submission_and_saved_evidence(
    rl_run, unrecorded, fork_some
):
    from support.audit import audit_run

    class RecordingTrainer(MemoryTrainer):
        calls = 0

        async def sample_async(self, prompt, **kwargs):
            response = await super().sample_async(prompt, **kwargs)
            RecordingTrainer.calls += 1
            if not (unrecorded and RecordingTrainer.calls == 1):
                response.sequences[0].sequence_id = f"seq-{RecordingTrainer.calls}"
            return response

    class BranchingRuntime(MemoryRuntime):
        async def sample(self, attempt):
            llm = attempt.services.model_access.llm()
            await llm.conversation().generate(messages=[{"role": "user", "content": "x"}])
            if fork_some and attempt.spec.identity.rollout_id.endswith("r1"):
                await llm.conversation().generate(messages=[{"role": "user", "content": "fork"}])
            return EpisodeResult()

    run, trainer, _ = rl_run(
        RecordingTrainer(), BranchingRuntime(), max_steps=1, trace="full", router_replay=True
    )
    outcome = await asyncio.wait_for(run.execute(), timeout=5)
    assert outcome.ok
    rows = trainer.rows[0]
    # Forks keep their provenance; one unrecorded generation takes the whole batch off.
    assert all(row.routed == (not unrecorded) for row in rows)
    batch = json.loads((run.out / "batches" / "step-0.json").read_text())
    assert batch["rows"] == [row_record(row) for row in rows]
    step = json.loads((run.out / "steps.jsonl").read_text().strip())
    assert step["r3/routed_datums"] == (0 if unrecorded else len(rows))
    assert step["r3/unrecorded_datums"] == (1 if unrecorded else 0)
    assert audit_run(run.out).datums == len(rows)


async def test_preflight_failure_closes_boundaries(rl_run):
    run, trainer, runtime = rl_run()

    async def failed():
        raise ValueError("invalid BYO agent")

    runtime.prepare = failed
    with pytest.raises(RuntimeExecutionFailure, match="invalid BYO") as captured:
        await asyncio.wait_for(run.execute(), timeout=5)
    assert isinstance(captured.value.__cause__, ValueError)
    assert trainer.closed and runtime.closed


async def test_a_failing_runtime_close_ends_the_run_uncleanly_and_still_closes_trainer(
    rl_run, caplog
):
    """Shutdown step 4: a close() failure is reported, and the rest of shutdown runs."""
    run, trainer, runtime = rl_run(max_steps=1)

    async def failed_close():
        raise RuntimeError("sandbox cleanup incomplete")

    runtime.close = failed_close
    outcome = await asyncio.wait_for(run.execute(), timeout=5)
    assert outcome.status == "unclean" and not outcome.ok and outcome.steps == 1
    assert "sandbox cleanup incomplete" in caplog.text
    assert trainer.closed


async def test_a_runtime_close_that_ignores_cancellation_is_reported_not_awaited(rl_run, caplog):
    run, _, runtime = rl_run(
        max_steps=1,
        **{
            "rollout.cleanup.timeout_seconds": 0.05,
            "rollout.cleanup.cancellation_grace_seconds": 0.05,
        },
    )
    release = asyncio.Event()

    async def stubborn():
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue

    runtime.close = stubborn
    try:
        outcome = await asyncio.wait_for(run.execute(), timeout=5)
        assert outcome.status == "unclean"
        assert "did not stop" in caplog.text
    finally:
        release.set()


@cancellation_matrix(
    triggers=tuple(Trigger),
    points={"grace": (Landing.NOTHING, Landing.CANCEL)},
    responses=tuple(Response),
)
async def test_a_stopped_runtime_close_gets_its_grace_whatever_cancels_the_run(
    rl_run, caplog, case
):
    """Shutdown step 4 under the cleanup deadline or a cancellation: close() is cancelled
    once and waited for through its grace; an outside cancellation still propagates."""
    deadline = 0.05 if case.trigger is Trigger.DEADLINE else 30
    grace = 0.2 if case.still_running else 30
    run, _, runtime = rl_run(
        **{
            "rollout.cleanup.timeout_seconds": deadline,
            "rollout.cleanup.cancellation_grace_seconds": grace,
        },
    )
    work = Work(case.response)
    runtime.close = work
    caller = asyncio.create_task(run.close_runtime(runtime))
    try:
        async with asyncio.timeout(5):
            await work.running.wait()
            if case.trigger is Trigger.CANCEL:
                caller.cancel()
            await work.cancelled.wait()
            if case.at("grace") is Landing.CANCEL:
                caller.cancel()
                await settle_loop()
            assert not caller.done()  # a cancellation never cuts the grace short
            work.let_go.set()
            outcome = (await asyncio.gather(caller, return_exceptions=True))[0]
        assert work.cancellations == 1
        reported = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        if case.cancelled:
            assert isinstance(outcome, asyncio.CancelledError)
            assert reported == []
        else:
            assert outcome is None
            [message] = reported
            expected = {
                Response.STOPS: "cancelled at its deadline",
                Response.FAILS: "failed while stopping",
                Response.IGNORES: "did not stop",
            }
            assert expected[case.response] in message
    finally:
        work.end()


@cancellation_matrix(
    triggers=(Trigger.CANCEL,),
    points={"gasp": (Landing.NOTHING, Landing.CANCEL)},
    responses=(Response.STOPS, Response.FAILS),
)
async def test_a_cancelled_step_saves_its_last_gasp_checkpoint_before_shutdown_goes_on(
    rl_run, case
):
    """A step cut off before its optimizer step leaves the last committed step
    resumable; a second cancellation during that save neither cuts it short nor is
    lost."""
    work, save = Work(case.response), Gate(WorkFailed("unused"))

    class Committing(MemoryTrainer):
        async def accumulate(self, rows):
            if self.policy_version == 0:
                return await super().accumulate(rows)
            await work()

        async def checkpoint_state_only(self, step):
            await save.passage()
            return f"state-{step}"

    run, trainer, runtime = rl_run(Committing(), max_steps=2)
    running = asyncio.create_task(run.execute())
    try:
        async with asyncio.timeout(10):
            await work.running.wait()
            running.cancel()
            await work.cancelled.wait()
            work.let_go.set()
            await save.entered.wait()
            if case.at("gasp") is Landing.CANCEL:
                running.cancel()
                await settle_loop()
            assert not (runtime.closed or trainer.closed)  # shutdown waits for the save
            save.open(Landing.NOTHING)
        outcome = await outcome_within(running, 10)
        stopped_by_cancel = case.response is Response.STOPS or case.at("gasp") is Landing.CANCEL
        assert isinstance(outcome, asyncio.CancelledError if stopped_by_cancel else WorkFailed)
        manifest = json.loads((run.out / "checkpoints" / "step-0.json").read_text())
        assert manifest["trainer_state_path"] == "state-0"
    finally:
        work.end()
        save.open(Landing.NOTHING)


@pytest.mark.parametrize("scope", ["dedicated", "serverless"])
async def test_the_config_a_run_saves_loads_back_as_that_config(rl_run, scope):
    """config.json is a config `loops` takes, so what a run saves is what it ran."""
    from loops_cookbook.rl.cli import load_training_config

    run, _, _ = rl_run(max_steps=1, **{"compute.scope": scope})
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok
    assert load_training_config(run.out / "config.json") == run.config


@pytest.mark.parametrize("trace", ["off", "samples", "transcripts", "full"])
async def test_trace_levels_keep_the_run_record_and_nothing_more_than_asked(rl_run, trace):
    """Evidence is opt-in; the run record is not, or a finished run answers nothing."""
    run, _, _ = rl_run(max_steps=1, trace=trace)
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok

    for name in ("config.json", "steps.jsonl", "rollouts.jsonl"):
        assert (run.out / name).exists(), f"{name} is the run record, not a trace"
    assert (run.out / "batches").exists() is (trace == "full")
    assert (run.out / "samples").exists() is (trace != "off")
    assert (run.out / "transcripts").exists() is (trace in {"transcripts", "full"})
    if trace in {"transcripts", "full"}:
        transcript = json.loads(next((run.out / "transcripts").glob("*.json")).read_text())
        (turn,) = transcript["turns"]
        assert turn["messages"] == [{"role": "user", "content": "x"}]
        assert (turn["disposition"], turn["trainable"]) == ("delivered", True)
        assert not list((run.out / "transcripts").glob("*.partial"))
        assert turn["reply"]["parsed"]["content"] is not None
    if trace == "off":
        return

    rows = [
        json.loads(line)
        for path in (run.out / "samples").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert rows
    for row in rows:
        assert ("prompt_ids" in row) is (trace == "full")
        assert ("prompt_tokens" in row) is (trace != "full")
        # Text-only chunks are reconstructible from prompt_ids, so they are not stored.
        assert "prompt_input" not in row


async def test_a_preparation_failure_is_raised_and_close_still_runs(rl_run, caplog):
    run, trainer, runtime = rl_run()

    async def failed_prepare():
        raise ValueError("primary preparation failure")

    async def failed_close():
        raise RuntimeError("secondary cleanup failure")

    runtime.prepare = failed_prepare
    runtime.close = failed_close
    with pytest.raises(RuntimeExecutionFailure, match="primary preparation failure"):
        await asyncio.wait_for(run.execute(), timeout=5)
    assert "secondary cleanup failure" in caplog.text
    assert trainer.closed


@pytest.mark.parametrize("error", [KeyError("dataset index"), ValueError("bad job"), OSError("io")])
async def test_an_unknown_preparation_error_names_its_phase(rl_run, error):
    run, trainer, runtime = rl_run()

    async def failed_prepare():
        raise error

    runtime.prepare = failed_prepare
    with pytest.raises(RuntimeExecutionFailure, match=r"prepare\(\)") as captured:
        await asyncio.wait_for(run.execute(), timeout=5)
    assert captured.value.__cause__ is error
    assert runtime.closed and trainer.closed


async def test_checkpoint_manifest_survives_rollout_logging_failure(rl_run, monkeypatch):
    from loops_cookbook.rl import run as run_module

    append = run_module._append_json

    def fail_rollout_log(path, record):
        if path.name == "rollouts.jsonl":
            raise OSError("rollout log failed")
        append(path, record)

    monkeypatch.setattr(run_module, "_append_json", fail_rollout_log)
    run, _, _ = rl_run(KeepingTrainer(), max_steps=1)
    outcome = await asyncio.wait_for(run.execute(), timeout=5)
    state = json.loads((run.out / "checkpoints" / "step-0.json").read_text())
    assert state["trainer_state_path"] == "bt://loops:test/state/0"
    assert state["next_step"] == 1
    # The update was applied and checkpointed before the record was written, so a
    # failed append must not throw it away -- and the step has to agree with the
    # manifest, or a resume repeats work that is already done.
    assert outcome.ok and outcome.next_step == state["next_step"]


def test_resume_requires_committed_manifest(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="manifest"):
        Manifest.read(path)


def test_a_manifest_from_another_recipe_is_refused(tmp_path):
    path = tmp_path / "sft.json"
    path.write_text(
        json.dumps(
            {
                "method": "sft",
                "trainer_state_path": "bt://loops:r/state",
                "next_step": 1,
                "digest": "d",
                "schedule": {},
                "run_id": "r",
                "run_directory": str(tmp_path),
            }
        )
    )
    with pytest.raises(ValueError, match="manifest"):
        Manifest.read(path)


async def test_assemble_builds_the_trainer_to_restore_the_manifests_remote_state(
    config, tmp_path, monkeypatch, loops_service
):
    remote_path = "bt://loops:source-run/trainer-state/step-7"
    Manifest(
        trainer_state_path=remote_path,
        next_step=8,
        next_group_index=21,
        policy_version=8,
        shuffle_seed=0,
        contract={},
        run_id="source-run",
        run_directory=str(tmp_path / "source-run"),
    ).write(tmp_path, 7)
    monkeypatch.setattr(recipe, "import_factory", lambda path: unused)
    manifest = manifest_path(tmp_path, 7)
    run = recipe.assemble(config(resume_from=str(manifest)))
    # The config still names the manifest; the trainer got the state it points at.
    assert run.config.training.resume_from == str(manifest)
    assert run.out == tmp_path / "source-run" and run.run_id == "source-run"
    loops_service.client.restored = 8
    await run.trainer.start()
    assert loops_service.client.loads == [("load_state_with_optimizer", remote_path)]
    assert run.trainer.resumed_step == 8


@pytest.fixture
def first_manifest(rl_run):
    """``await first_manifest(**settings)``: run one kept step; its manifest's path."""

    async def run_one(**settings):
        first, _, _ = rl_run(KeepingTrainer(), max_steps=1, **settings)
        assert (await first.execute()).ok
        return manifest_path(first.out, 0)

    return run_one


def resuming():
    """A memory trainer back at the policy version a one-step manifest records."""
    trainer = MemoryTrainer()
    trainer.policy_version = 1
    return trainer


async def test_the_contract_records_the_weights_a_run_started_from(first_manifest):
    weights = "bt://loops:source-run/weights/step-249"
    manifest = await first_manifest(init_from=weights)
    assert json.loads(manifest.read_text())["contract"]["training"]["init_from"] == weights


def test_assemble_uses_a_trainer_it_is_given_as_is(config, monkeypatch):
    monkeypatch.setattr(recipe, "import_factory", lambda path: unused)
    trainer = MemoryTrainer()
    assert recipe.assemble(config(), trainer=trainer).trainer is trainer


async def test_a_checkpoint_records_where_to_resume_and_what_a_resume_must_match(
    first_manifest,
):
    state = json.loads((await first_manifest()).read_text())
    assert state["next_step"] == 1 and state["next_group_index"] == 1
    contract = state["contract"]
    assert contract["tasks"] == [[TASK.id, TASK.digest]]
    assert contract["runtime_config_digest"].startswith("sha256:")
    assert contract["runtime_file_digests"] == {}
    assert contract["producers"] == {"main": True}
    assert {"harbor_config_digest", "task_ids", "reward_key"}.isdisjoint(contract)


@pytest.mark.parametrize(
    ("changed", "first_group"),
    [({}, "g0000001-"), ({"shuffle_seed": 42}, "g0000000-")],
    ids=["same order", "another order"],
)
async def test_a_resume_continues_the_task_order_or_restarts_one_that_changed(
    rl_run, first_manifest, changed, first_group
):
    """Another task order leaves the weights loadable and only makes the cursor
    meaningless, so the run starts that order from its beginning."""
    manifest = await first_manifest()
    # max_steps counts the run's updates, the one before the resume included.
    resumed, _, runtime = rl_run(resuming(), max_steps=2, resume_from=str(manifest), **changed)
    outcome = await resumed.execute()
    assert outcome.ok and (outcome.steps, outcome.next_step) == (1, 2)
    assert all(r.identity.group_id.startswith(first_group) for r in runtime.samples)


async def test_a_resume_with_no_update_left_completes_without_training(
    rl_run, first_manifest, caplog
):
    """As in Tinker, max_steps counts the run's updates: a resume past it trains nothing,
    and exits 0 so the same command run again completes again."""
    manifest = await first_manifest()

    class Unstarted(MemoryTrainer):
        async def start(self):
            raise AssertionError("a run with no update left started its trainer")

    resumed, _, _ = rl_run(Unstarted(), max_steps=1, resume_from=str(manifest))
    with caplog.at_level(logging.INFO):
        outcome = await resumed.execute()
    assert outcome.ok and (outcome.steps, outcome.next_step) == (0, 1)
    assert finished_message(1, *finished_hint(1)) in caplog.text


async def test_a_resume_without_a_step_limit_trains_on(rl_run, first_manifest):
    """max_steps = 0 is no limit, so a resume always has updates left."""
    manifest = await first_manifest()

    class Started(MemoryTrainer):
        async def start(self):
            raise RuntimeError("the trainer started")

    resumed, _, _ = rl_run(Started(), max_steps=0, resume_from=str(manifest))
    with pytest.raises(RuntimeError, match="the trainer started"):
        await asyncio.wait_for(resumed.execute(), timeout=5)


async def test_a_resume_with_no_update_left_still_reports_a_runtime_it_could_not_close(
    rl_run, first_manifest
):
    manifest = await first_manifest()
    resumed, _, runtime = rl_run(max_steps=1, resume_from=str(manifest))

    async def failed_close():
        raise RuntimeError("sandbox cleanup incomplete")

    runtime.close = failed_close
    outcome = await asyncio.wait_for(resumed.execute(), timeout=5)
    assert outcome.status == "unclean" and outcome.steps == 0 and outcome.next_step == 1


async def test_a_resume_into_another_base_model_is_refused(rl_run, first_manifest):
    """Weights trained for another base model cannot load at all, even when the run has
    no update left: `loops run` refuses what `loops check` refuses."""
    manifest = await first_manifest()
    incompatible, _, _ = rl_run(max_steps=1, resume_from=str(manifest), base_model="other")
    with pytest.raises(ValueError, match="cannot load"):
        await incompatible.execute()


async def test_a_run_directory_resumes_the_branch_written_last_not_the_furthest(rl_run):
    """The first process checkpoints steps 0 to 2; a resume from step 0 trains step 1
    again on a branch of its own. The run directory resumes that branch."""
    first, _, _ = rl_run(KeepingTrainer(), max_steps=3)
    assert (await first.execute()).ok
    assert {path.name for path in (first.out / "checkpoints").glob("step-*.json")} >= {
        "step-0.json",
        "step-2.json",
    }
    again = KeepingTrainer()
    again.policy_version = 1
    branch, _, _ = rl_run(again, max_steps=2, resume_from=str(manifest_path(first.out, 0)))
    assert (await branch.execute()).ok
    assert Manifest.resume_from(str(first.out)).next_step == 2


async def test_only_update_scoped_names_are_offered_the_optimizer_step_axis(rl_run):
    """A gauge a heartbeat also reports cannot be plotted per update."""
    events = []
    run, _, _ = rl_run(max_steps=1)
    run.add_step_sink(events.append)
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok

    (event,) = events
    assert event.step == 0
    # Produced by the update alone.
    assert {"reward/mean", "train/groups", "r3/routed_datums"} <= event.step_scoped
    # Also produced by the heartbeat, so it keeps the time axis.
    assert not {name for name in event.gauges if name.startswith("pipeline/")} & event.step_scoped


async def test_a_nonfinite_value_in_a_step_record_stops_the_run(rl_run, monkeypatch):
    """`allow_nan=False` is a correctness gate, not a formatting preference.

    A full disk is environmental and the update stands; a value that cannot be
    serialised means a number went wrong, and continuing would train for days on
    a run whose own record the acceptance audit rejects.
    """
    from loops_cookbook.rl import run as run_module

    monkeypatch.setattr(run_module, "trajectory_metrics", lambda _: {"x": float("nan")})
    run, _, _ = rl_run(max_steps=1)
    with pytest.raises(ValueError, match="JSON compliant"):
        await asyncio.wait_for(run.execute(), timeout=5)


async def test_a_cancellation_while_recording_a_step_is_not_swallowed(rl_run, monkeypatch):
    """The guard is for the environment failing, never for the run being stopped."""
    from loops_cookbook.rl import run as run_module

    def stop(path, record):
        raise asyncio.CancelledError()

    monkeypatch.setattr(run_module, "_append_json", stop)
    run, _, _ = rl_run(max_steps=1)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run.execute(), timeout=5)


async def test_the_snapshot_goes_quiet_before_the_pipeline_begins_draining(rl_run, monkeypatch):
    """Pipeline shutdown takes up to shutdown_drain_seconds, and observers poll
    throughout it. Frozen counts beside a climbing clock read as a live run."""
    from loops_cookbook.rl import pipeline as pipeline_module

    readings = []
    original = pipeline_module.Pipeline._drain

    async def watched(self):
        readings.append(run.snapshot())
        return await original(self)

    monkeypatch.setattr(pipeline_module.Pipeline, "_drain", watched)
    run, _, _ = rl_run(max_steps=1)
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok
    assert readings and all(reading is None for reading in readings), readings


def test_a_run_has_no_page_until_its_trainer_has_an_id(config):
    """Shown the moment provisioning finishes, which is when it becomes useful."""
    run = Run(config(), trainer=MemoryTrainer(), runtime_factory=unused, renderer_factory=unused)
    assert run.loops_url is None
    run.trainer.client = SimpleNamespace(run_id="run-1")
    assert run.loops_url == "https://app.baseten.co/training/loop/run-1"


async def test_an_interruption_announces_the_shutdown_rather_than_reporting_an_error(
    rl_run, caplog
):
    """Ctrl-C is not a failure, and what a reader needs next is how long teardown
    takes, that its noise is expected, and how unfinished cleanup will show."""
    import logging

    caplog.set_level(logging.INFO)
    run, _, _ = rl_run(shutdown_drain_seconds=90)
    task = asyncio.create_task(run.execute())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (line,) = [r for r in caplog.records if "Interrupted" in r.getMessage()]
    assert line.levelno == logging.WARNING, "a deliberate stop is not an error"
    message = line.getMessage()
    assert "90s" in message, "say how long the drain can take"
    assert "not confirmed is logged" in message, "say how unfinished cleanup shows"
    # The record on disk names the exception even though it stringifies to nothing.
    assert json.loads((run.out / "failure.json").read_text())["error"] == "CancelledError"


async def test_the_loops_run_id_is_written_down_not_only_logged(rl_run):
    """Asking "which run do I reuse?" an hour later should not mean scrolling back
    through a teardown, so it lands beside the run's other artifacts."""
    run, trainer, _ = rl_run(max_steps=1)
    trainer.client = SimpleNamespace(run_id="run-1")
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok
    record = json.loads((run.out / "run.json").read_text())
    assert record["run_id"] == "run-1"
    assert record["url"] == "https://app.baseten.co/training/loop/run-1"
    assert record["started_at"] > 0


async def test_the_snapshot_says_when_the_trainer_is_working(rl_run):
    """A gauge, not a notice, so W&B and the progress file record it too -- on a
    long collection phase "has it started training yet" is the question.

    Read from inside the commit, because that is the only window it is true in: by
    the time the step event fires, the update is done and the flag is back down.
    """
    inside, after = [], []
    run, trainer, _ = rl_run(max_steps=1)
    commit = trainer.commit

    async def watched(step, **stats):
        inside.append(run.snapshot())
        return await commit(step, **stats)

    trainer.commit = watched
    run.add_step_sink(lambda event: after.append(run.snapshot()))
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok

    (during,) = inside
    assert during["progress/training"] == 1.0
    assert during["progress/policy_version"] >= 0
    # And down again once the update is applied, so the region stops saying it.
    (settled,) = after
    assert settled["progress/training"] == 0.0


async def test_runtime_file_digests_cover_directories_and_resolve_against_the_config(
    rl_run, tmp_path
):
    from loops_cookbook.rl.run import _content_digest

    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "a.json").write_text("{}")
    run, _, _ = rl_run(
        KeepingTrainer(),
        config_dir=tmp_path,
        max_steps=1,
        **{"rollout_runtime.files": {"tasks": "tasks"}},
    )
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok
    contract = json.loads(manifest_path(run.out, 0).read_text())["contract"]
    assert contract["runtime_file_digests"] == {"tasks": _content_digest(tmp_path / "tasks")}
    before = _content_digest(tmp_path / "tasks")
    (tmp_path / "tasks" / "a.json").write_text('{"changed": true}')
    assert before is not None and _content_digest(tmp_path / "tasks") != before
    assert _content_digest(tmp_path / "missing") is None


async def test_shutdown_cancels_outstanding_attempts_cleans_them_up_then_closes(rl_run):
    """Shutdown order: stop admitting, cancel, clean up every attempt, then close()."""
    events: list[tuple[str, object]] = []
    pipelines = []

    class Outstanding(MemoryRuntime):
        quota = 0

        async def sample(self, attempt):
            if self.quota <= 0:
                events.append(("hung", attempt.spec.identity.attempt_id))
                await asyncio.Event().wait()
            self.quota -= 1
            return await super().sample(attempt)

        async def abort(self, attempt, reason):
            events.append(("abort", (attempt.spec.identity.attempt_id, reason)))

        async def close(self):
            (pipeline,) = pipelines
            events.append(("close", pipeline.capacity.snapshot()))
            await super().close()

    run, _, runtime = rl_run(runtime=Outstanding(), max_steps=3)
    runtime.quota = run.config.training.groups_per_step * run.config.training.group_size
    run.add_step_sink(lambda event: pipelines.append(run._pipeline))
    execution = asyncio.create_task(run.execute())
    while not (pipelines and any(kind == "hung" for kind, _ in events)):
        await asyncio.sleep(0.01)
    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(execution, timeout=10)

    hung = {value for kind, value in events if kind == "hung"}
    aborted = {value[0]: value[1] for kind, value in events if kind == "abort"}
    assert hung and set(aborted) == hung
    assert set(aborted.values()) == {AttemptAbortReason.CANCELLED}
    kinds = [kind for kind, _ in events]
    assert kinds.index("close") > max(i for i, kind in enumerate(kinds) if kind == "abort")
    closing = events[-1][1]
    assert closing["slots/sampling"] == 0 and closing["slots/verification"] == 0


async def test_each_declared_reasoning_effort_gets_its_own_renderer_from_the_factory(
    config, monkeypatch
):
    from loops_cookbook.rl import run as run_module

    built, probed = [], []

    def factory(settings):
        built.append(dict(settings.model.renderer_chat_template_kwargs))
        return Renderer()

    monkeypatch.setattr(run_module, "_warn_without_images", probed.append)
    settings = config(max_steps=1)
    settings = settings.model_copy(
        update={
            "model": settings.model.model_copy(
                update={
                    "renderer_chat_template_kwargs": {"enable_thinking": True, "x": 1},
                    "reasoning_efforts": {"none": {"enable_thinking": False}, "high": {}},
                }
            )
        }
    )
    run = Run(
        settings,
        MemoryTrainer(),
        runtime_factory=lambda s, *, context: MemoryRuntime(),
        renderer_factory=factory,
    )
    _runtime, _tasks, renderers = await run.preflight()
    assert built == [
        {"enable_thinking": True, "x": 1},  # the run renderer
        {"enable_thinking": True, "x": 1},  # high: nothing overridden
        {"enable_thinking": False, "x": 1},  # none
    ]
    assert sorted(renderers.by_effort) == ["high", "none"]
    assert len(probed) == 3  # every renderer is probed for images at preflight
    assert renderers.pool is not None  # conversations render off the event loop


async def test_a_committed_reward_trains_even_when_finish_fails(rl_run):
    """finish() is cleanup only: its failure is counted, never taken out of training."""

    class FailingFinish(MemoryRuntime):
        async def finish(self, attempt, rollout, disposition):
            raise RuntimeError("teardown failed after the reward was committed")

    trainer, runtime = MemoryTrainer(), FailingFinish()
    run, trainer, runtime = rl_run(
        trainer, runtime, max_steps=1, max_consecutive_rollout_failures=100
    )
    outcome = await asyncio.wait_for(run.execute(), timeout=5)
    assert outcome.steps == 1
    (rows,) = trainer.rows
    # Every graded rollout trained: a group with no reward variance would not have.
    assert len(rows) == len(runtime.samples) == run.config.training.group_size


class TwoProducerRuntime(MemoryRuntime):
    """Each episode generates once as ``main`` and once as ``critic``."""

    async def sample(self, attempt):
        for producer in ("main", "critic"):
            conversation = attempt.services.model_access.llm(producer).conversation()
            await conversation.generate(messages=[{"role": "user", "content": producer}])
        return EpisodeResult()


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_no_worker_keeps_a_finished_attempt_while_it_waits_for_its_next_rollout(rl_run, mode):
    """A finished attempt -- its conversations, lease and model access, megabytes for a
    long agent rollout -- is released once its rollout is finished. A pipeline worker
    must not hold the last one it handled while it samples or waits for the next, which
    can take as long as a whole rollout."""
    import gc
    import weakref

    finished = []  # weak references to each finished attempt's model access
    held = []

    def alive():
        return sum(ref() is not None for ref in finished)

    class Watched(MemoryRuntime):
        async def sample(self, attempt):
            # A finish still settling (its cleanup task) holds its attempt by right, so
            # wait for every finish to settle: each takes a few turns of the loop, and
            # a worker's reference never goes. The bound is a backstop, not a clock.
            for _ in range(1_000):
                if not alive():
                    break
                await asyncio.sleep(0)
            else:
                gc.collect()  # anything still alive is not merely awaiting the collector
            held.append(alive())
            return await super().sample(attempt)

        async def finish(self, attempt, rollout, disposition):
            finished.append(weakref.ref(attempt.services.model_access))

    runtime = Watched()
    run, _, runtime = rl_run(runtime=runtime, mode=mode, max_steps=3)
    # A backstop against a hang, generous for a loaded CI runner; nothing waits on it.
    outcome = await asyncio.wait_for(run.execute(), timeout=120)
    assert outcome.ok and outcome.steps == 3
    assert len(finished) >= 8 and held and not any(held)


@pytest.mark.parametrize(
    "trainable", [("main",), ("critic",), ("main", "critic")], ids=lambda t: "+".join(t)
)
async def test_a_declared_producer_is_sampled_and_trains_only_when_declared_trainable(
    rl_run, trainable
):
    producers = {name: {"trainable": name in trainable} for name in ("main", "critic")}
    run, trainer, runtime = rl_run(runtime=TwoProducerRuntime(), max_steps=1, producers=producers)
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok
    for rollout in runtime.samples:  # every rollout recorded both producers
        assert {s.producer for s in rollout.samples} == {"main", "critic"}
        assert {s.producer for s in rollout.samples if s.trainable} == set(trainable)
    (rows,) = trainer.rows
    assert len(rows) == len(runtime.samples) * len(trainable)  # one chain per trainable producer


async def test_a_transcript_write_cut_short_leaves_no_transcript(rl_run, monkeypatch):
    """Whole or absent: a reader never finds half an attempt's transcript."""

    from loops_cookbook.llm.transcript import SealedTranscript

    run, _, _ = rl_run(trace="full")
    wiring = run._lease_wiring()
    assert wiring.records_transcripts and wiring.on_transcript is not None
    written = Path.write_text

    def cut_short(path, text, *args, **kwargs):
        written(path, text[: len(text) // 2], *args, **kwargs)
        raise OSError("no space left on device")

    monkeypatch.setattr(Path, "write_text", cut_short)
    identity = AttemptIdentity("run", "g", "r", "a")
    await wiring.on_transcript(identity, SealedTranscript((), {}, {}))
    assert not (run.out / "transcripts" / "a.json").exists()
    assert run._trace_gauges()["trace/unwritten_transcripts"] == 1


@pytest.mark.parametrize("fails", [False, True], ids=["lands", "fails"])
async def test_a_cancelled_transcript_write_is_counted_only_when_it_failed(
    rl_run, monkeypatch, fails
):
    """A cancelled write runs to its end: one that lands is written, one that fails is
    counted, though the cancellation is what its caller sees."""
    import threading

    from loops_cookbook.llm.transcript import SealedTranscript

    run, _, _ = rl_run(trace="full")
    wiring = run._lease_wiring()
    assert wiring.on_transcript is not None
    entered, release = threading.Event(), threading.Event()
    written = Path.write_text

    def held(path, text, *args, **kwargs):
        entered.set()
        release.wait(5)
        if fails:
            raise OSError("no space left on device")
        return written(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", held)
    identity = AttemptIdentity("run", "g", "r", "a")
    writing = asyncio.ensure_future(wiring.on_transcript(identity, SealedTranscript((), {}, {})))
    await asyncio.to_thread(entered.wait, 5)
    writing.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await writing
    assert (run.out / "transcripts" / "a.json").exists() is not fails
    assert run._trace_gauges()["trace/unwritten_transcripts"] == int(fails)


async def test_a_sample_row_cut_short_leaves_the_rows_after_it_readable(rl_run, monkeypatch):
    """Only the row that failed is lost: the next starts a line of its own."""
    from support.builders import make_sample

    from loops_cookbook.rl import run as run_module

    run, _, _ = rl_run(trace="samples")
    wiring = run._lease_wiring()
    appended = run_module._append_json
    calls = []

    def append(path, record, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record)[:20])
            raise OSError("no space left on device")
        appended(path, record, **kwargs)

    monkeypatch.setattr(run_module, "_append_json", append)
    identity = AttemptIdentity("run", "g", "r", "a")
    for n in range(3):
        await wiring.on_sample(identity, make_sample(f"s{n}", (1,), (2,), (-0.1,)))
    lost, *rows = (run.out / "samples" / "a.jsonl").read_text().splitlines()
    assert [json.loads(row)["sample_id"] for row in rows] == ["s1", "s2"]
    assert len(lost) == 20
    assert run._trace_gauges()["trace/unwritten_samples"] == 1


async def test_a_transcript_is_written_off_the_event_loop(rl_run, monkeypatch):
    """A multi-megabyte write must not stall every gateway handler and attempt."""
    import threading

    from loops_cookbook.llm.transcript import SealedTranscript

    run, _, _ = rl_run(trace="full")
    wiring = run._lease_wiring()
    assert wiring.on_transcript is not None
    writers = []
    written = Path.write_text

    def spy(path, text, *args, **kwargs):
        writers.append(threading.current_thread())
        return written(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", spy)
    await wiring.on_transcript(AttemptIdentity("run", "g", "r", "a"), SealedTranscript((), {}, {}))
    assert writers and threading.main_thread() not in writers
    assert json.loads((run.out / "transcripts" / "a.json").read_text())["attempt_id"] == "a"


@contextlib.asynccontextmanager
async def _default_executor_held():
    """Every thread of the loop's default executor busy until the block exits, as a
    large batch's chunk POSTs and long polls held it in the SDK."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    released = threading.Event()
    busy = loop.run_in_executor(None, released.wait)
    try:
        yield
    finally:
        released.set()
        await busy


@pytest.mark.parametrize("record", ["sample", "transcript", "batch"])
async def test_a_run_records_while_the_default_executor_is_full(rl_run, record):
    """A run's own writes never wait on the loop's default executor, which the SDK, a
    provider's client and every new connection's host lookup share: a rollout that
    waits there to record a sample stops sampling until something else lets go."""
    from support.builders import make_sample

    from loops_cookbook.llm.transcript import SealedTranscript
    from loops_cookbook.rl.run import _record_batch

    run, _, _ = rl_run(trace="full")
    wiring = run._lease_wiring()
    identity = AttemptIdentity("run", "g", "r", "a")
    writes = {
        "sample": (
            lambda: wiring.on_sample(identity, make_sample("s", (1, 2), (3,), (-0.1,))),
            run.out / "samples" / "a.jsonl",
        ),
        "transcript": (
            lambda: wiring.on_transcript(identity, SealedTranscript((), {}, {})),
            run.out / "transcripts" / "a.json",
        ),
        "batch": (
            lambda: _record_batch(
                run.out / "batches" / "step-0.json", 0, 1, [], [long_row(4)], None
            ),
            run.out / "batches" / "step-0.json",
        ),
    }
    write, written = writes[record]
    async with _default_executor_held():
        # Waited on, not cancelled: a write is waited out through a cancellation.
        writing = asyncio.ensure_future(write())
        await asyncio.wait({writing}, timeout=5)
        assert writing.done() and written.exists()
    await writing


async def test_a_cancelled_transcript_write_is_over_only_once_its_thread_is(rl_run, monkeypatch):
    """The unsealed copy a cancelled seal writes next is the one left on disk."""
    import threading

    from loops_cookbook.llm.transcript import SealedTranscript

    run, _, _ = rl_run(trace="full")
    wiring = run._lease_wiring()
    assert wiring.on_transcript is not None
    entered, release, landed = threading.Event(), threading.Event(), threading.Event()
    written = Path.write_text

    def held(path, text, *args, **kwargs):
        if entered.is_set():
            return written(path, text, *args, **kwargs)
        entered.set()
        release.wait(5)
        try:
            return written(path, text, *args, **kwargs)
        finally:
            landed.set()

    monkeypatch.setattr(Path, "write_text", held)
    identity = AttemptIdentity("run", "g", "r", "a")
    first = asyncio.ensure_future(wiring.on_transcript(identity, SealedTranscript((), {}, {})))
    await asyncio.to_thread(entered.wait, 5)
    first.cancel()

    async def then_unsealed():
        with contextlib.suppress(asyncio.CancelledError):
            await first
        await wiring.on_transcript(identity, SealedTranscript((), {}, {}, sealed=False))

    after = asyncio.ensure_future(then_unsealed())
    await asyncio.sleep(0.05)
    release.set()
    await after
    await asyncio.to_thread(landed.wait, 5)
    await asyncio.sleep(0.05)  # the first write's replace, if it still has one to do
    assert json.loads((run.out / "transcripts" / "a.json").read_text())["sealed"] is False


async def test_a_steps_batch_is_prepared_in_slices_on_the_event_loop(rl_run, monkeypatch):
    """Scoring a large batch of long rollouts takes seconds of CPU; attempts, gateway
    calls and heartbeats wait at most a slice of it, never the whole. Each unit here
    takes 10 ms on the slices' clock, which only the work advances, so the default
    budget gives the loop a turn after every one, whatever the machine's speed."""
    import threading
    from functools import partial
    from itertools import pairwise

    from loops_cookbook.rl import run as run_module
    from loops_cookbook.rl.slices import in_slices

    scorers, seen, turns, cpu = [], [], 0, 0.0
    score_batch = run_module.score_batch

    def scored_in_slices(groups):
        nonlocal cpu
        scorers.append(threading.current_thread())
        for _ in range(5):
            seen.append(turns)
            cpu += 0.01  # a rollout's worth of CPU
            yield
        return (yield from score_batch(groups))

    async def count():
        nonlocal turns
        while True:
            turns += 1
            await asyncio.sleep(0)

    monkeypatch.setattr(run_module, "score_batch", scored_in_slices)
    monkeypatch.setattr(run_module, "in_slices", partial(in_slices, clock=lambda: cpu))
    run, _, _ = rl_run(max_steps=1)
    counter = asyncio.create_task(count())
    try:
        outcome = await asyncio.wait_for(run.execute(), timeout=5)
    finally:
        counter.cancel()
    assert outcome.ok and outcome.steps == 1
    # On the loop, not in a thread, where it would hold the GIL the loop needs.
    assert scorers == [threading.main_thread()]
    # The loop turned between every two slices.
    assert all(later > earlier for earlier, later in pairwise(seen)), seen


def long_row(tokens):
    from baseten.loops.models import Datum, EncodedTextChunk, ModelInput, TensorData

    from loops_cookbook.rl.grpo import TrainingRow

    ids = list(range(tokens))
    datum = Datum(
        model_input=ModelInput(chunks=[EncodedTextChunk(tokens=ids)]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=ids, dtype="int64", shape=[tokens]),
            "advantages": TensorData(data=[0.25] * tokens, dtype="float32", shape=[tokens]),
            "logprobs": TensorData(data=[-0.5] * tokens, dtype="float32", shape=[tokens]),
        },
    )
    return TrainingRow(datum, ((0, tokens),), (0,))


@pytest.mark.parametrize(
    "value",
    [
        {"a": 1, "b": [1.5, None, "x", True], "c": {}, "d": []},
        {"long": list(range(20_000)), "nested": [{"k": [1, 2]}, [3, [4]], ()]},
        [{"row": i, "data": [0.25] * 9_000} for i in range(3)],
        "a string",
    ],
)
def test_json_written_in_slices_is_the_text_json_dumps_writes(value):
    from loops_cookbook.rl.run import _dump_in_slices
    from loops_cookbook.rl.slices import whole

    parts = []
    whole(_dump_in_slices(value, parts.append))
    assert "".join(parts) == json.dumps(value, allow_nan=False)


def test_a_long_row_is_encoded_a_bounded_piece_at_a_time():
    """A 100k-token row is megabytes of text: tens of milliseconds as one call."""
    from loops_cookbook.rl.run import _dump_in_slices, datum_record

    parts, pieces = [], [0]
    work = _dump_in_slices(datum_record(long_row(100_000).datum), parts.append)
    for _ in work:
        pieces.append(sum(map(len, parts)) - sum(pieces))
    assert sum(map(len, parts)) > 1_000_000 and max(pieces) < 100_000


async def test_a_batch_is_encoded_on_the_loop_and_written_from_a_thread(tmp_path, monkeypatch):
    """A thread encoding would hold the GIL the loop needs; a write on the loop would
    block it on a slow disk or mount."""
    import threading

    from loops_cookbook.rl import run as run_module

    encoders, writers = [], []
    datum_record, opened = run_module.datum_record, Path.open

    def encoding(datum):
        encoders.append(threading.current_thread())
        return datum_record(datum)

    class Spied:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            writers.append(threading.current_thread())
            self.handle.close()

        def write(self, text):
            writers.append(threading.current_thread())
            return self.handle.write(text)

    rows = [long_row(10) for _ in range(2)]
    groups = [SimpleNamespace(members=[SimpleNamespace(identity=SimpleNamespace(rollout_id="r"))])]
    path = tmp_path / "batches" / "step-0.json"
    path.parent.mkdir()
    path.with_suffix(".json.partial").write_text("left by a run that crashed")
    monkeypatch.setattr(run_module, "datum_record", encoding)
    monkeypatch.setattr(Path, "open", lambda path, *a, **k: Spied(opened(path, *a, **k)))
    await run_module._record_batch(path, 0, 1, groups, rows, None)
    assert encoders == [threading.main_thread()] * 2
    assert writers and threading.main_thread() not in writers
    assert json.loads(path.read_text()) == {
        "step": 0,
        "policy_version": 1,
        "rollout_ids": ["r"],
        "rows": [{"datum": datum_record(row.datum), "actions": [[0, 10]]} for row in rows],
        "aggregation": None,
    }
    assert list(path.parent.iterdir()) == [path]


async def test_a_cancelled_batch_waits_out_its_write_and_leaves_no_file(tmp_path, monkeypatch):
    """A thread still writing when the step is cancelled would recreate the file the
    cleanup removed, so the cleanup waits for it."""
    import threading

    from loops_cookbook.rl import run as run_module

    started, release = threading.Event(), threading.Event()
    write = run_module._write

    def held(path, text, mode):
        started.set()
        release.wait()
        write(path, text, mode)

    monkeypatch.setattr(run_module, "_write", held)
    path = tmp_path / "batches" / "step-0.json"
    task = asyncio.create_task(run_module._record_batch(path, 0, 1, [], [long_row(4)], None))
    await asyncio.to_thread(started.wait)
    task.cancel()
    try:
        for _ in range(10):
            await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert list(path.parent.iterdir()) == []


async def test_a_batch_that_cannot_be_recorded_leaves_no_file(tmp_path):
    from loops_cookbook.rl.run import _record_batch

    @dataclasses.dataclass
    class Aggregation:
        denominator: float

    path = tmp_path / "batches" / "step-0.json"
    with pytest.raises(ValueError):
        await _record_batch(path, 0, 1, [], [long_row(4)], Aggregation(float("nan")))
    assert not path.exists() and not path.with_suffix(".json.partial").exists()


# On the loop's fake clock: its collections take real time on a large heap, never loop time.
@pytest.mark.looptime
@pytest.mark.parametrize("trace", ["off", "full"])
async def test_a_steps_rows_are_released_once_the_trainer_has_accumulated_them(
    rl_run, trace, monkeypatch
):
    """A step's rows are gigabytes for a large batch of long rollouts. Once accumulated,
    nothing may keep them: not through the commit, the checkpoint or the step record,
    and not while the next batch is collected, which can take as long as a step. Nor
    may the runner keep a trained step's groups, and their rollouts, through it."""
    import gc
    import weakref

    from loops_cookbook.rl.pipeline import Pipeline

    rows = []  # weak references to every row the trainer was given
    batches = []  # and to every list of groups the runner trained on

    def alive(refs):
        gc.collect()
        return sum(ref() is not None for ref in refs)

    class Forgetful(MemoryTrainer):
        """Keeps no row, as the Loops trainer client keeps none once it has sent them."""

        def __init__(self):
            super().__init__()
            self.alive_at = []

        async def accumulate(self, datums):
            rows.extend(weakref.ref(datum) for datum in datums)
            self.rows.append([None] * len(datums))  # only their count, for the report
            await asyncio.sleep(0)

        async def commit(self, step, **stats):
            self.alive_at.append(("commit", alive(rows)))
            return await super().commit(step, **stats)

        async def checkpoint(self, report, *, final=False, keep_sampler_weights=False):
            self.alive_at.append(("checkpoint", alive(rows)))
            return report

    class Batch(list):
        """The runner's list of groups, weakly referenceable."""

    next_batch = Pipeline.next_batch

    async def collect(pipeline):
        trainer.alive_at.append(("collecting", alive(rows), alive(batches)))
        groups = Batch(await next_batch(pipeline))
        batches.append(weakref.ref(groups))
        return groups

    monkeypatch.setattr(Pipeline, "next_batch", collect)
    trainer = Forgetful()
    run, trainer, _ = rl_run(trainer, max_steps=2, trace=trace)
    outcome = await asyncio.wait_for(run.execute(), timeout=5)
    assert outcome.ok and outcome.steps == 2
    assert len(rows) == 8 and [len(step) for step in trainer.rows] == [4, 4]
    assert trainer.alive_at == [
        ("collecting", 0, 0),
        ("commit", 0),
        ("checkpoint", 0),
        ("collecting", 0, 0),
        ("commit", 0),
        ("checkpoint", 0),
    ]
    # What the step reports is unchanged by dropping its rows.
    steps = records(run, "steps")
    assert [step["train/datums"] for step in steps] == [4, 4]
    assert [step["r3/routed_datums"] for step in steps] == [0, 0]


async def test_a_runtime_that_derives_no_rollout_clock_is_refused_one_before_any_attempt(
    rl_run,
):
    from loops_cookbook.runtime.context import RuntimeConfigurationError

    run, trainer, runtime = rl_run(rollout_timeout_seconds=None)
    with pytest.raises(
        RuntimeConfigurationError,
        match=r"\[rollout\] rollout_timeout_seconds is required: this rollout runtime derives no",
    ):
        await asyncio.wait_for(run.execute(), timeout=5)
    assert runtime.samples == [] and runtime.closed and trainer.closed


async def test_a_checkpoint_that_fails_after_publication_leaves_the_step_trained_and_resumable(
    rl_run,
):
    """The update is applied once published; only its artifacts are lost."""

    class FailsToCheckpoint(KeepingTrainer):
        async def checkpoint(self, report, *, final=False, keep_sampler_weights=False):
            if report.step == 1:
                raise OSError("checkpoint store unavailable")
            return await super().checkpoint(report, final=final)

    run, trainer, _ = rl_run(FailsToCheckpoint(scheduled=False), max_steps=3)
    with pytest.raises(OSError, match="checkpoint store unavailable"):
        await asyncio.wait_for(run.execute(), timeout=5)
    recorded = records(run)
    assert {record["disposition"] for record in recorded} == {"trained"}
    assert sorted({record["step"] for record in recorded}) == [0, 1]
    # The state saved on the way out is the one step 1 published, and is named for it.
    manifest = json.loads((run.out / "checkpoints" / "step-1.json").read_text())
    assert manifest["trainer_state_path"] == "state-1-v2"
    assert (manifest["next_step"], manifest["policy_version"]) == (2, trainer.policy_version)


@pytest.mark.parametrize("ending", ["interrupted", "aborted"])
async def test_an_open_ended_run_that_stops_while_collecting_keeps_its_last_update(rl_run, ending):
    # Kept only on the way out, as an open-ended run between cadences is.
    run, trainer, runtime = rl_run(
        KeepingTrainer(scheduled=False),
        max_steps=0,
        trainer_checkpoint_every=5,
        max_consecutive_unusable_groups=1,
    )
    grade = runtime.grade

    async def after_two_updates(attempt, rollout):
        if trainer.policy_version < 2:
            return await grade(attempt, rollout)
        if ending == "aborted":
            return GradeResult(reward=0.0)  # no variance: the run aborts
        await asyncio.Event().wait()  # collecting when the interrupt lands
        raise AssertionError("unreachable")

    runtime.grade = after_two_updates
    running = asyncio.create_task(run.execute())
    if ending == "interrupted":
        async with asyncio.timeout(5):
            while trainer.policy_version < 2:
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
        running.cancel()
    try:
        await asyncio.wait_for(running, timeout=10)
    except asyncio.CancelledError:
        pass
    manifest = json.loads((run.out / "checkpoints" / "step-1.json").read_text())
    assert manifest["trainer_state_path"] == "state-1-v2" and manifest["next_step"] == 2


async def test_a_commit_that_fails_after_its_optimizer_step_saves_no_mislabelled_state(rl_run):
    """Past the optimizer step the trainer holds neither step's state; none is saved."""

    class FailsToPublish(KeepingTrainer):
        async def commit(self, step, **stats):
            if step == 1:
                self.policy_version += 1  # the optimizer stepped
                raise ConnectionError("publish failed")
            return await super().commit(step, **stats)

    run, _, _ = rl_run(FailsToPublish(scheduled=False), max_steps=3)
    with pytest.raises(ConnectionError):
        await asyncio.wait_for(run.execute(), timeout=5)
    assert not (run.out / "checkpoints").exists()


async def test_every_rollouts_record_has_the_same_fields_meaning_the_same_thing(rl_run):
    """policy_version is the oldest version a rollout sampled at, whatever became of it."""

    class Mixed(MemoryRuntime):
        async def sample(self, attempt):
            if attempt.spec.identity.rollout_id.endswith("r3"):
                raise OSError("the sandbox died")  # never samples
            return await super().sample(attempt)

        async def grade(self, attempt, rollout):
            if rollout.identity.rollout_id.endswith("r0"):
                raise PermanentGradingFailure("the verifier wrote no reward")  # sampled
            return await super().grade(attempt, rollout)

    run, _, _ = rl_run(runtime=Mixed(), max_steps=1, max_group_losses=2)
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok
    by_rollout = {record["rollout_id"].rsplit("-", 1)[1]: record for record in records(run)}
    assert len({frozenset(record) for record in by_rollout.values()}) == 1  # one schema
    never, ungraded, trained = by_rollout["r3"], by_rollout["r0"], by_rollout["r1"]
    assert (never["policy_version"], never["sample_ids"], never["step"]) == (None, [], None)
    assert (ungraded["policy_version"], len(ungraded["sample_ids"]), ungraded["step"]) == (
        0,
        1,
        None,
    )
    assert (trained["policy_version"], trained["step"]) == (0, 0)
    assert never["task_digest"] == ungraded["task_digest"] == trained["task_digest"] == TASK.digest


@pytest.mark.parametrize(("limit", "dropped"), [(None, True), (64, False)])
async def test_rollout_metrics_limit_sets_how_many_agent_metrics_a_step_reports(
    rl_run, limit, dropped
):
    """An agent reporting 40 numbers: past the default of 32 the least reported go;
    ``[output] rollout_metrics_limit`` makes room for them all."""

    class Chatty(MemoryRuntime):
        async def grade(self, attempt, rollout):
            result = await super().grade(attempt, rollout)
            return dataclasses.replace(result, metadata={f"m{n:02}": float(n) for n in range(40)})

    options = {} if limit is None else {"rollout_metrics_limit": limit}
    run, _, _ = rl_run(runtime=Chatty(), mode="sync", max_steps=1, **options)
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok
    step = json.loads((run.out / "steps.jsonl").read_text().strip())
    reported = [
        key for key in step if key.startswith("rollout/m") and not key.endswith("_with_unusable")
    ]
    if dropped:
        assert len(reported) == 32 and step["rollout/dropped_keys"] == 8
    else:
        assert len(reported) == 40 and "rollout/dropped_keys" not in step


async def test_a_step_reports_rollout_metrics_with_the_unusable_groups_too(rl_run):
    """A group the policy solves outright has no variance: it is dropped from training,
    and from ``rollout/*``. Its members still count in ``rollout/<key>_with_unusable``,
    so an all-pass rate does not fall as the policy solves more tasks outright."""

    class FirstGroupSolved(MemoryRuntime):
        async def grade(self, attempt, rollout):
            if rollout.identity.group_id.startswith("g0000000"):
                # Every member solves it: no variance, dropped.
                return GradeResult(reward=1.0, metadata={"all_pass": True, "label": "a"})
            result = await super().grade(attempt, rollout)
            return dataclasses.replace(
                result, metadata={"all_pass": result.reward == 1.0, "label": "a"}
            )

    run, _, _ = rl_run(runtime=FirstGroupSolved(), mode="sync", max_steps=1)
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok
    step = json.loads((run.out / "steps.jsonl").read_text().strip())
    size = run.config.training.group_size
    # The trained group: one member of `size` passes. The dropped group: all of them.
    assert step["rollout/all_pass"] == 1 / size
    assert step["rollout/all_pass_with_unusable"] == (1 + size) / (2 * size)
    # Numbers only, as `rollout/*` reports them: labels are not repeated.
    assert not any(key.startswith("rollout/label") and "unusable" in key for key in step)


async def test_a_step_with_no_unusable_group_reports_its_rollout_metrics_with_unusable_too(rl_run):
    """Reported only on steps that dropped a group, the curve would sample just those
    steps, the ones a solved group lifts."""

    class Reporting(MemoryRuntime):
        async def grade(self, attempt, rollout):
            result = await super().grade(attempt, rollout)
            return dataclasses.replace(result, metadata={"all_pass": result.reward == 1.0})

    run, _, _ = rl_run(runtime=Reporting(), mode="sync", max_steps=1)
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok
    step = json.loads((run.out / "steps.jsonl").read_text().strip())
    assert step["reward/unusable_rollouts"] == 0
    assert step["rollout/all_pass_with_unusable"] == step["rollout/all_pass"]


async def test_a_step_reports_its_reward_with_the_unusable_groups_dropped_before_it(rl_run):
    class FirstGroupUniform(MemoryRuntime):
        async def grade(self, attempt, rollout):
            if rollout.identity.group_id.startswith("g0000000"):
                return GradeResult(reward=0.5)  # every member alike: no variance, dropped
            return await super().grade(attempt, rollout)

    run, _, _ = rl_run(runtime=FirstGroupUniform(), mode="sync", max_steps=1)
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok
    step = json.loads((run.out / "steps.jsonl").read_text().strip())
    size = run.config.training.group_size
    # The trained group rewards one member (r1) of `size`; the dropped one gave all 0.5.
    assert step["reward/unusable_rollouts"] == size
    assert step["reward/mean_with_unusable"] == (1 + 0.5 * size) / (2 * size)
    # Within the trained group: one member rewarded 1, the rest 0.
    spread = ((1 / size) * (1 - 1 / size)) ** 0.5
    assert step["reward/group_std_mean"] == pytest.approx(spread)
    assert step["reward/group_std_min"] == step["reward/group_std_max"] == pytest.approx(spread)
    # One update from a fresh trainer: nothing behind it, and its group completed.
    assert step["freshness/lag_mean"] == step["freshness/lag_p90"] == 0.0
    assert step["groups/completion_spread_mean_seconds"] >= 0.0


def test_lag_gauges_report_the_batch_freshness_distribution():
    from types import SimpleNamespace

    from loops_cookbook.rl.run import _lag_gauges

    rollouts = [SimpleNamespace(minimum_policy_version=v) for v in (6, 6, 5, 4, 2)] + [
        SimpleNamespace(minimum_policy_version=None)  # sampled nothing trainable
    ]
    gauges = _lag_gauges(6, rollouts)
    assert gauges["freshness/lag_mean"] == pytest.approx((0 + 0 + 1 + 2 + 4) / 5)
    assert gauges["freshness/lag_p90"] == 4.0
    assert _lag_gauges(6, []) == {}


def test_completion_spread_gauges_measure_how_long_first_finishers_wait():
    from types import SimpleNamespace

    from loops_cookbook.rl.run import _completion_spread_gauges

    groups = [
        SimpleNamespace(first_resolved_at=100.0, completed_at=160.0),
        SimpleNamespace(first_resolved_at=200.0, completed_at=500.0),
        SimpleNamespace(first_resolved_at=None, completed_at=None),  # never resolved
    ]
    gauges = _completion_spread_gauges(groups)
    assert gauges["groups/completion_spread_mean_seconds"] == pytest.approx(180.0)
    assert gauges["groups/completion_spread_max_seconds"] == 300.0
    assert _completion_spread_gauges([]) == {}


def test_a_run_whose_open_file_limit_is_too_low_is_refused_before_it_starts(config):
    import resource

    from loops_cookbook.rl.run import check_open_file_limit
    from loops_cookbook.runtime.context import RuntimeConfigurationError

    big = config(max_concurrent_rollouts=48, max_concurrent_verifiers=64)  # ~120 live
    with pytest.raises(RuntimeConfigurationError, match="ulimit -n"):
        check_open_file_limit(big, limits=(256, 4096))  # macOS's default soft limit
    check_open_file_limit(big, limits=(65536, 65536))
    check_open_file_limit(big, limits=(resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    check_open_file_limit(config(), limits=(256, 4096))  # a small run fits the default
    # One that only evaluates (loops eval) needs room for its eval rollouts alone.
    check_open_file_limit(big, training=False, limits=(256, 4096))


def test_eval_rollouts_count_toward_the_open_file_limit(config):
    from loops_cookbook.rl.run import check_open_file_limit
    from loops_cookbook.runtime.context import RuntimeConfigurationError

    sets = {
        "val": {"rollout_runtime": {"files": {"tasks": "val.jsonl"}}, "max_concurrent_rollouts": 64}
    }
    with pytest.raises(RuntimeConfigurationError, match="eval rollouts"):
        check_open_file_limit(
            config(**{"rollout_runtime.files": {"tasks": "t.jsonl"}, "eval.sets": sets}),
            limits=(256, 4096),
        )


@pytest.mark.parametrize(("verifiers", "refused"), [(1, False), (64, True)])
def test_an_eval_sets_group_graders_count_toward_the_open_file_limit(config, verifiers, refused):
    """A rollout frees its set's slot before its task is group-graded, on one of the
    eval's verifier slots: a set of 5 can have up to 5 more live, as many as verifiers."""
    from loops_cookbook.rl.run import check_open_file_limit
    from loops_cookbook.runtime.context import RuntimeConfigurationError

    sets = {
        "val": {"rollout_runtime": {"files": {"tasks": "val.jsonl"}}, "max_concurrent_rollouts": 5}
    }
    evals_only = config(
        **{
            "rollout_runtime.files": {"tasks": "t.jsonl"},
            "eval.sets": sets,
            "rollout.max_concurrent_verifiers": verifiers,
        }
    )
    if refused:
        with pytest.raises(RuntimeConfigurationError, match="up to 10 attempts"):
            check_open_file_limit(evals_only, training=False, limits=(256, 4096))
    else:
        check_open_file_limit(evals_only, training=False, limits=(256, 4096))


ENDED = "Loops reports run 4w50xmq INACTIVE"


def _breaks(trainer, where):
    """Make ``trainer`` fail at ``where``: a sampling request, or the first update."""

    async def broken(*args, **kwargs):
        raise RuntimeError("connection refused")

    if where == "sampling":
        trainer.sample_async = broken
    else:
        trainer.accumulate = broken


@pytest.mark.parametrize("where", ["sampling", "update"])
async def test_a_loops_run_that_ends_under_the_run_is_reported_as_such(rl_run, where):
    run, trainer, runtime = rl_run()
    _breaks(trainer, where)

    answers = iter([ENDED])

    async def ended():
        # Loops answers once; a second question means the run asked again for what
        # the driver had already reported.
        return next(answers, None)

    trainer.run_ended = ended
    outcome = await asyncio.wait_for(run.execute(), timeout=10)

    assert (outcome.status, outcome.reason, outcome.ok) == ("loops_run_ended", ENDED, False)
    assert json.loads((run.out / "failure.json").read_text())["status"] == "loops_run_ended"
    assert trainer.closed and runtime.closed


@pytest.mark.parametrize("answer", ["live", "unknown"])
async def test_a_failure_under_a_live_or_unknown_loops_run_stays_a_failure(rl_run, answer):
    run, trainer, _ = rl_run()
    _breaks(trainer, "update")

    async def run_ended() -> str | None:
        if answer == "unknown":
            raise OSError("the control plane is unreachable")
        return None

    trainer.run_ended = run_ended
    with pytest.raises(RuntimeError, match="connection refused"):
        await asyncio.wait_for(run.execute(), timeout=10)
    assert json.loads((run.out / "failure.json").read_text())["status"] == "failed"


async def test_a_cancellation_while_loops_answers_keeps_the_failure_record(rl_run):
    run, trainer, _ = rl_run()
    _breaks(trainer, "update")
    asking = asyncio.Event()

    async def slow():
        asking.set()
        await asyncio.sleep(60)

    trainer.run_ended = slow
    task = asyncio.create_task(run.execute())
    await asyncio.wait_for(asking.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert json.loads((run.out / "failure.json").read_text())["status"] == "failed"


async def test_a_resumed_async_run_samples_only_the_rollouts_its_steps_train(rl_run):
    """No rollout is sampled for a step past ``max_steps`` (#99): every one recorded
    trained, none cancelled at shutdown. A resume's limit counts the run's updates, the
    ones before it included; the pipeline's own tests count what a fresh run admits."""
    options = {"mode": "async", "max_concurrent_rollouts": 16, "max_steps_off_policy": 3}
    first, _, _ = rl_run(KeepingTrainer(), **options, max_steps=2)
    assert (await first.execute()).ok
    trainer = KeepingTrainer()
    trainer.policy_version = 1
    run, _, _ = rl_run(
        trainer, **options, max_steps=2, resume_from=str(manifest_path(first.out, 0))
    )
    assert (await run.execute()).ok
    recorded = records(run)
    assert {record["disposition"] for record in recorded} == {"trained"}
    assert len(recorded) == 3 * 4


async def test_a_resume_trains_the_groups_its_checkpoint_left_in_flight_first(rl_run):
    """An async run admits step 1's group before step 0's checkpoint. A resume from that
    checkpoint trains that group, not the one after it: the stream is not skipped."""
    options = {"mode": "async", "max_concurrent_rollouts": 16, "max_steps_off_policy": 3}
    first, _, _ = rl_run(KeepingTrainer(), **options, max_steps=2)
    assert (await first.execute()).ok
    manifest = Manifest.read(manifest_path(first.out, 0))
    assert (manifest.next_group_index, manifest.pending_group_indices) == (2, [1])
    trainer = KeepingTrainer()
    trainer.policy_version = 1
    run, _, runtime = rl_run(
        trainer, **options, max_steps=2, resume_from=str(manifest_path(first.out, 0))
    )
    assert (await run.execute()).ok
    assert {r.identity.group_id for r in runtime.samples} == {"g0000001-task-agent"}
    after = Manifest.read(manifest_path(run.out, 1))
    assert (after.next_group_index, after.pending_group_indices) == (2, [])


def test_a_manifest_without_pending_groups_still_resumes_at_its_cursor(tmp_path):
    """One written before manifests had pending groups: nothing to replay."""
    fields = {
        "trainer_state_path": "bt://loops:run/state/0",
        "next_step": 1,
        "next_group_index": 2,
        "policy_version": 1,
        "shuffle_seed": 0,
        "contract": {},
        "run_id": "run",
        "run_directory": str(tmp_path),
    }
    path = tmp_path / "step-0.json"
    path.write_text(json.dumps(fields))
    assert Manifest.read(path).pending_group_indices == []
    for pending in ([2], [1, 0], [1, 1], [-1]):
        path.write_text(json.dumps({**fields, "pending_group_indices": pending}))
        with pytest.raises(ValueError, match="pending_group_indices"):
            Manifest.read(path)
