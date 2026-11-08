"""Harbor trials of a prepared job through the real driver, capacity broker and lease.

Trials run in the offline directory-backed environment from tests/fixtures/offline.
"""

import asyncio
from types import SimpleNamespace as NS

import harbor_test_agents as agents
import harbor_test_gates as gates
import pytest
from support.configs import read_toml, write_toml

from loops_cookbook.harbor import attempts
from loops_cookbook.runtime.driver import Discard

from .jobs import (
    HILL_CLIMB,
    harbor,
    held_stops,
    live_trials,
    reported,
    separate_steps_task,
    separate_task,
    shared_task,
)


def rewards(root):
    """Every reward file a final verification wrote under ``root``'s trials."""
    return sorted((root / "work" / "trials" / "trials").glob("*/verifier/reward.txt"))


async def test_a_shared_trial_parks_at_final_verification_until_grading(tmp_path):
    r, runtime, task = await harbor(tmp_path, shared_task(tmp_path))
    admitted = await r.driver.sample(task, "g", "g-r1")
    assert r.capacity.snapshot()["slots/verification"] == 0  # not verifying yet
    assert live_trials(runtime) == 1 and rewards(tmp_path) == []  # parked, unverified
    graded = await r.driver.grade(admitted)
    await r.driver.drain()
    assert graded.reward == 1.0
    assert dict(graded.metadata) == {"turns": 1, "said": "ok"}
    assert [path.read_text().strip() for path in rewards(tmp_path)] == ["1"]
    assert live_trials(runtime) == 0
    assert r.capacity.snapshot()["slots/sampling"] == 0


async def test_a_stale_rollout_never_runs_its_final_verifier(tmp_path):
    r, runtime, task = await harbor(tmp_path, shared_task(tmp_path), max_steps_off_policy=0)
    admitted = await r.driver.sample(task, "g", "g-r1")
    r.version[0] = 1
    assert await r.driver.grade(admitted) is Discard.STALE_BEFORE_GRADING
    await r.driver.drain()
    assert rewards(tmp_path) == []
    assert live_trials(runtime) == 0


async def test_a_rollout_records_the_digest_of_the_task_its_trial_loaded(tmp_path):
    """The digest is Harbor's own content digest, from the sampling trial's lock."""
    import json

    r, _runtime, task = await harbor(tmp_path, shared_task(tmp_path))
    graded = await r.rollout("g-r1", task)
    [lock] = (tmp_path / "work" / "trials" / "trials").glob("*/lock.json")
    assert graded.task_digest == json.loads(lock.read_text())["task"]["digest"]
    assert graded.task_digest.startswith("sha256:")


async def test_a_separate_verifier_task_regrades_from_the_saved_trial(tmp_path):
    r, _runtime, task = await harbor(tmp_path, HILL_CLIMB)
    graded = await r.rollout("g-r1", task)
    assert graded.reward == 1.0
    grades = list((tmp_path / "work" / "trials" / "grades").iterdir())
    assert len(grades) == 1 and grades[0].name.endswith("-grade")


@pytest.mark.parametrize("verifier", ["regrade", "per_step"])
async def test_every_environment_of_an_attempt_receives_its_id(tmp_path, monkeypatch, verifier):
    """The agent's and every verifier's: a regrade trial's, or each step's own."""
    from baseline_environment import LocalEnvironment

    received = []
    construct = LocalEnvironment.__init__

    def recording(self, *args, **kwargs):
        received.append((kwargs.get("session_id"), kwargs.get("loops_attempt_id")))
        construct(self, *args, **kwargs)

    monkeypatch.setattr(LocalEnvironment, "__init__", recording)
    source = HILL_CLIMB if verifier == "regrade" else separate_steps_task(tmp_path)
    r, _runtime, task = await harbor(tmp_path, source)
    received.clear()  # preflight probes are their own attempts
    assert (await r.rollout("g-r1", task)).reward == 1.0
    verifiers = {
        session.split("__verifier__")[1] for session, _ in received if "__verifier__" in session
    }
    assert verifiers == ({"trial"} if verifier == "regrade" else {"s1", "s2"})
    (attempt,) = {attempt_id for _, attempt_id in received}
    assert attempt and len(attempt) == 32


async def test_a_multi_step_task_verifies_intermediate_steps_inside_sample(tmp_path):
    r, _runtime, task = await harbor(tmp_path, shared_task(tmp_path, steps=2))
    graded = await r.rollout("g-r1", task)
    assert graded.reward == 1.0
    assert len(agents.RUNS) == 2


async def test_each_steps_verification_holds_verification_capacity_not_sampling(
    tmp_path, monkeypatch
):
    """Intermediate verification inside sample() moves the attempt between phases."""
    import baseline_environment

    seen, holder = [], {}
    run = baseline_environment.LocalEnvironment.exec

    async def observed(self, command, *args, **kwargs):
        if "test.sh" in str(command):  # a step's verifier
            snapshot = holder["r"].capacity.snapshot()
            seen.append((snapshot["slots/sampling"], snapshot["slots/verification"]))
        return await run(self, command, *args, **kwargs)

    monkeypatch.setattr(baseline_environment.LocalEnvironment, "exec", observed)
    r, _runtime, task = await harbor(tmp_path, shared_task(tmp_path, steps=2))
    holder["r"] = r
    assert (await r.rollout("g-r1", task)).reward == 1.0
    assert seen and set(seen) == {(0, 1)}  # every step's verifier: verification, not sampling


async def test_a_multi_step_early_stop_fixes_its_reward_during_sample(tmp_path, monkeypatch):
    import baseline_environment

    task = shared_task(tmp_path, steps=2)
    toml = task / "task.toml"
    tables = read_toml(toml)
    tables["steps"][0]["min_reward"] = 2.0
    write_toml(toml, tables)
    verifications = []
    run = baseline_environment.LocalEnvironment.exec

    async def counted(self, command, *args, **kwargs):
        if "test.sh" in str(command):
            verifications.append(command)
        return await run(self, command, *args, **kwargs)

    monkeypatch.setattr(baseline_environment.LocalEnvironment, "exec", counted)
    r, _runtime, ref = await harbor(tmp_path, task)
    admitted = await r.driver.sample(ref, "g", "g-r1")
    assert len(agents.RUNS) == 1  # step 2 never ran
    before = len(verifications)
    graded = await r.driver.grade(admitted)
    assert graded.reward == 1.0 and len(verifications) == before
    await r.driver.drain()


@pytest.mark.parametrize("verifier", ["shared", "regrade", "per_step"])
async def test_every_sandbox_of_a_trial_is_the_attempts_and_released_with_it(tmp_path, verifier):
    """The agent's, a regrade's verifier, and each step's verifier alike."""
    from support.resources import FakeProvider, episodes, registry_at

    owners = []

    class Recording(FakeProvider):
        async def create(self, intent):
            owners.append((intent.scope.value, intent.owner_id))
            return await super().create(intent)

    provider = Recording()
    registry = registry_at(tmp_path / "state", provider, run_id="run")
    source = {
        "shared": lambda: shared_task(tmp_path),
        "regrade": lambda: HILL_CLIMB,
        "per_step": lambda: separate_steps_task(tmp_path),
    }[verifier]()
    r, _runtime, task = await harbor(
        tmp_path,
        source,
        job={"environment": {"import_path": "support.harbor_sandbox:ProvisionedLocalEnvironment"}},
        resources=episodes(registry),
    )
    graded = await r.rollout("g-r1", task)
    assert graded.reward == 1.0
    await r.driver.drain()
    created = [name for call, name in provider.calls if call == "create"]
    released = [name for call, name in provider.calls if call == "release"]
    expected = {"shared": 1, "regrade": 2, "per_step": 3}[verifier]  # agent + verifiers
    assert len(created) == expected and sorted(released) == sorted(created)
    assert set(owners) == {("episode", graded.identity.attempt_id)}
    assert provider.store == {} and registry.unresolved.results == ()
    await registry.close()


@pytest.mark.parametrize("agent", ["AnswerAgent", "SleepyAgent"], ids=["answers", "times-out"])
@pytest.mark.parametrize(
    ("sandbox", "kills"),
    [("OomKilledSandbox", 2), ("CleanSandbox", 0), ("UnreadableSandbox", None)],
    ids=["oom-killed", "clean", "unreadable"],
)
async def test_a_blaxel_trial_reports_its_sandbox_oom_kills_and_keeps_harbors_outcome(
    tmp_path, sandbox, kills, agent
):
    """A report only: the reward and the agent's own metadata are what Harbor recorded."""
    pytest.importorskip("blaxel")
    from support.blaxel_api import SETTINGS

    environment = {"import_path": f"harbor_test_environments:{sandbox}", "kwargs": {"ttl": "8h"}}
    timeout = 0.2 if agent == "SleepyAgent" else 30
    r, runtime, ref = await harbor(
        tmp_path,
        shared_task(tmp_path, agent_timeout=timeout),
        agent,
        job={"environment": environment},
        config={"provider_deadlines": {"blaxel": SETTINGS.deadlines}},
    )
    graded = await r.rollout("g-r1", ref)
    assert graded.reward == 1.0
    metrics = reported(runtime)
    assert dict(graded.metadata) == {"turns": 1, "said": "ok"}  # the agent's metadata only
    if kills is None:
        assert not any("oom" in name for name in metrics)
    else:
        assert metrics["runtime/harbor/sandbox_oom_kills/count"] == 1
        assert metrics["runtime/harbor/sandbox_oom_kills/max"] == kills
        assert metrics.get("runtime/harbor/oom_killed_trials", 0) == (1 if kills else 0)


async def test_a_daytona_trial_reads_the_runs_providers_daytona(tmp_path):
    """The attempt a sampling trial binds carries the table its environment reads."""
    pytest.importorskip("daytona")
    from support.daytona_api import CREDENTIALS, SETTINGS, FakeApi

    from loops_cookbook.providers.daytona.adapter import DaytonaAdapter

    adapter = DaytonaAdapter(SETTINGS, CREDENTIALS, FakeApi())
    r, _, ref = await harbor(
        tmp_path,
        shared_task(tmp_path),
        job={"environment": {"import_path": "harbor_test_daytona:LocalDaytona"}},
        config={
            "provider_deadlines": {"daytona": adapter.deadlines},
            "provider_capacity": {"daytona": adapter.capacity},
            "provider_settings": {"daytona": adapter.settings},
        },
    )
    assert (await r.rollout("g-r1", ref)).reward == 1.0


async def test_a_trials_sandbox_waits_for_room_to_sample_before_it_starts(tmp_path, monkeypatch):
    """Under backpressure a trial must not start a sandbox that would only sit idle."""
    import baseline_environment

    starts = []
    start = baseline_environment.LocalEnvironment.start

    async def recorded(self, force_build):
        starts.append(self.session_id)
        await start(self, force_build)

    monkeypatch.setattr(baseline_environment.LocalEnvironment, "start", recorded)
    r, _runtime, task = await harbor(tmp_path, shared_task(tmp_path))
    r.admission.value = 0  # downstream pressure: nothing new may sample
    rollout = asyncio.create_task(r.rollout("g-r1", task))
    async with asyncio.timeout(10):
        while r.capacity.snapshot()["admission/waiting"] < 1:
            await asyncio.sleep(0.01)
    assert starts == [] and not rollout.done()  # no sandbox started, none left idle
    r.admission.value = 1000
    r.capacity.notify()
    await asyncio.wait_for(rollout, 30)
    assert len(starts) == 1


async def test_sampling_is_held_while_the_agent_runs_not_during_environment_start(
    tmp_path, monkeypatch
):
    from harbor.trial.hooks import TrialEvent

    from loops_cookbook.harbor import integration

    r, _runtime, task = await harbor(tmp_path, shared_task(tmp_path))
    setup = integration.HookedTrialQueue._setup_hooks
    held = {}

    def recording(self, trial):
        setup(self, trial)
        for event in (TrialEvent.ENVIRONMENT_START, TrialEvent.AGENT_START):

            async def observe(hook_event, event=event):
                held[event] = r.capacity.snapshot()["slots/sampling"]

            trial.add_hook(event, observe)

    monkeypatch.setattr(integration.HookedTrialQueue, "_setup_hooks", recording)
    await r.rollout("g-r1", task)
    assert held == {TrialEvent.ENVIRONMENT_START: 0, TrialEvent.AGENT_START: 1}


async def test_a_regrade_trial_is_never_held_back_by_sampling_admission(tmp_path):
    """Grading runs as fast as verifiers allow; only sampling is backpressured."""
    r, _runtime, task = await harbor(tmp_path, HILL_CLIMB)
    admitted = await asyncio.wait_for(r.driver.sample(task, "g", "g-r1"), 30)
    r.admission.value = 0  # full backpressure: nothing new may be admitted to sample
    graded = await asyncio.wait_for(r.driver.grade(admitted), 30)
    await r.driver.drain()
    assert graded.reward == 1.0


async def test_a_finished_trial_waiting_for_its_regrade_holds_no_trial_slot(tmp_path):
    r, runtime, ref = await harbor(
        tmp_path,
        HILL_CLIMB,
        config={"limits": {"max_concurrent_rollouts": 1, "max_concurrent_verifiers": 1}},
    )
    assert reported(runtime)["runtime/harbor/trial_cap"] == 2
    # Three sampled rollouts wait for grading at once, past the cap of two trials.
    admitted = [await asyncio.wait_for(r.driver.sample(ref, "g", f"g-r{i}"), 30) for i in range(3)]
    graded = [await r.driver.grade(one) for one in admitted]
    assert [one.reward for one in graded] == [1.0] * 3
    await r.driver.drain()


async def test_regrades_verify_as_many_at_once_as_there_are_verification_permits(
    tmp_path, monkeypatch
):
    """Harbor's regrade queue adds no wait of its own inside grade()."""
    import baseline_environment

    verifying, all_in = 0, asyncio.Event()
    run = baseline_environment.LocalEnvironment.exec

    async def held(self, command, *args, **kwargs):
        nonlocal verifying
        if "test.sh)" in str(command):  # a verifier's run, held until all three are in
            verifying += 1
            if verifying == 3:
                all_in.set()
            await all_in.wait()
        return await run(self, command, *args, **kwargs)

    monkeypatch.setattr(baseline_environment.LocalEnvironment, "exec", held)
    r, _runtime, ref = await harbor(
        tmp_path,
        HILL_CLIMB,
        config={"limits": {"max_concurrent_rollouts": 1, "max_concurrent_verifiers": 3}},
    )
    admitted = [await asyncio.wait_for(r.driver.sample(ref, "g", f"g-r{i}"), 30) for i in range(3)]
    graded = await asyncio.wait_for(asyncio.gather(*map(r.driver.grade, admitted)), 30)
    assert [one.reward for one in graded] == [1.0] * 3
    await r.driver.drain()


async def test_a_regrade_retries_as_the_jobs_retry_block_allows(tmp_path, monkeypatch):
    """A regrade whose verifier fails once is graded on Harbor's second try."""
    import baseline_environment

    verifications = []
    run = baseline_environment.LocalEnvironment.exec

    async def fails_once(self, command, *args, **kwargs):
        if "test.sh" in str(command):
            verifications.append(command)
            if len(verifications) == 1:
                raise RuntimeError("the verifier's sandbox went away")
        return await run(self, command, *args, **kwargs)

    monkeypatch.setattr(baseline_environment.LocalEnvironment, "exec", fails_once)
    retry = {"max_retries": 1, "min_wait_sec": 0, "max_wait_sec": 0}
    r, _runtime, task = await harbor(tmp_path, HILL_CLIMB, job={"retry": retry})
    assert (await r.rollout("g-r1", task)).reward == 1.0
    assert verifications.count(verifications[0]) == 2  # the failed command, run again


async def test_sampling_trials_never_retry_inside_harbor(tmp_path):
    r, _runtime, task = await harbor(
        tmp_path, shared_task(tmp_path), "FlakyAgent", job={"retry": {"max_retries": 3}}
    )
    assert await r.rollout("g-r1", task) is Discard.ROLLOUT_FAILED
    assert agents.RUNS == ["flaky"]


async def test_a_lost_sandbox_is_infrastructure_retried_then_discarded(tmp_path):
    r, runtime, ref = await harbor(
        tmp_path, shared_task(tmp_path), "LostSandboxAgent", max_rollout_retries=1
    )
    assert await r.rollout("g-r1", ref) is Discard.ROLLOUT_FAILED  # never graded
    assert agents.RUNS.count("lost") == 2  # retried once
    assert r.driver.gauges() == {
        "retried/attempts/rollout_failed/sandbox_lost": 1.0,
        "discard/rollouts/rollout_failed/sandbox_lost": 1.0,
    }
    assert live_trials(runtime) == 0


async def test_an_excluded_trial_exception_is_a_permanent_rollout_failure(tmp_path):
    r, _runtime, ref = await harbor(
        tmp_path, shared_task(tmp_path), "RefusingAgent", max_rollout_retries=3
    )
    assert await r.rollout("g-r1", ref) is Discard.ROLLOUT_FAILED
    assert agents.RUNS == ["refused"]  # never re-dispatched


async def test_a_trial_exception_harbor_would_retry_is_retry_eligible(tmp_path):
    r, _runtime, ref = await harbor(
        tmp_path, shared_task(tmp_path), "FlakyAgent", max_rollout_retries=1
    )
    assert await r.rollout("g-r1", ref) is Discard.ROLLOUT_FAILED
    assert agents.RUNS == ["flaky", "flaky"]


async def test_an_excluded_exception_in_grading_is_a_permanent_grading_failure(
    tmp_path, monkeypatch
):
    """A regrade that times out, which the job's retry: block excludes, is never regraded."""
    import baseline_environment

    grading, verifications = asyncio.Event(), []
    run = baseline_environment.LocalEnvironment.exec

    async def hanging_verifier(self, command, *args, **kwargs):
        if grading.is_set() and "test" in str(command):
            verifications.append(command)
            await asyncio.Event().wait()  # past the verifier's own deadline
        return await run(self, command, *args, **kwargs)

    monkeypatch.setattr(baseline_environment.LocalEnvironment, "exec", hanging_verifier)
    r, _runtime, task = await harbor(
        tmp_path,
        separate_task(tmp_path, agent_timeout=30, verifier_timeout=0.2),
        job={"retry": {"max_retries": 2}},  # VerifierTimeoutError is in its default exclusions
    )
    admitted = await r.driver.sample(task, "g", "g-r1")
    grading.set()
    assert await r.driver.grade(admitted) is Discard.GRADING_FAILED
    await r.driver.drain()
    assert len(verifications) == 1  # never regraded
    assert r.driver.note("g-r1").cause == "grading_failed/verifier_timeout"


async def test_a_poisoned_leases_cause_takes_precedence_over_the_trial_exception(tmp_path, caplog):
    """A trial exception Harbor would retry does not make a sampler error retry-eligible."""
    r, _runtime, ref = await harbor(
        tmp_path, shared_task(tmp_path), "CatchingAgent", max_rollout_retries=2
    )

    async def broken(prompt, **kwargs):
        raise ConnectionError("sampler down")

    r.sampler.sample_async = broken
    assert await r.rollout("g-r1", ref) is Discard.ROLLOUT_FAILED
    assert agents.RUNS == ["caught"]  # never re-dispatched
    assert "sampling failed: g-r1: ModelTransportFailure" in caplog.text


async def test_a_harbor_failure_is_counted_under_harbors_name_for_it(tmp_path):
    task = shared_task(tmp_path)
    (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 1\n")  # writes no reward
    r, _runtime, ref = await harbor(tmp_path, task)
    assert await r.rollout("g-r1", ref) is Discard.GRADING_FAILED
    assert list(r.driver.gauges()) == ["discard/rollouts/grading_failed/reward_file_not_found"]


@pytest.mark.parametrize("steps", [0, 2], ids=["single-step", "multi-step"])
async def test_a_gated_reward_is_dropped_under_the_gates_reason(tmp_path, steps):
    gates.SEEN.clear()
    r, runtime, ref = await harbor(
        tmp_path,
        shared_task(tmp_path, steps=steps),
        config={"reward_gate": "harbor_test_gates:drop"},
    )
    assert await r.rollout("g-r1", ref) is Discard.GRADING_FAILED
    assert list(r.driver.gauges()) == ["discard/rollouts/grading_failed/flagged"]
    assert gates.SEEN == ["B"]  # read from the sandbox before it stopped
    assert list((tmp_path / "work" / "trials").rglob("reward.txt"))  # kept for analysis
    assert live_trials(runtime) == 0


async def test_a_failed_step_is_a_failure_even_when_the_gate_drops_the_reward(
    tmp_path, monkeypatch
):
    """A gate only turns a kept reward into a dropped one. A trial whose step failed
    ungraded is a failure with or without one, so it still counts toward the stop."""
    from loops_cookbook.harbor.runtime import _Pending

    publish = _Pending.publish

    class WithFailedStep:
        """The trial, as Harbor would leave it had its step failed ungraded."""

        def __init__(self, trial):
            self._trial = trial
            failed = NS(
                exception_info=NS(exception_type="RuntimeError", exception_message="step failed"),
                verifier_result=None,
            )
            self.result = NS(
                exception_info=None,
                step_results=[failed],
                trial_name=getattr(trial.result, "trial_name", None),
                trial_uri=getattr(trial.result, "trial_uri", None),
            )

        def __getattr__(self, name):
            return getattr(self._trial, name)

    async def publish_with_failed_step(pending, trial, result):
        await publish(pending, WithFailedStep(trial), result)

    monkeypatch.setattr(_Pending, "publish", publish_with_failed_step)
    task = shared_task(tmp_path, steps=2)
    toml = task / "task.toml"
    tables = read_toml(toml)
    tables["steps"][0]["min_reward"] = 2.0
    write_toml(toml, tables)
    r, runtime, ref = await harbor(tmp_path, task, config={"reward_gate": "harbor_test_gates:drop"})
    assert await r.rollout("g-r1", ref) is Discard.ROLLOUT_FAILED  # not dropped as "flagged"
    assert list(r.driver.gauges()) == ["discard/rollouts/rollout_failed/runtime"]
    await r.driver.drain()
    assert live_trials(runtime) == 0


@pytest.mark.parametrize(
    ("gate", "outcome", "calls"),
    [
        ("counted_keep", 1.0, ["keep"]),
        ("counted_drop", Discard.GRADING_FAILED, ["drop"]),  # a drop is never retried
        ("fails_once", 1.0, ["fails_once", "fails_once"]),  # trains once the gate passes
    ],
    ids=["keep", "drop", "fails_then_passes"],
)
async def test_a_regrade_is_gated_and_only_a_failed_gate_is_retried(
    tmp_path, monkeypatch, gate, outcome, calls
):
    """A separate verifier's regrade, under a job retry block that would retry a drop;
    every try verifies holding the regrade's verification permit."""
    import baseline_environment

    from loops_cookbook.runtime.api import Phase

    gates.CALLS.clear()
    held: list[bool] = []
    run = baseline_environment.LocalEnvironment.exec

    async def recording(self, command, *args, **kwargs):
        if "test.sh" in str(command):
            held.append(all(Phase.VERIFICATION in p.holds for p in runtime._pending.values()))
        return await run(self, command, *args, **kwargs)

    monkeypatch.setattr(baseline_environment.LocalEnvironment, "exec", recording)
    retry = {"max_retries": 2, "min_wait_sec": 0, "max_wait_sec": 0}
    r, runtime, task = await harbor(
        tmp_path,
        HILL_CLIMB,
        job={"retry": retry},
        config={"reward_gate": f"harbor_test_gates:{gate}"},
    )
    result = await r.rollout("g-r1", task)
    assert (result if result is Discard.GRADING_FAILED else result.reward) == outcome
    assert gates.CALLS == calls
    assert held and all(held)  # every verifier command, on every try, holds the permit
    if outcome is Discard.GRADING_FAILED:
        assert list(r.driver.gauges()) == ["discard/rollouts/grading_failed/flagged"]
    await r.driver.drain()
    assert live_trials(runtime) == 0 and not r.systemic


async def test_a_reason_may_start_with_a_digit(tmp_path):
    r, _runtime, ref = await harbor(
        tmp_path, shared_task(tmp_path), config={"reward_gate": "harbor_test_gates:digit"}
    )
    assert await r.rollout("g-r1", ref) is Discard.GRADING_FAILED
    assert list(r.driver.gauges()) == ["discard/rollouts/grading_failed/0_flagged"]


async def test_a_reward_the_gate_keeps_trains(tmp_path):
    r, _runtime, ref = await harbor(
        tmp_path, shared_task(tmp_path), config={"reward_gate": "harbor_test_gates:keep"}
    )
    assert (await r.rollout("g-r1", ref)).reward == 1.0


@pytest.mark.parametrize("gate", ["broken", "not_async"])
@pytest.mark.parametrize("steps", [0, 2], ids=["single-step", "multi-step"])
async def test_a_gate_that_fails_fails_grading_and_still_stops_the_sandbox(tmp_path, steps, gate):
    r, runtime, ref = await harbor(
        tmp_path,
        shared_task(tmp_path, steps=steps),
        config={"reward_gate": f"harbor_test_gates:{gate}"},
    )
    assert await r.rollout("g-r1", ref) is Discard.GRADING_FAILED
    assert list(r.driver.gauges()) == ["discard/rollouts/grading_failed/reward_gate"]
    await r.driver.drain()
    assert live_trials(runtime) == 0 and not r.systemic


@pytest.mark.parametrize("gate", ["truthy", "empty", "spaced", "capitalised"])
async def test_a_gate_that_returns_no_metric_name_fails_as_the_gate_failing(tmp_path, gate):
    r, runtime, ref = await harbor(
        tmp_path, shared_task(tmp_path), config={"reward_gate": f"harbor_test_gates:{gate}"}
    )
    assert await r.rollout("g-r1", ref) is Discard.GRADING_FAILED
    assert list(r.driver.gauges()) == ["discard/rollouts/grading_failed/reward_gate_result"]
    await r.driver.drain()
    assert live_trials(runtime) == 0 and not r.systemic


async def test_a_gate_that_never_returns_is_cut_by_the_grading_deadline(tmp_path):
    r, runtime, ref = await harbor(
        tmp_path,
        shared_task(tmp_path),
        config={"reward_gate": "harbor_test_gates:hangs"},
        grading_timeout_seconds=1.0,
        cleanup_timeout_seconds=20.0,
    )
    assert await r.rollout("g-r1", ref) is Discard.GRADING_FAILED
    # The gate runs inside grade(), whose own deadline began before verification did.
    assert list(r.driver.gauges()) == ["discard/rollouts/grading_failed/hook_deadline"]
    await r.driver.drain()
    assert live_trials(runtime) == 0 and not r.systemic  # its sandbox stopped with it


@pytest.mark.parametrize("gate", ["hangs", "swallows", "blocks"])
async def test_a_gate_on_an_early_stop_is_bounded_by_the_grading_deadline(tmp_path, gate):
    """An early stop publishes during sample(), but its gate runs in grade(), under the
    grading deadline: past it the trial fails, even when the gate swallows the
    cancellation and returns a keep."""
    task = shared_task(tmp_path, steps=2)
    toml = task / "task.toml"
    tables = read_toml(toml)
    tables["steps"][0]["min_reward"] = 2.0
    for step in tables["steps"]:
        step["verifier"] = {"timeout_sec": 1}  # the grading deadline must exceed it
    write_toml(toml, tables)
    r, runtime, ref = await harbor(
        tmp_path,
        task,
        config={
            "reward_gate": f"harbor_test_gates:{gate}",
            "limits": {"grading_timeout_seconds": 2.0},  # the rollout runtime's, as a run sets both
        },
        grading_timeout_seconds=2.0,
        cleanup_timeout_seconds=20.0,
    )
    assert await asyncio.wait_for(r.rollout("g-r1", ref), 30) is Discard.GRADING_FAILED
    assert list(r.driver.gauges()) == ["discard/rollouts/grading_failed/hook_deadline"]
    await r.driver.drain()
    assert live_trials(runtime) == 0 and not r.systemic  # its sandbox stopped with it


async def test_a_gate_that_ignores_its_cancellation_is_kept_as_any_grade_that_does(tmp_path):
    """On an early stop too: the gate runs in grade(), so the driver keeps its attempt live,
    with its sandbox, until it ends, and discards the rollout as unquiesced."""
    gates.IGNORING.clear()
    task = shared_task(tmp_path, steps=2)
    toml = task / "task.toml"
    tables = read_toml(toml)
    tables["steps"][0]["min_reward"] = 2.0
    for step in tables["steps"]:
        step["verifier"] = {"timeout_sec": 1}  # the grading deadline must exceed it
    write_toml(toml, tables)
    r, runtime, ref = await harbor(
        tmp_path,
        task,
        config={
            "reward_gate": "harbor_test_gates:ignores",
            "limits": {"grading_timeout_seconds": 2.0},
        },
        grading_timeout_seconds=2.0,
        cleanup_timeout_seconds=20.0,
    )
    try:
        assert await asyncio.wait_for(r.rollout("g-r1", ref), 15) is Discard.UNQUIESCED
        assert list(r.driver.gauges()) == ["discard/rollouts/unquiesced/grade"]
        assert r.driver.unquiesced == ("grade:g-r1",)
        assert live_trials(runtime) == 1  # kept live, sandbox and all, while the gate runs
    finally:
        gates.IGNORING.set()

    async def kept_hook_ends():
        while r.driver.unquiesced:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(kept_hook_ends(), 5)  # then the driver cleans up what it owed
    await r.driver.drain()
    assert live_trials(runtime) == 0 and not r.systemic


@pytest.mark.parametrize("steps", [0, 2], ids=["single-step", "multi-step-early-stop"])
async def test_a_failed_gate_is_handed_over_before_the_sandbox_stops(tmp_path, monkeypatch, steps):
    """As a reward is: sample() and grade() never wait on a stop that hangs after it."""
    grading, released = held_stops(monkeypatch)
    grading.set()  # every stop is held from the start
    task = shared_task(tmp_path, steps=steps)
    if steps:
        toml = task / "task.toml"
        tables = read_toml(toml)
        tables["steps"][0]["min_reward"] = 2.0  # stops after the first step
        write_toml(toml, tables)
    r, runtime, ref = await harbor(
        tmp_path,
        task,
        config={"reward_gate": "harbor_test_gates:broken"},
        grading_timeout_seconds=10.0,
        cleanup_timeout_seconds=20.0,
    )
    assert await asyncio.wait_for(r.rollout("g-r1", ref), 5) is Discard.GRADING_FAILED
    assert list(r.driver.gauges()) == ["discard/rollouts/grading_failed/reward_gate"]
    released.set()
    await r.driver.drain()
    assert live_trials(runtime) == 0 and not r.systemic


async def test_a_verifiers_output_never_changes_harbors_result(tmp_path):
    """Text is not a signal: a verifier that prints a disk-full message still scores."""
    task = shared_task(tmp_path)
    test = task / "tests" / "test.sh"
    test.write_text(
        test.read_text().replace(
            "#!/bin/sh\n", "#!/bin/sh\necho 'OSError: [Errno 28] No space left on device'\n", 1
        )
    )
    r, _runtime, ref = await harbor(tmp_path, task)
    assert (await r.rollout("g-r1", ref)).reward == 1.0
    assert r.driver.gauges() == {}


@pytest.mark.parametrize("verifier", ["shared", "separate"])
@pytest.mark.parametrize(
    ("agent", "recorded"),
    [
        ("SleepyAgent", "AgentTimeoutError"),
        ("ExitingAgent", "NonZeroAgentExitCodeError"),
        ("ApiErrorAgent", "ApiRateLimitError"),
        ("LocalApiErrorAgent", "LocalApiError"),
    ],
    ids=["timeout", "exit", "api-error", "api-error-defined-outside-harbor"],
)
async def test_a_failed_agent_is_graded_as_harbor_grades_it(tmp_path, verifier, agent, recorded):
    """Its verifier's reward trains; the cookbook only reports what Harbor recorded."""
    timeout = 0.2 if agent == "SleepyAgent" else 30
    task = (
        shared_task(tmp_path, agent_timeout=timeout)
        if verifier == "shared"
        else separate_task(tmp_path, agent_timeout=timeout)
    )
    r, runtime, ref = await harbor(tmp_path, task, agent, max_rollout_retries=2)
    outcome = await r.rollout("g-r1", ref)
    assert outcome.reward == 1.0  # graded and trained, through a regrade if separate
    assert len(agents.RUNS) == 1  # never retried
    assert live_trials(runtime) == 0
    metrics = reported(runtime)  # once per sampling trial, never for a regrade
    assert metrics["runtime/harbor/sampling_trials"] == 1
    assert metrics[f"runtime/harbor/trial_exceptions/{recorded}"] == 1
    # The progress display's trials row reads exactly these.
    from loops_cookbook.console import HARBOR_SLOTS

    ((live, _, cap),) = HARBOR_SLOTS
    assert {live, cap, "runtime/harbor/preparing_trials"} <= metrics.keys()
    assert metrics["runtime/harbor/preparing_trials"] == 0  # accepted, then finished


async def _environment_fails_to_start(monkeypatch):
    import baseline_environment

    async def refused(self, force_build):
        agents.RUNS.append("start")
        raise RuntimeError("the sandbox would not start")

    monkeypatch.setattr(baseline_environment.LocalEnvironment, "start", refused)


async def _logs_fail_to_upload(monkeypatch):
    from harbor.trial.trial import Trial

    async def lost(self):
        raise RuntimeError("the agent's logs could not be uploaded")

    monkeypatch.setattr(Trial, "_upload_agent_logs", lost)


async def _verification_skipped(monkeypatch):
    from harbor.trial.single_step import SingleStepTrial

    async def agent_only(self):
        await self._run_agent()

    monkeypatch.setattr(SingleStepTrial, "_run", agent_only)


@pytest.mark.parametrize(
    ("agent", "break_harbor"),
    [
        ("FlakyAgent", None),
        ("ImpostorExitAgent", None),
        ("AnswerAgent", _environment_fails_to_start),
        ("ExitingAgent", _logs_fail_to_upload),
        ("AnswerAgent", _verification_skipped),
    ],
    ids=[
        "agent-crash",
        "exception-named-like-harbors",
        "environment-start",
        "after-a-verified-agent-exit",
        "nothing-recorded",
    ],
)
async def test_a_separate_verifier_trial_harbor_ended_before_verification_is_retried(
    tmp_path, monkeypatch, agent, break_harbor
):
    """Only Harbor carrying the trial on to its verification grades it; a name never does."""
    if break_harbor is not None:
        await break_harbor(monkeypatch)
    r, runtime, ref = await harbor(
        tmp_path, separate_task(tmp_path, agent_timeout=30), agent, max_rollout_retries=1
    )
    assert await r.rollout("g-r1", ref) is Discard.ROLLOUT_FAILED
    gauges = r.driver.gauges()
    assert {key.split("/")[0] for key in gauges} == {"retried", "discard"}
    assert all("/rollout_failed/" in key for key in gauges)
    assert len(agents.RUNS) == 2  # retried once
    assert not (tmp_path / "work" / "trials" / "grades").exists()  # never regraded
    assert live_trials(runtime) == 0


async def test_an_agent_deadline_that_cut_its_only_generation_leaves_nothing_to_train(
    tmp_path,
):
    r, runtime, task = await harbor(tmp_path, shared_task(tmp_path, agent_timeout=0.2))

    async def hangs(prompt, **kwargs):
        await asyncio.Event().wait()

    r.sampler.sample_async = hangs
    assert await r.rollout("g-r1", task) is Discard.UNVERSIONED
    assert r.runtime is runtime and live_trials(runtime) == 0


async def test_a_multi_step_agent_deadline_that_cut_its_only_generation_leaves_nothing(
    tmp_path,
):
    r, runtime, ref = await harbor(tmp_path, shared_task(tmp_path, agent_timeout=0.2, steps=2))

    async def hangs(prompt, **kwargs):
        await asyncio.Event().wait()

    r.sampler.sample_async = hangs
    assert await r.rollout("g-r1", ref) is Discard.UNVERSIONED
    assert live_trials(runtime) == 0


async def test_a_graded_agent_deadline_drops_only_the_generation_it_cut(tmp_path):
    """The agent answered, then its deadline cut a second call: the episode trains."""
    r, runtime, task = await harbor(
        tmp_path, shared_task(tmp_path, agent_timeout=0.5), "AnswerThenHangAgent"
    )
    sample = r.sampler.sample_async
    calls = []

    async def second_call_hangs(prompt, **kwargs):
        calls.append(prompt)
        if len(calls) > 1:
            await asyncio.Event().wait()
        return await sample(prompt, **kwargs)

    r.sampler.sample_async = second_call_hangs
    graded = await r.rollout("g-r1", task)
    assert graded.reward == 1.0  # the verifier's reward for the answer it did write
    assert len(graded.samples) == 1  # the cut generation is not evidence
    assert len(calls) == 2 and live_trials(runtime) == 0


async def test_a_task_without_an_agent_deadline_is_bounded_by_the_rollout_clock(tmp_path):
    task = shared_task(tmp_path)
    write_toml(task / "task.toml", read_toml(task / "task.toml"), {"agent": None})
    r, runtime, ref = await harbor(tmp_path, task, "SleepyAgent", rollout_timeout_seconds=0.5)
    assert await asyncio.wait_for(r.rollout("g-r1", ref), 10) is Discard.AGENT_TIMEOUT
    assert live_trials(runtime) == 0


async def test_a_task_edited_mid_run_trains_and_its_rollouts_record_what_changed(tmp_path):
    """A task's content is the user's to change.

    Each rollout records the digest of the task as its trial loaded it.
    """
    task = shared_task(tmp_path)
    r, _runtime, ref = await harbor(tmp_path, task)
    before = await r.rollout("g-r1", ref)
    (task / "tests" / "test.sh").write_text("#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n")
    after = await r.rollout("g-r2", ref)
    assert before.reward == after.reward == 1.0
    assert before.task_digest != after.task_digest


async def test_an_agent_without_the_binding_fails_at_trial_setup(tmp_path, monkeypatch):
    r, runtime, task = await harbor(tmp_path, shared_task(tmp_path))
    monkeypatch.setattr(agents.AnswerAgent, "bind_loops_services", None)
    assert await r.rollout("g-r1", task) is Discard.ROLLOUT_FAILED
    assert r.driver.live_attempts == 0 and live_trials(runtime) == 0


def _recording_setup(monkeypatch) -> list:
    """Trials as the sampling queue sets them up, for assertions on Harbor state."""
    from loops_cookbook.harbor import integration

    trials = []
    setup = integration.HookedTrialQueue._setup_hooks

    def recording(self, trial):
        trials.append(trial)
        setup(self, trial)

    monkeypatch.setattr(integration.HookedTrialQueue, "_setup_hooks", recording)
    return trials


async def test_a_trial_whose_binding_fails_closes_its_logger(tmp_path, monkeypatch):
    r, _runtime, task = await harbor(tmp_path, shared_task(tmp_path))
    trials = _recording_setup(monkeypatch)
    monkeypatch.setattr(agents.AnswerAgent, "bind_loops_services", None)
    assert await r.rollout("g-r1", task) is Discard.ROLLOUT_FAILED
    (trial,) = trials
    assert trial._log_handler is None


async def test_holds_are_released_when_harbor_skips_its_end_and_cancel_hooks(tmp_path, monkeypatch):
    """Harbor finalization errors can skip every releasing hook; the adapter still lets go.

    The agent fails after AGENT_START took the sampling hold, every hook that would
    release it is removed, and the driver's own release of leftover holds is
    disabled, so only the adapter's finally can free it before the attempt aborts.
    """
    from harbor.trial.hooks import TrialEvent

    from loops_cookbook.harbor import integration
    from loops_cookbook.runtime import driver as driver_module

    r, runtime, task = await harbor(tmp_path, shared_task(tmp_path), "RefusingAgent")
    setup = integration.HookedTrialQueue._setup_hooks
    held_while_running = []

    def without_releases(self, trial):
        setup(self, trial)
        for event in (
            TrialEvent.AGENT_END,
            TrialEvent.VERIFICATION_START,
            TrialEvent.END,
            TrialEvent.CANCEL,
        ):
            trial._hooks[event].clear()

        async def running(event):
            held_while_running.append(r.capacity.snapshot()["slots/sampling"])

        trial.add_hook(TrialEvent.AGENT_START, running)

    monkeypatch.setattr(integration.HookedTrialQueue, "_setup_hooks", without_releases)
    monkeypatch.setattr(driver_module, "_release_leftovers", lambda phases: None)
    held_at_abort = []
    abort = runtime.abort

    async def recording(attempt, reason):
        # Before the driver closes the attempt, which would release anything left.
        snapshot = r.capacity.snapshot()
        held_at_abort.append(snapshot["slots/sampling"] + snapshot["slots/verification"])
        await abort(attempt, reason)

    runtime.abort = recording
    assert await r.rollout("g-r1", task) is Discard.ROLLOUT_FAILED
    assert held_while_running == [1]  # there was a hold to release
    assert held_at_abort == [0]
    assert live_trials(runtime) == 0


async def test_abort_and_finish_leave_no_retained_trial(tmp_path):
    r, runtime, task = await harbor(tmp_path, shared_task(tmp_path))
    admitted = await r.driver.sample(task, "g", "g-r1")
    assert live_trials(runtime) == 1
    r.driver.discard_admitted(admitted)
    await r.driver.drain()
    assert live_trials(runtime) == 0


async def test_close_cancels_retained_trials_and_their_attempts_leave_the_directory(tmp_path):

    r, runtime, task = await harbor(tmp_path, shared_task(tmp_path))
    admitted = await r.driver.sample(task, "g", "g-r1")
    attempt_id = admitted.rollout.identity.attempt_id
    assert attempts.attempt_for(attempt_id) is not None
    await runtime.close()
    assert live_trials(runtime) == 0
    with pytest.raises(RuntimeError, match="no live attempt"):
        attempts.attempt_for(attempt_id)


async def test_close_reports_a_retained_trial_whose_sandbox_would_not_stop(tmp_path, monkeypatch):
    import baseline_environment

    from loops_cookbook.harbor.runtime import SandboxStopFailure

    closing = asyncio.Event()
    stop = baseline_environment.LocalEnvironment.stop

    async def failing(self, delete):
        if closing.is_set():
            raise RuntimeError("the sandbox would not stop")
        return await stop(self, delete)

    monkeypatch.setattr(baseline_environment.LocalEnvironment, "stop", failing)
    r, runtime, task = await harbor(tmp_path, shared_task(tmp_path))
    await r.driver.sample(task, "g", "g-r1")  # retained, parked at its final verification
    closing.set()
    with pytest.raises(SandboxStopFailure, match="could not stop"):
        await runtime.close()
    assert live_trials(runtime) == 0  # the trial is still forgotten


@pytest.mark.parametrize("task", ["shared", "separate"])
async def test_the_reward_is_committed_before_any_sandbox_stops(tmp_path, monkeypatch, task):
    # Every stop is held until grade() has returned, so the reward cannot wait for
    # one; the deadline leaves room for a real verifier subprocess on a busy machine.
    grading, released = held_stops(monkeypatch)
    path = shared_task(tmp_path) if task == "shared" else HILL_CLIMB
    r, runtime, ref = await harbor(
        tmp_path, path, grading_timeout_seconds=3.0, cleanup_timeout_seconds=20.0
    )
    admitted = await r.driver.sample(ref, "g", "g-r1")
    grading.set()
    graded = await r.driver.grade(admitted)
    assert graded.reward == 1.0  # returned while every sandbox stop is still held
    assert live_trials(runtime) == 1  # teardown is still running, owned by finish()
    released.set()
    await r.driver.drain()
    assert live_trials(runtime) == 0 and not r.systemic


async def test_a_blocked_teardown_keeps_the_reward_and_counts_as_cleanup(tmp_path, monkeypatch):
    grading, _released = held_stops(monkeypatch)  # never released: the stop blocks
    r, _runtime, ref = await harbor(
        tmp_path, shared_task(tmp_path), cleanup_timeout_seconds=0.3, max_consecutive_failures=1
    )
    admitted = await r.driver.sample(ref, "g", "g-r1")
    grading.set()
    graded = await r.driver.grade(admitted)
    assert graded.reward == 1.0
    await r.driver.drain()
    assert r.systemic and "cleanup" in str(r.systemic[0])
    assert live_trials(_runtime) == 0  # the cut-short settle still discarded its trial


@pytest.mark.parametrize("task", ["shared", "separate"])
async def test_a_failed_sandbox_stop_keeps_the_reward_and_counts_as_cleanup(
    tmp_path, monkeypatch, task
):
    """Harbor records a stop that raises; the cookbook still counts it as a cleanup failure."""
    import baseline_environment

    grading = asyncio.Event()
    stop = baseline_environment.LocalEnvironment.stop

    async def failing(self, delete):
        if grading.is_set():
            raise RuntimeError("the sandbox would not stop")
        return await stop(self, delete)

    monkeypatch.setattr(baseline_environment.LocalEnvironment, "stop", failing)
    path = shared_task(tmp_path) if task == "shared" else HILL_CLIMB
    r, runtime, ref = await harbor(tmp_path, path, max_consecutive_failures=1)
    admitted = await r.driver.sample(ref, "g", "g-r1")
    grading.set()
    graded = await r.driver.grade(admitted)
    assert graded.reward == 1.0  # committed before any stop
    await r.driver.drain()
    assert r.systemic and "cleanup" in str(r.systemic[0])
    assert live_trials(runtime) == 0


async def test_a_network_policy_restore_failure_after_verify_publishes_no_reward(
    tmp_path, monkeypatch
):
    import baseline_environment

    task = shared_task(tmp_path)
    no_network = {"network_mode": "no-network"}
    toml = task / "task.toml"
    write_toml(toml, read_toml(toml), {"agent": no_network, "verifier": no_network})

    grading = asyncio.Event()

    async def restore_fails(self, policy):
        restricted = getattr(self, "_restricted", False)
        self._restricted = policy.network_mode.value != "public"
        if grading.is_set() and restricted and not self._restricted:
            raise OSError("policy restore failed")

    monkeypatch.setattr(
        baseline_environment.LocalEnvironment, "_apply_network_policy", restore_fails
    )
    r, _runtime, ref = await harbor(tmp_path, task)
    admitted = await r.driver.sample(ref, "g", "g-r1")
    grading.set()
    assert await r.driver.grade(admitted) is Discard.GRADING_FAILED
    await r.driver.drain()
