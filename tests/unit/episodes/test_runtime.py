"""The Episode adapter through the real driver, capacity broker, lease and resource registry.

Every episode here is ``Scripted``: its hooks do what the test's ``script`` says, and
every call is recorded, so a test states one hostile behaviour and reads back what
the cookbook did about it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import sys
import threading
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from types import FunctionType, MethodType, ModuleType
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict
from support.cancellation import (
    Gate,
    Landing,
    Response,
    Trigger,
    Work,
    cancellation_matrix,
    outcome_within,
    settle_loop,
)
from support.episode_rig import Scripted, Unsandboxed, answer, calls, live_sandboxes, score, until
from support.resources import FAST, Fake, FakeClient

from loops_cookbook.episodes import Episode, Grading
from loops_cookbook.episodes.api import Model
from loops_cookbook.episodes.runtime import TIMED_OUT
from loops_cookbook.model_access import ModelAccessMode
from loops_cookbook.model_access.lease import AttemptLease, DirectModelAccess
from loops_cookbook.model_access.openai.access import HttpModelAccess
from loops_cookbook.runtime import GradeResult
from loops_cookbook.runtime.api import AttemptContext, Phase
from loops_cookbook.runtime.driver import Discard


class SandboxLost(RuntimeError):
    pass


class RetrySafe(Scripted):
    grading_retries = 2


class RefinedFake(Fake):
    """A kind an adapter may provide for an episode written for ``Fake``."""


async def test_the_sandbox_is_the_kind_the_adapter_provides_a_refinement_of_the_episodes(
    make, script
):
    built = await make(provides=RefinedFake)
    await built.rig.rollout("g-r1", built.task)
    (episode,) = script.of("init")
    assert type(episode.sandbox) is RefinedFake


async def test_an_attempt_admits_provisions_sets_up_runs_grades_and_closes_once(make, script):
    built = await make()
    graded = await built.rig.rollout("g-r1", built.task)
    assert graded.reward == 1.0 and graded.metadata == {"answer": "ok"}
    (episode,) = script.of("init")
    assert [event for event, _ in script.events] == ["init", "setup", "run", "grade", "close"]
    # The sandbox existed before the episode was built, and is gone with the attempt.
    assert episode.sandbox_at_setup is episode.sandbox is not None
    assert type(episode.sandbox) is Fake  # the kind the adapter provides
    assert episode.sandbox.environment["FAKE_NAME"] == episode.sandbox.name
    assert [op for op, _ in built.provider.calls if op == "create"] == ["create"]
    # A sandbox table of provider only is the kind's spec with its defaults.
    (created,) = built.provider.created
    assert (created.kind, dict(created.specification)) == ("box", {"image": "fake-image"})
    assert live_sandboxes(built.provider) == []
    assert not episode.phase.timed_out
    assert built.rig.metrics.snapshot()[f"runtime/{TIMED_OUT}/mean"] == 0.0


@pytest.mark.parametrize("budget", [262144, None])
async def test_run_is_told_the_runs_trajectory_budget(make, script, budget):
    built = await make(trajectory_tokens=budget)
    await built.rig.rollout("g-r1", built.task)
    (episode,) = script.of("init")
    assert episode.sampling.trajectory_tokens == budget


async def raise_lost(episode, *args):
    raise SandboxLost("the environment went away")


@pytest.mark.parametrize("stage", ["setup", "run"])
async def test_raising_in_setup_or_run_retries_with_a_fresh_episode_and_sandbox(
    make, script, stage
):
    built = await make(max_rollout_retries=1)
    setattr(script, stage, raise_lost)
    outcome = await built.rig.rollout("g-r1", built.task)
    await built.rig.driver.drain()
    assert outcome is Discard.ROLLOUT_FAILED
    assert built.rig.driver.note("g-r1").cause == "rollout_failed/sandbox_lost"
    first, second = script.of("init")
    assert first is not second and first.phase.attempt_id != second.phase.attempt_id
    assert first.sandbox is not second.sandbox
    assert script.of("close") == [first, second]
    assert script.of("grade") == []
    assert [op for op, _ in built.provider.calls].count("create") == 2
    assert live_sandboxes(built.provider) == []


async def test_raising_in_grade_is_a_named_grading_failure_and_the_episode_still_closes(
    make, script
):
    built = await make()
    script.grade = raise_lost
    assert await built.rig.rollout("g-r1", built.task) is Discard.GRADING_FAILED
    assert built.rig.driver.note("g-r1").cause == "grading_failed/sandbox_lost"
    assert len(script.of("close")) == 1 and live_sandboxes(built.provider) == []


async def test_a_retry_safe_episode_is_graded_again_and_the_same_episode_answers(make, script):
    tries = []

    async def flaky(episode, rollout):
        tries.append(episode)
        if len(tries) == 1:
            raise SandboxLost("the grader's host went away")
        return 1.0

    built = await make(RetrySafe)
    script.grade = flaky
    assert (await built.rig.rollout("g-r1", built.task)).reward == 1.0
    assert len(tries) == 2 and tries[0] is tries[1]


async def test_a_close_that_raises_is_a_cleanup_failure_never_the_rollouts_outcome(make, script):
    built = await make(max_consecutive_failures=1)
    script.close = raise_lost
    graded = await built.rig.rollout("g-r1", built.task)
    assert graded.reward == 1.0  # the rollout committed; its cleanup failed after
    attempt = script.of("init")[0].phase.attempt_id
    expected = (
        f"1 consecutive cleanup infrastructure failures: rollout runtime finish() for g-r1 attempt "
        f"{attempt} failed: SandboxLost: the environment went away"
    )
    assert [str(error) for error in built.rig.systemic] == [expected]
    assert live_sandboxes(built.provider) == []  # released after close() failed


async def test_a_partial_run_is_graded_from_what_init_set_and_the_timeout_flag(make, script):
    async def partial(episode, model):
        await answer(episode, model)  # delivered: it trains
        episode.answer = "half-written"
        await asyncio.sleep(30)

    async def lenient(episode, rollout):
        return 0.25 if episode.phase.timed_out and episode.answer == "half-written" else 0.0

    built = await make(agent_timeout=0.2, rollout_timeout=1.0)
    script.run, script.grade = partial, lenient
    graded = await built.rig.rollout("g-r1", built.task)
    assert graded.reward == 0.25 and graded.metadata == {}
    assert len(graded.samples) == 1
    assert built.rig.metrics.snapshot()[f"runtime/{TIMED_OUT}/mean"] == 1.0


class Forgetful(Scripted):
    """Sets ``result`` only once run() gets far enough; grade() reads it regardless."""

    async def grade(self, grading: Grading):
        return float(self.result == "ok")  # pyright: ignore[reportAttributeAccessIssue]


async def test_a_grade_that_forgot_a_partial_run_fails_as_a_named_grading_failure(make, script):
    async def slow(episode, model):
        await answer(episode, model)
        await asyncio.sleep(30)
        episode.result = "ok"

    built = await make(Forgetful, agent_timeout=0.2, rollout_timeout=1.0)
    script.run = slow
    assert await built.rig.rollout("g-r1", built.task) is Discard.GRADING_FAILED
    assert built.rig.driver.note("g-r1").cause == "grading_failed/attribute"
    assert len(script.of("close")) == 1


@pytest.mark.parametrize("shortcut", [False, True], ids=["llm", "model_conversation"])
async def test_a_model_stashed_on_the_episode_refuses_once_run_has_ended(make, script, shortcut):
    refusals: list[str] = []

    async def stash(episode, model):
        episode.model, episode.llm = model, model.llm()
        episode.conversation = model.conversation() if shortcut else episode.llm.conversation()
        reply = await episode.conversation.generate(messages=[{"role": "user", "content": "q"}])
        episode.answer = reply.content
        return {"answer": reply.content}

    async def reuse(episode, rollout):
        for call in (
            lambda: episode.model.conversation() if shortcut else episode.model.llm(),
            lambda: episode.conversation.generate(messages=[{"role": "user", "content": "x"}]),
        ):
            try:
                result = call()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as error:  # noqa: BLE001 - what the refusal was
                refusals.append(f"{type(error).__name__}: {error}")
        return 1.0

    built = await make()
    script.run, script.grade = stash, reuse
    graded = await built.rig.rollout("g-r1", built.task)
    assert graded.reward == 1.0 and built.rig.sampler.calls == 1
    assert refusals == [
        "ModelAccessError: model access for this attempt is sealed",
        "CaptureIntegrityFailure: model access for this attempt is sealed",
    ]


async def test_grade_that_samples_through_a_stashed_model_fails_without_a_retry(make, script):
    async def stash(episode, model):
        episode.model = model
        return await answer(episode, model)

    async def judge_with_the_policy(episode, rollout):
        await (
            episode.model.llm().conversation().generate(messages=[{"role": "user", "content": "x"}])
        )
        return 1.0

    built = await make(RetrySafe)
    script.run, script.grade = stash, judge_with_the_policy
    assert await built.rig.rollout("g-r1", built.task) is Discard.GRADING_FAILED
    assert built.rig.driver.note("g-r1").cause == "grading_failed/model_access"
    assert (built.rig.sampler.calls, len(script.of("grade"))) == (1, 1)


async def test_grade_metadata_that_is_not_json_fails_grading_without_a_retry(make, script):
    async def unserialisable(episode, rollout):
        return GradeResult(reward=1.0, metadata={"when": object()})

    built = await make(RetrySafe)
    script.grade = unserialisable
    assert await built.rig.rollout("g-r1", built.task) is Discard.GRADING_FAILED
    assert len(script.of("grade")) == 1


async def test_episode_metadata_that_is_not_json_fails_the_rollout_without_a_retry(make, script):
    async def unserialisable(episode, model):
        await answer(episode, model)
        return {"when": object()}

    built = await make(max_rollout_retries=2)
    script.run = unserialisable
    assert await built.rig.rollout("g-r1", built.task) is Discard.ROLLOUT_FAILED
    assert (len(script.of("run")), built.rig.sampler.calls) == (1, 1)


async def test_an_undeclared_producer_stops_the_run_even_when_the_episode_swallows_it(make, script):
    async def misnamed(episode, model):
        with contextlib.suppress(Exception):  # the episode hides it; the run must not
            model.llm("critic")
        return await answer(episode, model)

    built = await make(max_rollout_retries=2)
    script.run = misnamed
    assert await built.rig.rollout("g-r1", built.task) is Discard.ROLLOUT_FAILED
    assert len(script.of("run")) == 1
    assert [type(error.__cause__).__name__ for error in built.rig.systemic] == [
        "UndeclaredProducer"
    ]


def lingering(seconds: float):
    """An agent that ignores the first cancellation for ``seconds``, then returns."""

    async def run(episode, model):
        await answer(episode, model)
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.sleep(seconds)  # a second cancellation still stops it
        return {"lingered": True}

    return run


async def test_a_slow_cancel_inside_the_backstop_is_still_graded_as_timed_out(make, script):
    built = await make(agent_timeout=0.3, rollout_timeout=1.0, grace=0.2)
    script.run = lingering(0.3)
    script.grade = lambda episode, rollout: asyncio.sleep(0, float(episode.phase.timed_out))
    graded = await built.rig.rollout("g-r1", built.task)
    assert graded.reward == 1.0 and graded.metadata == {"lingered": True}


async def test_a_cancel_slower_than_the_backstop_is_discarded_by_the_rollout_clock(make, script):
    built = await make(agent_timeout=0.3, rollout_timeout=1.0, grace=0.2)
    script.run = lingering(5.0)
    assert await built.rig.rollout("g-r1", built.task) is Discard.AGENT_TIMEOUT
    await built.rig.driver.drain()
    assert built.rig.driver.note("g-r1").cause == "agent_timeout/rollout_clock"
    assert script.of("grade") == [] and len(script.of("close")) == 1
    assert built.rig.systemic == [] and live_sandboxes(built.provider) == []


def stubborn_until(stop):
    """An episode hook that ignores cancellation until the test lets it go."""

    async def hook(episode, *args):
        while not stop.is_set():
            try:
                await stop.wait()
            except asyncio.CancelledError:
                continue

    return hook


@pytest.mark.parametrize(("hook", "fails"), [("run", False), ("close", False), ("close", True)])
async def test_a_kept_hooks_sandbox_is_released_once_it_ends(make, script, hook, fails):
    # run() is kept as sample(), close() as finish(); neither releases its sandbox while
    # it may still use it. A release that fails then is reported, not counted again.
    built = await make(
        agent_timeout=0.1,
        rollout_timeout=0.3,
        grace=0.1,
        cleanup_timeout_seconds=0.1,
        max_consecutive_failures=2,
    )
    stop = asyncio.Event()
    setattr(script, hook, stubborn_until(stop))
    try:
        await built.rig.rollout("g-r1", built.task)
        await built.rig.driver.drain()
        assert len(live_sandboxes(built.provider)) == 1
        if fails:
            built.provider.release_errors.append(ConnectionError("provider down"))
        stop.set()
        await settle_loop()
        await built.rig.driver.drain()
        assert built.rig.driver.unquiesced == ()
        assert built.rig.systemic == []
        assert len(live_sandboxes(built.provider)) == (1 if fails else 0)
        assert len(script.of("close")) == 1  # once kept run() ends; never twice
    finally:
        stop.set()


async def test_a_kept_cleanup_hooks_release_outlasts_a_cancelled_drain(make, script, caplog):
    # Shutdown may stop waiting for the drain; the release still runs to its end and
    # its failure is still reported.
    built = await make(
        agent_timeout=0.1, rollout_timeout=0.3, grace=0.1, cleanup_timeout_seconds=0.1
    )
    stop = asyncio.Event()
    script.close = stubborn_until(stop)
    try:
        await built.rig.rollout("g-r1", built.task)
        await built.rig.driver.drain()
        built.provider.blocked.add("release")
        built.provider.release_errors.append(ConnectionError("provider down"))
        stop.set()
        await until(lambda: ("release", "fake-sandbox") in built.provider.calls)
        draining = asyncio.create_task(built.rig.driver.drain())
        await settle_loop()
        draining.cancel()
        built.provider.unblocked.set()
        with pytest.raises(asyncio.CancelledError):
            await draining
        await built.rig.driver.drain()
        assert "cleanup failed: g-r1: CleanupIncomplete" in caplog.text
    finally:
        stop.set()
        built.provider.unblocked.set()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
async def test_an_agent_process_is_killed_at_the_agent_deadline_and_the_episode_graded(
    make, script, tmp_path
):
    pid_file = tmp_path / "agent.pid"
    program = f"import os, pathlib, time; pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)"

    async def agent(episode, model):
        await answer(episode, model)
        await episode.phase.processes.run([sys.executable, "-c", program])

    built = await make(agent_timeout=1.0, rollout_timeout=3.0, grace=0.5)
    script.run = agent
    script.grade = lambda episode, rollout: asyncio.sleep(0, float(episode.phase.timed_out))
    assert (await built.rig.rollout("g-r1", built.task)).reward == 1.0
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


async def test_the_sandbox_is_created_off_both_clocks_and_without_a_sampling_slot(make, script):
    """A slow create, longer than the setup clock, while the only sampling slot is taken."""
    running, release = asyncio.Event(), asyncio.Event()

    async def first_holds_the_slot(episode, model):
        if not running.is_set():
            running.set()
            await release.wait()
        return await answer(episode, model)

    built = await make(
        sampling_slots=1,
        agent_timeout=3.0,
        rollout_timeout=5.0,
        setup_timeout_seconds=0.3,
    )
    script.run = first_holds_the_slot
    first = asyncio.create_task(built.rig.rollout("g-r1", built.task))
    await asyncio.wait_for(running.wait(), 5)
    built.provider.delay = 0.5  # the second sandbox takes longer than the setup clock allows
    second = asyncio.create_task(built.rig.rollout("g-r2", built.task))
    await until(lambda: len(script.of("setup")) == 2)
    # The second sandbox is up and set up while the first still holds the only slot.
    assert len(live_sandboxes(built.provider)) == 2
    await until(lambda: built.rig.capacity.snapshot()["queue/sampling_waiting"] == 1.0)
    assert built.rig.capacity.snapshot()["slots/sampling"] == 1.0
    release.set()
    graded = await asyncio.gather(first, second)
    assert [rollout.reward for rollout in graded] == [1.0, 1.0]
    assert live_sandboxes(built.provider) == []


async def cut(episode, model):
    await answer(episode, model)
    await asyncio.sleep(30)


PATHS = {
    "graded": {},
    "setup raises": {"setup": raise_lost},
    "run raises": {"run": raise_lost},
    "grade raises": {"grade": raise_lost},
    "close raises": {"close": raise_lost},
    "cut by the agent deadline": {"run": cut},
}


@pytest.mark.parametrize("path", sorted(PATHS))
async def test_every_path_closes_the_episode_once_and_releases_its_sandbox(make, script, path):
    built = await make(agent_timeout=0.2, rollout_timeout=1.0)
    for stage, hook in PATHS[path].items():
        setattr(script, stage, hook)
    await built.rig.rollout("g-r1", built.task)
    await built.rig.driver.drain()
    assert script.of("close") == script.of("init") and len(script.of("close")) == 1
    assert live_sandboxes(built.provider) == []


async def test_a_cancelled_attempt_closes_its_episode_once_and_releases_its_sandbox(make, script):
    started = asyncio.Event()

    async def wait(episode, model):
        started.set()
        await asyncio.sleep(30)

    built = await make()
    script.run = wait
    attempt = asyncio.create_task(built.rig.driver.sample(built.task, "g", "g-r1"))
    await asyncio.wait_for(started.wait(), 5)
    attempt.cancel()
    with pytest.raises(asyncio.CancelledError):
        await attempt
    assert len(script.of("close")) == 1 and live_sandboxes(built.provider) == []


class UnsandboxedSampling(Unsandboxed):
    sandbox_lifetime = "sampling"


@pytest.mark.parametrize("episode", [Unsandboxed, UnsandboxedSampling], ids=["episode", "sampling"])
async def test_without_a_sandbox_the_context_has_none_and_nothing_is_provisioned(
    make, script, episode
):
    async def grade(episode, rollout):
        assert episode.phase.grading_sandbox is None
        return await score(episode, rollout)

    built = await make(episode, sandbox=False)
    script.grade = grade
    assert (await built.rig.rollout("g-r1", built.task)).reward == 1.0
    await built.rig.driver.drain()
    assert script.of("init")[0].sandbox is None and built.provider.calls == []


async def test_each_attempt_gets_its_own_copy_of_its_task(make, script):
    async def vandal(episode, model):
        episode.task.data["options"].append("vandalised")  # pyright: ignore[reportAttributeAccessIssue]
        return await answer(episode, model)

    built = await make(tasks=({"id": "t", "options": ["a"]},))
    script.run = vandal
    await built.rig.rollout("g-r1", built.task)
    await built.rig.rollout("g-r2", built.task)
    assert [episode.task.data["options"] for episode in script.of("init")] == [
        ["a", "vandalised"],
        ["a", "vandalised"],
    ]


async def test_grade_may_return_a_grade_result_with_its_own_metadata(make, script):
    built = await make()
    script.grade = lambda episode, rollout: asyncio.sleep(
        0, GradeResult(reward=0.5, metadata={"verdict": "half"})
    )
    graded = await built.rig.rollout("g-r1", built.task)
    assert graded.reward == 0.5 and graded.metadata == {"answer": "ok", "verdict": "half"}


class Http(Scripted):
    model_access = ModelAccessMode.OPENAI_HTTP
    grading_retries = 2


@pytest.mark.parametrize(
    ("episode", "mode", "retry"),
    [(Scripted, ModelAccessMode.DIRECT, 0), (Http, ModelAccessMode.OPENAI_HTTP, 2)],
)
async def test_capabilities_are_the_episode_classes_own(make, episode, mode, retry):
    capabilities = (await make(episode)).runtime.capabilities
    assert capabilities.model_access_modes == frozenset({mode})
    assert capabilities.grading_retries == retry
    assert capabilities.internally_acquired_phases == frozenset({Phase.SAMPLING})


async def test_backpressure_holds_the_sandbox_back_until_the_attempt_is_admitted(make, script):
    built = await make()
    built.rig.admission.value = 0  # the pipeline wants nothing new sampled
    attempt = asyncio.create_task(built.rig.rollout("g-r1", built.task))
    await until(lambda: built.rig.capacity.snapshot()["admission/waiting"] == 1.0)
    assert built.provider.calls == [] and script.events == []
    built.rig.admission.value = 1000
    built.rig.capacity.notify()
    assert (await asyncio.wait_for(attempt, 5)).reward == 1.0


async def test_run_returning_something_other_than_metadata_is_discarded_without_a_retry(
    make, script
):
    async def prose(episode, model):
        await answer(episode, model)
        return "the answer is ok"

    built = await make(max_rollout_retries=2)
    script.run = prose
    assert await built.rig.rollout("g-r1", built.task) is Discard.ROLLOUT_FAILED
    assert built.rig.driver.note("g-r1").cause == "rollout_failed/invalid_episode_metadata"
    assert len(script.of("init")) == 1 and len(script.of("close")) == 1


WRITE_MINE = "import os, pathlib; pathlib.Path('mine.txt').write_text(os.environ['WHO'])"
READ_MINE = (
    "import pathlib, sys; p = pathlib.Path('mine.txt'); "
    "sys.stdout.write(p.read_text() if p.exists() else 'nothing')"
)


async def test_concurrent_attempts_run_their_processes_in_directories_of_their_own(make, script):
    both = asyncio.Barrier(2)

    async def agent(episode, model):
        await answer(episode, model)
        await episode.phase.processes.run(
            [sys.executable, "-c", WRITE_MINE], env={"WHO": episode.phase.attempt_id}
        )
        await both.wait()  # each has written before either reads
        read = await episode.phase.processes.run([sys.executable, "-c", READ_MINE])
        return {"read": read.stdout_tail, "me": episode.phase.attempt_id}

    built = await make()
    script.run = agent
    graded = await asyncio.gather(
        built.rig.rollout("g-r1", built.task), built.rig.rollout("g-r2", built.task)
    )
    assert [rollout.metadata["read"] for rollout in graded] == [
        rollout.metadata["me"] for rollout in graded
    ]


async def test_a_retry_does_not_see_the_failed_attempts_files_and_both_are_removed(
    make, script, tmp_path
):
    reads = []

    async def agent(episode, model):
        await answer(episode, model)
        read = await episode.phase.processes.run([sys.executable, "-c", READ_MINE])
        reads.append(read.stdout_tail)
        await episode.phase.processes.run([sys.executable, "-c", WRITE_MINE], env={"WHO": "failed"})
        if len(reads) == 1:
            raise SandboxLost("the environment went away")

    built = await make(max_rollout_retries=1)
    script.run = agent
    assert (await built.rig.rollout("g-r1", built.task)).reward == 1.0
    assert reads == ["nothing", "nothing"]
    attempts = tmp_path / "work" / "attempts"
    assert not attempts.exists() or list(attempts.iterdir()) == []


async def test_a_slow_directory_removal_leaves_the_event_loop_free(
    make, script, tmp_path, monkeypatch
):
    """A large or slow attempt directory is removed off the loop: other rollouts, clocks
    and heartbeats keep running while it goes."""
    removing, loop_answered = threading.Event(), threading.Event()
    answered_in_time: list[bool] = []
    real_rmtree = shutil.rmtree
    attempts = tmp_path / "work" / "attempts"

    def slow_rmtree(path, *args, **kwargs):
        if Path(path).parent == attempts:
            removing.set()
            answered_in_time.append(loop_answered.wait(timeout=2))
        real_rmtree(path, *args, **kwargs)

    async def answer_while_removing():
        await until(removing.is_set)
        loop_answered.set()

    async def agent(episode, model):
        await episode.phase.processes.run([sys.executable, "-c", WRITE_MINE], env={"WHO": "me"})
        return await answer(episode, model)

    monkeypatch.setattr(shutil, "rmtree", slow_rmtree)
    built = await make()
    script.run = agent
    graded, _ = await asyncio.gather(built.rig.rollout("g-r1", built.task), answer_while_removing())
    assert graded.reward == 1.0
    assert answered_in_time == [True]
    assert list(attempts.iterdir()) == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
async def test_an_agent_that_traps_sigterm_is_killed_and_graded_inside_the_backstop(make, script):
    """The tightest deadline check allows: the kill grace and the cancel grace both fit."""
    trap = (
        "import signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "sys.stdout.write('ready'); sys.stdout.flush(); time.sleep(30)"
    )

    async def agent(episode, model):
        await answer(episode, model)
        await episode.phase.processes.run([sys.executable, "-c", trap])

    # 0.5 + 0.6 + 0.2 = 1.3 < 1.4: legal, and SIGKILL lands 1.1s in, past agent + cancel grace.
    built = await make(agent_timeout=0.5, kill_grace=0.6, grace=0.2, rollout_timeout=1.4)
    script.run = agent
    script.grade = lambda episode, rollout: asyncio.sleep(0, float(episode.phase.timed_out))
    graded = await built.rig.rollout("g-r1", built.task)
    assert graded.reward == 1.0


async def blocked(episode, model):
    await answer(episode, model)
    await asyncio.sleep(30)


@pytest.mark.parametrize(
    ("target", "sandboxes"),
    [
        (2, 3),  # full target: two sampling, and one pre-started waiting for a slot
        (1, 1),  # backpressure: one sampling, and no pre-start place at all
    ],
)
async def test_sandboxes_waiting_for_a_slot_are_bounded_by_the_prestart_places(
    make, script, target, sandboxes
):
    built = await make(sampling_slots=2, prestart=1, agent_timeout=3.0, rollout_timeout=5.0)
    built.rig.admission.value = target
    script.run = blocked
    rollouts = [
        asyncio.create_task(built.rig.rollout(f"g-r{index}", built.task)) for index in range(5)
    ]
    # Settled: every rollout beyond the pre-start places is held back at admission.
    await until(lambda: built.rig.capacity.snapshot()["admission/waiting"] == 5 - sandboxes)
    await until(lambda: len(script.of("setup")) == sandboxes)
    await until(lambda: len(script.of("run")) == min(target, 2))
    assert len(live_sandboxes(built.provider)) == sandboxes
    snapshot = built.rig.capacity.snapshot()
    assert snapshot["slots/sampling"] == min(target, 2)
    assert snapshot["admission/prestarted"] == sandboxes - min(target, 2)
    for rollout in rollouts:
        rollout.cancel()
    await asyncio.gather(*rollouts, return_exceptions=True)
    await built.rig.driver.drain()
    assert live_sandboxes(built.provider) == []


async def test_a_cancel_during_provisioning_leaves_nothing_live(make, script):
    built = await make()
    built.provider.gate = asyncio.Event()  # the create hangs at the provider
    attempt = asyncio.create_task(built.rig.driver.sample(built.task, "g", "g-r1"))
    await until(lambda: built.provider.in_flight == 1)
    attempt.cancel()
    await settle_loop()  # the cancellation reaches the create, which is already sent
    built.provider.gate.set()  # the create lands inside the grace, and then the cancel goes on
    with pytest.raises(asyncio.CancelledError):
        await attempt
    await built.rig.driver.drain()
    assert script.events == []  # no episode was ever built, so none is closed
    assert built.rig.driver.live_attempts == 0
    assert live_sandboxes(built.provider) == []


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    tries: int = 3  # filled in: the raw line has no tries


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    marks: list[str] = []


class Declared(Scripted, Episode[Prompt, Rubric]):
    execution_kind = Fake


SEEN: list[tuple[dict[str, Any], int, str, list[str]]] = []


async def keep_and_spoil(episode: Scripted, *args: Any) -> None:
    """Record what the episode was given, then change it, as a careless episode might."""
    task, options = episode.task.value, episode.options
    SEEN.append((dict(episode.task.data), task.tries, options.url, list(options.marks)))
    task.tries = 0
    options.marks.append("spoiled")
    raise SandboxLost("again")


async def test_each_attempt_gets_its_own_validated_task_and_options(make, script):
    SEEN.clear()
    built = await make(Declared, options={"url": "http://judge"}, max_rollout_retries=1)
    script.setup = keep_and_spoil
    assert await built.rig.rollout("g-r1", built.task) is Discard.ROLLOUT_FAILED
    first, second = SEEN  # the retry saw none of what the first attempt changed
    assert first == second == ({"id": "t", "prompt": "q", "tries": 3}, 3, "http://judge", [])


def reachable(root: object) -> Iterator[object]:
    """Every object ``root`` reaches through attributes, containers, bound methods and closures."""
    seen: set[int] = set()
    pending = [root]
    while pending:
        current = pending.pop()
        if id(current) in seen or isinstance(current, (type, ModuleType, str, bytes, int, float)):
            continue
        seen.add(id(current))
        yield current
        if isinstance(current, Mapping):
            pending.extend([*current.keys(), *current.values()])
        elif isinstance(current, (list, tuple, set, frozenset)):
            pending.extend(current)
        if isinstance(current, MethodType):
            pending.append(current.__self__)
        if isinstance(current, FunctionType):
            pending.extend(cell.cell_contents for cell in current.__closure__ or ())
        pending.extend(getattr(current, "__dict__", {}).values())
        for kind in type(current).__mro__:
            for name in getattr(kind, "__slots__", ()):
                if hasattr(current, name):
                    pending.append(getattr(current, name))


async def test_nothing_the_closing_phase_reaches_serves_the_model(make, script):
    served = []

    async def run(episode, model):
        served.append(model)
        return await answer(episode, model)

    script.run = run
    built = await make()
    await built.rig.rollout("g-r1", built.task)
    (episode,) = script.of("init")
    policy = (Model, AttemptLease, DirectModelAccess, HttpModelAccess, AttemptContext)
    assert any(isinstance(item, policy) for item in reachable(served))  # the walk finds it
    assert [item for item in reachable(episode.phase) if isinstance(item, policy)] == []


# ------------------------------------------------------- the sandbox's client


async def client_of(episode: Scripted) -> FakeClient:
    assert episode.sandbox is not None
    return await episode.sandbox.client()


async def test_the_client_is_attached_once_and_closed_after_close_before_the_release(make, script):
    seen: list[FakeClient] = []
    at_close: list[int] = []

    async def reach(episode, *args):
        seen.append(await client_of(episode))

    async def run(episode, model):
        await reach(episode)
        return await answer(episode, model)

    async def grade(episode, rollout):
        await reach(episode)
        return await score(episode, rollout)

    async def close(episode):
        await reach(episode)
        at_close.append(seen[0].closed)

    built = await make()
    script.setup, script.run, script.grade, script.close = reach, run, grade, close
    assert (await built.rig.rollout("g-r1", built.task)).reward == 1.0
    await built.rig.driver.drain()
    first = seen[0]
    assert len(seen) == 4 and all(client is first for client in seen)
    assert calls(built.provider, "client") == 1
    assert at_close == [0] and first.closed == 1  # still open in close(), closed after it
    operations = [op for op, _ in built.provider.calls]
    assert operations.index("close_client") < operations.index("release")


async def test_concurrent_first_calls_attach_one_client(make, script):
    pair: list[FakeClient] = []

    async def both(episode, *args):
        pair.extend(await asyncio.gather(client_of(episode), client_of(episode)))

    built = await make()
    built.provider.delay = 0.05  # the first attach is still in flight when the second asks
    script.setup = both
    await built.rig.rollout("g-r1", built.task)
    await built.rig.driver.drain()
    assert pair[0] is pair[1] and calls(built.provider, "client") == 1
    assert pair[0].closed == 1


async def test_an_attempt_that_never_asks_for_the_client_attaches_none(make, script):
    built = await make()
    await built.rig.rollout("g-r1", built.task)
    await built.rig.driver.drain()
    assert calls(built.provider, "client") == calls(built.provider, "close_client") == 0


async def test_the_attach_is_bounded_by_the_providers_attach_deadline(make, script):
    async def attach(episode, model):
        await client_of(episode)

    built = await make(deadlines=replace(FAST, attach_seconds=0.1), agent_timeout=3.0)
    built.provider.blocked = {"client"}  # the provider never answers
    script.run = attach
    assert await built.rig.rollout("g-r1", built.task) is Discard.ROLLOUT_FAILED
    assert built.rig.driver.note("g-r1").cause == "rollout_failed/timeout"
    assert calls(built.provider, "close_client") == 0


async def test_a_sandbox_kept_past_the_attempt_gives_no_client(make, script):
    built = await make()
    await built.rig.rollout("g-r1", built.task)
    await built.rig.driver.drain()
    (episode,) = script.of("init")
    with pytest.raises(RuntimeError, match="its sandbox's client is closed"):
        await client_of(episode)
    assert calls(built.provider, "client") == 0


@pytest.mark.parametrize("gone", [False, True], ids=["alive", "gone"])
async def test_alive_is_the_providers_fact(make, script, gone):
    async def grade(episode, rollout):
        if gone:
            built.provider.dead.add(episode.sandbox.name)
        return float(await episode.sandbox.alive())

    built = await make()
    script.grade = grade
    assert (await built.rig.rollout("g-r1", built.task)).reward == float(not gone)
    assert calls(built.provider, "alive") == 1


CLEANUP = 0.1


@pytest.mark.parametrize("hook", ["finish", "abort"])
@cancellation_matrix(
    triggers=(Trigger.DEADLINE,),
    points={"close_client": (Landing.NOTHING, Landing.FAILURE)},
    responses=tuple(Response),
)
async def test_the_client_is_closed_once_close_has_ended_and_never_while_it_runs(
    make, script, hook, case
):
    """``close()`` is cut by the cleanup deadline; the client closes only once it has ended."""
    work = Work(case.response)
    clients: list[FakeClient] = []

    async def setup(episode):
        clients.append(await client_of(episode))

    async def close(episode):
        await work()

    built = await make(
        cleanup_timeout_seconds=CLEANUP,
        grace=0.2 if case.still_running else 2.0,
        agent_timeout=1.0,
        rollout_timeout=4.0,
    )
    gate = Gate(ConnectionError("the client would not close"))
    built.provider.closing = gate
    script.setup, script.close = setup, close
    if hook == "abort":
        script.run = raise_lost
    rollout = asyncio.create_task(built.rig.rollout("g-r1", built.task))
    try:
        await asyncio.wait_for(work.cancelled.wait(), 5)
        if case.still_running:
            await until(lambda: built.rig.driver.unquiesced != ())
            assert not gate.entered.is_set() and clients[0].closed == 0
            assert live_sandboxes(built.provider) != []  # everything close() holds stays held
            work.end()
        else:
            work.let_go.set()
        await asyncio.wait_for(gate.entered.wait(), 5)
        assert work.finished  # close() has ended before its client closes
        gate.open(case.at("close_client"))
        await outcome_within(rollout, 5)
        await built.rig.driver.drain()
    finally:
        work.end()
    assert work.cancellations == 1
    assert calls(built.provider, "close_client") == 1
    assert clients[0].closed == (0 if case.at("close_client") is Landing.FAILURE else 1)
    if not case.still_running:
        assert live_sandboxes(built.provider) == []


class SamplingSandbox(Scripted):
    sandbox_lifetime = "sampling"


@pytest.mark.parametrize("timed_out", [False, True])
async def test_sampling_sandbox_is_released_before_grading_admission(make, script, timed_out):
    built = await make(episode=SamplingSandbox, agent_timeout=0.02 if timed_out else 1.0)

    async def run(episode, model):
        await client_of(episode)
        result = await answer(episode, model)
        if timed_out:
            await asyncio.Event().wait()
        return result

    async def grade(episode, rollout):
        assert live_sandboxes(built.provider) == []
        assert built.provider.clients[0].closed == 1
        assert episode.phase.timed_out is timed_out
        with pytest.raises(RuntimeError, match="client is closed"):
            await client_of(episode)
        with pytest.raises(RuntimeError, match="client is closed"):
            await episode.sandbox.alive()
        return await score(episode, rollout)

    script.run, script.grade = run, grade
    admitted = await built.rig.driver.sample(built.task, "g", "g-r1")
    assert live_sandboxes(built.provider) == []  # before requesting verification capacity
    assert script.of("grade") == []
    graded = await built.rig.driver.grade(admitted)
    assert graded.reward == 1.0
    await built.rig.driver.drain()
    assert calls(built.provider, "create") == calls(built.provider, "release") == 1
    assert calls(built.provider, "close_client") == 1
    assert len(script.of("close")) == 1


async def test_unconfirmed_sampling_release_does_not_enter_grading_and_cleanup_retries(
    make, script
):
    built = await make(episode=SamplingSandbox, max_rollout_retries=0)
    built.provider.release_errors = [ConnectionError("release failed")]
    assert await built.rig.rollout("g-r1", built.task) is Discard.ROLLOUT_FAILED
    assert script.of("grade") == []
    await built.rig.driver.drain()
    assert live_sandboxes(built.provider) == []
    assert calls(built.provider, "release") == 2
    assert len(script.of("close")) == 1


async def test_sampling_sandbox_client_close_failure_keeps_cleanup_available(make, script):
    built = await make(episode=SamplingSandbox, max_rollout_retries=0)
    script.setup = lambda episode: client_of(episode)
    built.provider.closing = Gate(ConnectionError("close failed"))
    task = asyncio.create_task(built.rig.rollout("g-r1", built.task))
    await built.provider.closing.entered.wait()
    built.provider.closing.open(Landing.FAILURE)
    assert await task is Discard.ROLLOUT_FAILED
    await built.rig.driver.drain()
    assert script.of("grade") == []
    assert live_sandboxes(built.provider) == []


async def test_sampling_sandbox_release_survives_cancellation(make, script):
    built = await make(episode=SamplingSandbox)
    built.provider.blocked = {"release"}
    task = asyncio.create_task(built.rig.driver.sample(built.task, "g", "g-r1"))
    await until(lambda: calls(built.provider, "release") > 0)
    task.cancel()
    built.provider.unblocked.set()  # the release in flight completes
    await asyncio.gather(task, return_exceptions=True)
    await built.rig.driver.drain()
    assert script.of("grade") == []
    assert live_sandboxes(built.provider) == []
    assert calls(built.provider, "release") == 1  # the cancelled release, not a retry


class SamplingRetrySafe(SamplingSandbox):
    grading_retries = 1


async def test_sampling_lifetime_grading_retries_do_not_recreate_the_sandbox(make, script):
    built = await make(episode=SamplingRetrySafe)
    tries = 0

    async def grade(episode, rollout):
        nonlocal tries
        tries += 1
        assert live_sandboxes(built.provider) == []
        if tries == 1:
            raise ConnectionError("judge unavailable")
        return await score(episode, rollout)

    script.grade = grade
    assert (await built.rig.rollout("g-r1", built.task)).reward == 1.0
    await built.rig.driver.drain()
    assert tries == 2
    assert calls(built.provider, "create") == calls(built.provider, "release") == 1
