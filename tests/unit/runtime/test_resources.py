"""Episode resources through the real driver, for a rollout runtime that is not Harbor.

The scripted runtime provisions, borrows, and releases through
``attempt.services.resources`` against ``FakeProvider``: the same contracts the
Harbor composition relies on, with no Harbor in the loop.
"""

import asyncio

import pytest
from support.harness import TASK, Runtime, default_sample, generate, rig
from support.resources import FakeProvider, episode, episodes, registry_at, request, run_resources

from loops_cookbook.resources.api import CleanupIncomplete, CleanupStatus
from loops_cookbook.runtime.api import EpisodeResult, GradeResult, Phase
from loops_cookbook.runtime.driver import Discard, RuntimeExecutionFailure, cleanup_failures


def resourced(tmp_path, provider, runtime, **settings):
    registry = registry_at(tmp_path, provider, run_id="run")
    return registry, rig(runtime, resources=episodes(registry), **settings)


def _task(rollout_id="g-r1"):
    return TASK, "g", rollout_id


async def test_an_episodes_resources_are_released_when_it_ends(tmp_path):
    provider = FakeProvider()
    owned = []

    async def sample(attempt):
        handle = await attempt.services.resources.provision(request("sandbox"))
        owned.append(handle.intent)
        env = attempt.services.resources.usage_environment(handle.intent.resource_id)
        assert env["FAKE_NAME"] == "fake-sandbox"
        return await default_sample(attempt)

    registry, r = resourced(tmp_path, provider, Runtime(on_sample=sample))
    rollout = await r.rollout("g-r1")
    assert rollout.reward == 1.0
    assert provider.store == {}
    [intent] = owned
    assert intent.owner_id == r.runtime.named("sample")[0][1].spec.identity.attempt_id
    assert registry.unresolved.results == ()
    await registry.close()


async def test_a_run_resource_is_borrowed_by_episodes_and_outlives_them(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider, run_id="run")
    run = run_resources(registry, run_id="run")
    shared = await run.provision(request("cache"))

    async def sample(attempt):
        async with await attempt.services.resources.acquire(shared.intent.resource_id) as lease:
            assert lease.handle == shared
        await attempt.services.resources.acquire(shared.intent.resource_id)  # held to the end
        return await default_sample(attempt)

    r = rig(Runtime(on_sample=sample), resources=episodes(registry))
    await r.rollout("g-r1")
    await r.rollout("g-r2")
    assert list(provider.store) == [shared.resource.provider_id]
    report = await run.close()
    assert report.complete
    assert provider.store == {}
    await registry.close()


async def test_the_setup_clock_excludes_resource_service_time(tmp_path):
    provider = FakeProvider(delay=0.1)

    async def sample(attempt):
        # Environment start, outside any hold: the setup clock's territory.
        await attempt.services.resources.provision(request("sandbox"))
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            await generate(attempt)
        return EpisodeResult()

    registry, r = resourced(
        tmp_path,
        provider,
        Runtime(on_sample=sample),
        phases=frozenset({Phase.SAMPLING}),
        setup_timeout_seconds=0.05,
    )
    rollout = await r.rollout("g-r1")
    assert rollout.reward == 1.0  # a slow provider create did not count as setup
    await registry.close()


def runner_owned(work):
    async def sample(attempt):
        # The runner's sampling hold spans all of sample(), provisioning included.
        await attempt.services.resources.provision(request("sandbox"))
        await work()
        return await default_sample(attempt)

    return sample, frozenset()


def runtime_owned(work):
    async def sample(attempt):
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            await attempt.services.resources.provision(request("sandbox"))
            await work()
            await generate(attempt)
        return EpisodeResult()

    return sample, frozenset({Phase.SAMPLING})


@pytest.mark.parametrize("hold", [runner_owned, runtime_owned])
async def test_the_rollout_clock_excludes_resource_service_time_inside_a_sampling_hold(
    tmp_path, hold
):
    sample, phases = hold(lambda: asyncio.sleep(0))
    registry, r = resourced(
        tmp_path,
        FakeProvider(delay=0.2),
        Runtime(on_sample=sample),
        phases=phases,
        rollout_timeout_seconds=0.05,
    )
    rollout = await r.rollout("g-r1")
    assert rollout.reward == 1.0  # a create longer than the rollout clock is not the agent's
    await registry.close()


@pytest.mark.parametrize("hold", [runner_owned, runtime_owned])
async def test_agent_work_after_a_slow_create_still_runs_out_the_rollout_clock(tmp_path, hold):
    sample, phases = hold(lambda: asyncio.sleep(0.2))
    registry, r = resourced(
        tmp_path,
        FakeProvider(delay=0.2),
        Runtime(on_sample=sample),
        phases=phases,
        rollout_timeout_seconds=0.1,
    )
    assert await r.rollout("g-r1") is Discard.AGENT_TIMEOUT
    await registry.close()


async def test_the_cleanup_clock_excludes_resource_service_time(tmp_path):
    provider = FakeProvider()
    owned = {}

    async def sample(attempt):
        handle = await attempt.services.resources.provision(request("sandbox"))
        owned[attempt.spec.identity.attempt_id] = handle.intent.resource_id
        return await default_sample(attempt)

    async def finish(attempt, rollout, disposition):
        provider.delay = 0.1
        await attempt.services.resources.release(owned[attempt.spec.identity.attempt_id])

    registry, r = resourced(
        tmp_path,
        provider,
        Runtime(on_sample=sample, on_finish=finish),
        cleanup_timeout_seconds=0.05,
        max_consecutive_failures=1,
    )
    await r.rollout("g-r1")
    assert not r.systemic  # a slow release is not a cleanup-hook timeout
    assert provider.store == {}
    await registry.close()


async def test_an_in_flight_release_survives_hook_cancellation(tmp_path):
    provider = FakeProvider()
    releasing = asyncio.Event()

    async def sample(attempt):
        handle = await attempt.services.resources.provision(request("sandbox"))
        provider.gate = asyncio.Event()
        releasing.set()
        await attempt.services.resources.release(handle.intent.resource_id)
        return await default_sample(attempt)

    registry, r = resourced(tmp_path, provider, Runtime(on_sample=sample))
    sampling = asyncio.create_task(r.driver.sample(*_task()))
    await releasing.wait()
    await asyncio.sleep(0)
    sampling.cancel()
    for _ in range(20):
        await asyncio.sleep(0)
    assert provider.in_flight == 1  # the release is still at the provider
    provider.gate.set()
    await asyncio.gather(sampling, return_exceptions=True)
    await registry.close()
    assert provider.store == {}
    assert registry.unresolved.results == ()


async def test_cancelling_an_attempt_mid_create_leaves_nothing_behind(tmp_path):
    """The create finishes at the provider after the cancel; the attempt still releases it."""
    provider = FakeProvider(gate=asyncio.Event())

    async def sample(attempt):
        await attempt.services.resources.provision(request("sandbox"))
        return await default_sample(attempt)

    registry, r = resourced(tmp_path, provider, Runtime(on_sample=sample))
    sampling = asyncio.create_task(r.driver.sample(*_task()))
    async with asyncio.timeout(2):
        while provider.in_flight == 0:  # the create is at the provider
            await asyncio.sleep(0)
    sampling.cancel()
    await asyncio.sleep(0.05)
    provider.gate.set()
    with pytest.raises(asyncio.CancelledError):
        await sampling
    await r.driver.drain()
    assert provider.store == {} and registry.unresolved.results == ()
    await registry.close()


async def test_a_shared_prerequisite_survives_one_waiters_cancellation(tmp_path):
    """Two attempts wait for one image; cancelling the first must not abandon it."""
    provider = FakeProvider(gate=asyncio.Event())
    provider.needs["box"] = [request("image", kind="image")]
    waiting = []

    async def sample(attempt):
        waiting.append(attempt.spec.identity.rollout_id)
        await attempt.services.resources.provision(request("sandbox"))
        return await default_sample(attempt)

    registry, r = resourced(tmp_path, provider, Runtime(on_sample=sample))
    first = asyncio.create_task(r.driver.sample(*_task("g-r1")))
    second = asyncio.create_task(r.rollout("g-r2"))
    async with asyncio.timeout(2):
        while len(waiting) < 2 or provider.in_flight == 0:  # both wait on the image build
            await asyncio.sleep(0)
    first.cancel()
    await asyncio.sleep(0.05)
    provider.gate.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    graded = await asyncio.wait_for(second, 5)
    assert not isinstance(graded, Discard)  # the second attempt got its image and sandbox
    assert [verb for verb, name in provider.calls if name == "fake-image"].count("create") == 1
    await r.driver.drain()
    assert {resource.name for resource in provider.store.values()} == {"fake-image"}  # the run's
    await registry.release_all()
    assert provider.store == {} and registry.unresolved.results == ()
    await registry.close()


async def test_incomplete_episode_cleanup_counts_and_keeps_every_identity(tmp_path, caplog):
    provider = FakeProvider()

    async def sample(attempt):
        await attempt.services.resources.provision(request("sandbox"))
        provider.release_errors.append(ConnectionError("reset"))
        raise ValueError("the agent failed")

    registry, r = resourced(tmp_path, provider, Runtime(on_sample=sample))
    assert await r.rollout() is Discard.ROLLOUT_FAILED
    [unresolved] = registry.unresolved.results
    assert unresolved.status is CleanupStatus.UNCONFIRMED
    primaries = [record.exc_info[1] for record in caplog.records if record.exc_info]
    attached = [failure for primary in primaries for failure in cleanup_failures(primary)]
    [incomplete] = [failure for failure in attached if isinstance(failure, CleanupIncomplete)]
    assert incomplete.report.unresolved == (unresolved,)
    await registry.close()


async def test_resource_contracts_hold_without_a_model_call(tmp_path):
    """A rollout runtime may use resources in grading too; the attempt still owns them."""
    provider = FakeProvider()

    async def grade(attempt, rollout):
        handle = await attempt.services.resources.provision(request("verifier"))
        assert handle.intent.owner_id == attempt.spec.identity.attempt_id
        return GradeResult(reward=1.0)

    async def sample(attempt):
        await generate(attempt)
        return EpisodeResult()

    registry, r = resourced(tmp_path, provider, Runtime(on_sample=sample, on_grade=grade))
    await r.rollout("g-r1")
    assert provider.store == {}
    await registry.close()


async def test_a_resource_service_that_fails_to_close_is_a_cleanup_failure_not_a_leak():
    class Broken:
        async def close(self):
            raise OSError("journal unavailable")

    r = rig(Runtime(), resources=lambda identity, phases: Broken(), max_consecutive_failures=1)
    rollout = await r.rollout("g-r1")
    assert rollout.reward == 1.0
    assert r.systemic and "cleanup" in str(r.systemic[0])  # counted, as cleanup
    assert r.driver.live_attempts == 0  # the attempt still settled
    assert r.capacity.snapshot()["slots/sampling"] == 0


async def test_a_failed_resource_close_is_kept_beside_a_failed_cleanup_hook(caplog):
    class Broken:
        async def close(self):
            raise OSError("journal unavailable")

    async def finish(attempt, rollout, disposition):
        raise RuntimeError("sandbox stop failed")

    r = rig(Runtime(on_finish=finish), resources=lambda identity, phases: Broken())
    await r.rollout("g-r1")
    logged = [record.exc_info[1] for record in caplog.records if record.exc_info]
    [failure] = [
        error
        for error in logged
        if isinstance(error, RuntimeExecutionFailure) and error.phase == "finish"
    ]
    assert any(isinstance(extra, OSError) for extra in cleanup_failures(failure))


@pytest.mark.parametrize("unconfirmed", [False, True])
async def test_early_release_returns_truthful_proof_and_preserves_recovery(tmp_path, unconfirmed):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider, run_id="run")
    resources = episode(registry)
    handle = await resources.provision(request("sandbox"))
    if unconfirmed:
        provider.release_errors = [ConnectionError("delete failed")]
    result = await resources.release(handle.intent.resource_id)
    assert result.resource_id == handle.intent.resource_id
    assert result.status is (CleanupStatus.UNCONFIRMED if unconfirmed else CleanupStatus.RELEASED)
    assert (await resources.close()).complete
    assert provider.store == {}
    await registry.close()


async def test_early_release_reports_deferred_then_scope_cleanup_releases_dependants(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider, run_id="run")
    resources = episode(registry)
    sandbox = await resources.provision(request("sandbox"))
    await resources.provision(request("child", depends_on=(sandbox.intent.resource_id,)))
    result = await resources.release(sandbox.intent.resource_id)
    assert result.status is CleanupStatus.DEFERRED
    assert len(provider.store) == 2
    assert (await resources.close()).complete
    assert provider.store == {}
    # Proven repeats do not delete again.
    released = await resources.release(sandbox.intent.resource_id)
    assert released.status is CleanupStatus.RELEASED
    assert [op for op, _ in provider.calls].count("release") == 2
    await registry.close()
