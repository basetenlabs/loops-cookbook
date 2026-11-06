"""Separate Episode verification environments through the real driver and registry."""

from __future__ import annotations

import asyncio

import pytest
from support.cancellation import settle_loop
from support.episode_rig import Scripted, answer, calls, live_sandboxes, score, until
from support.resources import Fake, FakeSpec, not_created

from loops_cookbook.episodes import Episode
from loops_cookbook.episodes.tasks import task_digest
from loops_cookbook.runtime.driver import Discard


class Separate(Scripted, Episode[None, None]):
    execution_kind = Fake
    grading_kind = Fake


class GradingOnly(Scripted, Episode[None, None]):
    execution_kind = None
    grading_kind = Fake


class SamplingSeparate(Separate):
    sandbox_lifetime = "sampling"


class RetrySeparate(Separate):
    grading_retries = 1


class OtherSpec(FakeSpec):
    image: str = "verifier-image"


class Other(Fake):
    provider = "other"
    spec = OtherSpec


class OtherSeparate(Scripted, Episode[None, None]):
    execution_kind = Fake
    grading_kind = Other


class RetryOtherSeparate(OtherSeparate):
    grading_retries = 1


@pytest.mark.parametrize("episode", [Separate, SamplingSeparate, GradingOnly])
async def test_verifier_is_lazy_and_both_clients_close_under_registered_cleanup(
    make, script, episode
):
    built = await make(
        episode=episode, sandbox=episode is not GradingOnly, grading_sandbox={"provider": "fake"}
    )
    seen = []

    async def run(episode, model):
        assert not hasattr(episode.phase, "grading_sandbox")
        if episode.sandbox is not None:
            seen.append(await episode.sandbox.client())
        return await answer(episode, model)

    async def grade(episode, rollout):
        assert built.rig.capacity.snapshot()["slots/verification"] == 1
        seen.append(await episode.phase.grading_sandbox.client())
        assert episode.phase.grading_sandbox.name != (
            None if episode.sandbox is None else episode.sandbox.name
        )
        return await score(episode, rollout)

    script.run, script.grade = run, grade
    admitted = await built.rig.driver.sample(built.task, "g", "g-r1")
    assert [request.logical_key for request in built.provider.created] == (
        [] if episode is GradingOnly else ["sandbox"]
    )
    assert (await built.rig.driver.grade(admitted)).reward == 1
    await built.rig.driver.drain()
    assert [request.logical_key for request in built.provider.created][-1] == "grading_sandbox"
    assert all(client.closed == 1 for client in seen)
    assert live_sandboxes(built.provider) == []
    assert len(script.of("close")) == 1
    with pytest.raises(RuntimeError, match="client is closed"):
        await script.of("init")[0].phase.grading_sandbox.client()


async def test_different_provider_kind_and_effective_spec_are_validated_and_recorded(make, script):
    built = await make(
        episode=OtherSeparate,
        grading_provider="other",
        grading_kind=Other,
        grading_sandbox={"provider": "other"},
    )

    async def grade(episode, rollout):
        assert type(episode.phase.grading_sandbox) is Other
        await episode.phase.grading_sandbox.client()
        return await score(episode, rollout)

    script.grade = grade
    assert (await built.rig.rollout("g-r1", built.task)).reward == 1
    await built.rig.driver.drain()
    assert calls(built.provider, "create") == calls(built.verifier_provider, "create") == 1
    assert live_sandboxes(built.provider) == live_sandboxes(built.verifier_provider) == []
    assert built.evidence.digests["grading_sandbox"] == task_digest({"image": "verifier-image"})


async def test_grading_retries_reuse_the_same_verifier_and_client(make, script):
    built = await make(episode=RetrySeparate, grading_sandbox={"provider": "fake"})
    clients = []

    async def grade(episode, rollout):
        clients.append(await episode.phase.grading_sandbox.client())
        if len(clients) == 1:
            raise ConnectionError("check unavailable")
        return await score(episode, rollout)

    script.grade = grade
    assert (await built.rig.rollout("g-r1", built.task)).reward == 1
    await built.rig.driver.drain()
    assert len(clients) == 2 and clients[0] is clients[1]
    assert clients[0].closed == 1
    assert calls(built.provider, "create") == 2


async def test_discarded_ungraded_rollout_never_creates_verifier(make, script):
    built = await make(episode=Separate, grading_sandbox={"provider": "fake"})
    admitted = await built.rig.driver.sample(built.task, "g", "g-r1")
    built.rig.driver.discard_admitted(admitted)
    await built.rig.driver.drain()
    assert [request.logical_key for request in built.provider.created] == ["sandbox"]
    assert live_sandboxes(built.provider) == []


async def test_verifier_provision_failure_is_a_grading_failure_and_releases_agent(make, script):
    built = await make(episode=Separate, grading_sandbox={"provider": "fake"})
    admitted = await built.rig.driver.sample(built.task, "g", "g-r1")
    built.provider.create_errors = [not_created("quota")]
    assert await built.rig.driver.grade(admitted) is Discard.GRADING_FAILED
    await built.rig.driver.drain()
    assert script.of("grade") == []
    assert live_sandboxes(built.provider) == []


async def test_slow_verifier_provisioning_counts_toward_grading_deadline(make, script):
    built = await make(
        episode=Separate,
        grading_sandbox={"provider": "fake"},
        grading_timeout_seconds=0.02,
        grace=0.5,
    )
    admitted = await built.rig.driver.sample(built.task, "g", "g-r1")
    built.provider.delay = 0.05
    assert await built.rig.driver.grade(admitted) is Discard.GRADING_FAILED
    await built.rig.driver.drain()
    assert script.of("grade") == []
    assert live_sandboxes(built.provider) == []


async def test_cancelled_verifier_provision_is_reconciled_and_both_sandboxes_released(make, script):
    built = await make(episode=Separate, grading_sandbox={"provider": "fake"})
    admitted = await built.rig.driver.sample(built.task, "g", "g-r1")
    built.provider.gate = asyncio.Event()
    grading = asyncio.create_task(built.rig.driver.grade(admitted))
    await until(lambda: calls(built.provider, "create") == 2)
    grading.cancel()
    await settle_loop()
    built.provider.gate.set()
    with pytest.raises(asyncio.CancelledError):
        await grading
    await built.rig.driver.drain()
    assert script.of("grade") == []
    assert live_sandboxes(built.provider) == []


async def test_nonquiescent_grader_keeps_both_sandboxes_live_until_it_ends(make, script):
    built = await make(
        episode=Separate,
        grading_sandbox={"provider": "fake"},
        grading_timeout_seconds=0.02,
        grace=0.02,
    )
    started, stop = asyncio.Event(), asyncio.Event()

    async def grade(episode, rollout):
        await episode.phase.grading_sandbox.client()
        started.set()
        while not stop.is_set():
            try:
                await stop.wait()
            except asyncio.CancelledError:
                continue
        return 1.0

    script.grade = grade
    admitted = await built.rig.driver.sample(built.task, "g", "g-r1")
    grading = asyncio.create_task(built.rig.driver.grade(admitted))
    await started.wait()
    assert await grading is Discard.UNQUIESCED
    assert len(live_sandboxes(built.provider)) == 2
    assert script.of("close") == []
    stop.set()
    await settle_loop()
    await built.rig.driver.drain()
    assert live_sandboxes(built.provider) == []


async def test_waiting_for_verification_capacity_does_not_provision_verifier(make, script):
    built = await make(episode=Separate, grading_sandbox={"provider": "fake"}, verification_slots=1)
    entered, leave = asyncio.Event(), asyncio.Event()

    async def grade(episode, rollout):
        if len(script.of("grade")) == 1:
            entered.set()
            await leave.wait()
        return await score(episode, rollout)

    script.grade = grade
    first = await built.rig.driver.sample(built.task, "g", "g-r1")
    active = asyncio.create_task(built.rig.driver.grade(first))
    await entered.wait()
    second = await built.rig.driver.sample(built.task, "g", "g-r2")
    waiting = asyncio.create_task(built.rig.driver.grade(second))
    await until(lambda: built.rig.capacity.snapshot()["queue/verification_waiting"] == 1)
    assert [request.logical_key for request in built.provider.created].count("grading_sandbox") == 1
    leave.set()
    assert [result.reward for result in await asyncio.gather(active, waiting)] == [1, 1]
    await built.rig.driver.drain()
    assert [request.logical_key for request in built.provider.created].count("grading_sandbox") == 2
    assert live_sandboxes(built.provider) == []


async def test_both_attached_clients_are_closed_even_when_first_close_fails(make, script):
    built = await make(episode=Separate, grading_sandbox={"provider": "fake"})

    async def setup(episode):
        await episode.sandbox.client()

    async def grade(episode, rollout):
        await episode.phase.grading_sandbox.client()
        return await score(episode, rollout)

    closing = built.provider.close_client

    async def close_client(client):
        await closing(client)
        if client.name.endswith("-sandbox"):
            raise ConnectionError("close acknowledgement failed")

    built.provider.close_client = close_client
    script.setup, script.grade = setup, grade
    assert (await built.rig.rollout("g-r1", built.task)).reward == 1
    await built.rig.driver.drain()
    assert len(built.provider.clients) == 2
    assert [client.closed for client in built.provider.clients] == [1, 1]
    assert live_sandboxes(built.provider) == []
    assert built.rig.driver.note("g-r1").cause == "cleanup_failed/connection"


async def test_failed_kind_initialization_retries_without_losing_registered_attachment(
    make, script
):
    class FlakyKind(Other):
        attempts = 0

        def __init__(self, name, environment, session):
            type(self).attempts += 1
            if self.attempts == 1:
                raise ConnectionError("kind initialization failed")
            super().__init__(name, environment, session)

    built = await make(
        episode=RetryOtherSeparate,
        grading_provider="other",
        grading_kind=FlakyKind,
        grading_sandbox={"provider": "other"},
    )

    async def grade(episode, rollout):
        await episode.phase.grading_sandbox.client()
        return await score(episode, rollout)

    script.grade = grade
    assert (await built.rig.rollout("g-r1", built.task)).reward == 1
    await built.rig.driver.drain()
    assert FlakyKind.attempts == 2
    assert calls(built.verifier_provider, "create") == calls(built.verifier_provider, "client") == 1
    assert built.verifier_provider.clients[0].closed == 1
    assert live_sandboxes(built.verifier_provider) == []
