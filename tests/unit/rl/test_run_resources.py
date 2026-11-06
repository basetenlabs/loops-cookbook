"""What a run hands its runtime from its providers, and the run resources it releases."""

import asyncio
import dataclasses
import time

import pytest
from support.resources import FAST, FakeProvider, registry_at, request, run_resources

from loops_cookbook.resources.api import ProvisioningFailure
from loops_cookbook.runtime.api import RunLimits

from .conftest import KeepingTrainer, MemoryRuntime


class Provisioning(MemoryRuntime):
    """Provisions a run resource named ``name`` as it prepares; ``kept`` holds its handle,
    or, when ``waits`` is false, the provisioning still under way."""

    def __init__(self, name="shared-cache", *, waits=True):
        super().__init__()
        self.name, self.waits = name, waits
        self.kept = []

    async def prepare(self):
        provisioning = self.context.run_resources.provision(request(self.name))
        if self.waits:
            self.kept.append(await provisioning)
        else:
            self.kept.append(asyncio.ensure_future(provisioning))
        return await super().prepare()


@pytest.mark.parametrize("providers", [0, 2], ids=["no_providers", "two_providers"])
async def test_the_runtime_receives_the_runs_actual_limits(rl_run, providers):
    adapters = {
        f"p{index}": FakeProvider(
            name=f"p{index}", deadlines=dataclasses.replace(FAST, attach_seconds=index + 1)
        )
        for index in range(providers)
    }
    run, _, runtime = rl_run(
        adapters=adapters,
        max_steps=1,
        max_concurrent_rollouts=3,
        max_concurrent_verifiers=2,
        rollout_timeout_seconds=41,
        prestart_allowance=0.5,
        **{"rollout.grading.timeout_seconds": 17, "sampling.max_trajectory_tokens": 65536},
    )
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok
    deadlines = {name: adapter.deadlines for name, adapter in adapters.items()}
    limits = RunLimits(
        3,
        2,
        41,
        17,
        cancellation_grace_seconds=1,
        process_kill_grace_seconds=1,
        provider_deadlines=deadlines,
        prestart_places=2,
        trajectory_tokens=65536,
    )
    assert runtime.context.limits == limits


async def test_the_runtime_receives_each_providers_own_settings(rl_run):
    """A provider's environment reads its table from the run, not a second copy."""
    pytest.importorskip("daytona")
    from support.daytona_api import CREDENTIALS, SETTINGS, FakeApi

    from loops_cookbook.providers.daytona.adapter import DaytonaAdapter

    run, _, runtime = rl_run(
        adapters={"daytona": DaytonaAdapter(SETTINGS, CREDENTIALS, FakeApi())}, max_steps=1
    )
    assert (await asyncio.wait_for(run.execute(), timeout=5)).ok
    assert runtime.context.provider_settings == {"daytona": SETTINGS}


def test_provider_sections_need_their_adapters(rl_run):
    with pytest.raises(ValueError, match="providers"):
        rl_run(providers={"fake": {}})


async def test_a_resumed_run_keeps_its_run_id_and_releases_what_its_last_process_left(
    rl_run, caplog
):
    provider = FakeProvider()
    first, _, _ = rl_run(KeepingTrainer(), adapters={"fake": provider}, max_steps=1)
    assert (await first.execute()).ok
    state_dir = first.out / "state"
    # A later process of the run dies holding a run resource and an unprovable one.
    crashed = registry_at(state_dir, provider, run_id=first.run_id)
    await run_resources(crashed, run_id=first.run_id).provision(request("left"))
    stuck = await run_resources(crashed, run_id=first.run_id).provision(request("stuck"))
    provider.stubborn.add("fake-stuck")
    crashed.journal.close()

    class Recording(MemoryRuntime):
        async def sample(self, attempt):
            provider.calls.append(("sample", attempt.spec.identity.attempt_id))
            return await super().sample(attempt)

    trainer = KeepingTrainer()
    trainer.policy_version = 1
    provider.calls.clear()
    resumed, _, _ = rl_run(
        trainer,
        Recording(),
        adapters={"fake": provider},
        max_steps=2,
        resume_from=str(first.out / "checkpoints" / "step-0.json"),
    )
    assert resumed.run_id == first.run_id and resumed.out == first.out
    outcome = await resumed.execute()
    verbs = [verb for verb, _ in provider.calls]
    assert ("release", "fake-left") in provider.calls
    assert provider.calls.index(("release", "fake-left")) < verbs.index("sample")
    assert "fake-left" not in {resource.name for resource in provider.store.values()}
    # The unprovable one is reported, kept for `loops cleanup`, and the run went ahead.
    assert outcome.steps == 1 and outcome.status == "unclean"
    assert "loops cleanup" in caplog.text and stuck.intent.resource_id in caplog.text
    reopened = registry_at(state_dir, provider, mode="reconcile")
    assert [r.resource_id for r in reopened.unresolved.results] == [stuck.intent.resource_id]
    reopened.journal.close()


async def test_shutdown_releases_the_run_scope(rl_run):
    provider = FakeProvider()
    run, _, _ = rl_run(runtime=Provisioning(), adapters={"fake": provider}, max_steps=1)
    outcome = await run.execute()
    assert outcome.ok and outcome.status == "completed"
    assert ("create", "fake-shared-cache") in provider.calls and provider.store == {}


async def test_a_shutdown_that_cannot_release_a_resource_is_unclean_and_says_how_to_recover(
    rl_run, caplog
):
    provider = FakeProvider()
    provider.stubborn.add("fake-shared-cache")
    run, _, runtime = rl_run(runtime=Provisioning(), adapters={"fake": provider}, max_steps=1)
    outcome = await run.execute()
    assert outcome.steps == 1 and outcome.status == "unclean"
    (handle,) = runtime.kept
    assert "loops cleanup" in caplog.text and handle.intent.resource_id in caplog.text
    reopened = registry_at(run.out / "state", provider, mode="reconcile")
    assert [r.resource_id for r in reopened.unresolved.results] == [handle.intent.resource_id]
    reopened.journal.close()


async def test_a_shutdown_whose_adapter_fails_to_close_still_reports_what_is_left(rl_run, caplog):
    """The registry's close raises, but the unreleased resource is still reported for
    ``loops cleanup``."""
    provider = FakeProvider()
    provider.stubborn.add("fake-shared-cache")

    async def fails():
        raise ConnectionError("the client would not close")

    provider.close = fails
    run, _, runtime = rl_run(runtime=Provisioning(), adapters={"fake": provider}, max_steps=1)
    with pytest.raises(ConnectionError, match="would not close"):
        await run.execute()
    (handle,) = runtime.kept
    assert "loops cleanup" in caplog.text and handle.intent.resource_id in caplog.text


async def test_shutdown_ends_within_its_bound_with_a_run_create_still_in_flight(rl_run, caplog):
    """A create that would hold shutdown for minutes is cut off at shutdown_drain_seconds,
    recorded as ambiguous, and reported for `loops cleanup`."""

    @dataclasses.dataclass
    class SlowCreate(FakeProvider):
        async def create(self, intent):
            if intent.desired_name == "fake-slow":
                await asyncio.Event().wait()  # an image build, as far as shutdown can tell
            return await super().create(intent)

    provider = SlowCreate(deadlines=dataclasses.replace(FAST, create_seconds=600))
    run, _, runtime = rl_run(
        runtime=Provisioning("slow", waits=False),
        adapters={"fake": provider},
        max_steps=1,
        shutdown_drain_seconds=0.3,
    )
    started = time.monotonic()
    outcome = await run.execute()
    assert time.monotonic() - started < 5  # not the create's 600 s
    [slow] = runtime.kept
    with pytest.raises(ProvisioningFailure, match="closing"):
        await slow
    assert outcome.steps == 1 and outcome.status == "unclean"
    assert "loops cleanup" in caplog.text and "JournalError" not in caplog.text


async def test_a_run_whose_journal_cannot_open_still_closes_its_adapters(rl_run, tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    provider = FakeProvider()
    run, _, _ = rl_run(adapters={"fake": provider}, **{"resources.state_dir": str(blocker)})
    with pytest.raises(OSError):
        await asyncio.wait_for(run.execute(), timeout=5)
    assert provider.closed == 1
