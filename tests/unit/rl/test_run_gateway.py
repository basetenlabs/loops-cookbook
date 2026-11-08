"""A rollout runtime that reaches the model over the run's OpenAI-compatible gateway."""

import asyncio

import pytest
from support.audit import audit_run
from support.fakes import MemoryTrainer

from loops_cookbook.model_access.api import ModelAccessMode
from loops_cookbook.runtime.api import EpisodeResult, GradeResult, RuntimeCapabilities
from loops_cookbook.runtime.context import RuntimeConfigurationError

from .conftest import MemoryRuntime, records


class HttpRuntime(MemoryRuntime):
    """The memory runtime, reaching the model over the gateway with the official client."""

    capabilities = RuntimeCapabilities(model_access_modes=frozenset({ModelAccessMode.OPENAI_HTTP}))

    def __init__(self):
        super().__init__()
        self.connections = []

    async def sample(self, attempt):
        import openai

        connection = await attempt.services.model_access.http_connection()
        self.connections.append(connection)
        async with openai.AsyncOpenAI(
            base_url=connection.base_url,
            api_key=connection.api_key.get_secret_value(),
            max_retries=0,
        ) as client:
            await client.chat.completions.create(
                model=connection.model, messages=[{"role": "user", "content": "x"}]
            )
        return EpisodeResult()


class HeldTrainer(MemoryTrainer):
    """Holds a generation open until released, so a client can walk away from it."""

    def __init__(self):
        super().__init__()
        self.held = asyncio.Event()
        self.release = asyncio.Event()

    async def sample_async(self, prompt, **kwargs):
        if kwargs["sampling_params"].seed == 7:
            self.held.set()
            await self.release.wait()
        return await super().sample_async(prompt, **kwargs)


class DroppingHttpRuntime(HttpRuntime):
    """Each episode abandons one request, and keeps one of three candidates it received."""

    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    async def sample(self, attempt):
        import aiohttp
        import openai

        connection = await attempt.services.model_access.http_connection()
        auth = {"Authorization": f"Bearer {connection.api_key.get_secret_value()}"}
        async with aiohttp.ClientSession(headers=auth) as session:
            self.trainer.held.clear()
            self.trainer.release.clear()
            abandoned = asyncio.create_task(
                session.post(
                    connection.base_url + "/chat/completions",
                    json={
                        "model": connection.model,
                        "messages": [{"role": "user", "content": "x"}],
                        "seed": 7,
                    },
                )
            )
            await self.trainer.held.wait()
            abandoned.cancel()  # the client walks away while its generation runs
            await asyncio.sleep(0.05)
            self.trainer.release.set()
            async with openai.AsyncOpenAI(
                base_url=connection.base_url, api_key=auth["Authorization"][7:], max_retries=0
            ) as client:
                _kept, *losers = await asyncio.gather(
                    *(
                        client.chat.completions.create(
                            model=connection.model, messages=[{"role": "user", "content": "x"}]
                        )
                        for _ in range(3)
                    )
                )
            for loser in losers:
                async with session.post(
                    connection.base_url + "/loops/discard", json={"id": loser.id}
                ) as response:
                    assert response.status == 200
        return EpisodeResult()


@pytest.mark.parametrize("disposition", ["trained", "discarded"])
async def test_each_rollout_record_counts_its_undelivered_and_agent_discarded_generations(
    rl_run, disposition
):
    """Dropped at seal and discarded by the agent are separate facts, per rollout and per run."""
    trainer = HeldTrainer()
    runtime = DroppingHttpRuntime(trainer)
    if disposition == "discarded":

        async def uniform(attempt, rollout):
            return GradeResult(reward=0.0)

        runtime.grade = uniform
    run, _, _ = rl_run(
        trainer,
        runtime,
        max_steps=1,
        max_concurrent_rollouts=1,
        max_consecutive_unusable_groups=1,
        trace="full",
    )
    outcome = await asyncio.wait_for(run.execute(), 20)
    recorded = records(run)
    assert recorded and {r["disposition"] for r in recorded} == {disposition}
    assert {
        (r["generations_undelivered"], r["generations_discarded_by_agent"]) for r in recorded
    } == {(1, 2)}
    gauges = run._discard_gauges()
    sealed = len(recorded)
    assert gauges["discard/undelivered_generations"] == sealed
    assert gauges["discard/generations"] == 3 * sealed  # both kinds, as before
    if disposition == "trained":
        assert outcome.ok
        assert all(len(r["sample_ids"]) == 1 for r in recorded)  # only the kept candidate
        audit_run(run.out)


async def test_a_runtime_trains_over_the_gateway_and_shutdown_revokes_its_credentials(rl_run):
    import aiohttp

    run, _, runtime = rl_run(runtime=HttpRuntime(), max_steps=1)
    outcome = await asyncio.wait_for(run.execute(), 20)
    assert outcome.ok and outcome.steps == 1
    assert {c.model for c in runtime.connections} == {run.config.model.base_model}
    assert len({c.api_key.get_secret_value() for c in runtime.connections}) == len(
        runtime.connections
    )  # one credential per attempt
    assert all(sample.sample_id for rollout in runtime.samples for sample in rollout.samples)
    connection = runtime.connections[0]
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(connection.base_url + "/models") as response:
                assert response.status == 401  # the gateway may even be gone entirely
        except aiohttp.ClientConnectionError:
            pass


async def test_shutdown_closes_the_gateway_listener(rl_run):
    import aiohttp

    run, _, runtime = rl_run(runtime=HttpRuntime(), max_steps=1)
    assert (await asyncio.wait_for(run.execute(), 20)).ok
    origin = runtime.connections[0].base_url.removesuffix("/v1")
    # Step 5 itself: the listener is gone, not merely refusing a sealed credential.
    async with aiohttp.ClientSession() as session:
        with pytest.raises(aiohttp.ClientConnectionError):
            await session.get(origin + "/v1/models")


async def test_the_gateway_gets_the_configured_endpoint_and_model(rl_run):
    endpoint = {
        "bind": "0.0.0.0:0",
        "advertised_base_url": "http://sandbox-host:9000",
        "model_name": "policy-x",
    }

    async def gateway(runtime, **settings):
        run, _, _ = rl_run(runtime=runtime, **settings)
        opened = await run.preflight()
        try:
            return run._open_gateway()
        finally:
            await run.close_runtime(opened[0])
            await run.close_resources()

    http = await gateway(HttpRuntime(), **{"rollout_runtime.openai": endpoint})
    assert http is not None and http.model == "policy-x"
    assert http._settings.bind == "0.0.0.0:0"
    assert http._settings.advertised_base_url == "http://sandbox-host:9000"
    assert await gateway(MemoryRuntime()) is None


async def test_gateway_settings_for_a_runtime_that_declares_direct_access_are_refused(
    rl_run,
):
    run, _, runtime = rl_run(**{"rollout_runtime.openai": {"model_name": "policy"}})
    with pytest.raises(
        RuntimeConfigurationError, match=r"\[rollout_runtime.openai\] is set, and the"
    ):
        await asyncio.wait_for(run.execute(), timeout=5)
    assert runtime.samples == []
