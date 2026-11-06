"""The Loops trainer both recipes share: provisioning, starting from saved state, what
it keeps on the way out, and whether its Loops run has ended."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from baseten.loops.models import Datum, ModelInput, TensorData
from support.builders import make_config, make_rollout, make_sample
from support.fakes import FakeClient, Future
from support.rl_sft import LoopsClient, sft_config

from loops_cookbook.rl import trainer as rl
from loops_cookbook.rl.grpo import build_rows
from loops_cookbook.sft import trainer as sft
from loops_cookbook.trainer import Capacity, stepped_version

INIT_FROM = "bt://loops:source/weights/step-249"
RESUME_FROM = "bt://loops:run-1/state/step-6"


class Recipe:
    """One recipe's trainer, built from section overrides, and one update as its run makes it."""

    def __init__(self, name, tmp_path):
        self.name = name
        self._tmp_path = tmp_path

    def trainer(self, client=None, *, resume_from=None, **sections):
        if self.name == "rl":
            config = make_config(
                **{
                    f"{section}.{key}": value
                    for section, overrides in sections.items()
                    for key, value in overrides.items()
                }
            )
            trainer = rl.Trainer(config, resume_from)
        else:
            trainer = sft.Trainer(sft_config(self._tmp_path, **sections), resume_from)
        if client is not None:
            trainer.client = client
        return trainer

    async def update(self, trainer, step=0):
        if self.name == "sft":
            return await trainer.update(step, [sft_datum()], learning_rate=2e-4)
        await trainer.accumulate(rl_datums())
        return await trainer.commit(
            step,
            groups=1,
            rollouts=2,
            rewards=[1.0, 0.0],
            advantages_=[0.5, -0.5],
            sampled_versions=[0],
        )


def rl_datums():
    rows = []
    for index in range(2):
        rollout = make_rollout(
            task_id="t",
            group_id="g",
            rollout_id=f"r{index}",
            samples=(make_sample(f"s{index}", (1, 2, 3), (4, 5), (-0.1, -0.2), 0),),
        )
        rows.extend(build_rows(rollout, 0.5))
    return rows


def sft_datum():
    return Datum(
        model_input=ModelInput.from_ints([0, 1]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=[1, 1], dtype="int64", shape=[2]),
            "weights": TensorData(data=[0.0, 1.0], dtype="float32", shape=[2]),
        },
    )


@pytest.fixture(params=["rl", "sft"])
def recipe(request, tmp_path):
    return Recipe(request.param, tmp_path)


def fail_once(client, error):
    """The client's next policy-version read raises ``error``; later reads answer."""

    def read(version):
        client.read_version = None
        raise error

    client.read_version = read


# --- provisioning ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("sections", "opened", "provisioned"),
    [
        (
            {},
            {"reuse_from_run_id": None, "availability_model": "standard"},
            {"replicas": 1, "seed": None},
        ),
        (
            {"compute": {"trainer_replicas": 2, "spot_job": True}, "training": {"seed": 17}},
            {"reuse_from_run_id": None, "availability_model": "spot"},
            {"replicas": 2, "seed": 17},
        ),
        (
            {"compute": {"reuse_from_run_id": "warm-run"}},
            {"reuse_from_run_id": "warm-run", "availability_model": "standard"},
            {"replicas": 1, "seed": None},
        ),
    ],
    ids=["defaults", "replicas, spot and seed", "pod reuse"],
)
async def test_compute_and_seed_settings_reach_provisioning(
    recipe, loops_service, sections, opened, provisioned
):
    """Unset settings leave the Loops defaults; only RL asks for a sampler."""
    await recipe.trainer(**sections).start()
    assert loops_service.opened == [opened]
    [call] = loops_service.provisioned
    assert {key: call[key] for key in provisioned} == provisioned
    assert call["with_sampler"] == (recipe.name == "rl")


@pytest.mark.parametrize(
    ("scope", "deployment_id"), [("dedicated", "7wljz0q"), ("serverless", None)]
)
async def test_a_run_is_created_on_the_configured_scope_and_its_record_agrees(
    recipe, loops_service, scope, deployment_id
):
    """The scope is asked for explicitly, never left to Loops; a serverless run is one
    with no trainer deployment of its own, as its run record states."""
    loops_service.deployment_id = deployment_id
    trainer = recipe.trainer(compute={"scope": scope})
    await trainer.start()
    [call] = loops_service.provisioned
    assert call["scope"] == scope
    assert trainer.capacity is Capacity(scope)


@pytest.mark.parametrize(
    ("scope", "deployment_id"), [("dedicated", None), ("serverless", "7wljz0q")]
)
async def test_a_run_whose_record_disagrees_with_its_scope_is_refused(
    recipe, loops_service, scope, deployment_id
):
    """A run on other capacity than asked for would time evals and bind sampler handles
    by the wrong rules, so it does not start."""
    loops_service.deployment_id = deployment_id
    trainer = recipe.trainer(compute={"scope": scope})
    with pytest.raises(RuntimeError, match=f"asked for {scope} capacity but reports"):
        await trainer.start()


@pytest.mark.parametrize("deactivates", [True, False], ids=["ended", "left up"])
async def test_a_refused_run_is_ended_whatever_keep_run_says_or_said_to_be_what_loops_placed(
    recipe, loops_service, caplog, deactivates
):
    """Its ID is not in the run record yet, so nothing else would end it; left up, it
    is reported as the capacity it bills as, not the one asked for."""
    loops_service.deployment_id = "7wljz0q"  # dedicated, though serverless was asked for
    if not deactivates:
        loops_service.deactivate_error = RuntimeError("Loops is down")
    trainer = recipe.trainer(compute={"scope": "serverless", "keep_run": True})
    with pytest.raises(RuntimeError, match="asked for serverless capacity"):
        await trainer.start()
    await trainer.close()
    assert loops_service.deactivated == (["run-1"] if deactivates else [])
    assert ("may be billing" in caplog.text) is not deactivates
    assert "serverless slot" not in caplog.text


async def test_a_refused_run_is_ended_even_when_its_client_fails_to_close(recipe, loops_service):
    """Ended first, then closed: a close that fails neither leaves it up nor hides why
    it was refused."""
    loops_service.client.aclose = AsyncMock(side_effect=RuntimeError("close failed"))
    trainer = recipe.trainer(compute={"scope": "serverless"})
    with pytest.raises(RuntimeError, match="asked for serverless capacity"):
        await trainer.start()
    assert loops_service.deactivated == ["run-1"]


async def test_a_refused_run_cancelled_while_it_is_ended_says_it_may_be_up(
    recipe, loops_service, caplog
):
    """Its ID is in no run record yet, so the log is the only place that names it."""
    ending = asyncio.Event()

    async def deactivate_run_async(run_id):
        ending.set()
        await asyncio.Event().wait()

    loops_service.deactivate_run_async = deactivate_run_async
    trainer = recipe.trainer(compute={"scope": "serverless"})
    start = asyncio.create_task(trainer.start())
    await asyncio.wait_for(ending.wait(), 5)
    start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start
    assert "Loops run run-1 is still up and may be billing" in caplog.text


@pytest.mark.parametrize(
    ("scope", "deployment_id"), [("dedicated", "7wljz0q"), ("serverless", None)]
)
async def test_weights_made_servable_for_an_eval_come_from_a_run_on_the_configured_scope(
    loops_service, scope, deployment_id
):
    """`loops eval` loads trainer state on a run of its own: placed like the run it
    evaluates, never where Loops would choose."""
    loops_service.deployment_id = deployment_id
    trainer = rl.Trainer(make_config(**{"compute.scope": scope}), None)
    await trainer.servable_weights("bt://loops:r/weights/step-3", "eval-step-3")
    [call] = loops_service.provisioned
    assert call["scope"] == scope


@pytest.mark.parametrize("fails", [False, True])
async def test_cancelled_provisioning_settles_before_cleanup(loops_service, fails, caplog):
    entered, release = threading.Event(), threading.Event()
    client = FakeClient()
    client.aclose = AsyncMock()

    def provision(**kwargs):
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test provisioning timed out")
        if fails:
            raise RuntimeError("provisioning failed")
        return client

    loops_service.create_lora_training_client = provision
    t = rl.Trainer(make_config())
    startup = asyncio.create_task(t.start())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        startup.cancel()
        await asyncio.sleep(0)
        startup.cancel()
        await asyncio.sleep(0)
        assert not startup.done()
        assert t.client is None
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(startup, timeout=3)
    await t.close()
    assert client.published == []
    if fails:
        assert t.client is None
        assert "Provisioning failed while startup" in caplog.text
        client.aclose.assert_not_awaited()
    else:
        assert t.client is client
        assert client.run_id in caplog.text
        client.aclose.assert_awaited_once()


async def test_start_refuses_a_model_loops_does_not_serve_before_provisioning(
    monkeypatch, loops_capabilities, loops_service
):
    monkeypatch.setenv("BASETEN_API_KEY", "key")
    loops_capabilities.models = [{"model_name": "Qwen/Qwen3.5-9B", "max_enabled_seq_len": 4096}]
    t = rl.Trainer(make_config(base_model="Qwen/Qwen3.5-9B", max_seq_len=8192))

    with pytest.raises(ValueError, match="max_seq_len 8192 is above the 4096"):
        await t.start()

    # The check asks the client that provisions, so the client exists; nothing is provisioned.
    assert not loops_service.provisioned


# --- what a start loads -----------------------------------------------------------


@pytest.mark.parametrize(
    ("training", "resume_from", "loads"),
    [
        ({}, None, []),
        # The weights alone, with fresh Adam moments.
        ({"init_from": INIT_FROM}, None, [("load_state", INIT_FROM)]),
        (
            {"init_from": INIT_FROM, "init_from_optimizer": True},
            None,
            [("load_state_with_optimizer", INIT_FROM)],
        ),
        # load_state would reset the Adam moments that made the run mean something.
        ({}, RESUME_FROM, [("load_state_with_optimizer", RESUME_FROM)]),
        # The resumed state already holds the weights the run started from.
        ({"init_from": INIT_FROM}, RESUME_FROM, [("load_state_with_optimizer", RESUME_FROM)]),
    ],
    ids=["fresh", "init_from", "init_from with optimizer", "resume", "resume of an init_from run"],
)
async def test_a_start_loads_the_state_it_resumes_or_the_weights_it_starts_from(
    recipe, loops_service, training, resume_from, loads
):
    await recipe.trainer(resume_from=resume_from, training=training).start()
    assert loops_service.client.loads == loads


@pytest.mark.parametrize(
    ("training", "resume_from"),
    [({"init_from": INIT_FROM}, None), ({}, RESUME_FROM)],
    ids=["init_from", "resume"],
)
async def test_a_load_that_reports_no_step_is_refused(recipe, loops_service, training, resume_from):
    """The SDK defaults a missing ``step`` to 0, a real version: read as one, a run
    started from step-6 weights would count every rollout of its first publication
    as stale."""
    loops_service.client.restored = None
    with pytest.raises(RuntimeError, match="no restored step"):
        await recipe.trainer(resume_from=resume_from, training=training).start()


# --- versions and restarts --------------------------------------------------------


@pytest.mark.parametrize(
    ("started_at", "step"),
    [(0, 0), (250, 0), (257, 7)],
    ids=["fresh", "init_from", "resume of an init_from run"],
)
async def test_the_last_applied_update_is_kept_on_the_way_out(recipe, started_at, step):
    """A run started from another run's weights keeps the source's policy version, so
    its trainer is ahead of its own step count; that is not a restart."""
    client = LoopsClient(policy_version=started_at)
    trainer = recipe.trainer(client)
    await recipe.update(trainer, step)
    assert await trainer.checkpoint_state_only(step) == f"bt://loops:run-1/state/step-{step}"
    assert [name for name, _ in client.state_checkpoints] == [f"step-{step}"]
    # Nothing staged for the sampler: resuming does not need it.
    assert not client.sampler_checkpoints


async def test_a_way_out_that_cannot_save_keeps_nothing_and_does_not_raise(recipe):
    client = LoopsClient()
    trainer = recipe.trainer(client)
    await recipe.update(trainer)
    client.save_state_async = AsyncMock(side_effect=RuntimeError("trainer is gone"))
    assert await trainer.checkpoint_state_only(0) is None


@pytest.mark.parametrize("started_at", [0, 250], ids=["fresh", "init_from"])
async def test_a_trainer_that_restarted_is_not_kept_on_the_way_out(recipe, started_at):
    """A trainer pod that restarted after step 0 is back on base weights at version 0:
    saving it as ``step-0`` would give a resume an untrained state under a trained name."""
    client = LoopsClient(policy_version=started_at)
    trainer = recipe.trainer(client)
    await recipe.update(trainer)
    client.policy_version = 0
    assert await trainer.checkpoint_state_only(0) is None
    assert not client.state_checkpoints


async def test_a_step_this_trainer_did_not_apply_is_not_kept(recipe):
    client = LoopsClient()
    trainer = recipe.trainer(client)
    assert await trainer.checkpoint_state_only(0) is None
    await recipe.update(trainer, step=3)
    assert await trainer.checkpoint_state_only(2) is None
    assert not client.state_checkpoints


async def test_a_trainer_whose_version_cannot_be_read_is_not_kept(recipe):
    """A trainer that cannot say which step it holds is not trusted to name one."""
    client = LoopsClient()
    trainer = recipe.trainer(client)
    await recipe.update(trainer)
    fail_once(client, httpx.ConnectError("trainer is gone"))
    assert await trainer.checkpoint_state_only(0) is None
    assert not client.state_checkpoints


async def test_a_trainer_that_serves_no_version_is_kept(recipe):
    """A serverless trainer has no version to read back; its state is the broker's
    lineage, not a pod that can restart under the run."""
    client = LoopsClient()

    def serverless(version):
        raise NotImplementedError("policy_version is unavailable for serverless trainers")

    client.read_version = serverless
    trainer = recipe.trainer(client)
    await recipe.update(trainer)
    assert await trainer.checkpoint_state_only(0) == "bt://loops:run-1/state/step-0"


def test_a_step_result_with_no_step_count_is_refused():
    """A serverless trainer's version is the step count its result reports; there is
    no read to fall back on."""
    with pytest.raises(RuntimeError, match="no step count"):
        stepped_version(SimpleNamespace(metrics={"grad_norm": 0.25}))


@pytest.mark.parametrize("when", ["between steps", "after the update"])
async def test_an_update_on_a_restarted_trainer_is_refused_and_not_kept(recipe, when):
    """A restarted trainer reports a version at or below the one the last update left;
    recording it would let the way out save untrained state under a trained name."""
    client = LoopsClient(policy_version=250)
    trainer = recipe.trainer(client)
    await recipe.update(trainer)
    if when == "between steps":
        client.policy_version = 0
    else:
        client.read_version = lambda version: 0
    with pytest.raises(RuntimeError, match="did not advance"):
        await recipe.update(trainer, step=1)
    assert await trainer.checkpoint_state_only(1) is None
    assert not client.state_checkpoints


async def test_a_restart_caught_at_the_next_update_keeps_no_earlier_step(recipe):
    """Fresh, step 0 leaves version 1; restarted, step 1 leaves version 1 again. Step 0's
    record matches that trainer, which holds only step 1's batch."""
    client = LoopsClient()
    trainer = recipe.trainer(client)
    await recipe.update(trainer)
    client.policy_version = 0
    with pytest.raises(RuntimeError, match=r"did not advance \(1 -> 1\)"):
        await recipe.update(trainer, step=1)
    assert await trainer.checkpoint_state_only(0) is None
    assert not client.state_checkpoints


async def test_an_update_whose_version_was_never_read_keeps_nothing_after_it(recipe):
    """Restarted, step 1 leaves version 1 again but its read fails: step 0's record
    matches a trainer holding only step 1's batch, and a later update must not make
    the trainer look accounted for again."""
    client = LoopsClient()
    trainer = recipe.trainer(client)
    await recipe.update(trainer)
    client.policy_version = 0
    fail_once(client, ConnectionError("policy_version read failed"))
    with pytest.raises(ConnectionError):
        await recipe.update(trainer, step=1)
    assert client.optim_steps == 2, "the failure is the trainer's read, not the update"
    assert await trainer.checkpoint_state_only(0) is None
    with pytest.raises(RuntimeError, match="never recorded"):
        await recipe.update(trainer, step=2)
    assert await trainer.checkpoint_state_only(2) is None
    assert not client.state_checkpoints


async def test_an_init_from_trainer_restarted_before_its_first_update_is_refused(
    recipe, loops_service
):
    """Back on base weights, its first update leaves version 1: an advance from 0,
    but not from the 250 the run started at."""
    client = loops_service.client
    client.restored = 250
    trainer = recipe.trainer(training={"init_from": INIT_FROM})
    await trainer.start()
    client.policy_version = 0
    with pytest.raises(RuntimeError, match=r"did not advance \(250 -> 1\)"):
        await recipe.update(trainer)
    assert await trainer.checkpoint_state_only(0) is None
    assert not client.state_checkpoints


@pytest.mark.parametrize(
    ("training", "resume_from"),
    [({}, None), ({"init_from": INIT_FROM}, None), ({}, RESUME_FROM)],
    ids=["fresh", "init_from", "resume"],
)
async def test_the_trainer_never_reads_its_policy_version_on_the_event_loop(
    recipe, loops_service, training, resume_from
):
    """A stalled read on the loop would stop every other task, cancellation included."""
    client = loops_service.client
    read_on = []

    def record(version):
        read_on.append(threading.get_ident())
        return version

    client.read_version = record
    trainer = recipe.trainer(resume_from=resume_from, training=training)
    await trainer.start()
    await recipe.update(trainer)
    assert await trainer.checkpoint_state_only(0)
    assert read_on
    assert threading.get_ident() not in read_on


def test_a_stalled_version_read_does_not_hold_up_the_way_out(recipe):
    """The read is a blocking SDK call that a timeout abandons rather than stops; exit
    must not wait for it (``asyncio.run`` waits on its executor's threads)."""
    client = LoopsClient()
    trainer = recipe.trainer(client)
    released = threading.Event()

    def stall(version):
        released.wait()
        return version

    async def go_down():
        await recipe.update(trainer)
        client.read_version = stall
        return await trainer.checkpoint_state_only(0, timeout=0.05)

    release = threading.Timer(5.0, released.set)
    release.start()
    started = time.monotonic()
    try:
        assert asyncio.run(go_down()) is None
        assert time.monotonic() - started < 2.0
    finally:
        release.cancel()
        released.set()
    assert not client.state_checkpoints


# --- the reduction Loops applies ---------------------------------------------------


@pytest.mark.parametrize(
    ("training", "metrics"),
    [
        ({}, {}),
        ({}, {"skip_token_normalization": 0.0}),
        ({"loss_aggregation": "token_mean"}, {"skip_token_normalization": 1.0}),
    ],
)
async def test_an_update_loops_does_not_confirm_the_reduction_of_is_refused(
    recipe, training, metrics
):
    client = LoopsClient()
    client.forward_backward_async = AsyncMock(
        return_value=Future(SimpleNamespace(loss=1.0, metrics=metrics))
    )
    trainer = recipe.trainer(client, training=training)
    with pytest.raises(RuntimeError, match="did not confirm"):
        await recipe.update(trainer)
    assert client.optim_steps == 0


# --- whether the Loops run has ended, as its control plane answers -----------------


class LoopsApi:
    """The run and trainer deployment records, as Loops answered them on 4 Oct 2026."""

    def __init__(self, run="ACTIVE", trainer="ACTIVE"):
        self.run, self.trainer = run, trainer
        self.deployment_id: str | None = "7wljz0q"
        self.requests = []
        self.hold_deployment: asyncio.Event | None = None
        self.stopped = False

    def answer(self, request):
        self.requests.append(request)
        if isinstance(self.run, int):
            return httpx.Response(self.run, json={"error": "nope"}, request=request)
        if request.url.path == "/v1/loops/runs/4w50xmq":
            body = {
                "run": {
                    "id": "4w50xmq",
                    "deployment_id": self.deployment_id,
                    "status": {"name": self.run},
                }
            }
        else:
            assert request.url.path == "/v1/loops/deployments/7wljz0q"
            body = {"deployment": {"id": "7wljz0q", "status": {"name": self.trainer}}}
        return httpx.Response(200, json=body, request=request)


@pytest.fixture
def loops_api(monkeypatch):
    """A started trainer's run, its HTTP answered by a ``LoopsApi``."""
    from baseten.loops import ServiceClient

    api = LoopsApi()

    async def handle_async(transport, request):
        if api.hold_deployment is not None and "/deployments/" in request.url.path:
            try:
                await api.hold_deployment.wait()
            except asyncio.CancelledError:
                api.stopped = True
                raise
        return api.answer(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async)
    monkeypatch.setattr(ServiceClient, "_create_session", lambda self: "session-1")
    t = rl.Trainer(make_config())
    t._service = ServiceClient(base_url="https://api.example", api_key="secret")
    t.client = SimpleNamespace(run_id="4w50xmq")
    return t, api


@pytest.mark.parametrize(
    ("run", "trainer", "reason"),
    [
        ("INACTIVE", "ACTIVE", "Loops reports run 4w50xmq INACTIVE"),
        *(
            ("ACTIVE", status, f"Loops reports run 4w50xmq's trainer deployment 7wljz0q {status}")
            for status in ("PREEMPTED", "FAILED", "STOPPED")
        ),
        *(("ACTIVE", status, None) for status in ("ACTIVE", "DEPLOYING", "SCALED_TO_ZERO")),
    ],
)
async def test_a_run_has_ended_when_loops_marks_it_inactive_or_its_trainer_is_gone(
    loops_api, run, trainer, reason
):
    t, api = loops_api
    api.run, api.trainer = run, trainer
    assert await t.run_ended() == reason
    assert {request.headers["Authorization"] for request in api.requests} == {"Api-Key secret"}


async def test_a_live_serverless_run_has_not_ended(loops_api):
    """A serverless run has no trainer deployment of its own to ask about (Loops
    answered ``deployment_id: null`` for run 4w5g2v3 on 21 Oct 2026)."""
    t, api = loops_api
    api.deployment_id = None
    assert await t.run_ended() is None
    assert [request.url.path for request in api.requests] == ["/v1/loops/runs/4w50xmq"]


async def test_a_status_loops_cannot_give_raises(loops_api):
    t, api = loops_api
    api.run = 503
    with pytest.raises(httpx.HTTPStatusError):
        await t.run_ended()


async def test_a_trainer_not_started_has_no_run_to_ask_about():
    assert await rl.Trainer(make_config()).run_ended() is None


async def test_cancelling_the_question_stops_its_request(loops_api):
    """A check cancelled at shutdown leaves no request running behind it."""
    t, api = loops_api
    api.hold_deployment = asyncio.Event()
    asking = asyncio.create_task(t.run_ended())
    for _ in range(50):
        await asyncio.sleep(0)
    asking.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asking
    assert api.stopped


class EndingService:
    """The ``ServiceClient`` a closing trainer ends its run through: ``deactivate_run_async``
    answers with ``outcome`` -- ``None``, an exception to raise, or ``"hang"``."""

    def __init__(self, outcome=None):
        self.outcome = outcome
        self.deactivated: list[str] = []

    async def deactivate_run_async(self, run_id):
        self.deactivated.append(run_id)
        if self.outcome == "hang":
            await asyncio.Event().wait()
        elif self.outcome is not None:
            raise self.outcome


def closing_trainer(service, **compute):
    t = rl.Trainer(make_config(**compute))
    t._service = service
    t.client = SimpleNamespace(run_id="4w50xmq", aclose=AsyncMock())
    return t


async def test_closing_a_trainer_ends_its_run_unless_keep_run(caplog):
    caplog.set_level("INFO")
    t = closing_trainer(service := EndingService(), keep_run=False)
    await t.close()
    t.client.aclose.assert_awaited_once()
    assert service.deactivated == ["4w50xmq"]
    assert "Deactivated Loops run 4w50xmq" in caplog.text
    assert "still up" not in caplog.text


@pytest.mark.parametrize("capacity", [Capacity.DEDICATED, Capacity.SERVERLESS])
async def test_by_default_the_run_stays_up_and_says_how_to_end_it(caplog, capacity):
    """Only a dedicated run is said to bill: a serverless one holds a slot, not pods."""
    t = closing_trainer(service := EndingService())
    t.capacity = capacity
    await t.close()
    assert service.deactivated == []
    assert "loops deactivate 4w50xmq" in caplog.text
    assert ("may be billing" in caplog.text) is (capacity is Capacity.DEDICATED)
    assert ("holds a serverless slot" in caplog.text) is (capacity is Capacity.SERVERLESS)


@pytest.mark.parametrize(
    "outcome",
    [
        httpx.HTTPStatusError(
            "503 Service Unavailable: try again",
            request=httpx.Request("POST", "https://api.example"),
            response=httpx.Response(503),
        ),
        "hang",
    ],
    ids=["refused", "unanswered"],
)
async def test_a_run_loops_does_not_end_in_time_is_reported_not_raised(
    monkeypatch, caplog, outcome
):
    monkeypatch.setattr("loops_cookbook.trainer.DEACTIVATE_TIMEOUT_SECONDS", 0.05)
    t = closing_trainer(EndingService(outcome), keep_run=False)
    await asyncio.wait_for(t.close(), timeout=5)
    assert "Could not deactivate Loops run 4w50xmq" in caplog.text
    assert "loops deactivate 4w50xmq" in caplog.text


async def test_a_trainer_that_never_started_has_no_run_to_end():
    t = rl.Trainer(make_config())
    t._service = service = EndingService()
    await t.close()
    assert service.deactivated == []
