"""Accumulate/commit and checkpointing."""

from __future__ import annotations

import asyncio
import json
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from baseten.loops.models import LoadWeightsResponse, TensorData
from baseten.loops.training_client import TrainingClient
from support.builders import make_config, make_rollout, make_sample
from support.fakes import FakeClient, Future, scored_like_sampler

from loops_cookbook import trainer as trainer_module
from loops_cookbook.rl.grpo import build_rows
from loops_cookbook.rl.loss import scored_datum
from loops_cookbook.rl.trainer import Trainer
from loops_cookbook.trainer import Capacity


def datums(n, *, recorded=False):
    """``n`` training rows; ``recorded`` ones name their sampler records."""
    out = []
    for i in range(n):
        traj = make_rollout(
            task_id="t",
            group_id="g",
            rollout_id=f"r{i}",
            samples=(
                make_sample(
                    f"s{i}",
                    (1, 2, 3),
                    (4, 5),
                    (-0.1, -0.2),
                    0,
                    sequence_id=f"seq-{i}" if recorded else None,
                ),
            ),
        )
        out.extend(build_rows(traj, 0.5))
    return out


def trainer(fake_client, checkpoint_every=None, **overrides):
    """``checkpoint_every`` sets both cadences; tests that split them pass their own."""
    if checkpoint_every is not None:
        overrides.setdefault("trainer_checkpoint_every", checkpoint_every)
        overrides.setdefault("sampler_checkpoint_every", checkpoint_every)
    resume_from = overrides.pop("resume_from", None)
    t = Trainer(make_config(**overrides), resume_from)
    t.client = fake_client
    return t


async def commit(t, step=0, rollouts=4, final=False):
    """One update as the runner makes it: commit, then checkpoint."""
    report = await t.commit(
        step,
        groups=1,
        rollouts=rollouts,
        rewards=[1.0, 0.0] * (rollouts // 2),
        advantages_=[0.5, -0.5] * (rollouts // 2),
        sampled_versions=[0],
    )
    return await t.checkpoint(report, final=final)


async def test_the_new_policy_version_is_recorded_before_the_sampler_can_serve_it(fake_client):
    # The paired sampler can hot-swap before the publication call returns, so a
    # response from the new weights must not look newer than the trainer's version.
    t = trainer(fake_client, checkpoint_every=0)
    during_publication = []

    async def publish(name="default", *, record_stability_info=False):
        during_publication.append(t.policy_version)
        return fake_client.sampling_client

    fake_client.save_weights_and_get_sampling_client_async = publish
    await t.accumulate(datums(2))
    report = await commit(t)
    assert during_publication == [1]
    assert report.policy_version_after == t.policy_version == 1


async def test_stage_timings_accumulate_and_reset_between_updates(monkeypatch, fake_client):
    ticks = iter(range(30))
    monkeypatch.setattr("loops_cookbook.rl.trainer.monotonic", lambda: next(ticks))
    t = trainer(fake_client, checkpoint_every=0)
    await t.accumulate(datums(2))
    await t.accumulate(datums(2))
    report = await commit(t)
    assert report.timings == {
        "forward_backward": 2,
        "optimizer": 1,
        "publication": 1,
        "checkpoint": 1,
    }
    assert report.gauges()["time/forward_backward_seconds"] == 2
    await t.accumulate(datums(4))
    second = await commit(t, step=1)
    assert second.timings["forward_backward"] == 1
    assert report.timings["forward_backward"] == 2


async def test_delivery_shape_does_not_change_the_batch():
    """Chunked delivery preserves batch accounting and makes one optimizer call."""
    totals = {}
    for chunks in ([12], [4, 4, 4], [1] * 12):
        client = FakeClient()
        t = trainer(client, checkpoint_every=0)
        rows, offset = datums(12), 0
        for size in chunks:
            await t.accumulate(rows[offset : offset + size])
            offset += size
        report = await commit(t, rollouts=12)
        totals[len(chunks)] = (report.datums, report.tokens, report.trained_positions)
        assert client.fb_calls == chunks
        assert client.optim_steps == 1
    assert len(set(totals.values())) == 1, totals


@pytest.mark.parametrize(
    ("training", "sent"),
    [
        ({}, ("cispo", {})),
        ({"loss_fn": "importance_sampling"}, ("importance_sampling", {})),
        ({"loss_fn": "cispo"}, ("cispo", {})),
        (
            {"loss_fn": "cispo", "loss_fn_config": {"clip_high_threshold": 5.0}},
            ("cispo", {"clip_high_threshold": 5.0}),
        ),
        (
            {"loss_fn": "ppo", "loss_fn_config": {"clip_low_threshold": 0.9}},
            ("ppo", {"clip_low_threshold": 0.9}),
        ),
        ({"loss_fn": "dro", "loss_fn_config": {"beta": 0.1}}, ("dro", {"beta": 0.1})),
        ({"loss_fn": "dppo", "loss_fn_config": {"kl_tau": 0.01}}, ("dppo", {"kl_tau": 0.01})),
    ],
)
@pytest.mark.parametrize(
    ("aggregation", "skip"),
    [({}, True), ({"loss_aggregation": "token_mean"}, False)],
    ids=["sum", "token_mean"],
)
async def test_a_built_in_loss_reaches_loops_with_the_settings_given_and_its_reduction(
    fake_client, training, sent, aggregation, skip
):
    """A key left unset keeps Loops' default; the reduction is always stated."""
    t = trainer(fake_client, checkpoint_every=0, **training, **aggregation)
    await t.accumulate(datums(2))
    await commit(t, rollouts=2)
    loss_fn, settings = sent
    assert fake_client.loss_fns == [(loss_fn, {**settings, "skip_token_normalization": skip})]


class CustomLossTransport(FakeClient):
    """The pinned SDK's own custom-loss path over a fake transport."""

    forward_backward_custom_async = TrainingClient.forward_backward_custom_async
    _get_custom_loss_forward_data = TrainingClient._get_custom_loss_forward_data

    def __init__(self):
        super().__init__()
        self.sent = []

    async def forward_async(self, data, loss_fn, loss_fn_config=None):
        outputs = [
            {"logprobs": TensorData(data=[-1.0] * n, dtype="float32", shape=[n])}
            for n in (d.model_input.length for d in data)
        ]
        return Future(
            SimpleNamespace(loss_fn_outputs=outputs, metrics={"skip_token_normalization": 1.0})
        )

    async def forward_backward_async(self, data, loss_fn, loss_fn_config=None):
        self.sent.extend(data)
        return await super().forward_backward_async(data, loss_fn, loss_fn_config)


async def test_a_custom_loss_gradient_reaches_loops_over_every_scored_position():
    pytest.importorskip("torch")
    client = CustomLossTransport()
    t = trainer(
        client,
        checkpoint_every=0,
        loss_fn="custom",
        custom_loss={"factory": "support.losses:make_policy_gradient"},
    )
    first, second = datums(2), datums(1)
    await t.accumulate(first)
    await t.accumulate(second)
    report = await commit(t, rollouts=3)
    assert [loss_fn for loss_fn, _ in client.loss_fns] == ["cross_entropy"] * 2
    for sent, row in zip(client.sent, first + second, strict=True):
        scored = scored_datum(row).loss_fn_inputs
        assert sent.loss_fn_inputs["target_tokens"] == scored["target_tokens"]
        # L = -Σ adv·lp over the actions, so -dL/dlp = adv: the weights Loops trains on.
        assert sent.loss_fn_inputs["weights"].tolist() == pytest.approx(
            row.datum.loss_fn_inputs["advantages"].tolist()
        )
    # train/loss is the loss's own value per call, datum-weighted across calls. Each
    # datum adds 2 positions x 0.5 advantage x 1 nat: calls of 2.0 and 1.0.
    assert report.loss == pytest.approx((2.0 * 2 + 1.0 * 1) / 3)
    gauges = report.gauges()
    assert gauges["loss/datums"] == pytest.approx((2 * 2 + 1 * 1) / 3)
    assert "train/mismatch_kl" not in gauges


async def test_the_sampler_is_stamped_with_the_version_it_was_published_at(fake_client):
    """A serverless sampling client is pinned to its publication and reports no
    version; the lease takes this one instead."""
    t = trainer(fake_client, checkpoint_every=0)
    for step in range(2):
        await t.accumulate(datums(2))
        await commit(t, step=step, rollouts=2)
        assert t.sampler_version == t.policy_version == step + 1


async def test_step_checkpoints_are_named_by_optimizer_step(fake_client):
    t = trainer(fake_client, checkpoint_every=2, checkpoint_ttl_seconds=99)
    for step in range(4):
        await t.accumulate(datums(2))
        report = await commit(t, step=step, rollouts=2)
        assert report.checkpoint == (f"step-{step}" if step % 2 else None)
    assert fake_client.sampler_checkpoints == [("step-1", 99), ("step-3", 99)]
    assert fake_client.state_checkpoints == [("step-1", 99), ("step-3", 99)]
    # The per-step sync still happened on every step.
    assert fake_client.published == ["default"] * 4


async def test_a_retention_of_zero_keeps_checkpoints_without_a_ttl(fake_client):
    t = trainer(fake_client, checkpoint_every=1, checkpoint_ttl_seconds=0)
    await t.accumulate(datums(2))
    await commit(t, rollouts=2)
    assert fake_client.sampler_checkpoints == [("step-0", None)]
    assert fake_client.state_checkpoints == [("step-0", None)]


async def test_save_every_zero_keeps_nothing_but_still_syncs(fake_client):
    t = trainer(fake_client, checkpoint_every=0)
    await t.accumulate(datums(2))
    assert (await commit(t, rollouts=2)).checkpoint is None
    assert not fake_client.sampler_checkpoints and not fake_client.state_checkpoints
    assert fake_client.published == ["default"]


async def test_the_final_update_is_kept_whatever_the_cadence_says(fake_client):
    """A run that ends between save_every boundaries still leaves something to resume."""
    t = trainer(fake_client, checkpoint_every=5, checkpoint_ttl_seconds=7)
    await t.accumulate(datums(2))
    skipped = await commit(t, step=1, rollouts=2)
    assert skipped.checkpoint is None and not fake_client.state_checkpoints
    await t.accumulate(datums(2))
    kept = await commit(t, step=2, rollouts=2, final=True)
    assert kept.checkpoint == "step-2"
    assert fake_client.state_checkpoints == [("step-2", 7)]


async def test_the_final_update_is_kept_even_with_checkpointing_disabled(fake_client):
    """`save_every = 0` asks not to checkpoint along the way, not to end with nothing."""
    t = trainer(fake_client, checkpoint_every=0)
    await t.accumulate(datums(2))
    assert (await commit(t, rollouts=2, final=True)).checkpoint == "step-0"
    assert [name for name, _ in fake_client.state_checkpoints] == ["step-0"]


async def test_an_eval_keeps_the_weights_the_cadence_would_skip_at_the_path_published(
    fake_client,
):
    t = trainer(fake_client, checkpoint_every=0, checkpoint_ttl_seconds=7)
    await t.accumulate(datums(2))
    report = await t.commit(
        3, groups=1, rollouts=2, rewards=[1.0, 0.0], advantages_=[0.5, -0.5], sampled_versions=[0]
    )
    kept = await t.checkpoint(report, keep_sampler_weights=True)
    assert fake_client.sampler_checkpoints == [("step-3", 7)]
    assert kept.sampler_weights == "bt://loops:run-1/sampler_weights/step-3"
    assert kept.checkpoint == "step-3"
    assert not fake_client.state_checkpoints  # only the weights were asked for


@pytest.mark.parametrize("path", ["", None, 7], ids=["empty", "none", "not a string"])
@pytest.mark.parametrize("for_an_eval", [True, False])
async def test_weights_without_a_servable_path_fail_only_an_eval_that_needs_them(
    fake_client, for_an_eval, path
):
    async def no_path(name="default", ttl_seconds=None):
        return Future(SimpleNamespace(version=1, path=path))

    fake_client.save_weights_for_sampler_async = no_path
    t = trainer(fake_client, checkpoint_every=1)
    await t.accumulate(datums(2))
    report = await t.commit(
        0, groups=1, rollouts=2, rewards=[1.0, 0.0], advantages_=[0.5, -0.5], sampled_versions=[0]
    )
    if for_an_eval:
        # The eval cannot serve weights it cannot name; the update stands.
        kept = await t.checkpoint(report, keep_sampler_weights=True)
        assert kept.checkpoint == "step-0" and kept.sampler_weights is None
    else:
        kept = await t.checkpoint(report)  # the cadence's own save, as before evals
        assert kept.checkpoint == "step-0" and kept.sampler_weights is None


class SamplersApi:
    """Loops' sampler endpoints as a run's ServiceClient reaches them; it records each
    request, and answers deactivation with ``deactivate_statuses`` in turn, the last
    repeated."""

    def __init__(self, monkeypatch, deactivate_statuses=(200,)):
        from baseten.loops import ServiceClient

        self.created: list[dict] = []
        self.deactivated: list[httpx.Request] = []
        self.deactivate_statuses = list(deactivate_statuses)
        api = self

        def handle(transport, request):
            return api.handle(request)

        async def handle_async(transport, request):
            return await api.handle_async(request)

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async)
        self.client = ServiceClient(base_url="https://api.example", api_key="secret")

    def handle(self, request):
        if request.url.path.endswith("/samplers"):
            self.created.append(json.loads(request.content))
            sampler = {"id": "5wo2nv3", "base_url": "https://sampler.example"}
            return httpx.Response(200, json={"sampler": sampler}, request=request)
        return httpx.Response(200, json={"session": {"id": "session-1"}}, request=request)

    async def handle_async(self, request):
        if request.url.path.endswith(("/samplers", "/sessions")):
            # baseten-loops 0.33 creates a sampler over its async client.
            return self.handle(request)
        if request.url.path.endswith("/deactivate"):
            self.deactivated.append(request)
            statuses = self.deactivate_statuses
            status = statuses.pop(0) if len(statuses) > 1 else statuses[0]
            return httpx.Response(status, json={}, request=request)
        # Loops resolves a checkpoint path before serving it.
        checkpoint = {
            "run_id": "r",
            "checkpoint_id": "step-3",
            "checkpoint_type": "sampler_weights",
            "created_at": "2026-10-29T00:00:00Z",
            "base_model": "m",
            "lora_adapter_config": {"r": 16},
        }
        return httpx.Response(200, json={"checkpoints": [checkpoint]}, request=request)


@pytest.mark.parametrize(
    ("model_path", "serves"),
    [
        (None, {"base_model": "m"}),
        (
            "bt://loops:r/sampler_weights/step-3",
            {"model_path": "bt://loops:r/sampler_weights/step-3"},
        ),
    ],
    ids=["base model", "checkpoint"],
)
async def test_an_eval_sampler_is_a_standalone_sampler_in_the_runs_session(
    monkeypatch, model_path, serves
):
    """Loops is asked for a standalone sampler (never the run's paired one) on the
    run's own session, serving what the eval names at the model's sequence length."""
    api = SamplersApi(monkeypatch)
    t = Trainer(make_config(**{"model.base_model": "m", "model.max_seq_len": 4096}))
    t._service = api.client
    sampler = await t.eval_sampler(model_path, ready_timeout=30.0)
    [body] = api.created
    assert {key: body[key] for key in serves} == serves
    assert body["session_id"] == "session-1" and body["max_seq_length"] == 4096
    assert sampler.checkpoint_path == model_path


@pytest.mark.parametrize("scope", ["dedicated", "serverless"])
async def test_an_eval_sampler_runs_on_the_runs_scope(monkeypatch, scope):
    """An eval sampler is placed where the run is, not where Loops would choose: the
    eval schedule follows the run's capacity."""
    api = SamplersApi(monkeypatch)
    t = Trainer(make_config(**{"compute.scope": scope}))
    t._service = api.client
    await t.eval_sampler(None, ready_timeout=30.0)
    [body] = api.created
    assert body["scope"] == scope


@pytest.mark.parametrize(
    ("statuses", "attempts", "deactivated"),
    [
        ((200,), 1, True),
        ((503, 500, 200), 3, True),
        ((500,), 5, False),
        ((404,), 1, False),
    ],
    ids=["at once", "after transient failures", "failing past its retries", "refused"],
)
async def test_the_eval_sampler_started_is_the_one_deactivated_retried_and_a_failure_only_logged(
    monkeypatch, caplog, statuses, attempts, deactivated
):
    """A transient failure is retried, since a sampler left up bills; one that outlasts
    the retries, or a refusal, is logged with the sampler's ID and never raised."""
    monkeypatch.setattr(trainer_module, "DEACTIVATE_RETRY_DELAYS", (0.0,) * 4)
    api = SamplersApi(monkeypatch, deactivate_statuses=statuses)
    t = Trainer(make_config(**{"model.base_model": "m"}))
    t._service = api.client
    caplog.set_level("INFO", logger="loops_cookbook.trainer")
    await t.deactivate_sampler(await t.eval_sampler(None, ready_timeout=30.0))
    assert len(api.deactivated) == attempts
    for request in api.deactivated:
        assert str(request.url) == "https://api.example/v1/loops/samplers/5wo2nv3/deactivate"
        assert request.method == "POST" and request.headers["Authorization"] == "Api-Key secret"
    assert ("Deactivated eval sampler 5wo2nv3." in caplog.text) is deactivated
    assert ("Eval sampler 5wo2nv3 was NOT deactivated" in caplog.text) is not deactivated


@pytest.mark.parametrize(("retention", "ttl"), [(0, None), (99, 99)], ids=["keep", "expire"])
async def test_weights_kept_only_as_trainer_state_are_loaded_and_saved_for_a_sampler(
    monkeypatch, fake_client, retention, ttl
):
    t = trainer(fake_client, checkpoint_ttl_seconds=retention)
    loaded = []

    async def load_state_async(path):
        loaded.append(path)
        return Future(LoadWeightsResponse(path=path, step=30))

    async def provision(**_):
        return fake_client

    fake_client.load_state_async = load_state_async
    monkeypatch.setattr(t, "_provision", provision)
    path, version = await t.servable_weights("bt://loops:r/weights/step-29", "eval-step-29")
    assert loaded == ["bt://loops:r/weights/step-29"]
    assert (path, version) == ("bt://loops:run-1/sampler_weights/eval-step-29", 30)
    # Kept as long as the run keeps its checkpoints: a retention of 0 sends no TTL.
    assert fake_client.sampler_checkpoints == [("eval-step-29", ttl)]


@pytest.mark.parametrize(
    ("statuses", "deactivated"), [((503, 200), True), ((404,), False)], ids=["retried", "refused"]
)
async def test_the_run_a_trainer_made_for_an_eval_is_deactivated_and_a_failure_only_logged(
    monkeypatch, caplog, statuses, deactivated
):
    monkeypatch.setattr(trainer_module, "DEACTIVATE_RETRY_DELAYS", (0.0,) * 4)
    api = SamplersApi(monkeypatch, deactivate_statuses=statuses)
    t = Trainer(make_config(**{"model.base_model": "m"}))
    t._service, t.client = api.client, SimpleNamespace(run_id="qvv1gxq")
    caplog.set_level("INFO", logger="loops_cookbook.trainer")
    await t.deactivate_run()
    assert {str(r.url) for r in api.deactivated} == {
        "https://api.example/v1/loops/runs/qvv1gxq/deactivate"
    }
    assert ("Deactivated loops run qvv1gxq." in caplog.text) is deactivated
    assert ("Loops run qvv1gxq was NOT deactivated" in caplog.text) is not deactivated


async def test_a_local_eval_sampler_has_nothing_on_loops_to_deactivate(caplog):
    from baseten.loops import ServiceClient

    t = Trainer(make_config(**{"model.base_model": "m"}))
    t._service = ServiceClient.local(
        trainer_url="http://localhost:8000", sampler_url="http://localhost:8001"
    )
    await t.deactivate_sampler(await t.eval_sampler(None, ready_timeout=30.0))
    assert "NOT deactivated" not in caplog.text


@pytest.mark.parametrize(
    ("capacity", "serverless"),
    [(Capacity.SERVERLESS, True), (Capacity.DEDICATED, False), (None, False)],
    ids=["serverless", "dedicated", "not started"],
)
def test_a_trainer_on_serverless_capacity_says_so(fake_client, capacity, serverless):
    t = trainer(fake_client)
    t.capacity = capacity
    assert t.serverless is serverless


async def test_commit_with_nothing_accumulated_is_refused(fake_client):
    t = trainer(fake_client)
    with pytest.raises(ValueError, match="nothing accumulated"):
        await commit(t, rollouts=2)


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), -float("inf")])
async def test_nonfinite_loss_blocks_commit_even_after_valid_accumulation(loss):
    client = FakeClient()
    t = trainer(client)
    await t.accumulate(datums(2))
    rows = datums(2)
    client.forward_backward_async = AsyncMock(
        return_value=Future(
            SimpleNamespace(
                loss=loss,
                metrics={"skip_token_normalization": 1.0},
                loss_fn_outputs=scored_like_sampler([row.datum for row in rows]),
            )
        )
    )
    with pytest.raises(RuntimeError, match="Nonfinite"):
        await t.accumulate(rows)
    with pytest.raises(RuntimeError, match="Nonfinite"):
        await commit(t, rollouts=2)
    with pytest.raises(RuntimeError, match="Nonfinite"):
        await t.accumulate(datums(2))
    assert client.forward_backward_async.await_count == 1
    assert client.optim_steps == 0 and client.published == []


async def test_over_length_datum_never_reaches_the_trainer():
    client = FakeClient()
    t = trainer(client, max_seq_len=3)
    with pytest.raises(ValueError, match="max_seq_len"):
        await t.accumulate(datums(1))
    assert client.fb_calls == []


async def test_checkpoint_records_the_path_resume_needs(fake_client):
    """The name alone cannot be fed back to load_state."""
    t = trainer(fake_client, checkpoint_every=1)
    await t.accumulate(datums(2))
    report = await commit(t, step=0, rollouts=2)
    assert report.checkpoint == "step-0"
    assert report.checkpoint_state_path == "bt://loops:run-1/state/step-0"


@pytest.mark.parametrize("replicas", [None, 4])
async def test_sampler_replicas_reach_provisioning(loops_service, replicas):
    """Unset, the sampler gets the Loops default."""
    await Trainer(make_config(sampler_replicas=replicas)).start()
    assert [call["sampler_num_replicas"] for call in loops_service.provisioned] == [replicas]


@pytest.mark.parametrize("resume_from", [None, "bt://loops:source/state/step-6"])
async def test_start_publishes_the_version_it_restored_before_any_rollout(
    loops_service, resume_from
):
    t = Trainer(make_config(), resume_from)
    await t.start()
    assert t.policy_version == t.sampler_version == (7 if resume_from else 0)
    assert loops_service.client.published == ["default"]


@pytest.mark.parametrize("replay", [True, False])
async def test_every_publication_records_samples_exactly_when_replay_is_on(loops_service, replay):
    """Replay names the sampler's record of each generation, so every sampler a
    replaying run publishes records; one that does not replay records nothing."""
    t = Trainer(make_config(router_replay=replay), None)
    await t.start()
    await t.accumulate(datums(2))
    await commit(t)
    assert loops_service.client.recorded == [replay, replay]


async def test_start_settles_sampler_readiness_before_returning(loops_service):
    """Startup settles the cold start, so no agent deadline pays for it."""
    sampler = await Trainer(make_config()).start()
    assert sampler is loops_service.client.sampling_client
    assert sampler.ready_calls == 1


async def test_weight_publishes_do_not_re_settle_readiness(loops_service):
    """Only startup waits: a poll every step would sit in the weight sync's way."""
    client = loops_service.client
    t = trainer(client, checkpoint_every=0)
    await t.start()
    await t.accumulate(datums(4))
    await commit(t, step=0)

    # The step published new weights, and settled readiness no further.
    assert len(client.published) > 1
    assert client.sampling_client.ready_calls == 1


async def test_optimizer_settings_and_metrics_reach_the_step(fake_client):
    t = trainer(
        fake_client,
        checkpoint_every=0,
        grad_clip_norm=0.7,
        adam_epsilon=1e-8,
        adam_beta1=0.8,
        adam_beta2=0.99,
        weight_decay=0.01,
    )
    await t.accumulate(datums(2))
    report = await commit(t, rollouts=2)
    params = fake_client.adam_params
    assert (params.grad_clip_norm, params.eps, params.beta1, params.beta2, params.weight_decay) == (
        0.7,
        1e-8,
        0.8,
        0.99,
        0.01,
    )
    assert report.gauges()["optim/grad_norm"] == 0.25
    assert report.gauges()["optim/step_seconds"] == 2.0


@pytest.mark.parametrize(
    ("replay", "routed"),
    [(True, [True, False, False]), (False, [True, True])],
    ids=["replay with a route missing", "replay off with routes"],
)
async def test_a_batch_whose_provenance_is_not_normalized_is_refused_before_any_call(
    replay, routed
):
    client = FakeClient()
    t = trainer(client, checkpoint_every=0, router_replay=replay)
    recorded, plain = datums(len(routed), recorded=True), datums(len(routed))
    rows = [
        with_spans if keeps else without
        for with_spans, without, keeps in zip(recorded, plain, routed, strict=True)
    ]
    with pytest.raises(ValueError, match="normalize_batch_provenance"):
        await t.accumulate(rows)
    assert client.fb_calls == []


async def test_replay_mode_cannot_change_mid_optimizer_window():
    client = FakeClient()
    t = trainer(client, checkpoint_every=0, router_replay=True)
    rows = datums(2, recorded=True)
    await t.accumulate(rows)
    assert client.fb_spans == [[row.datum.model_input_spans for row in rows]]
    assert all(spans is not None for spans in client.fb_spans[0])
    with pytest.raises(ValueError, match="entire optimizer batch"):
        await t.accumulate(datums(1))
    assert client.fb_calls == [2]
    await commit(t, rollouts=2)
    await t.accumulate(datums(1))
    assert client.fb_spans[-1] == [None]


class PerCallMetrics(FakeClient):
    """A call over a long datum reports one set of trainer metrics, any other call another."""

    async def forward_backward_async(self, data, loss_fn, loss_fn_config=None):
        long = data[0].model_input.length > 10
        return Future(
            SimpleNamespace(
                loss=9.0 if long else 3.0,
                metrics={
                    "mismatch_kl": 0.6 if long else 0.03,
                    "mismatch_abs_logprob_diff": 0.9 if long else 0.3,
                    "mismatch_ess": 0.5 if long else 0.8,
                    "not_a_number": "n/a",
                    "nonfinite": float("nan"),
                    "skip_token_normalization": 1.0,
                },
                loss_fn_outputs=scored_like_sampler(data),
            )
        )


async def test_trainer_metrics_are_kept_per_call_and_averaged_per_update_by_datums():
    long_rollout = make_rollout(
        "t",
        "g",
        "long",
        (make_sample("long", (1, 2, 3), tuple(range(4, 14)), (-0.1,) * 10, 0),),
    )
    rows = build_rows(long_rollout, 0.5) + datums(2)
    t = trainer(PerCallMetrics(), checkpoint_every=0)
    await t.accumulate(rows[:1])
    await t.accumulate(rows[1:])
    report = await commit(t, rollouts=3)
    # Datum-weighted across calls: one long datum, two short; the worst call is kept.
    assert report.loss == pytest.approx(5.0)
    assert report.mismatch_kl == pytest.approx(0.22)
    assert report.mismatch_kl_max_call == 0.6
    gauges = report.gauges()
    assert gauges["train/mismatch_kl_max_call"] == 0.6
    assert gauges["train/mismatch_abs_logprob_diff"] == pytest.approx((0.9 + 2 * 0.3) / 3)
    assert gauges["train/mismatch_ess"] == pytest.approx((0.5 + 2 * 0.8) / 3)
    assert "train/not_a_number" not in gauges and "train/nonfinite" not in gauges
    assert "train/skip_token_normalization" not in gauges  # an echo, checked, not a metric
    calls = report.forward_backward_reports
    assert [call.routed for call in calls] == [False, False]
    assert [call.trained_positions for call in calls] == [10, 4]
    assert [call.loss for call in calls] == [9.0, 3.0]
    assert [call.metrics["mismatch_ess"] for call in calls] == [0.5, 0.8]

    # A new optimizer window must not retain the previous calls' diagnostics.
    await t.accumulate(datums(1))
    next_report = await commit(t, step=1, rollouts=2)
    assert next_report.loss == 3.0
    assert next_report.mismatch_kl == pytest.approx(0.03)
    assert next_report.mismatch_kl_max_call == 0.03
    assert next_report.gauges()["train/mismatch_ess"] == pytest.approx(0.8)
    assert len(next_report.forward_backward_reports) == 1


async def test_eval_samplers_starting_never_hold_the_threads_training_runs_on():
    """A sampler start holds a thread until the sampler is ready; however many are
    starting, a call training makes on the default pool still runs."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    release = threading.Event()

    class Service:
        def create_sampling_client(self, **kwargs):
            release.wait(10)
            return "sampler"

        async def create_sampling_client_async(self, **kwargs):
            return await asyncio.to_thread(partial(self.create_sampling_client, **kwargs))

    loop = asyncio.get_running_loop()
    default = ThreadPoolExecutor(max_workers=2)  # a small controller's pool
    loop.set_default_executor(default)
    t = Trainer(make_config(**{"model.base_model": "m"}))
    t._service = Service()
    starts = [asyncio.create_task(t.eval_sampler(None, ready_timeout=30.0)) for _ in range(3)]
    await asyncio.sleep(0.1)
    try:
        assert await asyncio.wait_for(asyncio.to_thread(lambda: "trained"), 2) == "trained"
    finally:
        release.set()
    assert await asyncio.gather(*starts) == ["sampler"] * 3
    await t.close()
    default.shutdown(wait=False)


async def test_an_eval_sampler_start_waiting_for_a_thread_spends_its_own_timeout():
    """The start timeout bounds the whole start, a wait for a free thread included."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    release = threading.Event()
    asked: list[float] = []

    class Service:
        def create_sampling_client(self, *, ready_timeout, **kwargs):
            asked.append(ready_timeout)
            release.wait(10)
            return "sampler"

    t = Trainer(make_config(**{"model.base_model": "m"}))
    t._service = Service()
    t._eval_starts = ThreadPoolExecutor(max_workers=1)
    first = asyncio.create_task(t.eval_sampler(None, ready_timeout=30.0))
    while not asked:  # the first holds the only thread
        await asyncio.sleep(0.01)
    try:
        # It fails at its deadline, while the thread it waits for is still busy.
        with pytest.raises(TimeoutError, match="0.2s"):
            await asyncio.wait_for(t.eval_sampler(None, ready_timeout=0.2), 2)
    finally:
        release.set()
    assert await first == "sampler"
    assert asked == [pytest.approx(30.0, abs=0.1)]  # the second never reached Loops
    await t.close()


class OffsetScores(FakeClient):
    """Loops scoring every position 0.3 above the sampler."""

    async def forward_backward_async(self, data, loss_fn, loss_fn_config=None):
        result = await (
            await super().forward_backward_async(data, loss_fn, loss_fn_config)
        ).result_async()
        for output in result.loss_fn_outputs:
            output["logprobs"] = TensorData(
                data=[value + 0.3 for value in output["logprobs"].data],
                dtype="float32",
                shape=output["logprobs"].shape,
            )
        return Future(result)


async def test_each_update_reports_the_gap_on_its_trained_tokens_by_policy_lag():
    t = trainer(OffsetScores(), checkpoint_every=0)
    await t.accumulate(datums(2))
    first = (await commit(t)).gauges()
    # The next update's rows were sampled at version 0; the trainer is now at 1.
    await t.accumulate(datums(2))
    second = (await commit(t, step=1)).gauges()
    for gauges, lag in ((first, "lag_0"), (second, "lag_1")):
        assert {k: v for k, v in gauges.items() if k.startswith("mismatch/")} == pytest.approx(
            {
                "mismatch/abs_logprob_diff": 0.3,
                "mismatch/abs_logprob_diff_max": 0.3,
                "mismatch/abs_logprob_diff/turn_1": 0.3,
                f"mismatch/abs_logprob_diff/{lag}": 0.3,
            }
        )


async def test_a_custom_loss_update_reports_the_gap_on_the_logprobs_it_was_given():
    pytest.importorskip("torch")
    t = trainer(
        CustomLossTransport(),
        checkpoint_every=0,
        loss_fn="custom",
        custom_loss={"factory": "support.losses:make_policy_gradient"},
    )
    await t.accumulate(datums(2))
    gauges = (await commit(t, rollouts=2)).gauges()
    # Loops scores -1 everywhere; the sampler gave -0.1 and -0.2.
    assert gauges["mismatch/abs_logprob_diff"] == pytest.approx(0.85)
    assert gauges["mismatch/abs_logprob_diff_max"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    "outputs",
    [
        lambda data: scored_like_sampler(data)[:-1],
        lambda data: [
            {"logprobs": TensorData(data=[-1.0], dtype="float32", shape=[1])} for _ in data
        ],
        lambda data: [{"logprobs": TensorData(data=[-1.0], dtype="float32", shape=[1])}, {}],
        lambda data: [*scored_like_sampler(data)[:-1], {}],
    ],
    ids=["a datum missing", "a wrong length", "partly scored, wrong length", "partly scored"],
)
async def test_a_trainer_answer_that_does_not_fit_its_rows_blocks_the_step(outputs):
    """Loops holds the call's gradients by then, so nothing may step on them."""
    client = FakeClient()
    t = trainer(client)
    rows = datums(2)
    client.forward_backward_async = AsyncMock(
        return_value=Future(
            SimpleNamespace(
                loss=1.0,
                metrics={"skip_token_normalization": 1.0},
                loss_fn_outputs=outputs([row.datum for row in rows]),
            )
        )
    )
    with pytest.raises(RuntimeError, match="for 1 of 2 datums|logprobs for a datum of"):
        await t.accumulate(rows)
    with pytest.raises(RuntimeError, match="did not fit its rows invalidated"):
        await commit(t, rollouts=2)
    assert client.optim_steps == 0 and client.published == []


@pytest.mark.parametrize("outputs", [[], [{}, {}]], ids=["none", "without logprobs"])
async def test_an_answer_without_logprobs_trains_and_reports_no_gap(outputs, caplog):
    client = FakeClient()
    t = trainer(client, checkpoint_every=0)
    client.forward_backward_async = AsyncMock(
        return_value=Future(
            SimpleNamespace(
                loss=1.0, metrics={"skip_token_normalization": 1.0}, loss_fn_outputs=outputs
            )
        )
    )
    await t.accumulate(datums(2))
    gauges = (await commit(t, rollouts=2)).gauges()
    assert client.optim_steps == 1
    assert not [key for key in gauges if key.startswith("mismatch/")]
    assert "mismatch/* is not reported" in caplog.text
