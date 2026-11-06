"""Billable numerical contract for built-in Loops RL normalization.

Run with LOOPS_LIVE_LOOPS=1 and the custom-loss extra; LOOPS_CONTRACT_TRAINER_REPLICAS
runs it on more than one trainer copy. One trainer, no sampler or sandbox: use
zero-learning-rate optimizer steps to compare built-in gradient norms, for every
loss_aggregation mode and for DRO's and DPPO's penalties under token_mean, against
explicitly reduced custom objectives at fixed weights. It also records how far split
forward_backward calls move logprobs, and holds two calls of one batch to a single
call of its doubled objective, which checks accumulation apart from that drift. The
run is deactivated even when an assertion fails; normalization-receipt.json retains
the evidence.
"""

import asyncio
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import re
import uuid
from pathlib import Path

import httpx
import pytest
from baseten.loops import ServiceClient
from baseten.loops.models import AdamParams, Datum, ModelInput, TensorData

from loops_cookbook.rl.aggregation import (
    BUILTIN_NORMALIZATION_CONTRACT,
    aggregate_loss,
    trainer_averages,
)
from loops_cookbook.rl.config import TrainingSettings
from loops_cookbook.rl.grpo import TrainingRow
from loops_cookbook.trainer import check_token_normalization, token_normalization

pytestmark = [pytest.mark.live_loops, pytest.mark.asyncio]


def contract_groups():
    groups = []
    for lengths, advantages in (((1, 3, 2), (-0.5, 0.5, 0)), ((2, 6), (-0.5, 0.5))):
        rows = []
        for length, advantage in zip(lengths, advantages, strict=True):
            # Mask a prompt and a gap between action spans. Give ignored positions
            # nonzero advantages so the test requires BOTH parts of the RL mask.
            actions = {1, *range(4, 4 + length - 1)}
            size = max(actions) + 1
            tokens = list(range(10 + len(groups) * 20, 10 + len(groups) * 20 + size + 1))
            targets = [tokens[i + 1] if i in actions else -100 for i in range(size)]
            rows.append(
                TrainingRow(
                    Datum(
                        model_input=ModelInput.from_ints(tokens[:-1]),
                        loss_fn_inputs={
                            "target_tokens": TensorData(data=targets, dtype="int64", shape=[size]),
                            "advantages": TensorData(
                                data=[advantage if i in actions else 7.0 for i in range(size)],
                                dtype="float32",
                                shape=[size],
                            ),
                            "logprobs": TensorData(
                                data=[0.0] * size, dtype="float32", shape=[size]
                            ),
                        },
                    ),
                    ((1, 2), (4, 4 + length - 1)),
                    (0, 0),
                )
            )
        groups.append(rows)
    return groups


async def logprobs(client, datums):
    bare = [
        datum.model_copy(
            update={
                "loss_fn_inputs": {
                    "target_tokens": datum.loss_fn_inputs["target_tokens"],
                    "weights": TensorData(
                        data=[1.0] * datum.model_input.length,
                        dtype="float32",
                        shape=[datum.model_input.length],
                    ),
                }
            }
        )
        for datum in datums
    ]
    result = await (await client.forward_async(bare, "cross_entropy")).result_async()
    return [
        output["logprobs"].to_torch().nan_to_num().tolist() for output in result.loss_fn_outputs
    ]


def explicit_cispo(datums, probabilities, denominators):
    """An oracle with explicit reduction and unscaled advantages."""
    import torch

    loss = sum(
        -(
            (lp - datum.loss_fn_inputs["logprobs"].to_torch()).exp().clamp(0.8, 1.2).detach()
            * lp
            * datum.loss_fn_inputs["advantages"].to_torch()
            * (datum.loss_fn_inputs["target_tokens"].to_torch() != -100)
        ).sum()
        / denominator
        for datum, lp, denominator in zip(datums, probabilities, denominators, strict=True)
    )
    assert isinstance(loss, torch.Tensor)
    return loss, {}


DPPO_KL_TAU = 0.05
"""Above Loops' 1e-3 default, so the KL term carries weight against the policy term."""


def explicit_dro(datums, probabilities, denominators, beta=0.05):
    """DRO's per-token loss on active positions, its penalty divided like its advantage."""
    import torch

    loss = sum(
        (
            (
                -lp * datum.loss_fn_inputs["advantages"].to_torch()
                + 0.5 * beta * (lp - datum.loss_fn_inputs["logprobs"].to_torch()) ** 2
            )
            * active_mask(datum)
        ).sum()
        / denominator
        for datum, lp, denominator in zip(datums, probabilities, denominators, strict=True)
    )
    assert isinstance(loss, torch.Tensor)
    return loss, {}


def explicit_dppo(datums, probabilities, denominators, kl_tau=None, bound=0.2):
    """Loops' DPPO-Binary-TV per-token loss on active positions, its KL term included."""
    import torch

    kl_tau = DPPO_KL_TAU if kl_tau is None else kl_tau

    def per_token(datum, lp):
        old = datum.loss_fn_inputs["logprobs"].to_torch()
        advantages = datum.loss_fn_inputs["advantages"].to_torch()
        active = active_mask(datum)
        moved = lp.exp() - old.exp()
        keep = active & ~torch.where(advantages > 0, moved > bound, moved < -bound)
        ratio = (lp - old).clamp(-20, 20).exp()
        return -(keep * advantages * ratio) + kl_tau * active * (lp - old) ** 2

    loss = sum(
        per_token(datum, lp).sum() / denominator
        for datum, lp, denominator in zip(datums, probabilities, denominators, strict=True)
    )
    assert isinstance(loss, torch.Tensor)
    return loss, {}


def explicit_linear(datums, probabilities, denominators):
    """No importance ratio: forward-value noise cannot reach this gradient's weights."""
    import torch

    loss = sum(
        -(lp * datum.loss_fn_inputs["advantages"].to_torch() * active_mask(datum)).sum()
        / denominator
        for datum, lp, denominator in zip(datums, probabilities, denominators, strict=True)
    )
    assert isinstance(loss, torch.Tensor)
    return loss, {}


def active_mask(datum):
    return (datum.loss_fn_inputs["target_tokens"].to_torch() != -100) & (
        datum.loss_fn_inputs["advantages"].to_torch() != 0
    )


async def deactivate(run_id):
    root = os.getenv("LOOPS_BASE_URL", "https://api.baseten.co").rstrip("/")
    headers = {"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"}
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{root}/v1/loops/runs/{run_id}/deactivate", headers=headers)
        response.raise_for_status()
        return response.status_code


@pytest.fixture
async def live_trainer(tmp_path, caplog):
    assert os.environ.get("BASETEN_API_KEY"), "Live normalization contract requires BASETEN_API_KEY"
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.fail("Live normalization contract requires the custom-loss extra (PyTorch)")
    caplog.set_level(logging.INFO, logger="baseten.loops")
    receipt = {
        "contract": BUILTIN_NORMALIZATION_CONTRACT,
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sdk_version": importlib.metadata.version("baseten-loops"),
        "model": os.getenv("LOOPS_CONTRACT_MODEL", "Qwen/Qwen3.5-4B"),
        # The contract must hold however many trainer copies split the update.
        "trainer_replicas": int(os.getenv("LOOPS_CONTRACT_TRAINER_REPLICAS", "1")),
        # Replicas size a deployment of the run's own; serverless would drop them.
        "scope": "dedicated",
        "cases": [],
    }
    client = None
    path = tmp_path / "normalization-receipt.json"
    print(f"Normalization receipt: {path}", flush=True)
    try:
        client = await asyncio.to_thread(
            ServiceClient().create_lora_training_client,
            base_model=receipt["model"],
            rank=8,
            replicas=receipt["trainer_replicas"],
            scope=receipt["scope"],
            seed=17,
            max_seq_len=128,
            with_sampler=False,
            name=f"normalization-contract-{uuid.uuid4().hex[:8]}",
            ready_timeout=900,
            timeout=300,
        )
        receipt["run_id"] = client.run_id
        print(f"Normalization trainer ready: {client.run_id}", flush=True)
        yield client, receipt
    finally:
        # The SDK logs ownership before its readiness wait, which also lets us
        # clean up a provisioned run when readiness fails before returning a client.
        run_id = receipt.get("run_id")
        if run_id is None:
            found = re.findall(r"Provisioned Loops run (\w+)", caplog.text)
            if found:
                run_id = found[-1]
                receipt["run_id"] = run_id
        try:
            if run_id is not None:
                receipt["deactivated"] = await deactivate(run_id)
                print(f"Normalization trainer deactivated: {run_id}", flush=True)
        finally:
            path.write_text(json.dumps(receipt, indent=2, allow_nan=False))
            if client is not None:
                await client.aclose()


async def test_optimizer_normalization_contract(live_trainer):
    async with asyncio.timeout(1800):
        await verify_normalization(live_trainer)


def split(rows, chunks):
    out, start = [], 0
    for size in chunks:
        out.append(rows[start : start + size])
        start += size
    assert start == len(rows)
    return out


async def verify_normalization(live_trainer):
    client, receipt = live_trainer
    groups = contract_groups()
    rows = [row.datum for group in groups for row in group]
    initial = await logprobs(client, rows)
    print("Normalization baseline logprobs captured", flush=True)
    # Split forward calls score each row with different batch neighbours. Loops'
    # logprobs depend on them (0.06 nats seen); recorded, and bounded only grossly.
    split_logprobs = [
        values for chunk in split(rows, (1, 3, 1)) for values in await logprobs(client, chunk)
    ]
    receipt["split_logprob_max_abs_diff"] = max(
        abs(a - b)
        for whole_row, split_row in zip(initial, split_logprobs, strict=True)
        for a, b in zip(whole_row, split_row, strict=True)
    )
    print(f"Split logprob drift: {receipt['split_logprob_max_abs_diff']:.4g}", flush=True)
    assert receipt["split_logprob_max_abs_diff"] < 0.25
    # Grad norms are measured before Adam. A zero learning rate keeps the same
    # parameters for every comparison; optimizer moments cannot affect the norm.
    # Each optim_step clears accumulated gradients for the next case.
    adam = AdamParams(learning_rate=0, grad_clip_norm=0, weight_decay=0)
    # Independently specified from this fixture's 14 action tokens: two belong
    # to a zero-advantage episode, so Loops must count 12.
    expected_active_tokens = 12

    def with_old_logprobs(offset):
        for datum, values in zip(rows, initial, strict=True):
            datum.loss_fn_inputs["logprobs"] = TensorData(
                data=[value - offset for value in values], dtype="float32", shape=[len(values)]
            )

    async def step(
        label,
        *,
        chunks=(5,),
        oracle=None,
        denominators=None,
        batch=None,
        training=None,
        repeat=1,
    ):
        """One optimizer window: an explicit custom oracle, sent ``repeat`` times, or a
        built-in loss."""
        print(f"Normalization case {label}: starting", flush=True)
        if oracle is not None:
            for chunk, chunk_denominators in repeat * list(
                zip(split(rows, chunks), split(denominators, chunks), strict=True)
            ):
                bare = [
                    datum.model_copy(
                        update={
                            "loss_fn_inputs": {
                                "target_tokens": datum.loss_fn_inputs["target_tokens"],
                            }
                        }
                    )
                    for datum in chunk
                ]
                future = await client.forward_backward_custom_async(
                    bare,
                    lambda _, lp, chunk=chunk, chunk_denominators=chunk_denominators: oracle(
                        chunk, lp, chunk_denominators
                    ),
                )
                await future.result_async()
        else:
            averaged = trainer_averages(training)
            settings = {**training.loss_fn_config, **token_normalization(averaged)}
            for chunk in split([row.datum for row in batch.rows], chunks):
                future = await client.forward_backward_async(chunk, training.loss_fn, settings)
                check_token_normalization((await future.result_async()).metrics, averaged)
        result = await (await client.optim_step_async(adam)).result_async()
        metrics = result.metrics or {}
        case = {"case": label, "metrics": metrics}
        receipt["cases"].append(case)
        assert metrics.get("num_loss_tokens") == repeat * expected_active_tokens, (
            "Loops normalization contract changed: expected targeted, nonzero-advantage "
            f"tokens={repeat * expected_active_tokens}, got {metrics.get('num_loss_tokens')}; "
            f"case={label}"
        )
        norm = metrics.get("grad_norm")
        assert norm is not None and math.isfinite(norm) and norm > 1e-8, metrics
        after = await logprobs(client, rows)
        case["logprobs"] = after
        for actual, expected in zip(after, initial, strict=True):
            assert actual == pytest.approx(expected, abs=0.003), (
                f"Parameters changed despite zero learning rate: {label}"
            )
        print(f"Normalization case {label}: grad_norm={norm:.8g}", flush=True)
        return norm

    async def compare(label, oracle, denominators, training):
        expected_norm = await step(
            f"{label}/explicit-custom", oracle=oracle, denominators=denominators
        )
        batch = aggregate_loss(groups, training)
        # Where Loops averages, it divides the accumulated gradient at optim_step, and
        # dividing before backward rounds differently (0.1% seen). The same-chunks
        # control then runs the oracle undivided and divides its norm afterwards.
        late = expected_active_tokens if trainer_averages(training) else 1
        control = [denominator / late for denominator in denominators]
        for chunks in ((5,), (1, 3, 1)):
            case = f"{label}/{chunks}"
            # The oracle reduces in its loss; the built-in loss through aggregate_loss
            # and the reduction it asks Loops for. Same chunks, same forward values.
            chunked_norm = (
                await step(
                    f"{case}/explicit-custom", chunks=chunks, oracle=oracle, denominators=control
                )
                / late
            )
            norm = await step(f"{case}/builtin", chunks=chunks, batch=batch, training=training)
            comparison = {
                "case": case,
                "whole_batch_relative_error": abs(norm / expected_norm - 1),
                "same_chunks_relative_error": abs(norm / chunked_norm - 1),
            }
            receipt["comparisons"].append(comparison)
            print(f"Normalization comparison: {comparison}", flush=True)
            assert norm == pytest.approx(expected_norm, rel=0.02), case
            assert norm == pytest.approx(chunked_norm, rel=1e-4), (
                f"Normalization mismatch: {comparison}"
            )

    def settings(loss_fn, mode, **loss_fn_config):
        return TrainingSettings(
            max_steps=0,
            learning_rate=1e-5,
            loss_fn=loss_fn,
            loss_aggregation=mode,
            loss_fn_config=loss_fn_config,
            group_size=3,
            max_group_losses=1,
            groups_per_step=2,
            **({"loss_token_budget": 10} if mode == "fixed_budget" else {}),
        )

    receipt["comparisons"] = []
    for regime, offset in (("clipped", 10), ("on_policy", 0)):
        # Saturating the clip removes differences between the custom scoring
        # forward and native backward forward from the importance weight.
        with_old_logprobs(offset)
        for mode, denominators in (
            ("sum", [1] * 5),
            ("token_mean", [12] * 5),
            ("fixed_budget", [60] * 5),
            ("task_mean", [12] * 3 + [16] * 2),
        ):
            await compare(
                f"{regime}/cispo/{mode}", explicit_cispo, denominators, settings("cispo", mode)
            )
    # Four nats off policy, DRO's penalty gradient (beta * 4 = 0.2) is a large share of
    # its advantage term's (0.5) without cancelling it at any token, so dividing only
    # the advantages would show and every active token keeps a gradient.
    with_old_logprobs(4)
    await compare(
        "off_policy/dro/token_mean", explicit_dro, [12] * 5, settings("dro", "token_mean")
    )
    # One nat off policy: the ratio stays near e, the probability bound drops some
    # positive-advantage tokens, and the KL gradient is a sizeable share of the total.
    with_old_logprobs(1)
    await compare(
        "off_policy/dppo/token_mean",
        explicit_dppo,
        [12] * 5,
        settings("dppo", "token_mean", kl_tau=DPPO_KL_TAU),
    )

    # The same batch sent twice sees the same neighbours, so the same activations: two
    # calls must accumulate to one call's doubled gradient, isolating the summing from
    # the forward drift that split calls carry.
    with_old_logprobs(0)
    doubled = await step("accumulation/once-doubled", oracle=explicit_linear, denominators=[6] * 5)
    twice = await step(
        "accumulation/twice", oracle=explicit_linear, denominators=[12] * 5, repeat=2
    )
    receipt["accumulation_relative_error"] = abs(twice / doubled - 1)
    assert twice == pytest.approx(doubled, rel=1e-3), receipt["accumulation_relative_error"]
    receipt["passed"] = True
