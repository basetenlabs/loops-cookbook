from array import array
from dataclasses import asdict

import pytest
from baseten.loops.models import TensorData
from support.builders import make_config, make_rollout, make_sample

from loops_cookbook.rl.aggregation import active_positions, aggregate_loss, trainer_averages
from loops_cookbook.rl.grpo import build_rows


def row(name, length, advantage):
    sample = make_sample(
        name,
        (1, 2),
        (3,) * length,
        (-1.0,) * length,
        sequence_id=f"seq-{name}",
    )
    return build_rows(make_rollout("t", "g", name, (sample,)), advantage)[0]


@pytest.fixture
def groups():
    # The first group's middle reward produces zero advantage, but its actions
    # still belong to that task's token mean. Different group lengths expose
    # accidental per-row or per-chunk normalization.
    return [
        [row("a", 1, -0.5), row("b", 3, 0.5), row("zero", 2, 0.0)],
        [row("c", 2, -0.5), row("d", 6, 0.5)],
    ]


def settings(mode, **overrides):
    return make_config(
        **{
            "loss_aggregation": mode,
            "group_size": 3,
            "max_group_losses": 1,
            "groups_per_step": 2,
            **({"loss_token_budget": 10} if mode == "fixed_budget" else {}),
            **overrides,
        }
    ).training


@pytest.mark.parametrize(("mode", "denominator"), [("sum", None), ("token_mean", 12)])
def test_loops_reductions_send_the_rows_unchanged(groups, mode, denominator):
    """sum is Loops' default; token_mean is Loops dividing by the active tokens."""
    batch = aggregate_loss(groups, settings(mode))
    originals = [row for group in groups for row in group]
    assert all(a is b for a, b in zip(batch.rows, originals, strict=True))
    assert batch.report.active_tokens == 12
    assert batch.report.action_tokens == 14
    assert batch.report.denominator == denominator
    assert batch.report.advantage_scale_min == batch.report.advantage_scale_max == 1


@pytest.mark.parametrize(
    ("mode", "averages"),
    [("sum", False), ("token_mean", True), ("fixed_budget", False), ("task_mean", False)],
)
def test_only_token_mean_asks_loops_to_average(mode, averages):
    assert trainer_averages(settings(mode)) is averages


@pytest.mark.parametrize("mode", ["fixed_budget", "task_mean"])
def test_only_advantages_change_and_originals_are_untouched(groups, mode):
    originals = [row for group in groups for row in group]
    before = [row.datum.model_dump() for row in originals]
    batch = aggregate_loss(groups, settings(mode))
    for row, weighted_row in zip(originals, batch.rows, strict=True):
        original, weighted = row.datum, weighted_row.datum
        assert weighted_row.actions == row.actions
        assert weighted.model_input is original.model_input
        assert weighted.model_input_spans == original.model_input_spans is not None
        assert weighted.loss_fn_input_spans == original.loss_fn_input_spans
        assert weighted.loss_fn_inputs["target_tokens"] is original.loss_fn_inputs["target_tokens"]
        assert weighted.loss_fn_inputs["logprobs"] is original.loss_fn_inputs["logprobs"]
        assert active_positions(weighted) == active_positions(original)
    assert before == [row.datum.model_dump() for row in originals]
    if mode == "fixed_budget":
        # Six configured rollouts, even though this selected batch has only five.
        assert batch.report.denominator == 60
        assert batch.report.advantage_scale_min == pytest.approx(1 / 60)
    else:
        assert batch.report.denominator is None
        assert batch.report.group_denominators == (12, 16)


def test_multiturn_actions_use_one_episode_budget_and_tool_tokens_are_ignored():
    rollout = make_rollout(
        "t",
        "g",
        "multi",
        (
            make_sample("first", (1, 2), (3, 4), (-1.0, -1.0)),
            make_sample("second", (1, 2, 3, 4, 90, 91), (5,), (-1.0,)),
        ),
    )
    original = build_rows(rollout, 0.5)
    batch = aggregate_loss([original], settings("fixed_budget"))
    assert len(batch.rows) == 1
    assert batch.report.action_tokens == batch.report.active_tokens == 3
    assert batch.report.denominator == 60
    assert batch.rows[0].datum.loss_fn_inputs["target_tokens"].data == [2, 3, 4, 90, 91, 5]
    assert batch.rows[0].actions == ((1, 3), (5, 6))
    assert batch.rows[0].datum.loss_fn_inputs["advantages"].data == pytest.approx(
        [0, 0.5 / 60, 0.5 / 60, 0, 0, 0.5 / 60]
    )


def test_custom_loss_keeps_unscaled_inputs(groups):
    training = make_config(
        loss_fn="custom", custom_loss={"factory": "support.losses:make_policy_gradient"}
    ).training
    batch = aggregate_loss(groups, training)
    assert batch.report is None
    assert batch.rows == [row for group in groups for row in group]


def test_active_mask_is_evaluated_after_float32_conversion():
    underflow = row("underflow", 1, 1e-50)
    assert active_positions(underflow.datum) == 0
    with pytest.raises(ValueError, match="nonzero float32"):
        aggregate_loss([[underflow]], settings("token_mean"))


@pytest.mark.parametrize("value", [float("inf"), 1e40])
def test_float32_overflow_is_rejected_before_submission(value):
    overflow = row("overflow", 1, 1.0)
    overflow.datum.loss_fn_inputs["advantages"] = TensorData(
        data=[0, value], dtype="float32", shape=[2]
    )
    with pytest.raises(ValueError, match="finite in float32"):
        aggregate_loss([[overflow]], settings("token_mean"))


def test_scaling_cannot_silently_drop_active_tokens():
    with pytest.raises(ValueError, match="active-token mask"):
        aggregate_loss(
            [[row("tiny", 1, 1e-40)]], settings("fixed_budget", loss_token_budget=10**10)
        )


@pytest.mark.parametrize("groups", [[], [[]]])
def test_empty_task_groups_are_rejected(groups):
    with pytest.raises(ValueError, match="action tokens"):
        aggregate_loss(groups, settings("task_mean"))


@pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo", "cispo"])
@pytest.mark.parametrize("mode", ["sum", "token_mean", "fixed_budget", "task_mean"])
@pytest.mark.parametrize("chunks", [[5], [2, 3], [1, 1, 1, 1, 1]])
def test_policy_gradients_match_direct_reduction_across_transport_chunks(
    groups, loss_fn, mode, chunks
):
    """Compare autograd with the requested objective, independently of 1/D scaling.

    Simulate the backend's summed chunk gradients, divided by the active tokens only
    when the run asks Loops to average. The logits have ratios below, inside and above the clipping range,
    for both signs of advantage and for masked/zero-advantage positions.
    """
    torch = pytest.importorskip("torch")
    batch = aggregate_loss(groups, settings(mode, loss_fn=loss_fn))
    originals = [row.datum for group in groups for row in group]

    def token_losses(datum, offset):
        old = torch.tensor(datum.loss_fn_inputs["logprobs"].data, dtype=torch.float32)
        weights = torch.tensor(datum.loss_fn_inputs["advantages"].data, dtype=torch.float32)
        mask = torch.tensor(datum.loss_fn_inputs["target_tokens"].data) != -100
        ratios = torch.tensor([0.4, 1.0, 1.8] * len(old))[: len(old)]
        new = old + ratios.log() + offset
        ratio = (new - old).exp()
        if loss_fn == "importance_sampling":
            losses = -ratio * weights
        elif loss_fn == "ppo":
            losses = -torch.minimum(ratio * weights, ratio.clamp(0.8, 1.2) * weights)
        else:
            losses = -ratio.clamp(0.8, 1.2).detach() * weights * new
        return (losses * mask).sum()

    reference = torch.zeros(len(originals), requires_grad=True)
    sums = [token_losses(datum, reference[i]) for i, datum in enumerate(originals)]
    if mode == "sum":
        objective = sum(sums)
    elif mode == "token_mean":
        objective = sum(sums) / 12
    elif mode == "fixed_budget":
        objective = sum(sums) / 60
    else:
        objective = (sum(sums[:3]) / 6 + sum(sums[3:]) / 8) / 2
    objective.backward()

    delivered = torch.zeros(len(originals), requires_grad=True)
    start = 0
    for size in chunks:
        stop = start + size
        sum(token_losses(batch.rows[i].datum, delivered[i]) for i in range(start, stop)).backward()
        start = stop
    assert start == len(originals)
    divisor = batch.report.active_tokens if trainer_averages(settings(mode)) else 1
    assert (delivered.grad / divisor).numpy() == pytest.approx(
        reference.grad.numpy(), rel=2e-6, abs=1e-8
    )
    assert asdict(batch.report)["active_tokens"] == 12


def test_weighted_float32_storage_matches_backend_precision(groups):
    batch = aggregate_loss(groups, settings("task_mean"))
    for weighted in batch.rows:
        values = weighted.datum.loss_fn_inputs["advantages"].data
        assert list(array("f", values)) == values
