"""Reduce a complete optimizer batch before transport splits it into requests.

Loops adds gradients up across forward_backward calls until optim_step. ``sum``
leaves it at that, Loops' default. ``token_mean`` asks Loops to divide the whole
loss by the targeted tokens with nonzero float32 advantages, penalty terms included.
``fixed_budget`` and ``task_mean`` keep the sum and multiply advantages by 1 / D:
for IS, PPO and CISPO a positive advantage multiplier scales the entire policy
gradient.

This module owns that choice. Group membership and counts refer to the
whole update, never to a datum, generation, or transport chunk. Prepared rows may
be split for delivery, but must not be mixed with rows from another preparation
in the same optimizer update.
"""

from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import compress, repeat
from math import isfinite
from operator import mul, ne

from baseten.loops.models import Datum, TensorData

from loops_cookbook.rl.config import TrainingSettings
from loops_cookbook.rl.grpo import IGNORE_INDEX, TrainingRow
from loops_cookbook.rl.slices import Sliced, whole

BUILTIN_NORMALIZATION_CONTRACT = "loops.rl.sum_or_active_token_mean.v2"
"""Cookbook name for required trainer behavior, not a negotiated SDK capability.

Check with tests/live/test_trainer_normalization.py before releases.
"""


# The per-token passes below are chained builtins (map, compress, count) so they run
# in C: a batch holds tens of millions of positions, and a generator expression over
# them costs seconds of the event loop's time.


def trainer_averages(training: TrainingSettings) -> bool:
    """Whether Loops divides this run's loss by its active tokens (``token_mean``)."""
    return training.loss_aggregation == "token_mean"


def active_positions(datum: Datum) -> int:
    """Targeted positions with nonzero float32 advantages: token_mean's count, as Loops
    counts it. A recipe's rows carry advantages only at actions."""
    weights = _float32(datum.loss_fn_inputs["advantages"].data)
    targets = datum.loss_fn_inputs["target_tokens"].data
    if len(targets) != len(weights):
        raise ValueError("target_tokens and advantages must be the same length")
    targeted = list(compress(weights, map(ne, targets, repeat(IGNORE_INDEX))))
    return len(targeted) - targeted.count(0)


def _float32(values: Iterable[float]) -> array:
    try:
        result = array("f", values)
    except OverflowError as error:
        raise ValueError("advantages must be finite in float32") from error
    if not all(map(isfinite, result)):
        raise ValueError("advantages must be finite in float32")
    return result


@dataclass(frozen=True)
class AggregationReport:
    mode: str
    active_tokens: int
    action_tokens: int
    groups: int
    denominator: int | None
    group_denominators: tuple[int, ...]
    advantage_scale_min: float
    advantage_scale_max: float
    backend_normalization: str = BUILTIN_NORMALIZATION_CONTRACT

    def gauges(self) -> dict[str, float]:
        # Only the divisions the cookbook makes itself. Under sum and token_mean Loops
        # reduces, and its own count is optim/num_loss_tokens; the scales are 1 / these.
        values: dict[str, float] = {}
        if self.mode == "fixed_budget":
            assert self.denominator is not None
            values["denominator"] = float(self.denominator)
        if self.group_denominators:
            values["group_denominator_min"] = float(min(self.group_denominators))
            values["group_denominator_max"] = float(max(self.group_denominators))
        return {f"aggregation/{key}": value for key, value in values.items()}


@dataclass(frozen=True)
class AggregatedBatch:
    rows: list[TrainingRow]
    report: AggregationReport | None


def aggregate_loss(
    groups: Sequence[Sequence[TrainingRow]], training: TrainingSettings
) -> AggregatedBatch:
    """Weight all rows for one optimizer update; inputs remain unchanged.

    task_mean includes zero-advantage action tokens in each group's token mean.
    fixed_budget uses the configured rollout count, even when members were lost.
    Custom losses receive the original rows and own their reduction.
    """
    return whole(aggregate_loss_in_slices(groups, training))


def aggregate_loss_in_slices(
    groups: Sequence[Sequence[TrainingRow]], training: TrainingSettings
) -> Sliced[AggregatedBatch]:
    """``aggregate_loss``, pausing after each row it reads or weights."""
    rows = [row for group in groups for row in group]
    if training.loss_fn == "custom":
        return AggregatedBatch(rows, None)
    counts = [sum(row.trained_positions for row in group) for group in groups]
    if not counts or any(count == 0 for count in counts):
        raise ValueError("loss aggregation requires action tokens in every task group")
    active = 0
    for row in rows:
        active += active_positions(row.datum)
        yield
    if active == 0:
        raise ValueError("loss aggregation requires nonzero float32 advantages")
    mode = training.loss_aggregation
    denominator, group_denominators = None, ()
    if mode in ("sum", "token_mean"):
        # Loops sums, or divides by ``active`` itself: the rows go as they are.
        denominator = active if mode == "token_mean" else None
        scales = [1.0] * len(groups)
    else:
        if mode == "fixed_budget":
            assert training.loss_token_budget is not None  # validated by TrainingSettings
            denominator = training.rollouts_per_step * training.loss_token_budget
            scales = [1 / denominator] * len(groups)
        else:
            group_denominators = tuple(len(groups) * count for count in counts)
            scales = [1 / value for value in group_denominators]
        rows = []
        for group, scale in zip(groups, scales, strict=True):
            for row in group:
                rows.append(row.with_datum(_scale_advantages(row.datum, scale)))
                yield
    return AggregatedBatch(
        rows,
        AggregationReport(
            mode,
            active,
            sum(counts),
            len(groups),
            denominator,
            group_denominators,
            min(scales),
            max(scales),
        ),
    )


def _scale_advantages(datum: Datum, scale: float) -> Datum:
    if not isfinite(scale) or scale <= 0:
        raise ValueError("loss aggregation requires a finite positive advantage scale")
    original = datum.loss_fn_inputs["advantages"]
    weights = _float32(original.data)
    scaled = _float32(map(mul, weights, repeat(scale)))
    # A positive finite scale keeps every zero zero, so the mask changes only where a
    # nonzero weight underflows to zero, and then there are more zeros.
    if scaled.count(0) != weights.count(0):
        raise ValueError("loss aggregation would change the float32 active-token mask")
    return datum.model_copy(
        update={
            "loss_fn_inputs": {
                **datum.loss_fn_inputs,
                "advantages": TensorData(
                    data=scaled.tolist(), dtype="float32", shape=original.shape
                ),
            }
        }
    )
