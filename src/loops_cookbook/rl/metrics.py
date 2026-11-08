"""Per-update summaries of an RL batch: stage durations, trajectories and agent metadata."""

import logging
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from statistics import median

from loops_cookbook.console import some_names
from loops_cookbook.evidence import JsonValue, Rollout

logger = logging.getLogger(__name__)


@dataclass
class DurationSummary:
    """Cumulative wall-clock duration statistics, kept in bounded memory."""

    count: int = 0
    total: float = 0.0
    maximum: float = 0.0

    def observe(self, seconds: float) -> None:
        self.count += 1
        self.total += seconds
        self.maximum = max(self.maximum, seconds)

    def gauges(self, name: str) -> dict[str, float]:
        return {
            f"time/{name}/count": float(self.count),
            f"time/{name}/total_seconds": self.total,
            f"time/{name}/mean_seconds": self.total / self.count if self.count else 0.0,
            f"time/{name}/max_seconds": self.maximum,
        }


def trajectory_metrics(rollouts: Sequence[Rollout]) -> dict[str, float]:
    """Return metrics for the selected batch. Token entropy is a sampled estimate."""
    if not rollouts:
        return {}
    samples = [sample for rollout in rollouts for sample in rollout.samples]
    trainable = [sample for sample in samples if sample.trainable]
    token_count = sum(len(sample.completion_ids) for sample in trainable)
    lengths = [sum(len(s.completion_ids) for s in r.samples) for r in rollouts]
    # The final context each rollout reached: its longest prompt plus completion.
    sequences = [
        max((len(s.prompt_ids) + len(s.completion_ids) for s in r.samples), default=0)
        for r in rollouts
    ]
    out = {
        "trajectory/generations": float(len(samples)),
        "trajectory/generations_per_rollout": len(samples) / len(rollouts),
        "trajectory/sampled_tokens": float(sum(lengths)),
        "trajectory/sampled_tokens_per_rollout": sum(lengths) / len(rollouts),
        "trajectory/sampled_tokens_min": float(min(lengths)),
        "trajectory/sampled_tokens_median": float(median(lengths)),
        "trajectory/sampled_tokens_max": float(max(lengths)),
        "trajectory/sequence_tokens_min": float(min(sequences)),
        "trajectory/sequence_tokens_median": float(median(sequences)),
        "trajectory/sequence_tokens_mean": sum(sequences) / len(rollouts),
        "trajectory/sequence_tokens_max": float(max(sequences)),
        "trajectory/trainable_sampled_tokens": float(token_count),
    }
    if token_count:
        out["sampling/entropy_estimate"] = (
            -sum(sum(sample.logprobs) for sample in trainable) / token_count
        )
    # These count why model generations stopped, not why the agent ended.
    stops = Counter(sample.stop_reason or "unknown" for sample in samples)
    out.update({f"sampling/stop/{reason}": float(count) for reason, count in stops.items()})
    # These generations were cut by the trajectory budget, not by the max_tokens they
    # asked for. Their limit was lowered to the room the prompt left, and they stopped
    # on reaching it.
    out["sampling/cut_at_budget"] = float(
        sum(
            sample.clamped_from is not None and sample.stop_reason == "length" for sample in samples
        )
    )
    return out


def metadata_means(metadata: Sequence[Mapping[str, JsonValue]]) -> dict[str, float]:
    """Return each numeric metadata key's mean over the entries that report it.

    It averages the way ``rollout_metrics`` does. Booleans count as 0 and 1, and labels
    are left out.
    """
    numbers: dict[str, list[float]] = defaultdict(list)
    for entry in metadata:
        for key, value in entry.items():
            if isinstance(value, (bool, int, float)) and isfinite(value):
                numbers[key].append(float(value))
    return {key: sum(values) / len(values) for key, values in numbers.items()}


def rollout_metrics(rollouts: Sequence[Rollout], *, limit: int = 32) -> dict[str, float]:
    """Summarize what the agents reported about their own episodes.

    Numbers are averaged over the batch, and labels are counted. That gives an agent
    its own view of a step next to the reward, such as how many turns it took or how
    often it ran out of them. The recipe never needs to know what any of it means. Booleans
    average into a rate, which is what a flag is worth across a batch.

    The keys are the agent's. A label whose value differs per rollout, such as an ID
    or a timestamp, would make one metric name per rollout and bury everything else,
    so such labels are dropped. Past ``limit`` names, the metrics kept are the ones
    the most rollouts reported, so a stray identifier can't displace the numbers the
    run is watching. Each rule logs its own warning, and ``rollout/dropped_keys``
    counts everything either rule left out.
    """
    if not rollouts:
        return {}
    numbers: dict[str, list[float]] = defaultdict(list)
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    for rollout in rollouts:
        for key, value in rollout.metadata.items():
            if isinstance(value, (bool, int, float)) and isfinite(value):
                numbers[key].append(float(value))
            elif isinstance(value, str):
                labels[key][value] += 1
    # Label counts are absolute, and the batch size varies, because a group may be
    # selected with as few as `min_group_size` members. To compare across steps, a
    # count such as "completed: 40" needs a denominator. `train/rollouts` already carries
    # it on the same axis, from the same members, so a second name would be a synonym.
    out = {f"rollout/{key}": sum(values) / len(values) for key, values in numbers.items()}
    coverage = {f"rollout/{key}": len(values) for key, values in numbers.items()}
    identifiers = []
    # A label that never repeats a value identifies each rollout instead of describing
    # it. Its counts are then all 1, so summarizing it reveals nothing, and dropping it
    # loses nothing. This test needs no threshold, because a batch of any size answers
    # for itself. A key with one rare outcome among many common ones still repeats, so
    # it stays.
    for key, counts in labels.items():
        if max(counts.values()) == 1:
            identifiers.append(key)
            continue
        for value, count in counts.items():
            out[f"rollout/{key}/{value}"] = float(count)
            coverage[f"rollout/{key}/{value}"] = count
    over_limit: list[str] = []
    if len(out) > limit:
        kept = sorted(out, key=lambda name: (-coverage[name], name))[:limit]
        over_limit = sorted(set(out) - set(kept))
        out = {name: out[name] for name in kept}
    # Each rule gets its own warning because the remedy differs. A limit drop described
    # as an identifier would send the reader to fix keys that were never wrong.
    if identifiers:
        logger.warning(
            "Dropped %d agent metadata key(s) from step metrics as identifiers: %s. A "
            "label whose value differs per rollout makes a metric name, not a metric.",
            len(identifiers),
            some_names(sorted(identifiers)),
        )
    if over_limit:
        logger.warning(
            "Dropped %d rollout metric(s) past [output] rollout_metrics_limit = %d, "
            "keeping the ones the most rollouts reported, then by name: %s. Raise "
            "[output] rollout_metrics_limit to keep them.",
            len(over_limit),
            limit,
            some_names(sorted(over_limit)),
        )
    if identifiers or over_limit:
        out["rollout/dropped_keys"] = float(len(identifiers) + len(over_limit))
    return out
