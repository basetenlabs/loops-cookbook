"""How far the trainer's logprobs are from the sampler's, on the tokens that train.

The trainer scores every datum it trains on. At an action position, its logprob and
the sampler's should agree up to numerics and routing. When the action is older than
the trainer's weights, policy drift adds to the gap. ``MismatchTotals`` splits the gap
two ways, so that each cause can be read on its own:

- By turn, which is the action's place in its chain. Router replay should keep late
  turns as close as early ones.
- By lag, which is how many updates the trainer's weights are past the action's
  policy. At lag 0, the gap comes from numerics and routing alone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from loops_cookbook.rl.grpo import TrainingRow
from loops_cookbook.rl.slices import Sliced, whole

_TURNS = ((1, "1"), (2, "2"), (4, "3-4"), (8, "5-8"), (16, "9-16"))


def _turn_bucket(turn: int) -> str:
    return next((name for last, name in _TURNS if turn <= last), "17+")


def _lag_bucket(lag: int) -> str:
    return str(lag) if lag < 2 else "2+"


class UnfitScores(RuntimeError):
    """The trainer's logprobs don't fit the rows sent.

    A datum is missing, or one has the wrong length.
    """


def action_scores(rows: Sequence[TrainingRow], scored: Sequence[Any]) -> list[list[Any]]:
    """Return each row's trainer logprobs on its actions, one slice per action.

    It takes Python lists and tensors alike. It keeps only the action positions, so a
    caller never copies a whole datum's logprobs to build the gauges.
    """
    if len(scored) != len(rows):
        raise UnfitScores(f"Loops returned logprobs for {len(scored)} of {len(rows)} datums")
    slices = []
    for row, values in zip(rows, scored, strict=True):
        length = len(row.datum.loss_fn_inputs["logprobs"].data)
        if len(values) != length:
            raise UnfitScores(
                f"Loops returned {len(values)} logprobs for a datum of {length} targets"
            )
        slices.append([values[start:stop] for start, stop in row.actions])
    return slices


@dataclass
class MismatchTotals:
    """Sums of |trainer - sampler| logprob over the tokens of one optimizer step."""

    sums: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    largest: float = 0.0
    nonfinite: int = 0

    def add(
        self,
        rows: Sequence[TrainingRow],
        scores: Sequence[Sequence[Sequence[float | None]]],
        version: int,
    ) -> None:
        """Add one call's rows.

        ``scores`` is ``action_scores`` of the trainer's logprobs, under weights at
        policy ``version``. A nonfinite or null trainer logprob is counted, not averaged.
        """
        whole(self.add_in_slices(rows, scores, version))

    def add_in_slices(
        self,
        rows: Sequence[TrainingRow],
        scores: Sequence[Sequence[Sequence[float | None]]],
        version: int,
    ) -> Sliced[None]:
        """Run ``add``, pausing after each action, for ``in_slices``."""
        for row, actions in zip(rows, scores, strict=True):
            sampler = row.datum.loss_fn_inputs["logprobs"].data
            for turn, ((start, _), policy, trainer) in enumerate(
                zip(row.actions, row.policies, actions, strict=True), start=1
            ):
                # The trainer may leave a position unscored, as null, which SFT scoring
                # also sees.
                gaps = [
                    abs(float(value) - float(sampler[start + i]))
                    for i, value in enumerate(trainer)
                    if value is not None
                ]
                finite = [gap for gap in gaps if math.isfinite(gap)]
                self.nonfinite += len(trainer) - len(finite)
                total = sum(finite)
                for key in (
                    "all",
                    f"turn_{_turn_bucket(turn)}",
                    f"lag_{_lag_bucket(version - policy)}",
                ):
                    self.sums[key] = self.sums.get(key, 0.0) + total
                    self.tokens[key] = self.tokens.get(key, 0) + len(finite)
                self.largest = max(self.largest, *finite, 0.0)
                yield

    def merge(self, other: MismatchTotals) -> None:
        """Fold ``other``'s sums into these."""
        for key, total in other.sums.items():
            self.sums[key] = self.sums.get(key, 0.0) + total
            self.tokens[key] = self.tokens.get(key, 0) + other.tokens[key]
        self.largest = max(self.largest, other.largest)
        self.nonfinite += other.nonfinite

    def gauges(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.nonfinite:
            out["mismatch/nonfinite_tokens"] = float(self.nonfinite)
        if not self.tokens.get("all"):
            return out
        out["mismatch/abs_logprob_diff_max"] = self.largest
        for key, total in self.sums.items():
            if self.tokens[key]:
                name = "mismatch/abs_logprob_diff" + ("" if key == "all" else f"/{key}")
                out[name] = total / self.tokens[key]
        return out
