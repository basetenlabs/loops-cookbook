"""Group-centered advantages and token-exact Loops training rows.

The advantage is ``reward - mean(group rewards)``, not divided by the group's
standard deviation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from statistics import fmean

from baseten.loops import PromptProvenanceSpan, SampledProvenanceSpan
from baseten.loops.models import Datum, EncodedTextChunk, TensorData

from loops_cookbook.evidence import Rollout, Sample, TrajectoryError
from loops_cookbook.rl.slices import Sliced, whole

IGNORE_INDEX = -100
"""The target at an image position, which has no token to predict."""


# Weak-referenceable: a run checks that a step's rows are freed once they are sent.
@dataclass(frozen=True, slots=True, weakref_slot=True)
class TrainingRow:
    """A Loops datum and the positions it trains.

    The datum's targets are the next token at every text position, prompt and
    environment tokens included: token provenance checks each target against the
    sampler's record, so none can be masked out by its id. Which positions are the
    agent's own -- the ones a recipe trains -- is ``actions``, and nothing else.
    """

    datum: Datum
    actions: tuple[tuple[int, int], ...]
    """Half-open target ranges, ascending and disjoint: one per trainable generation."""
    policies: tuple[int, ...]
    """The policy version that sampled each of ``actions``, in the same order."""

    def __post_init__(self) -> None:
        if len(self.policies) != len(self.actions):
            raise ValueError("a row needs one policy version per action")

    @property
    def trained_positions(self) -> int:
        return sum(stop - start for start, stop in self.actions)

    @property
    def routed(self) -> bool:
        """Whether the datum names the recorded samples its tokens came from."""
        return self.datum.model_input_spans is not None

    def with_datum(self, datum: Datum) -> TrainingRow:
        return TrainingRow(datum, self.actions, self.policies)


def advantages(rewards: list[float]) -> list[float]:
    if not all(isfinite(reward) for reward in rewards):
        raise TrajectoryError("rewards must be finite")
    mean = fmean(rewards) if rewards else 0.0
    return [reward - mean for reward in rewards]


def build_rows(
    trajectory: Rollout,
    advantage: float,
    chains: Sequence[Sequence[Sample]] | None = None,
) -> list[TrainingRow]:
    """One row per chain with a trainable generation; ``chains`` are the trajectory's
    own ``chains``, when the caller has already computed them."""
    return whole(build_rows_in_slices(trajectory, advantage, chains))


def build_rows_in_slices(
    trajectory: Rollout,
    advantage: float,
    chains: Sequence[Sequence[Sample]] | None = None,
) -> Sliced[list[TrainingRow]]:
    """``build_rows``, pausing after each chain: a forked rollout's rows are each a
    pass over a full-length sequence, so many long chains are not one unit."""
    if not isfinite(advantage):
        raise TrajectoryError("advantage must be finite")
    parts = trajectory.chains if chains is None else chains
    if not parts:
        raise TrajectoryError(f"rollout {trajectory.identity.rollout_id} has no samples")
    result = []
    for chain in parts:
        if not any(sample.trainable for sample in chain):
            continue
        result.append(_row(chain, advantage))
        yield
    if not result:
        raise TrajectoryError("rollout contains no trainable generations")
    return result


def _row(chain: Sequence[Sample], advantage: float) -> TrainingRow:
    last = chain[-1]
    # The final sampled token is a target, never an input. Images live only in the
    # prompt, so trimming a completion token never cuts a chunk.
    length = len(last.prompt_ids) + len(last.completion_ids) - 1
    # Built from the compact evidence for this row alone: its own lists and chunks.
    prompt = last.prompt_input
    model_input = prompt
    if len(last.completion_ids) > 1:
        model_input = model_input.append(EncodedTextChunk(tokens=list(last.completion_ids[:-1])))
    # The next token at every position; an image position has none.
    sequence = [
        token
        for chunk in prompt.chunks
        for token in (
            chunk.tokens if isinstance(chunk, EncodedTextChunk) else [IGNORE_INDEX] * chunk.length
        )
    ]
    sequence.extend(last.completion_ids)
    targets = sequence[1:]
    weights = [0.0] * length
    logprobs = [0.0] * length
    actions = []
    policies = []
    # Each generation's positions, written a slice at a time rather than a token at a
    # time: a chain's generations are prefixes of its last, so they fit.
    for sample in chain:
        if not sample.trainable:
            continue
        start = len(sample.prompt_ids) - 1
        stop = start + len(sample.completion_ids)
        if stop > length:
            raise TrajectoryError("a chain's generations must be prefixes of its last")
        # An action's target is what it sampled, whatever chunk the last prompt holds there.
        targets[start:stop] = sample.completion_ids
        weights[start:stop] = [advantage] * (stop - start)
        logprobs[start:stop] = sample.logprobs
        actions.append((start, stop))
        policies.append(sample.policy_version)
    datum = Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData(data=targets, dtype="int64", shape=[length]),
            "advantages": TensorData(data=weights, dtype="float32", shape=[length]),
            "logprobs": TensorData(data=logprobs, dtype="float32", shape=[length]),
        },
    )
    spans = chain_provenance(chain)
    if spans is not None:
        datum = datum.with_provenance(model_input=spans[0], loss_fn_inputs=spans[1])
    return TrainingRow(datum, tuple(actions), tuple(policies))


type Spans = list[PromptProvenanceSpan | SampledProvenanceSpan]


def chain_provenance(chain: Sequence[Sample]) -> tuple[Spans, Spans] | None:
    """Where each of a chain row's inputs and targets was sampled; ``None`` when a
    generation in it was not recorded.

    Every position names the earliest record that holds it as the sampler saw it. A
    generation's own tokens are its sampled tokens, except its last, which the sampler
    never ran forward: as an input, that one comes from the next generation's prompt,
    as do the environment's tokens between the two. Targets are the same sequence one
    position on, so a generation's sampled tokens are all its own.
    """
    if any(sample.sequence_id is None for sample in chain):
        return None
    inputs: Spans = []
    targets: Spans = []
    input_end, target_end = 0, 1
    for sample in chain:
        sid = sample.sequence_id
        assert sid is not None
        prompt, completion = len(sample.prompt_ids), len(sample.completion_ids)
        if prompt > input_end:
            inputs.append(
                PromptProvenanceSpan(sequence_id=sid, offset=input_end, length=prompt - input_end)
            )
        if completion > 1:
            inputs.append(SampledProvenanceSpan(sequence_id=sid, length=completion - 1))
        if prompt > target_end:
            targets.append(
                PromptProvenanceSpan(sequence_id=sid, offset=target_end, length=prompt - target_end)
            )
        targets.append(SampledProvenanceSpan(sequence_id=sid, length=completion))
        input_end, target_end = prompt + completion - 1, prompt + completion
    return inputs, targets


def without_provenance(row: TrainingRow) -> TrainingRow:
    if not row.routed:
        return row
    return row.with_datum(
        row.datum.model_copy(update={"model_input_spans": None, "loss_fn_input_spans": None})
    )


def normalize_batch_provenance(
    rows: list[TrainingRow], *, enabled: bool = True
) -> list[TrainingRow]:
    """Replay the complete optimizer batch, or train it entirely without replay: the
    trainer refuses a request that mixes the two."""
    if enabled and all(row.routed for row in rows):
        return rows
    return [without_provenance(row) for row in rows]
