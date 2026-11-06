"""A step's batch is prepared in slices on the event loop, and exactly as before.

``reference_prepare`` is the per-token implementation the sliced one replaced, kept
verbatim: every row, reward, advantage, aggregation report and gauge must match it,
and so must every refusal.
"""

import asyncio
import random
import weakref
from array import array
from dataclasses import asdict, replace
from math import isfinite
from types import SimpleNamespace

import pytest
from baseten.loops import PromptProvenanceSpan, SampledProvenanceSpan
from baseten.loops.models import (
    Datum,
    EncodedTextChunk,
    ImageChunk,
    ModelInput,
    TensorData,
)
from support.builders import make_config, make_rollout

from loops_cookbook.evidence import PromptChunks, Sample, TokenBuffers, TrajectoryError
from loops_cookbook.rl.aggregation import AggregatedBatch, AggregationReport
from loops_cookbook.rl.grpo import (
    IGNORE_INDEX,
    TrainingRow,
    advantages,
    build_rows,
    normalize_batch_provenance,
)
from loops_cookbook.rl.metrics import rollout_metrics, trajectory_metrics
from loops_cookbook.rl.run import PreparedBatch, prepare_batch
from loops_cookbook.rl.slices import in_slices, whole

# ------------------------------------------------- the per-token reference, verbatim


def reference_build_datums(trajectory, advantage, chains=None):
    if not isfinite(advantage):
        raise TrajectoryError("advantage must be finite")
    parts = trajectory.chains if chains is None else chains
    if not parts:
        raise TrajectoryError(f"rollout {trajectory.identity.rollout_id} has no samples")
    result = []
    for chain in parts:
        if not any(sample.trainable for sample in chain):
            continue
        last = chain[-1]
        tokens = last.prompt_ids + last.completion_ids
        length = len(tokens) - 1
        model_input = last.prompt_input.model_copy(deep=True)
        if len(last.completion_ids) > 1:
            model_input = model_input.append(
                EncodedTextChunk(tokens=list(last.completion_ids[:-1]))
            )
        images = set()
        position = 0
        for chunk in last.prompt_input.chunks:
            if not isinstance(chunk, EncodedTextChunk):
                images.update(range(position, position + chunk.length))
            position += chunk.length
        targets = [
            IGNORE_INDEX if position + 1 in images else tokens[position + 1]
            for position in range(length)
        ]
        weights = [0.0] * length
        logprobs = [0.0] * length
        actions = []
        policies = []
        for sample in chain:
            if not sample.trainable:
                continue
            for offset, probability in enumerate(sample.logprobs):
                position = len(sample.prompt_ids) - 1 + offset
                targets[position] = sample.completion_ids[offset]
                weights[position] = advantage
                logprobs[position] = probability
            start = len(sample.prompt_ids) - 1
            actions.append((start, start + len(sample.completion_ids)))
            policies.append(sample.policy_version)
        datum = Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": TensorData(data=targets, dtype="int64", shape=[length]),
                "advantages": TensorData(data=weights, dtype="float32", shape=[length]),
                "logprobs": TensorData(data=logprobs, dtype="float32", shape=[length]),
            },
        )
        if all(sample.sequence_id is not None for sample in chain):
            datum = datum.with_provenance(
                model_input=reference_spans(chain, range(length), forwarded=True),
                loss_fn_inputs=reference_spans(chain, range(1, length + 1), forwarded=False),
            )
        result.append(TrainingRow(datum, tuple(actions), tuple(policies)))
    if not result:
        raise TrajectoryError("rollout contains no trainable generations")
    return result


def reference_spans(chain, positions, *, forwarded):
    """Each sequence position's source, a position at a time: the first generation that
    held it -- as an input, one that ran it forward, so not its own last token."""
    spans = []
    for position in positions:
        for sample in chain:
            end = len(sample.prompt_ids) + len(sample.completion_ids) - (1 if forwarded else 0)
            if position < end:
                break
        prompt = len(sample.prompt_ids)
        kind, index = ("prompt", position) if position < prompt else ("sampled", position - prompt)
        previous = spans[-1] if spans else None
        if (
            previous is not None
            and previous.sequence_id == sample.sequence_id
            and previous.type == kind
            and previous.offset + previous.length == index
        ):
            spans[-1] = previous.model_copy(update={"length": previous.length + 1})
            continue
        span = PromptProvenanceSpan if kind == "prompt" else SampledProvenanceSpan
        spans.append(span(sequence_id=sample.sequence_id, offset=index, length=1))
    return spans


def reference_trained_positions(row):
    return sum(stop - start for start, stop in row.actions)


def reference_active_positions(datum):
    weights = reference_float32(datum.loss_fn_inputs["advantages"].data)
    return sum(
        target != IGNORE_INDEX and weight != 0
        for target, weight in zip(datum.loss_fn_inputs["target_tokens"].data, weights, strict=True)
    )


def reference_float32(values):
    try:
        result = array("f", values)
    except OverflowError as error:
        raise ValueError("advantages must be finite in float32") from error
    if not all(isfinite(value) for value in result):
        raise ValueError("advantages must be finite in float32")
    return result


def reference_aggregate_loss(groups, training):
    rows = [row for group in groups for row in group]
    if training.loss_fn == "custom":
        return AggregatedBatch(rows, None)
    counts = [sum(reference_trained_positions(row) for row in group) for group in groups]
    if not counts or any(count == 0 for count in counts):
        raise ValueError("loss aggregation requires action tokens in every task group")
    active = sum(reference_active_positions(row.datum) for row in rows)
    if active == 0:
        raise ValueError("loss aggregation requires nonzero float32 advantages")
    mode = training.loss_aggregation
    denominator, group_denominators = None, ()
    if mode in ("sum", "token_mean"):
        denominator = active if mode == "token_mean" else None
        scales = [1.0] * len(groups)
    else:
        if mode == "fixed_budget":
            denominator = training.rollouts_per_step * training.loss_token_budget
            scales = [1 / denominator] * len(groups)
        else:
            group_denominators = tuple(len(groups) * count for count in counts)
            scales = [1 / value for value in group_denominators]
        rows = [
            TrainingRow(reference_scale_advantages(row.datum, scale), row.actions, row.policies)
            for group, scale in zip(groups, scales, strict=True)
            for row in group
        ]
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


def reference_scale_advantages(datum, scale):
    if not isfinite(scale) or scale <= 0:
        raise ValueError("loss aggregation requires a finite positive advantage scale")
    original = datum.loss_fn_inputs["advantages"]
    weights = reference_float32(original.data)
    scaled = reference_float32([weight * scale for weight in weights])
    if any((before != 0) != (after != 0) for before, after in zip(weights, scaled, strict=True)):
        raise ValueError("loss aggregation would change the float32 active-token mask")
    return datum.model_copy(
        update={
            "loss_fn_inputs": {
                **datum.loss_fn_inputs,
                "advantages": TensorData(data=list(scaled), dtype="float32", shape=original.shape),
            }
        }
    )


def reference_prepare(groups, training):
    datums_by_group, rewards, advs, versions = [], [], [], []
    forked = 0
    for group in groups:
        group_rewards = [rollout.reward for rollout in group.members]
        if any(reward is None for reward in group_rewards):
            raise ValueError("cannot train an ungraded rollout")
        group_advs = advantages(group_rewards)
        datums = []
        for rollout, advantage in zip(group.members, group_advs, strict=True):
            chains = rollout.chains
            forked += len(chains) > 1
            datums.extend(reference_build_datums(rollout, advantage, chains))
            versions.extend(rollout.policy_versions)
        datums_by_group.append(datums)
        rewards.extend(group_rewards)
        advs.extend(group_advs)
    batch = reference_aggregate_loss(datums_by_group, training)
    selected = [rollout for group in groups for rollout in group.members]
    unrecorded = (
        sum(row.datum.model_input_spans is None for row in batch.rows)
        if training.router_replay
        else 0
    )
    rows = normalize_batch_provenance(batch.rows, enabled=training.router_replay)
    return PreparedBatch(
        rows,
        batch.report,
        rewards,
        advs,
        versions,
        forked,
        sum(row.datum.model_input_spans is not None for row in rows),
        unrecorded,
        {**trajectory_metrics(selected), **rollout_metrics(selected)},
    )


# ----------------------------------------------------------------------- evidence


def prompt_input(rng, ids, image):
    """Chunks covering ``ids``: an image over its first ``image`` positions when they
    fit, then text cut at random points."""
    chunks, start = [], 0
    if image and len(ids) > image:
        chunks.append(ImageChunk(data=b"image bytes", format="png", expected_tokens=image))
        start = image
    while start < len(ids):
        stop = rng.randint(start + 1, len(ids))
        chunks.append(EncodedTextChunk(tokens=list(ids[start:stop])))
        start = stop
    return ModelInput(chunks=chunks)


def random_rollout(rng, name, reward, *, turns=8, scale=1, recorded=0.7):
    """Generations that extend, fork from and interleave conversations, with spans and
    tuples, images, untrainable turns, mixed policy versions and sampler records."""
    buffers = TokenBuffers()
    ends: list[tuple[str, tuple[int, ...]]] = []  # each conversation's tokens so far
    images: dict[str, int] = {}
    samples = []
    for turn in range(rng.randint(1, turns)):
        draw = rng.random()
        tool = tuple(rng.randrange(1000) for _ in range(rng.randint(0, 5 * scale)))
        if ends and draw < 0.6:
            conversation, history = rng.choice(ends)
            prompt = history + tool
        elif ends and draw < 0.8:
            conversation, history = rng.choice(ends)
            conversation = f"{conversation}-fork{turn}"
            prompt = history[: rng.randint(1, len(history))] + tool
        else:
            conversation = f"c{turn}"
            prompt = tuple(rng.randrange(1000) for _ in range(rng.randint(1, 12 * scale)))
            images[conversation] = rng.choice([0, 0, 2])
        prompt = prompt or (rng.randrange(1000),)
        completion = tuple(rng.randrange(1000) for _ in range(rng.randint(1, 6 * scale)))
        samples.append(
            Sample(
                sample_id=f"{name}-s{turn}",
                producer=rng.choice(["main", "judge"]),
                conversation_id=conversation,
                trainable=rng.random() < 0.8,
                prompt_ids=buffers.span(conversation, prompt) if rng.random() < 0.5 else prompt,
                completion_ids=completion,
                logprobs=tuple(-rng.random() * 4 for _ in completion),
                policy_version=rng.choice([3, 3, 4]),
                prompt_chunks=PromptChunks.of(
                    prompt_input(rng, prompt, images.get(conversation.split("-")[0], 0)), prompt
                ),
                stop_reason=rng.choice(["stop", "length", None]),
                sequence_id=f"{name}-{turn}" if rng.random() < recorded else None,
            )
        )
        ends.append((conversation, prompt + completion))
    if not any(sample.trainable for sample in samples):
        samples[-1] = replace(samples[-1], trainable=True)
    return make_rollout(
        "t",
        "g",
        name,
        samples,
        reward,
        metadata={"turns": len(samples), "solved": reward > 0.5, "outcome": rng.choice("ab")},
    )


def random_groups(seed, *, groups=3, size=4, rewards=(0.0, 0.5, 1.0), **shape):
    rng = random.Random(seed)
    return [
        SimpleNamespace(
            members=[
                random_rollout(rng, f"g{g}r{r}", rng.choice(rewards), **shape) for r in range(size)
            ]
        )
        for g in range(groups)
    ]


def training(mode, loss_fn="cispo", router_replay=True, budget=10):
    settings = make_config(
        loss_fn="cispo" if loss_fn == "custom" else loss_fn,
        loss_aggregation=mode,
        router_replay=router_replay,
        group_size=4,
        groups_per_step=3,
        max_group_losses=0,
        **({"loss_token_budget": budget} if mode == "fixed_budget" else {}),
    ).training
    # Preparation reads only the name of a custom loss, not its table.
    return settings.model_copy(update={"loss_fn": loss_fn})


def outcome(prepare):
    """What a preparation returns or refuses, as exact text: a float that changed, an
    int that became a float, or a reworded refusal all show."""
    try:
        batch = prepare()
    except (ValueError, TrajectoryError) as error:
        return ("refused", type(error).__name__, str(error))
    return (
        [repr(row.datum.model_dump()) for row in batch.rows],
        [row.datum.model_dump_json() for row in batch.rows],
        [row.actions for row in batch.rows],
        repr(asdict(batch.aggregation)) if batch.aggregation is not None else None,
        repr(batch.rewards),
        repr(batch.advantages),
        batch.versions,
        batch.forked_rollouts,
        batch.routed_datums,
        batch.unrecorded_datums,
        repr(batch.rollout_gauges),
    )


SETTINGS = [
    ("sum", "cispo", True),
    ("token_mean", "cispo", True),
    ("fixed_budget", "cispo", False),
    ("fixed_budget", "ppo", True),
    ("task_mean", "importance_sampling", True),
    ("sum", "custom", False),
]


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize(("mode", "loss_fn", "router_replay"), SETTINGS)
@pytest.mark.parametrize("recorded", [0.7, 1.0], ids=["some-recorded", "all-recorded"])
def test_a_sliced_preparation_is_exactly_the_per_token_one(
    seed, mode, loss_fn, router_replay, recorded
):
    groups = random_groups(seed, recorded=recorded)
    settings = training(mode, loss_fn, router_replay)
    expected = outcome(lambda: reference_prepare(groups, settings))
    assert outcome(lambda: whole(prepare_batch(groups, settings))) == expected


@pytest.mark.parametrize(
    ("rewards", "mode", "budget", "refusal"),
    [
        # Every group's rewards are equal: nothing to learn from anywhere.
        ((0.5,), "token_mean", 10, "nonzero float32 advantages"),
        # Advantages that are float32 subnormals, scaled below the smallest one.
        ((0.0, 1e-44), "fixed_budget", 10**6, "active-token mask"),
        # An advantage beyond float32.
        ((0.0, 1e39), "token_mean", 10, "finite in float32"),
    ],
)
def test_refusals_are_the_per_token_ones(rewards, mode, budget, refusal):
    groups = random_groups(1, rewards=rewards)
    settings = training(mode, budget=budget)
    expected = outcome(lambda: reference_prepare(groups, settings))
    assert expected[0] == "refused" and refusal in expected[2]
    assert outcome(lambda: whole(prepare_batch(groups, settings))) == expected


def test_a_chain_whose_generations_are_not_its_prefixes_is_refused():
    (rollout,) = random_groups(2, groups=1, size=1, turns=1)[0].members
    (sample,) = rollout.samples
    longer = replace(sample, sample_id="longer", completion_ids=(1,) * 50, logprobs=(-1.0,) * 50)
    with pytest.raises(TrajectoryError, match="prefixes of its last"):
        build_rows(rollout, 0.5, [(longer, sample)])


# --------------------------------------------------------------------- the loop


async def test_the_loop_gets_a_turn_between_slices():
    """With no budget, every unit of the batch -- a rollout, a row -- is followed by a
    turn of the loop, whatever the machine's speed."""
    groups = random_groups(3)
    turns = 0

    async def count():
        nonlocal turns
        while True:
            turns += 1
            await asyncio.sleep(0)

    counter = asyncio.create_task(count())
    await asyncio.sleep(0)
    try:
        prepared = await in_slices(prepare_batch(groups, training("fixed_budget")), budget=0)
    finally:
        counter.cancel()
    rollouts = sum(len(group.members) for group in groups)
    # One per rollout scored, per row weighed and scaled, per group released. Counting
    # a row's actions reads its ranges, not its tokens, so it is no unit of its own.
    assert turns >= rollouts + 2 * len(prepared.rows)


async def test_a_cancelled_preparation_releases_what_it_built():
    """Cancelled between slices, the work is closed: what it built so far goes then,
    not when the cancellation's traceback does, after the run's teardown."""

    class Rows:
        pass

    built: list[weakref.ref[Rows]] = []

    def work():
        rows = Rows()
        built.append(weakref.ref(rows))
        while True:
            yield

    task = asyncio.create_task(in_slices(work(), budget=0))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as held:
        await task
    assert held.value.__traceback__ is not None
    assert built[0]() is None


def test_a_rollout_that_trains_nothing_is_in_the_baseline_and_has_no_rows():
    rng = random.Random(0)
    solved, failed = random_rollout(rng, "a", 1.0), random_rollout(rng, "b", 0.0)
    cut = replace(make_rollout("t", "g", "c", (), 0.0), discarded_versions=(3,))
    batch = whole(prepare_batch([SimpleNamespace(members=[solved, failed, cut])], training("sum")))
    # The cut-off rollout's 0 is in the mean: the solved one's advantage is 2/3, not 1/2.
    up, down, _ = advantages([1.0, 0.0, 0.0])
    assert batch.advantages == [up, down, down] and up == pytest.approx(2 / 3)
    rows = build_rows(solved, up) + build_rows(failed, down)
    assert [row.datum.model_dump() for row in batch.rows] == [
        row.datum.model_dump() for row in normalize_batch_provenance(rows, enabled=True)
    ]
