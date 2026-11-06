import random
from dataclasses import replace

import pytest
from baseten.loops.models import EncodedTextChunk, ImageChunk, ModelInput
from baseten.loops.provenance import validate_provenance
from support.builders import make_rollout, make_sample

from loops_cookbook.evidence import TrajectoryError
from loops_cookbook.rl.grpo import (
    IGNORE_INDEX,
    advantages,
    build_rows,
    build_rows_in_slices,
    normalize_batch_provenance,
)


def sample(identity, prompt, completion, version=1, conversation="main", recorded=False):
    return make_sample(
        identity,
        tuple(prompt),
        tuple(completion),
        tuple(-0.5 for _ in completion),
        version,
        conversation_id=conversation,
        sequence_id=f"seq-{identity}" if recorded else None,
    )


def targets(row):
    return row.datum.loss_fn_inputs["target_tokens"].data


def resolve(spans, samples):
    """The tokens a list of spans names, read from the samples' own evidence."""
    by_id = {s.sequence_id: s for s in samples}
    tokens = []
    for span in spans:
        source = by_id[span.sequence_id]
        ids = source.prompt_ids if span.type == "prompt" else source.completion_ids
        assert span.offset + span.length <= len(ids), "a span runs past its sequence"
        tokens.extend(ids[span.offset : span.offset + span.length])
    return tokens


def assert_provenance_names_the_row(row, chain):
    """The trainer's checks, made here: both lists tile their field, and every span
    names exactly the tokens at its positions."""
    datum = row.datum
    shape = datum.loss_fn_inputs["target_tokens"].shape
    validate_provenance(
        datum.model_input_spans,
        datum.loss_fn_input_spans,
        datum.model_input.length,
        shape,
        len(targets(row)),
    )
    assert resolve(datum.model_input_spans, chain) == datum.model_input.to_ints()
    assert resolve(datum.loss_fn_input_spans, chain) == targets(row)
    # The sampler never ran a generation's last token forward, so it has no input
    # capture of its own: no input span reads it from the sampled tokens.
    by_id = {s.sequence_id: s for s in chain}
    for span in datum.model_input_spans:
        if span.type == "sampled":
            assert span.offset + span.length <= len(by_id[span.sequence_id].completion_ids) - 1


# Rows and their actions


def test_linear_rows_train_each_action_once_with_real_targets():
    rollout = make_rollout(
        "task",
        "group",
        "rollout",
        (sample("a", [1, 2], [3, 4]), sample("b", [1, 2, 3, 4, 5], [6], 3)),
    )
    (row,) = build_rows(rollout, 0.0)
    assert row.trained_positions == 3
    assert row.actions == ((1, 3), (4, 5))
    assert targets(row) == [2, 3, 4, 5, 6]
    assert rollout.minimum_policy_version == 1
    assert rollout.internal_delta == 2


def test_auxiliary_samples_keep_context_without_actions_or_freshness_weight():
    auxiliary = replace(sample("aux", [1], [2], version=0), producer="summary", trainable=False)
    main = sample("main", [1, 2, 3], [4], version=3)
    rollout = make_rollout("t", "g", "r", (auxiliary, main))
    (row,) = build_rows(rollout, 1)
    assert targets(row) == [2, 3, 4]
    assert row.actions == ((2, 3),)
    assert row.datum.loss_fn_inputs["advantages"].data == [0.0, 0.0, 1.0]
    assert rollout.minimum_policy_version == 3
    assert rollout.internal_delta == 0


def test_a_zero_advantage_row_still_counts_its_actions():
    """Positions are trained by the mask, not by their advantage: a group whose rewards
    all tie still trains its agent's tokens."""
    (row,) = build_rows(make_rollout("t", "g", "r", (sample("a", [1, 2], [3, 4]),)), 0.0)
    assert set(row.datum.loss_fn_inputs["advantages"].data) == {0.0}
    assert row.trained_positions == 2


def test_image_positions_have_no_target():
    image = ImageChunk(data=b"img", format="png", expected_tokens=2)
    prompt_input = ModelInput(
        chunks=[EncodedTextChunk(tokens=[1]), image, EncodedTextChunk(tokens=[4])]
    )
    first = make_sample("a", (1, 7, 7, 4), (5, 6), (-0.5, -0.5), 1, prompt_input=prompt_input)
    (row,) = build_rows(make_rollout("t", "g", "r", (first,)), 1)
    assert targets(row) == [IGNORE_INDEX, IGNORE_INDEX, 4, 5, 6]
    assert row.actions == ((3, 5),)


def test_fork_trains_history_once_per_chain():
    rollout = make_rollout(
        "task",
        "group",
        "rollout",
        (sample("a", [1], [2]), sample("b", [9, 2, 8], [3], conversation="fork")),
    )
    rows = build_rows(rollout, -0.5)
    assert len(rows) == 2
    assert targets(rows[1]) == [2, 8, 3]
    assert rows[1].actions == ((2, 3),)
    assert sum(row.trained_positions for row in rows) == 2


def test_duplicate_identity_rejected_but_identical_independent_draws_allowed():
    first = sample("a", [1], [2])
    with pytest.raises(TrajectoryError, match="only once"):
        make_rollout("t", "g", "r", (first, first))
    rollout = make_rollout("t", "g", "r", (first, sample("b", [1], [2])))
    assert sum(row.trained_positions for row in build_rows(rollout, 1)) == 2


def test_invalid_sampling_evidence_and_rewards_rejected():
    with pytest.raises(TrajectoryError):
        make_sample("a", (1,), (2,), (), 1)
    with pytest.raises(TrajectoryError):
        sample("a", [1], [2], version=None)
    with pytest.raises(TrajectoryError, match="sequence_id"):
        make_sample("a", (1,), (2,), (-0.5,), 1, sequence_id="")
    with pytest.raises(TrajectoryError):
        advantages([float("nan"), 0])
    assert advantages([1, 0]) == [0.5, -0.5]


def test_a_forked_rollout_pauses_after_each_chains_row():
    """Each chain's row is a pass over its full length, so a rollout with many long
    chains is as many slices, not one."""
    forked = make_rollout(
        "t", "g", "r", tuple(sample(str(i), [100 + i], [i + 1]) for i in range(5))
    )
    work = build_rows_in_slices(forked, 0.5)
    pauses = 0
    while True:
        try:
            next(work)
        except StopIteration as done:
            rows = done.value
            break
        pauses += 1
    assert pauses == len(rows) == 5
    assert rows == build_rows(forked, 0.5)


def test_input_matches_the_samplers_forwarded_length():
    rollout = make_rollout(
        "t", "g", "r", (sample("a", [1, 2], [3]), sample("b", [1, 2, 3, 4], [5, 6]))
    )
    (row,) = build_rows(rollout, 0.5)
    datum = row.datum
    assert datum.model_input.to_ints() == [1, 2, 3, 4, 5]
    assert targets(row)[-1] == 6
    assert all(
        tensor.shape == [datum.model_input.length] for tensor in datum.loss_fn_inputs.values()
    )


# Chains


def test_interleaved_conversations_keep_their_own_prefix_chains():
    first = sample("first", [1], [2], version=2)
    auxiliary = replace(
        sample("aux", [9], [8], version=0, conversation="summary"),
        producer="summary",
        trainable=False,
    )
    second = sample("second", [1, 2, 3], [4], version=4)
    rollout = make_rollout("t", "g", "r", (first, auxiliary, second))
    assert [[s.sample_id for s in chain] for chain in rollout.chains] == [
        ["first", "second"],
        ["aux"],
    ]
    (row,) = build_rows(rollout, 1)
    assert targets(row) == [2, 3, 4]
    assert row.actions == ((0, 1), (2, 3))
    assert rollout.minimum_policy_version == 2


def test_return_to_older_chain_reuses_eligible_tail_without_duplicate_training():
    rollout = make_rollout(
        "t",
        "g",
        "r",
        (sample("first", [1], [2]), sample("compact", [9], [8]), sample("revisit", [1, 2, 3], [4])),
    )
    assert [[s.sample_id for s in chain] for chain in rollout.chains] == [
        ["first", "revisit"],
        ["compact"],
    ]
    assert sum(row.trained_positions for row in build_rows(rollout, 1)) == 3


def test_sibling_branch_splits_only_on_actual_prefix_divergence():
    rollout = make_rollout(
        "t",
        "g",
        "r",
        (
            sample("parent", [1], [2]),
            sample("left", [1, 2, 3], [4], conversation="left"),
            sample("right", [1, 2, 5], [6], conversation="right"),
        ),
    )
    assert [[s.sample_id for s in chain] for chain in rollout.chains] == [
        ["parent", "left"],
        ["right"],
    ]
    rows = build_rows(rollout, 1)
    assert sum(row.trained_positions for row in rows) == 3
    assert targets(rows[1]) == [2, 5, 6]
    assert rows[1].actions == ((2, 3),)


def test_interleaved_producer_labels_do_not_split_compatible_tokens():
    rollout = make_rollout(
        "t",
        "g",
        "r",
        (
            sample("first", [1], [2], version=3),
            replace(
                sample("aux", [1, 2, 3], [4], version=0, conversation="other"),
                producer="aux",
                trainable=False,
            ),
            sample("last", [1, 2, 3, 4, 5], [6], version=3, conversation="restored"),
        ),
    )
    (row,) = build_rows(rollout, 1)
    assert targets(row) == [2, 3, 4, 5, 6]
    assert row.actions == ((0, 1), (4, 5))
    assert rollout.minimum_policy_version == 3
    assert rollout.internal_delta == 0


def test_longest_eligible_tail_selected_with_stable_ties():
    rollout = make_rollout(
        "t",
        "g",
        "r",
        (
            sample("long", [1, 2, 3], [4]),
            sample("short", [1], [2]),
            sample("tie", [1], [2]),
            sample("long-child", [1, 2, 3, 4, 5], [6]),
            sample("short-child", [1, 2, 7], [8]),
        ),
    )
    assert [[s.sample_id for s in chain] for chain in rollout.chains] == [
        ["long", "long-child"],
        ["short", "short-child"],
        ["tie"],
    ]
    assert sum(row.trained_positions for row in build_rows(rollout, 1)) == 5


# Provenance


def test_a_single_generation_names_its_prompt_and_sampled_tokens():
    first = sample("a", [1, 2, 3], [4, 5], recorded=True)
    (row,) = build_rows(make_rollout("t", "g", "r", (first,)), 1)
    spans = [(s.type, s.offset, s.length) for s in row.datum.model_input_spans]
    assert spans == [("prompt", 0, 3), ("sampled", 0, 1)]
    spans = [(s.type, s.offset, s.length) for s in row.datum.loss_fn_input_spans]
    assert spans == [("prompt", 1, 2), ("sampled", 0, 2)]
    assert_provenance_names_the_row(row, (first,))


def test_a_turns_last_token_and_the_observation_come_from_the_next_prompt():
    first = sample("a", [1, 2], [3, 4], recorded=True)
    second = sample("b", [1, 2, 3, 4, 5, 6], [7], recorded=True)
    (row,) = build_rows(make_rollout("t", "g", "r", (first, second)), 1)
    spans = [(s.sequence_id, s.type, s.offset, s.length) for s in row.datum.model_input_spans]
    assert spans == [
        ("seq-a", "prompt", 0, 2),
        ("seq-a", "sampled", 0, 1),
        ("seq-b", "prompt", 3, 3),
    ]
    assert_provenance_names_the_row(row, (first, second))


def test_one_token_turns_have_no_sampled_input():
    first = sample("a", [1], [2], recorded=True)
    second = sample("b", [1, 2], [3], recorded=True)
    (row,) = build_rows(make_rollout("t", "g", "r", (first, second)), 1)
    assert {s.type for s in row.datum.model_input_spans} == {"prompt"}
    assert_provenance_names_the_row(row, (first, second))


def test_turns_under_different_policies_keep_their_own_records():
    first = sample("a", [1], [2, 3], version=3, recorded=True)
    second = sample("b", [1, 2, 3, 4], [5], version=4, recorded=True)
    (row,) = build_rows(make_rollout("t", "g", "r", (first, second)), 1)
    assert {s.sequence_id for s in row.datum.loss_fn_input_spans} == {"seq-a", "seq-b"}
    assert_provenance_names_the_row(row, (first, second))


def test_each_fork_names_its_own_chain():
    parent = sample("parent", [1], [2], recorded=True)
    left = sample("left", [1, 2, 3], [4], conversation="left", recorded=True)
    right = sample("right", [1, 2, 5], [6], conversation="right", recorded=True)
    rollout = make_rollout("t", "g", "r", (parent, left, right))
    rows = build_rows(rollout, 1)
    assert all(row.routed for row in rows)
    for row, chain in zip(rows, rollout.chains, strict=True):
        assert_provenance_names_the_row(row, chain)


def test_an_unrecorded_generation_leaves_its_row_without_provenance():
    first = sample("a", [1], [2], recorded=True)
    second = sample("b", [1, 2, 3], [4])
    (row,) = build_rows(make_rollout("t", "g", "r", (first, second)), 1)
    assert not row.routed
    assert row.datum.loss_fn_input_spans is None


def random_chain(rng: random.Random, turns: int):
    """A chain of recorded generations: each prompt the whole history so far plus a
    possibly empty observation, each completion at least one token."""
    history = [rng.randrange(1, 50) for _ in range(rng.randrange(1, 4))]
    chain = []
    for turn in range(turns):
        if turn:
            history.extend(rng.randrange(1, 50) for _ in range(rng.randrange(0, 4)))
        completion = [rng.randrange(1, 50) for _ in range(rng.randrange(1, 4))]
        produced = make_sample(
            f"s{turn}",
            tuple(history),
            tuple(completion),
            tuple(-0.5 for _ in completion),
            rng.randrange(0, 3),
            producer="main" if rng.random() < 0.7 else "aux",
            trainable=True if turn == 0 else rng.random() < 0.7,
            sequence_id=f"seq-{turn}",
        )
        chain.append(produced)
        history.extend(completion)
    return tuple(chain)


@pytest.mark.parametrize("seed", range(200))
def test_provenance_names_every_position_of_any_chain(seed):
    rng = random.Random(seed)
    chain = random_chain(rng, rng.randrange(1, 6))
    rollout = make_rollout("t", "g", "r", chain)
    assert len(rollout.chains) == 1
    (row,) = build_rows(rollout, 1)
    assert row.routed
    assert_provenance_names_the_row(row, chain)
    assert row.trained_positions == sum(len(s.completion_ids) for s in chain if s.trainable)


# Batch normalization


def test_one_unrecorded_row_takes_the_whole_batch_off_replay():
    recorded = [
        build_rows(make_rollout("t", "g", str(i), (sample(str(i), [1], [2], recorded=True),)), 1)[0]
        for i in range(3)
    ]
    unrecorded = build_rows(make_rollout("t", "g", "x", (sample("x", [1], [2]),)), 1)
    normalized = normalize_batch_provenance(recorded + unrecorded)
    assert not any(row.routed for row in normalized)
    for before, after in zip(recorded + unrecorded, normalized, strict=True):
        assert after.actions == before.actions
        assert before.datum.model_dump(
            exclude={"model_input_spans", "loss_fn_input_spans"}
        ) == after.datum.model_dump(exclude={"model_input_spans", "loss_fn_input_spans"})


def test_a_fully_recorded_batch_keeps_its_provenance():
    rows = [
        build_rows(make_rollout("t", "g", str(i), (sample(str(i), [1], [2], recorded=True),)), 1)[0]
        for i in range(4)
    ]
    assert normalize_batch_provenance(rows) is rows
    assert not any(row.routed for row in normalize_batch_provenance(rows, enabled=False))
