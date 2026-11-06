import pytest
from baseten.loops.models import Datum, ModelInput, TensorData

from loops_cookbook.rl.grpo import TrainingRow
from loops_cookbook.rl.mismatch import MismatchTotals, UnfitScores, action_scores

PREFIX = "mismatch/abs_logprob_diff"


def row(policies):
    """One two-token action per turn, sampled at that turn's policy version, with a
    sampler logprob of -1 everywhere and a one-token observation between turns."""
    length = 1 + 3 * len(policies)
    actions = tuple((1 + 3 * i, 3 + 3 * i) for i in range(len(policies)))
    datum = Datum(
        model_input=ModelInput.from_ints(list(range(length))),
        loss_fn_inputs={
            "target_tokens": TensorData(
                data=list(range(1, length + 1)), dtype="int64", shape=[length]
            ),
            "advantages": TensorData(data=[0.0] * length, dtype="float32", shape=[length]),
            "logprobs": TensorData(data=[-1.0] * length, dtype="float32", shape=[length]),
        },
    )
    return TrainingRow(datum, actions, tuple(policies))


def scored(row, gaps, outside=0.0):
    """The trainer's logprobs: the sampler's plus ``gaps[i]`` on action i, plus
    ``outside`` at every position that does not train."""
    values = [-1.0 + outside] * row.datum.model_input.length
    for (start, stop), gap in zip(row.actions, gaps, strict=True):
        values[start:stop] = [-1.0 + gap] * (stop - start)
    return values


@pytest.mark.parametrize(
    ("turn", "bucket"),
    [
        (1, "1"),
        (2, "2"),
        (3, "3-4"),
        (4, "3-4"),
        (5, "5-8"),
        (8, "5-8"),
        (9, "9-16"),
        (16, "9-16"),
        (17, "17+"),
        (40, "17+"),
    ],
)
def test_an_actions_gap_lands_in_its_turn_bucket_only(turn, bucket):
    r = row([0] * turn)
    totals = MismatchTotals()
    totals.add([r], action_scores([r], [scored(r, [0.0] * (turn - 1) + [0.5])]), version=0)
    gauges = totals.gauges()
    turns = {key: value for key, value in gauges.items() if "/turn_" in key}
    assert [key for key, value in turns.items() if value] == [f"{PREFIX}/turn_{bucket}"]


def test_gaps_are_token_means_split_by_turn_and_policy_lag():
    # Trainer at version 5. First row: turns at versions 5, 4, 2 (lags 0, 1, 3).
    first, second = row([5, 4, 2]), row([5])
    totals = MismatchTotals()
    totals.add([first], action_scores([first], [scored(first, [0.1, -0.3, 0.6])]), version=5)
    totals.add([second], action_scores([second], [scored(second, [0.2])]), version=5)
    assert totals.gauges() == pytest.approx(
        {
            PREFIX: (0.1 + 0.3 + 0.6 + 0.2) / 4,
            f"{PREFIX}/turn_1": (0.1 + 0.2) / 2,
            f"{PREFIX}/turn_2": 0.3,
            f"{PREFIX}/turn_3-4": 0.6,
            f"{PREFIX}/lag_0": (0.1 + 0.2) / 2,
            f"{PREFIX}/lag_1": 0.3,
            f"{PREFIX}/lag_2+": 0.6,
            f"{PREFIX}_max": 0.6,
        }
    )


@pytest.mark.parametrize("outside", [5.0, float("nan")])
def test_positions_that_do_not_train_are_ignored(outside):
    r = row([0, 0])
    totals = MismatchTotals()
    totals.add([r], action_scores([r], [scored(r, [0.25, 0.25], outside=outside)]), version=0)
    assert totals.gauges()[PREFIX] == pytest.approx(0.25)
    assert totals.gauges()[f"{PREFIX}_max"] == pytest.approx(0.25)


def test_nothing_added_reports_nothing():
    assert MismatchTotals().gauges() == {}


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda r: [], "for 0 of 1 datums"),
        (lambda r: [scored(r, [0.0])[:-1]], "logprobs for a datum of"),
        (lambda r: [scored(r, [0.0])] * 2, "for 2 of 1 datums"),
    ],
)
def test_a_trainer_answer_that_does_not_fit_the_rows_is_refused(make, match):
    r = row([0])
    with pytest.raises(UnfitScores, match=match):
        action_scores([r], make(r))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_nonfinite_trainer_logprob_is_counted_not_averaged(bad):
    r = row([0, 0])
    values = scored(r, [0.25, 0.25])
    values[r.actions[1][0]] = bad
    totals = MismatchTotals()
    totals.add([r], action_scores([r], [values]), version=0)
    gauges = totals.gauges()
    assert gauges["mismatch/nonfinite_tokens"] == 1
    assert gauges[PREFIX] == pytest.approx(0.25)
    assert gauges[f"{PREFIX}/turn_2"] == pytest.approx(0.25)


def test_a_null_trainer_logprob_is_counted_not_averaged():
    """A position the trainer left unscored is nonfinite, as SFT scoring treats it."""
    r = row([0, 0])
    values = scored(r, [0.25, 0.25])
    values[r.actions[1][0]] = None
    totals = MismatchTotals()
    totals.add([r], action_scores([r], [values]), version=0)
    gauges = totals.gauges()
    assert gauges["mismatch/nonfinite_tokens"] == 1
    assert gauges[PREFIX] == pytest.approx(0.25)


def test_adding_in_slices_pauses_after_each_action_and_adds_the_same():
    rows = [row([0, 1]), row([1, 1, 2])]
    scores = action_scores(rows, [scored(r, [0.1] * len(r.policies)) for r in rows])
    whole_totals, sliced = MismatchTotals(), MismatchTotals()
    whole_totals.add(rows, scores, version=2)
    work = sliced.add_in_slices(rows, scores, version=2)
    pauses = sum(1 for _ in work)
    assert pauses == 5
    merged = MismatchTotals()
    merged.merge(sliced)
    assert merged.gauges() == whole_totals.gauges()


def test_action_scores_keep_only_the_action_positions():
    r = row([0, 0])
    assert action_scores([r], [list(range(r.datum.model_input.length))]) == [[[1, 2], [4, 5]]]


def test_a_row_needs_one_policy_per_action():
    r = row([0, 0])
    with pytest.raises(ValueError, match="one policy version per action"):
        TrainingRow(r.datum, r.actions, (0,))
