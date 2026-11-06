import asyncio
import sys
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from support.builders import make_rollout, make_sample

from loops_cookbook.metrics import WANDB_GROUP_LIMIT, Metrics, wandb_group
from loops_cookbook.rl.metrics import rollout_metrics, trajectory_metrics


def test_trajectory_metrics_preserve_token_weighting_and_trainability():
    first = make_sample("a", (1,), (2, 3), (-0.2, -0.4), 0, stop_reason="stop")
    second = make_sample("b", (1, 2, 3, 4), (5,), (-0.6,), 1, stop_reason="length")
    second = replace(second, max_tokens=1, clamped_from=8)  # cut by the budget
    first = replace(first, max_tokens=2, clamped_from=8)  # lowered, but it ended first
    untrained = make_sample("c", (9,), (8,), (-10.0,), 1, trainable=False)
    metrics = trajectory_metrics([make_rollout("t", "g", "r", (first, second, untrained))])
    assert metrics["trajectory/generations"] == 3
    assert metrics["trajectory/sampled_tokens"] == 4
    assert metrics["trajectory/trainable_sampled_tokens"] == 3
    assert abs(metrics["sampling/entropy_estimate"] - 0.4) < 1e-12
    assert metrics["sampling/stop/length"] == 1
    assert metrics["sampling/stop/unknown"] == 1
    assert metrics["sampling/cut_at_budget"] == 1


def test_trajectory_metrics_summarize_each_rollouts_length():
    # One rollout's turns grow its context: 2 + 3, then 6 + 4 (its final context, 10).
    turns = (
        make_sample("a", (1, 2), (3, 4, 5), (-0.1,) * 3, 0),
        make_sample("b", (1, 2, 3, 4, 5, 6), (7, 8, 9, 10), (-0.1,) * 4, 0),
    )
    short = (make_sample("c", (1,), (2,), (-0.1,), 0),)
    medium = (make_sample("d", (1, 2, 3), (4, 5), (-0.1,) * 2, 0),)
    metrics = trajectory_metrics(
        [
            make_rollout("t", "g", "r0", turns),
            make_rollout("t", "g", "r1", short),
            make_rollout("t", "g", "r2", medium),
        ]
    )
    assert metrics["trajectory/sampled_tokens_min"] == 1
    assert metrics["trajectory/sampled_tokens_median"] == 2
    assert metrics["trajectory/sampled_tokens_max"] == 7
    assert metrics["trajectory/sequence_tokens_min"] == 2
    assert metrics["trajectory/sequence_tokens_median"] == 5
    assert metrics["trajectory/sequence_tokens_mean"] == 17 / 3
    assert metrics["trajectory/sequence_tokens_max"] == 10


def test_empty_batch_has_no_trajectory_metrics():
    assert trajectory_metrics([]) == {}
    assert rollout_metrics([]) == {}


def _reported(*reports):
    return [make_rollout("t", "g", f"r{i}", metadata=m) for i, m in enumerate(reports)]


def test_agent_reported_numbers_average_and_labels_count():
    metrics = rollout_metrics(
        _reported(
            {"turns": 4, "termination_reason": "completed", "hit_limit": False},
            {"turns": 8, "termination_reason": "completed", "hit_limit": True},
            {"turns": 6, "termination_reason": "truncated", "hit_limit": False},
        )
    )
    assert metrics["rollout/turns"] == 6
    assert metrics["rollout/termination_reason/completed"] == 2
    assert metrics["rollout/termination_reason/truncated"] == 1
    # A flag averages into the rate it describes.
    assert abs(metrics["rollout/hit_limit"] - 1 / 3) < 1e-12


def test_unreportable_metadata_is_left_out_rather_than_coerced():
    metrics = rollout_metrics(
        # A non-finite number cannot reach here at all: rollout metadata is JSON.
        _reported({"fine": 1.5, "nested": {"a": 1}, "seq": [1, 2]})
    )
    assert metrics == {"rollout/fine": 1.5}


@pytest.mark.parametrize("others", [True, False], ids=["beside other keys", "alone"])
def test_a_label_that_never_repeats_is_dropped_because_it_measures_nothing(caplog, others):
    """Every count would be 1, so summarizing it reveals nothing to lose. The regression:
    an early-sorting unique key used to delete every real metric."""
    extra = {"turns": 1, "termination_reason": "completed"} if others else {}
    reports = [{"attempt_id": f"{i:040x}", **extra} for i in range(40)]
    with caplog.at_level("WARNING"):
        metrics = rollout_metrics(_reported(*reports))
    kept = {"rollout/turns": 1.0, "rollout/termination_reason/completed": 40} if others else {}
    assert metrics == {**kept, "rollout/dropped_keys": 1.0}
    assert "attempt_id" in caplog.text


def test_the_total_cap_keeps_what_the_most_rollouts_reported(caplog):
    shared = {"turns": 7}
    reports = [{**shared, f"only_on_{i}": i} for i in range(20)]
    with caplog.at_level("WARNING"):
        metrics = rollout_metrics(_reported(*reports), limit=4)
    assert metrics["rollout/turns"] == 7
    assert len(metrics) == 4 + 1  # the cap, plus dropped_keys
    assert metrics["rollout/dropped_keys"] == len(reports) + 1 - 4


# A real run's case: every rollout reported every key, so the limit, not an
# identifier, dropped the four that sort last.
_PAST_THE_LIMIT = [{f"metric_{i:02d}": float(i) for i in range(36)}] * 3
_AN_IDENTIFIER = [{"attempt_id": f"{i:x}", "turns": i} for i in range(3)]


@pytest.mark.parametrize(
    ("reports", "says", "never_says"),
    [
        pytest.param(
            _PAST_THE_LIMIT,
            ["past [output] rollout_metrics_limit = 32", "rollout/metric_35"],
            ["differs per rollout"],
            id="count limit",
        ),
        pytest.param(
            _AN_IDENTIFIER,
            ["differs per rollout", "attempt_id"],
            ["rollout_metrics_limit"],
            id="identifier",
        ),
    ],
)
def test_a_drop_warning_names_the_rule_that_dropped_the_key(caplog, reports, says, never_says):
    with caplog.at_level("WARNING"):
        rollout_metrics(_reported(*reports))
    for text in says:
        assert text in caplog.text
    for text in never_says:
        assert text not in caplog.text


def test_keys_dropped_by_both_rules_are_reported_under_each(caplog):
    reports = [{"attempt_id": f"{i:x}", "turns": i, "tool_errors": 0} for i in range(3)]
    with caplog.at_level("WARNING"):
        metrics = rollout_metrics(_reported(*reports), limit=1)
    by_rule = [record.getMessage() for record in caplog.records]
    assert len(by_rule) == 2
    identifier, limit = sorted(by_rule, key=lambda message: "limit" in message)
    assert "attempt_id" in identifier and "turns" not in identifier
    assert "rollout/turns" in limit and "attempt_id" not in limit
    assert "rollout_metrics_limit = 1" in limit
    # One count over both rules, as before.
    assert metrics["rollout/dropped_keys"] == 2


def test_a_long_drop_list_says_how_many_it_left_unnamed(caplog):
    with caplog.at_level("WARNING"):
        rollout_metrics(_reported(*_PAST_THE_LIMIT), limit=26)
    assert "rollout/metric_26" in caplog.text
    assert "rollout/metric_31" not in caplog.text
    assert "and 5 more" in caplog.text


def test_metadata_is_summarized_only_for_the_selected_batch():
    """A rollout that reported nothing must not dilute another rollout's average."""
    assert rollout_metrics(_reported({}, {"turns": 10})) == {"rollout/turns": 10}


class FakeWandb:
    """``wandb`` as the mirror sees it: one run per ``init``, recording what it is told.

    ``opened`` keeps each ``init``'s arguments and ``rows`` each ``(values, commit)``
    logged; ``threads`` the thread each ``init`` and ``finish`` ran on. ``on_init`` runs
    first in ``init`` and may raise or block; ``log_error`` makes every ``log`` raise;
    a name in ``rejected`` makes ``define_metric`` raise.
    """

    def __init__(self):
        self.opened, self.rows, self.definitions, self.threads = [], [], [], []
        self.finished = 0
        self.on_init = None
        self.log_error = None
        self.rejected = set()

    def init(self, **kwargs):
        self.threads.append(threading.current_thread())
        if self.on_init is not None:
            self.on_init(**kwargs)
        self.opened.append(kwargs)
        return SimpleNamespace(log=self._log, define_metric=self._define, finish=self._finish)

    def _log(self, values, commit):
        if self.log_error is not None:
            raise self.log_error
        self.rows.append((values, commit))

    def _define(self, name, **kwargs):
        if name in self.rejected:
            raise RuntimeError("rejected by the remote")
        self.definitions.append(((name,), kwargs))

    def _finish(self):
        self.threads.append(threading.current_thread())
        self.finished += 1


@pytest.fixture
def fake_wandb(monkeypatch):
    wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    return wandb


async def test_the_slow_calls_happen_off_the_event_loop(config, fake_wandb):
    """`init` and `finish` take seconds; the loop is orchestrating rollouts meanwhile."""
    metrics = Metrics(config(wandb_project="test"))
    await metrics.start()
    await metrics.close()
    assert len(fake_wandb.threads) == 2
    assert threading.main_thread() not in fake_wandb.threads


async def test_a_run_that_never_opened_is_still_safe_to_log_to_and_close(config, fake_wandb):
    """A W&B outage must cost the run nothing but the mirror."""

    def fail(**kwargs):
        raise RuntimeError("remote unavailable")

    fake_wandb.on_init = fail
    metrics = Metrics(config(wandb_project="test"))
    await metrics.start()
    metrics.log({"reward/mean": 0.5}, 0, {"reward/mean"})
    await metrics.close()


async def test_a_logging_failure_never_reaches_the_run(config, fake_wandb, caplog):
    fake_wandb.log_error = RuntimeError("remote rejected the row")
    metrics = Metrics(config(wandb_project="test"))
    await metrics.start()
    metrics.log({"reward/mean": 0.5}, 0, {"reward/mean"})
    await metrics.close()
    assert "local metrics are intact" in caplog.text


async def test_heartbeat_and_update_are_committed_without_waiting_for_next_update(
    config, fake_wandb
):
    records, definitions = fake_wandb.rows, fake_wandb.definitions
    metrics = Metrics(config(wandb_project="test"))
    await metrics.start()
    try:
        metrics.log({"pipeline/sampled": 4, "time/sampling_queue/total_seconds": 0.2})
        metrics.log(
            {"reward/mean": 0.5, "pipeline/sampled": 6}, step=0, step_scoped={"reward/mean"}
        )
        metrics.log({"pipeline/sampled": 8})
        assert all(commit for _, commit in records)
        assert "train/update" not in records[0][0]
        # The axis is the step index the run artifacts use, not a 1-based count.
        assert records[1][0]["train/update"] == 0
        assert "train/update" not in records[2][0]
        assert (
            ("reward/mean",),
            {"step_metric": "train/update", "step_sync": False},
        ) in definitions
        for name in ("pipeline/sampled", "time/sampling_queue/total_seconds"):
            assert (
                (name,),
                {"step_metric": "telemetry/elapsed_seconds", "step_sync": False},
            ) in definitions
        assert sum(args == ("pipeline/sampled",) for args, _ in definitions) == 1
        assert (("train/update",), {"hidden": True, "step_sync": False}) in definitions
        # `pipeline/sampled` rode along in the stepped record too, but a heartbeat
        # reports it as well, so it keeps the axis it can always be plotted on.
        assert (
            ("pipeline/sampled",),
            {"step_metric": "telemetry/elapsed_seconds", "step_sync": False},
        ) in definitions
    finally:
        await metrics.close()
    rows = len(records)
    assert fake_wandb.finished == 1
    # Nothing reaches a closed mirror.
    metrics.log({"late": 1}, 301, {"late"})
    assert len(records) == rows


@pytest.mark.parametrize(
    ("kwargs", "valid"),
    [
        ({"step": 0, "step_scoped": {"reward/mean"}}, True),
        ({}, True),
        # The mistake worth making impossible: a per-update record that never says
        # which names are per-update would pin them to the time axis for the run.
        ({"step": 0}, False),
        ({"step_scoped": {"reward/mean"}}, False),
    ],
)
def test_a_per_update_record_must_say_which_names_are_per_update(config, kwargs, valid):
    metrics = Metrics(config(wandb_project=None))
    if valid:
        metrics.log({"reward/mean": 0.5}, **kwargs)
    else:
        with pytest.raises(ValueError, match="together"):
            metrics.log({"reward/mean": 0.5}, **kwargs)


def test_batch_size_does_not_decide_which_labels_survive():
    """No threshold, so develop's 8-rollout example keeps what a 64-rollout one keeps."""
    reasons = ["completed", "truncated", "unusable_reply"]
    for size in (6, 8, 12, 64):
        reports = [{"termination_reason": reasons[i % len(reasons)]} for i in range(size)]
        metrics = rollout_metrics(_reported(*reports))
        assert sum(1 for name in metrics if "termination_reason" in name) == len(reasons), size
        assert "rollout/dropped_keys" not in metrics


def test_a_rare_outcome_among_common_ones_is_still_a_label():
    """The rule is per key, so one context_overflow in 64 is kept, not treated as an id."""
    reports = [{"outcome": "completed" if i else "context_overflow"} for i in range(64)]
    metrics = rollout_metrics(_reported(*reports))
    assert metrics["rollout/outcome/context_overflow"] == 1
    assert metrics["rollout/outcome/completed"] == 63


async def test_a_cancelled_start_is_finished_by_the_close_that_follows_it(config, fake_wandb):
    """`wandb.init` runs in a thread and cannot be cancelled; if it opens a run after
    the caller's `finally` has already closed, nothing finishes it and it bills as
    `running` forever. So a cancelled start returns only once the thread has."""
    opened, release = threading.Event(), threading.Event()

    def held(**kwargs):
        opened.set()
        release.wait(5)

    fake_wandb.on_init = held
    metrics = Metrics(config(wandb_project="test"))
    task = asyncio.create_task(metrics.start())
    assert await asyncio.to_thread(opened.wait, 2)
    task.cancel()
    threading.Timer(0.2, release.set).start()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await metrics.close()
    assert fake_wandb.finished == 1


async def test_one_unacceptable_metric_name_does_not_block_every_other_row(config, fake_wandb):
    """A rejected `define_metric` used to abort the send, losing every metric and
    logging a traceback per update for the life of the run."""
    fake_wandb.rejected.add("bad/name")
    metrics = Metrics(config(wandb_project="test"))
    await metrics.start()
    metrics.log({"good/name": 1.0, "bad/name": 2.0})
    metrics.log({"good/name": 3.0})
    await metrics.close()
    assert [values["good/name"] for values, _ in fake_wandb.rows] == [1.0, 3.0]


async def test_a_resumed_run_joins_the_group_of_the_run_it_continues(config, fake_wandb):
    """Each process opens its own W&B run; the cookbook run id, which a resume keeps, groups them."""
    for _ in range(2):
        metrics = Metrics(config(wandb_project="test"), group="spot-1790896196-44bb518f")
        await metrics.start()
        await metrics.close()
    opened = fake_wandb.opened
    assert [kwargs["group"] for kwargs in opened] == ["spot-1790896196-44bb518f"] * 2
    assert all("resume" not in kwargs and "id" not in kwargs for kwargs in opened)


async def test_a_long_run_name_keeps_the_mirror_and_a_unique_stable_group(config, fake_wandb):
    """W&B refuses a group over 128 characters and fails `init` over it, so a long,
    descriptive `run_name` used to train on with no W&B at all."""

    def refuse_long_groups(**kwargs):
        if len(kwargs["group"]) > WANDB_GROUP_LIMIT:
            raise RuntimeError("invalid parameters: 128 limit exceeded for GroupName")

    fake_wandb.on_init = refuse_long_groups
    name = "nemo3super-sft-lr1e-04-linear-wu10-clip1-bs8-ep2-r64-score0.9-floor0.25" * 2
    # A resume reopens the same run id; another run differs only in its random tail.
    run_ids = [f"{name}-1791590400-44bb518f"] * 2 + [f"{name}-1791590400-0f3a9c2e"]
    for run_id in run_ids:
        metrics = Metrics(config(wandb_project="test", run_name=name), group=run_id)
        await metrics.start()
        await metrics.close()
    opened = fake_wandb.opened
    groups = [kwargs["group"] for kwargs in opened]
    assert len(groups) == 3 and all(len(group) == WANDB_GROUP_LIMIT for group in groups)
    assert all(group.startswith(name[:100]) for group in groups)
    assert groups[0] == groups[1] != groups[2]
    # Only the group is cut: W&B takes a display name of a thousand characters.
    assert {kwargs["name"] for kwargs in opened} == {name}


@pytest.mark.parametrize("length", [1, WANDB_GROUP_LIMIT])
def test_a_run_id_wandb_accepts_is_its_own_group(length):
    assert wandb_group("r" * length) == "r" * length
