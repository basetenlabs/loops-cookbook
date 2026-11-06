"""The region is optional, shares one console with the log, and never touches theirs."""

import contextlib
import io
import logging
import re
import sys

import pytest
from rich.console import Console

from loops_cookbook.console import (
    STAGES,
    elapsed,
    groups,
    harbor,
    live_region,
    log_handler,
    note,
    paint,
    region,
    stages,
    summarise,
    trainer,
    use_rich,
    workers,
)


class Terminal(io.StringIO):
    def isatty(self):
        return True


@pytest.fixture
def installed(monkeypatch):
    """Install the handler an application would, and take it away afterwards."""
    monkeypatch.setenv("TERM", "xterm-256color")
    terminal = Terminal()
    with logging_to(log_handler(stream=terminal)):
        yield terminal


ANSI = r"\x1b\[[0-9;?]*[a-zA-Z]"


def rendered(terminal):
    return re.sub(ANSI, "", terminal.getvalue())


@contextlib.contextmanager
def logging_to(handler, level=None):
    """The root logger writes only through ``handler`` (at ``level``) until the block ends."""
    root = logging.getLogger()
    original, original_level = list(root.handlers), root.level
    root.handlers = [handler]
    if level is not None:
        root.setLevel(level)
    try:
        yield root
    finally:
        root.handlers = original
        root.setLevel(original_level)


def drawn(*region_args, width=200):
    """The region as a terminal ``width`` columns wide shows it, without escapes."""
    buffer = io.StringIO()
    Console(file=buffer, width=width).print(region(*region_args))
    return re.sub(ANSI, "", buffer.getvalue())


def test_a_redirected_stream_never_gets_control_characters():
    """A captured log is read long after the run; carriage returns ruin it."""
    assert use_rich("auto", stream=io.StringIO()) is False
    # Even asked for explicitly: there is nothing to animate in a file.
    assert use_rich("rich", stream=io.StringIO()) is False
    assert use_rich("plain", stream=Terminal()) is False
    assert use_rich("auto", stream=Terminal()) is True


@pytest.mark.parametrize("is_tty", [False, True])
def test_dumb_terminal_keeps_logs_without_live_redraws(monkeypatch, is_tty):
    monkeypatch.setenv("TERM", "dumb")
    stream = Terminal() if is_tty else io.StringIO()
    with logging_to(log_handler(stream=stream)) as root:
        with live_region(True) as live:
            if not is_tty:
                assert live is None
            paint(live, "progress update")
            if live is not None:
                live.refresh()
            assert "progress update" not in stream.getvalue()
            root.warning("ordinary log output")
            assert "ordinary log output" in stream.getvalue()
        assert "\x1b" not in stream.getvalue()


def test_the_stream_that_decides_is_the_stream_that_is_drawn_on(monkeypatch):
    """`2> run.log` must not fill the log with a region; `> run.log` must keep it."""
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.setattr(sys, "stdout", Terminal())
    assert use_rich("auto") is False
    monkeypatch.setattr(sys, "stderr", Terminal())
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert use_rich("auto") is True


BUSY = {
    "progress/step": 21,
    "pipeline/elapsed_s": 272,
    "groups/selected": 2,
    "progress/groups_needed": 4,
    "harbor/live_rollouts": 100,
    "admission/active_rollouts": 48,
    "runtime/harbor/live_trials": 96,
    "runtime/harbor/trial_cap": 96,
    "queue/graded_rollouts": 17,
    "queue/trainable_rollouts": 12,
    "progress/rollout_slots": 48,
    "progress/verifier_slots": 48,
    "runtime/harbor/preparing_trials": 0,
    "admission/waiting": 44,
    "admission/prestarted": 9,
    "slots/sampling": 39,
    "queue/verification_waiting": 1,
    "slots/verification": 3,
}


def test_busy_is_one_snapshot_a_harbor_run_could_report():
    """The rest of this file reads ``BUSY`` as a real run's numbers, so they must add up.

    `trials` counts what Harbor holds: every stage except `prep` (not accepted yet)
    and the two graded banks (out of Harbor entirely). It is more than the two worker
    caps put together: the gap is queued + to-grade.
    """
    outside = ("prep", "graded", "ready")
    inside = [key for key, label in STAGES if label not in outside]
    live = BUSY["runtime/harbor/live_trials"]
    assert sum(int(BUSY.get(key, 0)) for key in inside) == live
    assert live <= BUSY["runtime/harbor/trial_cap"]
    assert live > BUSY["admission/active_rollouts"] + BUSY["slots/verification"]


def test_the_trial_cap_sits_beside_the_batch_progress():
    """It is the cap that binds when grading is slow, so it belongs on the line you
    read for progress rather than buried among the stages."""
    assert "trials 96/96" in summarise(BUSY)
    # And it is gone from the stages walk, where it duplicated their sum.
    assert "trials" not in stages(BUSY)


def test_the_stages_run_in_the_order_a_rollout_passes_through_them():
    """Harbor prepares, it waits for an agent slot, the sandbox starts, the agent
    runs, it waits for a verifier, the verifier runs, the graded rollout waits for
    its group and then for the optimizer. Any other order invites reading a queue
    as a later stage than it is."""
    assert stages(BUSY) == (
        "prep 0  queued 44  setup 9  sampling 39  to-grade 1  grading 3  graded 17  ready 12"
    )


def test_harbor_and_worker_limits_are_kept_apart():
    """One row invited reading them as a partition: Harbor's cap covers rollouts
    holding no worker permit at all, so it is deliberately larger than the two
    worker caps put together and is not their sum."""
    assert harbor(BUSY) == "trials 96/96"
    assert workers(BUSY) == "rollouts 48/48  verifiers 3/48"
    assert "/" not in stages(BUSY), "the stages walk carries plain counts only"


def test_the_trainer_says_whether_it_is_working_and_what_it_serves():
    """A run is idle here most of the time while collection is busy, and "has it
    started training yet" is the question on a long collection phase."""
    assert trainer({"progress/policy_version": 3}) == "idle  policy v3"
    assert (
        trainer({"progress/step": 7, "progress/training": 1.0, "progress/policy_version": 7})
        == "training step 7  policy v7"
    )
    assert trainer({}) == "idle  policy v0"


def test_every_stage_holds_its_place_even_at_zero():
    """A field that vanishes when empty makes the line jump and hides the answer."""
    line = stages({"runtime/harbor/trial_cap": 96})
    for stage in ("prep", "queued", "setup", "sampling", "to-grade", "grading", "graded", "ready"):
        assert f"{stage} 0" in line, stage


def test_the_clock_advances_only_with_the_snapshot_that_carries_it():
    """Proof of life: a spinner would keep turning with the loop blocked."""
    assert elapsed({"pipeline/elapsed_s": 0}) == "0:00:00"
    assert elapsed({"pipeline/elapsed_s": 3671}) == "1:01:11"
    assert elapsed({}) == "0:00:00"


def test_there_is_no_region_where_the_log_is_not_already_going():
    """Without a console-backed handler a region would be a second writer, and two
    writers on one terminal interleave mid-redraw and corrupt both."""
    with logging_to(logging.StreamHandler(Terminal())), live_region(True) as live:
        assert live is None


def test_the_region_draws_on_the_very_console_the_log_writes_through(installed):
    """Sharing it is structural rather than arranged: the region asks the handler."""
    with live_region(True) as live:
        assert live is not None
        assert live.console is logging.getLogger().handlers[0].console


def test_a_disabled_region_is_a_no_op_even_with_a_console_present(installed):
    with live_region(False) as live:
        assert live is None
    assert paint(live, "ignored") is None


def test_the_pinned_line_reaches_the_terminal(installed):
    with live_region(True) as live:
        paint(live, "starting up")
    assert "starting up" in rendered(installed)


def test_the_region_draws_the_batch_as_a_bar_with_every_stage_beside_it(installed):
    """The bar is what a batch's progress looks like; the stages say why it stalled."""
    with live_region(True) as live:
        paint(
            live,
            region(
                {
                    "progress/step": 21,
                    "groups/selected": 2,
                    "progress/groups_needed": 4,
                    "slots/sampling": 39,
                    "progress/rollout_slots": 48,
                }
            ),
        )
    text = rendered(installed)
    assert "step 21" in text and "2/4 groups" in text
    assert "sampling 39" in text
    for stage in ("queued", "setup", "to-grade", "grading"):
        assert stage in text, stage
    # A bar, not just numbers.
    assert "\u2501" in text or "\u2578" in text or "\u2588" in text


def test_a_log_record_during_the_region_reaches_the_same_terminal(installed):
    """The whole point of one console: the log scrolls above the pinned line."""
    with live_region(True) as live:
        paint(live, "step 3  groups 1/8")
        logging.getLogger("run").error("something happened")
    text = rendered(installed)
    assert "something happened" in text and "step 3  groups 1/8" in text


def test_a_traceback_is_rendered_plainly_rather_than_drawn(installed):
    """Rich draws framed tracebacks at about a millisecond per stack frame, and ours
    unwind through asyncio, Harbor and httpx -- a burst of rollout failures would
    block the event loop for seconds formatting its own account of them."""
    with live_region(True):
        try:
            raise ValueError("boom")
        except ValueError:
            logging.getLogger("run").exception("failed")
    text = rendered(installed)
    assert "Traceback" in text and "ValueError: boom" in text
    # The drawn form frames each stack frame in a panel; the plain one does not.
    assert "╭" not in text, "rich is drawing box frames around tracebacks again"


def test_output_a_program_writes_to_stdout_still_goes_to_stdout(installed, monkeypatch):
    """`myrun > out.txt` must not lose its stdout into the terminal for the run."""
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    with live_region(True):
        print("output that belongs in the file")
    assert "output that belongs in the file" in captured.getvalue()
    assert "output that belongs in the file" not in rendered(installed)


def test_a_second_region_declines_the_terminal_rather_than_fighting_for_it(installed):
    """Two Lives on one console redraw over each other, and rich does not stop it."""
    with live_region(True) as outer:
        assert outer is not None
        with live_region(True) as inner:
            assert inner is None, "a second Live was started on the same terminal"
    # Released again, so a later run in the same process still gets a region.
    with live_region(True) as again:
        assert again is not None


def test_a_region_that_fails_to_start_does_not_poison_the_terminal(installed, monkeypatch):
    """Ownership is claimed before the region is built, so a failure in between
    must hand it back -- otherwise no run in this process ever draws again."""
    import rich.live

    monkeypatch.setattr(rich.live, "Live", lambda *a, **k: 1 / 0)
    with pytest.raises(ZeroDivisionError), live_region(True):
        pass
    monkeypatch.undo()
    with live_region(True) as live:
        assert live is not None, "the terminal was never handed back"


def test_the_application_never_has_its_own_logging_rearranged(installed):
    """A library that displaced handlers had to guess which it was allowed to take."""
    root = logging.getLogger()
    before = list(root.handlers)
    mine = logging.StreamHandler(io.StringIO())
    root.addHandler(mine)
    try:
        with live_region(True):
            assert root.handlers == [*before, mine]
        assert root.handlers == [*before, mine]
    finally:
        root.removeHandler(mine)


def test_a_console_pointed_at_a_file_is_not_a_terminal_to_draw_on():
    """An application may log through rich into a file. Drawing a region there
    writes redraw frames into a log nobody watches, and shows the user nothing."""
    from rich.logging import RichHandler

    written = io.StringIO()
    with logging_to(RichHandler(console=Console(file=written))):
        with live_region(True) as live:
            assert live is None
            paint(live, "step 1  groups 0/8")
        assert "groups" not in written.getvalue()


def test_a_detach_handle_can_be_called_twice():
    """It is handed to callers, so the obvious mistake must not be an exception."""
    from support.builders import make_config
    from support.fakes import MemoryTrainer

    run_cls = __import__("loops_cookbook.rl.run", fromlist=["Run"]).Run
    settings = make_config()

    def unused(*args, **kwargs):
        raise AssertionError("this run never preflights")

    run = run_cls(settings, MemoryTrainer(), runtime_factory=unused, renderer_factory=unused)
    detach = run.add_step_sink(lambda event: None)
    detach()
    detach()


def test_the_region_pins_a_footer_under_the_counts(installed):
    """A run's own page is worth keeping on screen rather than scrolling away."""
    url = "https://app.baseten.co/training/loop/run-1"
    with live_region(True) as live:
        paint(live, region({"progress/step": 1}, url))
    assert url in rendered(installed).replace("\n", "")


def test_the_region_has_no_footer_row_before_there_is_a_link(installed):
    with live_region(True) as live:
        paint(live, region({"progress/step": 1}, None))
    assert "https://" not in rendered(installed)


def test_a_copyable_line_is_never_cropped_to_the_terminal_width(installed):
    """A path is only useful whole, and rich crops to width unless told not to."""
    path = "  tail -f outputs/" + "a-long-run-name" * 8 + "/progress.jsonl"
    note(path)
    assert path.strip() in rendered(installed).replace("\n", "")


def test_a_copyable_line_falls_back_to_the_log_without_a_console():
    """No console means no columns to escape, so the log is the right place."""
    captured = io.StringIO()
    # The level is set explicitly: the fallback logs at INFO, and inheriting whatever
    # level a previous test left behind is how this passed by accident.
    with logging_to(logging.StreamHandler(captured), logging.INFO):
        note("tail -f somewhere/progress.jsonl")
    assert "tail -f somewhere/progress.jsonl" in captured.getvalue()


@pytest.mark.parametrize("notice", [None, "Shutting down"], ids=["plain", "notice"])
def test_the_region_is_always_the_same_number_of_lines(notice):
    """A row that wraps changes the region's height, and `Live` erases the height it
    last drew -- so a frame a line taller or shorter strands the old one or eats a
    real log line. That is the duplicated bars and missing output."""
    heights = {
        width: len(
            drawn(BUSY, "https://example.test/loop/abc", notice, width=width).rstrip().splitlines()
        )
        for width in (120, 100, 80, 60, 40)
    }
    assert len(set(heights.values())) == 1, f"height varies with width: {heights}"


def test_the_group_line_walks_a_group_through_its_life_then_totals_it():
    """Live states first, in the order a group lives them, and the running totals
    last: `trained 1  discarded 9` is the only line that says whether a run is
    converting work into updates at all."""
    line = groups(
        {
            "groups/filling": 12,
            "groups/ready": 2,
            "groups/selected": 2,
            "progress/groups_needed": 2,
            "pipeline/groups_trained": 1,
            "discard/groups/total": 9,
        }
    )
    assert line == "filling 12  backlog 2  batch 2/2  trained 1  discarded 9"
    assert line.index("filling") < line.index("backlog") < line.index("batch")
    assert line.index("batch") < line.index("trained"), "totals come after live states"


def test_every_group_line_field_holds_its_place_at_zero():
    line = groups({})
    for field in ("filling", "backlog", "batch", "trained", "discarded"):
        assert f"{field} 0" in line, field


EPISODES = {key: value for key, value in BUSY.items() if not key.startswith("runtime/harbor/")}


def test_a_run_on_another_runtime_draws_no_harbor_rows():
    """Trials and `prep` are Harbor's; beside an Episode run they are zeros that mean
    nothing and read as a stalled Harbor."""
    text = drawn(EPISODES)
    assert "harbor" not in text and "trials" not in text and "prep" not in text
    assert "trials" not in summarise(EPISODES) and "prep" not in summarise(EPISODES)
    assert "queued 44" in text and "rollouts 48/48" in text


def test_a_harbor_run_draws_its_rows_from_the_first_frame():
    """The cap is reported when the runtime prepares, so the rows are there before
    any trial is, and the region never grows a row mid-run."""
    first = {"runtime/harbor/trial_cap": 96}
    text = drawn(first)
    assert "trials 0/96" in text and "prep 0" in text
    assert "trials 0/96" in summarise(first) and "prep 0" in summarise(first)
