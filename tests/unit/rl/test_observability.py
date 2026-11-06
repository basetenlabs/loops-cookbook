"""Telemetry attaches from outside the run, and must never be able to end it."""

import asyncio
import json
import logging
import re
from types import SimpleNamespace

import pytest
from support.fakes import MemoryTrainer

from loops_cookbook.rl.observability import observe, poll
from loops_cookbook.rl.run import Run, StepEvent


def unused(*args, **kwargs):
    raise AssertionError("this run never preflights")


@pytest.fixture
def terminal(monkeypatch):
    """Configure logging the way the CLI does, which is what makes a region possible."""
    from loops_cookbook.console import log_handler

    monkeypatch.setenv("TERM", "xterm-256color")
    stream = Terminal()
    root = logging.getLogger()
    original = list(root.handlers)
    root.handlers = [log_handler(stream=stream)]
    try:
        yield stream
    finally:
        root.handlers = original


class Terminal:
    """A stream that claims a terminal, so the region turns itself on."""

    def __init__(self):
        self.text = ""

    def write(self, chunk):
        self.text += chunk
        return len(chunk)

    def flush(self):
        pass

    def isatty(self):
        return True


@pytest.fixture
def recorder(monkeypatch):
    """What reaches the W&B adapter, which a recording stand-in replaces."""
    events = []

    class RecordingMetrics:
        def __init__(self, config, *, group=None):
            pass

        async def start(self):
            events.append("open")

        def log(self, values, step=None, step_scoped=None):
            events.append(("log", step))

        async def close(self):
            events.append("close")

    monkeypatch.setattr("loops_cookbook.telemetry.Metrics", RecordingMetrics)
    return events


@pytest.fixture
def observed(config):
    """``observed(**settings)``: a run to observe, never executed."""

    def build(**settings):
        return Run(
            config(**settings),
            trainer=MemoryTrainer(),
            runtime_factory=unused,
            renderer_factory=unused,
        )

    return build


@pytest.fixture
def run(observed):
    return observed(progress_every_seconds=0)


async def test_a_poller_hands_each_reading_to_its_sink():
    seen, received = [], asyncio.Event()

    def sink(gauges):
        seen.append(gauges)
        received.set()

    task = asyncio.create_task(poll(lambda: {"a": 1.0}, sink, 0.001))
    try:
        await asyncio.wait_for(received.wait(), timeout=2)
        assert seen[0] == {"a": 1.0}
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_a_source_with_nothing_to_say_is_skipped_but_an_idle_one_is_not():
    """Observers outlive the pipeline at both ends, and silence is not a stalled
    run -- but an empty reading is a state, so it is reported like any other."""
    seen = []
    task = asyncio.create_task(poll(lambda: None, seen.append, 0.001))
    try:
        await asyncio.sleep(0.05)
        assert seen == []
        idle = asyncio.create_task(poll(dict, seen.append, 0.001))
        try:
            async with asyncio.timeout(2):
                while not seen:
                    await asyncio.sleep(0.001)
            assert seen[0] == {}
        finally:
            idle.cancel()
            with pytest.raises(asyncio.CancelledError):
                await idle
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_a_failing_observer_never_ends_the_run_and_names_itself_once(caplog):
    """A sink that keeps breaking must not turn into its own log flood."""
    calls = []

    def sink(_):
        calls.append(1)
        raise RuntimeError("sink is broken")

    task = asyncio.create_task(poll(lambda: {"a": 1.0}, sink, 0.001), name="the-broken-one")
    try:
        async with asyncio.timeout(2):
            while len(calls) < 5:
                await asyncio.sleep(0.001)
        assert not task.done()
        warnings = [r for r in caplog.records if "Observer" in r.message]
        assert len(warnings) == 1
        assert "the-broken-one" in warnings[0].getMessage()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_the_sink_is_open_before_the_run_starts_and_drained_after_it_ends(run, recorder):
    """Opened first so no billable work goes unreported; drained last so none is lost."""
    async with observe(run):
        recorder.append("execute")
    assert recorder == ["open", "execute", "close"]


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_the_sink_is_drained_however_the_run_ends(run, recorder, failure):
    """Cancellation is the path that matters: Ctrl-C still has rows to flush."""
    with pytest.raises(failure):
        async with observe(run):
            raise failure()
    assert recorder == ["open", "close"]


async def test_observing_a_run_leaves_nothing_running_behind_it(observed, recorder):
    """With a poller that exists: one left running would outlive the run it watches."""
    watched = observed(progress_every_seconds=0.001)
    watched._pipeline = SimpleNamespace(snapshot=lambda: {"groups/ready": 1.0})
    before = asyncio.all_tasks()
    async with asyncio.timeout(2), observe(watched):
        while not [event for event in recorder if event[0] == "log"]:
            await asyncio.sleep(0.001)
        assert asyncio.all_tasks() > before, "the test must have a poller to leak"
    assert asyncio.all_tasks() <= before


async def test_an_update_reaches_both_the_standard_sink_and_the_callers_own(run, recorder):
    mine = []
    run.add_step_sink(mine.append)
    async with observe(run):
        run._emit_step(StepEvent(7, {"reward/mean": 1.0}, frozenset({"reward/mean"})))
    assert ("log", 7) in recorder
    # Attaching the standard sinks must not silently detach anyone else's.
    assert [event.step for event in mine] == [7]


async def test_a_sink_attached_after_observing_starts_still_receives_updates(run, recorder):
    """The run is the registry, so nobody has to attach before anybody else."""
    late = []
    async with observe(run):
        run.add_step_sink(late.append)
        run._emit_step(StepEvent(1, {}, frozenset()))
    assert [event.step for event in late] == [1]


async def test_the_standard_sink_is_gone_once_observing_ends(run, recorder):
    async with observe(run):
        pass
    run._emit_step(StepEvent(2, {}, frozenset()))
    assert ("log", 2) not in recorder


async def test_nothing_is_reported_when_progress_reporting_is_off(run, recorder, caplog):
    """The default is silence, and it says how to turn reporting on."""
    caplog.set_level(logging.INFO)
    async with observe(run):
        await asyncio.sleep(0.01)
    assert not [event for event in recorder if event[0] == "log"]
    assert "progress_every_seconds" in caplog.text


async def test_the_heartbeat_goes_to_its_own_file_and_to_wandb_not_to_the_log(
    observed, recorder, caplog, tmp_path
):
    """A hundred gauges is a wall of numbers in a log you are reading for events,
    and exactly what you want to tail on its own."""
    caplog.set_level(logging.INFO)
    run = observed(progress_every_seconds=0.001, output_dir=str(tmp_path))
    # Deliberately not creating run.out first: the heartbeat may fire before
    # `execute` has made it, and the file has to appear anyway.
    run._pipeline = SimpleNamespace(snapshot=lambda: {"groups/ready": 3.0})
    async with asyncio.timeout(2), observe(run):
        while not [event for event in recorder if event[0] == "log"]:
            await asyncio.sleep(0.001)
    assert ("log", None) in recorder, "W&B still gets every heartbeat"
    rows = [json.loads(line) for line in (run.out / "progress.jsonl").read_text().splitlines()]
    assert rows and rows[0]["groups/ready"] == 3.0 and "at" in rows[0]
    # Not in the log at INFO, but the log does say where to find it.
    assert "groups/ready" not in caplog.text
    assert "tail -f" in caplog.text and "progress.jsonl" in caplog.text


async def test_the_progress_region_shows_the_run_state_on_a_terminal(observed, recorder, terminal):
    """The one path a redirected log never takes, so it is driven on a fake tty."""
    run = observed(console="rich", progress_every_seconds=0, groups_per_step=4)
    run._pipeline = SimpleNamespace(
        snapshot=lambda: {"groups/selected": 3.0, "slots/sampling": 7.0}
    )
    async with asyncio.timeout(3), observe(run):
        while "3/4 groups" not in terminal.text:
            await asyncio.sleep(0.05)
        assert "sampling 7" in terminal.text
        # Every stage keeps its place, so the line never jumps as numbers change.
        for stage in ("setup", "sampling", "grading", "queued"):
            assert stage in terminal.text


async def test_a_poller_that_dies_outright_does_not_become_the_runs_outcome(
    observed, recorder, monkeypatch
):
    """`poll` guards its source and sink, but anything escaping that -- a logging
    handler that throws, say -- kills the task, and awaiting a dead task re-raises.
    A watcher's death must not replace the outcome of a run that succeeded."""

    async def dies(source, sink, interval):
        raise RuntimeError("poller died")

    monkeypatch.setattr("loops_cookbook.rl.observability.poll", dies)
    watched = observed(progress_every_seconds=0.001)
    async with observe(watched):
        await asyncio.sleep(0.01)


async def test_the_region_says_something_while_the_run_is_still_provisioning(
    observed, recorder, terminal
):
    """Sandboxes and the trainer take minutes to come up, and the run has no state
    to report until they have. A blank pinned line for the longest quiet window of
    the run is the exact thing the region exists to prevent."""
    watched = observed(console="rich", progress_every_seconds=0)
    assert watched.snapshot() is None, "this test is about the window before any state exists"
    async with observe(watched):
        await asyncio.sleep(0.05)
    # Control codes are not content: rich writes those even for an empty region.
    drawn = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", terminal.text).strip()
    assert drawn, "the region drew nothing at all while starting up"


async def test_console_plain_declines_the_region_even_on_a_terminal(observed, recorder, terminal):
    """The setting is the opt-out; whether a terminal exists was already settled."""
    run = observed(console="plain", progress_every_seconds=0)
    run._pipeline = SimpleNamespace(snapshot=lambda: {"groups/ready": 3.0})
    async with observe(run):
        await asyncio.sleep(0.05)
    # The startup line is painted the moment a region opens, so its absence is
    # deterministic where waiting for a 4 Hz poller tick would not be.
    assert "starting up" not in terminal.text


async def test_the_tail_hint_is_short_enough_to_read_on_one_line(
    observed, recorder, caplog, tmp_path, monkeypatch
):
    """The absolute run path wraps, and this is the one line a reader copies whole."""
    caplog.set_level(logging.INFO)
    monkeypatch.chdir(tmp_path)
    run = observed(progress_every_seconds=60, output_dir=str(tmp_path / "outputs"))
    async with observe(run):
        pass
    (hint,) = [r.getMessage() for r in caplog.records if "tail -f" in r.getMessage()]
    assert str(tmp_path) not in hint, "the hint should be relative to where you are"
    assert hint.endswith("progress.jsonl") and len(hint) < 100


async def test_every_run_leaves_a_log_behind(observed, recorder, tmp_path, caplog):
    """After the fact you want the lines the terminal filtered out, not the ones it
    chose to show, so the file takes everything the run logged."""
    caplog.set_level(logging.INFO)
    run = observed(progress_every_seconds=0, output_dir=str(tmp_path))
    async with observe(run):
        logging.getLogger("loops_cookbook.rl.run").info("step 0: reward 0.5")
        # Third-party chatter is dropped from the terminal but kept on disk.
        logging.getLogger("harbor.trial.trial").info("trial started")
    written = (run.out / "run.log").read_text()
    assert "step 0: reward 0.5" in written
    assert "trial started" in written
    assert "loops_cookbook.rl.run" in written, "the file keeps the logger name"


async def test_the_runs_log_names_where_its_config_came_from(observed, recorder, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    run = observed(progress_every_seconds=0, output_dir=str(tmp_path))
    async with observe(run, ["Config: a -> b", "  .env: none"]):
        pass
    assert "Config: a -> b" in (run.out / "run.log").read_text()


async def test_the_log_handler_is_taken_away_again(observed, recorder, tmp_path):
    """Two runs in one process must not each keep a handler on the root logger."""
    run = observed(progress_every_seconds=0, output_dir=str(tmp_path))
    before = list(logging.getLogger().handlers)
    async with observe(run):
        assert len(logging.getLogger().handlers) == len(before) + 1
    assert logging.getLogger().handlers == before


async def test_a_log_that_cannot_be_opened_does_not_stop_the_run(observed, recorder, tmp_path):
    blocked = tmp_path / "taken"
    blocked.write_text("not a directory")
    run = observed(progress_every_seconds=0, output_dir=str(blocked / "runs"))
    async with observe(run):
        pass


async def test_the_region_pins_a_shutdown_notice_once_ctrl_c_lands(observed, recorder, terminal):
    """Teardown takes minutes and buries itself in Harbor's per-trial output, so
    "this is shutting down" has to stay on screen rather than scroll away."""
    run = observed(console="rich", progress_every_seconds=0)
    async with asyncio.timeout(3), observe(run):
        # The pipeline is already gone by the time a drain is running, so the
        # notice has to survive an empty snapshot.
        assert run.snapshot() is None
        run._notice = "Shutdown requested -- draining"
        while "Shutdown requested" not in terminal.text:
            await asyncio.sleep(0.05)
