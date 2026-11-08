"""Telemetry sinks, which attach to a run from outside it.

A run produces two kinds of observable data, and they are not the same thing.

An optimizer update is an **event**. It carries a step axis, and only the run knows
when it happens. Dropping one loses a row that nothing else will reproduce, so the run
pushes each event to the sinks registered on it.

Pipeline occupancy is **state**. No observer needs every transition, and announcing
each transition would put telemetry on the orchestration path. So each observer pulls
``snapshot()`` on its own timer. A thousand state changes between two frames cost
nothing extra, and a quiet pipeline costs one call.

Neither sink belongs to ``Run``. The run offers a snapshot and an event. This module
decides what is done with them, such as W&B, a terminal region or nothing at all, so
adding a third sink never touches the training loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

from loops_cookbook.console import REFRESH_PER_SECOND, live_region, note, paint, region
from loops_cookbook.metrics import Metrics
from loops_cookbook.telemetry import telemetry

if TYPE_CHECKING:
    from loops_cookbook.rl.run import Run

logger = logging.getLogger(__name__)

Gauges = dict[str, float]
Snapshot = Callable[[], Gauges | None]
Sink = Callable[[Gauges], None]


async def poll(source: Snapshot, sink: Sink, interval: float) -> None:
    """Hand ``source()`` to ``sink`` every ``interval`` seconds, until cancelled.

    Every pulled observer uses this one mechanism. Observers differ only in their sink
    and their period.

    A source says it has nothing to report by returning ``None``, and that reading is
    skipped. A run has no state while it is still provisioning, or once its pipeline
    has stopped, and observers outlive both. Reporting nothing as a reading would
    claim a stalled pipeline where there is no pipeline yet. An empty reading is a
    different answer. It is a state, and it is reported like any other.

    Telemetry must never end a run, so a failing sink is named once and then tolerated
    in silence. An observer that logs its own breakage every tick is worse than the
    gap it leaves.
    """
    warned = False
    while True:
        await asyncio.sleep(interval)
        try:
            if (reading := source()) is not None:
                sink(reading)
        except Exception:
            if not warned:
                warned = True
                task = asyncio.current_task()
                logger.warning(
                    "Observer %s failed. The run continues without it.",
                    task.get_name() if task else "?",
                    exc_info=True,
                )


@asynccontextmanager
async def observe(run: Run, preamble: Sequence[str] = ()) -> AsyncIterator[None]:
    """Attach this run's telemetry for the duration of its execution.

    Write ``async with observe(run): await run.execute()`` to get the standard sinks.
    Build a ``Run`` without it to get a silent run, or to wire your own sinks against
    the same two surfaces. ``preamble`` is logged first, into the run's log.

    Teardown runs inside out. The pollers stop, then the region comes down, and then
    the sink is closed. Closing uploads, and uploading under a live region would leave
    the terminal frozen on a stale line with nothing to say why.
    """
    config = run.config
    progress = run.out / "progress.jsonl"

    def report(metrics: Metrics, gauges: Gauges) -> None:
        """Write a heartbeat to its own file and to W&B, and to the log only at DEBUG.

        The file matters most. A hundred gauges are a wall of numbers in a log you
        read for events, but they are exactly what you want to tail on their own. The
        file is also the only proof of life a run has when there is no terminal to pin
        a region to. `steps.jsonl` shows this run on the step axis, and this file shows
        it on the time axis, which is the split W&B already makes.
        """
        logger.debug("pipeline %s", gauges)
        metrics.log(gauges)
        try:
            # Create the directory here instead of assuming it. This runs before
            # `execute` creates the run directory, so a short interval would otherwise
            # race the mkdir.
            progress.parent.mkdir(parents=True, exist_ok=True)
            with progress.open("a") as handle:
                handle.write(json.dumps({"at": time(), **gauges}, allow_nan=False) + "\n")
        except OSError:
            logger.warning("Could not append to %s. Progress is still in W&B.", progress)

    if config.output.progress_every_seconds:
        # Show a relative path where possible. The absolute run path is long enough to
        # wrap onto a second line, and this is the one line a reader wants to copy whole.
        try:
            shown = Path(progress).relative_to(Path.cwd())
        except ValueError:
            shown = progress
        logger.info("Progress every %gs, to its own file:", config.output.progress_every_seconds)
        note(f"  tail -f {shown}")
    else:
        logger.info(
            "Progress reporting is off. Optimizer steps can be far apart, and "
            "nothing reports in between. To follow the pipeline while a batch "
            "collects, set [output] progress_every_seconds. Its default is 60."
        )
    async with telemetry(run.out, config, group=run.run_id, preamble=preamble) as metrics:
        detach = run.add_step_sink(
            lambda event: metrics.log(event.gauges, event.step, event.step_scoped)
        )
        try:
            # The application settled whether a terminal is being driven when it
            # installed its log handler. Only the opt-out is left to check here.
            with live_region(config.output.console != "plain") as live:
                tasks = []
                if live is not None:
                    # Sandboxes and the trainer take minutes to come up, and the run
                    # has no state until they have, so the poller has nothing to draw
                    # yet. The region exists to fill that long quiet stretch, so it
                    # shows a line instead of staying blank.
                    paint(live, "starting up")
                    tasks.append(
                        asyncio.create_task(
                            poll(
                                # Return an empty reading, not None, while there
                                # is a notice. The pipeline is already gone while a
                                # shutdown drains, and that is exactly when the line
                                # saying so has to stay up.
                                lambda: run.snapshot() or ({} if run.notice else None),
                                lambda gauges: paint(
                                    live, region(gauges, run.loops_url, run.notice)
                                ),
                                1 / REFRESH_PER_SECOND,
                            ),
                            name="console-region",
                        )
                    )
                if config.output.progress_every_seconds:
                    tasks.append(
                        asyncio.create_task(
                            poll(
                                run.snapshot,
                                lambda gauges: report(metrics, gauges),
                                config.output.progress_every_seconds,
                            ),
                            name="progress-report",
                        )
                    )
                try:
                    yield
                finally:
                    # Use `gather` instead of awaiting each task. Awaiting a task that
                    # died re-raises its exception into a successful run. A suppressor
                    # for CancelledError can't tell the task's cancellation from this
                    # one's, so it would swallow a real Ctrl-C here.
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            detach()
