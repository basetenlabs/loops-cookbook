"""Telemetry sinks, attached to a run from outside it.

A run produces two kinds of observable data and they are not the same thing.

An optimizer update is an **event**: it carries a step axis, only the run knows
when it happens, and dropping one loses a row that nothing else will reproduce.
So the run pushes it, to the sinks registered on it.

Pipeline occupancy is **state**: no observer needs every transition, and paying
per transition to announce them would put telemetry on the orchestration path.
So each observer pulls ``snapshot()`` on its own timer -- a thousand state changes
between two frames cost nothing extra, and a quiet pipeline costs one call.

Neither sink belongs to ``Run``. The run offers a snapshot and an event, and what
is done with them -- W&B, a terminal region, nothing at all -- is decided here, so
that adding a third sink never touches the training loop again.
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

    The one mechanism every pulled observer uses; they differ only in their sink
    and their period.

    A source says it has nothing to report by returning ``None``, and that is
    skipped. A run has no state while it is still provisioning, or once its
    pipeline has stopped, and observers outlive both -- reporting that would claim
    a stalled pipeline where there is not yet a pipeline at all. An empty reading
    is not the same answer: it is a state, and it gets reported like any other.

    Telemetry must never end a run, so a failing sink is named once and then
    tolerated in silence: an observer that logs its own breakage every tick is
    worse than the gap it leaves.
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
                    "Observer %s failed; the run continues without it.",
                    task.get_name() if task else "?",
                    exc_info=True,
                )


@asynccontextmanager
async def observe(run: Run, preamble: Sequence[str] = ()) -> AsyncIterator[None]:
    """Attach this run's telemetry for the duration of its execution.

    ``async with observe(run): await run.execute()`` -- the composition root for
    anyone who wants the standard sinks. Build a ``Run`` without it to get a silent
    one, or to wire your own against the same two surfaces. ``preamble`` is logged
    first, into the run's log.

    Teardown runs inside out: the pollers stop, then the region comes down, then
    the sink is closed. Closing uploads, and doing that under a live region would
    leave the terminal frozen on a stale line with nothing to say why.
    """
    config = run.config
    progress = run.out / "progress.jsonl"

    def report(metrics: Metrics, gauges: Gauges) -> None:
        """A heartbeat: to its own file, to W&B, and to the log only at DEBUG.

        The file is the point. A hundred gauges is a wall of numbers in a log you
        are reading for events, but it is exactly what you want to tail on its own,
        and it is the only proof of life a run has when there is no terminal to pin
        a region to. `steps.jsonl` is this run on the step axis; this is the same
        run on the time axis, which is the split W&B already makes.
        """
        logger.debug("pipeline %s", gauges)
        metrics.log(gauges)
        try:
            # Made here, not assumed: this is entered before `execute` creates the
            # run directory, so a short interval would otherwise race the mkdir.
            progress.parent.mkdir(parents=True, exist_ok=True)
            with progress.open("a") as handle:
                handle.write(json.dumps({"at": time(), **gauges}, allow_nan=False) + "\n")
        except OSError:
            logger.warning("Could not append to %s; progress is still in W&B", progress)

    if config.output.progress_every_seconds:
        # Relative where it can be: the absolute run path is long enough to wrap
        # onto a second line, which is the one line a reader wants to copy whole.
        try:
            shown = Path(progress).relative_to(Path.cwd())
        except ValueError:
            shown = progress
        logger.info("Progress every %gs, to its own file:", config.output.progress_every_seconds)
        note(f"  tail -f {shown}")
    else:
        logger.info(
            "Progress reporting is off. Optimizer steps can be far apart and "
            "nothing reports in between, so set progress_every_seconds (60 is "
            "a good start) to follow the pipeline while a batch collects."
        )
    async with telemetry(run.out, config, group=run.run_id, preamble=preamble) as metrics:
        detach = run.add_step_sink(
            lambda event: metrics.log(event.gauges, event.step, event.step_scoped)
        )
        try:
            # Whether a terminal is being driven was settled when the application
            # installed its log handler; all that is left here is the opt-out.
            with live_region(config.output.console != "plain") as live:
                tasks = []
                if live is not None:
                    # Sandboxes and the trainer take minutes to come up and the run
                    # has no state until they have, so the poller has nothing to draw
                    # yet. A blank line through the longest quiet stretch of the run
                    # is the very thing the region is for.
                    paint(live, "starting up")
                    tasks.append(
                        asyncio.create_task(
                            poll(
                                # An empty reading rather than none while there
                                # is a notice: the pipeline is already gone by
                                # the time a shutdown is draining, and that is
                                # exactly when the line saying so has to stay up.
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
                    # `gather` rather than awaiting each: awaiting a task that died
                    # re-raises its exception into a successful run, and a suppressor
                    # for CancelledError cannot tell the task's cancellation from
                    # this one's, so a real Ctrl-C would be swallowed here.
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            detach()
