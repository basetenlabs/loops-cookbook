"""What every run records about itself while it runs: its log, and its metrics.

Each recipe's composition root enters ``telemetry`` around its run; RL's ``observe``
adds its live region and progress pollers inside it. The ordering is the subtle part,
so it lives here once.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from loops_cookbook.metrics import Metrics

logger = logging.getLogger(__name__)


@contextmanager
def transcript(path: Path) -> Iterator[bool]:
    """Record everything the run logs to ``path``, and take the handler away after.

    Added by the recipe rather than by the CLI because only the run knows its own
    directory, and beside its other artifacts is where anyone debugging afterwards
    will look. Adding a handler is not the displacement the old region did --
    nothing of the caller's is touched, and this one is removed on the way out.
    """
    root = logging.getLogger()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(path, encoding="utf-8")
    except OSError:
        yield False
        return
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    try:
        yield True
    finally:
        root.removeHandler(handler)
        handler.close()


@asynccontextmanager
async def telemetry(
    out: Path, config, *, group: str, preamble: Sequence[str] = ()
) -> AsyncIterator[Metrics]:
    """Record the run's log to ``out/run.log`` and mirror its metrics, for the duration.

    ``config`` is the recipe's (W&B reads its ``output`` settings and records the rest);
    ``group`` ties a resumed run's W&B run to the one it continues. ``preamble`` is
    logged first, as where the config came from (``ConfigSource.describe``).
    """
    log = out / "run.log"
    # Always, and unfiltered. The terminal handler drops third-party chatter below
    # WARNING, but that is a reading aid, not a decision about what is worth keeping
    # -- after the fact the buried lines are usually the ones that explain what happened.
    with transcript(log) as recorded:
        if not recorded:
            logger.warning("Could not open %s; this run leaves no log behind", log)
        for line in preamble:
            logger.info("%s", line)
        metrics = Metrics(config, group=group)
        try:
            # Inside the transcript, so its failure is in the run's log, and inside the
            # try: `wandb.init` takes seconds, and a Ctrl-C landing in it would
            # otherwise leave a remote run open that nothing ever finishes.
            await metrics.start()
            yield metrics
        finally:
            await metrics.close()
