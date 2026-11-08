"""What every run records about itself while it runs, which is its log and its metrics.

Each recipe's composition root enters ``telemetry`` around its run. RL's ``observe``
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
    """Record everything the run logs to ``path``, and remove the handler afterwards.

    The recipe adds this handler rather than the CLI, because only the run knows its
    own directory. Beside the run's other artifacts is where anyone debugging
    afterwards will look. Adding a handler touches none of the caller's handlers, and
    this one is removed on the way out.
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
    """Record the run's log to ``out/run.log`` and mirror its metrics while the context is open.

    ``config`` is the recipe's config. W&B reads its ``output`` settings and records the
    rest. ``group`` ties a resumed run's W&B run to the one it continues. ``preamble``
    is logged first, such as where the config came from, from ``ConfigSource.describe``.
    """
    log = out / "run.log"
    # The run log is always written, and unfiltered. The terminal handler drops
    # third-party chatter below WARNING, but that is a reading aid, not a decision about
    # what is worth keeping. After the fact, the buried lines usually explain what
    # happened.
    with transcript(log) as recorded:
        if not recorded:
            logger.warning("Could not open %s, so this run leaves no log behind", log)
        for line in preamble:
            logger.info("%s", line)
        metrics = Metrics(config, group=group)
        try:
            # This is inside the transcript, so its failure is in the run's log. It is
            # also inside the try. `wandb.init` takes seconds, and a Ctrl-C landing in
            # it would otherwise leave a remote run open that nothing ever finishes.
            await metrics.start()
            yield metrics
        finally:
            await metrics.close()
