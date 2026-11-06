"""Optional, best-effort W&B mirror of run metrics; local artifacts remain authoritative."""

import asyncio
import hashlib
import logging
from time import monotonic

from loops_cookbook.cancellation import outlast
from loops_cookbook.threads import in_thread

logger = logging.getLogger(__name__)

# W&B refuses a longer group name, and fails the whole `init` over it.
WANDB_GROUP_LIMIT = 128


def wandb_group(run_id: str) -> str:
    """``run_id`` as a W&B group: itself if W&B accepts it, else its head and a digest.

    A run id is as long as ``run_name`` makes it, so a descriptive name used to cost
    the run its W&B mirror. Cutting it alone would drop the time and random id at its
    tail, which are what keep it unique; the digest of the whole keeps that, and is
    the same on a resume, which must join the group of the run it continues.
    """
    if len(run_id) <= WANDB_GROUP_LIMIT:
        return run_id
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    return f"{run_id[: WANDB_GROUP_LIMIT - len(digest) - 1]}-{digest}"


class Metrics:
    """Mirror run metrics to W&B. Optional and best effort; local artifacts win.

    No queue and no worker of our own. ``log`` is a socket write to W&B's own
    background service -- 145us for a hundred-key row, against a call rate of one
    per optimizer update and one per heartbeat -- so buffering it behind a thread
    only bought a second copy of the batching W&B already does, and a daemon the
    interpreter could kill mid-flush. Only ``init`` and ``finish`` are slow, and
    those happen once each, off the loop.
    """

    def __init__(self, config, *, group: str | None = None):
        self._started = monotonic()
        self._config = config
        # The cookbook run id, which a resume keeps. Each process opens its own W&B
        # run -- resuming the remote one would replay rows onto `_step` -- so the
        # group is what ties a resumed run back to the one it continues.
        self._group = group
        self._run = None
        self._defined: set[str] = set()
        if config.output.wandb_project:
            try:
                import wandb  # noqa: F401  (presence check; `start` imports it again)
            except ImportError as error:
                raise ValueError(
                    "wandb_project is set but wandb is not installed; "
                    "install the 'wandb' extra (uv sync --extra wandb)"
                ) from error

    async def start(self) -> None:
        """Open the remote run, off the event loop -- ``wandb.init`` takes seconds."""
        if not self._config.output.wandb_project:
            return
        import wandb

        settings = self._config.model_dump(mode="json")
        group = None if self._group is None else wandb_group(self._group)
        if group != self._group:
            logger.info("W&B group %s: run id %s cut to W&B's limit", group, self._group)

        def open_run() -> None:
            # Assigned inside the thread, so the handle exists however the await ends.
            self._run = wandb.init(
                project=settings["output"]["wandb_project"],
                name=settings["output"]["run_name"],
                group=group,
                config=settings,
            )

        opening = in_thread(open_run)
        try:
            await asyncio.shield(opening)
        except asyncio.CancelledError:
            # The thread cannot be stopped. Returning before it does would let the
            # caller's `close` run first, and the run it opens later would never finish.
            await outlast(opening)
            raise
        except Exception:
            logger.exception("W&B initialization failed; continuing with local metrics")
            self._run = None
            return
        # Separately: the run is open now, and dropping the handle over a failed
        # axis definition would lose every metric *and* leave it running forever.
        # `log` defines any name it finds missing, so this is only the head start.
        try:
            for axis in ("train/update", "telemetry/elapsed_seconds"):
                self._run.define_metric(axis, hidden=True, step_sync=False)
                self._defined.add(axis)
        except Exception:
            logger.exception("W&B axis setup failed; metrics may plot against time")

    def log(self, values, step=None, step_scoped=None):
        """Record one event; ``step_scoped`` names the values an update alone reports.

        Those get the optimizer-step axis. Everything else is also reported by
        heartbeats, which have no step to plot against, so it takes the time axis --
        and a name keeps one axis for the life of the run, because that is how often
        it can be reported, not how it is spelled.

        Required whenever a step is given, rather than defaulting to empty: a
        forgotten argument would silently pin a per-update metric to the time axis
        for the whole run, and the first definition of a name is the only one.
        """
        if (step is None) != (step_scoped is None):
            raise ValueError("pass step and step_scoped together, or neither")
        if self._run is None:
            return
        record = {**values, "telemetry/elapsed_seconds": monotonic() - self._started}
        if step is not None:
            # The same number the step artifacts use. A 1-based axis would read
            # better on a fresh run and disagree with steps.jsonl,
            # batches/step-N.json and a checkpoint's next_step on every resumed one.
            record["train/update"] = step
        scoped = frozenset(step_scoped or ())
        for name in record.keys() - self._defined:
            # Per name, and marked either way: one name the remote will not accept
            # must not cost every other metric its row, nor be retried every update
            # for the life of the run.
            self._defined.add(name)
            axis = "train/update" if name in scoped else "telemetry/elapsed_seconds"
            try:
                self._run.define_metric(name, step_metric=axis, step_sync=False)
            except Exception:
                logger.exception("W&B rejected the metric %s; it may plot against time", name)
        try:
            # History rows include heartbeats between optimizer updates. Let W&B
            # advance its own row index; commit every event now.
            self._run.log(record, commit=True)
        except Exception:
            logger.exception("W&B logging failed; local metrics are intact")

    async def close(self) -> None:
        """Flush and close the remote run, off the event loop; never raises."""
        run, self._run = self._run, None
        if run is None:
            return
        try:
            await in_thread(run.finish)
        except Exception:
            logger.exception("W&B finalization failed; local metrics are intact")
