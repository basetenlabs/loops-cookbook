"""An optional, best-effort W&B mirror of run metrics. Local artifacts remain authoritative."""

import asyncio
import hashlib
import logging
from time import monotonic

from loops_cookbook.cancellation import outlast
from loops_cookbook.threads import in_thread

logger = logging.getLogger(__name__)

# W&B refuses a longer group name, and it fails the whole `init` over one.
WANDB_GROUP_LIMIT = 128


def wandb_group(run_id: str) -> str:
    """Return ``run_id`` as a W&B group, which is the ID itself if W&B accepts it, or else its head and a digest.

    A run ID is as long as ``run_name`` makes it, so a descriptive name could exceed
    W&B's limit. Cutting the ID alone would drop the time and random ID at its tail,
    which keep it unique. The digest of the whole ID keeps it unique. The digest is also
    the same on a resume, which must join the group of the run it continues.
    """
    if len(run_id) <= WANDB_GROUP_LIMIT:
        return run_id
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    return f"{run_id[: WANDB_GROUP_LIMIT - len(digest) - 1]}-{digest}"


class Metrics:
    """Mirror run metrics to W&B. The mirror is optional and best effort, and local artifacts win.

    There is no queue and no worker thread of our own. ``log`` is a socket write to
    W&B's own background service. It takes about 145 microseconds for a hundred-key
    row, and it is called once per optimizer update and once per heartbeat. Buffering
    it behind a thread would add a second copy of the batching W&B already does, and a
    daemon that the interpreter could kill mid-flush. Only ``init`` and ``finish`` are
    slow, and each happens once, off the event loop.
    """

    def __init__(self, config, *, group: str | None = None):
        self._started = monotonic()
        self._config = config
        # This is the cookbook run ID, which a resume keeps. Each process opens its own
        # W&B run, because resuming the remote one would replay rows onto `_step`. So
        # the group is what ties a resumed run back to the one it continues.
        self._group = group
        self._run = None
        self._defined: set[str] = set()
        if config.output.wandb_project:
            try:
                import wandb  # noqa: F401  (A presence check. `start` imports it again.)
            except ImportError as error:
                raise ValueError(
                    "wandb_project is set but wandb is not installed. "
                    "Install the 'wandb' extra with uv sync --extra wandb"
                ) from error

    async def start(self) -> None:
        """Open the remote run off the event loop, because ``wandb.init`` takes seconds."""
        if not self._config.output.wandb_project:
            return
        import wandb

        settings = self._config.model_dump(mode="json")
        group = None if self._group is None else wandb_group(self._group)
        if group != self._group:
            logger.info("W&B group %s: run ID %s was cut to W&B's limit", group, self._group)

        def open_run() -> None:
            # Assign the run inside the thread, so the handle exists however the await ends.
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
            # The thread cannot be stopped. Returning before it ends would let the
            # caller's `close` run first, and the run that it opens later would never
            # finish.
            await outlast(opening)
            raise
        except Exception:
            logger.exception("W&B initialization failed. Continuing with local metrics")
            self._run = None
            return
        # Define the axes separately. The run is open now, and dropping the handle over
        # a failed axis definition would lose every metric and leave the run open
        # forever. `log` defines any name it finds missing, so this is only a head start.
        try:
            for axis in ("train/update", "telemetry/elapsed_seconds"):
                self._run.define_metric(axis, hidden=True, step_sync=False)
                self._defined.add(axis)
        except Exception:
            logger.exception("W&B axis setup failed, so metrics may plot against time")

    def log(self, values, step=None, step_scoped=None):
        """Record one event. ``step_scoped`` names the values that only an update reports.

        Those values get the optimizer-step axis. Heartbeats also report everything
        else, and heartbeats have no step to plot against, so everything else takes the
        time axis. A name keeps one axis for the life of the run, because the axis
        depends on how often the value can be reported, not on how it is spelled.

        ``step_scoped`` is required whenever a step is given, rather than defaulting to
        empty. A forgotten argument would silently pin a per-update metric to the time
        axis for the whole run, because the first definition of a name is the only one.
        """
        if (step is None) != (step_scoped is None):
            raise ValueError("pass step and step_scoped together, or neither")
        if self._run is None:
            return
        record = {**values, "telemetry/elapsed_seconds": monotonic() - self._started}
        if step is not None:
            # Use the same number that the step artifacts use. A 1-based axis would read
            # better on a fresh run, but it would disagree with steps.jsonl,
            # batches/step-N.json and a checkpoint's next_step on every resumed run.
            record["train/update"] = step
        scoped = frozenset(step_scoped or ())
        for name in record.keys() - self._defined:
            # Define each name separately, and mark it defined either way. One name that
            # the remote refuses must not cost every other metric its row, and it must
            # not be retried every update for the life of the run.
            self._defined.add(name)
            axis = "train/update" if name in scoped else "telemetry/elapsed_seconds"
            try:
                self._run.define_metric(name, step_metric=axis, step_sync=False)
            except Exception:
                logger.exception("W&B rejected the metric %s, so it may plot against time", name)
        try:
            # History rows include heartbeats between optimizer updates. Let W&B advance
            # its own row index, and commit every event now.
            self._run.log(record, commit=True)
        except Exception:
            logger.exception("W&B logging failed. Local metrics are intact")

    async def close(self) -> None:
        """Flush and close the remote run off the event loop. This never raises."""
        run, self._run = self._run, None
        if run is None:
            return
        try:
            await in_thread(run.finish)
        except Exception:
            logger.exception("W&B finalization failed. Local metrics are intact")
