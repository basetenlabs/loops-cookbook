"""An SFT run: render every conversation first, then step through the epochs on Loops.

Everything that can be wrong with the data is found before a trainer exists. The
whole dataset is rendered and each conversation either becomes supervised sequences
or is refused with its reason, so ``loops check`` and the start of ``loops run``
report the same thing, and a run never discovers a bad row an hour in.

What the run directory holds is listed in docs/reference/sft.md.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import multiprocessing
import os
import pickle
import random
import sys
import traceback
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from functools import partial
from itertools import islice
from pathlib import Path
from statistics import median
from time import monotonic, time
from typing import Any, Literal, Protocol

from pydantic import Field

from loops_cookbook import run_record
from loops_cookbook.cancellation import see_through
from loops_cookbook.llm.factory import create_renderer, describe
from loops_cookbook.llm.rendering import RenderingError
from loops_cookbook.run_record import loops_url, record_failure, record_identity
from loops_cookbook.sft.config import Config, TrainingSettings
from loops_cookbook.sft.data import Row, load_dataset
from loops_cookbook.sft.sequences import (
    Refused,
    Supervised,
    SupervisedSequence,
    TrainingRenderer,
    check_renderer,
    digest,
    supervise,
)
from loops_cookbook.sft.trainer import Score, TrainerPort, Update
from loops_cookbook.threads import in_thread
from loops_cookbook.trainer import ended_under

logger = logging.getLogger(__name__)


class MetricSink(Protocol):
    """Receives every update's gauges; ``Metrics`` is one."""

    def log(
        self,
        values: Mapping[str, float],
        step: int | None = None,
        step_scoped: Iterable[str] | None = None,
    ) -> None: ...


# --- data ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Refusal:
    row: str
    reason: str


@dataclass(frozen=True)
class Split:
    """One split of the data, rendered: its sequences, and what did not become one."""

    name: str
    rows: int
    sequences: tuple[SupervisedSequence, ...]
    per_turn_rows: int
    refusals: tuple[Refusal, ...]
    ignored: dict[str, int]
    """Fields outside the contract, and how many rows carried each; never trained."""

    def summary(self) -> dict[str, Any]:
        lengths = sorted(sequence.length for sequence in self.sequences)
        tokens = sum(lengths)
        trained = sum(sequence.trained_tokens for sequence in self.sequences)
        return {
            "rows": self.rows,
            "sequences": len(self.sequences),
            "rows_split_per_turn": self.per_turn_rows,
            "rows_refused": len(self.refusals),
            "tokens": tokens,
            "trained_tokens": trained,
            "trained_fraction": trained / tokens if tokens else 0.0,
            "sequence_tokens": {
                "median": median(lengths) if lengths else 0,
                "p95": lengths[min(len(lengths) - 1, math.ceil(0.95 * len(lengths)) - 1)]
                if lengths
                else 0,
                "max": lengths[-1] if lengths else 0,
            },
            "refusals": [
                {"row": refusal.row, "reason": refusal.reason} for refusal in self.refusals
            ],
            "ignored_fields": self.ignored,
        }

    def describe(self) -> list[str]:
        if not self.rows:
            return []
        summary = self.summary()
        lengths = summary["sequence_tokens"]
        lines = [
            (
                f"{self.name}: {self.rows} rows -> {len(self.sequences)} sequences, "
                f"{summary['trained_tokens']} of {summary['tokens']} tokens trained "
                f"({summary['trained_fraction']:.1%}); tokens per sequence median "
                f"{lengths['median']:g}, p95 {lengths['p95']}, max {lengths['max']}"
            )
        ]
        if self.per_turn_rows:
            lines.append(
                f"  {self.per_turn_rows} row(s) split into one sequence per trained "
                "turn: the template renders their history differently from how each turn "
                "was generated"
            )
        if self.ignored:
            lines.append(
                "  fields outside the contract, ignored: "
                + ", ".join(f"{path} ({rows} rows)" for path, rows in self.ignored.items())
            )
        if self.refusals:
            lines.append(f"  {len(self.refusals)} row(s) refused and not trained:")
            lines.extend(f"    {refusal.row}: {refusal.reason}" for refusal in self.refusals[:20])
            if len(self.refusals) > 20:
                lines.append(f"    ... and {len(self.refusals) - 20} more, in data.json")
        return lines


SERIAL_SECONDS = 5.0
"""How long a split renders in this process before the rest of its rows go to worker
processes. A dataset that renders in seconds is not worth starting them for; long
agent traces render for hours on one core."""

DEFAULT_WORKERS = 4
"""The most worker processes a split renders in unless ``[data] render_workers`` says.
Each holds its own copy of the renderer and its tokenizer, hundreds of megabytes, and
the machine may be running other work, so the default takes a few cores, not all."""

PROGRESS_SECONDS = 30.0
"""How often, at most, a split still rendering logs how far it has got; it logs as
rows finish, so one slow row holds it up."""


async def render_split(
    name: str,
    rows: Sequence[Row],
    renderer: TrainingRenderer | None,
    *,
    max_seq_len: int,
    workers: int | None,
) -> Split:
    """Every row of a split, supervised or refused, in the rows' order.

    Rows are independent, so where they render does not change what they become: a
    worker renders a row with a copy of ``renderer`` exactly as this process would.
    Should the workers fail (a renderer that pickles here but not in a fresh process,
    such as a class defined in a notebook; a worker killed for memory), the rows they
    had not returned render here. ``workers`` is how many processes; unset, the cores
    this process may use less one, at most ``DEFAULT_WORKERS``.

    Rows render off the event loop, so a SIGTERM or Ctrl-C is seen while they do.
    Cancelled, the split starts no other batch and raises once the rows it started end.
    """
    outcomes: dict[int, Supervised | str] = {}
    progress = _Progress(name, len(rows))
    if workers is None:
        # One core left for this process and whatever else the machine runs.
        workers = min(_usable_cores() - 1, DEFAULT_WORKERS)
    indexed = list(enumerate(rows))
    while len(outcomes) < len(indexed):
        if workers > 1 and progress.seconds > SERIAL_SECONDS:
            if _portable(renderer):
                break
            workers = 1
        outcomes |= await _render_here(indexed, len(outcomes), renderer, max_seq_len, progress)
    rest = indexed[len(outcomes) :]
    if rest:
        workers = min(workers, len(rest))
        logger.info("%s: rendering the other %d rows in %d processes", name, len(rest), workers)
        outcomes |= await _render_in_processes(
            name, rest, renderer, workers=workers, max_seq_len=max_seq_len, progress=progress
        )
        missing = [(index, row) for index, row in rest if index not in outcomes]
        done = 0
        while done < len(missing):
            batch = await _render_here(missing, done, renderer, max_seq_len, progress)
            outcomes |= batch
            done += len(batch)
    sequences: list[SupervisedSequence] = []
    refusals: list[Refusal] = []
    per_turn_rows = 0
    ignored = Counter(path for row in rows for path in row.example.ignored())
    for index, row in enumerate(rows):
        outcome = outcomes[index]
        if isinstance(outcome, str):
            refusals.append(Refusal(row.name, outcome))
            continue
        sequences.extend(outcome.sequences)
        per_turn_rows += outcome.per_turn
    return Split(
        name,
        len(rows),
        tuple(sequences),
        per_turn_rows,
        tuple(refusals),
        dict(sorted(ignored.items())),
    )


BATCH_SECONDS = 0.2
"""How long a thread of this process renders rows before handing them back. Each hand-
back costs about as much as rendering a token row, so rows go in batches; a Ctrl-C or
SIGTERM waits at most this long, or one row, for the batch to end."""


async def _render_here(
    rows: Sequence[tuple[int, Row]],
    start: int,
    renderer: TrainingRenderer | None,
    max_seq_len: int,
    progress: _Progress,
) -> dict[int, Supervised | str]:
    """Rows from ``start`` on, rendered in a thread of this process for about
    ``BATCH_SECONDS`` and at least one row; a cancellation waits for the batch."""
    batch = await see_through(in_thread(_supervise_batch, rows, start, renderer, max_seq_len))
    for _ in batch:
        progress.advance()
    return batch


def _supervise_batch(
    rows: Sequence[tuple[int, Row]], start: int, renderer: TrainingRenderer | None, max_seq_len: int
) -> dict[int, Supervised | str]:
    deadline = monotonic() + BATCH_SECONDS
    batch: dict[int, Supervised | str] = {}
    for position in range(start, len(rows)):
        index, row = rows[position]
        batch[index] = _supervise(row, renderer, max_seq_len)
        if monotonic() >= deadline:
            break
    return batch


async def _render_in_processes(
    name: str,
    rows: Sequence[tuple[int, Row]],
    renderer: TrainingRenderer | None,
    *,
    workers: int,
    max_seq_len: int,
    progress: _Progress,
) -> dict[int, Supervised | str]:
    """What the workers made of ``rows``, by index; a row missing is left to this process.

    Only as many rows as workers are submitted at once. A row waiting in the pool's
    queue would still start after a cancellation; a row not yet submitted never does.
    So a cancellation while the workers start still waits for each to render one row.
    """
    loop = asyncio.get_running_loop()
    # Spawned, not forked: this process may already run threads (the tokenizer's,
    # telemetry's), which a fork would copy mid-flight.
    pool = ProcessPoolExecutor(
        workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_adopt,
        initargs=(renderer,),
    )
    supervise = partial(_supervise_in_worker, max_seq_len=max_seq_len)
    queued = iter(rows)
    pending: dict[asyncio.Future[Supervised | str], int] = {}
    outcomes: dict[int, Supervised | str] = {}
    try:
        for index, row in islice(queued, workers):
            pending[loop.run_in_executor(pool, supervise, row)] = index
        while pending:
            finished, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for future in finished:
                outcomes[pending.pop(future)] = future.result()
                progress.advance()
                for index, row in islice(queued, 1):
                    pending[loop.run_in_executor(pool, supervise, row)] = index
    except BrokenProcessPool as error:
        logger.warning(
            "%s: the rendering processes stopped (%s); rendering the other %d rows in this process",
            name,
            error,
            len(rows) - len(outcomes),
        )
    finally:
        for future in pending:
            # A row a worker is rendering still finishes; shutdown waits for it. A
            # failure already recorded is read here, so it is not reported again.
            if not future.cancel() and not future.cancelled():
                future.exception()
        await see_through(in_thread(pool.shutdown, cancel_futures=True))
    return outcomes


def _supervise(row: Row, renderer: TrainingRenderer | None, max_seq_len: int) -> Supervised | str:
    """What ``row`` became, or why it was refused: the reason alone, since the refusal's
    traceback would keep everything the row rendered alive until the split is done."""
    try:
        return supervise(row, renderer, max_seq_len=max_seq_len)
    except Refused as refusal:
        return str(refusal)


def _usable_cores() -> int:
    """The cores this process may run on: on Linux, only those it is pinned to
    (``taskset``, a container's CPU set). A container's CPU quota is not visible here."""
    if sys.platform == "linux":
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _portable(renderer: TrainingRenderer | None) -> bool:
    """Whether ``renderer`` can be copied into a worker; a custom one may not pickle,
    and its own ``__reduce__`` or ``__getstate__`` may raise anything."""
    try:
        pickle.dumps(renderer)
    except Exception as error:  # noqa: BLE001 - a custom __reduce__ may raise anything
        logger.info("rendering stays in one process: the renderer does not pickle (%s)", error)
        return False
    return True


_worker_renderer: TrainingRenderer | None = None
"""A rendering worker's copy of the split's renderer, set once as the worker starts."""


def _adopt(renderer: TrainingRenderer | None) -> None:
    global _worker_renderer
    _worker_renderer = renderer


def _supervise_in_worker(row: Row, *, max_seq_len: int) -> Supervised | str:
    return _supervise(row, _worker_renderer, max_seq_len)


class _Progress:
    """Logs a split's rendering every ``PROGRESS_SECONDS``, so a long one is not silent."""

    def __init__(self, name: str, rows: int) -> None:
        self.name, self.rows, self.done = name, rows, 0
        self.started = self.logged = monotonic()

    @property
    def seconds(self) -> float:
        return monotonic() - self.started

    def advance(self) -> None:
        self.done += 1
        if monotonic() - self.logged >= PROGRESS_SECONDS and self.done < self.rows:
            self.logged = monotonic()
            logger.info(
                "%s: rendered %d of %d rows in %.0fs", self.name, self.done, self.rows, self.seconds
            )


def build_renderer(config: Config) -> TrainingRenderer:
    renderer = create_renderer(
        config.model.base_model,
        tokenizer_source=config.model.tokenizer_source,
        chat_template_kwargs=config.model.renderer_chat_template_kwargs,
        renderer=config.model.renderer,
    )
    try:
        check_renderer(renderer)
    except (ValueError, RenderingError) as error:
        raise ValueError(f"{describe(renderer)}: {error}") from None
    return renderer  # pyright: ignore[reportReturnType]


# --- schedule --------------------------------------------------------------------------


@dataclass(frozen=True)
class Schedule:
    """Which sequences each update trains, and at what learning rate."""

    sequences: int
    training: TrainingSettings
    shuffle_seed: int

    @classmethod
    def of(cls, sequences: int, config: Config) -> Schedule:
        return cls(sequences, config.training, config.data.shuffle_seed)

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(self.sequences / self.training.batch_size)

    @property
    def total_steps(self) -> int:
        planned = self.training.epochs * self.steps_per_epoch
        max_steps = self.training.step_limit
        return min(planned, max_steps) if max_steps else planned

    def _binding(self) -> tuple[str, ...]:
        """The settings that end the run at ``total_steps``: one, or both when
        ``max_steps`` equals the epochs' updates."""
        max_steps, planned = self.training.step_limit, self.training.epochs * self.steps_per_epoch
        if max_steps is None or max_steps > planned:
            return ("epochs",)
        return ("max_steps",) if max_steps < planned else ("max_steps", "epochs")

    @property
    def bound(self) -> str:
        """The settings that end the run at ``total_steps``, with their values."""
        values = {
            "max_steps": f"max_steps = {self.training.max_steps}",
            "epochs": f"{self.training.epochs} epoch(s)",
        }
        return ", ".join(values[name] for name in self._binding())

    def more(self, kept: str) -> str:
        """How to train past the last update, given ``kept``, the trainer state it left.
        A decaying learning rate reached its end there, and ``total_steps`` is part of
        the contract, so only a new run trains more."""
        if self.training.lr_schedule != "constant":
            return (
                f"The {self.training.lr_schedule} learning rate decayed over all "
                f"{self.total_steps} update(s); to train more, start a new run with "
                f"--training.init_from {kept} (add --training.init_from_optimizer true to "
                "keep the Adam moments)"
            )
        return f"Raise {' and '.join(self._binding())} to train more"

    def batch(self, step: int) -> list[int]:
        """Sequence indices for ``step``: epoch ``e`` is a shuffle seeded ``seed + e``,
        so a resumed run picks up the order where it stopped."""
        epoch, position = divmod(step, self.steps_per_epoch)
        order = list(range(self.sequences))
        random.Random(self.shuffle_seed + epoch).shuffle(order)
        size = self.training.batch_size
        return order[position * size : (position + 1) * size]

    def epoch(self, step: int) -> float:
        """Epochs completed once ``step`` has trained."""
        return (step + 1) / self.steps_per_epoch

    def learning_rate(self, step: int) -> float:
        training = self.training
        peak, warmup = training.learning_rate, training.warmup_steps
        if step < warmup:
            return peak * (step + 1) / warmup
        if training.lr_schedule == "constant":
            return peak
        progress = (step - warmup) / max(1, self.total_steps - warmup)
        if training.lr_schedule == "linear":
            return peak * (1 - progress)
        return peak * 0.5 * (1 + math.cos(math.pi * progress))

    def contract(self) -> dict[str, Any]:
        """What decides each step's batch and learning rate; a resume must keep it, or
        ``next_step`` would mean a different update. The update count is part of it only
        when the learning rate decays towards it."""
        training = self.training
        contract: dict[str, Any] = {
            "batch_size": training.batch_size,
            "shuffle_seed": self.shuffle_seed,
            "learning_rate": training.learning_rate,
            "lr_schedule": training.lr_schedule,
            "warmup_steps": training.warmup_steps,
        }
        if training.lr_schedule != "constant":
            contract["total_steps"] = self.total_steps
        return contract


@dataclass(frozen=True)
class Plan:
    """Everything a run does before it provisions anything; ``check`` prints it."""

    renderer: TrainingRenderer | None
    train: Split
    validation: Split
    digest: str
    """What trains and what is held out, exactly; a resumed run refuses anything else."""
    schedule: Schedule

    def describe(self) -> list[str]:
        training = self.schedule.training
        lines = [
            f"Renderer: {describe(self.renderer)}"
            if self.renderer
            else "Renderer: none (the tokens format is trained as written)"
        ]
        lines += [*self.train.describe(), *self.validation.describe()]
        lines.append(
            f"Schedule: {self.schedule.total_steps} update(s), "
            f"{self.schedule.steps_per_epoch} per epoch of up to {training.batch_size} "
            f"sequences; learning rate {training.learning_rate:g}, {training.lr_schedule}"
            + (f" after {training.warmup_steps} warmup update(s)" if training.warmup_steps else "")
        )
        if not self.validation.sequences:
            lines.append("Validation: none; set [data] validation or validation_fraction")
        return lines


async def renderer_for(config: Config) -> TrainingRenderer | None:
    """The renderer a chat dataset needs; ``None`` for a tokens dataset, which is never
    rendered. Building it may download the tokenizer, so that runs off the event loop."""
    if config.data.format != "chat":
        return None
    return await see_through(in_thread(build_renderer, config))


async def prepare(config: Config, renderer: TrainingRenderer | None = None) -> Plan:
    """``plan`` with the renderer a chat dataset needs, built here when none is given."""
    if renderer is None:
        renderer = await renderer_for(config)
    return await plan(config, renderer)


async def plan(config: Config, renderer: TrainingRenderer | None) -> Plan:
    """Load, validate and render the whole dataset, and schedule it; nothing remote is
    touched. A tokens dataset is never rendered, so it needs no renderer."""
    data = config.data
    dataset = load_dataset(
        data.train,
        format=data.format,
        validation=data.validation,
        validation_fraction=data.validation_fraction,
        seed=data.shuffle_seed,
    )
    budget = config.model.max_seq_len
    workers = data.render_workers
    train = await render_split(
        "train", dataset.train, renderer, max_seq_len=budget, workers=workers
    )
    if not train.sequences:
        raise ValueError("no training row can be trained:\n" + "\n".join(train.describe()))
    validation = await render_split(
        "validation", dataset.validation, renderer, max_seq_len=budget, workers=workers
    )
    return Plan(
        renderer,
        train,
        validation,
        digest(train.sequences, validation.sequences),
        Schedule.of(len(train.sequences), config),
    )


# --- the run ---------------------------------------------------------------------------


Status = Literal["completed", "interrupted", "failed", "loops_run_ended"]


@dataclass(frozen=True)
class Outcome:
    status: Status
    steps: int
    """Updates completed by this invocation, excluding restored ones."""
    reason: str | None = None
    next_step: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "completed"


class Manifest(run_record.Manifest):
    """An SFT run's resume point: also exactly what trains, and the schedule that decides
    each update's batch and learning rate, both of which a resume is held to."""

    method: Literal["sft"] = "sft"
    next_step: int = Field(ge=1)
    digest: str
    schedule: dict[str, Any]


def check_resume(resume: Manifest, data_digest: str, schedule: Schedule) -> None:
    """Hold a resume to what the run trained, so ``next_step`` means the same update:
    ``loops check`` and ``loops run`` refuse the same changes."""
    if resume.digest != data_digest:
        raise ValueError(
            "the data (or its file's name), renderer or max_seq_len changed since the "
            "checkpoint was written; resume trains exactly what the run trained, so "
            "start a new run"
        )
    recorded, now = resume.schedule, schedule.contract()
    changed = sorted(
        name for name in recorded.keys() | now.keys() if recorded.get(name) != now.get(name)
    )
    if changed:
        raise ValueError(
            f"{', '.join(changed)} changed since the checkpoint was written, which "
            "changes which sequences each update trains or at what learning rate; "
            "resume with the run's own"
        )


class Run:
    def __init__(
        self,
        config: Config,
        trainer: TrainerPort,
        renderer: TrainingRenderer | None,
        *,
        resume: Manifest | None = None,
    ) -> None:
        """The composition root (``sft.recipe``) builds the trainer, restoring ``resume``'s
        state when there is one, and hands it in. ``renderer`` is built as the run
        prepares its data when it is ``None`` and the dataset is chat."""
        self.config = config
        self.out = run_record.new_run_directory(config.output.output_dir, config.output.run_name)
        self.run_id = self.out.name
        self._resume = resume
        if resume is not None:
            self.out, self.run_id = Path(resume.run_directory), resume.run_id
        self._manifest_order = run_record.ManifestOrder(self.out)
        self.trainer = trainer
        self._renderer = renderer

    @property
    def loops_url(self) -> str | None:
        return loops_url(self._loops_run_id())

    def _loops_run_id(self) -> str | None:
        return self.trainer.run_id

    def _record(self, name: str) -> str:
        """``name.json``, or for a resume ``name-from-step-N.json``: a resume is another
        invocation in the run's directory, and must not overwrite what the first one
        recorded -- least of all when it is refused."""
        if self._resume is None:
            return f"{name}.json"
        return f"{name}-from-step-{self._resume.next_step}.json"

    async def execute(self, sink: MetricSink | None = None) -> Outcome:
        """Train, reporting every update's gauges to ``sink``. Every SFT gauge is an
        update's, so all of them take the step axis."""
        config = self.config
        self.out.mkdir(parents=True, exist_ok=True)
        step = self._resume.next_step if self._resume else 0
        (self.out / self._record("config")).write_text(config.model_dump_json(indent=2))
        first = step
        # The last update Loops applied, and the last one kept: they differ when a run
        # fails after an update but before keeping it.
        applied, last_kept = step - 1, None
        data_digest: str | None = None
        schedule: Schedule | None = None
        status, reason = "completed", None
        try:
            planned = await prepare(config, self._renderer)
            schedule, data_digest = planned.schedule, planned.digest
            (self.out / self._record("data")).write_text(
                json.dumps(
                    {
                        "digest": planned.digest,
                        "train": planned.train.summary(),
                        "validation": planned.validation.summary(),
                    },
                    indent=2,
                )
            )
            for line in planned.describe():
                logger.info(line)
            if self._resume:
                self._check_resume(data_digest, schedule)
                if run_record.finished(step, schedule.total_steps):
                    # After the resume checks, so `loops run` refuses what `loops check`
                    # refuses; before the trainer, so nothing is provisioned.
                    run_record.report_finished(
                        schedule.total_steps,
                        schedule.bound,
                        schedule.more(self._resume.trainer_state_path),
                    )
                    return Outcome(status, 0, reason, step)
            # Each datum is built as it is sent: held for the whole dataset, they would
            # cost many times the sequences they are built from.
            train, validation = planned.train.sequences, planned.validation.sequences
            await self.trainer.start()
            if run_id := self._loops_run_id():
                logger.info("Loops run %s: %s", run_id, self.loops_url)
                record_identity(self.out, run_id, name=self._record("run"))

            async def evaluate(at: int) -> dict[str, float]:
                if not validation:
                    return {}
                scored: Score = await self.trainer.evaluate(
                    [sequence.datum() for sequence in validation]
                )
                logger.info("step %d: validation nll %.4f", at, scored.nll)
                return {"val/nll": scored.nll, "val/trained_tokens": float(scored.trained_tokens)}

            if step == 0 and validation:
                # Before any update, at step -1: the curve's origin, the base model's fit.
                baseline = await evaluate(-1)
                if sink is not None:
                    sink.log(baseline, -1, baseline.keys())
            every = config.eval.every_steps
            while step < schedule.total_steps:
                batch = [train[index].datum() for index in schedule.batch(step)]
                update = await self.trainer.update(step, batch, schedule.learning_rate(step))
                applied = step
                final = step == schedule.total_steps - 1
                gauges = _gauges(update, schedule.epoch(step))
                if final or (every and not (step + 1) % every):
                    gauges |= await evaluate(step)
                name, state_path = await self.trainer.checkpoint(step, final=final)
                gauges["train/checkpointed"] = float(name is not None or state_path is not None)
                self._record_step(update, gauges, name, state_path)
                if state_path:
                    self._write_manifest(step, state_path, data_digest, schedule)
                    last_kept = step
                if sink is not None:
                    sink.log(gauges, step, gauges.keys())
                step += 1
        except BaseException as error:
            interrupted = isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
            status = "interrupted" if interrupted else "failed"
            reason = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
            self._record_failure(error, status, applied + 1)
            if schedule is not None and applied >= first and last_kept != applied:
                # The last applied update, kept on the way out so a resume continues
                # after it rather than applying it twice.
                path = await self.trainer.checkpoint_state_only(applied)
                if path and data_digest is not None:
                    self._write_manifest(applied, path, data_digest, schedule)
            # Asked last, so a cancellation while Loops answers loses neither the record
            # nor the update kept above. Whatever noticed first, a Loops run that ended
            # under the run is the cause: recorded again, as such.
            if isinstance(error, Exception) and (
                ended := await ended_under(self.trainer.run_ended)
            ):
                status, reason = "loops_run_ended", ended
                self._record_failure(error, status, applied + 1, quiet=True)
                logger.error(
                    "The Loops run had ended under the run (%s): recorded as loops_run_ended",
                    ended,
                )
            if not isinstance(error, Exception):
                raise
        finally:
            await self.trainer.close()
        return Outcome(status, applied + 1 - first, reason, applied + 1)

    def _record_step(
        self, update: Update, gauges: dict[str, float], name: str | None, state_path: str | None
    ) -> None:
        record = {
            "step": update.step,
            "recorded_at": time(),
            **gauges,
            **({"checkpoint": name} if name else {}),
            **({"trainer_state_path": state_path} if state_path else {}),
        }
        with (self.out / "steps.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        validation = f" val {gauges['val/nll']:.4f}" if "val/nll" in gauges else ""
        logger.info(
            "step %d: nll %.4f%s lr %.3g (%d sequences, %d trained tokens, %.1fs)",
            update.step,
            update.score.nll,
            validation,
            update.learning_rate,
            update.sequences,
            update.score.trained_tokens,
            update.seconds,
        )

    def _check_resume(self, data_digest: str, schedule: Schedule) -> None:
        assert self._resume is not None
        check_resume(self._resume, data_digest, schedule)

    def _write_manifest(
        self, step: int, state_path: str, data_digest: str, schedule: Schedule
    ) -> None:
        Manifest(
            trainer_state_path=state_path,
            next_step=step + 1,
            digest=data_digest,
            schedule=schedule.contract(),
            run_id=self.run_id,
            run_directory=str(self.out),
            seq=self._manifest_order.next(),
        ).write(self.out, step)

    def _record_failure(
        self, error: BaseException, status: Status, step: int, *, quiet: bool = False
    ) -> None:
        """Write what ended the run; ``quiet`` rewrites a record already announced,
        under a status found since."""
        record = {
            "status": status,
            "at": time(),
            "next_step": step,
            "error": f"{type(error).__name__}: {error}",
            "traceback": "".join(traceback.format_exception(error)),
        }
        record_failure(self.out, record, name=self._record("failure"))
        if not quiet:
            logger.error("Run %s at step %d: %s", status, step, record["error"])


def _gauges(update: Update, epoch: float) -> dict[str, float]:
    gauges = {
        "train/nll": update.score.nll,
        "train/trained_tokens": float(update.score.trained_tokens),
        "train/tokens": float(update.tokens),
        "train/sequences": float(update.sequences),
        "train/learning_rate": update.learning_rate,
        "train/epoch": epoch,
        "time/update_seconds": update.seconds,
    }
    gauges.update({f"optim/{name}": value for name, value in update.optimizer.items()})
    return gauges
