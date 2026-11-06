"""The files every recipe's run directory shares: which Loops run, how it ended, and
where to resume from.

`config.json` is what was asked for, `run.json` is what was created, `failure.json` is
how it ended and `checkpoints/step-N.json` is where to pick up. A manifest is one
model, written and read through the same fields: what every resume needs is here, and
each recipe adds what decides its next update.
"""

from __future__ import annotations

import glob
import json
import logging
import re
from pathlib import Path
from time import time
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


def loops_url(run_id: str | None) -> str | None:
    """The Loops run's page on Baseten, or ``None`` before the trainer has an id."""
    return f"https://app.baseten.co/training/loop/{run_id}" if run_id else None


def warn_still_up(run_id: str, *, serverless: bool) -> None:
    """Say that the Loops run is still up -- kept with ``[compute].keep_run``, so on
    dedicated Loops ``reuse_from_run_id`` can start the next one warm, or not deactivated in time -- and
    how to end it. A dedicated run bills until it ends. A serverless run has no
    deployment of its own and costs nothing while it waits, but it holds one of the
    workspace's serverless slots."""
    held = "holds a serverless slot" if serverless else "may be billing"
    logger.warning(
        "Loops run %s is still up and %s. Deactivate it with: "
        "loops deactivate %s (or use the Loops dashboard).",
        run_id,
        held,
        run_id,
    )


def read_run_ids(out: Path) -> list[str]:
    """Every Loops run ID that run directory ``out`` recorded in ``record_identity``, once
    each, in the order the runs started: ``run.json``, then each SFT resume's
    ``run-from-step-N.json``. An RL resume rewrites ``run.json``."""
    paths = sorted(out.glob("run-from-step-*.json"), key=_resumed_at)
    if (out / "run.json").is_file():
        paths.insert(0, out / "run.json")
    if not paths:
        raise ValueError(f"{out} holds no run record (run.json)")
    run_ids = [_recorded_run_id(path) for path in paths]
    return list(dict.fromkeys(run_ids))


def _resumed_at(path: Path) -> int:
    match = re.fullmatch(r"run-from-step-(\d+)\.json", path.name)
    return int(match.group(1)) if match else -1


def _recorded_run_id(path: Path) -> str:
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: no run record to read ({error})") from error
    run_id = record.get("run_id") if isinstance(record, dict) else None
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{path} names no run_id")
    return run_id


def new_run_directory(output_dir: str, name: str) -> Path:
    """A fresh run directory's path, ``<output_dir>/<name>-<time>-<id>``; not created."""
    return (Path(output_dir).expanduser() / f"{name}-{int(time())}-{uuid4().hex[:8]}").resolve()


def last_trained(out: Path) -> float | None:
    """When run directory ``out`` last recorded an update or kept trainer state, by the
    times the run wrote into its records: the last ``steps.jsonl`` record's
    ``recorded_at`` and each manifest's ``written_at``. A copy, a move or a restore keeps
    them, where it resets file times. A record written before runs kept those times
    counts by its file's modification time. ``None`` before either."""
    times = []
    steps = out / "steps.jsonl"
    if steps.is_file() and (line := _last_line(steps)):
        times.append(_recorded_time(line, "recorded_at", steps))
    for path in manifests(out).values():
        times.append(_recorded_time(path.read_bytes(), "written_at", path))
    return max(times, default=None)


def _last_line(path: Path, chunk: int = 1 << 16) -> bytes:
    """The last line of ``path``, read from its end: ``steps.jsonl`` grows by one
    record per update, for as many updates as the run makes."""
    with path.open("rb") as handle:
        end = handle.seek(0, 2)
        data = b""
        while end > 0:
            start = max(0, end - chunk)
            handle.seek(start)
            data = handle.read(end - start) + data
            end = start
            _, newline, last = data.rstrip(b"\n").rpartition(b"\n")
            if newline or end == 0:
                return last
        return b""


def _recorded_time(record: bytes, key: str, path: Path) -> float:
    """``record``'s ``key`` time, or ``path``'s modification time without one."""
    try:
        value = json.loads(record).get(key)
    except (ValueError, AttributeError):
        value = None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return path.stat().st_mtime


def latest_run(output_dir: str, name: str) -> Path | None:
    """The run directory ``new_run_directory`` made for ``name`` that trained most
    recently, or ``None`` when none has. A resumed run counts from its last update,
    not from when it started. Only ``<name>-<time>-<id>`` matches, so run name ``rl``
    does not take a directory of run name ``rl-sweep``."""
    pattern = re.compile(rf"{re.escape(name)}-[0-9]+-[0-9a-f]{{8}}")
    found = [
        (trained, path.name, path)
        for path in Path(output_dir).expanduser().glob(f"{glob.escape(name)}-*")
        if pattern.fullmatch(path.name)
        and path.is_dir()
        and (trained := last_trained(path)) is not None
    ]
    # So a wrong pick can be explained afterwards: every run passed over, and why.
    for trained, run, _ in sorted(found, reverse=True):
        logger.debug("Run %s last trained at %.3f", run, trained)
    return max(found)[2].resolve() if found else None


def finished(next_step: int, end: int | None) -> bool:
    """Whether a resume at ``next_step`` has no update left before ``end``: ``max_steps``
    counts the run's updates, the ones before the resume included, as in Tinker."""
    return end is not None and next_step >= end


def finished_message(end: int, bound: str, more: str) -> str:
    """What ``loops run`` logs and ``loops check`` prints for a resume with nothing to
    train: the ``end`` it reached, the ``bound`` setting that put it there, and ``more``,
    how to train past it."""
    return f"The run already made all {end} update(s) ({bound}); nothing to train. {more}."


def report_finished(end: int, bound: str, more: str) -> None:
    """Log ``finished_message``. Not an error: the same command, run again after the run
    completed, completes again without training."""
    logger.info("%s", finished_message(end, bound, more))


def record_identity(out: Path, run_id: str, *, name: str = "run.json") -> None:
    """Write down which Loops run this is. Never raises: the run is already provisioned,
    and a disk error here must not fail it."""
    record = {"run_id": run_id, "url": loops_url(run_id), "started_at": time()}
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(json.dumps(record, indent=2))
    except OSError:
        logger.warning("Could not write the run record; the Loops run is %s", run_id)


def record_failure(out: Path, record: dict[str, Any], *, name: str = "failure.json") -> Path | None:
    """Write how the run ended; its path, or ``None``. Never raises."""
    destination = out / name
    try:
        out.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record, indent=2))
    except Exception:
        logger.exception("Could not write the failure record")
        return None
    return destination


def manifest_path(out: Path, step: int) -> Path:
    """Where step ``step``'s resume manifest lives in run directory ``out``."""
    return out / "checkpoints" / f"step-{step}.json"


_MANIFEST_NAME = re.compile(r"step-([0-9]+)\.json")


def manifests(out: Path) -> dict[int, Path]:
    """Every resume manifest in run directory ``out``, by step."""
    found = {}
    for path in (out / "checkpoints").glob("step-*.json"):
        if match := _MANIFEST_NAME.fullmatch(path.name):
            found[int(match.group(1))] = path
    return found


def _seq(path: Path) -> int | None:
    """The ``seq`` the manifest at ``path`` was written with; ``None`` for one written
    before manifests had it. A ``ValueError`` when the file cannot say."""
    try:
        recorded = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{path} cannot be read as a checkpoint manifest: {error}") from None
    seq = recorded.get("seq") if isinstance(recorded, dict) else None
    if not isinstance(recorded, dict) or not (seq is None or (type(seq) is int and seq >= 0)):
        raise ValueError(f"{path} is not a checkpoint manifest with a readable seq")
    return seq


def newest_manifest(out: Path) -> Path | None:
    """The manifest run directory ``out`` wrote last, or ``None`` when it has none.

    The highest ``seq``; manifests from before ``seq`` are older than any with it, and
    among themselves the highest step's is newest. A manifest whose ``seq`` cannot be
    read could be the newest, so it refuses the choice rather than being skipped.
    """
    found = manifests(out)
    if not found:
        return None

    def order(step: int) -> tuple[bool, int]:
        seq = _seq(found[step])
        return (True, seq) if seq is not None else (False, step)

    return found[max(found, key=order)]


class ManifestOrder:
    """Numbers one process's manifests, in the order it writes them, after every one
    already in its run directory: so a resume from an older manifest still writes the
    newest, and ``resume_from`` a run directory takes the work written last."""

    def __init__(self, out: Path) -> None:
        self._out = out
        self._next: int | None = None

    def next(self) -> int:
        """The ``seq`` of the manifest about to be written. The first call reads the
        run directory; one whose ``seq`` cannot be read is passed over, since a resume
        from the directory refuses it anyway."""
        if self._next is None:
            seqs = []
            for path in manifests(self._out).values():
                try:
                    seqs.append(_seq(path))
                except ValueError:
                    continue
            self._next = max((seq for seq in seqs if seq is not None), default=-1) + 1
        seq, self._next = self._next, self._next + 1
        return seq


class Manifest(BaseModel):
    """Where to pick a run up: the trainer state to restore, the next update, and the
    run directory it continues in. A recipe subclasses it with its own fields."""

    # Strict: a manifest is machine-written, so `true` or `3.0` where a count belongs is
    # a hand edit or a corruption, refused rather than resumed at a coerced step.
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    trainer_state_path: str = Field(min_length=1)
    next_step: int = Field(ge=0)
    run_id: str = Field(min_length=1)
    run_directory: str = Field(min_length=1)
    seq: int | None = Field(default=None, ge=0)
    """Its place in the order the run directory's manifests were written (see
    ``ManifestOrder``); ``None`` in one written before manifests had it."""
    written_at: float | None = None
    """When it was written, in seconds since the epoch (``last_trained``); ``None`` in
    one written before manifests had it."""

    @classmethod
    def resume_from(cls, setting: str | None) -> Self | None:
        """The manifest a ``[training] resume_from`` setting names, or ``None`` unset.

        The setting names a manifest, or a run directory, which resumes from the
        manifest it wrote last (``newest_manifest``): what a retry after a stopped run
        wants, without knowing where it stopped, even after a resume from an older
        manifest left a branch with higher steps behind. A run directory given is the
        one the resume continues in, wherever its manifests say the run was.
        """
        if not setting:
            return None
        path = Path(setting).expanduser()
        if not path.is_dir():
            return cls.read(path)
        newest = newest_manifest(path)
        if newest is None:
            raise ValueError(
                f"{path} holds no checkpoint manifest (checkpoints/step-N.json) to resume from"
            )
        manifest = cls.read(newest)
        recorded = Path(manifest.run_directory).expanduser()
        # Absolute, as a fresh run's: later manifests carry it, and one of them may be
        # resumed from another working directory.
        directory = path.resolve()
        if recorded.resolve() == directory:
            return manifest
        logger.warning(
            "Resuming in %s, the run directory given; its manifest %s names %s",
            directory,
            newest.name,
            recorded,
        )
        return manifest.model_copy(update={"run_directory": str(directory)})

    @classmethod
    def read(cls, path: Path) -> Self:
        """The manifest at ``path``, or a ``ValueError`` saying why it is not one."""
        path = path.expanduser()
        try:
            return cls.model_validate_json(path.read_text())
        except ValidationError as error:
            raise ValueError(
                f"{path} is not a checkpoint manifest this run can resume:\n{error}"
            ) from None

    def write(self, out: Path, step: int) -> Path:
        """Write it as step ``step``'s, whole or not at all: a resume reads it."""
        target = manifest_path(out, step)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        written = self.model_copy(update={"written_at": time()})
        temporary.write_text(written.model_dump_json(indent=2))
        temporary.replace(target)
        logger.info("Resume manifest: %s", target)
        return target
