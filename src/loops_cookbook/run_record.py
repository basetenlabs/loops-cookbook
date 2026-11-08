"""The files that every recipe's run directory shares, which say which Loops run it was, how it ended and where to resume.

`config.json` is what was asked for, and `run.json` is what was created.
`failure.json` is how the run ended, and `checkpoints/step-N.json` is where to pick
up. A manifest is one model, written and read through the same fields. What every
resume needs is here, and each recipe adds what decides its next update.
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
    """Return the Loops run's page on Baseten, or ``None`` before the trainer has an ID."""
    return f"https://app.baseten.co/training/loop/{run_id}" if run_id else None


def warn_still_up(run_id: str, *, serverless: bool) -> None:
    """Warn that the Loops run is still up, and say how to end it.

    A run stays up when ``[compute].keep_run`` keeps it, so that on dedicated capacity
    ``reuse_from_run_id`` can start the next run warm. A run also stays up when it was
    not deactivated in time. A dedicated run bills until it ends. A serverless run has
    no deployment of its own and costs nothing while it waits, but it holds one of the
    workspace's serverless slots.
    """
    held = "holds a serverless slot" if serverless else "may be billing"
    logger.warning(
        "Loops run %s is still up and %s. Deactivate it with "
        "loops deactivate %s, or use the Loops dashboard.",
        run_id,
        held,
        run_id,
    )


def read_run_ids(out: Path) -> list[str]:
    """Return every Loops run ID that run directory ``out`` recorded through ``record_identity``.

    Each ID appears once, in the order the runs started. That order is ``run.json``,
    then each SFT resume's ``run-from-step-N.json``. An RL resume rewrites ``run.json``.
    """
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
    """Return a fresh run directory's path, ``<output_dir>/<name>-<time>-<id>``, without creating it."""
    return (Path(output_dir).expanduser() / f"{name}-{int(time())}-{uuid4().hex[:8]}").resolve()


def last_trained(out: Path) -> float | None:
    """Return when run directory ``out`` last recorded an update or kept trainer state.

    The answer comes from the times the run wrote into its records. Those are the last
    ``steps.jsonl`` record's ``recorded_at`` and each manifest's ``written_at``. A copy,
    a move or a restore keeps them, while it resets file times. A record without such a
    time, written by an older version of the cookbook, counts by its file's
    modification time. The result is ``None`` before the run did either.
    """
    times = []
    steps = out / "steps.jsonl"
    if steps.is_file() and (line := _last_line(steps)):
        times.append(_recorded_time(line, "recorded_at", steps))
    for path in manifests(out).values():
        times.append(_recorded_time(path.read_bytes(), "written_at", path))
    return max(times, default=None)


def _last_line(path: Path, chunk: int = 1 << 16) -> bytes:
    """Return the last line of ``path``, read from its end.

    ``steps.jsonl`` grows by one record per update, for as many updates as the run
    makes, so reading from the start would get slower as the run goes on.
    """
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
    """Return ``record``'s ``key`` time, or ``path``'s modification time when it has none."""
    try:
        value = json.loads(record).get(key)
    except (ValueError, AttributeError):
        value = None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return path.stat().st_mtime


def latest_run(output_dir: str, name: str) -> Path | None:
    """Return the run directory for ``name`` that trained most recently, or ``None`` when none has.

    The directories are those that ``new_run_directory`` made. A resumed run counts
    from its last update, not from when it started. Only ``<name>-<time>-<id>``
    matches, so run name ``rl`` does not take a directory of run name ``rl-sweep``.
    """
    pattern = re.compile(rf"{re.escape(name)}-[0-9]+-[0-9a-f]{{8}}")
    found = [
        (trained, path.name, path)
        for path in Path(output_dir).expanduser().glob(f"{glob.escape(name)}-*")
        if pattern.fullmatch(path.name)
        and path.is_dir()
        and (trained := last_trained(path)) is not None
    ]
    # Log every run passed over and why, so that a wrong pick can be explained afterwards.
    for trained, run, _ in sorted(found, reverse=True):
        logger.debug("Run %s last trained at %.3f", run, trained)
    return max(found)[2].resolve() if found else None


def finished(next_step: int, end: int | None) -> bool:
    """Return whether a resume at ``next_step`` has no update left before ``end``.

    ``max_steps`` counts the run's updates including the ones before the resume, as in
    Tinker.
    """
    return end is not None and next_step >= end


def finished_message(end: int, bound: str, more: str) -> str:
    """Return what ``loops run`` logs and ``loops check`` prints for a resume with nothing to train.

    The message names the ``end`` the run reached, the ``bound`` setting that put it
    there, and ``more``, which says how to train past it.
    """
    return f"The run already made all {end} update(s) ({bound}); nothing to train. {more}."


def report_finished(end: int, bound: str, more: str) -> None:
    """Log ``finished_message``.

    This is not an error. The same command, run again after the run completed, completes
    again without training.
    """
    logger.info("%s", finished_message(end, bound, more))


def record_identity(out: Path, run_id: str, *, name: str = "run.json") -> None:
    """Write down which Loops run this is.

    This never raises, because the run is already provisioned, and a disk error here
    must not fail it.
    """
    record = {"run_id": run_id, "url": loops_url(run_id), "started_at": time()}
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(json.dumps(record, indent=2))
    except OSError:
        logger.warning("Could not write the run record. The Loops run is %s", run_id)


def record_failure(out: Path, record: dict[str, Any], *, name: str = "failure.json") -> Path | None:
    """Write how the run ended, and return the record's path or ``None``. This never raises."""
    destination = out / name
    try:
        out.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record, indent=2))
    except Exception:
        logger.exception("Could not write the failure record")
        return None
    return destination


def manifest_path(out: Path, step: int) -> Path:
    """Return where step ``step``'s resume manifest lives in run directory ``out``."""
    return out / "checkpoints" / f"step-{step}.json"


_MANIFEST_NAME = re.compile(r"step-([0-9]+)\.json")


def manifests(out: Path) -> dict[int, Path]:
    """Return every resume manifest in run directory ``out``, by step."""
    found = {}
    for path in (out / "checkpoints").glob("step-*.json"):
        if match := _MANIFEST_NAME.fullmatch(path.name):
            found[int(match.group(1))] = path
    return found


def _seq(path: Path) -> int | None:
    """Return the ``seq`` that the manifest at ``path`` was written with.

    The result is ``None`` for a manifest written before manifests had a ``seq``. Raise
    ``ValueError`` when the file cannot say.
    """
    try:
        recorded = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{path} cannot be read as a checkpoint manifest: {error}") from None
    seq = recorded.get("seq") if isinstance(recorded, dict) else None
    if not isinstance(recorded, dict) or not (seq is None or (type(seq) is int and seq >= 0)):
        raise ValueError(f"{path} is not a checkpoint manifest with a readable seq")
    return seq


def newest_manifest(out: Path) -> Path | None:
    """Return the manifest that run directory ``out`` wrote last, or ``None`` when it has none.

    The newest manifest has the highest ``seq``. Manifests from before ``seq`` existed
    are older than any with one, and among themselves the highest step's is newest. A
    manifest whose ``seq`` cannot be read could be the newest, so it makes the choice
    fail rather than being skipped.
    """
    found = manifests(out)
    if not found:
        return None

    def order(step: int) -> tuple[bool, int]:
        seq = _seq(found[step])
        return (True, seq) if seq is not None else (False, step)

    return found[max(found, key=order)]


class ManifestOrder:
    """Numbers one process's manifests in the order it writes them, after every manifest already in its run directory.

    So a resume from an older manifest still writes the newest one, and ``resume_from``
    a run directory takes the work written last.
    """

    def __init__(self, out: Path) -> None:
        self._out = out
        self._next: int | None = None

    def next(self) -> int:
        """Return the ``seq`` of the manifest about to be written.

        The first call reads the run directory. A manifest whose ``seq`` cannot be read
        is passed over, because a resume from the directory refuses it anyway.
        """
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
    """Where to pick a run up.

    A manifest names the trainer state to restore, the next update, and the run
    directory the run continues in. A recipe subclasses it with its own fields.
    """

    # The model is strict. A manifest is machine-written, so `true` or `3.0` where a
    # count belongs is a hand edit or a corruption. It is refused rather than resumed at
    # a coerced step.
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    trainer_state_path: str = Field(min_length=1)
    next_step: int = Field(ge=0)
    run_id: str = Field(min_length=1)
    run_directory: str = Field(min_length=1)
    seq: int | None = Field(default=None, ge=0)
    """The manifest's place in the order the run directory's manifests were written.

    ``ManifestOrder`` assigns it. It is ``None`` in a manifest written before manifests
    had it."""
    written_at: float | None = None
    """When the manifest was written, in seconds since the epoch, for ``last_trained``.

    It is ``None`` in a manifest written before manifests had it."""

    @classmethod
    def resume_from(cls, setting: str | None) -> Self | None:
        """Return the manifest that a ``[training] resume_from`` setting names, or ``None`` when it is unset.

        The setting names a manifest, or a run directory. A run directory resumes from
        the manifest it wrote last, which ``newest_manifest`` finds. That is what a
        retry after a stopped run wants, without knowing where the run stopped. It holds
        even after a resume from an older manifest left a branch with higher steps
        behind. The resume continues in the run directory given, wherever its manifests
        say the run was.
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
        # Make the directory absolute, as a fresh run's is. Later manifests carry it,
        # and one of them may be resumed from another working directory.
        directory = path.resolve()
        if recorded.resolve() == directory:
            return manifest
        logger.warning(
            "Resuming in %s, the run directory given, although its manifest %s names %s",
            directory,
            newest.name,
            recorded,
        )
        return manifest.model_copy(update={"run_directory": str(directory)})

    @classmethod
    def read(cls, path: Path) -> Self:
        """Read the manifest at ``path``, or raise ``ValueError`` saying why it is not one."""
        path = path.expanduser()
        try:
            return cls.model_validate_json(path.read_text())
        except ValidationError as error:
            raise ValueError(
                f"{path} is not a checkpoint manifest this run can resume:\n{error}"
            ) from None

    def write(self, out: Path, step: int) -> Path:
        """Write the manifest as step ``step``'s, whole or not at all, because a resume reads it."""
        target = manifest_path(out, step)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        written = self.model_copy(update={"written_at": time()})
        temporary.write_text(written.model_dump_json(indent=2))
        temporary.replace(target)
        logger.info("Resume manifest: %s", target)
        return target
