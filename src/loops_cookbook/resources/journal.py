"""The run's durable resource journal: append, fsync, and replay, nothing else.

Each record is one line of canonical JSON carrying the schema version, a sequence
number, the event, the resource it concerns, its typed payload, and the digest of
the record before it; the record's own digest covers all of that. Replay checks
order and the digest chain. A record is durable once ``append()`` returns: the
line is fsynced, and so is the directory when the file is new.

A corrupt or torn record is never repaired and never replaced by a fresh journal.
The journal opens in cleanup-only mode instead: every identity replayed before the
damage is kept and nothing more can be appended. What followed the damage is
unknown, so no cleanup of a damaged journal is ever complete, and the file stays
until a person sets it aside. A journal of another schema version does not open.

One writer at a time: the journal holds an exclusive advisory lock for its whole
lifetime, and opening fails while another process holds it. The lock and fsync
ordering are trustworthy only on local filesystems, so the state directory is
checked first.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from loops_cookbook.evidence import JsonValue

SCHEMA_VERSION = 1
JOURNAL = Path("resources") / "v1" / "journal.jsonl"
_GENESIS = "sha256:" + "0" * 64
# Filesystems on which advisory locks and fsync ordering cannot be relied on.
_REJECTED = ("nfs", "nfs4", "cifs", "smb", "smbfs", "smb2", "smb3", "afpfs", "webdav")


class JournalError(RuntimeError):
    """The journal cannot be opened or written as the single, durable writer."""


class UnsupportedJournal(JournalError):
    """The journal was written by another schema version; this one cannot read it."""


class JournalLocked(JournalError):
    """Another process holds the journal: a live run, or a cleanup already running."""


class UnsupportedStateDirectory(JournalError):
    """The state directory is on a filesystem whose locks or fsync cannot be trusted."""


@dataclass(frozen=True, slots=True)
class Record:
    sequence: int
    event: str
    resource_id: str
    payload: Mapping[str, JsonValue]
    recorded_at_ns: int  # wall clock when the record was made durable


def canonical(value: JsonValue) -> bytes:
    """The canonical JSON bytes the journal and specification digests are taken over."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def digest(value: JsonValue) -> str:
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def filesystem_type(path: Path) -> str:
    """The type of the filesystem backing ``path``, as the kernel names it."""
    if platform.system() == "Darwin":
        return _darwin_filesystem_type(path)
    return _linux_filesystem_type(path, Path("/proc/self/mountinfo").read_text())


def _linux_filesystem_type(path: Path, mountinfo: str) -> str:
    """From ``/proc/self/mountinfo``: the longest mount point containing ``path``."""
    target = str(path.resolve())
    best, kind = "", "unknown"
    for line in mountinfo.splitlines():
        fields, _, tail = line.partition(" - ")
        parts = fields.split()
        if len(parts) < 5 or not tail:
            continue
        mount_point = parts[4].replace("\\040", " ")
        inside = target == mount_point or target.startswith(mount_point.rstrip("/") + "/")
        if inside and len(mount_point) >= len(best):
            best, kind = mount_point, tail.split()[0]
    return kind


def _darwin_filesystem_type(path: Path) -> str:
    import ctypes
    import ctypes.util

    class StatFs(ctypes.Structure):
        _fields_ = [
            ("f_bsize", ctypes.c_uint32),
            ("f_iosize", ctypes.c_int32),
            ("f_blocks", ctypes.c_uint64),
            ("f_bfree", ctypes.c_uint64),
            ("f_bavail", ctypes.c_uint64),
            ("f_files", ctypes.c_uint64),
            ("f_ffree", ctypes.c_uint64),
            ("f_fsid", ctypes.c_int32 * 2),
            ("f_owner", ctypes.c_uint32),
            ("f_type", ctypes.c_uint32),
            ("f_flags", ctypes.c_uint32),
            ("f_fssubtype", ctypes.c_uint32),
            ("f_fstypename", ctypes.c_char * 16),
            ("f_mntonname", ctypes.c_char * 1024),
            ("f_mntfromname", ctypes.c_char * 1024),
            ("f_flags_ext", ctypes.c_uint32),
            ("f_reserved", ctypes.c_uint32 * 7),
        ]

    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    result = StatFs()
    if libc.statfs(os.fsencode(str(path)), ctypes.byref(result)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(path))
    return result.f_fstypename.decode()


def check_state_directory(path: Path, *, kind: Callable[[Path], str] = filesystem_type) -> None:
    """Refuse a network or user-space filesystem; accept local ones, overlays included."""
    path.mkdir(parents=True, exist_ok=True)
    name = kind(path).lower()
    if name.startswith("fuse") or any(
        name == bad or name.startswith(bad + ".") for bad in _REJECTED
    ):
        raise UnsupportedStateDirectory(
            f"[resources].state_dir {path} is on {name}; its locks and fsync are not "
            "reliable, so the resource journal needs a local filesystem"
        )


class Journal:
    """One run's resource journal, held open and locked by this process."""

    def __init__(self, path: Path, lock: int, records: list[Record], damage: str | None):
        self.path = path
        self._lock = lock
        self._records = records
        self._damage = damage
        self._last = _GENESIS
        self._sequence = 0

    @classmethod
    def open(cls, state_dir: Path, *, kind: Callable[[Path], str] = filesystem_type) -> Journal:
        """Lock the run's journal and replay it; fails at once if another process holds it."""
        check_state_directory(state_dir, kind=kind)
        path = state_dir / JOURNAL
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = os.open(path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(lock)
            raise JournalLocked(
                f"the resource journal {path} is held by another process; stop it first"
            ) from None
        journal = cls(path, lock, [], None)
        try:
            journal._replay()
        except BaseException:
            journal.close()
            raise
        return journal

    @property
    def records(self) -> tuple[Record, ...]:
        return tuple(self._records)

    @property
    def damage(self) -> str | None:
        """Why the journal is cleanup-only, or ``None`` when it replayed intact."""
        return self._damage

    def _replay(self) -> None:
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        for number, line in enumerate(data.split(b"\n")):
            if not line:
                continue
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    raise TypeError("a record is not a JSON object")
                self._accept(entry)
            except (ValueError, KeyError, TypeError) as error:
                where = (
                    "a torn final record" if not data.endswith(b"\n") else f"record {number + 1}"
                )
                self._damage = f"{where} in {self.path} is unreadable: {error}"
                return
            except UnsupportedJournal:
                raise
            except JournalError as error:
                self._damage = str(error)
                return

    def _accept(self, entry: dict[str, JsonValue]) -> None:
        if entry.get("schema_version") != SCHEMA_VERSION:
            raise UnsupportedJournal(
                f"unsupported resource journal schema {entry.get('schema_version')!r}"
            )
        claimed = entry.pop("digest")
        if entry.get("previous_digest") != self._last or claimed != digest(entry):
            raise JournalError(
                f"the resource journal's digest chain breaks at sequence {entry.get('sequence')}"
            )
        sequence = entry["sequence"]
        if type(sequence) is not int or sequence != self._sequence + 1:
            raise JournalError(f"the resource journal skips from {self._sequence} to {sequence}")
        payload = entry["payload"]
        event, resource_id = entry["event"], entry["resource_id"]
        recorded_at = entry.get("recorded_at_ns")
        if (
            not isinstance(payload, dict)
            or not isinstance(event, str)
            or not isinstance(resource_id, str)
            or type(recorded_at) is not int
        ):
            raise JournalError(f"record {sequence} of the resource journal is malformed")
        self._records.append(Record(sequence, event, resource_id, payload, recorded_at))
        self._sequence, self._last = sequence, str(claimed)

    def append(self, event: str, resource_id: str, payload: Mapping[str, JsonValue]) -> Record:
        """Durably record ``event``; the record exists once this returns."""
        if self._damage is not None:
            raise JournalError(f"the resource journal is cleanup-only: {self._damage}")
        if self._lock < 0:
            raise JournalError(f"the resource journal {self.path} is closed")
        recorded_at = time.time_ns()
        entry: dict[str, JsonValue] = {
            "schema_version": SCHEMA_VERSION,
            "sequence": self._sequence + 1,
            "event": event,
            "resource_id": resource_id,
            "payload": dict(payload),
            "recorded_at_ns": recorded_at,
            "previous_digest": self._last,
        }
        claimed = digest(entry)
        new = not self.path.exists()
        try:
            with self.path.open("ab") as stream:
                stream.write(canonical({**entry, "digest": claimed}) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            if new:
                _fsync_directory(self.path.parent)
        except BaseException as error:
            # The file may now end in a partial record, or hold one whose durability
            # is unknown; nothing may be appended after it, or replay would lose it.
            self._damage = f"an append to {self.path} failed: {type(error).__name__}: {error}"
            raise
        record = Record(self._sequence + 1, event, resource_id, dict(payload), recorded_at)
        self._records.append(record)
        self._sequence, self._last = record.sequence, claimed
        return record

    def close(self) -> None:
        if self._lock >= 0:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            os.close(self._lock)
            self._lock = -1


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
