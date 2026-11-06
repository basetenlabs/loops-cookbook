"""The resource journal: durable, chained, single-writer, and never silently replaced."""

import json
from pathlib import Path

import pytest

from loops_cookbook.resources.journal import (
    JOURNAL,
    Journal,
    JournalError,
    JournalLocked,
    UnsupportedJournal,
    UnsupportedStateDirectory,
    _linux_filesystem_type,
    canonical,
    check_state_directory,
    digest,
)


def local(path):
    return "apfs"


def opened(tmp_path):
    return Journal.open(tmp_path, kind=local)


def test_records_replay_in_order_in_a_fresh_journal(tmp_path):
    journal = opened(tmp_path)
    journal.append("intent", "r1", {"name": "a"})
    journal.append("created", "r1", {"provider_id": "p1"})
    journal.close()
    replayed = opened(tmp_path)
    assert [(r.sequence, r.event, r.resource_id) for r in replayed.records] == [
        (1, "intent", "r1"),
        (2, "created", "r1"),
    ]
    assert replayed.damage is None
    replayed.append("released", "r1", {})
    replayed.close()


def _three_records(tmp_path):
    journal = opened(tmp_path)
    for name in ("r1", "r2", "r3"):
        journal.append("intent", name, {})
    journal.close()
    return tmp_path / JOURNAL


def _torn_tail(lines):
    lines[-1] = lines[-1][:10]


def _tampered_middle(lines):
    edited = json.loads(lines[1])
    edited["payload"] = {"tampered": True}
    lines[1] = json.dumps(edited).encode()


def _garbage_first(lines):
    lines[0] = b"garbage"


def _not_an_object(lines):
    lines.insert(1, b"[1, 2]")


@pytest.mark.parametrize(
    ("damage", "kept"),
    [
        (_torn_tail, ["r1", "r2"]),
        (_tampered_middle, ["r1"]),
        (_garbage_first, []),
        (_not_an_object, ["r1"]),
    ],
    ids=["torn_tail", "tampered_middle", "garbage_first", "not_an_object"],
)
def test_damage_keeps_what_replayed_takes_no_records_and_never_starts_fresh(tmp_path, damage, kept):
    path = _three_records(tmp_path)
    lines = path.read_bytes().rstrip(b"\n").split(b"\n")
    damage(lines)
    path.write_bytes(b"\n".join(lines) + (b"" if damage is _torn_tail else b"\n"))
    damaged = opened(tmp_path)
    assert [record.resource_id for record in damaged.records] == kept
    assert damaged.damage is not None and path.exists()
    with pytest.raises(JournalError, match="cleanup-only"):
        damaged.append("intent", "r4", {})
    damaged.close()
    opened(tmp_path).close()  # the lock was released


def test_a_journal_of_another_schema_version_does_not_open(tmp_path):
    journal = opened(tmp_path)
    journal.append("intent", "r1", {})
    journal.close()
    path = tmp_path / JOURNAL
    entry = json.loads(path.read_bytes())
    entry.pop("digest")
    entry["schema_version"] = 2
    path.write_bytes(canonical({**entry, "digest": digest(entry)}) + b"\n")
    for _ in range(2):  # refused again: the lock was not leaked
        with pytest.raises(UnsupportedJournal, match="schema"):
            opened(tmp_path)


def test_a_second_writer_fails_while_the_lock_is_held(tmp_path):
    journal = opened(tmp_path)
    with pytest.raises(JournalLocked):
        opened(tmp_path)
    journal.close()
    opened(tmp_path).close()


@pytest.mark.parametrize("kind", ["nfs", "nfs4", "cifs", "smb3", "fuse", "fuse.sshfs", "fuseblk"])
def test_network_and_user_space_filesystems_are_rejected(tmp_path, kind):
    with pytest.raises(UnsupportedStateDirectory, match=kind):
        check_state_directory(tmp_path, kind=lambda p: kind)


@pytest.mark.parametrize("kind", ["ext4", "xfs", "btrfs", "apfs", "overlay", "tmpfs", "zfs"])
def test_local_filesystems_and_container_overlays_are_accepted(tmp_path, kind):
    check_state_directory(tmp_path, kind=lambda p: kind)


def test_the_linux_mount_table_is_read_by_the_longest_containing_mount():
    mountinfo = (
        "22 1 0:21 / / rw - overlay overlay rw\n"
        "30 22 0:40 / /mnt/share rw - nfs4 server:/export rw\n"
        "31 22 0:41 / /mnt/share/local rw - ext4 /dev/sdb rw\n"
    )
    assert _linux_filesystem_type(Path("/mnt/share/run"), mountinfo) == "nfs4"
    assert _linux_filesystem_type(Path("/mnt/share/local/run"), mountinfo) == "ext4"
    assert _linux_filesystem_type(Path("/home/run"), mountinfo) == "overlay"


def test_this_machines_temporary_directory_is_a_supported_state_directory(tmp_path):
    check_state_directory(tmp_path)


def test_a_failed_append_stops_the_journal_so_nothing_lands_after_it(tmp_path, monkeypatch):
    journal = opened(tmp_path)
    journal.append("intent", "a", {})

    def fails(descriptor):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr("loops_cookbook.resources.journal.os.fsync", fails)
    with pytest.raises(OSError):
        journal.append("intent", "b", {})
    monkeypatch.undo()
    assert journal.damage is not None and "append" in journal.damage
    with pytest.raises(JournalError, match="cleanup-only"):
        journal.append("intent", "c", {})
    journal.close()


def test_a_closed_journal_takes_no_records(tmp_path):
    journal = opened(tmp_path)
    journal.close()
    with pytest.raises(JournalError, match="closed"):
        journal.append("intent", "a", {})
