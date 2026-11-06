"""What every recipe's run directory shares: the resume manifest's own rules."""

import importlib
import json
import logging
import os

import pytest

from loops_cookbook.run_record import Manifest, ManifestOrder, finished, latest_run

VALID = {
    "trainer_state_path": "bt://loops:r/state/step-3",
    "next_step": 4,
    "run_id": "r",
    "run_directory": "/runs/r",
}


@pytest.mark.parametrize("next_step", [True, 4.0, "4"], ids=["boolean", "float", "string"])
def test_a_manifest_holding_what_a_hand_edit_would_write_is_refused(tmp_path, next_step):
    """Machine-written, so a count that is not a JSON integer is refused, not coerced."""
    path = tmp_path / "step-3.json"
    path.write_text(json.dumps({**VALID, "next_step": next_step}))
    with pytest.raises(ValueError, match="manifest"):
        Manifest.read(path)


def test_a_manifest_reads_back_what_it_wrote_and_when(tmp_path):
    written = Manifest(**VALID)
    read = Manifest.read(written.write(tmp_path, 3))
    assert read.written_at is not None
    assert read.model_copy(update={"written_at": None}) == written


def test_the_latest_run_is_the_one_its_records_say_trained_last(tmp_path):
    """A copy, a move or a restore resets file times; the times a run records stay.
    A step record longer than one read from the file's end is still found."""

    def run(name, *, steps=None, manifest=None, copied_at):
        out = tmp_path / name
        (out / "checkpoints").mkdir(parents=True)
        written = []
        if steps is not None:
            records = [{"step": 0, "recorded_at": steps - 1, "pad": "x" * 100_000}]
            records.append({"step": 1, "recorded_at": steps, "pad": "x" * 100_000})
            written.append(out / "steps.jsonl")
            written[-1].write_text("".join(json.dumps(record) + "\n" for record in records))
        if manifest is not None:
            written.append(out / "checkpoints" / "step-1.json")
            written[-1].write_text(json.dumps({**VALID, "written_at": manifest}))
        for path in written:
            os.utime(path, (copied_at, copied_at))

    run("a-100-0000000a", steps=300, manifest=300, copied_at=1_000)  # copied last
    run("a-200-0000000b", manifest=400, copied_at=500)
    run("a-300-0000000c", steps=350, copied_at=600)
    assert latest_run(str(tmp_path), "a") == (tmp_path / "a-200-0000000b").resolve()


def test_a_run_recorded_before_runs_kept_times_counts_by_its_files(tmp_path):
    for name, modified in (("a-100-0000000a", 200), ("a-200-0000000b", 100)):
        steps = tmp_path / name / "steps.jsonl"
        steps.parent.mkdir()
        steps.write_text(json.dumps({"step": 0}) + "\n")
        os.utime(steps, (modified, modified))
    assert latest_run(str(tmp_path), "a") == (tmp_path / "a-100-0000000a").resolve()


def test_no_resume_setting_names_no_manifest():
    assert Manifest.resume_from(None) is None


def _run_directory(tmp_path, steps, seqs=None):
    """A run directory holding a manifest for each of ``steps``, written in that order,
    numbered by ``seqs`` (``None``: by a version that wrote no ``seq``)."""
    for step, seq in zip(steps, seqs or [None] * len(steps), strict=True):
        fields = {**VALID, "next_step": step + 1, "run_directory": str(tmp_path)}
        if seq is not None:
            fields["seq"] = seq
        Manifest(**fields).write(tmp_path, step)
    return tmp_path


def test_a_run_directory_resumes_from_its_highest_step_when_none_is_numbered(tmp_path):
    """Manifests from before ``seq``: by step number, not by name, so step-10 is later
    than step-9."""
    run = _run_directory(tmp_path, [2, 10, 9])
    (run / "checkpoints" / "step-latest.json").write_text("not a manifest")
    assert Manifest.resume_from(str(run)).next_step == 11


def test_a_run_directory_resumes_from_its_newest_written_manifest_not_its_highest_step(
    tmp_path,
):
    """A resume from step 3 branched the run: what it wrote is the latest work, though
    the abandoned branch reached a higher step."""
    run = _run_directory(tmp_path, [1, 2, 3, 4, 5, 6, 4, 5], seqs=[0, 1, 2, 3, 4, 5, 6, 7])
    assert Manifest.resume_from(str(run)).next_step == 6


def test_a_numbered_manifest_is_newer_than_any_unnumbered_one(tmp_path):
    """Unnumbered manifests were written by a version before ``seq``, so before any
    numbered one."""
    run = _run_directory(tmp_path, [8, 9], seqs=[None, None])
    _run_directory(tmp_path, [3], seqs=[0])
    assert Manifest.resume_from(str(run)).next_step == 4


def test_a_process_numbers_its_manifests_after_every_one_already_written(tmp_path):
    """Each process starts after the highest ``seq`` on disk, whatever step it resumed
    from, and numbers its own in the order it writes them."""
    run = _run_directory(tmp_path, [1, 2, 9], seqs=[None, 4, 7])
    (run / "checkpoints" / "step-0.json").write_text("{")
    order = ManifestOrder(run)
    assert [order.next(), order.next()] == [8, 9]
    assert ManifestOrder(tmp_path / "fresh").next() == 0


@pytest.mark.parametrize("seq", ['"1"', "-1"], ids=["text", "negative"])
def test_a_manifest_with_an_unusable_seq_is_refused_and_not_numbered_after(tmp_path, seq):
    """A ``seq`` that is not a count could be the newest, so it refuses the directory,
    naming the file; the next write numbers after the readable ones."""
    run = _run_directory(tmp_path, [3, 5], seqs=[0, 1])
    (run / "checkpoints" / "step-2.json").write_text(f'{{"seq": {seq}}}')
    with pytest.raises(ValueError, match="step-2.json"):
        Manifest.resume_from(str(run))
    assert ManifestOrder(run).next() == 2


def test_a_run_directory_resumes_where_it_is_not_where_it_was_recorded(tmp_path, caplog):
    """A run directory moved or copied resumes in the directory given, and says once
    that its manifests name another."""
    run = tmp_path / "copy"
    for step in (1, 4):
        Manifest(**{**VALID, "next_step": step + 1, "run_directory": "/elsewhere/run"}).write(
            run, step
        )
    with caplog.at_level(logging.WARNING):
        resumed = Manifest.resume_from(str(run))
    assert resumed.run_directory == str(run) and resumed.next_step == 5
    [said] = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert "/elsewhere/run" in said and str(run) in said


def test_a_run_directory_given_relative_is_continued_by_its_absolute_path(tmp_path, monkeypatch):
    """Later manifests carry the directory a resume continues in; relative, it would
    point elsewhere when one of them is resumed from another working directory."""
    Manifest(**{**VALID, "run_directory": "/elsewhere/run"}).write(tmp_path / "copy", 1)
    monkeypatch.chdir(tmp_path)
    resumed = Manifest.resume_from("copy")
    assert resumed.run_directory == str(tmp_path.resolve() / "copy")


def test_a_run_directory_resumed_where_it_was_recorded_says_nothing(tmp_path, caplog):
    run = _run_directory(tmp_path, [1, 4])
    with caplog.at_level(logging.WARNING):
        assert Manifest.resume_from(str(run)).run_directory == str(run)
    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []


def test_a_run_directory_with_no_manifest_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no checkpoint manifest"):
        Manifest.resume_from(str(tmp_path))


def test_a_corrupt_latest_manifest_is_refused_not_skipped(tmp_path):
    """Falling back to an older one would retrain steps without saying so."""
    run = _run_directory(tmp_path, [3, 5])
    (run / "checkpoints" / "step-5.json").write_text("{")
    with pytest.raises(ValueError, match="step-5.json"):
        Manifest.resume_from(str(run))


def test_a_named_manifest_resumes_as_named_even_below_the_latest(tmp_path):
    run = _run_directory(tmp_path, [3, 5, 7])
    assert Manifest.resume_from(str(run / "checkpoints" / "step-3.json")).next_step == 4


@pytest.mark.parametrize(
    ("kind", "extra"),
    [
        pytest.param(
            "loops_cookbook.rl.run",
            {"next_group_index": 8, "policy_version": 3, "shuffle_seed": 0, "contract": {}},
            id="rl",
        ),
        pytest.param("loops_cookbook.sft.run", {"digest": "sha256:d", "schedule": {}}, id="sft"),
    ],
)
def test_each_recipe_resumes_from_a_run_directory_as_its_own_manifest(tmp_path, kind, extra):
    recipe = importlib.import_module(kind).Manifest
    for step in (1, 4):
        recipe(**{**VALID, **extra, "next_step": step + 1, "run_directory": str(tmp_path)}).write(
            tmp_path, step
        )
    resumed = recipe.resume_from(str(tmp_path))
    assert type(resumed) is recipe and resumed.next_step == 5


def test_a_resume_at_max_steps_has_no_update_left():
    """``max_steps`` counts the run's updates, those before the resume included."""
    assert not finished(4, 5)
    assert not finished(4, None)
    assert finished(5, 5)
