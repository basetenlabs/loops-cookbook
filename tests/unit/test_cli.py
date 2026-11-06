"""The CLI helpers both recipes share: loading a config and resolving the files it names."""

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel
from support.configs import SFT_MINIMAL, write_toml

from loops_cookbook.cli import exit_code, load_config, run_main


class Files(BaseModel):
    one: str | int | None = None
    many: dict[str, str] = {}


class Settings(BaseModel):
    data: Files = Files()
    sets: dict[str, Files] = {}


REFERENCES = ("data.one", "data.many.*")


@pytest.fixture
def named(tmp_path, monkeypatch):
    """``named(tables)``: a config written in CONFIG, loaded from the shell's directory SHELL,
    and ``where``, which expands CONFIG, SHELL, ROOT and HOME in an expected value."""
    (tmp_path / "config").mkdir()
    (tmp_path / "shell").mkdir()
    monkeypatch.chdir(tmp_path / "shell")
    places = {
        "CONFIG": tmp_path / "config",
        "SHELL": tmp_path / "shell",
        "ROOT": tmp_path,
        "HOME": Path.home(),
    }

    def where(value):
        if not isinstance(value, str):
            return value
        for name, place in places.items():
            value = value.replace(name, str(place))
        return value

    def write(tables):
        return write_toml(tmp_path / "config" / "run.toml", tables), where

    return write


@pytest.mark.parametrize(
    ("tables", "overrides", "one", "many"),
    [
        pytest.param({"data": {"one": "a.jsonl"}}, {}, "CONFIG/a.jsonl", {}, id="from the file"),
        pytest.param({}, {"data.one": "a.jsonl"}, "SHELL/a.jsonl", {}, id="from a flag"),
        pytest.param({"data": {"one": "/abs/a.jsonl"}}, {}, "/abs/a.jsonl", {}, id="absolute"),
        pytest.param(
            {"data": {"many": {"x": "x.txt", "y": "y.txt"}}},
            {"data.many": {"y": "flag-y.txt"}},
            None,
            {"x": "CONFIG/x.txt", "y": "SHELL/flag-y.txt"},
            id="a table mixing file and flag entries",
        ),
        pytest.param(
            {"data": {"many": {"x": "x.txt"}}},
            {"data.many": {"tasks.v1": "flag.txt"}},
            None,
            {"x": "CONFIG/x.txt", "tasks.v1": "SHELL/flag.txt"},
            id="a flag's entry whose name has a dot",
        ),
        pytest.param({}, {}, None, {}, id="no data section"),
        pytest.param({"data": {"one": 3}}, {}, 3, {}, id="a value that is not a path"),
    ],
)
def test_a_file_a_setting_names_resolves_against_whoever_named_it(
    named, tables, overrides, one, many
):
    path, where = named(tables)
    loaded = load_config(Settings, path, overrides, REFERENCES)
    assert loaded.data.one == where(one)
    assert loaded.data.many == {key: where(value) for key, value in many.items()}


@pytest.mark.parametrize(
    ("tables", "overrides", "sets"),
    [
        pytest.param(
            {"sets": {"val": {"many": {"tasks": "val.jsonl"}}}},
            {},
            {"val": "CONFIG/val.jsonl"},
            id="from the file",
        ),
        pytest.param(
            {"sets": {"val": {"many": {"tasks": "val.jsonl"}}}},
            {"sets": {"hard": {"many": {"tasks": "hard.jsonl"}}}},
            {"val": "CONFIG/val.jsonl", "hard": "SHELL/hard.jsonl"},
            id="a set a flag adds beside the file's",
        ),
        pytest.param(
            {"sets": {"val": {"many": {"tasks": "val.jsonl"}}}},
            {"sets.val.many.tasks": "flag.jsonl"},
            {"val": "SHELL/flag.jsonl"},
            id="a flag for one entry",
        ),
        pytest.param(
            {},
            {"sets": {"val.v2": {"many": {"tasks": "v2.jsonl"}}}},
            {"val.v2": "SHELL/v2.jsonl"},
            id="a flag's set whose name has a dot",
        ),
        pytest.param({}, {}, {}, id="no sets"),
    ],
)
def test_each_named_tables_files_resolve_against_whoever_named_them(named, tables, overrides, sets):
    """A ``*`` inside a reference stands for every key there, as each eval set."""
    path, where = named(tables)
    loaded = load_config(Settings, path, overrides, ("sets.*.many.*",))
    assert {name: files.many["tasks"] for name, files in loaded.sets.items()} == {
        name: where(value) for name, value in sets.items()
    }


def test_a_reference_to_no_setting_is_refused(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text("")
    with pytest.raises(ValueError, match="no setting data.nme"):
        load_config(Settings, path, {}, ("data.nme",))


def test_paths_are_resolved_before_validation(tmp_path):
    """So a setting validated as a path sees the absolute one."""
    path = write_toml(tmp_path / "run.toml", {"data": {"one": "a.jsonl"}})
    assert Path(load_config(Settings, path, {}, REFERENCES).data.one).is_absolute()


class Named(BaseModel):
    source: str | None = None


@pytest.mark.parametrize(
    ("tables", "overrides", "source"),
    [
        pytest.param({"source": "./tok"}, {}, "CONFIG/tok", id="a path from the file"),
        pytest.param({}, {"source": "./tok"}, "SHELL/tok", id="a path from a flag"),
        pytest.param({"source": "../tok"}, {}, "ROOT/tok", id="a parent path"),
        pytest.param({"source": "/abs/tok"}, {}, "/abs/tok", id="absolute"),
        pytest.param({"source": "~/tok"}, {}, "HOME/tok", id="home"),
        pytest.param({"source": "org/repo"}, {}, "org/repo", id="a repo id from the file"),
        pytest.param({}, {"source": "org/repo"}, "org/repo", id="a repo id from a flag"),
        pytest.param({"source": "tok"}, {}, "tok", id="a bare name is a repo id"),
        pytest.param({}, {}, None, id="unset"),
    ],
)
def test_a_path_or_name_resolves_only_a_path(named, tables, overrides, source):
    path, where = named(tables)
    assert load_config(Named, path, overrides, (), ("source",)).source == where(source)


def test_a_path_or_name_to_no_setting_is_refused(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text("")
    with pytest.raises(ValueError, match="no setting sorce"):
        load_config(Named, path, {}, (), ("sorce",))


def rl_config():
    from support.builders import config_values

    from loops_cookbook.rl import cli

    return cli.load_training_config, config_values()


def sft_config():
    from loops_cookbook.sft import cli

    return cli.load_config, SFT_MINIMAL


@pytest.mark.parametrize("recipe", [rl_config, sft_config])
@pytest.mark.parametrize(
    ("tokenizer", "loaded"),
    [("./tokenizer", "CONFIG/tokenizer"), ("org/tokenizer-repo", "org/tokenizer-repo")],
)
def test_each_recipe_reads_a_local_tokenizer_from_beside_its_config(
    tmp_path, monkeypatch, recipe, tokenizer, loaded
):
    load, values = recipe()
    (tmp_path / "config" / "tokenizer").mkdir(parents=True)
    path = tmp_path / "config" / "run.json"
    path.write_text(json.dumps({**values, "model": {**values["model"], "tokenizer": tokenizer}}))
    monkeypatch.chdir(tmp_path)
    config = load(path)
    assert config.model.tokenizer == loaded.replace("CONFIG", str(tmp_path / "config"))


STOPPABLE = """
import asyncio, os, signal, sys
from loops_cookbook.cli import Terminated, run_main

# Ctrl-C as at a terminal, even when the test runs where SIGINT is ignored.
signal.signal(signal.SIGINT, signal.default_int_handler)
AGAIN = {again!r}

async def stoppable(name):
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        print(name, "stopping", flush=True)
        if AGAIN == name:
            os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.sleep(0.05)
        print(name, "stopped", flush=True)
        raise

async def main():
    global leftover
    if AGAIN == "leftover":
        # Not awaited by main: the runner cancels it as it shuts down.
        leftover = asyncio.create_task(stoppable("leftover"))
    print("started", flush=True)
    await stoppable("main")

try:
    run_main(main())
except Terminated:
    print("terminated", flush=True)
    sys.exit(143)
except KeyboardInterrupt:
    print("interrupted", flush=True)
    sys.exit(130)
"""


STOPS = {
    "SIGTERM": (signal.SIGTERM, None, 143, ["main stopping", "main stopped", "terminated"]),
    "Ctrl-C": (signal.SIGINT, None, 130, ["main stopping", "main stopped", "interrupted"]),
    # A second SIGTERM does not cut the stop short.
    "SIGTERM twice": (
        signal.SIGTERM,
        "main",
        143,
        ["main stopping", "main stopped", "terminated"],
    ),
    # Nor the runner's shutdown: a task main left running is cancelled, and finishes too.
    "SIGTERM twice, a task left running": (
        signal.SIGTERM,
        "leftover",
        143,
        [
            "main stopping",
            "main stopped",
            "leftover stopping",
            "leftover stopped",
            "terminated",
        ],
    ),
}


@pytest.fixture(scope="module")
def stopped():
    """Each of ``STOPS`` run in its own process, side by side: ``(exit code, lines)``.

    Each process runs ``STOPPABLE`` and is sent its signal once it is waiting; with
    ``again``, the task of that name sends a second SIGTERM as it stops.
    """
    processes = {
        name: subprocess.Popen(
            [sys.executable, "-c", STOPPABLE.format(again=again)],
            stdout=subprocess.PIPE,
            text=True,
        )
        for name, (_, again, _, _) in STOPS.items()
    }
    results = {}
    try:
        for name, process in processes.items():
            assert process.stdout is not None
            assert process.stdout.readline() == "started\n"
            process.send_signal(STOPS[name][0])
        for name, process in processes.items():
            out, _ = process.communicate(timeout=30)
            results[name] = (process.returncode, out.splitlines())
    finally:
        for process in processes.values():
            process.kill()
            process.wait()
    return results


@pytest.mark.parametrize("stop", sorted(STOPS))
def test_a_stop_signal_takes_the_graceful_stop_and_a_second_does_not_cut_it_short(stopped, stop):
    _, _, code, lines = STOPS[stop]
    assert stopped[stop] == (code, lines)


@pytest.mark.parametrize("before", [signal.SIG_DFL, signal.SIG_IGN, lambda *_: None])
def test_sigterm_handling_ends_with_the_run(before):
    """What SIGTERM did before the run, it does after."""

    async def main() -> str:
        return "done"

    previous = signal.signal(signal.SIGTERM, before)
    try:
        assert run_main(main()) == "done"
        assert signal.getsignal(signal.SIGTERM) is before
    finally:
        signal.signal(signal.SIGTERM, previous)


@pytest.mark.parametrize(
    ("status", "code"),
    [
        ("completed", 0),
        ("loops_run_ended", 75),
        ("aborted", 1),
        ("failed", 1),
        ("unclean", 1),
        ("interrupted", 1),
    ],
)
def test_a_finished_runs_exit_code_says_whether_a_retry_can_work(status, code):
    assert exit_code(status) == code
