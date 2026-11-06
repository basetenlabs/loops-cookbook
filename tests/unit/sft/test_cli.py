"""``loops``: config loading, help, and check."""

import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest
from support.configs import SFT_MINIMAL, write_toml

from loops_cookbook import command
from loops_cookbook.cli import settings
from loops_cookbook.run_record import finished_message
from loops_cookbook.sft import cli
from loops_cookbook.sft import run as run_module
from loops_cookbook.sft.config import Config

from .conftest import FakeTrainer


def test_a_data_path_is_relative_to_the_config_that_names_it(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    data = {"data": {"train": "data/train.jsonl", "validation": "data/val.jsonl"}}
    path = write_toml(tmp_path / "configs" / "sft.toml", SFT_MINIMAL, data)
    monkeypatch.chdir(tmp_path)
    config = cli.load_config(path, {"data.validation": "held_out.jsonl"})
    assert config.data.train == str(tmp_path / "configs" / "data" / "train.jsonl")
    # A flag names a path from the shell, not from the config.
    assert config.data.validation == str(tmp_path / "held_out.jsonl")


@pytest.mark.parametrize("name", ["check", "run"])
def test_help_lists_every_setting(name, capsys):
    with pytest.raises(SystemExit) as exited:
        command.main([name, "--help"])
    assert exited.value.code == 0
    rendered = capsys.readouterr().out
    for path, _ in settings(Config):
        if path == "recipe":
            continue
        assert f"--{path}" in rendered, path


def test_a_misspelt_flag_is_answered_with_the_setting(capsys, tmp_path):
    path = write_toml(tmp_path / "x.toml", {"recipe": "sft"})
    with pytest.raises(SystemExit):
        command.main(["check", str(path), "--learning_rate", "1"])
    assert "--training.learning_rate" in capsys.readouterr().err


class Tokenizer:
    def decode(self, ids):
        return "".join(chr(token) for token in ids)


TOKENS = [{"input_ids": [1, ord("h"), ord("i")], "weights": [0, 1, 1]}]


@pytest.mark.parametrize(
    "model,source",
    [
        ({}, "m"),
        ({"base_model": "org/served", "tokenizer": "org/tokenizer-repo"}, "org/tokenizer-repo"),
    ],
)
async def test_check_reads_a_tokens_dataset_with_the_tokenizer_alone(
    sft_config, monkeypatch, model, source
):
    from loops_cookbook.sft import recipe, run

    def no_renderer(config):
        raise AssertionError("a tokens dataset is never rendered")

    loaded = []

    def load(name):
        loaded.append(name)
        return Tokenizer()

    monkeypatch.setattr(run, "build_renderer", no_renderer)
    config = sft_config(rows=TOKENS, data={"format": "tokens"}, model=model)
    output = "\n".join((await recipe.check(config, load=load)).describe())
    assert loaded == [source]
    assert "'hi'" in output
    assert "Renderer: none" in output


async def test_check_shows_token_ids_when_the_tokenizer_does_not_load(sft_config):
    from loops_cookbook.sft import recipe

    def load(name):
        raise OSError("gated")

    config = sft_config(rows=TOKENS, data={"format": "tokens"}, model={"tokenizer": "org/t"})
    output = "\n".join((await recipe.check(config, load=load)).describe())
    assert "the org/t tokenizer did not load (gated)" in output
    assert "'[104, 105]'" in output


@pytest.mark.skipif(os.name != "posix", reason="CLI signal handlers require a Unix event loop")
@pytest.mark.parametrize(
    "stage,command",
    [
        ("renderer", "check"),
        ("renderer", "run"),
        ("rows", "check"),
        ("rows", "run"),
        ("preview", "check"),
    ],
)
def test_sigterm_during_preparation_joins_workers_before_exit(command, stage, sft_config, tmp_path):
    config = tmp_path / "sft.json"
    settings = (
        sft_config(rows=TOKENS, data={"format": "tokens"})
        if stage == "preview"
        else sft_config(data={"render_workers": 2})
    )
    config.write_text(settings.model_dump_json())
    script = tmp_path / "stoppable.py"
    script.write_text(
        f"""
import json, multiprocessing, sys, time
from pathlib import Path
sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})
from unit.sft.conftest import FakeTrainer
from unit.sft.test_project import GatedChatRenderer
from loops_cookbook import command
from loops_cookbook.sft import recipe, run

directory = Path({str(tmp_path)!r})

def renderer_for(config):
    if {stage!r} != "rows":
        (directory / "loading").touch()
        deadline = time.monotonic() + 20
        while not (directory / "release").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("renderer was never released")
            time.sleep(0.01)
    return GatedChatRenderer(directory)

class Trainer(FakeTrainer):
    async def start(self):
        raise AssertionError("cancelled preparation must not provision a trainer")

    async def close(self):
        await super().close()
        (directory / "trainer-closed").touch()

if __name__ == "__main__":
    run.build_renderer = renderer_for
    recipe._tokenizer_decode = lambda *_: (renderer_for(None).detokenize, ())
    recipe.Trainer = lambda *args: Trainer()
    run.SERIAL_SECONDS = -1
    code = command.main([{command!r}, {str(config)!r}])
    (directory / "children").write_text(json.dumps([p.pid for p in multiprocessing.active_children()]))
    sys.exit(code)
"""
    )

    def wait_for(predicate, message):
        deadline = time.monotonic() + 10
        while not predicate():
            assert process.poll() is None, log.read_text()
            assert time.monotonic() < deadline, message
            time.sleep(0.01)

    log = tmp_path / "output.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for(
                lambda: (
                    (tmp_path / "loading").exists()
                    if stage != "rows"
                    else len(list(tmp_path.glob("started-*"))) == 2
                ),
                "preparation did not start",
            )
            process.send_signal(signal.SIGTERM)
            wait_for(
                lambda: "SIGTERM; stopping" in log.read_text(),
                "SIGTERM handler could not run while preparation was blocked",
            )
            process.send_signal(signal.SIGTERM)
            wait_for(
                lambda: "SIGTERM again" in log.read_text(),
                "repeated SIGTERM interrupted cleanup",
            )
            assert process.poll() is None, "active preparation must finish before exit"
            (tmp_path / "release").touch()
            assert process.wait(timeout=20) == 143, log.read_text()
        finally:
            (tmp_path / "release").touch()
            if process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
    assert (tmp_path / "children").read_text() == "[]"
    assert len(list(tmp_path.glob("started-*"))) == (2 if stage == "rows" else 0)
    if command == "run" and stage == "renderer":
        # The renderer is built before the run starts, so no run or trainer begins.
        assert not (tmp_path / "trainer-closed").exists()
        assert not list((tmp_path / "runs").glob("*"))
    elif command == "run":
        assert (tmp_path / "trainer-closed").exists()


def test_a_renderer_that_cannot_be_built_is_a_config_error_and_starts_no_run(
    sft_config, tmp_path, monkeypatch, capsys
):
    config = tmp_path / "sft.json"
    config.write_text(sft_config().model_dump_json())

    def refuse(config):
        raise ValueError("the chat template is refused")

    monkeypatch.setattr(run_module, "build_renderer", refuse)
    with pytest.raises(SystemExit) as exited:
        command.main(["run", str(config)])
    assert exited.value.code == 2
    assert "the chat template is refused" in capsys.readouterr().err
    assert not list((tmp_path / "runs").glob("*"))


async def test_check_holds_a_resume_to_what_its_run_trained(sft_config):
    """`loops check` refuses what `loops run` would, changed data or schedule, says when
    the run is finished, and names the run it continues."""
    from loops_cookbook.run_record import manifest_path
    from loops_cookbook.sft import recipe

    rows = [{**TOKENS[0], "input_ids": [1, ord("h"), ord(c)]} for c in "ijk"]
    training = {"batch_size": 1, "max_steps": 3}
    first = recipe.assemble(
        sft_config(rows=rows, data={"format": "tokens"}, training=training),
        trainer=FakeTrainer(keep_state_every=1),
    )
    assert (await first.execute()).ok

    def resumed(step, rows=rows, **changes):
        settings = {**training, **changes, "resume_from": str(manifest_path(first.out, step))}
        return sft_config(rows=rows, data={"format": "tokens"}, training=settings)

    output = "\n".join((await recipe.check(resumed(0), load=lambda name: Tokenizer())).describe())
    assert f"Resume: {first.out} at step 1; checked" in output
    output = "\n".join((await recipe.check(resumed(2), load=lambda name: Tokenizer())).describe())
    # The line `loops run` logs, so the two cannot drift apart.
    assert "Resume: " + finished_message(3, "max_steps = 3", "Raise max_steps to train more") in (
        output
    )
    with pytest.raises(ValueError, match="the data"):
        await recipe.check(resumed(0, rows=rows[:2]), load=lambda name: Tokenizer())
    with pytest.raises(ValueError, match="batch_size changed"):
        await recipe.check(resumed(0, batch_size=2), load=lambda name: Tokenizer())
