"""Check and train a loaded project through the public package entry point."""

import asyncio
import json
import multiprocessing
import os
import sys
import threading
import time

import pytest
from support.cancellation import outcome_within, settle_loop
from support.rl_sft import chat_row

from loops_cookbook.project import load
from loops_cookbook.sft import recipe
from loops_cookbook.sft import run as run_module

from .conftest import ChatRenderer, FakeTrainer


async def test_loaded_project_keeps_files_overrides_and_output_after_chdir(
    sft_config, tmp_path, monkeypatch, capsys
):
    rows = [{"input_ids": [1, 2, 3], "weights": [0, 1, 1]}]
    config = sft_config(
        rows=rows,
        data={"format": "tokens", "train": "train.jsonl"},
        output={"output_dir": "results"},
    )
    source = tmp_path / "sft.json"
    source.write_text(config.model_dump_json())
    caller = tmp_path / "caller"
    caller.mkdir()
    (caller / "held_out.jsonl").write_text(json.dumps(rows[0]) + "\n")
    monkeypatch.chdir(caller)
    monkeypatch.setattr(sys, "path", list(sys.path))
    trainers = []

    def trainer(*args):
        result = FakeTrainer()
        trainers.append(result)
        return result

    monkeypatch.setattr(recipe, "Trainer", trainer)
    # The preview decodes with ids, so the check never fetches the "m" tokenizer.
    monkeypatch.setattr(recipe, "_tokenizer_decode", lambda *_: (str, ()))
    project = load("../sft.json", overrides={"data.validation": "held_out.jsonl"})
    monkeypatch.chdir(tmp_path)
    report = await project.check()
    planned = report.plan
    assert not trainers
    assert len(planned.train.sequences) == len(planned.validation.sequences) == 1
    assert capsys.readouterr().out == ""
    assert "Trained text of" in "\n".join(report.describe())
    outcome = await project.train()
    assert outcome.ok and outcome.steps == planned.schedule.total_steps
    assert len(trainers) == 1 and trainers[0].closed
    assert len(trainers[0].updates) == outcome.steps
    assert list((caller / "results").glob("*/config.json"))


@pytest.mark.parametrize("outcome", ["complete", "error", "cancel"])
async def test_check_keeps_the_loop_responsive_and_settles_preparation(
    outcome, sft_config, tmp_path, monkeypatch
):
    source = tmp_path / "sft.json"
    source.write_text(sft_config().model_dump_json())
    monkeypatch.setattr(sys, "path", list(sys.path))
    project = load(source)
    loop = asyncio.get_running_loop()
    entered, release = asyncio.Event(), threading.Event()
    failure = ValueError("renderer unavailable")
    rendered = []

    class RecordingRenderer(ChatRenderer):
        def render_conversation(self, messages, tools=()):
            rendered.append(messages)
            return super().render_conversation(messages, tools)

    def renderer_for(config):
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(5):
            raise TimeoutError("check blocked the event loop during renderer preparation")
        if outcome == "error":
            raise failure
        return RecordingRenderer()

    monkeypatch.setattr(run_module, "build_renderer", renderer_for)
    checking = asyncio.create_task(project.check())
    try:
        await asyncio.wait_for(entered.wait(), 10)
        assert not checking.done(), "the event loop must run while preparation is blocked"
        if outcome == "cancel":
            for _ in range(2):
                checking.cancel()
                await settle_loop()
                assert not checking.done(), "cancellation must wait for the worker to settle"
    finally:
        release.set()
        result = await outcome_within(checking, 5)
    if outcome == "cancel":
        assert isinstance(result, asyncio.CancelledError)
        assert not rendered, "cancellation during preparation must skip dataset rendering"
    elif outcome == "error":
        assert result is failure
    else:
        assert isinstance(result, recipe.CheckReport) and len(result.plan.train.sequences) == 10


class GatedChatRenderer(ChatRenderer):
    """File gates work across the spawned rendering processes as well as threads."""

    def __init__(self, directory):
        self.directory = directory

    def render_conversation(self, messages, tools=()):
        question = messages[0]["content"]
        (self.directory / f"started-{question}").write_text(str(os.getpid()))
        deadline = time.monotonic() + 20
        while not (self.directory / "release").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("render was never released")
            time.sleep(0.01)
        return super().render_conversation(messages, tools)


@pytest.mark.parametrize("workers", [1, 2], ids=["serial", "processes"])
async def test_cancelled_check_finishes_active_rows_and_stops_before_the_rest(
    workers, sft_config, tmp_path, monkeypatch
):
    config = sft_config(
        rows=[chat_row((f"q{i}", "a")) for i in range(10)],
        validation=[chat_row(("validation", "a"))],
        data={"render_workers": workers},
    )
    source = tmp_path / "sft.json"
    source.write_text(config.model_dump_json())
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(run_module, "SERIAL_SECONDS", -1)
    # One row per batch: a cancellation is seen at every batch's end, so at every row.
    monkeypatch.setattr(run_module, "BATCH_SECONDS", 0)
    monkeypatch.setattr(run_module, "build_renderer", lambda config: GatedChatRenderer(tmp_path))
    checking = asyncio.create_task(load(source).check())
    try:
        async with asyncio.timeout(20):
            while len(list(tmp_path.glob("started-*"))) < workers:
                await asyncio.sleep(0.01)
        checking.cancel()
        await settle_loop()
        assert not checking.done(), "active renders still own their worker processes"
    finally:
        (tmp_path / "release").touch()
        result = await outcome_within(checking, 20)
    assert isinstance(result, asyncio.CancelledError)
    started = list(tmp_path.glob("started-*"))
    assert len(started) == workers, "cancelled checks must not render queued rows or validation"
    worker_pids = {int(path.read_text()) for path in started}
    assert worker_pids.isdisjoint(child.pid for child in multiprocessing.active_children())
