"""An SFT run against a fake trainer: rendering, schedule, records, resume and failure."""

import asyncio
import hashlib
import json
import logging
import math
import os
import shutil
import weakref
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace

import pytest
from baseten.loops.models import EncodedTextChunk, ImageChunk, ModelInput
from support.rl_sft import chat_row

from loops_cookbook.llm.rendering import RenderedPrompt
from loops_cookbook.run_record import finished_message, manifest_path
from loops_cookbook.sft import recipe
from loops_cookbook.sft import run as run_module
from loops_cookbook.sft.run import Manifest, Schedule, plan

from .conftest import ChatRenderer, FakeTrainer, Merging


def run_with(config, trainer=None, renderer=None):
    """The run `loops run` would assemble, with a fake trainer and template."""
    trainer = trainer or FakeTrainer()
    return recipe.assemble(config, trainer=trainer, renderer=renderer or ChatRenderer()), trainer


class RecordingSink:
    """Every update's gauge names, by step, as `Metrics.log` receives them."""

    def __init__(self):
        self.seen = []

    def log(self, values, step=None, step_scoped=None):
        self.seen.append((step, set(values)))


@pytest.fixture
def manifest_after(sft_config):
    """``await manifest_after(steps, **training)``: run ``steps`` updates; the manifest of
    the last."""

    async def run(steps, **training):
        first, _ = run_with(sft_config(training={"max_steps": steps, **training}))
        await first.execute()
        return manifest_path(first.out, steps - 1)

    return run


def records(run):
    return [json.loads(line) for line in (run.out / "steps.jsonl").read_text().splitlines()]


# --- schedule --------------------------------------------------------------------


def test_each_epoch_trains_every_sequence_once_in_its_own_order(sft_config):
    schedule = Schedule.of(10, sft_config(training={"batch_size": 4, "epochs": 3}))
    assert (schedule.steps_per_epoch, schedule.total_steps) == (3, 9)
    epochs = [
        [index for step in range(epoch * 3, epoch * 3 + 3) for index in schedule.batch(step)]
        for epoch in range(3)
    ]
    assert all(sorted(order) == list(range(10)) for order in epochs)
    assert epochs[0] != epochs[1]
    assert Schedule.of(10, sft_config(training={"batch_size": 4, "epochs": 3})).batch(
        4
    ) == schedule.batch(4)


@pytest.mark.parametrize(("max_steps", "total"), [(2, 2), (0, 9)], ids=["mid_epoch", "no_limit"])
def test_max_steps_stops_mid_epoch_and_0_runs_every_epoch(sft_config, max_steps, total):
    training = {"batch_size": 4, "epochs": 3, "max_steps": max_steps}
    assert Schedule.of(10, sft_config(training=training)).total_steps == total


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("constant", [0.5, 1.0, 1.0, 1.0, 1.0, 1.0]),
        # Decay runs over the updates after warmup and never reaches zero: an update
        # at learning rate 0 would cost a step and change nothing.
        ("linear", [0.5, 1.0, 1.0, 0.75, 0.5, 0.25]),
        ("cosine", [0.5, 1.0, *(0.5 * (1 + math.cos(math.pi * k / 4)) for k in range(4))]),
    ],
)
def test_learning_rate_warms_up_then_follows_its_schedule(sft_config, kind, expected):
    config = sft_config(
        training={
            "learning_rate": 1.0,
            "lr_schedule": kind,
            "warmup_steps": 2,
            "max_steps": 6,
            "batch_size": 1,
        }
    )
    schedule = Schedule.of(100, config)
    assert [schedule.learning_rate(step) for step in range(6)] == pytest.approx(expected)


# --- rendering -------------------------------------------------------------------


class UnpicklableChatRenderer(ChatRenderer):
    def __init__(self):
        self.closure = lambda: None


def _load_only_in(pid):
    if os.getpid() != pid:
        raise RuntimeError("this renderer loads only in the process that made it")
    return ChatRenderer()


class ParentOnlyChatRenderer(ChatRenderer):
    """Pickles here but fails to load in a worker, as a class defined in a notebook does."""

    def __reduce__(self):
        return (_load_only_in, (os.getpid(),))


class TrackedPrompt(RenderedPrompt):
    """A prompt a test can hold a weak reference to."""


class TrackingChatRenderer(ChatRenderer):
    """Notes, as each prompt renders, which of the prompts it rendered before are alive."""

    def __init__(self):
        self.rendered = []
        self.alive = []

    def render(self, messages, tools=()):
        self.alive.append([prompt() is not None for prompt in self.rendered])
        prompt = TrackedPrompt.text(super().render(messages, tools).ids)
        self.rendered.append(weakref.ref(prompt))
        return prompt


def mixed_rows():
    """Rows that train and rows refused for their length, interleaved."""
    return [chat_row((f"q{i}", "x" * (300 if i % 3 == 1 else i + 1))) for i in range(7)]


class UnstartedPool:
    """A pool whose processes never start, so the split renders here: for tests of how
    many processes a split asks for, without paying for spawning them."""

    def __init__(self, workers, **kwargs):
        pass

    def submit(self, *args, **kwargs):
        raise BrokenProcessPool("not started in this test")

    def shutdown(self, **kwargs):
        pass


@pytest.fixture
def parallel_render(monkeypatch, caplog):
    """``parallel_render(cores)``: a split goes to worker processes at once, on a machine
    with ``cores`` usable cores; the render's log is captured."""

    def on(cores):
        monkeypatch.setattr(run_module, "SERIAL_SECONDS", -1.0)
        monkeypatch.setattr(run_module, "_usable_cores", lambda: cores)
        caplog.set_level(logging.INFO, logger=run_module.__name__)

    return on


async def test_rows_rendered_in_worker_processes_become_what_they_would_here(
    sft_config, parallel_render, caplog
):
    config = sft_config(rows=mixed_rows(), validation=[chat_row(("v", "w"))])
    here = await plan(config, ChatRenderer())
    parallel_render(3)
    elsewhere = await plan(config, ChatRenderer())
    assert "train: rendering the other 7 rows in 2 processes" in caplog.text
    assert here.train.refusals and here.train.sequences
    assert (elsewhere.train, elsewhere.validation) == (here.train, here.validation)
    assert elsewhere.digest == here.digest


async def test_a_renderer_that_does_not_pickle_renders_in_this_process(
    sft_config, parallel_render, caplog
):
    config = sft_config(rows=mixed_rows())
    here = await plan(config, ChatRenderer())
    parallel_render(3)
    planned = await plan(config, UnpicklableChatRenderer())
    assert "the renderer does not pickle" in caplog.text
    assert "processes" not in caplog.text
    assert planned.digest == here.digest


async def test_rows_the_workers_could_not_render_render_in_this_process(
    sft_config, parallel_render, caplog
):
    config = sft_config(rows=mixed_rows())
    here = await plan(config, ChatRenderer())
    parallel_render(3)
    planned = await plan(config, ParentOnlyChatRenderer())
    assert "rendering the other 7 rows in this process" in caplog.text
    assert planned.digest == here.digest


@pytest.mark.parametrize(
    ("cores", "render_workers", "processes"),
    [
        (64, None, 4),  # at most four by default
        (3, None, 2),  # one core left for everything else
        (2, None, None),  # two cores: this process alone
        (2, 3, 3),  # render_workers sets it, whatever the cores
        (64, 2, 2),
        (64, 1, None),  # one worker is this process
    ],
)
async def test_how_many_processes_a_split_renders_in(
    sft_config, parallel_render, caplog, monkeypatch, cores, render_workers, processes
):
    monkeypatch.setattr(run_module, "ProcessPoolExecutor", UnstartedPool)
    parallel_render(cores)
    data = {} if render_workers is None else {"render_workers": render_workers}
    planned = await plan(sft_config(rows=mixed_rows(), data=data), ChatRenderer())
    if processes is None:
        assert "processes" not in caplog.text
    else:
        assert f"train: rendering the other 7 rows in {processes} processes" in caplog.text
    assert planned.train.sequences


async def test_a_refused_row_lets_go_of_what_it_rendered(sft_config):
    renderer = TrackingChatRenderer()
    planned = await plan(
        sft_config(rows=[chat_row(("q", "x" * 300)), chat_row(("q", "a"))]), renderer
    )
    assert len(planned.train.refusals) == 1
    assert renderer.alive[1] == [False]


async def test_a_long_render_logs_its_progress(sft_config, monkeypatch, caplog):
    monkeypatch.setattr(run_module, "PROGRESS_SECONDS", 0.0)
    with caplog.at_level(logging.INFO, logger=run_module.__name__):
        await plan(sft_config(rows=mixed_rows()), ChatRenderer())
    assert "train: rendered 1 of 7 rows" in caplog.text
    assert "rendered 7 of 7" not in caplog.text


# --- what trains -----------------------------------------------------------------


class Pictured(ChatRenderer):
    """Renders each conversation's first token as an image placeholder."""

    def render_conversation(self, messages, tools=()):
        rendered = super().render_conversation(messages, tools)
        chunks = [
            ImageChunk(data=b"picture", format="png", expected_tokens=1),
            EncodedTextChunk(tokens=list(rendered.ids[1:])),
        ]
        return replace(rendered, model_input=ModelInput(chunks=chunks))


class DatumRecorder(FakeTrainer):
    """Hashes every datum it is sent, in the order it is sent them."""

    def __init__(self):
        super().__init__()
        self.sent = hashlib.sha256()

    async def update(self, step, datums, learning_rate):
        self.sent.update(f"update {step}".encode())
        for datum in datums:
            self.sent.update(datum.model_dump_json().encode())
        return await super().update(step, datums, learning_rate)

    async def evaluate(self, datums):
        self.sent.update(b"evaluate")
        for datum in datums:
            self.sent.update(datum.model_dump_json().encode())
        return await super().evaluate(datums)


def tool_row():
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    return {
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "content": "done"},
            {"role": "assistant", "content": "ok", "weight": 0},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "fine"},
        ],
        "id": "kept aside",
    }


CASES = {
    "chat": (ChatRenderer, [*mixed_rows(), tool_row()], {}),
    "per turn": (Merging, [chat_row(("a", "bc"), ("d", "ef")), chat_row(("g", "hi"))], {}),
    "images": (Pictured, mixed_rows(), {}),
    "tokens": (
        None,
        [{"input_ids": [1, 2, 3 + i, 300 + i], "weights": [0, 1, i % 2, 1]} for i in range(5)],
        {"format": "tokens"},
    ),
}

GOLDEN = {
    "chat": (
        "0defed13f17eff15a8cd253fc458e0c03226f165d5e0fdd9ab0d096623ad09f6",
        "f2bbcc5bc6d243dd96b96cc0a1ed8b8a861a164fbbc5824083605b2283ae1879",
    ),
    "per turn": (
        "6356ffed982d8b5d71a882102e725a8a1cf68b63863af27cd9b3a88e606189da",
        "339341d32956377b94b26b6f980de5e98f27e09f2a9c0e1b884061f0a839ed08",
    ),
    "images": (
        "dc38c7d3d8249a4ad782f8626a6d575d7acd2b11124dd2f8172d6a9723c17674",
        "5a184cd0eed649b87348b9027ba5b798620487038e89ee53e19ac886ccb8c77e",
    ),
    "tokens": (
        "4357d9e9148a0bcd6507f2403bc8cc2f11cca7530c57c36853df3cd44c5bddf3",
        "a516441dc274bc441b7eea08fb875628a9a77e27a8c914b665e40092f4fff11a",
    ),
}
"""Each case's data digest, and a hash of its data.json and every datum the trainer is
sent, recorded before sequences were held compactly (#104). Runs resume from manifests
holding the digest, so the same data must keep it however sequences are held. The datum
hashes were re-recorded for baseten-loops 0.33, whose ``Datum`` swaps
``routed_experts_ref`` for the two provenance span fields: the data digests are unchanged."""


@pytest.mark.parametrize("case", CASES)
async def test_what_trains_and_its_digest_are_what_runs_recorded(sft_config, case):
    renderer, rows, data = CASES[case]
    config = sft_config(
        rows=rows, validation=rows[:2], data=data, training={"batch_size": 2, "epochs": 2}
    )
    trainer = DatumRecorder()
    run = recipe.assemble(config, trainer=trainer, renderer=renderer and renderer())
    outcome = await run.execute()
    assert outcome.ok, outcome.reason
    recorded = (run.out / "data.json").read_text()
    trainer.sent.update(recorded.encode())
    assert (json.loads(recorded)["digest"], trainer.sent.hexdigest()) == GOLDEN[case]


# --- the run ---------------------------------------------------------------------


async def test_a_run_records_every_update_and_scores_validation_on_cadence(sft_config):
    config = sft_config(validation=[chat_row(("vq", "va"))] * 2, eval={"every_steps": 2})
    run, trainer = run_with(config)
    sink = RecordingSink()
    outcome = await run.execute(sink)
    seen = sink.seen

    assert outcome.ok and outcome.steps == 6
    assert [step for step, *_ in trainer.updates] == list(range(6))
    steps = records(run)
    assert [record["step"] for record in steps] == list(range(6))
    assert all(record["train/nll"] > 0 for record in steps)
    # Before the first update, every second update, and after the last.
    assert [step for step, names in seen if "val/nll" in names] == [-1, 1, 3, 5]
    assert trainer.closed
    data = json.loads((run.out / "data.json").read_text())
    assert data["train"]["sequences"] == 10 and data["validation"]["sequences"] == 2


async def test_the_last_update_is_always_kept_with_a_resume_manifest(sft_config):
    run, trainer = run_with(sft_config())
    await run.execute()
    assert trainer.kept == [5]
    manifest = json.loads((manifest_path(run.out, 5)).read_text())
    assert manifest["next_step"] == 6 and manifest["trainer_state_path"] == "bt://state/step-5"


async def test_a_resumed_run_continues_where_its_manifest_left_off(sft_config):
    first, _ = run_with(sft_config(training={"max_steps": 3}), FakeTrainer(keep_state_every=2))
    await first.execute()
    manifest = manifest_path(first.out, 1)

    resumed, again = run_with(sft_config(training={"resume_from": str(manifest)}))
    outcome = await resumed.execute()

    assert resumed.out == first.out
    assert [step for step, *_ in again.updates] == [2, 3, 4, 5]
    assert outcome.ok and outcome.steps == 4


async def test_a_resume_after_the_last_epoch_completes_without_training(sft_config):
    """Without max_steps the epochs bound the run: once the data is read, a resume of a
    finished run completes and starts no trainer, as in Tinker."""
    first, _ = run_with(sft_config(), FakeTrainer(keep_state_every=1))
    await first.execute()

    resumed, again = run_with(sft_config(training={"resume_from": str(first.out)}))
    outcome = await resumed.execute()

    assert outcome.ok and outcome.steps == 0 and outcome.next_step == 6
    assert again.client is None and again.updates == []


async def test_a_resume_at_max_steps_completes_without_training(sft_config, manifest_after, caplog):
    manifest = await manifest_after(3)
    resumed, trainer = run_with(sft_config(training={"max_steps": 3, "resume_from": str(manifest)}))
    with caplog.at_level(logging.INFO):
        outcome = await resumed.execute()
    assert outcome.ok and outcome.steps == 0 and outcome.next_step == 3
    assert trainer.client is None and trainer.updates == []
    assert finished_message(3, "max_steps = 3", "Raise max_steps to train more") in caplog.text


async def test_a_resume_with_no_update_left_is_still_held_to_its_run(sft_config, manifest_after):
    """`loops run` refuses what `loops check` refuses, whether or not an update is left."""
    manifest = await manifest_after(3)
    training = {"max_steps": 3, "batch_size": 2, "resume_from": str(manifest)}
    resumed, trainer = run_with(sft_config(training=training))
    outcome = await resumed.execute()
    assert outcome.status == "failed" and "batch_size changed" in (outcome.reason or "")
    assert trainer.client is None


@pytest.mark.parametrize(
    ("training", "more"),
    [
        ({"max_steps": 3}, "Raise max_steps to train more"),
        ({"epochs": 1}, "Raise epochs to train more"),
        # 10 sequences in batches of 4 are 3 updates an epoch: both limits end the run.
        ({"max_steps": 3, "epochs": 1}, "Raise max_steps and epochs to train more"),
        (
            {"max_steps": 3, "lr_schedule": "linear"},
            "start a new run with --training.init_from bt://loops:r/weights/step-2",
        ),
    ],
    ids=["max_steps", "epochs", "both", "decaying"],
)
def test_a_finished_run_says_how_to_train_more(sft_config, training, more):
    """A decaying learning rate reached its end at the last update, and the update count
    is part of its contract: more updates need a new run, not a larger max_steps."""
    schedule = Schedule.of(10, sft_config(training=training))
    assert more in schedule.more("bt://loops:r/weights/step-2")


async def test_a_run_directory_resumes_the_branch_written_last_not_the_furthest(sft_config):
    """A resume from step 1 of a run that reached step 5 branches it; resuming the run
    directory then continues that branch, at step 3, not the abandoned step 6."""
    first, _ = run_with(sft_config(), FakeTrainer(keep_state_every=1))
    await first.execute()
    branch, _ = run_with(
        sft_config(training={"resume_from": str(manifest_path(first.out, 1)), "max_steps": 3}),
        FakeTrainer(keep_state_every=1),
    )
    await branch.execute()
    resumed, trainer = run_with(sft_config(training={"resume_from": str(first.out)}))
    await resumed.execute()
    assert [step for step, *_ in trainer.updates] == [3, 4, 5]


async def test_a_copied_run_directory_resumes_in_the_copy(sft_config, tmp_path):
    """The directory given is the run directory: the resume writes there, and the
    original is left as it was."""
    first, _ = run_with(sft_config(training={"max_steps": 3}), FakeTrainer(keep_state_every=1))
    await first.execute()
    copy = tmp_path / "copy"
    shutil.copytree(first.out, copy)
    before = sorted(path.name for path in (first.out / "checkpoints").iterdir())

    resumed, trainer = run_with(sft_config(training={"resume_from": str(copy)}))
    await resumed.execute()

    assert resumed.out == copy and [step for step, *_ in trainer.updates] == [3, 4, 5]
    assert manifest_path(copy, 5).exists()
    assert sorted(path.name for path in (first.out / "checkpoints").iterdir()) == before


async def test_resume_refuses_data_that_changed(sft_config, manifest_after):
    manifest = await manifest_after(1)

    changed = sft_config(
        rows=[chat_row((f"other{i}", f"a{i}")) for i in range(10)],
        training={"resume_from": str(manifest)},
    )
    resumed, trainer = run_with(changed)
    outcome = await resumed.execute()
    assert not outcome.ok and "changed since the checkpoint" in outcome.reason
    assert trainer.updates == []


async def test_a_failed_run_records_why_and_keeps_its_last_update(sft_config):
    run, trainer = run_with(sft_config(), FakeTrainer(fail_at=3))
    outcome = await run.execute()

    assert outcome.status == "failed" and outcome.next_step == 3
    assert "trainer went away" in json.loads((run.out / "failure.json").read_text())["error"]
    assert trainer.last_gasp == [2]
    assert json.loads((manifest_path(run.out, 2)).read_text())["next_step"] == 3
    assert trainer.closed


async def test_rows_that_cannot_train_are_reported_not_trained(sft_config):
    rows = [chat_row(("q", "a")), chat_row(("q", "x" * 300))]
    run, trainer = run_with(sft_config(rows=rows, training={"batch_size": 1, "epochs": 1}))
    await run.execute()
    train = json.loads((run.out / "data.json").read_text())["train"]
    assert train["sequences"] == 1 and train["rows_refused"] == 1
    assert "over max_seq_len 256" in train["refusals"][0]["reason"]
    assert len(trainer.updates) == 1


async def test_a_dataset_with_nothing_trainable_fails_before_the_trainer_starts(sft_config):
    run, trainer = run_with(sft_config(rows=[chat_row(("q", "x" * 300))]))
    outcome = await run.execute()
    assert not outcome.ok and "no training row can be trained" in outcome.reason
    assert trainer.client is None


async def test_a_run_failing_after_an_update_resumes_after_it_not_at_it(sft_config):
    """The update was applied before keeping it failed, so the state saved on the way
    out includes it, and the manifest must not ask for it again."""
    run, trainer = run_with(sft_config(), FakeTrainer(fail_keeping=3))
    outcome = await run.execute()

    assert outcome.status == "failed" and outcome.next_step == 4 and outcome.steps == 4
    assert trainer.last_gasp == [3]
    assert json.loads((manifest_path(run.out, 3)).read_text())["next_step"] == 4
    assert not (manifest_path(run.out, 2)).exists()


async def test_a_failure_after_the_update_was_kept_needs_no_last_gasp(sft_config):
    run, trainer = run_with(sft_config(), FakeTrainer(keep_state_every=1))

    class FailingSink:
        def log(self, values, step=None, step_scoped=None):
            if step == 2:
                raise RuntimeError("metrics went away")

    outcome = await run.execute(FailingSink())
    assert outcome.next_step == 3 and trainer.last_gasp == []
    assert sorted(path.name for path in (run.out / "checkpoints").iterdir()) == [
        "step-0.json",
        "step-1.json",
        "step-2.json",
    ]


@pytest.mark.parametrize(
    ("training", "data"),
    [({"batch_size": 2}, {}), ({}, {"shuffle_seed": 7})],
    ids=["batch size", "shuffle seed"],
)
async def test_resume_refuses_a_different_order_of_the_same_data(
    sft_config, manifest_after, training, data
):
    manifest = await manifest_after(1)
    resumed, trainer = run_with(
        sft_config(training={"resume_from": str(manifest), **training}, data=data)
    )
    outcome = await resumed.execute()
    assert not outcome.ok and "which sequences each update trains" in outcome.reason
    assert trainer.updates == []


@pytest.mark.parametrize("scope", ["dedicated", "serverless"])
async def test_the_config_a_run_saves_loads_back_as_that_config(sft_config, scope):
    """config.json is a config `loops` takes, so what a run saves is what it ran."""
    from loops_cookbook.sft.cli import load_config

    config = sft_config(compute={"scope": scope})
    run, _ = run_with(config)
    assert (await run.execute()).ok
    assert load_config(run.out / "config.json") == config


async def test_a_resumed_run_keeps_the_config_it_started_with(sft_config, manifest_after):
    manifest = await manifest_after(1)
    out = manifest.parent.parent
    original = (out / "config.json").read_text()
    resumed, _ = run_with(
        sft_config(training={"resume_from": str(manifest), "learning_rate": 5e-5})
    )
    await resumed.execute()
    assert resumed.out == out and (out / "config.json").read_text() == original
    resumed_with = json.loads((out / "config-from-step-1.json").read_text())
    assert resumed_with["training"]["learning_rate"] == 5e-5


async def test_a_tokens_dataset_trains_without_a_renderer(sft_config, monkeypatch):
    def no_renderer(config):
        raise AssertionError("a tokens dataset is never rendered")

    monkeypatch.setattr(run_module, "build_renderer", no_renderer)
    rows = [{"input_ids": [1, 2, 3], "weights": [0, 1, 1]} for _ in range(4)]
    config = sft_config(rows=rows, data={"format": "tokens"}, training={"epochs": 1})
    trainer = FakeTrainer()
    run = recipe.assemble(config, trainer=trainer)
    outcome = await run.execute()
    assert outcome.ok, outcome.reason
    assert len(trainer.updates) == 1


@pytest.mark.parametrize(
    "change",
    [{"learning_rate": 2e-4}, {"warmup_steps": 1}, {"lr_schedule": "linear"}, {"max_steps": 4}],
    ids=["peak", "warmup", "schedule", "length of a decaying schedule"],
)
async def test_resume_refuses_a_different_learning_rate_for_the_same_step(
    sft_config, manifest_after, change
):
    manifest = await manifest_after(2, lr_schedule="cosine")
    resumed, trainer = run_with(
        sft_config(
            training={
                "lr_schedule": "cosine",
                "max_steps": 2,
                **change,
                "resume_from": str(manifest),
            }
        )
    )
    outcome = await resumed.execute()
    assert not outcome.ok and "at what learning rate" in outcome.reason
    assert trainer.updates == []


async def test_a_constant_learning_rate_can_resume_for_longer(sft_config, manifest_after):
    manifest = await manifest_after(2)
    resumed, trainer = run_with(sft_config(training={"max_steps": 4, "resume_from": str(manifest)}))
    outcome = await resumed.execute()
    assert outcome.ok and [step for step, *_ in trainer.updates] == [2, 3]


async def test_a_resume_records_its_own_loops_run_beside_the_first(sft_config, manifest_after):
    manifest = await manifest_after(2)
    resumed, _ = run_with(sft_config(training={"max_steps": 3, "resume_from": str(manifest)}))
    await resumed.execute()
    assert (resumed.out / "run.json").exists()
    assert json.loads((resumed.out / "run-from-step-2.json").read_text())["run_id"] == "loops-run-1"


async def test_fields_outside_the_contract_are_reported_not_trained(sft_config):
    rows = [{**chat_row((f"q{i}", f"a{i}")), "id": i} for i in range(4)]
    run, _ = run_with(sft_config(rows=rows, training={"epochs": 1}))
    await run.execute()
    assert json.loads((run.out / "data.json").read_text())["train"]["ignored_fields"] == {"id": 4}


async def test_a_refused_resume_leaves_the_first_runs_records_as_they_were(sft_config):
    run, _ = run_with(sft_config(), FakeTrainer(fail_at=3, keep_state_every=1))
    await run.execute()
    manifest = manifest_path(run.out, 2)
    recorded = {name: (run.out / name).read_text() for name in ("data.json", "failure.json")}

    changed = [chat_row((f"other{i}", f"a{i}")) for i in range(10)]
    resumed, _ = run_with(sft_config(rows=changed, training={"resume_from": str(manifest)}))
    outcome = await resumed.execute()

    assert not outcome.ok
    assert {name: (run.out / name).read_text() for name in recorded} == recorded
    assert (
        "changed since" in json.loads((run.out / "failure-from-step-3.json").read_text())["error"]
    )


def test_a_manifest_from_another_recipe_is_refused(tmp_path):
    path = tmp_path / "rl.json"
    path.write_text(
        json.dumps(
            {
                "method": "rl",
                "trainer_state_path": "bt://loops:r/state",
                "next_step": 1,
                "next_group_index": 0,
                "policy_version": 1,
                "shuffle_seed": 0,
                "contract": {},
                "run_id": "r",
                "run_directory": str(tmp_path),
            }
        )
    )
    with pytest.raises(ValueError, match="manifest"):
        Manifest.read(path)


async def test_assemble_builds_the_trainer_to_restore_the_manifests_state(
    sft_config, loops_service
):
    first, _ = run_with(sft_config(training={"max_steps": 2}))
    await first.execute()
    config = sft_config(training={"resume_from": str(manifest_path(first.out, 1))})
    run = recipe.assemble(config, renderer=ChatRenderer())
    assert run.out == first.out
    await run.trainer.start()
    assert loops_service.client.loads == [("load_state_with_optimizer", "bt://state/step-1")]


class TokenizerChatRenderer(ChatRenderer):
    """The fake chat template, keeping the tokenizer it was built with."""

    def __init__(self, tokenizer, chat_template_kwargs):
        self.tokenizer = tokenizer

    def parse(self, prompt_ids, completion_ids, tools=()):
        raise NotImplementedError


def test_the_renderer_is_built_on_the_configured_tokenizer(sft_config, tmp_path):
    from support.local_tokenizer import save_tokenizer

    from loops_cookbook.sft.run import build_renderer

    source = str(save_tokenizer(tmp_path / "tokenizer"))
    config = sft_config(
        model={
            "base_model": "org/no-such-repo",
            "tokenizer": source,
            "renderer": f"{__name__}:TokenizerChatRenderer",
        }
    )
    assert build_renderer(config).tokenizer.name_or_path == source


async def test_a_failure_under_an_ended_loops_run_is_reported_as_such(sft_config):
    ended = "Loops reports run loops-run-1 INACTIVE"
    run, _ = run_with(sft_config(), FakeTrainer(fail_at=3, ended=ended))
    outcome = await run.execute()

    assert (outcome.status, outcome.reason, outcome.next_step) == ("loops_run_ended", ended, 3)
    assert json.loads((run.out / "failure.json").read_text())["status"] == "loops_run_ended"


async def test_a_cancellation_while_loops_answers_keeps_the_record_and_the_last_update(
    sft_config,
):
    run, trainer = run_with(sft_config(), FakeTrainer(fail_at=3))
    asking = asyncio.Event()

    async def slow():
        asking.set()
        await asyncio.sleep(60)

    trainer.run_ended = slow
    task = asyncio.create_task(run.execute())
    await asyncio.wait_for(asking.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert json.loads((run.out / "failure.json").read_text())["status"] == "failed"
    assert trainer.last_gasp == [2]
