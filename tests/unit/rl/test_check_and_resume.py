"""``loops check`` on an RL config, and what a resumed run reports, for any runtime.

The runtime is the in-process direct pattern's, as its config names it; nothing here
depends on which runtime it is.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from support.builders import task_file
from support.configs import Tables, episode_config, write_toml
from support.resources import PROVIDERS, FakeProvider, registry_at, request
from support.resume import resumed_offline

from loops_cookbook.resources.api import ResourceScope
from loops_cookbook.rl import cli

ROOT = Path(__file__).resolve().parents[3]
PATTERN = ROOT / "patterns" / "custom_runtime" / "in_process_direct"
TASKS = [
    {"name": "b", "instruction": "Say B", "answer": "B"},
    {"name": "c", "instruction": "Say C", "answer": "C"},
]
RUNTIME = {"episode": "direct_episode:AnswerEpisode", "agent_timeout_seconds": 20}


@pytest.fixture(autouse=True)
def pattern(monkeypatch):
    """The pattern's episode module, importable as its config names it."""
    monkeypatch.syspath_prepend(str(PATTERN))


def write_config(tmp_path: Path, *tables: Tables) -> Path:
    config = episode_config(tmp_path, RUNTIME["episode"], TASKS)
    return write_toml(tmp_path / "train.toml", config, *tables)


# ------------------------------------------------------------------ the model


def test_check_reports_the_model_loops_serves_and_the_sequence_budget_it_allows(
    loops_check, tmp_path, loops_capabilities, monkeypatch
):
    monkeypatch.setenv("BASETEN_API_KEY", "key")
    loops_capabilities.models = [{"model_name": "m", "max_enabled_seq_len": 4096}]

    code, out, err = loops_check(write_config(tmp_path))

    assert code == 0, err
    assert "Loops model: m; max_seq_len model default (enabled up to 4096)" in out


def test_check_refuses_a_model_loops_does_not_serve_before_building_the_runtime(
    loops_check, tmp_path, loops_capabilities, monkeypatch
):
    monkeypatch.setenv("BASETEN_API_KEY", "key")
    loops_capabilities.models = [{"model_name": "m2"}]

    code, out, err = loops_check(write_config(tmp_path))

    assert code == 2
    assert "[model] base_model 'm' is not a model Loops serves; closest: m2" in err
    assert "Rollout runtime:" not in out


def test_check_passes_offline_and_says_the_model_was_not_checked(
    loops_check, tmp_path, monkeypatch
):
    monkeypatch.setenv("BASETEN_API_KEY", "key")  # the default fake: Loops is out of reach

    code, out, err = loops_check(write_config(tmp_path))

    assert code == 0, err
    assert "Loops model: m not checked (ConnectError from" in out


# ------------------------------------------------------------ check of a resume


def _prior_run(tmp_path: Path, provider: FakeProvider, *, base_model: str = "m"):
    """A prior run's directory: one resource left in its journal, and a step manifest."""
    directory = tmp_path / "runs" / "prior-1-deadbeef"
    journal = registry_at(directory / "state", provider, run_id=directory.name)
    manifest = directory / "checkpoints" / "step-3.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "trainer_state_path": "bt://loops:x/state/step-3",
                "next_step": 4,
                "next_group_index": 8,
                "policy_version": 4,
                "shuffle_seed": 0,
                "contract": {"training": {"base_model": base_model, "lora_rank": 32}},
                "run_id": directory.name,
                "run_directory": str(directory),
            }
        )
    )
    return directory, journal, manifest


def _files(directory: Path) -> dict[str, bytes]:
    return {str(path): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("prior", ["stopped", "live"])
def test_check_of_a_resume_leaves_the_prior_run_and_its_resources_untouched(
    loops_check, tmp_path, monkeypatch, prior
):
    provider = FakeProvider()
    monkeypatch.setitem(PROVIDERS, "check", provider)
    directory, journal, manifest = _prior_run(tmp_path, provider)
    asyncio.run(journal.provision(request("a"), scope=ResourceScope.EPISODE, owner_id="attempt-1"))
    if prior == "stopped":
        journal.journal.close()  # the process died holding a resource
    before, store, calls = _files(directory), dict(provider.store), len(provider.calls)
    fake = {"adapter": "support.resources:create_fake", "instance": "check"}
    resume = {"providers": {"fake": fake}, "training": {"resume_from": manifest}}
    code, _, err = loops_check(write_config(tmp_path, resume))
    assert code == 0, err
    assert (_files(directory), provider.store, provider.calls[calls:]) == (before, store, [])


def test_check_refuses_a_resume_whose_weights_cannot_load(loops_check, tmp_path):
    _, journal, manifest = _prior_run(tmp_path, FakeProvider(), base_model="other")
    journal.journal.close()
    code, _, err = loops_check(write_config(tmp_path, {"training": {"resume_from": manifest}}))
    assert code == 2
    assert "this checkpoint's weights cannot load into this run (base_model: 'other' -> 'm')" in err


def test_check_takes_init_from_as_a_loops_uri_not_a_path(loops_check, tmp_path):
    weights = "bt://loops:source-run/weights/step-249"
    path = write_config(tmp_path, {"training": {"init_from": weights}})
    code, _, err = loops_check(path)
    assert code == 0, err
    assert cli.load_training_config(path).training.init_from == weights


# ---------------------------------------------------------------- a resumed run


@pytest.fixture
def settings(config, tmp_path):
    """``settings(**overrides)``: a one-step run of the pattern's runtime."""

    def build(**overrides):
        return config(
            **{
                "rollout_runtime.factory": "loops_cookbook.episodes:create_runtime",
                "rollout_runtime.config": RUNTIME,
                "rollout_runtime.files": {"tasks": str(task_file(tmp_path, TASKS))},
                "compute.sampler_replica_affinity": True,
                "max_steps": 1,
                **overrides,
            }
        )

    return build


@pytest.mark.parametrize("setting", ["tokenizer", "renderer"])
async def test_a_resume_reports_a_changed_tokenizer_or_renderer(
    settings, tmp_path, caplog, setting
):
    value = {"tokenizer": str(tmp_path), "renderer": "elsewhere:renderer"}[setting]
    with caplog.at_level("WARNING", logger="loops_cookbook.rl.run"):
        outcome, _, _ = await resumed_offline(settings, **{f"model.{setting}": value})
    assert outcome.ok
    assert f"Resuming with changed settings ({setting}" in caplog.text


@pytest.mark.parametrize(
    "setting,recorded",
    [
        ("renderer_source", "the cookbook's for org/model: pkg:factory"),
        ("renderer_class", "AnotherRenderer"),
    ],
    ids=["selected-elsewhere", "another-class"],
)
async def test_a_resume_reports_a_renderer_chosen_differently_with_the_same_setting(
    settings, caplog, setting, recorded
):
    """With no ``[model] renderer`` either time, the cookbook's selection or Prime's
    renderer can still change between the halves of a run."""

    def chosen_differently(manifest):
        manifest["contract"]["training"][setting] = recorded

    with caplog.at_level("WARNING", logger="loops_cookbook.rl.run"):
        outcome, _, _ = await resumed_offline(settings, chosen_differently)
    assert outcome.ok
    assert f"Resuming with changed settings ({setting})" in caplog.text


async def test_a_resume_from_a_record_made_before_tokenizer_and_renderer_reports_no_change(
    settings, caplog
):
    def forget(manifest):
        training = manifest["contract"]["training"]
        # as a run before they were kept
        for name in ("tokenizer", "renderer", "renderer_source", "renderer_class"):
            del training[name]

    with caplog.at_level("WARNING", logger="loops_cookbook.rl.run"):
        outcome, _, _ = await resumed_offline(settings, forget)
    assert outcome.ok
    assert "changed settings" not in caplog.text
