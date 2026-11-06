"""Where a run executes: ``[compute] scope``, stated by every config, and the settings
that size a dedicated deployment, which a serverless run is refused."""

from __future__ import annotations

import argparse

import pytest
from pydantic import ValidationError
from support.configs import MINIMAL, SFT_MINIMAL, merged, write_toml

from loops_cookbook import command
from loops_cookbook.cli import add_setting_flags
from loops_cookbook.docs import render_knob_glossary
from loops_cookbook.rl.config import Config as RlConfig
from loops_cookbook.settings import is_dedicated_only
from loops_cookbook.sft.config import Config as SftConfig

RECIPES = {"rl": (RlConfig, MINIMAL), "sft": (SftConfig, SFT_MINIMAL)}

# A value that differs from each dedicated-only setting's default.
SET_TO = {
    "spot_job": True,
    "reuse_from_run_id": "warm-run",
    "trainer_replicas": 2,
    "sampler_replicas": 2,
}


def compute(recipe: str, **settings):
    """The recipe's ``[compute]`` from exactly ``settings``, so a test can leave scope out."""
    model, minimal = RECIPES[recipe]
    return model.model_validate({**minimal, "compute": settings}).compute


def dedicated_only(recipe: str) -> list[str]:
    model = RECIPES[recipe][0].model_fields["compute"].annotation
    return [name for name, info in model.model_fields.items() if is_dedicated_only(info)]


def test_the_dedicated_only_settings_are_the_ones_that_size_a_deployment():
    assert dedicated_only("rl") == [
        "spot_job",
        "reuse_from_run_id",
        "trainer_replicas",
        "sampler_replicas",
    ]
    assert dedicated_only("sft") == ["spot_job", "reuse_from_run_id", "trainer_replicas"]


@pytest.mark.parametrize("recipe", RECIPES)
def test_every_config_states_its_scope(recipe):
    with pytest.raises(ValidationError, match=r"compute\.scope\n  Field required"):
        compute(recipe)


@pytest.mark.parametrize(
    ("recipe", "name"),
    [(recipe, name) for recipe in RECIPES for name in dedicated_only(recipe)],
)
def test_a_serverless_run_refuses_each_dedicated_only_setting(recipe, name):
    """Loops ignores them on serverless capacity, so they are refused, never dropped."""
    with pytest.raises(ValidationError, match=f"{name} configure the run's own deployments"):
        compute(recipe, scope="serverless", **{name: SET_TO[name]})
    assert getattr(compute(recipe, scope="dedicated", **{name: SET_TO[name]}), name) == SET_TO[name]


@pytest.mark.parametrize(
    ("recipe", "name"),
    [(recipe, name) for recipe in RECIPES for name in dedicated_only(recipe)],
)
def test_a_dedicated_only_setting_at_its_default_is_accepted_on_serverless(recipe, name):
    """A saved config writes every default, and must load back as the config it saved."""
    model = RECIPES[recipe][0].model_fields["compute"].annotation
    default = model.model_fields[name].get_default(call_default_factory=True)
    assert getattr(compute(recipe, scope="serverless", **{name: default}), name) == default


@pytest.mark.parametrize("recipe", RECIPES)
def test_a_serverless_run_keeps_the_settings_both_scopes_share(recipe):
    assert compute(recipe, scope="serverless", keep_run=False).keep_run is False


def test_the_glossary_and_the_flags_say_which_settings_are_dedicated_only():
    glossary = render_knob_glossary().splitlines()
    parser = argparse.ArgumentParser()
    add_setting_flags(parser, {"rl": RlConfig, "sft": SftConfig})
    help_by_flag = {action.option_strings[0]: action.help for action in parser._actions[1:]}
    for name in ("spot_job", "trainer_replicas", "scope", "keep_run", "sampler_replica_affinity"):
        rows = [line for line in glossary if line.startswith(f"| `{name}` |")]
        marked = name in dedicated_only("rl")
        assert rows and all(row.endswith("Dedicated only. |") is marked for row in rows), name
        assert help_by_flag[f"--compute.{name}"].count("Dedicated only.") == int(marked), name


@pytest.mark.parametrize("recipe", RECIPES)
def test_a_dedicated_only_flag_on_a_serverless_config_is_refused(recipe, tmp_path, capsys):
    """An override is as much a declaration as the file: Loops would ignore it either way."""
    body = merged(RECIPES[recipe][1], {"compute": {"scope": "serverless"}})
    path = write_toml(tmp_path / "config.toml", body)
    with pytest.raises(SystemExit) as exited:
        command.main(["check", str(path), "--compute.spot_job", "true"])
    assert exited.value.code == 2
    assert "spot_job configure the run's own deployments" in capsys.readouterr().err
