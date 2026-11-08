"""Public configuration contract: nested sections, dotted overrides, validation."""

import argparse
import dataclasses
import json
import re

import pytest
from baseten.loops.models import AdamParams
from pydantic import ValidationError
from support.builders import REQUIRED, config_values, make_config
from support.configs import MINIMAL, write_toml

from loops_cookbook import command
from loops_cookbook.cli import flag_overrides, settings
from loops_cookbook.rl import cli
from loops_cookbook.rl.config import Config


class Checked:
    """A check's report that says nothing; these tests read the config it was given."""

    def describe(self) -> list[str]:
        return []


def _build_config(args: argparse.Namespace) -> Config:
    """The config ``loops`` builds from a parsed command line: the file, then its flags."""
    return cli.load_training_config(args.config, flag_overrides(args, Config))


@pytest.fixture
def checked(monkeypatch):
    """``checked(path, *flags)``: the config ``loops check`` validates from a file and flags."""
    seen = []

    async def check(config, source):
        seen.append(config)
        return Checked()

    monkeypatch.setitem(command.RECIPES, "rl", dataclasses.replace(cli.RECIPE, check=check))

    def run(path, *flags):
        assert command.main(["check", str(path), *flags]) == 0
        return seen.pop()

    return run


def write(tmp_path, tables=None, name="train.toml"):
    """MINIMAL, with ``tables`` merged in, as ``train.toml`` in ``tmp_path``."""
    return write_toml(tmp_path / name, MINIMAL, tables or {})


def test_training_does_not_own_agent_behavior():
    for forbidden in (
        "tasks_dir",
        "max_turns",
        "tool_output_max_chars",
        "environment_type",
        "system_prompt",
    ):
        values = config_values()
        values["rollout_runtime"][forbidden] = "anything"
        with pytest.raises(ValidationError, match="Extra inputs"):
            Config.model_validate(values)


def test_runtime_configuration_reaches_the_model_unchanged():
    opaque = {"path": "/bin/bash", "endpoint": "/v1/chat", "api_key_file": "x", "n": [1, {"a": 2}]}
    config = make_config(**{"rollout_runtime.config": opaque})
    assert config.rollout_runtime.config == opaque


def test_every_setting_has_one_dotted_path():
    paths = [path for path, _ in settings(Config)]
    assert len(paths) == len(set(paths))
    assert "rollout.grading.timeout_seconds" in paths
    assert "rollout.cleanup.cancellation_grace_seconds" in paths


@pytest.mark.parametrize(
    "tables",
    [
        {"unknown": {"x": 1}},
        {"harbor": {"harbor_config": "job.yaml"}},
        {"training": {"base_model": "m"}},
        {"sampling": {"typo": 1}},
        {"rollout": {"max_concurrent_resources": 8}},
        {"resources": {"state": "x"}},
    ],
)
def test_unknown_or_misplaced_settings_are_rejected(tmp_path, tables):
    path = write(tmp_path, tables)
    with pytest.raises(ValidationError, match="Extra inputs"):
        cli.load_training_config(path)


def test_a_section_must_be_a_table(tmp_path):
    path = write(tmp_path, {"model": "m"})
    with pytest.raises(ValidationError, match="model"):
        cli.load_training_config(path)


@pytest.mark.parametrize(
    "missing",
    [
        "rollout.setup_timeout_seconds",
        "rollout.grading.timeout_seconds",
        "rollout_runtime.factory",
        "output.run_name",
    ],
)
def test_deadlines_the_factory_and_the_run_name_are_mandatory(missing):
    values = config_values()
    *sections, leaf = missing.split(".")
    target = values
    for section in sections:
        target = target[section]
    del target[leaf]
    with pytest.raises(ValidationError, match=leaf):
        Config.model_validate(values)


@pytest.mark.parametrize(
    "setting",
    [
        "rollout.rollout_timeout_seconds",
        "rollout.grading.timeout_seconds",
        "rollout.cleanup.cancellation_grace_seconds",
        "rollout.cleanup.process_kill_grace_seconds",
    ],
)
def test_deadlines_are_positive(setting):
    with pytest.raises(ValidationError):
        make_config(**{setting: 0})


def test_a_wandb_project_name_wandb_would_refuse_is_refused_at_load():
    """`wandb.init` refuses it only once the run trains, and the mirror is best effort."""
    assert make_config(wandb_project="p" * 128).output.wandb_project == "p" * 128
    with pytest.raises(ValidationError, match="wandb_project"):
        make_config(wandb_project="p" * 129)


def test_an_optimizer_to_start_from_needs_the_weights_it_comes_with():
    with pytest.raises(ValidationError, match="set init_from"):
        make_config(init_from_optimizer=True)


def test_there_is_no_separate_live_episode_limit():
    """Live attempts are bounded as on develop: sampling slots and admission."""
    with pytest.raises(ValidationError):
        make_config(**{"rollout.max_live_episodes": 8})


def test_with_no_producer_table_there_is_one_trainable_producer_main(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config_values()))
    producers = cli.load_training_config(path).rollout.producers
    assert {name: spec.trainable for name, spec in producers.items()} == {"main": True}


def test_the_producer_table_declares_each_producer_and_whether_it_trains(tmp_path):
    producers = {"main": True, "critic": True, "judge": False}
    path = write(
        tmp_path,
        {"rollout": {"producers": {name: {"trainable": t} for name, t in producers.items()}}},
    )
    producers = cli.load_training_config(path).rollout.producers
    assert {name: spec.trainable for name, spec in producers.items()} == {
        "main": True,
        "critic": True,
        "judge": False,
    }


@pytest.mark.parametrize(
    ("producers", "error"),
    [
        ({}, "at least one trainable"),
        ({"judge": {"trainable": False}}, "at least one trainable"),
        ({"main": {"trainable": True}, "": {"trainable": False}}, "nonempty"),
        ({"main": {}}, "trainable"),
        ({"main": {"trainable": True, "weight": 2}}, "weight"),
    ],
)
def test_a_producer_table_is_refused_at_load_when_it_cannot_train(producers, error):
    with pytest.raises(ValidationError, match=error):
        make_config(**{"rollout.producers": producers})


@pytest.mark.parametrize("old", ["default_producer", "allowed_producers", "trainable_producers"])
def test_the_old_producer_settings_are_gone(old):
    with pytest.raises(ValidationError, match=old):
        make_config(**{f"rollout.{old}": ["main"]})


def test_model_access_is_unset_direct_or_openai_http():
    assert make_config().rollout_runtime.model_access is None  # the runtime declares it
    assert make_config(model_access="openai_http").rollout_runtime.model_access == "openai_http"
    with pytest.raises(ValidationError, match="model_access"):
        make_config(model_access="grpc")


def test_the_gateway_settings_default_to_loopback_and_apply_only_to_http(tmp_path):
    from loops_cookbook.rl.cli import load_training_config

    openai = {"advertised_base_url": "http://10.0.0.2:9000", "model_name": "policy"}
    path = write(tmp_path, {"rollout_runtime": {"openai": openai}})
    with pytest.raises(ValidationError, match="only when model_access"):
        load_training_config(path, {"rollout_runtime.model_access": "direct"})
    config = load_training_config(path)
    assert config.rollout_runtime.openai is not None
    assert config.rollout_runtime.openai.bind == "127.0.0.1:0"
    assert config.rollout_runtime.openai.advertised_base_url == "http://10.0.0.2:9000"
    with pytest.raises(ValidationError):
        make_config(
            **{
                "rollout_runtime.model_access": "openai_http",
                "rollout_runtime.openai": {"bind": "nope"},
            }
        )


def test_slots_can_be_less_than_group_size():
    assert make_config(max_concurrent_rollouts=1).rollout.max_concurrent_rollouts == 1


def test_structured_overrides(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config_values()))
    result = _build_config(
        argparse.Namespace(
            config=path,
            **{
                "rollout.producers": '{"main": {"trainable": true}, "summary": {"trainable": false}}'
            },
        )
    )
    assert {name: spec.trainable for name, spec in result.rollout.producers.items()} == {
        "main": True,
        "summary": False,
    }


@pytest.mark.parametrize(
    ("table", "override", "merged"),
    [
        (
            {"rollout_runtime": {"config": {"kept": 1, "changed": 2, "nested": {"x": 1, "y": 2}}}},
            ("rollout_runtime.config", {"changed": 3, "added": 4, "nested": {"y": 5}}),
            lambda config: (
                config.rollout_runtime.config
                == {"kept": 1, "changed": 3, "added": 4, "nested": {"x": 1, "y": 5}}
            ),
        ),
        (
            {
                "providers": {
                    "blaxel": {"region": "us-pdx-1", "ttl": "1h"},
                    "fake": {"adapter": "a:b"},
                }
            },
            ("providers", {"blaxel": {"ttl": "2h"}}),
            lambda config: (
                config.providers
                == {"blaxel": {"region": "us-pdx-1", "ttl": "2h"}, "fake": {"adapter": "a:b"}}
            ),
        ),
        (
            {"rollout_runtime": {"openai": {"advertised_base_url": "http://10.0.0.2:9000"}}},
            ("rollout_runtime.openai", {"model_name": "policy"}),
            lambda config: (
                config.rollout_runtime.openai is not None
                and config.rollout_runtime.openai.advertised_base_url == "http://10.0.0.2:9000"
                and config.rollout_runtime.openai.model_name == "policy"
            ),
        ),
    ],
    ids=["runtime_config", "providers", "gateway"],
)
def test_a_table_override_merges_into_the_files_table(tmp_path, table, override, merged):
    """Each key given on the command line replaces its own; the file's other keys stay."""
    path = write(tmp_path, table)
    dotted, value = override
    result = _build_config(argparse.Namespace(config=path, **{dotted: json.dumps(value)}))
    assert merged(result)


def test_renderer_kwargs_json_cli_override(tmp_path, checked):
    config = checked(
        write(tmp_path),
        "--model.renderer_chat_template_kwargs",
        '{"reasoning_effort":"low","clear_thinking":false}',
    )
    assert config.model.renderer_chat_template_kwargs == {
        "reasoning_effort": "low",
        "clear_thinking": False,
    }


def test_renderer_options_are_json_values():
    from loops_cookbook.rl.config import ModelSettings

    nested = {"effort": "low", "budget": [1, 2.5, None], "flags": {"on": True}}
    assert (
        ModelSettings(
            base_model="m", lora_rank=32, renderer_chat_template_kwargs=nested
        ).renderer_chat_template_kwargs
        == nested
    )
    with pytest.raises(ValidationError):
        ModelSettings(
            lora_rank=32, base_model="m", renderer_chat_template_kwargs={"when": object()}
        )


def test_dotted_override_reaches_a_nested_section(tmp_path, checked):
    config = checked(
        write(tmp_path),
        "--rollout.max_concurrent_rollouts",
        "3",
        "--rollout.grading.timeout_seconds",
        "7",
    )
    assert (config.rollout.max_concurrent_rollouts, config.rollout.grading.timeout_seconds) == (
        3,
        7.0,
    )


@pytest.mark.parametrize(
    ("training", "error"),
    [
        ({"loss_fn": "dpo"}, "loss_fn"),
        (
            {"loss_fn": "importance_sampling", "loss_fn_config": {"clip_high_threshold": 5.0}},
            "Extra inputs",
        ),
        ({"loss_fn": "cispo", "loss_fn_config": {"clip_hihg_threshold": 5.0}}, "Extra inputs"),
        ({"loss_fn": "ppo", "loss_fn_config": {"clip_low_threshold": -1}}, "greater than or equal"),
        ({"loss_fn": "cispo", "loss_fn_config": {"clip_low_threshold": 1.5}}, "at least"),
        ({"loss_fn": "ppo", "loss_fn_config": {"clip_high_threshold": 0.5}}, "at least"),
        ({"loss_fn": "dro", "loss_fn_config": {"beta": -0.1}}, "greater than or equal"),
        ({"loss_fn": "dppo", "loss_fn_config": {"dppo_mask_low": -0.1}}, "greater than or equal"),
        ({"loss_fn": "custom"}, "needs a \\[training.custom_loss\\] table"),
        ({"loss_fn": "cispo", "custom_loss": {"factory": "m:f"}}, 'only when loss_fn = "custom"'),
        (
            {"loss_fn": "custom", "custom_loss": {"factory": "m:f"}, "loss_fn_config": {"a": 1}},
            "takes its settings in",
        ),
    ],
)
def test_loss_settings_belong_to_their_loss(training, error):
    values = config_values()
    values["training"] = {**values["training"], **training}
    with pytest.raises(ValidationError, match=error):
        Config.model_validate(values)


@pytest.mark.parametrize(
    "training",
    [
        {"loss_fn": "ppo", "loss_fn_config": {"clip_low_threshold": 0, "clip_high_threshold": 0}},
        {"loss_fn": "cispo", "loss_fn_config": {"clip_low_threshold": 1, "clip_high_threshold": 1}},
        {"loss_fn": "dppo", "loss_fn_config": {"adv_tau": -1.0, "kl_tau": 0.01}},
    ],
)
def test_loss_settings_take_what_the_loops_trainer_takes(training):
    """The trainer's own bounds: thresholds may be 0 or equal; the taus are unbounded."""
    assert make_config(**training).training.loss_fn_config == training["loss_fn_config"]


def test_every_built_in_loss_fn_has_its_settings_declared():
    from typing import get_args

    from loops_cookbook.rl.config import LOSS_FN_CONFIGS, TrainingSettings

    choices = get_args(TrainingSettings.model_fields["loss_fn"].annotation)
    assert set(choices) - {"custom"} == set(LOSS_FN_CONFIGS)


def test_cispo_and_loops_sum_are_the_defaults():
    training = make_config().training
    assert training.loss_fn == "cispo"
    assert training.loss_aggregation == "sum"
    assert training.loss_fn_config == {}
    assert training.loss_token_budget is None


@pytest.mark.parametrize(
    "training",
    [
        {"loss_fn": "dro", "loss_aggregation": "sum"},
        {"loss_fn": "dro", "loss_aggregation": "token_mean"},
        {"loss_fn": "dppo", "loss_aggregation": "token_mean"},
        {"loss_fn": "custom", "custom_loss": {"factory": "m:f"}, "loss_aggregation": "sum"},
        {"loss_fn": "importance_sampling", "loss_aggregation": "task_mean"},
    ],
)
def test_each_loss_accepts_the_reductions_that_mean_the_same_for_it(training):
    assert make_config(**training).training.loss_aggregation == training["loss_aggregation"]


@pytest.mark.parametrize(
    "training,error",
    [
        ({"loss_aggregation": "sequence_mean"}, "loss_aggregation"),
        ({"loss_aggregation": "fixed_budget"}, "requires loss_token_budget"),
        ({"loss_token_budget": 10}, "only to fixed_budget"),
        ({"loss_aggregation": "task_mean", "loss_token_budget": 10}, "only to fixed_budget"),
        ({"loss_aggregation": "fixed_budget", "loss_token_budget": 0}, "greater than or equal"),
        ({"loss_fn": "dro", "loss_aggregation": "task_mean"}, "penalty terms"),
        (
            {"loss_fn": "dppo", "loss_aggregation": "fixed_budget", "loss_token_budget": 10},
            "penalty terms",
        ),
        (
            {
                "loss_fn": "custom",
                "custom_loss": {"factory": "m:f"},
                "loss_aggregation": "task_mean",
            },
            "reduce inside the loss function",
        ),
        (
            {
                "loss_fn": "custom",
                "custom_loss": {"factory": "m:f"},
                "loss_aggregation": "token_mean",
            },
            "reduce inside the loss function",
        ),
    ],
)
def test_invalid_aggregation_is_rejected_before_training(training, error):
    with pytest.raises(ValueError, match=error):
        make_config(**training)


def test_reuse_run_id_is_configurable_and_not_a_file_reference(tmp_path, checked):
    path = write(tmp_path, {"compute": {"reuse_from_run_id": "warm-run"}})
    config = cli.load_training_config(path)
    assert config.compute.reuse_from_run_id == "warm-run"
    assert config.training.resume_from is None
    flagged = checked(path, "--compute.reuse_from_run_id", "other")
    assert flagged.compute.reuse_from_run_id == "other"


def test_reuse_run_id_rejects_empty_values():
    with pytest.raises(ValidationError):
        make_config(reuse_from_run_id="")


def test_reuse_is_refused_for_more_than_one_trainer_replica():
    """Loops provisions fresh capacity then, so the reuse request would be ignored."""
    with pytest.raises(ValidationError, match="trainer_replicas=2"):
        make_config(reuse_from_run_id="warm-run", trainer_replicas=2)
    assert (
        make_config(reuse_from_run_id="warm-run", sampler_replicas=4).compute.sampler_replicas == 4
    )


def test_paths_resolve_against_whoever_named_them(tmp_path, monkeypatch):
    """A config's own reference travels with it; a flag means what the shell typed.

    The second half is the one that matters in practice: tab completion produces a
    path relative to the shell, and resolving that against the config file sends it
    somewhere the caller never named.
    """
    training = tmp_path / "settings"
    training.mkdir()
    path = write(
        training,
        {
            "rollout_runtime": {"files": {"job": "job.yaml"}},
            "training": {"resume_from": "checkpoints/step-0.json"},
        },
    )
    monkeypatch.chdir(tmp_path)
    config = cli.load_training_config(path)
    assert config.training.resume_from == str(training / "checkpoints" / "step-0.json")
    assert config.rollout_runtime.files["job"] == str(training / "job.yaml")
    overridden = _build_config(
        argparse.Namespace(config=path, **{"training.resume_from": "other.json"})
    )
    assert overridden.training.resume_from == str(tmp_path / "other.json")
    files = cli.load_training_config(path, {"rollout_runtime.files": {"job": "shell.yaml"}})
    assert files.rollout_runtime.files["job"] == str(tmp_path / "shell.yaml")
    absolute = cli.load_training_config(
        path, {"training.resume_from": str(tmp_path / "absolute.json")}
    )
    assert absolute.training.resume_from == str(tmp_path / "absolute.json")
    cleared = cli.load_training_config(path, {"training.resume_from": None})
    assert cleared.training.resume_from is None


@pytest.mark.parametrize("value", [42, True, [], {}])
def test_path_types_are_validated_before_path_operations(tmp_path, value):
    values = config_values(**{"training.resume_from": value})
    path = tmp_path / "train.json"
    path.write_text(json.dumps(values))
    with pytest.raises(ValidationError, match="resume_from"):
        cli.load_training_config(path)


def test_resource_settings_default_and_validate(tmp_path):
    path = write(
        tmp_path,
        {
            "resources": {"state_dir": "elsewhere"},
            "providers": {"custom": {"adapter": "my_provider:create", "zone": "eu-1"}},
        },
    )
    config = cli.load_training_config(path)
    assert config.resources.state_dir == "elsewhere"
    assert config.providers == {"custom": {"adapter": "my_provider:create", "zone": "eu-1"}}
    assert config.rollout.max_concurrent_provisioning == 8
    assert make_config().providers == {}
    with pytest.raises(ValidationError, match="max_concurrent_provisioning"):
        make_config(**{"rollout.max_concurrent_provisioning": 0})


def test_reasoning_efforts_are_a_table_of_template_overrides(tmp_path):
    path = write(
        tmp_path,
        {
            "model": {
                "reasoning_efforts": {
                    "high": {"reasoning_effort": "high"},
                    "none": {"enable_thinking": False},
                }
            }
        },
    )
    from loops_cookbook.rl.cli import load_training_config

    config = load_training_config(path)
    assert config.model.reasoning_efforts == {
        "high": {"reasoning_effort": "high"},
        "none": {"enable_thinking": False},
    }


def test_a_reasoning_effort_an_openai_client_cannot_send_is_rejected():
    from loops_cookbook.rl.config import ModelSettings

    with pytest.raises(ValidationError, match="OpenAI client can send"):
        ModelSettings(lora_rank=32, base_model="m", reasoning_efforts={"turbo": {}})


def test_the_accepted_efforts_are_the_pinned_schemas():
    import typing

    from loops_cookbook.model_access.openai.schema.models import CreateChatCompletionRequest
    from loops_cookbook.rl.config import REASONING_EFFORTS

    annotation = CreateChatCompletionRequest.model_fields["reasoning_effort"].annotation
    literals = {v for arg in typing.get_args(annotation) for v in typing.get_args(arg)}
    assert set(REASONING_EFFORTS) == literals


def test_async_needs_a_policy_version_of_slack_to_overlap_anything():
    """With 0, every rollout sampled while a step trains is stale; sync is what 0 means."""
    with pytest.raises(ValidationError, match="max_steps_off_policy >= 1"):
        make_config(mode="async", max_steps_off_policy=0)
    make_config(mode="async", max_steps_off_policy=1)


STALENESS_LIMITS = ["max_steps_off_policy", "max_policy_delta_in_trajectory"]


@pytest.mark.parametrize("limit", STALENESS_LIMITS)
def test_async_requires_each_staleness_limit(limit):
    values = config_values(mode="async")
    del values["training"][limit]
    with pytest.raises(ValidationError, match=f'mode = "async" requires {limit}'):
        Config.model_validate(values)


@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("limit", STALENESS_LIMITS)
def test_sync_refuses_a_staleness_limit_and_says_to_remove_it(limit, value):
    """0 too, as configs from before sync fixed the limits held: the fix is removal."""
    with pytest.raises(
        ValidationError, match=f'{limit} applies only to mode = "async".*Remove them'
    ):
        make_config(mode="sync", **{limit: value})


# The rollout clock is required only of a runtime that derives none, which check decides.
@pytest.mark.parametrize("path", sorted(set(REQUIRED) - {"rollout.rollout_timeout_seconds"}))
def test_every_required_setting_is_refused_when_missing(path):
    values = config_values()
    *sections, leaf = path.split(".")
    table = values
    for section in sections:
        table = table[section]
    del table[leaf]
    with pytest.raises(ValidationError, match=rf"{re.escape(path)}\n  Field required"):
        Config.model_validate(values)


def test_max_steps_0_runs_without_a_limit():
    assert make_config(max_steps=0).training.step_limit is None
    assert make_config(max_steps=3).training.step_limit == 3


def test_an_unset_optimizer_setting_is_left_to_loops():
    assert make_config().training.adam_params(1e-5) == AdamParams(learning_rate=1e-5)
    stated = make_config(
        adam_epsilon=1e-8, adam_beta1=0.8, adam_beta2=0.99, weight_decay=0.1, grad_clip_norm=0
    ).training.adam_params(1e-5)
    assert stated == AdamParams(
        learning_rate=1e-5, eps=1e-8, beta1=0.8, beta2=0.99, weight_decay=0.1, grad_clip_norm=0
    )


def test_async_admits_at_most_the_cap_ahead():
    """Groups admitted further ahead than the cap would be stale before their batch trains."""
    make_config(mode="async", max_steps_off_policy=4, admission_steps_ahead=1)
    make_config(mode="async", max_steps_off_policy=4, admission_steps_ahead=4)
    make_config(mode="async", max_steps_off_policy=4, admission_steps_ahead=0.5)
    with pytest.raises(ValidationError, match="must not exceed max_steps_off_policy"):
        make_config(mode="async", max_steps_off_policy=2, admission_steps_ahead=2.5)
    with pytest.raises(ValidationError, match="greater than 0"):
        make_config(mode="async", max_steps_off_policy=2, admission_steps_ahead=0)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"runtime.factory": "x:y"}, r"\[runtime\] is now \[rollout_runtime\]"),
        (
            {"eval.sets.val.runtime.files.tasks": "tasks.jsonl"},
            r"\[eval\.sets\.<name>\.runtime\] is now \[eval\.sets\.<name>\.rollout_runtime\]",
        ),
    ],
)
def test_the_old_runtime_table_is_refused_naming_its_new_name(override, message):
    """An old config's [runtime] would otherwise read as a typo, not as a rename."""
    with pytest.raises(ValidationError, match=message):
        make_config(**override)


def test_admission_headroom_is_refused_saying_what_replaced_it():
    """A config that set it reads as a typo otherwise, not as a knob that moved."""
    with pytest.raises(ValidationError, match="admission_headroom was removed"):
        make_config(**{"training.admission_headroom": 1})


def test_admission_steps_ahead_is_refused_in_sync_mode():
    """Sync has no window to size, so setting one is a misconfiguration, not a no-op."""
    with pytest.raises(ValidationError, match='applies only to mode = "async"'):
        make_config(mode="sync", max_steps_off_policy=3, admission_steps_ahead=1)


def test_cleanup_deadlines_default_to_values_every_workload_can_keep():
    cleanup = Config.model_validate(config_values()).rollout.cleanup
    assert (
        cleanup.timeout_seconds,
        cleanup.cancellation_grace_seconds,
        cleanup.process_kill_grace_seconds,
    ) == (600, 5, 10)


def test_the_tokenizer_defaults_to_the_base_model():
    model = make_config().model
    assert model.tokenizer is None
    assert model.tokenizer_source == model.base_model


@pytest.mark.parametrize("tokenizer", ["org/tokenizer-repo", "DIR", "DIR/."])
def test_the_tokenizer_names_a_repo_or_a_directory(tmp_path, tokenizer):
    tokenizer = tokenizer.replace("DIR", str(tmp_path))
    model = make_config(base_model="org/served", tokenizer=tokenizer).model
    assert (model.base_model, model.tokenizer_source) == ("org/served", tokenizer)


@pytest.mark.parametrize(
    ("tokenizer", "match"),
    [
        ("", "tokenizer"),
        ("./no-such-directory", "./no-such-directory is not a directory"),
        ("DIR/missing", "is not a directory"),
        ("DIR/file", "is not a directory"),
        ("~/no-such-directory-for-a-tokenizer", "is not a directory"),
    ],
)
def test_a_tokenizer_that_is_empty_or_names_no_directory_is_refused(tmp_path, tokenizer, match):
    (tmp_path / "file").write_text("")
    with pytest.raises(ValidationError, match=match):
        make_config(tokenizer=tokenizer.replace("DIR", str(tmp_path)))


def test_the_run_renders_with_the_configured_tokenizer(tmp_path):
    from support.local_tokenizer import save_tokenizer

    from loops_cookbook.rl.recipe import configured_renderer

    source = str(save_tokenizer(tmp_path))
    config = make_config(
        base_model="org/no-such-repo",
        tokenizer=source,
        renderer=f"{__name__}:TokenizerRenderer",
    )
    assert configured_renderer(config).tokenizer.name_or_path == source


def test_sampler_replica_affinity_defaults_on(config):
    """A routing hint Loops serves without where it cannot pin, so it costs nothing on."""
    assert config().compute.sampler_replica_affinity is True


def test_unusable_group_limit(config):
    assert config().training.max_consecutive_unusable_groups == 16
    limited = config(max_consecutive_unusable_groups=0)
    assert limited.training.max_consecutive_unusable_groups == 0
    with pytest.raises(ValidationError):
        config(max_consecutive_unusable_groups=-1)


class TokenizerRenderer:
    """A renderer a config names: keeps the tokenizer it was built with; renders nothing."""

    stop_sequences = ()

    def __init__(self, tokenizer, chat_template_kwargs):
        self.tokenizer = tokenizer

    def render(self, messages, tools=()):
        raise NotImplementedError

    def parse(self, prompt_ids, completion_ids, tools=()):
        raise NotImplementedError


def test_an_eval_set_names_files_the_runtime_already_has(tmp_path):
    path = write(
        tmp_path,
        {
            "rollout_runtime": {"files": {"tasks": "train.jsonl"}},
            "eval": {
                "sets": {
                    "val": {
                        "rollout_runtime": {"files": {"tasks": "val.jsonl"}},
                        "samples_per_task": 2,
                    }
                }
            },
        },
    )
    config = cli.load_training_config(path)
    # Resolved against the config that named it, as training's own files are.
    assert config.eval.sets["val"].rollout_runtime.files == {
        "tasks": str(path.parent / "val.jsonl")
    }
    assert config.rollout_runtime.files["tasks"] == str(path.parent / "train.jsonl")
    assert config.eval.sets["val"].samples_per_task == 2


def test_an_eval_set_naming_a_file_the_runtime_lacks_is_refused():
    with pytest.raises(ValidationError) as error:
        make_config(
            **{
                "rollout_runtime.files": {"tasks": "train.jsonl", "job": "harbor.yaml"},
                "eval.sets": {"val": {"rollout_runtime": {"files": {"taks": "val.jsonl"}}}},
            }
        )
    assert "eval set 'val' names file(s) taks, which [rollout_runtime.files] does not have" in str(
        error.value
    )
    assert "(job, tasks)" in str(error.value)


@pytest.mark.parametrize("name", ["Val", "a/b", "", "-val"])
def test_an_eval_set_name_must_fit_a_path_and_a_metric_name(name):
    with pytest.raises(ValidationError, match="eval set names"):
        make_config(
            **{
                "rollout_runtime.files": {"tasks": "train.jsonl"},
                "eval.sets": {name: {"rollout_runtime": {"files": {"tasks": "val.jsonl"}}}},
            }
        )


@pytest.mark.parametrize(
    ("has_images", "replay", "refused"),
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_replay_is_refused_for_a_runtime_whose_prompts_carry_images(has_images, replay, refused):
    """Replay records every generation, and Loops records text prompts only."""
    values = {"has_images": has_images, "router_replay": replay}
    if refused:
        with pytest.raises(ValidationError, match="cannot record image prompts"):
            make_config(**values)
    else:
        assert make_config(**values).rollout_runtime.has_images is has_images
