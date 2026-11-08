"""CLI contract: outcomes, actionable errors, dotenv and import-path behaviour."""

import io
import logging
import os
import sys

import pytest
from support.configs import MINIMAL, write_toml

from loops_cookbook import command
from loops_cookbook.cli import load_environment, make_config_dir_importable, settings
from loops_cookbook.rl.config import Config


def test_third_party_chatter_is_dropped_but_never_its_warnings():
    """Harbor sets every per-trial logger to DEBUG, so the handler has to filter."""
    from loops_cookbook.cli import ThirdPartyChatter

    keep = ThirdPartyChatter().filter

    def record(name, level):
        return logging.LogRecord(name, level, "", 0, "message", None, None)

    assert not keep(record("harbor.trial.trial.lr-abc", logging.DEBUG))
    assert not keep(record("httpx", logging.INFO))
    # A third party reporting a real problem is still worth seeing.
    assert keep(record("harbor.trial.trial.lr-abc", logging.WARNING))
    assert keep(record("blaxel.core.sandbox", logging.ERROR))
    # Ours is never filtered, including the Blaxel environment we own.
    assert keep(record("loops_cookbook.rl.run", logging.INFO))
    assert keep(record("loops_cookbook.harbor.blaxel", logging.DEBUG))


@pytest.fixture
def minimal(tmp_path):
    """A real config: ``loops`` reads it for the recipe before it parses the flags."""
    return str(write_toml(tmp_path / "x.toml", MINIMAL))


@pytest.mark.parametrize(
    ("status", "code"), [("completed", 0), ("loops_run_ended", 75), ("aborted", 1), ("unclean", 1)]
)
def test_cli_exits_with_the_outcomes_code(minimal, monkeypatch, status, code):
    from loops_cookbook.rl.run import Outcome

    async def finished(config, *, source):
        return Outcome(status, 0, {}, None if status == "completed" else "why")

    monkeypatch.setattr("loops_cookbook.rl.recipe.train", finished)
    assert command.main(["run", minimal]) == code


def test_an_rl_config_missing_a_section_is_refused_naming_it(tmp_path, capsys):
    path = write_toml(tmp_path / "run.toml", {"recipe": "rl", "model": {"base_model": "m"}})
    with pytest.raises(SystemExit) as error:
        command.main(["run", str(path)])
    assert error.value.code == 2
    err = capsys.readouterr().err
    assert "runtime\n  Field required" in err and "rollout\n  Field required" in err


@pytest.mark.parametrize("body", ["[]", "null", "42", '"not an object"'])
def test_non_object_training_config_has_actionable_error(tmp_path, body, capsys):
    path = tmp_path / "train.json"
    path.write_text(body)
    with pytest.raises(SystemExit) as error:
        command.main(["check", str(path)])
    assert error.value.code == 2
    assert "valid dictionary" in capsys.readouterr().err


def test_dotenv_beside_config_is_loaded_without_overriding_the_shell(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("LOOPSRL_TEST_FROM_FILE=file\nLOOPSRL_TEST_SHELL=file\n")
    monkeypatch.delenv("LOOPSRL_TEST_FROM_FILE", raising=False)
    monkeypatch.setenv("LOOPSRL_TEST_SHELL", "shell")
    monkeypatch.chdir(tmp_path / "..")
    load_environment(tmp_path)
    assert os.environ["LOOPSRL_TEST_FROM_FILE"] == "file"
    assert os.environ["LOOPSRL_TEST_SHELL"] == "shell"
    monkeypatch.delenv("LOOPSRL_TEST_FROM_FILE")


def test_config_directory_is_importable_for_agent_modules(tmp_path):
    make_config_dir_importable((tmp_path / "nested").resolve())
    assert str((tmp_path / "nested").resolve()) in sys.path
    sys.path.remove(str((tmp_path / "nested").resolve()))


@pytest.mark.parametrize("name", ["check", "run"])
def test_help_renders_for_every_command(name, capsys):
    """argparse %-formats help, so a description mentioning a percentage crashed it."""
    with pytest.raises(SystemExit) as exited:
        command.main([name, "--help"])
    assert exited.value.code == 0
    rendered = capsys.readouterr().out
    for path, _ in settings(Config):
        if path == "recipe":
            continue
        assert f"--{path}" in rendered, path
    assert "%%" not in rendered


def test_the_top_level_help_does_not_list_the_commands_twice():
    rendered = command.build_parser().format_help()
    assert rendered.count(command.COMMANDS["run"]) == 1
    assert "commands:" in rendered, "the commands read as argparse positionals"


def test_the_cli_installs_a_console_backed_handler_on_a_terminal(monkeypatch):
    """Configuring logging is the application's job, so this is where it happens."""
    from loops_cookbook.cli import configure_logging

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    root = logging.getLogger()
    original = list(root.handlers)
    try:
        monkeypatch.setattr(sys, "stderr", Terminal())
        configure_logging(verbose=False, style="auto")
        assert getattr(root.handlers[0], "console", None) is not None
        # Redirected output gets a plain handler, so a captured log stays readable.
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        configure_logging(verbose=False, style="auto")
        assert getattr(root.handlers[0], "console", None) is None
        assert len(root.handlers) == 1, "configuring twice must not stack handlers"
    finally:
        root.handlers = original


def test_third_party_chatter_is_filtered_whichever_handler_is_installed(monkeypatch):
    from loops_cookbook.cli import configure_logging

    root = logging.getLogger()
    original = list(root.handlers)
    try:
        configure_logging(verbose=False, style="plain")
        assert root.handlers[0].filters, "the noise filter was lost"
        configure_logging(verbose=True, style="plain")
        assert not root.handlers[0].filters, "--verbose must show everything"
    finally:
        root.handlers = original


@pytest.mark.parametrize(
    "flag",
    [
        "--reuse_from_run_id",
        "--reuse-from-run-id",
        "--compute.reuse_run_id",
        "--zzzzz",
        "--rollout.max_concurent_rollouts",
    ],
)
def test_every_invalid_flag_is_still_rejected(flag, capsys, minimal):
    """The guarantee the hint must never weaken: argparse decides and argparse exits."""
    with pytest.raises(SystemExit) as raised:
        command.main(["run", minimal, flag, "1"])
    assert raised.value.code == 2


@pytest.mark.parametrize("flag", ["--reuse_from_run_id", "--reuse-from-run-id"])
def test_a_bare_setting_name_is_corrected_to_its_section_path(flag, capsys, minimal):
    """It names one setting, so this is a definite answer rather than a suggestion."""
    with pytest.raises(SystemExit):
        command.main(["run", minimal, flag, "run-1"])
    assert f"{flag} is spelled --compute.reuse_from_run_id" in capsys.readouterr().err


def test_a_near_miss_gets_a_suggestion(capsys, minimal):
    with pytest.raises(SystemExit):
        command.main(["run", minimal, "--compute.reuse_run_id", "run-1"])
    assert "did you mean --compute.reuse_from_run_id" in capsys.readouterr().err


def test_a_valid_abbreviation_is_not_reported_as_a_typo(capsys, minimal):
    """argparse accepts unambiguous prefixes, so those are not what was rejected."""
    flag = "--rollout.max_concurrent_rollout"
    with pytest.raises(SystemExit):
        command.main(["run", minimal, flag, "8", "--nope", "1"])
    err = capsys.readouterr().err
    assert flag not in err.replace(f"{flag}s", "")


@pytest.mark.parametrize(
    ("flags", "table", "shown"),
    [
        ([], {}, "Loss: cispo\n"),
        (
            [
                "--training.loss_fn",
                "cispo",
                "--training.loss_fn_config",
                '{"clip_high_threshold": 5}',
            ],
            {},
            "Loss: cispo (clip_high_threshold 5)\n",
        ),
        (
            ["--training.loss_fn", "custom"],
            {"training": {"custom_loss": {"factory": "support.losses:make_policy_gradient"}}},
            "Loss: custom (support.losses:make_policy_gradient)\n",
        ),
    ],
    ids=["default", "cispo", "custom"],
)
def test_check_makes_the_loss_and_names_it(tmp_path, monkeypatch, loops_check, flags, table, shown):
    if table:
        pytest.importorskip("torch")
    code, out, err = loops_check(_pattern_config(tmp_path, monkeypatch, table), *flags)
    assert code == 0, err
    assert shown in out


def test_check_refuses_a_custom_loss_factory_that_makes_no_function(
    tmp_path, monkeypatch, loops_check
):
    pytest.importorskip("torch")
    table = {"training": {"custom_loss": {"factory": "support.losses:not_a_function"}}}
    path = _pattern_config(tmp_path, monkeypatch, table)
    code, _, err = loops_check(path, "--training.loss_fn", "custom")
    assert code == 2
    assert "returned int, not a function" in err


def _pattern_config(tmp_path, monkeypatch, table):
    """The in-process direct pattern's config, runnable from ``tmp_path``, plus ``table``."""
    import tomllib
    from pathlib import Path

    pattern = Path(__file__).resolve().parents[3] / "patterns/custom_runtime/in_process_direct"
    monkeypatch.syspath_prepend(str(pattern))  # the episode module, beside its config
    shipped = tomllib.loads((pattern / "train.toml").read_text())
    tasks = (pattern / shipped["rollout_runtime"]["files"]["tasks"]).resolve()
    here = {"rollout_runtime": {"files": {"tasks": tasks}}}
    return write_toml(tmp_path / "train.toml", shipped, here, table)


@pytest.mark.parametrize(
    "flags,shown",
    [
        ([], "Loss aggregation: sum"),
        (["--training.loss_aggregation", "token_mean"], "Loss aggregation: token_mean"),
        (["--training.loss_aggregation", "task_mean"], "Loss aggregation: task_mean"),
        (
            ["--training.loss_aggregation", "fixed_budget", "--training.loss_token_budget", "16"],
            "Loss aggregation: fixed_budget (16 rollouts × 16 action tokens)",
        ),
    ],
)
def test_check_names_the_complete_update_reduction(
    tmp_path, monkeypatch, loops_check, flags, shown
):
    code, out, err = loops_check(_pattern_config(tmp_path, monkeypatch, {}), *flags)
    assert code == 0, err
    assert shown in out


@pytest.mark.parametrize(
    ("table", "shown"),
    [
        ({}, "Producers: main (trainable); a call that names none uses main\n"),
        (
            {
                "rollout": {
                    "producers": {"main": {"trainable": True}, "judge": {"trainable": False}}
                }
            },
            "Producers: judge (not trained), main (trainable); a call that names none uses main\n",
        ),
        (
            {"rollout": {"producers": {"solver": {"trainable": True}}}},
            (
                "Producers: solver (trainable); a call that names none uses main, which is "
                "not declared, so such a call is refused\n"
            ),
        ),
    ],
    ids=["default", "with_main", "without_main"],
)
def test_check_names_the_producers_and_the_default_a_call_that_names_none_uses(
    tmp_path, monkeypatch, loops_check, table, shown
):
    code, out, err = loops_check(_pattern_config(tmp_path, monkeypatch, table))
    assert code == 0, err
    assert shown in out
