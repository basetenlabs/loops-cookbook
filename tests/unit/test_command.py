"""``loops``: the config's recipe picks the command."""

import os
import sys
from pathlib import Path

import pytest
from support.configs import MINIMAL, write_toml
from support.configs import SFT_MINIMAL as SFT

from loops_cookbook import command
from loops_cookbook.project import NoRunFound, load


@pytest.fixture
def finished(monkeypatch):
    """Both recipes' training, replaced by an outcome; records which one ran."""
    from loops_cookbook.rl.run import Outcome

    ran = []

    async def rl_train(config, *, source):
        ran.append(config.recipe)
        return Outcome("completed", 0, {}, None)

    async def sft_train(config, *, source):
        ran.append(config.recipe)
        return Outcome("completed", 0, {}, None)

    monkeypatch.setattr("loops_cookbook.rl.recipe.train", rl_train)
    monkeypatch.setattr("loops_cookbook.sft.recipe.train", sft_train)
    return ran


@pytest.mark.parametrize(("body", "recipe"), [(MINIMAL, "rl"), (SFT, "sft")])
def test_the_recipe_key_picks_the_command(tmp_path, finished, body, recipe):
    path = write_toml(tmp_path / "config.toml", body)
    assert command.main(["run", str(path)]) == 0
    assert finished == [recipe]


@pytest.fixture
def bundled(tmp_path, monkeypatch):
    """An installed package's bundled tree holding examples/demo/config.toml, an empty
    working directory, and the config file each run trained with."""
    from loops_cookbook.rl.run import Outcome

    tree = tmp_path / "site-packages" / "bundled"
    (tree / "examples" / "demo").mkdir(parents=True)
    write_toml(tree / "examples" / "demo" / "config.toml", SFT)
    monkeypatch.setattr("loops_cookbook.cli.BUNDLED", tree)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    used = []

    async def sft_train(config, *, source):
        used.append(source.file)
        return Outcome("completed", 0, {}, None)

    monkeypatch.setattr("loops_cookbook.sft.recipe.train", sft_train)
    return tree, work, used


def test_an_installed_example_runs_by_its_repository_path(bundled):
    tree, _, used = bundled
    assert command.main(["run", "examples/demo/config.toml"]) == 0
    assert used == [(tree / "examples" / "demo" / "config.toml").resolve()]


def test_a_file_at_the_path_wins_over_the_installed_example(bundled):
    _, work, used = bundled
    write_toml(work / "examples" / "demo" / "config.toml", SFT)
    assert command.main(["run", "examples/demo/config.toml"]) == 0
    assert used == [(work / "examples" / "demo" / "config.toml").resolve()]


def test_a_path_out_of_the_bundled_tree_is_not_looked_up(bundled, capsys):
    tree, _, used = bundled
    write_toml(tree.parent / "outside.toml", SFT)
    with pytest.raises(SystemExit) as error:
        command.main(["run", "../outside.toml"])
    assert error.value.code == 2
    assert used == []


WAYS = ("plain", "file link", "directory link")


@pytest.fixture
def named(tmp_path, monkeypatch):
    """``named(way, tables)``: a config in ``shared/``, named the ``way`` given.

    ``shared/`` and ``project/`` each hold their own data, ``.env`` and module, each
    saying which directory it is in, so whatever the config used names its directory.
    """
    shared, project = tmp_path / "shared", tmp_path / "project"
    for directory in (shared, project):
        directory.mkdir()
        (directory / "train.jsonl").touch()
        (directory / ".env").write_text(f"LOOPS_TEST_WHERE={directory.name}\n")
        (directory / "where_module.py").write_text(f"WHERE = {directory.name!r}\n")
    # No .env in the working directory, and every trace of a test's config undone after.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOOPS_TEST_WHERE", "unset")
    monkeypatch.delenv("LOOPS_TEST_WHERE")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "where_module", raising=False)

    def name(way, tables):
        write_toml(shared / "config.toml", tables)
        if way == "plain":
            return shared / "config.toml"
        if way == "file link":
            (project / "config.toml").symlink_to(shared / "config.toml")
            return project / "config.toml"
        (project / "configs").symlink_to(shared, target_is_directory=True)
        return project / "configs" / "config.toml"

    yield name
    sys.modules.pop("where_module", None)


@pytest.mark.parametrize("way", WAYS)
def test_a_config_means_what_its_real_file_means_however_it_is_named(
    named, monkeypatch, tmp_path, way
):
    from loops_cookbook.sft.run import Outcome

    path = named(way, SFT)
    seen = {}

    async def sft_train(config, *, source):
        import where_module

        seen.update(
            data=config.data.train,
            env=os.environ["LOOPS_TEST_WHERE"],
            module=where_module.WHERE,
            directory=source.directory,
        )
        return Outcome("completed", 0)

    monkeypatch.setattr("loops_cookbook.sft.recipe.train", sft_train)
    assert command.main(["run", str(path)]) == 0
    shared = (tmp_path / "shared").resolve()
    assert seen == {
        "data": str(shared / "train.jsonl"),
        "env": "shared",
        "module": "shared",
        "directory": shared,
    }


@pytest.mark.parametrize("way", WAYS)
def test_a_libraries_env_file_is_beside_the_real_config(named, tmp_path, way):
    project = load(named(way, SFT), env_file=".env")
    shared = (tmp_path / "shared").resolve()
    assert os.environ["LOOPS_TEST_WHERE"] == "shared"
    assert project.source.environment == (shared / ".env",)


def test_a_link_repointed_after_loading_moves_nothing(named, tmp_path):
    path = named("file link", SFT)
    project = load(path)
    path.unlink()
    write_toml(tmp_path / "project" / "other.toml", SFT)
    path.symlink_to(tmp_path / "project" / "other.toml")
    shared = (tmp_path / "shared").resolve()
    # What the recipe is handed, and what the check and the run's log say.
    assert project.source.directory == shared
    assert project.describe()[0] == f"Config: {path} -> {shared / 'config.toml'}"


@pytest.mark.parametrize("way", WAYS)
def test_an_rl_runtimes_config_dir_is_beside_the_real_config(named, monkeypatch, tmp_path, way):
    from loops_cookbook.rl import recipe

    given = []

    def assemble(config, *, config_dir, adapters):
        given.append(config_dir)
        raise ValueError("assembled")

    monkeypatch.setattr(recipe, "assemble", assemble)
    with pytest.raises(SystemExit):
        command.main(["run", str(named(way, MINIMAL))])
    assert given == [(tmp_path / "shared").resolve()]


@pytest.mark.parametrize("way", WAYS)
def test_check_first_says_what_the_config_used(named, monkeypatch, capsys, tmp_path, way):
    from loops_cookbook.sft import recipe

    class Report:
        def describe(self):
            return ["checked"]

    async def check(config):
        return Report()

    monkeypatch.setattr(recipe, "check", check)
    path = named(way, SFT)
    assert command.main(["check", str(path)]) == 0
    shared = (tmp_path / "shared").resolve()
    config = shared / "config.toml"
    assert capsys.readouterr().out.splitlines() == [
        f"Config: {path}" if way == "plain" else f"Config: {path} -> {config}",
        f"  paths, modules: beside {shared}",
        f"  .env: {shared / '.env'}",
        # The output directory is the one path in a config that is the shell's.
        f"  runs: {tmp_path.resolve() / 'runs'}",
        "checked",
    ]


@pytest.mark.parametrize(
    ("tables", "message"),
    [
        ({"recipe": "dpo"}, "'dpo' is not a recipe"),
        ({"recipe": ["rl"]}, "['rl'] is not a recipe"),
        ({"recipe": {"name": "rl"}}, "{'name': 'rl'} is not a recipe"),
        ({"model": {"base_model": "m"}}, "names no recipe"),
    ],
)
def test_a_config_whose_recipe_is_unknown_or_unstated_is_refused(tmp_path, capsys, tables, message):
    path = write_toml(tmp_path / "config.toml", tables)
    with pytest.raises(SystemExit) as exited:
        command.main(["check", str(path)])
    assert exited.value.code == 2
    assert message in capsys.readouterr().err


def test_help_lists_every_recipes_settings_and_needs_no_config(capsys):
    with pytest.raises(SystemExit) as exited:
        command.main(["run", "--help"])
    assert exited.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    # A setting only one recipe has says so, and a required one says so.
    assert "--runtime.factory TEXT (RL)" in output and "--data.train TEXT (SFT)" in output
    assert "Adam learning rate; under SFT's lr_schedule, its peak. Required." in output
    assert "(default: auto)" in output  # console: both recipes agree
    # The file's recipe key chooses the recipe, so it is no flag.
    assert "--recipe" not in output


def test_a_flag_for_the_other_recipes_setting_names_its_recipe(tmp_path, capsys):
    path = write_toml(tmp_path / "config.toml", SFT)
    with pytest.raises(SystemExit) as exited:
        command.main(["check", str(path), "--runtime.factory", "m:create"])
    assert exited.value.code == 2
    assert (
        f'--runtime.factory is an RL setting; {path} is recipe = "sft"' in capsys.readouterr().err
    )


def test_loops_check_prints_the_sft_report(tmp_path, monkeypatch, capsys):
    from loops_cookbook.sft import recipe

    class Report:
        def describe(self):
            return ["checked"]

    async def check(config):
        return Report()

    monkeypatch.setattr(recipe, "check", check)
    path = write_toml(tmp_path / "config.toml", SFT)
    assert command.main(["check", str(path)]) == 0
    assert capsys.readouterr().out.endswith("  runs: " + str(Path.cwd() / "runs") + "\nchecked\n")


def test_a_failed_rl_check_still_says_whether_loops_serves_the_model(tmp_path, monkeypatch, capsys):
    from loops_cookbook.rl import recipe

    async def served(model):
        return "Loops serves m"

    def assemble(*args, **kwargs):
        raise ValueError("runtime refused")

    monkeypatch.setattr(recipe, "check_served_model", served)
    monkeypatch.setattr(recipe, "assemble", assemble)
    path = write_toml(tmp_path / "config.toml", MINIMAL)
    with pytest.raises(SystemExit) as exited:
        command.main(["check", str(path)])
    assert exited.value.code == 2
    assert capsys.readouterr().err == "runtime refused\nLoops serves m\n"


def _runs(tmp_path, *runs):
    """Run directories under ``tmp_path/runs``, each ``(name, trained, manifest)``: when
    it last recorded an update (``None``: never), and whether it kept a manifest."""
    for name, trained, manifest in runs:
        directory = tmp_path / "runs" / name
        (directory / "checkpoints").mkdir(parents=True)
        written = []
        if trained is not None:
            written.append(directory / "steps.jsonl")
        if manifest:
            written.append(directory / "checkpoints" / "step-3.json")
        for path in written:
            path.write_text("{}")
            os.utime(path, (trained, trained))
    return tmp_path / "runs"


@pytest.fixture
def resumed(monkeypatch):
    """SFT training, replaced by a record of the ``resume_from`` it was given."""
    from loops_cookbook.sft.run import Outcome

    given = []

    async def sft_train(config, *, source):
        given.append(config.training.resume_from)
        return Outcome("completed", 0)

    monkeypatch.setattr("loops_cookbook.sft.recipe.train", sft_train)
    return given


def test_resume_continues_the_run_that_trained_last_not_the_one_started_last(tmp_path, resumed):
    runs = _runs(
        tmp_path,
        ("a-100-0000000a", 300, True),  # started first, resumed most recently
        ("a-200-0000000b", 200, True),
        ("a-300-0000000c", None, False),  # started last, never trained
        ("a-sweep-400-0000000d", 500, True),  # another run name that starts with this one
    )
    path = tmp_path / "config.toml"
    write_toml(path, SFT, {"output": {"run_name": "a", "output_dir": str(runs)}})
    assert command.main(["run", str(path), "--resume"]) == 0
    assert resumed == [str(runs / "a-100-0000000a")]


@pytest.mark.parametrize(
    ("runs", "message"),
    [
        ([("a-100-0000000a", None, False)], "--resume found no run"),
        # The latest run kept none; the older one that did is not taken in its place.
        (
            [("a-100-0000000a", 100, True), ("a-200-0000000b", 200, False)],
            "a-200-0000000b, kept no resume manifest",
        ),
    ],
    ids=["no run trained", "the latest run kept no manifest"],
)
def test_resume_with_no_run_to_resume_starts_nothing(tmp_path, resumed, capsys, runs, message):
    directory = _runs(tmp_path, *runs)
    path = tmp_path / "config.toml"
    write_toml(path, SFT, {"output": {"run_name": "a", "output_dir": str(directory)}})
    with pytest.raises(SystemExit) as exited:
        command.main(["run", str(path), "--resume"])
    assert exited.value.code == 2
    assert message in capsys.readouterr().err
    assert resumed == []


def test_a_project_says_why_it_cannot_resume_in_its_own_terms(tmp_path):
    """Not in the flags of the command that asked: a script calling it has none."""
    path = tmp_path / "config.toml"
    write_toml(path, SFT, {"output": {"run_name": "a", "output_dir": str(tmp_path / "runs")}})
    with pytest.raises(NoRunFound) as raised:
        load(path).resume_latest()
    assert raised.value.run_name == "a" and "--" not in str(raised.value)


def test_resume_and_resume_from_together_are_refused(tmp_path, resumed, capsys):
    _runs(tmp_path, ("a-100-0000000a", 100, True))
    path = tmp_path / "config.toml"
    write_toml(path, SFT, {"output": {"run_name": "a", "output_dir": str(tmp_path / "runs")}})
    with pytest.raises(SystemExit) as exited:
        command.main(["run", str(path), "--resume", "--training.resume_from", "elsewhere"])
    assert exited.value.code == 2
    assert "give one or the other" in capsys.readouterr().err
    assert resumed == []


@pytest.fixture
def evaluated(monkeypatch):
    """RL's after-the-fact eval, replaced by a record of what it was asked; it reports
    every set but ``missed``."""
    asked, missed = [], set()

    async def evaluate(config, directory, step, *, sets, source):
        asked.append((directory, step, list(sets)))
        return [name for name in sets or ["val", "hard"] if name not in missed]

    monkeypatch.setattr("loops_cookbook.rl.recipe.evaluate", evaluate)
    return asked, missed


def _rl_with_sets(tmp_path, runs):
    path = tmp_path / "config.toml"
    sets = {name: {"runtime": {"files": {"tasks": f"{name}.jsonl"}}} for name in ("val", "hard")}
    write_toml(
        path,
        MINIMAL,
        {
            "runtime": {"files": {"tasks": "train.jsonl"}},
            "output": {"run_name": "a", "output_dir": str(runs)},
            "eval": {"sets": sets},
        },
    )
    return path


@pytest.mark.parametrize(
    ("argv", "step", "sets", "missed", "code"),
    [
        (["--step", "29"], 29, [], set(), 0),
        (["--step", "base", "--set", "hard"], -1, ["hard"], set(), 0),
        (["--step", "3"], 3, [], {"hard"}, 1),
    ],
    ids=["every set", "base, one set", "a set that did not evaluate"],
)
def test_eval_runs_the_sets_on_a_step_of_the_latest_run(
    tmp_path, evaluated, argv, step, sets, missed, code
):
    asked, evaluated_missed = evaluated
    evaluated_missed.update(missed)
    runs = _runs(tmp_path, ("a-100-0000000a", 300, True), ("a-200-0000000b", 200, True))
    path = _rl_with_sets(tmp_path, runs)
    assert command.main(["eval", str(path), *argv]) == code
    assert asked == [(runs / "a-100-0000000a", step, sets)]


def test_eval_takes_the_run_named(tmp_path, evaluated):
    asked, _ = evaluated
    runs = _runs(tmp_path, ("a-100-0000000a", 300, True), ("a-200-0000000b", 200, True))
    path = _rl_with_sets(tmp_path, runs)
    assert (
        command.main(["eval", str(path), "--step", "1", "--run", str(runs / "a-200-0000000b")]) == 0
    )
    assert asked == [((runs / "a-200-0000000b").resolve(), 1, [])]


@pytest.mark.parametrize(
    ("config", "argv", "message"),
    [
        ("rl", ["--step", "x"], "'x' is not an update number or 'base'"),
        ("rl", ["--step", "-2"], "an update number is 0 or more"),
        ("rl", ["--step", "1", "--run", "nowhere"], "--run nowhere: no such run directory"),
        ("none", ["--step", "1"], "loops eval found no run"),
        ("sft", ["--step", "1"], "loops eval runs an RL config's [eval.sets]"),
    ],
    ids=["not a step", "negative", "no such run", "no run trained", "sft"],
)
def test_an_eval_that_cannot_run_starts_nothing(tmp_path, evaluated, capsys, config, argv, message):
    asked, _ = evaluated
    runs = _runs(tmp_path, ("a-100-0000000a", 300, True))
    if config == "sft":
        path = tmp_path / "config.toml"
        write_toml(path, SFT, {"output": {"run_name": "a", "output_dir": str(runs)}})
    else:
        path = _rl_with_sets(tmp_path, runs if config == "rl" else tmp_path / "empty")
    with pytest.raises(SystemExit) as exited:
        command.main(["eval", str(path), *argv])
    assert exited.value.code == 2
    assert message in capsys.readouterr().err
    assert asked == []
