"""Train with Baseten Loops, using one command for every recipe.

`loops check <config>` checks a config and launches nothing. `loops run <config>` trains
with it, and `loops run <config> --resume` continues the latest run of that config. The
config names its recipe with `recipe = "rl"` or `recipe = "sft"`. `loops eval <config>
--step N` runs an RL config's evals on update N of its latest run, after the fact.
`loops cleanup` releases the provider resources that a stopped RL run left behind.
`loops run` leaves its Loops run up on exit unless `[compute].keep_run` is false, and
`loops deactivate` ends a run that was left up.

You can override every setting with a flag that names its dotted section path, such as
`--training.learning_rate 2e-4` or `--rollout.grading.timeout_seconds 600`. A table or
list setting takes JSON. A table merges into the file's table. Each key it names
replaces that key, a nested table merges the same way, and every other key in the file
is kept. A list replaces the file's list. A flag for a setting that the config's recipe
does not have is refused. Run `loops run --help` to see every setting, grouped by the
TOML section it lives in.

The directory that holds the config is importable, so `[rollout_runtime].factory`,
`[model].renderer` and a Harbor job's agent can name modules beside it. Credentials
come from a `.env` beside the config or in the working directory, as .env.example
shows. A symlinked config is the file it points to, so its paths, modules and `.env`
are beside that file. `loops check` prints which config file, directory and `.env`
files it used.

The package ships the repository's examples/, patterns/ and tasks/. So a config path
from the repository, such as examples/counting/train.toml, also runs after
`pip install`. A file at that path in the working directory wins over the installed one.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from loops_cookbook import __version__
from loops_cookbook.cli import (
    TERMINATED_EXIT_CODE,
    ConfigSource,
    Recipe,
    Terminated,
    add_setting_flags,
    cli_settings,
    configure_logging,
    exit_code,
    flag_overrides,
    hint,
    load_environment,
    locate_config,
    make_config_dir_importable,
    run_main,
)
from loops_cookbook.project import (
    RECIPES,
    NoManifest,
    NoRunFound,
    Project,
    ResumeFromSet,
    load_as,
    recipe_of,
)
from loops_cookbook.rl import cli as rl
from loops_cookbook.rl import recipe as rl_recipe
from loops_cookbook.run_record import read_run_ids
from loops_cookbook.trainer import CredentialsError, deactivate_run

MODELS = {name.upper(): recipe.config for name, recipe in RECIPES.items()}
# These errors mean a config or its environment is wrong. The list holds every
# recipe's shared errors, then each recipe's own, so a config that turns out to be the
# other recipe's still gets a clear answer.
ERRORS = tuple(
    dict.fromkeys(
        (ValidationError, ValueError, OSError, CredentialsError)
        + tuple(error for recipe in RECIPES.values() for error in recipe.errors)
    )
)
COMMANDS = {
    "check": "Check a config against its recipe. It launches nothing.",
    "run": "Train with a config's recipe and settings.",
}
CLEANUP = "Release the provider resources that a stopped RL run left behind."
EVAL = "Run an RL config's eval sets on one update of a run, after the fact."
DEACTIVATE = "End a Loops run and its paired sampler. Its checkpoints stay."
PROG = "loops"
# A Loops run ID as Loops writes one, such as "3ydy8k3".
RUN_ID = re.compile(r"[A-Za-z0-9]+")


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for every command, with a flag for each setting of every recipe."""
    parser = argparse.ArgumentParser(
        prog=PROG, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    # The group has a title, so the commands appear under "commands:" rather than
    # argparse's default "positional arguments:".
    commands = parser.add_subparsers(
        dest="command", required=True, metavar="COMMAND", title="commands"
    )
    for name, summary in COMMANDS.items():
        child = commands.add_parser(name, help=summary, description=summary)
        child.add_argument("config", type=Path, metavar="CONFIG", help="Training TOML or JSON.")
        child.add_argument(
            "-v", "--verbose", action="store_true", help="Log at DEBUG instead of INFO."
        )
        child.add_argument(
            "--resume",
            action="store_true",
            help="Resume the latest run of this config. That is the "
            "<output_dir>/<run_name>-<time>-<id> that trained most recently, resumed from "
            "the manifest it wrote last. Exits 2 when no run has trained, or when that run "
            "kept no manifest. To name one run instead, use --training.resume_from.",
        )
        add_setting_flags(child, MODELS)
    evaluate = commands.add_parser("eval", help=EVAL, description=EVAL_DESCRIPTION)
    evaluate.add_argument("config", type=Path, metavar="CONFIG", help="Training TOML or JSON.")
    evaluate.add_argument(
        "--step",
        required=True,
        type=_step,
        metavar="N",
        help="The update to evaluate, as the run's logs number it, or 'base' for the base model.",
    )
    evaluate.add_argument(
        "--run",
        type=Path,
        metavar="DIR",
        help="The run directory. By default, the latest run of this config, as --resume takes.",
    )
    evaluate.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="NAME",
        help="Evaluate only this eval set. Repeat the flag for more. By default, every set.",
    )
    evaluate.add_argument(
        "-v", "--verbose", action="store_true", help="Log at DEBUG instead of INFO."
    )
    add_setting_flags(evaluate, MODELS)
    rl.add_cleanup_arguments(commands.add_parser("cleanup", help=CLEANUP, description=rl.CLEANUP))
    deactivate = commands.add_parser("deactivate", help=DEACTIVATE, description=DEACTIVATE)
    deactivate.add_argument(
        "run", metavar="RUN", help="A Loops run ID, or a run directory holding run.json."
    )
    deactivate.add_argument(
        "-v", "--verbose", action="store_true", help="Log at DEBUG instead of INFO."
    )
    return parser


EVAL_DESCRIPTION = f"""{EVAL}

The config's [eval.sets] evaluate update N's weights as an eval during the run would.
Each set writes eval/<set>/step-N.jsonl and its transcripts into the run directory,
beside the run's own files, and the eval's line goes to the log. If the run already
ran an eval at N, the eval runs again and its files are replaced.

The weights are those the run kept for N. Sampler weights come first. A run keeps them
for every eval that was due, even one that was skipped or failed, and on [output]
sampler_checkpoint_every. Otherwise the eval uses the trainer state the run kept on
trainer_checkpoint_every. A separate trainer first saves that state for a sampler, and
its Loops run is deactivated afterwards. Exits 1 when a set did not evaluate."""


def _step(value: str) -> int:
    if value == "base":
        return -1
    try:
        step = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an update number or 'base'") from None
    if step < 0:
        raise argparse.ArgumentTypeError(
            "an update number is 0 or more, and 'base' is the base model"
        )
    return step


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        hint(parser, argv, *MODELS.values())
        raise
    # Log plain output until the config says otherwise, so a failure to read the config
    # is still legible.
    configure_logging(verbose=args.verbose, style="plain")
    if args.command == "deactivate":
        return _deactivate(parser, args.run)
    args.config = locate_config(args.config)
    source = ConfigSource.of(args.config)
    source = replace(source, environment=load_environment(source.directory))
    try:
        if args.command == "cleanup":
            make_config_dir_importable(source.directory)
            return run_main(rl.cleanup(args))
        recipe = recipe_of(source.file)
        _refuse_other_recipes_flags(args, recipe)
        project = load_as(recipe, source, overrides=flag_overrides(args, recipe.config))
        if args.command == "eval":
            return _evaluate(project, args)
        if args.resume:
            project = _resume_latest(project)
        configure_logging(verbose=args.verbose, style=project.config.output.console)
        if args.command == "check":
            # Print the config's meaning first, so it is on screen while the check runs.
            for line in project.describe():
                print(line, flush=True)
            for line in run_main(project.check()).describe():
                print(line)
            return 0
        outcome = run_main(project.train())
        if not outcome.ok:
            logging.getLogger(PROG).error(
                "run %s after %d update(s): %s", outcome.status, outcome.steps, outcome.reason
            )
        return exit_code(outcome.status)
    except ERRORS as error:
        # An exception note says what the command found before it failed.
        parser.exit(2, "".join(f"{line}\n" for line in (str(error), *_notes(error))))
    except KeyboardInterrupt:
        return 130
    except Terminated:
        return TERMINATED_EXIT_CODE


def _evaluate(project: Project[Any], args: argparse.Namespace) -> int:
    """Run ``loops eval``, and return 0 when every set asked for evaluated the step."""
    if project.recipe is not rl.RECIPE:
        raise ValueError(
            f"loops eval runs an RL config's [eval.sets], but {args.config} is recipe = \""
            f'{project.recipe.name}", whose evals score validation NLL during the run'
        )
    configure_logging(verbose=args.verbose, style=project.config.output.console)
    if args.run is not None:
        directory = args.run.expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"--run {args.run}: no such run directory")
    else:
        try:
            directory = project.latest_run()
        except NoRunFound as error:
            raise ValueError(
                f"loops eval found no run, because no {error.run_name}-<time>-<id> directory "
                f"in {error.output_dir} has trained. Name one with --run"
            ) from None
    evaluated = run_main(
        rl_recipe.evaluate(
            project.config, directory, args.step, sets=args.sets, source=project.source
        )
    )
    wanted = args.sets or [name for name in project.config.eval.sets]
    missed = sorted(set(wanted) - set(evaluated))
    if missed:
        logging.getLogger(PROG).error(
            "eval set(s) %s did not evaluate. The log above says why",
            ", ".join(map(repr, missed)),
        )
        return 1
    return 0


def _deactivate(parser: argparse.ArgumentParser, run: str) -> int:
    """Run ``loops deactivate``, which ends each Loops run that ``run`` names.

    ``run`` is a run ID, or a run directory whose records name one run, or several
    after an SFT resume. Credentials come from a ``.env`` in the working directory.
    """
    load_environment(Path.cwd())
    log = logging.getLogger(PROG)
    try:
        if Path(run).is_dir():
            run_ids = read_run_ids(Path(run))
            log.info("%s recorded Loops run(s) %s", run, ", ".join(run_ids))
        elif RUN_ID.fullmatch(run):
            run_ids = [run]
        else:
            # Such a value is never sent, because a path in the URL would reach another
            # route.
            raise ValueError(f"{run} is not a Loops run ID or a run directory")
        for run_id in run_ids:
            log.info("Deactivating Loops run %s", run_id)
            run_main(deactivate_run(run_id))
            print(f"Deactivated Loops run {run_id}. Its checkpoints stay.")
    except (*ERRORS, httpx.HTTPError) as error:
        parser.exit(2, f"{error}\n")
    except KeyboardInterrupt:
        return 130
    except Terminated:
        return TERMINATED_EXIT_CODE
    return 0


def _notes(error: BaseException) -> list[str]:
    return getattr(error, "__notes__", [])


def _resume_latest[C: BaseModel](project: Project[C]) -> Project[C]:
    """Call ``project.resume_latest()``, wording its refusals in this command's flags."""
    name = project.path.name
    try:
        return project.resume_latest()
    except ResumeFromSet as error:
        raise ValueError(
            f"--resume takes the latest run of {name}, and [training] resume_from names "
            f"{error.resume_from}. Give one or the other"
        ) from None
    except NoRunFound as error:
        raise ValueError(
            f"--resume found no run, because no {error.run_name}-<time>-<id> directory in "
            f"{error.output_dir} has trained"
        ) from None
    except NoManifest as error:
        raise ValueError(
            f"--resume: the latest run of {name}, {error.directory}, kept no resume "
            "manifest. Name another run's manifest with --training.resume_from"
        ) from None


def _refuse_other_recipes_flags(args: argparse.Namespace, recipe: Recipe[Any]) -> None:
    """Refuse a flag that belongs to another recipe, and name that recipe."""
    own = {path for path, _ in cli_settings(recipe.config)}
    for label, model in MODELS.items():
        for path, _ in cli_settings(model):
            if path not in own and getattr(args, path, None) is not None:
                raise ValueError(
                    f'--{path} is an {label} setting; {args.config} is recipe = "{recipe.name}"'
                )


if __name__ == "__main__":
    raise SystemExit(main())
