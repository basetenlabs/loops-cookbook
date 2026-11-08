"""What the `loops` command does with an RL config, to train agents with RL on Baseten Loops.

`loops check` asks Loops whether it serves the model, builds the rollout runtime and
discovers its tasks. It launches nothing. `loops cleanup` releases the provider
resources that a stopped run left behind.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from loops_cookbook import cli
from loops_cookbook.cli import load_config_file
from loops_cookbook.evidence import JsonValue
from loops_cookbook.providers import create_adapters
from loops_cookbook.resources.journal import JournalError
from loops_cookbook.rl import recipe
from loops_cookbook.rl.config import Config
from loops_cookbook.rl.run import Outcome
from loops_cookbook.runtime.context import RuntimeConfigurationError
from loops_cookbook.runtime.driver import RuntimeExecutionFailure

CLEANUP = (
    "Release every provider resource that a stopped run left behind. The command reads "
    "the run's resource journal (--state-dir) or, when the journal is lost, finds the "
    "resources by the run's ownership labels (--run-id). It exits 0 when every resource "
    "is proven released, and 1 while any is not. A label sweep can never prove that "
    "nothing was missed, so it exits 3, not 0, once everything it listed is released. "
    "Exit code 2 means a configuration error or a held journal lock."
)
# A label sweep released everything it listed. Only the listings' completeness is unproven.
SWEEP_LISTING_UNCONFIRMED = 3


def load_training_config(path: Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load the training settings, resolving each file reference against whoever named it."""
    return RECIPE.load(path, overrides)


def add_cleanup_arguments(cleanup: argparse.ArgumentParser) -> None:
    cleanup.add_argument(
        "config",
        type=Path,
        metavar="CONFIG",
        help="The run's training TOML or JSON. Only its [providers] tables are read.",
    )
    source = cleanup.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--state-dir",
        type=Path,
        metavar="DIR",
        help="The run's resource state directory, as its unresolved-resource report names it.",
    )
    source.add_argument(
        "--run-id",
        metavar="ID",
        help=(
            "Sweep by ownership labels when the journal is gone. Use it only once the run "
            "has stopped. It exits 1 while a listed resource is unresolved or a listing "
            "failed, and 3 once every listed resource is released."
        ),
    )
    cleanup.add_argument(
        "-v", "--verbose", action="store_true", help="Log at DEBUG instead of INFO."
    )


def _providers(path: Path) -> dict[str, dict[str, JsonValue]]:
    """Return the ``[providers]`` tables of a training config, and nothing else from it."""
    tables = load_config_file(path).get("providers", {})
    return TypeAdapter(dict[str, dict[str, JsonValue]]).validate_python(tables)


async def cleanup(args: argparse.Namespace) -> int:
    from loops_cookbook.resources.cleanup import (
        only_listings_unconfirmed,
        reconcile_journal,
        sweep_labels,
    )

    adapters = await create_adapters(_providers(args.config))
    if args.state_dir is not None:
        source = f"journal {args.state_dir}"
        report = await reconcile_journal(args.state_dir.expanduser().resolve(), adapters)
    else:
        source = f"ownership labels of run {args.run_id}"
        report = await sweep_labels(adapters, args.run_id)
    for result in report.results:
        detail = f": {result.detail}" if result.detail else ""
        print(f"{result.resource_id} {result.status.value}{detail}")
    if report.complete:
        print(f"Cleanup complete from {source}: {len(report.results)} resource(s) accounted for.")
        return 0
    if args.state_dir is None and only_listings_unconfirmed(report):
        print(
            f"Released everything listed under {source}. A label sweep cannot prove that "
            "nothing else is left, because provider listings lag. Rerun it once they "
            "settle, and check the provider for anything still labeled "
            f"loops.run_id={args.run_id}.",
            file=sys.stderr,
        )
        return SWEEP_LISTING_UNCONFIRMED
    print(
        f"Cleanup incomplete from {source}: {len(report.unresolved)} resource(s) are not "
        "proven released. Rerun the same command once the provider settles.",
        file=sys.stderr,
    )
    return 1


async def _train(config: Config, source: cli.ConfigSource) -> Outcome:
    return await recipe.train(config, source=source)


RECIPE = cli.Recipe(
    name="rl",
    config=Config,
    check=lambda config, source: recipe.check(config, source=source),
    train=_train,
    references=(
        "training.resume_from",
        "rollout_runtime.files.*",
        "eval.sets.*.rollout_runtime.files.*",
    ),
    errors=(RuntimeConfigurationError, RuntimeExecutionFailure, JournalError),
)
