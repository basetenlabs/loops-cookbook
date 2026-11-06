"""Open a config as a project, to check and train it from Python as ``loops`` does.

The config names its recipe with ``recipe = "rl"`` or ``recipe = "sft"``, and that
recipe's ``Recipe`` checks and trains it. ``loops`` loads its config through
``load`` too, and adds the automatic ``.env`` loading a library leaves to its caller.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from pydantic import BaseModel

from loops_cookbook.cli import (
    ConfigSource,
    Outcome,
    Recipe,
    Report,
    load_config_file,
    make_config_dir_importable,
)
from loops_cookbook.evidence import JsonValue
from loops_cookbook.rl import cli as rl
from loops_cookbook.run_record import latest_run, manifests
from loops_cookbook.settings import RunConfig
from loops_cookbook.sft import cli as sft

logger = logging.getLogger(__name__)

RECIPES: dict[str, Recipe[Any]] = {recipe.name: recipe for recipe in (rl.RECIPE, sft.RECIPE)}
# Pyright checks here that every recipe's config has the sections ``run_config`` reads.
_RUN_CONFIGS: tuple[type[RunConfig], ...] = (rl.RECIPE.config, sft.RECIPE.config)


def run_config(config: BaseModel) -> RunConfig:
    """``config``'s sections every recipe has (``_RUN_CONFIGS``)."""
    return cast(RunConfig, config)


class NoRunFound(ValueError):
    """No ``<run_name>-<time>-<id>`` directory in ``output_dir`` has trained."""

    def __init__(self, output_dir: str, run_name: str) -> None:
        super().__init__(f"no {run_name}-<time>-<id> directory in {output_dir} has trained")
        self.output_dir, self.run_name = output_dir, run_name


class NoManifest(ValueError):
    """The latest run, ``directory``, kept no resume manifest."""

    def __init__(self, directory: Path) -> None:
        super().__init__(f"the latest run, {directory}, kept no resume manifest")
        self.directory = directory


class ResumeFromSet(ValueError):
    """``[training] resume_from`` already names ``resume_from``, another run to resume."""

    def __init__(self, resume_from: str) -> None:
        super().__init__(f"[training] resume_from already names {resume_from}")
        self.resume_from = resume_from


def recipe_of(path: Path) -> Recipe[Any]:
    """The recipe ``path`` names with its ``recipe`` key."""
    declared = load_config_file(path.expanduser()).get("recipe")
    if declared is None:
        raise ValueError(f'{path} names no recipe; add recipe = "rl" or recipe = "sft"')
    if not isinstance(declared, str) or declared not in RECIPES:
        raise ValueError(
            f"{path}: recipe = {declared!r} is not a recipe; use "
            + " or ".join(f'"{name}"' for name in RECIPES)
        )
    return RECIPES[declared]


@dataclass(frozen=True)
class Project[C: BaseModel]:
    """A loaded config, where it came from, and the recipe that checks and trains it."""

    recipe: Recipe[C]
    source: ConfigSource
    config: C

    @property
    def path(self) -> Path:
        """The config's path as it was named."""
        return self.source.path

    def describe(self) -> list[str]:
        """Where the config came from and where its runs go, as ``loops check`` prints
        them before the check's own lines."""
        return [*self.source.describe(), f"  runs: {run_config(self.config).output.output_dir}"]

    async def check(self) -> Report:
        """Check the config as ``loops check`` does; launches no trainer or sandbox.

        It may download model or data files, and an RL check writes a run under
        ``output_dir/preflight``. ``describe()`` gives the lines ``loops check`` prints.
        """
        return await self.recipe.check(self.config, self.source)

    async def train(self) -> Outcome:
        """Train as ``loops run`` does; returns the run's ``Outcome``."""
        return await self.recipe.train(self.config, self.source)

    def resume_latest(self) -> Project[C]:
        """This project set to resume its latest run, as ``loops run --resume`` does: the
        ``<output_dir>/<run_name>-<time>-<id>`` that trained most recently
        (``latest_run``), from the manifest it wrote last.

        ``NoRunFound`` when there is no such run, rather than a fresh run: a mistyped
        ``run_name`` would otherwise train from scratch under a resume's name.
        ``NoManifest`` when that run kept no manifest: an older run is never taken in its
        place. ``ResumeFromSet`` when ``resume_from`` is set, since the two would name
        different runs.
        """
        training = run_config(self.config).training
        if training.resume_from:
            raise ResumeFromSet(training.resume_from)
        directory = self.latest_run()
        if not manifests(directory):
            raise NoManifest(directory)
        logger.info("Resuming the latest run of %s: %s", self.path.name, directory)
        training = training.model_copy(update={"resume_from": str(directory)})
        return replace(self, config=self.config.model_copy(update={"training": training}))

    def latest_run(self) -> Path:
        """The run of this config that trained most recently (``latest_run``);
        ``NoRunFound`` when none has, so nothing starts in its place."""
        output = run_config(self.config).output
        directory = latest_run(output.output_dir, output.run_name)
        if directory is None:
            raise NoRunFound(output.output_dir, output.run_name)
        return directory


def load(
    path: str | Path,
    *,
    overrides: Mapping[str, JsonValue] | None = None,
    env_file: str | Path | None = None,
) -> Project[Any]:
    """The project ``path`` holds, validated and ready to check or train.

    A file the config names is relative to the config, a symlink followed to the file
    it points to; a dotted override that names one is relative to the working
    directory, as a ``loops`` flag is. ``output_dir``
    is resolved now, so a later change of directory does not move it. The config's
    directory becomes importable. ``env_file``, relative to the config, fills only
    variables the process does not set; without it no ``.env`` is read.
    """
    source = ConfigSource.of(Path(path))
    return load_as(recipe_of(source.file), source, overrides=overrides, env_file=env_file)


def load_as[C: BaseModel](
    recipe: Recipe[C],
    source: ConfigSource,
    *,
    overrides: Mapping[str, JsonValue] | None = None,
    env_file: str | Path | None = None,
) -> Project[C]:
    """``load`` for a config already known to be ``recipe``'s, as ``loops`` knows it.

    It reads the real file ``source`` reached, so everything comes from one directory.
    """
    config = recipe.load(source.file, overrides)
    if env_file is not None:
        named = Path(env_file).expanduser()
        if not named.is_absolute():
            named = source.directory / named
        with named.open() as stream:
            load_dotenv(stream=stream, override=False)
        source = replace(source, environment=(*source.environment, named))
    make_config_dir_importable(source.directory)
    output = run_config(config).output
    config = config.model_copy(
        update={
            "output": output.model_copy(
                update={"output_dir": str(Path(output.output_dir).expanduser().resolve())}
            )
        }
    )
    return Project(recipe, source, config)
