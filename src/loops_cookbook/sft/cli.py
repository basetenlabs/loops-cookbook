"""Fine-tune a model on fixed data with Baseten Loops.

`loops check` renders every conversation with the model's chat template and reports
what trains, what is split per turn and what is refused, and why; it launches nothing.
`loops run` does the same, then trains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loops_cookbook import cli
from loops_cookbook.sft import recipe
from loops_cookbook.sft.config import Config
from loops_cookbook.sft.data import DatasetError
from loops_cookbook.sft.run import Outcome


def load_config(path: Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load SFT settings; a file a setting names is relative to the config that names it,
    or to the shell when it came from a flag."""
    return RECIPE.load(path, overrides)


async def _train(config: Config, source: cli.ConfigSource) -> Outcome:
    return await recipe.train(config, source=source)


RECIPE = cli.Recipe(
    name="sft",
    config=Config,
    check=lambda config, source: recipe.check(config),
    train=_train,
    references=("data.train", "data.validation", "training.resume_from"),
    errors=(DatasetError,),
)
