"""A run trained offline: the baseline model's trainer and renderer, no Loops.

``baseline_model`` is importable because the root conftest puts
``tests/fixtures/offline`` on ``sys.path``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from baseline_model import BaselineTrainer, create_renderer

from loops_cookbook.rl import cli
from loops_cookbook.rl.recipe import assemble


def offline_config(path: Path, tmp_path: Path, overrides: dict[str, Any] | None = None):
    """The training config at ``path`` as an offline run needs it: replica affinity on
    (the baseline sampler reads the rollout from it) and output under ``tmp_path``."""
    return cli.load_training_config(
        path,
        {
            "compute.sampler_replica_affinity": True,
            "output.output_dir": str(tmp_path / "runs"),
            **(overrides or {}),
        },
    )


async def train_offline(
    config: Any,
    *,
    trainer: Any = None,
    renderer: Any = None,
    timeout: float = 60,
    **assemble_options: Any,
) -> tuple[Any, Any, Any]:
    """``(outcome, trainer, run)`` of ``config`` trained by ``trainer`` (a fresh
    ``BaselineTrainer`` by default) with ``renderer`` (the baseline one by default).

    A run that has not ended within ``timeout`` seconds fails the test rather than
    hanging the suite. ``assemble_options`` go to ``assemble`` as they are.
    """
    trainer = trainer or BaselineTrainer()
    renderer = renderer or create_renderer()
    run = assemble(
        config, trainer=trainer, renderer_factory=lambda _config: renderer, **assemble_options
    )
    outcome = await asyncio.wait_for(run.execute(), timeout)
    return outcome, trainer, run
