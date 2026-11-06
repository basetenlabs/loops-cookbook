"""A run resumed from its own checkpoint, trained offline.

``resumed_offline`` trains one step that writes a checkpoint manifest, lets the test
rewrite that manifest, and resumes from it: what a later run started with
``[training] resume_from`` sees.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from baseline_model import BaselineTrainer

from support.offline_run import train_offline


class Checkpointing(BaselineTrainer):
    """The baseline trainer, reporting a trainer checkpoint after each step."""

    async def checkpoint(self, report, *, final=False, keep_sampler_weights=False):
        report.checkpoint_state_path = "bt://loops:test/state/checkpoint"
        return report


async def resumed_offline(
    settings: Callable[..., Any],
    edit: Callable[[dict[str, Any]], None] = lambda manifest: None,
    **overrides: Any,
) -> tuple[Any, Any, Any]:
    """Train ``settings()`` one step, rewrite its step-0 manifest with ``edit``, then
    resume from it under ``settings(**overrides)``; the resumed run's
    ``(outcome, trainer, run)``."""
    outcome, _, run = await train_offline(settings(), trainer=Checkpointing())
    assert outcome.ok
    manifest = run.out / "checkpoints" / "step-0.json"
    recorded = json.loads(manifest.read_text())
    edit(recorded)
    manifest.write_text(json.dumps(recorded))
    trainer = Checkpointing()
    trainer.policy_version = 1
    resumed = settings(**{"training.resume_from": str(manifest), **overrides})
    return await train_offline(resumed, trainer=trainer)
