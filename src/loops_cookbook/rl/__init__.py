"""The RL core, which schedules rollouts, computes GRPO advantages and drives the trainer.

It knows nothing about any one environment.
"""

from loops_cookbook.rl.config import Config
from loops_cookbook.rl.pipeline import Pipeline
from loops_cookbook.rl.run import Outcome, Run
from loops_cookbook.rl.trainer import StepReport, Trainer
from loops_cookbook.trainer import CredentialsError

__all__ = [
    "Config",
    "CredentialsError",
    "Outcome",
    "Pipeline",
    "Run",
    "StepReport",
    "Trainer",
]
