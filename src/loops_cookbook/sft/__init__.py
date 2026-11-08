"""Supervised fine-tuning (SFT) on fixed data, which trains a LoRA adapter on Loops.

This package holds the core, which is the data contracts, the rule that picks the
targets, and the run. ``sft.cli`` and ``sft.recipe`` are the composition roots that
wire the core to Loops and telemetry. Import them from there.
"""

from loops_cookbook.sft.config import Config
from loops_cookbook.sft.data import (
    ChatExample,
    DatasetError,
    TokenExample,
    load_dataset,
    load_rows,
)
from loops_cookbook.sft.run import Outcome, Run, build_renderer
from loops_cookbook.sft.sequences import Refused, SupervisedSequence, supervise

__all__ = [
    "ChatExample",
    "Config",
    "DatasetError",
    "Outcome",
    "Refused",
    "Run",
    "SupervisedSequence",
    "TokenExample",
    "build_renderer",
    "load_dataset",
    "load_rows",
    "supervise",
]
