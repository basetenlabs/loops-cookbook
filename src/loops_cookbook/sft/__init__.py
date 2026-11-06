"""Supervised fine-tuning on fixed data: conversations in, a LoRA trained on Loops out.

The core: contracts, the target rule, the run. ``sft.cli`` and ``sft.recipe`` are the
composition roots that wire it to Loops and telemetry, imported from there.
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
