"""Supervised fine-tuning configuration: one TOML file, one nested model per section.

The model, compute and output sections are the RL recipe's, spelled the same; data,
training and evaluation are SFT's own. Secrets never appear here: the Loops SDK reads
its key from the environment, loaded from ``.env``.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from loops_cookbook.settings import (
    MAX_SEQ_LEN_DESCRIPTION,
    RECIPE_DESCRIPTION,
    ComputeSettings,
    ModelSettings,
    OutputSettings,
    Section,
)
from loops_cookbook.settings import TrainingSettings as _TrainingSettings
from loops_cookbook.sft.data import Format


class SftModelSettings(ModelSettings):
    max_seq_len: int = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        ge=1,
        description=MAX_SEQ_LEN_DESCRIPTION,
    )


class DataSettings(Section):
    format: Format = Field(
        default="chat",
        description="The contract every row meets. chat: OpenAI chat fine-tuning JSONL, "
        "rendered with the model's template, which decides the targets. tokens: "
        "input_ids and 0/1 weights per row, trained as given, with no checks that the "
        "targets are the model's own tokens.",
    )
    train: str = Field(
        description="Training rows as JSONL in [data] format, one per line. Relative to "
        "this config file, or to the shell when passed as a command-line flag."
    )
    validation: str | None = Field(
        default=None,
        description="Held-out rows in the same format, scored for validation "
        "NLL. Resolved like train. Set this or validation_fraction, or neither for no "
        "validation.",
    )
    validation_fraction: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description="Hold out this fraction of the training rows, chosen by shuffle_seed, "
        "for validation instead of a separate file.",
    )
    shuffle_seed: int = Field(
        default=0,
        description="Seed for the held-out rows and for each epoch's order; epoch e "
        "shuffles with shuffle_seed + e, so a resumed run sees the same order.",
    )
    render_workers: int | None = Field(
        default=None,
        ge=1,
        description="Worker processes that render a split still rendering after a few "
        "seconds; 1 renders in this process. Unset: the cores this process may use less "
        "one, at most 4. Each holds a copy of the renderer and its tokenizer. Changes how "
        "fast and how much memory check and the start of run take, never what trains.",
    )

    @model_validator(mode="after")
    def _one_validation_source(self) -> Self:
        if self.validation is not None and self.validation_fraction is not None:
            raise ValueError("set validation or validation_fraction, not both")
        return self


class TrainingSettings(_TrainingSettings):
    lr_schedule: Literal["constant", "linear", "cosine"] = Field(
        description="Learning rate after warmup: constant, or decayed towards zero by "
        "the last update, linearly or along a half cosine.",
    )
    warmup_steps: int = Field(
        ge=0,
        description="Updates over which the learning rate rises linearly to its peak; "
        "0 starts at the peak.",
    )
    batch_size: int = Field(
        ge=1,
        description="Training sequences per optimizer update. A conversation is one "
        "sequence, or one per trained turn when its template renders history "
        "differently from how each turn was generated.",
    )
    epochs: int = Field(ge=1, description="Passes over the training sequences.")
    loss_aggregation: Literal["sum", "token_mean"] = Field(
        default="sum",
        description="Reduction over each update's trained tokens. sum is Loops' default: "
        "no division; token_mean divides by the update's trained-token count.",
    )


class EvalSettings(Section):
    every_steps: int = Field(
        ge=0,
        description="Score validation NLL every N updates; 0 scores only before the "
        "first update and after the last.",
    )


class Config(Section):
    recipe: Literal["sft"] = Field(
        default="sft",
        description=RECIPE_DESCRIPTION,
        json_schema_extra={"cli": False},
    )
    model: SftModelSettings = Field(description="The model being fine-tuned.")
    data: DataSettings = Field(description="Where the conversations come from.")
    training: TrainingSettings = Field(description="The optimizer and the batch.")
    eval: EvalSettings = Field(description="When validation NLL is scored.")
    compute: ComputeSettings = Field(description="Where the run executes.")
    output: OutputSettings = Field(description="What the run leaves behind.")
