"""The supervised fine-tuning (SFT) configuration, with one nested model per TOML section.

The model, compute and output sections are the RL recipe's, spelled the same. The data,
training and eval sections are SFT's own. Secrets never appear here. The Loops SDK
reads its key from the environment, which can be loaded from ``.env``.
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
        description="The contract that every row meets. chat rows are OpenAI chat "
        "fine-tuning JSONL. The cookbook renders them with the model's template, which "
        "decides the targets. tokens rows give input_ids and 0/1 weights, and they train "
        "as given. Nothing checks that their targets are the model's own tokens.",
    )
    train: str = Field(
        description="The training rows, as JSONL with one row per line in the format "
        "that [data] format names. The path is relative to this config file, or to the "
        "shell's directory when it is passed as a command-line flag."
    )
    validation: str | None = Field(
        default=None,
        description="Held-out rows in the same format, which the run scores for "
        "validation negative log-likelihood (NLL). The path resolves like train. Set "
        "this or validation_fraction. Set neither for no validation.",
    )
    validation_fraction: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description="Hold out this fraction of the training rows for validation, "
        "instead of reading a separate file. shuffle_seed chooses the rows.",
    )
    shuffle_seed: int = Field(
        default=0,
        description="The seed that chooses the held-out rows and each epoch's order. "
        "Epoch e shuffles with shuffle_seed + e, so a resumed run sees the same order.",
    )
    render_workers: int | None = Field(
        default=None,
        ge=1,
        description="The number of worker processes that take over a split which is "
        "still rendering after a few seconds. 1 renders in this process. When it is "
        "unset, the cookbook uses one fewer than the cores this process may use, and at "
        "most 4. Each worker holds a copy of the renderer and its tokenizer. This "
        "setting changes how fast check and the start of a run are, and how much memory "
        "they use. It never changes what trains.",
    )

    @model_validator(mode="after")
    def _one_validation_source(self) -> Self:
        if self.validation is not None and self.validation_fraction is not None:
            raise ValueError("set validation or validation_fraction, not both")
        return self


class TrainingSettings(_TrainingSettings):
    lr_schedule: Literal["constant", "linear", "cosine"] = Field(
        description="The learning rate after warmup. constant keeps it at its peak. "
        "linear and cosine decay it towards zero by the last update, along a straight "
        "line or a half cosine.",
    )
    warmup_steps: int = Field(
        ge=0,
        description="The number of updates over which the learning rate rises "
        "linearly to its peak. 0 starts at the peak.",
    )
    batch_size: int = Field(
        ge=1,
        description="The number of training sequences in each optimizer update. A "
        "conversation is usually one sequence. It becomes one sequence per trained turn "
        "when its template renders history differently from how each turn was "
        "generated.",
    )
    epochs: int = Field(ge=1, description="The number of passes over the training sequences.")
    loss_aggregation: Literal["sum", "token_mean"] = Field(
        default="sum",
        description="How the loss combines each update's trained tokens. sum is Loops' "
        "default, and it does not divide. token_mean divides by the number of trained "
        "tokens in the update.",
    )


class EvalSettings(Section):
    every_steps: int = Field(
        ge=0,
        description="Score validation NLL every N updates. Validation is always scored "
        "after the last update, and before the first update of a run that is not a "
        "resume. 0 scores only at those two points.",
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
