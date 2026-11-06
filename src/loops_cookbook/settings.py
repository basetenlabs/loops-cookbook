"""Configuration sections every recipe shares: the trained model, compute, and output.

Each recipe's ``Config`` composes these with its own sections, and extends one where
it needs more, so a setting both recipes read is declared once and spelled the same
in both TOML files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, Self

from baseten.loops.models import AdamParams
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.fields import FieldInfo

from loops_cookbook.evidence import JsonValue


def is_local_path(value: str) -> bool:
    """Whether a setting that takes a path or a Hugging Face repo id names a path.

    A path starts with ``.``, ``/`` or ``~``; a repo id never does, since Hugging Face
    refuses a name starting with ``.``. Anything else is a repo id, and
    ``llm.factory.load_tokenizer`` refuses one a directory of the same name shadows.
    """
    return value.startswith((".", "/", "~"))


# Settings both recipes declare in a section of their own. One `loops` flag sets each,
# so the two declarations share one description.
RECIPE_DESCRIPTION = (
    'The recipe this file configures: "rl" or "sft". `loops` reads it to choose how '
    "to check and train."
)
MAX_SEQ_LEN_DESCRIPTION = (
    "Per-sequence token budget for the trainer. RL defaults to the model's. SFT requires "
    "it, so that check refuses a row longer than this before anything is provisioned."
)
MAX_STEPS_DESCRIPTION = (
    "Stop once the run has made this many optimizer updates, the ones before a resume "
    "included. 0 runs without a limit: RL until stopped, SFT every epoch."
)
LEARNING_RATE_DESCRIPTION = "Adam learning rate; under SFT's lr_schedule, its peak."


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


DEDICATED_ONLY: dict[str, JsonValue] = {"dedicated_only": True}
"""A ``[compute]`` setting's mark, as its ``json_schema_extra``: it configures the
run's own deployments, which a serverless run has none of. Loops ignores such a
setting on serverless capacity, so the section refuses it there at any value but
its default."""


def is_dedicated_only(info: FieldInfo) -> bool:
    extra = info.json_schema_extra
    return isinstance(extra, dict) and extra.get("dedicated_only") is True


class ModelSettings(Section):
    base_model: str = Field(description="Loops base model, e.g. Qwen/Qwen3.6-35B-A3B.")
    tokenizer: str | None = Field(
        default=None,
        min_length=1,
        description="Where the policy's tokenizer loads from: a Hugging Face repo id, or "
        "a local directory starting with ./, / or ~ (relative to this file); default is "
        "base_model. For a model Loops serves under a name with no Hugging Face repo. A "
        "private repo reads HF_TOKEN.",
    )
    lora_rank: int = Field(ge=1, description="LoRA rank for the trained adapter.")
    max_seq_len: int | None = Field(
        default=None,
        ge=1,
        description=MAX_SEQ_LEN_DESCRIPTION,
    )
    renderer: str | None = Field(
        default=None,
        min_length=1,
        description="Renderer factory as module:callable; default is the cookbook's "
        "renderer for the models it renders (docs/reference/renderers.md), otherwise "
        "Prime auto-detection from the tokenizer.",
    )
    renderer_chat_template_kwargs: dict[str, JsonValue] = Field(
        default_factory=dict[str, JsonValue],
        description="Model-specific renderer options (JSON on the command line).",
    )

    @field_validator("tokenizer")
    @classmethod
    def _local_tokenizer_is_a_directory(cls, value: str | None) -> str | None:
        if value is not None and is_local_path(value) and not Path(value).expanduser().is_dir():
            raise ValueError(
                f"{value} is not a directory; a local tokenizer is a directory of "
                "tokenizer files (a Hugging Face repo id does not start with ., / or ~)"
            )
        return value

    @property
    def tokenizer_source(self) -> str:
        """Where the policy's tokenizer loads from: ``tokenizer``, else ``base_model``."""
        return self.tokenizer or self.base_model


class ComputeSettings(Section):
    scope: Literal["serverless", "dedicated"] = Field(
        description="Where the run's trainer, samplers and evals run. serverless: Loops' "
        "shared capacity, billed for tokens; a model or configuration it cannot serve "
        "fails the run's creation. dedicated: deployments of the run's own, billed for "
        "GPU time until the run is deactivated."
    )
    spot_job: bool = Field(
        default=False,
        description="Use reclaimable Loops spot capacity; interrupted runs stop.",
        json_schema_extra=DEDICATED_ONLY,
    )
    reuse_from_run_id: str | None = Field(
        default=None,
        min_length=1,
        description="Reuse a prior run's warm pods with fresh LoRAs; does not restore a "
        "checkpoint.",
        json_schema_extra=DEDICATED_ONLY,
    )
    keep_run: bool = Field(
        default=True,
        description="Leave the Loops run up when the process exits. A dedicated run bills "
        "until it is deactivated with loops deactivate, and a later dedicated run's "
        "reuse_from_run_id can start warm on it; a serverless run holds a workspace slot. "
        "false deactivates the run on exit; its checkpoints stay.",
    )
    trainer_replicas: int = Field(
        default=1,
        ge=1,
        description="Parallel copies of the trainer. More raise update throughput at "
        "proportionally more GPUs; above 1 always provisions fresh capacity.",
        json_schema_extra=DEDICATED_ONLY,
    )

    @model_validator(mode="after")
    def _reuse_needs_one_trainer(self) -> Self:
        # Loops ignores reuse for more than one trainer replica; say so before paying.
        if self.reuse_from_run_id is not None and self.trainer_replicas > 1:
            raise ValueError(
                f"reuse_from_run_id cannot be combined with trainer_replicas="
                f"{self.trainer_replicas}; Loops provisions fresh capacity for more than "
                "one trainer replica"
            )
        return self

    @model_validator(mode="after")
    def _dedicated_settings_need_dedicated(self) -> Self:
        # By value, not by whether it was written: a saved config writes every default,
        # and must load back as the config it saved.
        if self.scope == "serverless":
            named = [
                name
                for name, info in type(self).model_fields.items()
                if is_dedicated_only(info)
                and getattr(self, name) != info.get_default(call_default_factory=True)
            ]
            if named:
                raise ValueError(
                    f"{', '.join(named)} configure the run's own deployments, which a "
                    'serverless run has none of; remove them, or set scope = "dedicated"'
                )
        return self


class OutputSettings(Section):
    run_name: str = Field(
        min_length=1,
        description="Prefix for the run directory and the Loops run name. Required, so "
        "that two configs in one output_dir do not share run directories: `loops run "
        "--resume` takes the latest one of this name.",
    )
    output_dir: str = Field(default="runs", description="Where run directories are created.")
    wandb_project: str | None = Field(
        default=None,
        # W&B's own limit: a longer name fails `wandb.init`, and with it the mirror,
        # only once the run is training.
        max_length=128,
        description="Mirror metrics to this W&B project (best effort).",
    )
    console: Literal["auto", "plain", "rich"] = Field(
        default="auto",
        description="Terminal output style. auto: coloured logs, and a live progress "
        "region where the recipe has one, when the log stream is a terminal and the "
        "console extra is installed, plain otherwise. Redirected output is always "
        "plain, so a captured log never contains control characters.",
    )
    trainer_checkpoint_every: int = Field(
        ge=0,
        description="Save resumable trainer state every N updates; 0 disables. The "
        "last update is always saved: the one at max_steps, and on the way out of a run "
        "that ends any other way (bounded, best effort).",
    )
    sampler_checkpoint_every: int = Field(
        ge=0,
        description="Stage servable adapter weights every N updates; 0 disables. The "
        "update at max_steps is always saved.",
    )
    checkpoint_ttl_seconds: int = Field(
        ge=0,
        description="Remote checkpoint retention, in seconds; 0 keeps them.",
    )

    @property
    def checkpoint_ttl(self) -> int | None:
        """The retention sent with each checkpoint save; ``None`` keeps it."""
        return self.checkpoint_ttl_seconds or None


class TrainingSettings(Section):
    """The ``[training]`` settings every recipe has; each recipe extends it with its own."""

    seed: int | None = Field(
        default=None, description="The Loops trainer's random seed. Unset: Loops chooses."
    )
    max_steps: int = Field(ge=0, description=MAX_STEPS_DESCRIPTION)
    learning_rate: float = Field(gt=0, description=LEARNING_RATE_DESCRIPTION)
    grad_clip_norm: float | None = Field(
        default=None,
        ge=0,
        description="Gradient norm clipping; 0 disables it. Unset: the Loops SDK's.",
    )
    adam_epsilon: float | None = Field(
        default=None, gt=0, description="Adam epsilon. Unset: the Loops SDK's."
    )
    adam_beta1: float | None = Field(
        default=None, ge=0, lt=1, description="Adam beta1. Unset: the Loops SDK's."
    )
    adam_beta2: float | None = Field(
        default=None, ge=0, lt=1, description="Adam beta2. Unset: the Loops SDK's."
    )
    weight_decay: float | None = Field(
        default=None, ge=0, description="Adam weight decay. Unset: the Loops SDK's."
    )
    resume_from: str | None = Field(
        default=None,
        description="A checkpoints/step-N.json manifest from a prior run (for SFT, of the "
        "same data), or that run's directory, which resumes from the manifest it wrote "
        "last. Relative to this config file, or to the shell when passed as a "
        "command-line flag.",
    )
    init_from: str | None = Field(
        default=None,
        description="Loops trainer weights a fresh run starts from, such as "
        "bt://loops:<run>/weights/step-N: the LoRA adapter, with fresh Adam moments; the "
        "run's steps count from 0, the trainer's policy version continues from the "
        "checkpoint's. Same base model and LoRA rank. A resume ignores it: the resumed "
        "state already holds those weights.",
    )
    init_from_optimizer: bool = Field(
        default=False,
        description="Also load init_from's optimizer state (Adam moments and step): a fork "
        "that continues the source run's optimizer under this run's config. The moments "
        "reflect the source run's gradient scale; from a checkpoint saved without them "
        "Loops loads the weights only. Requires init_from; a resume ignores it.",
    )

    @property
    def step_limit(self) -> int | None:
        """Updates this run makes; ``None`` runs without a limit."""
        return self.max_steps or None

    def adam_params(self, learning_rate: float) -> AdamParams:
        """The optimizer step's settings: those set here, and the Loops SDK's for the rest."""
        named = {
            "grad_clip_norm": self.grad_clip_norm,
            "eps": self.adam_epsilon,
            "beta1": self.adam_beta1,
            "beta2": self.adam_beta2,
            "weight_decay": self.weight_decay,
        }
        return AdamParams(
            learning_rate=learning_rate,
            **{name: value for name, value in named.items() if value is not None},
        )

    @model_validator(mode="after")
    def _optimizer_needs_weights(self) -> Self:
        if self.init_from_optimizer and not self.init_from:
            raise ValueError("init_from_optimizer loads init_from's optimizer; set init_from")
        return self


class RunConfig(Protocol):
    """The sections every recipe's config has, which ``loops`` reads whatever the recipe."""

    @property
    def output(self) -> OutputSettings: ...

    @property
    def training(self) -> TrainingSettings: ...
