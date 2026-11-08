"""Configuration sections that every recipe shares, for the trained model, compute and output.

Each recipe's ``Config`` combines these sections with its own, and extends one where it
needs more. A setting that both recipes read is therefore declared once, and it is
spelled the same in both TOML files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, Self

from baseten.loops.models import AdamParams
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.fields import FieldInfo

from loops_cookbook.evidence import JsonValue


def is_local_path(value: str) -> bool:
    """Return whether a setting that takes a path or a Hugging Face repo ID names a path.

    A path starts with ``.``, ``/`` or ``~``. A repo ID never does, because Hugging Face
    refuses a name that starts with ``.``. Anything else is a repo ID, and
    ``llm.factory.load_tokenizer`` refuses a repo ID that a directory of the same name
    shadows.
    """
    return value.startswith((".", "/", "~"))


# Both recipes declare these settings, each in a section of its own. One `loops` flag
# sets each setting, so the two declarations share one description.
RECIPE_DESCRIPTION = (
    'The recipe this file configures, either "rl" or "sft". `loops` reads it to choose '
    "how to check and train."
)
MAX_SEQ_LEN_DESCRIPTION = (
    "The trainer's token budget for each sequence. RL defaults to the model's own limit. "
    "SFT requires it, so that check can refuse a row longer than this before anything "
    "is provisioned."
)
MAX_STEPS_DESCRIPTION = (
    "The run stops once it has made this many optimizer updates, counting the updates "
    "made before a resume. 0 sets no limit. RL then runs until it is stopped, and SFT "
    "runs every epoch."
)
LEARNING_RATE_DESCRIPTION = (
    "The Adam learning rate. Under SFT's lr_schedule, this is the peak learning rate."
)


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


DEDICATED_ONLY: dict[str, JsonValue] = {"dedicated_only": True}
"""Mark a ``[compute]`` setting, through its ``json_schema_extra``, as dedicated only.

Such a setting configures the run's own deployments, and a serverless run has none.
Loops ignores the setting on serverless capacity, so the section refuses any value
other than its default there."""


def is_dedicated_only(info: FieldInfo) -> bool:
    extra = info.json_schema_extra
    return isinstance(extra, dict) and extra.get("dedicated_only") is True


class ModelSettings(Section):
    base_model: str = Field(description="The Loops base model, such as Qwen/Qwen3.6-35B-A3B.")
    tokenizer: str | None = Field(
        default=None,
        min_length=1,
        description="Where the policy's tokenizer loads from. Give a Hugging Face repo "
        "ID, or a local directory that starts with ./, / or ~. A local directory is "
        "relative to this file. The default is base_model. Set it for a model that "
        "Loops serves under a name with no Hugging Face repo. A private repo reads "
        "HF_TOKEN.",
    )
    lora_rank: int = Field(ge=1, description="The LoRA rank of the trained adapter.")
    max_seq_len: int | None = Field(
        default=None,
        ge=1,
        description=MAX_SEQ_LEN_DESCRIPTION,
    )
    renderer: str | None = Field(
        default=None,
        min_length=1,
        description="The renderer factory, written as module:callable. By default the "
        "cookbook uses its own renderer for the models it renders, which "
        "docs/reference/renderers.md lists. For other models, Prime detects the "
        "renderer from the tokenizer.",
    )
    renderer_chat_template_kwargs: dict[str, JsonValue] = Field(
        default_factory=dict[str, JsonValue],
        description="Model-specific renderer options. On the command line, pass them as JSON.",
    )

    @field_validator("tokenizer")
    @classmethod
    def _local_tokenizer_is_a_directory(cls, value: str | None) -> str | None:
        if value is not None and is_local_path(value) and not Path(value).expanduser().is_dir():
            raise ValueError(
                f"{value} is not a directory. A local tokenizer is a directory of "
                "tokenizer files, and a Hugging Face repo ID does not start with ., / or ~"
            )
        return value

    @property
    def tokenizer_source(self) -> str:
        """Where the policy's tokenizer loads from, which is ``tokenizer`` or else ``base_model``."""
        return self.tokenizer or self.base_model


class ComputeSettings(Section):
    scope: Literal["serverless", "dedicated"] = Field(
        description="Where the run's trainer, samplers and evals run. serverless uses "
        "Loops' shared capacity and bills for tokens. If that capacity cannot serve the "
        "model or the configuration, creating the run fails. dedicated gives the run "
        "deployments of its own, which bill for GPU time until the run is deactivated."
    )
    spot_job: bool = Field(
        default=False,
        description="Run on reclaimable Loops spot capacity. A run whose capacity is "
        "reclaimed stops.",
        json_schema_extra=DEDICATED_ONLY,
    )
    reuse_from_run_id: str | None = Field(
        default=None,
        min_length=1,
        description="Reuse a prior run's warm pods, with fresh LoRA adapters. It does "
        "not restore a checkpoint.",
        json_schema_extra=DEDICATED_ONLY,
    )
    keep_run: bool = Field(
        default=True,
        description="Leave the Loops run up when the process exits. A dedicated run "
        "bills until you deactivate it with loops deactivate, and a later dedicated "
        "run can start warm on it through reuse_from_run_id. A serverless run holds a "
        "workspace slot. false deactivates the run on exit, and its checkpoints stay.",
    )
    trainer_replicas: int = Field(
        default=1,
        ge=1,
        description="The number of parallel copies of the trainer. More copies raise "
        "update throughput and use proportionally more GPUs. Any value above 1 always "
        "provisions fresh capacity.",
        json_schema_extra=DEDICATED_ONLY,
    )

    @model_validator(mode="after")
    def _reuse_needs_one_trainer(self) -> Self:
        # Loops ignores reuse for more than one trainer replica, so say so before the
        # run pays for fresh capacity.
        if self.reuse_from_run_id is not None and self.trainer_replicas > 1:
            raise ValueError(
                f"reuse_from_run_id cannot be combined with trainer_replicas="
                f"{self.trainer_replicas}, because Loops provisions fresh capacity for "
                "more than one trainer replica"
            )
        return self

    @model_validator(mode="after")
    def _dedicated_settings_need_dedicated(self) -> Self:
        # Compare values rather than ask whether a setting was written. A saved config
        # writes every default, and it must load back as the config it saved.
        if self.scope == "serverless":
            named = [
                name
                for name, info in type(self).model_fields.items()
                if is_dedicated_only(info)
                and getattr(self, name) != info.get_default(call_default_factory=True)
            ]
            if named:
                raise ValueError(
                    f"{', '.join(named)} configure the run's own deployments, and a "
                    'serverless run has none. Remove them, or set scope = "dedicated"'
                )
        return self


class OutputSettings(Section):
    run_name: str = Field(
        min_length=1,
        description="The prefix for the run directory and the Loops run name. It is "
        "required so that two configs in one output_dir do not share run directories. "
        "`loops run --resume` takes the latest run directory with this name.",
    )
    output_dir: str = Field(
        default="runs", description="The directory where run directories are created."
    )
    wandb_project: str | None = Field(
        default=None,
        # This is W&B's own limit. A longer name fails `wandb.init`, and with it the
        # mirror, but only once the run is training.
        max_length=128,
        description="Mirror metrics to this W&B project. The mirror is best effort.",
    )
    console: Literal["auto", "plain", "rich"] = Field(
        default="auto",
        description="The terminal output style. auto gives coloured logs, plus a live "
        "progress region where the recipe has one, when the log stream is a terminal "
        "and the console extra is installed. Otherwise auto gives plain output. "
        "Redirected output is always plain, so a captured log never contains control "
        "characters.",
    )
    trainer_checkpoint_every: int = Field(
        ge=0,
        description="Save resumable trainer state every N updates. 0 turns this off. "
        "The last update is always saved. That is the update at max_steps, or the last "
        "update of a run that ends any other way, saved on the way out within a bounded "
        "time and on a best-effort basis.",
    )
    sampler_checkpoint_every: int = Field(
        ge=0,
        description="Stage servable adapter weights every N updates. 0 turns this off. "
        "The update at max_steps is always saved.",
    )
    checkpoint_ttl_seconds: int = Field(
        ge=0,
        description="How long Loops keeps each remote checkpoint, in seconds. 0 keeps "
        "checkpoints indefinitely.",
    )

    @property
    def checkpoint_ttl(self) -> int | None:
        """The retention sent with each checkpoint save. ``None`` keeps the checkpoint."""
        return self.checkpoint_ttl_seconds or None


class TrainingSettings(Section):
    """The ``[training]`` settings that every recipe has. Each recipe extends them with its own."""

    seed: int | None = Field(
        default=None,
        description="The Loops trainer's random seed. When it is unset, Loops chooses one.",
    )
    max_steps: int = Field(ge=0, description=MAX_STEPS_DESCRIPTION)
    learning_rate: float = Field(gt=0, description=LEARNING_RATE_DESCRIPTION)
    grad_clip_norm: float | None = Field(
        default=None,
        ge=0,
        description="The gradient norm to clip at. 0 turns clipping off. When it is "
        "unset, the Loops SDK's default applies.",
    )
    adam_epsilon: float | None = Field(
        default=None,
        gt=0,
        description="Adam's epsilon. When it is unset, the Loops SDK's default applies.",
    )
    adam_beta1: float | None = Field(
        default=None,
        ge=0,
        lt=1,
        description="Adam's beta1. When it is unset, the Loops SDK's default applies.",
    )
    adam_beta2: float | None = Field(
        default=None,
        ge=0,
        lt=1,
        description="Adam's beta2. When it is unset, the Loops SDK's default applies.",
    )
    weight_decay: float | None = Field(
        default=None,
        ge=0,
        description="Adam's weight decay. When it is unset, the Loops SDK's default applies.",
    )
    resume_from: str | None = Field(
        default=None,
        description="The run to resume. Give a checkpoints/step-N.json manifest from a "
        "prior run, or that run's directory, which resumes from the manifest the run "
        "wrote last. For SFT, the prior run must have trained the same data. The path "
        "is relative to this config file, or to the shell's directory when it is passed "
        "as a command-line flag.",
    )
    init_from: str | None = Field(
        default=None,
        description="The Loops trainer weights that a fresh run starts from, such as "
        "bt://loops:<run>/weights/step-N. The run loads the LoRA adapter with fresh Adam "
        "moments. The run's steps count from 0, and the trainer's policy version "
        "continues from the checkpoint's. The checkpoint must have the same base model "
        "and LoRA rank. A resume ignores this setting, because the resumed state "
        "already holds those weights.",
    )
    init_from_optimizer: bool = Field(
        default=False,
        description="Also load the optimizer state of init_from, meaning the Adam "
        "moments and step. The run then continues the source run's optimizer under "
        "this run's config. The moments reflect the source run's gradient scale. From a "
        "checkpoint saved without them, Loops loads only the weights. This setting "
        "requires init_from, and a resume ignores it.",
    )

    @property
    def step_limit(self) -> int | None:
        """The number of updates this run makes. ``None`` means no limit."""
        return self.max_steps or None

    def adam_params(self, learning_rate: float) -> AdamParams:
        """Build the optimizer step's settings from those set here and the Loops SDK's defaults."""
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
            raise ValueError(
                "init_from_optimizer loads the optimizer state of init_from, so set init_from"
            )
        return self


class RunConfig(Protocol):
    """The sections that every recipe's config has, which ``loops`` reads for any recipe."""

    @property
    def output(self) -> OutputSettings: ...

    @property
    def training(self) -> TrainingSettings: ...
