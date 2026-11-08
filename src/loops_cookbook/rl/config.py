"""The RL training configuration, read from one TOML file with one model per section.

The rollout runtime's own settings live under ``[rollout_runtime.config]``. They are
application data, and nothing here interprets them. The files a rollout runtime needs
resolved are named in ``[rollout_runtime.files]``. Secrets never appear in the config.
Each SDK, rollout runtime or adapter reads the variables it needs from the environment,
which is loaded from ``.env``.
"""

from __future__ import annotations

import math
import re
from typing import Literal, Self

from pydantic import Field, ValidationError, field_validator, model_validator

from loops_cookbook import settings
from loops_cookbook.evidence import JsonValue
from loops_cookbook.model_access.producers import DEFAULT_PRODUCER
from loops_cookbook.settings import RECIPE_DESCRIPTION
from loops_cookbook.settings import Section as _Section
from loops_cookbook.settings import TrainingSettings as _TrainingSettings


def _refuse_renamed_runtime(data: object, table: str) -> object:
    # Without this check, an old config's [runtime] table reads as a typo ("extra
    # inputs") instead of as a table that was renamed.
    if isinstance(data, dict) and "runtime" in data:
        raise ValueError(
            f"[{table}runtime] is now [{table}rollout_runtime]. Rename the table. Its "
            "keys are unchanged."
        )
    return data


class OpenAISettings(_Section):
    bind: str = Field(
        default="127.0.0.1:0",
        pattern=r"^.+:[0-9]+$",
        description="The host:port the gateway listens on. By default it listens on "
        "loopback, on a port the system picks.",
    )
    advertised_base_url: str | None = Field(
        default=None,
        pattern=r"^https?://[^/]+/?$",
        description="The origin agents use to reach the gateway, such as "
        "http://127.0.0.1:43210. By default it is the gateway's bound loopback address.",
    )
    model_name: str | None = Field(
        default=None,
        min_length=1,
        description="The one model name the gateway advertises. By default it is "
        "[model].base_model.",
    )


class RuntimeSettings(_Section):
    factory: str = Field(
        min_length=1,
        description="The rollout runtime factory, as module:callable. For the Harbor "
        "rollout runtime, use loops_cookbook.harbor:create_runtime. For a custom rollout "
        "runtime, name its module beside this config file.",
    )
    model_access: Literal["direct", "openai_http"] | None = Field(
        default=None,
        description="How the agent reaches the model. direct is the cookbook's llm() "
        "client, which only an in-process agent can use. openai_http is an "
        "OpenAI-compatible Chat Completions endpoint for each attempt. When unset, the run "
        "uses the model access the rollout runtime declares, which is an Episode class's "
        "model_access or the model access of a Harbor job's agents. Set it only for a "
        "rollout runtime that supports more than one.",
    )
    openai: OpenAISettings | None = Field(
        default=None,
        description="Settings for the Chat Completions gateway. They apply only when the "
        "model access is openai_http.",
    )
    config: dict[str, JsonValue] = Field(
        default_factory=dict[str, JsonValue],
        description="The rollout runtime's own settings. The cookbook hands them to the "
        "factory unchanged.",
    )
    files: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="Files the rollout runtime may resolve, as alias = path. A path in "
        "this file is relative to this config file. A path passed as a command-line flag "
        "is relative to the shell's working directory.",
    )
    has_images: bool = Field(
        default=False,
        description="Whether the agent's prompts carry images. When it is false, the run "
        "refuses every request that has an image. When it is true, loops check renders a "
        "probe image through every renderer. Loops can't record an image prompt, so router "
        "replay needs this to be false.",
    )

    @model_validator(mode="after")
    def _gateway(self) -> Self:
        if self.openai is not None and self.model_access == "direct":
            raise ValueError(
                '[rollout_runtime.openai] applies only when model_access = "openai_http"'
            )
        return self


# The Chat Completions ``reasoning_effort`` values that the pinned OpenAI schema
# enumerates. Direct access accepts the same set, so both kinds of model access agree.
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class ModelSettings(settings.ModelSettings):
    reasoning_efforts: dict[str, dict[str, JsonValue]] = Field(
        default_factory=dict[str, dict[str, JsonValue]],
        description="The reasoning_effort values a request may send. Each value maps to "
        "the renderer_chat_template_kwargs entries it overrides. The run rejects a "
        "request that sends a value not declared here.",
    )

    @field_validator("reasoning_efforts")
    @classmethod
    def _efforts(cls, efforts: dict[str, dict[str, JsonValue]]) -> dict[str, dict[str, JsonValue]]:
        unknown = sorted(set(efforts) - set(REASONING_EFFORTS))
        if unknown:
            raise ValueError(
                f"reasoning_efforts keys must be values an OpenAI client can send "
                f"({', '.join(REASONING_EFFORTS)}); got {', '.join(unknown)}"
            )
        return efforts


class SamplingSettings(_Section):
    temperature: float = Field(
        default=1.0,
        ge=0,
        description="The sampling temperature for a generation whose request names none. "
        "A request may set its own.",
    )
    max_generation_tokens: int = Field(
        ge=1,
        description="The length limit, in tokens, for a generation whose request names "
        "none. A request may set its own, so this value is not a cap. The run lowers "
        "either limit to the room the trajectory budget leaves.",
    )
    max_trajectory_tokens: int | None = Field(
        default=None,
        ge=1,
        description="The trajectory budget, in tokens, for the prompt plus the generation. "
        "The run lowers each call's max_tokens to the room its prompt leaves, and it "
        "refuses only a prompt that leaves no room. The budget always applies over HTTP. "
        "A direct call may pass its own. The default is max_seq_len, which is the "
        "capacity the trainer was configured for.",
    )


class ImportanceSamplingConfig(_Section):
    """The importance-sampling loss, ``-ratio * advantage``, which takes no settings."""


class ClipConfig(_Section):
    """Settings for the PPO and CISPO losses, which clip the importance ratio to [low, high]."""

    clip_low_threshold: float = Field(
        default=0.8, ge=0, description="The lowest importance ratio kept."
    )
    clip_high_threshold: float = Field(
        default=1.2, ge=0, description="The highest importance ratio kept."
    )

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.clip_high_threshold < self.clip_low_threshold:
            raise ValueError("clip_high_threshold must be at least clip_low_threshold")
        return self


class DROConfig(_Section):
    """Settings for the DRO loss, ``logprob * advantage - beta / 2 * log_ratio**2``."""

    beta: float = Field(default=0.05, ge=0, description="The weight of the log-ratio penalty.")


class DPPOConfig(_Section):
    """Settings for the DPPO loss.

    DPPO is a trust region on probability differences, plus a squared log-ratio KL term.
    """

    dppo_mask_high: float = Field(
        default=0.2, ge=0, description="The largest probability rise kept for a positive advantage."
    )
    dppo_mask_low: float = Field(
        default=0.2, ge=0, description="The largest probability fall kept for a negative advantage."
    )
    adv_tau: float = Field(default=1.0, description="The scale of the advantage.")
    kl_tau: float = Field(default=1e-3, description="The weight of the KL term.")


# The built-in losses whose inputs are GRPO's logprobs, advantages and loss mask. Each
# maps to its settings as the Loops trainer validates them, defaults included. Loops
# itself ignores a key it doesn't know, but these models refuse it.
LOSS_FN_CONFIGS: dict[str, type[_Section]] = {
    "importance_sampling": ImportanceSamplingConfig,
    "ppo": ClipConfig,
    "cispo": ClipConfig,
    "dro": DROConfig,
    "dppo": DPPOConfig,
}


def _loss_fn_config_keys() -> str:
    """List each built-in loss's settings and defaults, for the setting's description."""
    return ". ".join(
        f"{name} takes "
        + (
            ", ".join(f"{key} ({info.default:g})" for key, info in model.model_fields.items())
            or "no settings"
        )
        for name, model in LOSS_FN_CONFIGS.items()
    )


class CustomLossSettings(_Section):
    factory: str = Field(
        min_length=1,
        description="The loss factory, as module:callable, such as a module beside this "
        "config file. The factory takes this table's config and returns the loss function "
        "that forward_backward_custom runs. A custom loss needs the custom-loss extra.",
    )
    config: dict[str, JsonValue] = Field(
        default_factory=dict[str, JsonValue],
        description="The loss's own settings. The cookbook hands them to the factory unchanged.",
    )


class TrainingSettings(_TrainingSettings):
    mode: Literal["sync", "async"] = Field(
        default="sync",
        description="sync collects a batch and then trains on it, one batch at a time. "
        "async collects the next batch while the current one trains.",
    )
    group_size: int = Field(
        ge=2, description="The number of rollouts sampled for each task's group."
    )
    max_group_losses: int = Field(
        ge=0,
        description="How many rollouts a group may lose and still train. It is a "
        "tolerance rather than a minimum, so it keeps its meaning when group_size changes.",
    )
    groups_per_step: int = Field(ge=1, description="The number of groups in each optimizer update.")
    loss_fn: Literal["importance_sampling", "ppo", "cispo", "dro", "dppo", "custom"] = Field(
        description="The loss that forward_backward computes. Name one of Loops' built-in "
        "losses, or custom for your own loss from [training.custom_loss], which runs "
        "through forward_backward_custom.",
    )
    loss_fn_config: dict[str, float] = Field(
        default_factory=dict[str, float],
        description="The built-in loss's settings, which the cookbook hands to Loops as "
        "they are. A key left unset keeps Loops' default. Each loss's settings follow, "
        f"with defaults in parentheses. {_loss_fn_config_keys()}.",
    )
    custom_loss: CustomLossSettings | None = Field(
        default=None, description='Your own loss. It is required when loss_fn = "custom".'
    )
    loss_aggregation: Literal["sum", "token_mean", "fixed_budget", "task_mean"] = Field(
        default="sum",
        description="How the built-in loss is reduced over a complete optimizer update. "
        "sum is Loops' default, and it divides by nothing. token_mean divides by the "
        "update's action tokens that have a nonzero advantage. fixed_budget divides by "
        "group_size * groups_per_step * loss_token_budget. task_mean averages the task "
        "groups' means over action tokens. A custom loss reduces inside its own function "
        "and keeps sum.",
    )
    loss_token_budget: int | None = Field(
        default=None,
        ge=1,
        description="The action-token budget for one whole episode, across all its turns, "
        "that fixed_budget aggregation divides by. It is required with loss_aggregation "
        "= fixed_budget and refused with any other. It is a normalization constant, not "
        "a generation limit.",
    )
    router_replay: bool = Field(
        default=False,
        description="Whether to replay the expert routes of eligible batches, which is "
        "router replay (R3). It is off by default.",
    )
    max_consecutive_unusable_groups: int = Field(
        default=16,
        ge=0,
        description="The run aborts after this many groups in a row are rejected for "
        "training. 0 turns the check off. The count ignores infrastructure failures. It "
        "catches a wave of agent failures that were graded as zeros.",
    )
    max_steps_off_policy: int | None = Field(
        default=None,
        ge=0,
        description="In async mode, how many updates old a rollout's policy may be when "
        "the rollout trains. It is required with mode = async. Sync trains only on the "
        "current policy, so sync refuses it.",
    )
    admission_steps_ahead: float | None = Field(
        default=None,
        gt=0,
        description="In async mode, how many batches beyond the one in training that "
        "admission starts groups for. It may be at most max_steps_off_policy, which stays "
        "the staleness cap. A fraction admits part of a batch. With a lower value, groups "
        "wait less for the trainer and train closer to the current policy. A group that "
        "runs long still trains, up to the cap, and its finished members' slots go to new "
        "groups in the meantime. When unset, it equals max_steps_off_policy.",
    )

    @model_validator(mode="before")
    @classmethod
    def _removed_settings(cls, data: object) -> object:
        # Without this check, a config that sets it reads as a typo ("extra inputs")
        # instead of as a setting that moved into the scheduler.
        if isinstance(data, dict) and "admission_headroom" in data:
            raise ValueError(
                "admission_headroom was removed. Async admission always starts "
                "replacements for the groups that recent drops say won't train, up to a "
                "batch of them. Delete the setting. admission_steps_ahead sets how far "
                "ahead groups start. See docs/reference/scheduling.md."
            )
        return data

    max_policy_delta_in_trajectory: int | None = Field(
        default=None,
        ge=0,
        description="In async mode, how many updates apart the policy versions inside one "
        "rollout may be. It is required with mode = async. Sync samples each rollout from "
        "one policy, so sync refuses it.",
    )

    @model_validator(mode="after")
    def _losses_leave_a_rollout(self) -> Self:
        if self.max_group_losses >= self.group_size:
            raise ValueError(
                "max_group_losses must leave at least one rollout, so it has to be "
                "smaller than group_size"
            )
        return self

    @model_validator(mode="after")
    def _loss_settings(self) -> Self:
        if self.loss_fn == "custom":
            if self.custom_loss is None:
                raise ValueError('loss_fn = "custom" needs a [training.custom_loss] table')
            if self.loss_fn_config:
                raise ValueError(
                    "loss_fn_config holds a built-in loss's settings. A custom loss takes "
                    "its settings in [training.custom_loss] config"
                )
            return self
        if self.custom_loss is not None:
            raise ValueError('[training.custom_loss] applies only when loss_fn = "custom"')
        try:
            LOSS_FN_CONFIGS[self.loss_fn].model_validate(self.loss_fn_config)
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or 'loss_fn_config'}: {e['msg']}"
                for e in error.errors()
            )
            raise ValueError(f"loss_fn_config for {self.loss_fn}: {problems}") from None
        return self

    @model_validator(mode="after")
    def _aggregation_settings(self) -> Self:
        if self.loss_aggregation == "fixed_budget":
            if self.loss_token_budget is None:
                raise ValueError("fixed_budget aggregation requires loss_token_budget")
        elif self.loss_token_budget is not None:
            raise ValueError("loss_token_budget applies only to fixed_budget aggregation")
        if self.loss_aggregation == "sum":
            return self
        if self.loss_fn == "custom":
            raise ValueError(
                "loss_aggregation doesn't apply to a custom loss. Loops sums what the "
                "loss returns, so reduce inside the loss function"
            )
        if self.loss_aggregation != "token_mean" and self.loss_fn not in {
            "importance_sampling",
            "ppo",
            "cispo",
        }:
            raise ValueError(
                f"{self.loss_aggregation} applies only to importance_sampling, ppo and "
                "cispo. dro and dppo have penalty terms that scaling the advantages "
                "wouldn't reweight, so use sum or token_mean with them"
            )
        return self

    @model_validator(mode="after")
    def _staleness_is_async(self) -> Self:
        # Sync admits only the groups its batch takes, so no group crosses an update.
        # The limits are fixed at 0 in sync, and a stale sync rollout fails loudly.
        # Async samples the next batch while the current one trains, so it needs at
        # least one version of slack and both limits stated. The check lives here, not
        # on the field, so a sync config holding 0 is told to remove the key, not to
        # raise it.
        limits = ("max_steps_off_policy", "max_policy_delta_in_trajectory")
        given = [name for name in limits if getattr(self, name) is not None]
        if self.mode == "sync" and given:
            raise ValueError(
                f'{" and ".join(given)} applies only to mode = "async". Sync trains each '
                "rollout on the policy that sampled it. Remove them from the config, "
                "because a --training.mode sync flag can't remove them"
            )
        missing = [name for name in limits if getattr(self, name) is None]
        if self.mode == "async" and missing:
            raise ValueError(f'mode = "async" requires {" and ".join(missing)}')
        if self.mode == "async" and self.max_steps_off_policy == 0:
            raise ValueError(
                'mode = "async" needs max_steps_off_policy >= 1, because a rollout '
                "sampled while a batch trains is at least one update old"
            )
        return self

    @model_validator(mode="after")
    def _admission_within_the_cap(self) -> Self:
        if self.admission_steps_ahead is None:
            return self
        if self.mode == "sync":
            raise ValueError(
                'admission_steps_ahead applies only to mode = "async". Sync admits the '
                "next batch only once the last one has trained"
            )
        if self.admission_steps_ahead > self.steps_off_policy:
            raise ValueError(
                "admission_steps_ahead must not exceed max_steps_off_policy, because a "
                "group admitted for a batch further ahead is stale by the time that batch "
                "trains"
            )
        return self

    @property
    def steps_off_policy(self) -> int:
        """How many updates old a rollout's policy may be when it trains. It is 0 in sync."""
        return self.max_steps_off_policy or 0

    @property
    def policy_delta_in_trajectory(self) -> int:
        """How many updates apart the policy versions in one rollout may be. It is 0 in sync."""
        return self.max_policy_delta_in_trajectory or 0

    @property
    def steps_admitted_ahead(self) -> float:
        """How many batches beyond the one in training async admission starts groups for."""
        if self.admission_steps_ahead is None:
            return self.steps_off_policy
        return self.admission_steps_ahead

    @property
    def rollouts_per_step(self) -> int:
        return self.group_size * self.groups_per_step

    @property
    def min_group_size(self) -> int:
        """The number of graded rollouts a group needs to train, from its loss tolerance."""
        return self.group_size - self.max_group_losses


class GradingSettings(_Section):
    timeout_seconds: float = Field(
        gt=0,
        description="The deadline for each grade() try and each group grader call. It is required.",
    )


class CleanupSettings(_Section):
    # These settings have defaults because a wrong value only fails loudly, and one
    # value suits every workload. The rule is in docs/reference/KNOB_GLOSSARY.md.
    timeout_seconds: float = Field(
        default=600,
        gt=0,
        description="The deadline for each finish(), abort() and close() call. It is "
        "generous because it covers a Harbor trial's teardown. A value set too long only "
        "delays the report of a hook that hangs.",
    )
    cancellation_grace_seconds: float = Field(
        default=5,
        gt=0,
        description="How long a cancelled rollout runtime hook may take to stop before the "
        "run fails.",
    )
    process_kill_grace_seconds: float = Field(
        default=10,
        gt=0,
        description="How long an agent process may take to exit after SIGTERM before it "
        "is killed. This time is spent before a cancelled hook's own grace starts.",
    )


class ProducerSpec(_Section):
    trainable: bool = Field(description="Whether this producer's generations receive gradient.")


class RolloutSettings(_Section):
    max_concurrent_rollouts: int = Field(
        ge=1, description="How many rollouts may hold sampling capacity at once."
    )
    max_concurrent_verifiers: int = Field(
        ge=1, description="How many rollouts may hold verification capacity at once."
    )
    prestart_allowance: float = Field(
        default=0.1,
        ge=0,
        description="How many admitted rollouts may start their environment before a "
        "sampling slot is free, as a fraction of max_concurrent_rollouts. The aim is "
        "enough ready environments that a freed slot is taken at once, without paying for "
        "many idle ones.",
    )
    max_concurrent_provisioning: int = Field(
        default=8,
        ge=1,
        description="The run-wide limit on provider operations in flight, which are "
        "creates, reconciliation, releases and absence checks. Each provider's create "
        "window also starts at this value and never grows past it. Refusals from a "
        "provider narrow its window.",
    )
    max_consecutive_rollout_failures: int = Field(
        default=64,
        ge=0,
        description="The run aborts after this many failures in a row in one stage. The "
        "stages are sampling infrastructure, grading infrastructure, cleanup "
        "infrastructure and a group grader. 0 turns the check off.",
    )
    shuffle_seed: int = Field(default=0, description="The seed for the task order.")
    shutdown_drain_seconds: float = Field(
        default=120,
        gt=0,
        description="How long in-flight attempts get to settle on shutdown. The run's "
        "resource creates and releases then get the same time again to finish. Whatever "
        "is left is cut off, given a few seconds to stop, and reported for `loops "
        "cleanup`. An operation that ignores the cut-off is reported, and it holds the "
        "process until it stops.",
    )
    producers: dict[str, ProducerSpec] = Field(
        default={DEFAULT_PRODUCER: ProducerSpec(trainable=True)},
        description="Every producer a rollout runtime may use, by name, and whether its "
        "generations receive gradient. A call that names no producer uses main, and the "
        "run refuses an undeclared name. Setting this table replaces the default, so list "
        "main if calls rely on it.",
    )
    rollout_timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        description="A backstop on how long a rollout may hold sampling capacity, not "
        "counting resource-service time such as a sandbox create. A rollout runtime "
        "bounds its own agent below this value. When unset, the run uses the clock the "
        "rollout runtime derives, and an Episode derives one from its agent deadline. It "
        "is required for a rollout runtime that derives no clock.",
    )
    setup_timeout_seconds: float = Field(
        gt=0,
        description="The bound on the time an attempt spends before sealing while it "
        "holds neither sampling nor verification capacity, such as preparing an "
        "environment. It is required.",
    )
    max_rollout_retries: int = Field(
        default=0,
        ge=0,
        description="How many extra attempts a rollout may use after an episode failure, "
        "before it has a valid rollout. Use it as a last resort. Rollout runtimes retry "
        "their own steps, and the run never retries model-call failures here because the "
        "Loops SDK already retried them.",
    )
    grading: GradingSettings = Field(description="The deadline for each grading try.")
    cleanup: CleanupSettings = Field(
        default_factory=CleanupSettings, description="Cleanup deadlines for rollout runtime hooks."
    )

    @model_validator(mode="after")
    def _producers(self) -> Self:
        if any(not name for name in self.producers):
            raise ValueError("producer names must be nonempty")
        if not any(spec.trainable for spec in self.producers.values()):
            raise ValueError("[rollout.producers] must declare at least one trainable producer")
        return self

    @property
    def prestart_places(self) -> int:
        """How many admitted rollouts may start their environment ahead of a sampling slot."""
        return math.ceil(self.prestart_allowance * self.max_concurrent_rollouts)


class ComputeSettings(settings.ComputeSettings):
    sampler_replica_affinity: bool = Field(
        default=True,
        description="Whether to pin a rollout's requests to one sampler replica, so that "
        "its growing prompt keeps hitting a warm prefix cache. It is a routing hint. "
        "Where Loops can't pin a request, it serves the request without affinity.",
    )
    sampler_replicas: int | None = Field(
        default=None,
        ge=1,
        description="A fixed replica count for the paired sampler. When unset, the Loops "
        "backend default applies.",
        json_schema_extra=settings.DEDICATED_ONLY,
    )


class OutputSettings(settings.OutputSettings):
    progress_every_seconds: float = Field(
        default=60,
        ge=0,
        description="Seconds between progress reports while a batch collects. 0 turns "
        "the reports off. Optimizer steps can be tens of minutes apart, and nothing else "
        "reports in between. Without these reports, a long collection phase and a stall "
        "look the same. Every slots, queue and admission series in W&B comes from them.",
    )
    rollout_metrics_limit: int = Field(
        default=32,
        ge=1,
        description="The most rollout/* metrics a step reports from what the agents return "
        "about their episodes. Each label value counts as one metric. Past the limit, the "
        "run drops the keys that the fewest rollouts reported, and rollout/dropped_keys "
        "counts them. This stops a stray per-rollout ID from burying the rest. Raise the "
        "limit when an agent reports more.",
    )
    trace: Literal["off", "samples", "transcripts", "full"] = Field(
        default="off",
        description="The per-generation evidence to keep beside the run record. off "
        "keeps none. samples keeps one row per generation without token arrays, at about "
        "1% of the size of full. transcripts keeps those rows and each attempt's "
        "transcript, so you can read what the policy saw and said. A transcript holds "
        "every generation's request as the agent sent it and the reply it received, each "
        "marked with whether it is trainable. full keeps the transcripts, the token "
        "arrays and the exact training rows, for replay and audit. The run writes every "
        "attempt's transcript, and a failed attempt's as far as it got. Recording never "
        "changes what trains. A request or reply that isn't JSON, such as bytes or a set "
        "in a direct agent's history, is written as unrecordable.",
    )


EVAL_SET_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class EvalRuntimeSettings(_Section):
    files: dict[str, str] = Field(
        min_length=1,
        description="The [rollout_runtime.files] entries this set replaces. The rollout "
        "runtime, its config and every other file stay the same as training's.",
    )


class EvalSamplingSettings(_Section):
    temperature: float | None = Field(
        default=None,
        ge=0,
        description="The sampling temperature for this set. When unset, it is "
        "[sampling].temperature.",
    )


class EvalSetSettings(_Section):
    """One eval set's settings.

    A key at the path of a training setting overrides that setting for this set. The
    other keys belong to the set alone.
    """

    rollout_runtime: EvalRuntimeSettings = Field(
        description="Overrides [rollout_runtime] for this set."
    )

    @model_validator(mode="before")
    @classmethod
    def _renamed_runtime(cls, data: object) -> object:
        return _refuse_renamed_runtime(data, "eval.sets.<name>.")

    sampling: EvalSamplingSettings = Field(
        default_factory=EvalSamplingSettings, description="Overrides [sampling] for this set."
    )
    samples_per_task: int = Field(
        default=1, ge=1, description="The number of rollouts for each task."
    )
    max_tasks: int | None = Field(
        default=None,
        ge=1,
        description="Evaluate only the first N tasks. When unset, the set evaluates every task.",
    )
    max_concurrent_rollouts: int = Field(
        default=16,
        ge=1,
        description="How many of this set's rollouts may run at once, on top of "
        "training's. The set shares training's sandbox providers, so those providers see "
        "training's limit plus every set's.",
    )


class EvalSettings(_Section):
    every_steps: int = Field(
        default=0,
        ge=0,
        description="Evaluate every N updates. 0 evaluates only before the first update "
        "and after the last. Evaluating after the last update needs [training] max_steps.",
    )
    sampler_start_timeout_seconds: float = Field(
        default=900,
        gt=0,
        description="How long an eval's sampler may take to become ready before the run "
        "skips that set's eval. On dedicated Loops, each eval starts a sampler deployment.",
    )
    sets: dict[str, EvalSetSettings] = Field(
        default_factory=dict[str, EvalSetSettings],
        description="Held-out task sets, by name. With no sets, the run runs no evals.",
    )

    @field_validator("sets")
    @classmethod
    def _names(cls, sets: dict[str, EvalSetSettings]) -> dict[str, EvalSetSettings]:
        bad = sorted(name for name in sets if not EVAL_SET_NAME.match(name))
        if bad:
            raise ValueError(
                "eval set names go into paths and metric names, so they are lowercase "
                f"letters, digits, '_' and '-': {', '.join(bad)}"
            )
        return sets

    def due(self, step: int, *, final: bool) -> bool:
        """Whether update ``step`` is evaluated, either on the cadence or as the final update."""
        if not self.sets:
            return False
        return final or bool(self.every_steps) and (step + 1) % self.every_steps == 0


class ResourcesSettings(_Section):
    state_dir: str = Field(
        default="state",
        min_length=1,
        description="The directory of the resource journal, relative to the run directory. "
        "It must be on a local filesystem.",
    )


class Config(_Section):
    """The settings for an RL run, with one field for each table of the config file.

    ``loops check`` and ``loops run`` load it from the TOML file and the command-line
    flags. Validation refuses settings that can't work together before anything is
    provisioned, such as a ``max_trajectory_tokens`` above the model's ``max_seq_len``.
    """

    recipe: Literal["rl"] = Field(
        default="rl",
        description=RECIPE_DESCRIPTION,
        json_schema_extra={"cli": False},
    )
    rollout_runtime: RuntimeSettings = Field(description="Where episodes come from.")
    model: ModelSettings = Field(description="The policy being trained.")
    sampling: SamplingSettings = Field(description="Defaults for what the policy produces.")
    training: TrainingSettings = Field(description="The optimizer and the batch.")
    rollout: RolloutSettings = Field(description="Concurrency, deadlines and producers.")
    compute: ComputeSettings = Field(description="Where the run executes.")
    output: OutputSettings = Field(description="What the run leaves behind.")
    resources: ResourcesSettings = Field(
        default_factory=ResourcesSettings, description="Where resource ownership is journaled."
    )
    eval: EvalSettings = Field(
        default_factory=EvalSettings, description="Held-out evals during training."
    )
    providers: dict[str, dict[str, JsonValue]] = Field(
        default_factory=dict[str, dict[str, JsonValue]],
        description="Resource providers, by name. Each table holds that provider's "
        "settings and deadlines, never credentials. A provider that isn't built in names "
        "its adapter factory as adapter = module:callable.",
    )

    @model_validator(mode="before")
    @classmethod
    def _renamed_runtime(cls, data: object) -> object:
        return _refuse_renamed_runtime(data, "")

    @model_validator(mode="after")
    def _trajectory_fits_the_trainer(self) -> Self:
        # The cap and max_seq_len are two halves of one decision. The cap is how much
        # you want the agent to build, and max_seq_len is how much room the trainer was
        # given. A cap above capacity fails at training time, after the rollout is paid
        # for.
        cap, capacity = self.sampling.max_trajectory_tokens, self.model.max_seq_len
        if cap is not None and capacity is not None and cap > capacity:
            raise ValueError(
                f"max_trajectory_tokens ({cap}) exceeds max_seq_len ({capacity}), so the "
                "trainer would refuse the datum"
            )
        return self

    @model_validator(mode="after")
    def _eval_files_are_runtime_files(self) -> Self:
        known = sorted(self.rollout_runtime.files)
        for name, eval_set in self.eval.sets.items():
            unknown = sorted(set(eval_set.rollout_runtime.files) - set(known))
            if unknown:
                raise ValueError(
                    f"eval set {name!r} names file(s) {', '.join(unknown)}, which "
                    f"[rollout_runtime.files] does not have ({', '.join(known) or 'none'})"
                )
        return self

    @model_validator(mode="after")
    def _replay_records_text_only(self) -> Self:
        # Router replay names the sampler's record of every generation, and Loops
        # records only tokenized text prompts. An image prompt would fail to sample.
        if self.rollout_runtime.has_images and self.training.router_replay:
            raise ValueError(
                "router_replay records every generation, and Loops cannot record image "
                "prompts. Set [training] router_replay = false when [rollout_runtime] has_images "
                "= true."
            )
        return self

    @property
    def trajectory_tokens(self) -> int | None:
        """The cap to enforce on each trajectory, which is the trainer's capacity when unset."""
        return self.sampling.max_trajectory_tokens or self.model.max_seq_len
