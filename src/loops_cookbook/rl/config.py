"""Training configuration: one TOML file, one nested model per section.

The runtime's own settings are opaque application data under ``[rollout_runtime.config]``;
nothing here interprets them. Files a runtime needs resolved are named explicitly in
``[rollout_runtime.files]``. Secrets never appear here: each SDK, runtime, or adapter reads
the variables it needs from the environment, loaded from ``.env``.
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
    # Otherwise an old config's [runtime] reads as a typo ("extra inputs"), not as a
    # table that was renamed.
    if isinstance(data, dict) and "runtime" in data:
        raise ValueError(
            f"[{table}runtime] is now [{table}rollout_runtime]. Rename the table; its "
            "keys are unchanged."
        )
    return data


class OpenAISettings(_Section):
    bind: str = Field(
        default="127.0.0.1:0",
        pattern=r"^.+:[0-9]+$",
        description="Where the gateway listens, as host:port; loopback and an ephemeral "
        "port by default.",
    )
    advertised_base_url: str | None = Field(
        default=None,
        pattern=r"^https?://[^/]+/?$",
        description="The origin agents use, such as http://127.0.0.1:43210; default is the "
        "bound loopback address.",
    )
    model_name: str | None = Field(
        default=None,
        min_length=1,
        description="The one model the gateway advertises; default is [model].base_model.",
    )


class RuntimeSettings(_Section):
    factory: str = Field(
        min_length=1,
        description="Rollout runtime factory as module:callable. Harbor: "
        "loops_cookbook.harbor:create_runtime; a custom rollout runtime: its module beside "
        "this file.",
    )
    model_access: Literal["direct", "openai_http"] | None = Field(
        default=None,
        description="How the agent reaches the model. direct: the cookbook's llm() client, "
        "for in-process agents; openai_http: an OpenAI-compatible Chat Completions "
        "endpoint per attempt. Unset: the one the rollout runtime declares (an Episode class's "
        "model_access, a Harbor job's agents'); set only for a rollout runtime that supports "
        "several.",
    )
    openai: OpenAISettings | None = Field(
        default=None,
        description="The Chat Completions gateway; only with model_access = openai_http.",
    )
    config: dict[str, JsonValue] = Field(
        default_factory=dict[str, JsonValue],
        description="Opaque rollout runtime settings, handed to the factory unchanged.",
    )
    files: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="Files the rollout runtime may resolve, as alias = path. Relative to this "
        "config file, or to the shell when passed as a command-line flag.",
    )
    has_images: bool = Field(
        default=False,
        description="Whether the agent's prompts carry images. Loops cannot record an "
        "image prompt, so router replay needs this false.",
    )

    @model_validator(mode="after")
    def _gateway(self) -> Self:
        if self.openai is not None and self.model_access == "direct":
            raise ValueError(
                '[rollout_runtime.openai] applies only when model_access = "openai_http"'
            )
        return self


# The Chat Completions ``reasoning_effort`` values, as the pinned OpenAI schema
# enumerates them; direct access accepts the same set, so both paths agree.
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class ModelSettings(settings.ModelSettings):
    reasoning_efforts: dict[str, dict[str, JsonValue]] = Field(
        default_factory=dict[str, dict[str, JsonValue]],
        description="Accepted reasoning_effort values, each mapped to the template "
        "options it overrides; a request selects one, and an undeclared value is "
        "rejected.",
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
        description="Default sampling temperature for a generation that names none; a "
        "request may set its own.",
    )
    max_generation_tokens: int = Field(
        ge=1,
        description="Default length limit for a generation that names none; a request "
        "may set its own, so this is not a cap. Either is lowered to the room the "
        "trajectory budget leaves.",
    )
    max_trajectory_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Trajectory budget, in tokens, for prompt plus generation: each "
        "call's max_tokens is lowered to the room its prompt leaves, and only a prompt "
        "that leaves none is refused. Over HTTP it always applies; a direct call may "
        "pass its own. Defaults to max_seq_len, which is the capacity the trainer was "
        "configured for.",
    )


class ImportanceSamplingConfig(_Section):
    """``-ratio * advantage``; it takes no settings."""


class ClipConfig(_Section):
    """PPO and CISPO clip the importance ratio to [low, high]."""

    clip_low_threshold: float = Field(default=0.8, ge=0, description="Lowest ratio kept.")
    clip_high_threshold: float = Field(default=1.2, ge=0, description="Highest ratio kept.")

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.clip_high_threshold < self.clip_low_threshold:
            raise ValueError("clip_high_threshold must be at least clip_low_threshold")
        return self


class DROConfig(_Section):
    """``logprob * advantage - beta / 2 * log_ratio**2``."""

    beta: float = Field(default=0.05, ge=0, description="Weight of the log-ratio penalty.")


class DPPOConfig(_Section):
    """A probability-difference trust region, plus a squared log-ratio KL."""

    dppo_mask_high: float = Field(
        default=0.2, ge=0, description="Largest probability rise a positive advantage keeps."
    )
    dppo_mask_low: float = Field(
        default=0.2, ge=0, description="Largest probability fall a negative advantage keeps."
    )
    adv_tau: float = Field(default=1.0, description="Scale of the advantage.")
    kl_tau: float = Field(default=1e-3, description="Weight of the KL term.")


# The built-in losses whose inputs are GRPO's (logprobs, advantages, a loss mask), each
# with its settings as the Loops trainer validates them, defaults included. Loops itself
# ignores a key it does not know; these refuse it.
LOSS_FN_CONFIGS: dict[str, type[_Section]] = {
    "importance_sampling": ImportanceSamplingConfig,
    "ppo": ClipConfig,
    "cispo": ClipConfig,
    "dro": DROConfig,
    "dppo": DPPOConfig,
}


def _loss_fn_config_keys() -> str:
    """Each built-in loss's settings with their defaults, for the setting's description."""
    return "; ".join(
        f"{name}: "
        + (
            ", ".join(f"{key} ({info.default:g})" for key, info in model.model_fields.items())
            or "none"
        )
        for name, model in LOSS_FN_CONFIGS.items()
    )


class CustomLossSettings(_Section):
    factory: str = Field(
        min_length=1,
        description="Loss factory as module:callable, such as a module beside this file; "
        "it takes config and returns the loss function forward_backward_custom runs. "
        "Needs the custom-loss extra.",
    )
    config: dict[str, JsonValue] = Field(
        default_factory=dict[str, JsonValue],
        description="The loss's own settings, handed to the factory unchanged.",
    )


class TrainingSettings(_TrainingSettings):
    mode: Literal["sync", "async"] = Field(
        default="sync",
        description="sync: one batch at a time. async: overlap collection and training.",
    )
    group_size: int = Field(ge=2, description="Rollouts sampled per task group.")
    max_group_losses: int = Field(
        ge=0,
        description="Rollouts a group may lose and still train. A tolerance rather "
        "than a floor, so it keeps its meaning when group_size changes.",
    )
    groups_per_step: int = Field(ge=1, description="Groups per optimizer update.")
    loss_fn: Literal["importance_sampling", "ppo", "cispo", "dro", "dppo", "custom"] = Field(
        description="The loss forward_backward computes: one of Loops', or custom, your "
        "own from [training.custom_loss], run through forward_backward_custom.",
    )
    loss_fn_config: dict[str, float] = Field(
        default_factory=dict[str, float],
        description="The built-in loss's settings, handed to Loops as they are; a key "
        f"left unset keeps Loops' default. {_loss_fn_config_keys()}.",
    )
    custom_loss: CustomLossSettings | None = Field(
        default=None, description='Your own loss; required with loss_fn = "custom".'
    )
    loss_aggregation: Literal["sum", "token_mean", "fixed_budget", "task_mean"] = Field(
        default="sum",
        description="Built-in loss reduction over a complete optimizer update. sum is "
        "Loops' default: no division; token_mean divides by the update's action tokens "
        "with nonzero advantage; fixed_budget divides by group_size * groups_per_step * "
        "loss_token_budget; task_mean averages each task group's action-token mean. "
        "Custom losses reduce in their own function and keep sum.",
    )
    loss_token_budget: int | None = Field(
        default=None,
        ge=1,
        description="Action-token budget per whole episode for fixed_budget aggregation, "
        "across all turns; required only with loss_aggregation = fixed_budget. "
        "A normalization constant, not a generation limit.",
    )
    router_replay: bool = Field(
        default=False, description="Replay eligible batches' expert routes (R3); off by default."
    )
    max_consecutive_unusable_groups: int = Field(
        default=16,
        ge=0,
        description="Abort after this many consecutive groups rejected for training; 0 "
        "disables. Independent of infrastructure failures: it catches a wave of agent "
        "failures graded as zeros.",
    )
    max_steps_off_policy: int | None = Field(
        default=None,
        ge=0,
        description="Async: oldest policy version, in updates, a rollout may train on. "
        "Required with mode = async; sync trains only on the current policy and refuses it.",
    )
    admission_steps_ahead: float | None = Field(
        default=None,
        gt=0,
        description="Async: batches beyond the one training that admission starts groups "
        "for, at most max_steps_off_policy, which stays the staleness cap; fractions "
        "admit part of a batch. Lower, groups wait less for the trainer and train closer "
        "to the current policy, while one that runs long still trains up to the cap, its "
        "finished members' slots lent to new groups meanwhile. Unset: "
        "max_steps_off_policy.",
    )

    @model_validator(mode="before")
    @classmethod
    def _removed_settings(cls, data: object) -> object:
        # Otherwise a config that set it reads as a typo ("extra inputs"), not as a
        # knob that moved into the scheduler.
        if isinstance(data, dict) and "admission_headroom" in data:
            raise ValueError(
                "admission_headroom was removed: async admission always starts "
                "replacements for the groups recent drops say will not train, up to a "
                "batch of them. Delete it; admission_steps_ahead sets how far ahead "
                "groups start. See docs/reference/scheduling.md."
            )
        return data

    max_policy_delta_in_trajectory: int | None = Field(
        default=None,
        ge=0,
        description="Async: policy-version spread allowed inside one rollout. Required "
        "with mode = async; sync samples each rollout from one policy and refuses it.",
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
                    "loss_fn_config is a built-in loss's; a custom loss takes its settings "
                    "in [training.custom_loss] config"
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
                "loss_aggregation does not reach a custom loss: Loops sums what it "
                "returns, so reduce inside the loss function"
            )
        if self.loss_aggregation != "token_mean" and self.loss_fn not in {
            "importance_sampling",
            "ppo",
            "cispo",
        }:
            raise ValueError(
                f"{self.loss_aggregation} applies only to importance_sampling, ppo and "
                "cispo: dro and dppo have penalty terms advantage scaling would not "
                "reweight; use sum or token_mean"
            )
        return self

    @model_validator(mode="after")
    def _staleness_is_async(self) -> Self:
        # Sync admits only the groups its batch takes, so none crosses an update: the
        # limits are fixed at 0 there, and a sync rollout that is stale fails loudly.
        # Async samples the next batch while the current one trains, so it needs at
        # least one version of slack and both limits stated. Checked here, not by the
        # field, so a sync config holding 0 is told to remove the key, not raise it.
        limits = ("max_steps_off_policy", "max_policy_delta_in_trajectory")
        given = [name for name in limits if getattr(self, name) is not None]
        if self.mode == "sync" and given:
            raise ValueError(
                f'{" and ".join(given)} applies only to mode = "async": sync trains each '
                "rollout on the policy that sampled it. Remove them from the config; a "
                "--training.mode sync flag cannot"
            )
        missing = [name for name in limits if getattr(self, name) is None]
        if self.mode == "async" and missing:
            raise ValueError(f'mode = "async" requires {" and ".join(missing)}')
        if self.mode == "async" and self.max_steps_off_policy == 0:
            raise ValueError(
                'mode = "async" needs max_steps_off_policy >= 1: a rollout sampled while '
                "a batch trains is at least one update old"
            )
        return self

    @model_validator(mode="after")
    def _admission_within_the_cap(self) -> Self:
        if self.admission_steps_ahead is None:
            return self
        if self.mode == "sync":
            raise ValueError(
                'admission_steps_ahead applies only to mode = "async": sync admits the '
                "next batch only once the last has trained"
            )
        if self.admission_steps_ahead > self.steps_off_policy:
            raise ValueError(
                "admission_steps_ahead must not exceed max_steps_off_policy: a group "
                "admitted for a batch further ahead is stale by the time that batch trains"
            )
        return self

    @property
    def steps_off_policy(self) -> int:
        """Oldest policy version, in updates, a rollout may train on; 0 in sync."""
        return self.max_steps_off_policy or 0

    @property
    def policy_delta_in_trajectory(self) -> int:
        """Policy-version spread allowed inside one rollout; 0 in sync."""
        return self.max_policy_delta_in_trajectory or 0

    @property
    def steps_admitted_ahead(self) -> float:
        """Batches beyond the one training that async admission starts groups for."""
        if self.admission_steps_ahead is None:
            return self.steps_off_policy
        return self.admission_steps_ahead

    @property
    def rollouts_per_step(self) -> int:
        return self.group_size * self.groups_per_step

    @property
    def min_group_size(self) -> int:
        """Graded rollouts a group needs to train, derived from its loss tolerance."""
        return self.group_size - self.max_group_losses


class GradingSettings(_Section):
    timeout_seconds: float = Field(
        gt=0, description="Deadline for each grade() try, and each group grader call. Required."
    )


class CleanupSettings(_Section):
    # Defaults, because a wrong value only fails loudly and one value suits every
    # workload; see the rule in docs/reference/KNOB_GLOSSARY.md.
    timeout_seconds: float = Field(
        default=600,
        gt=0,
        description="Deadline for each finish(), abort(), and close(). Generous: it "
        "covers a Harbor trial's teardown, and one set too long only delays the report "
        "of a hook that hangs.",
    )
    cancellation_grace_seconds: float = Field(
        default=5,
        gt=0,
        description="How long a cancelled rollout runtime hook may take to stop before the run fails.",
    )
    process_kill_grace_seconds: float = Field(
        default=10,
        gt=0,
        description="How long an agent process may take to exit after SIGTERM before it "
        "is killed; spent before a cancelled hook's own grace starts.",
    )


class ProducerSpec(_Section):
    trainable: bool = Field(description="Whether this producer's generations receive gradient.")


class RolloutSettings(_Section):
    max_concurrent_rollouts: int = Field(
        ge=1, description="Concurrent rollouts holding sampling capacity."
    )
    max_concurrent_verifiers: int = Field(
        ge=1, description="Concurrent rollouts holding verification capacity."
    )
    prestart_allowance: float = Field(
        default=0.1,
        ge=0,
        description="Admitted rollouts that may start their environment before a sampling "
        "slot is free, as a fraction of max_concurrent_rollouts: enough ready "
        "environments that a freed slot is taken at once, without paying for many idle.",
    )
    max_concurrent_provisioning: int = Field(
        default=8,
        ge=1,
        description="Run-wide limit on provider operations in flight: creates, "
        "reconciliation, releases, and absence checks. Also where each provider's "
        "create window starts, and its cap: refusals narrow it.",
    )
    max_consecutive_rollout_failures: int = Field(
        default=64,
        ge=0,
        description="Abort after this many consecutive failures in one stage: sampling, "
        "grading and cleanup infrastructure, or a group grader; 0 disables.",
    )
    shuffle_seed: int = Field(default=0, description="Seed for the task order.")
    shutdown_drain_seconds: float = Field(
        default=120,
        gt=0,
        description="Time allowed for in-flight attempts to settle on shutdown, and again "
        "for the run's resource creates and releases to finish; what is left is cut off, "
        "given a few seconds to stop, and reported for `loops cleanup`. An operation "
        "that ignores being cut off is reported, and holds the process until it stops.",
    )
    producers: dict[str, ProducerSpec] = Field(
        default={DEFAULT_PRODUCER: ProducerSpec(trainable=True)},
        description="Every producer a rollout runtime may use, by name, and whether its generations "
        "receive gradient; a call that names no producer uses main, and an undeclared name "
        "is refused. Setting the table replaces the default, so list main if calls rely "
        "on it.",
    )
    rollout_timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Backstop on a rollout's time holding sampling capacity, less "
        "resource-service time such as a sandbox create; a rollout runtime bounds its own agent "
        "below it. Unset: the rollout runtime's derived clock (Episodes derive one from their "
        "agent deadline); required for a rollout runtime that derives none.",
    )
    setup_timeout_seconds: float = Field(
        gt=0,
        description="Bound on an attempt's time before sealing outside sampling and "
        "verification capacity. Required.",
    )
    max_rollout_retries: int = Field(
        default=0,
        ge=0,
        description="Extra attempts a rollout may use after an episode failure before it "
        "has a valid rollout. A last resort: rollout runtimes retry their own steps, and model-call "
        "failures are never retried here because the Loops SDK already retried them.",
    )
    grading: GradingSettings = Field(description="Per-try grading deadline.")
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
        """Admitted rollouts that may start their environment ahead of a sampling slot."""
        return math.ceil(self.prestart_allowance * self.max_concurrent_rollouts)


class ComputeSettings(settings.ComputeSettings):
    sampler_replica_affinity: bool = Field(
        default=True,
        description="Pin a rollout's requests to one sampler replica so its growing "
        "prompt keeps hitting a warm prefix cache. A routing hint: where Loops cannot pin "
        "a request, it is served without affinity.",
    )
    sampler_replicas: int | None = Field(
        default=None,
        ge=1,
        description="Fixed replica count for the paired sampler. Unset: the Loops backend default.",
        json_schema_extra=settings.DEDICATED_ONLY,
    )


class OutputSettings(settings.OutputSettings):
    progress_every_seconds: float = Field(
        default=60,
        ge=0,
        description="Seconds between progress reports while a batch collects; 0 is off. "
        "Optimizer steps can be tens of minutes apart and nothing else reports in "
        "between, so without it a long collection phase and a stall look identical. "
        "Every slots, queue and admission series in W&B comes from these reports.",
    )
    rollout_metrics_limit: int = Field(
        default=32,
        ge=1,
        description="Most rollout/* metrics a step reports from what the agents return "
        "about their episodes, counting each label value as one. Past it, the keys the "
        "fewest rollouts reported are dropped (rollout/dropped_keys counts them), so a "
        "stray per-rollout id cannot bury the rest; raise it when an agent reports more.",
    )
    trace: Literal["off", "samples", "transcripts", "full"] = Field(
        default="off",
        description="Per-generation evidence to keep beside the run record. off: none. "
        "samples: one row per generation without token arrays, about 1% of full. "
        "transcripts: those rows and each attempt's transcript (every generation's "
        "request as the agent sent it and the reply it received, marked with whether it "
        "is trainable), to read what the policy saw and said. full: transcripts, token "
        "arrays and the exact training rows, for replay and audit. Every attempt's "
        "transcript is written, a failed one's as far as it got, and recording never "
        "changes what trains: a request or reply that is not JSON (bytes or a set in a "
        "direct agent's history) is written as unrecordable.",
    )


EVAL_SET_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class EvalRuntimeSettings(_Section):
    files: dict[str, str] = Field(
        min_length=1,
        description="Replaces these [rollout_runtime.files] entries for this set; the rollout runtime, "
        "its config and every other file are training's.",
    )


class EvalSamplingSettings(_Section):
    temperature: float | None = Field(
        default=None, ge=0, description="Unset: [sampling].temperature."
    )


class EvalSetSettings(_Section):
    """A key at the path of a training setting overrides it for this set; the others
    are the set's own."""

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
    samples_per_task: int = Field(default=1, ge=1, description="Rollouts per task.")
    max_tasks: int | None = Field(
        default=None, ge=1, description="Evaluate the first N tasks; unset: all."
    )
    max_concurrent_rollouts: int = Field(
        default=16,
        ge=1,
        description="This set's rollouts at once, on top of training's: they share "
        "training's sandbox providers, so those see training's limit plus every set's.",
    )


class EvalSettings(_Section):
    every_steps: int = Field(
        default=0,
        ge=0,
        description="Evaluate every N updates; 0 evaluates only before the first update "
        "and after the last. After the last needs [training] max_steps.",
    )
    sampler_start_timeout_seconds: float = Field(
        default=900,
        gt=0,
        description="How long an eval's sampler may take to be ready before that set's "
        "eval is skipped. On dedicated Loops each eval starts a sampler deployment.",
    )
    sets: dict[str, EvalSetSettings] = Field(
        default_factory=dict[str, EvalSetSettings],
        description="Held-out task sets, by name. None: no evals.",
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
        """Whether update ``step`` is evaluated: on the cadence, or the final one."""
        if not self.sets:
            return False
        return final or bool(self.every_steps) and (step + 1) % self.every_steps == 0


class ResourcesSettings(_Section):
    state_dir: str = Field(
        default="state",
        min_length=1,
        description="Directory of the resource journal, relative to the run directory. "
        "It must be on a local filesystem.",
    )


class Config(_Section):
    recipe: Literal["rl"] = Field(
        default="rl",
        description=RECIPE_DESCRIPTION,
        json_schema_extra={"cli": False},
    )
    rollout_runtime: RuntimeSettings = Field(description="Where episodes come from.")
    model: ModelSettings = Field(description="The policy being trained.")
    sampling: SamplingSettings = Field(description="Defaults for what the policy produces.")
    training: TrainingSettings = Field(description="The optimizer and the batch.")
    rollout: RolloutSettings = Field(description="Concurrency, deadlines, and producers.")
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
        description="Resource providers by name. Each table is that provider's settings "
        "and deadlines, never credentials; one that is not built in names its adapter "
        "factory as adapter = module:callable.",
    )

    @model_validator(mode="before")
    @classmethod
    def _renamed_runtime(cls, data: object) -> object:
        return _refuse_renamed_runtime(data, "")

    @model_validator(mode="after")
    def _trajectory_fits_the_trainer(self) -> Self:
        # The two halves of one decision: the cap is what you want the agent to
        # build, max_seq_len is what the trainer was given room for. A cap above
        # capacity fails at training time, after paying for the rollout.
        cap, capacity = self.sampling.max_trajectory_tokens, self.model.max_seq_len
        if cap is not None and capacity is not None and cap > capacity:
            raise ValueError(
                f"max_trajectory_tokens ({cap}) exceeds max_seq_len ({capacity}); the "
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
        # Replay names the sampler's record of every generation, and Loops records
        # tokenized text prompts only: an image prompt would fail to sample.
        if self.rollout_runtime.has_images and self.training.router_replay:
            raise ValueError(
                "router_replay records every generation, and Loops cannot record image "
                "prompts: set [training] router_replay = false with [rollout_runtime] has_images = true"
            )
        return self

    @property
    def trajectory_tokens(self) -> int | None:
        """The cap to enforce per trajectory; the trainer's capacity when unset."""
        return self.sampling.max_trajectory_tokens or self.model.max_seq_len
