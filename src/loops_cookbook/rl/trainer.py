"""The Loops trainer for RL, which accumulates gradients, steps, checkpoints and publishes."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from time import monotonic
from typing import Protocol

from baseten.loops.models import Datum, TensorData
from baseten.loops.sampling_client import SamplingClient

from loops_cookbook.rl.aggregation import (
    BUILTIN_NORMALIZATION_CONTRACT,
    AggregationReport,
    trainer_averages,
)
from loops_cookbook.rl.config import Config
from loops_cookbook.rl.grpo import TrainingRow
from loops_cookbook.rl.loss import forward_backward_custom, load_custom_loss
from loops_cookbook.rl.mismatch import MismatchTotals, UnfitScores, action_scores
from loops_cookbook.rl.slices import in_slices
from loops_cookbook.trainer import (
    Capacity,
    LoopsTrainer,
    check_token_normalization,
    token_normalization,
    updated_version,
)

logger = logging.getLogger(__name__)


class TrainerPort(Protocol):
    """What the runner needs from a trainer. ``Trainer`` is the Loops implementation.

    The protocol is internal and not exported. Tests and ``loops check`` pass ``Run``
    their own trainers.
    """

    sampler: SamplingClient | None
    sampler_version: int | None
    # The [compute] scope. A started run's record agrees with it.
    capacity: Capacity
    resumed_step: int

    @property
    def policy_version(self) -> int: ...
    @property
    def serverless(self) -> bool: ...
    async def start(self) -> SamplingClient: ...
    async def accumulate(self, rows: list[TrainingRow]) -> None: ...
    async def commit(
        self,
        step: int,
        *,
        groups: int,
        rollouts: int,
        rewards: list[float],
        advantages_: list[float],
        sampled_versions: list[int],
    ) -> StepReport: ...
    async def checkpoint(
        self, report: StepReport, *, final: bool = False, keep_sampler_weights: bool = False
    ) -> StepReport: ...
    async def checkpoint_state_only(self, step: int) -> str | None: ...
    async def run_ended(self) -> str | None: ...
    async def eval_sampler(
        self, model_path: str | None, *, ready_timeout: float
    ) -> SamplingClient: ...
    async def deactivate_sampler(self, sampler: SamplingClient) -> None: ...
    def connect(self) -> object: ...
    async def servable_weights(self, state_path: str, name: str) -> tuple[str, int]: ...
    async def deactivate_run(self) -> None: ...
    async def close(self) -> None: ...


@dataclass(frozen=True)
class ForwardBackwardReport:
    """One public SDK call, whose result may already aggregate transport chunks."""

    datums: int
    trained_positions: int
    routed: bool
    loss: float
    mismatch_kl: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    """Every finite numeric metric the trainer returned for the call, verbatim."""


@dataclass
class StepReport:
    """One optimizer step, as a flat record."""

    step: int
    policy_version_before: int
    policy_version_after: int
    groups: int
    rollouts: int
    datums: int
    tokens: int
    trained_positions: int
    loss: float
    reward_mean: float
    reward_std: float
    advantage_abs_mean: float
    sampled_version_min: int
    sampled_version_max: int
    forward_backward_calls: int = 1
    checkpoint: str | None = None
    """The name of the step checkpoint written after this step, when one was written."""
    checkpoint_state_path: str | None = None
    """The path to give back to ``resume_from``. The name alone is not enough."""
    sampler_weights: str | None = None
    """The servable path of the weights kept after this step, when they were kept."""
    mismatch_kl: float | None = None
    mismatch_kl_max_call: float | None = None
    forward_backward_reports: tuple[ForwardBackwardReport, ...] = ()
    optimizer_metrics: dict[str, float] = field(default_factory=dict)
    loss_metrics: dict[str, float] = field(default_factory=dict)
    """A custom loss's own metrics, averaged over datums."""
    trainer_metrics: dict[str, float] = field(default_factory=dict)
    """The trainer's own metrics for a built-in loss, averaged over datums.

    They include every metric other than ``mismatch_kl``, such as
    ``mismatch_abs_logprob_diff`` and ``mismatch_ess``.
    """
    timings: dict[str, float] = field(default_factory=dict)
    mismatch: dict[str, float] = field(default_factory=dict)
    """The trainer/sampler logprob gap on trained tokens, by turn and policy lag."""
    aggregation: AggregationReport | None = None
    """The whole-update reduction that the runner supplied, before transport chunking."""

    @property
    def trainable_fraction(self) -> float:
        """The share of sent tokens that carry gradient.

        A low share means long prompts and short completions. That is the cost of
        agentic tasks, and it is worth watching.
        """
        return self.trained_positions / self.tokens if self.tokens else 0.0

    def gauges(self) -> dict[str, float]:
        out = {
            **{f"time/{name}_seconds": value for name, value in self.timings.items()},
            "train/loss": self.loss,
            "train/groups": float(self.groups),
            "train/rollouts": float(self.rollouts),
            "train/datums": float(self.datums),
            "train/tokens": float(self.tokens),
            "train/trained_positions": float(self.trained_positions),
            "train/trainable_fraction": self.trainable_fraction,
            "train/policy_version": float(self.policy_version_after),
            "train/forward_backward_calls": float(self.forward_backward_calls),
            "train/checkpointed": float(self.checkpoint is not None),
            "reward/mean": self.reward_mean,
            "reward/std": self.reward_std,
            "advantage/abs_mean": self.advantage_abs_mean,
            "freshness/lag_max": float(self.policy_version_before - self.sampled_version_min),
            "freshness/version_span": float(self.sampled_version_max - self.sampled_version_min),
        }
        if self.mismatch_kl is not None:
            # This measures numerical agreement between the sampler and the trainer.
            # It is logged and never gated. A threshold picked here would kill runs
            # that are merely unusual, and the right response is a human reading a curve.
            out["train/mismatch_kl"] = self.mismatch_kl
        if self.mismatch_kl_max_call is not None:
            out["train/mismatch_kl_max_call"] = self.mismatch_kl_max_call
        out.update({f"optim/{key}": value for key, value in self.optimizer_metrics.items()})
        out.update({f"loss/{key}": value for key, value in self.loss_metrics.items()})
        out.update({f"train/{key}": value for key, value in self.trainer_metrics.items()})
        out.update(self.mismatch)
        if self.aggregation is not None:
            out.update(self.aggregation.gauges())
        return out


class Trainer(LoopsTrainer):
    """One Loops run, which is a LoRA trainer and its paired sampler."""

    def __init__(self, config: Config, resume_from: str | None = None) -> None:
        """Set up the trainer without provisioning it.

        ``resume_from`` is the remote trainer state that a checkpoint manifest names. A
        fresh run starts from the weights ``[training] init_from`` names, if it names any.
        """
        super().__init__(
            model=config.model,
            compute=config.compute,
            output=config.output,
            resume_from=resume_from,
            init_from=config.training.init_from,
            init_from_optimizer=config.training.init_from_optimizer,
        )
        self.config = config
        self._custom_loss = load_custom_loss(config.training)
        self.sampler: SamplingClient | None = None
        self.sampler_version: int | None = None
        """The policy version ``sampler`` was published at."""
        self._version = 0
        self.resumed_step = 0
        # Once this trainer's accumulated gradients can't be stepped, this says why.
        self._invalid_gradients: str | None = None
        self._warned_unscored = False
        self._reset_window()

    async def start(self) -> SamplingClient:
        """Provision the trainer, restore any saved state, publish, and return the sampler."""
        if self.config.training.loss_fn != "custom":
            logger.info(
                "Built-in loss aggregation %s requires trainer normalization contract %s",
                self.config.training.loss_aggregation,
                BUILTIN_NORMALIZATION_CONTRACT,
            )
        client = await self._provision(
            with_sampler=True,
            sampler_replicas=self.config.compute.sampler_replicas,
            seed=self.config.training.seed,
        )
        self.resumed_step = await self._restore()
        # Publish before any rollout, so that a known version serves the first sample
        # instead of whatever the sampler happened to hold. Router replay names the
        # sampler's record of each generation, so a run that replays has every sample
        # recorded.
        self.sampler = await client.save_weights_and_get_sampling_client_async(
            name="default", record_stability_info=self.config.training.router_replay
        )
        self.sampler_version = self._version
        logger.info("Published initial adapter at policy version %d", self._version)
        # The SDK defers readiness to first use, and first use is a rollout inside
        # Harbor's agent timeout. A cold start belongs to startup, not to that budget.
        logger.info("Waiting for sampler to be ready...")
        started = monotonic()
        await self.sampler.ensure_ready_async()
        logger.info("Sampler is ready (after %.0fs)", monotonic() - started)
        return self.sampler

    async def _restore(self) -> int:
        resumed = await super()._restore()
        self._version = self._started_at
        return resumed

    @property
    def policy_version(self) -> int:
        return self._version

    def _check_lengths(self, datums: list[Datum]) -> None:
        """Refuse a sequence longer than the run's sequence budget.

        The check is per *datum*, not per batch, because ``max_seq_len`` bounds one
        sequence. Splitting a large batch is left to the SDK. It splits ``data`` when the
        request body would exceed its 24 MiB cap, and it dispatches one op per chunk.
        The trainer accumulates gradients across those calls until ``optim_step``, so
        split calls are mathematically equivalent to one.
        """
        budget = self.config.model.max_seq_len
        if budget is None:
            return
        for datum in datums:
            length = datum.model_input.length
            if length > budget:
                raise ValueError(
                    f"a single datum is {length} tokens, over the {budget}-token "
                    "max_seq_len. Shorten the agent's context or raise max_seq_len."
                )

    def _action_scores(
        self, rows: list[TrainingRow], outputs: list[dict[str, TensorData]]
    ) -> list[list[Sequence[float]]] | None:
        """Return the trainer's logprobs on each row's actions, for the mismatch gauges.

        Loops already holds this call's gradients, so an answer that doesn't fit the
        rows invalidates them before this raises. A backend that returns no logprobs
        costs only the gauges.
        """
        scored = sum("logprobs" in output for output in outputs)
        if not scored:
            if not self._warned_unscored:
                self._warned_unscored = True
                logger.warning("Loops returned no logprobs, so mismatch/* is not reported")
            return None
        try:
            if scored < len(outputs):
                raise UnfitScores(f"Loops returned logprobs for {scored} of {len(outputs)} datums")
            return action_scores(rows, [output["logprobs"].data for output in outputs])
        except UnfitScores:
            self._invalid_gradients = "A trainer answer that did not fit its rows"
            raise

    async def accumulate(self, rows: list[TrainingRow]) -> None:
        """Accumulate a complete batch without stepping the optimizer.

        The caller normalizes replay eligibility with ``normalize_batch_provenance``
        before it records or submits the batch, so what it records is what trains. A
        batch that is not normalized is refused. Several calls may share an optimizer
        step only when they use one consistent replay mode. Loops accumulates their
        gradients until that step, reduced as ``loss_aggregation`` asks.
        """
        if self.client is None:
            raise RuntimeError("Trainer.start() has not been awaited")
        if not rows:
            return
        if self._invalid_gradients:
            raise RuntimeError(f"{self._invalid_gradients} invalidated this trainer's gradients")
        datums = [row.datum for row in rows]
        self._check_lengths(datums)
        routed = all(row.routed for row in rows)
        if any(row.routed for row in rows) and not (routed and self.config.training.router_replay):
            raise ValueError(
                "the batch's provenance is not normalized. Pass it through "
                "normalize_batch_provenance(enabled=[training] router_replay) first."
            )
        if self._window_routed is not None and routed != self._window_routed:
            raise ValueError(
                "normalize routes across the entire optimizer batch before accumulation"
            )
        self._window_routed = routed
        started = monotonic()
        training = self.config.training
        if training.loss_fn == "custom":
            # It was loaded from these settings, which require it with a custom loss_fn.
            assert self._custom_loss is not None
            loss, loss_metrics, scores = await forward_backward_custom(
                self.client, rows, self._custom_loss
            )
            mismatch, server_metrics = None, {}
        else:
            averaged = trainer_averages(training)
            future = await self.client.forward_backward_async(
                datums,
                training.loss_fn,
                {**training.loss_fn_config, **token_normalization(averaged)},
            )
            result = await future.result_async()
            check_token_normalization(result.metrics, averaged)
            loss, loss_metrics = result.loss, {}
            scores = self._action_scores(rows, result.loss_fn_outputs)
            # The reduction's echo was checked above, and it doesn't measure the step.
            server_metrics = {
                key: float(value)
                for key, value in (result.metrics or {}).items()
                if key != "skip_token_normalization"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            }
            mismatch = server_metrics.get("mismatch_kl")
        self._timings["forward_backward"] = (
            self._timings.get("forward_backward", 0.0) + monotonic() - started
        )
        if not math.isfinite(loss):
            self._invalid_gradients = "Nonfinite loss"
            raise RuntimeError(
                "Nonfinite trainer loss. The optimizer step and publication are refused."
            )
        self._pending_datums += len(datums)
        self._pending_tokens += sum(d.model_input.length for d in datums)
        trained = sum(row.trained_positions for row in rows)
        self._pending_trained += trained
        # Match the SDK's response aggregation, not the server's global token mean.
        # The SDK's chunk merging doesn't keep the server's token denominators.
        self._loss_numerator += loss * len(datums)
        for key, value in loss_metrics.items():
            total = self._loss_metric_numerators.get(key, 0.0)
            self._loss_metric_numerators[key] = total + value * len(datums)
        # Weight these by datum, like mismatch_kl, which keeps its own fields. The
        # trainer's per-call means arrive already weighted by token.
        for key, value in server_metrics.items():
            if key != "mismatch_kl":
                total = self._trainer_metric_numerators.get(key, 0.0)
                self._trainer_metric_numerators[key] = total + value * len(datums)
        self._forward_backward_reports.append(
            ForwardBackwardReport(
                len(datums),
                trained,
                bool(routed),
                loss,
                mismatch,
                server_metrics,
            )
        )
        if mismatch is not None:
            self._mismatch_numerator += mismatch * len(datums)
            self._mismatch_datums += len(datums)
            self._mismatch_max_call = (
                mismatch
                if self._mismatch_max_call is None
                else max(self._mismatch_max_call, mismatch)
            )
        self._calls += 1
        if scores is not None:
            # This runs last, on the loop in slices. Everything above already accounts
            # for the gradients Loops holds, so a cancellation here costs only these
            # gauges.
            gaps = MismatchTotals()
            await in_slices(gaps.add_in_slices(rows, scores, self.policy_version))
            self._gaps.merge(gaps)

    async def commit(
        self,
        step: int,
        *,
        groups: int,
        rollouts: int,
        rewards: list[float],
        advantages_: list[float],
        sampled_versions: list[int],
    ) -> StepReport:
        """Step the optimizer over everything accumulated, then publish.

        Once this returns, the update is applied and the sampler serves it.
        ``checkpoint()`` keeps it, in a separate step that can fail without undoing it.
        """
        if self.client is None:
            raise RuntimeError("Trainer.start() has not been awaited")
        if self._invalid_gradients:
            raise RuntimeError(f"{self._invalid_gradients} invalidated this trainer's gradients")
        if not self._pending_datums:
            raise ValueError(f"step {step}: commit with nothing accumulated")

        before = self._version
        started = monotonic()
        self._applying()
        optim = await self.client.optim_step_async(
            self.config.training.adam_params(self.config.training.learning_rate)
        )
        optim_result = await optim.result_async()
        self._timings["optimizer"] = monotonic() - started
        optimizer_metrics = {}
        for key, value in (optim_result.metrics or {}).items():
            if key not in {
                "grad_norm",
                "learning_rate",
                "num_tokens",
                "num_loss_tokens",
                "step_seconds",
                "tps",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            }:
                continue
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                optimizer_metrics[key] = float(value)
            else:
                logger.warning(
                    "Omitting nonfinite or nonnumeric optimizer metric %s=%r", key, value
                )

        # The trainer's version is its optimizer step count, so it advances here,
        # not at publication. The paired sampler can serve the new weights before
        # the publication call returns, so record the version before publishing.
        after = await updated_version(self.client, optim_result)
        self._applied(step, after)
        self._version = after

        # The per-step weight sync goes to "default". The paired sampler hot-swaps to
        # it, and it is overwritten every step by design. A serverless sampling client
        # is pinned to this publication instead, because the SDK gives each serverless
        # save a name of its own. So its samples take ``sampler_version``, and leases
        # read the new handle at their next generation.
        started = monotonic()
        self.sampler = await self.client.save_weights_and_get_sampling_client_async(
            record_stability_info=self.config.training.router_replay
        )
        self.sampler_version = after
        self._timings["publication"] = monotonic() - started

        mean = sum(rewards) / len(rewards)
        var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        report = StepReport(
            step=step,
            policy_version_before=before,
            policy_version_after=after,
            groups=groups,
            rollouts=rollouts,
            datums=self._pending_datums,
            tokens=self._pending_tokens,
            trained_positions=self._pending_trained,
            loss=self._loss_numerator / self._pending_datums,
            reward_mean=mean,
            reward_std=var**0.5,
            advantage_abs_mean=(
                sum(abs(a) for a in advantages_) / len(advantages_) if advantages_ else 0.0
            ),
            sampled_version_min=min(sampled_versions) if sampled_versions else before,
            sampled_version_max=max(sampled_versions) if sampled_versions else before,
            forward_backward_calls=self._calls,
            mismatch_kl=(
                self._mismatch_numerator / self._mismatch_datums if self._mismatch_datums else None
            ),
            mismatch_kl_max_call=self._mismatch_max_call,
            forward_backward_reports=tuple(self._forward_backward_reports),
            optimizer_metrics=optimizer_metrics,
            loss_metrics={
                key: total / self._pending_datums
                for key, total in self._loss_metric_numerators.items()
            },
            trainer_metrics={
                key: total / self._pending_datums
                for key, total in self._trainer_metric_numerators.items()
            },
            timings=dict(self._timings),
            mismatch=self._gaps.gauges(),
        )
        self._reset_window()
        return report

    async def checkpoint(
        self, report: StepReport, *, final: bool = False, keep_sampler_weights: bool = False
    ) -> StepReport:
        """Return the committed step's report, with what was kept of the step.

        ``final`` marks the last update the runner intends to make. That update is
        checkpointed whatever the cadences say, because a run that ends on a step they
        skip would otherwise leave nothing to resume or serve.
        ``keep_sampler_weights`` keeps this step's sampler weights whatever their
        cadence says, for an eval to serve.
        """
        started = monotonic()
        checkpoint, state_path, weights = await self._checkpoint(
            report.step,
            final=final,
            version=report.policy_version_after,
            keep_sampler_weights=keep_sampler_weights,
        )
        return replace(
            report,
            checkpoint=checkpoint,
            checkpoint_state_path=state_path,
            sampler_weights=weights,
            timings={**report.timings, "checkpoint": monotonic() - started},
        )

    def _reset_window(self) -> None:
        self._timings: dict[str, float] = {}
        self._window_routed: bool | None = None
        self._pending_datums = 0
        self._pending_tokens = 0
        self._pending_trained = 0
        self._loss_numerator = 0.0
        self._calls = 0
        self._mismatch_numerator = 0.0
        self._mismatch_datums = 0
        self._mismatch_max_call = None
        self._forward_backward_reports = []
        self._gaps = MismatchTotals()
        self._loss_metric_numerators: dict[str, float] = {}
        self._trainer_metric_numerators: dict[str, float] = {}
