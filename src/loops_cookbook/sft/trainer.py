"""The Loops transport for SFT, which makes one update per batch and scores validation without gradients.

This module computes negative log-likelihood (NLL) from the per-token logprobs that
Loops returns, for training and validation alike. The NLL is the sum of ``-logprob``
over the trained positions, divided by their count. It is computed the same way
whichever ``loss_aggregation`` the optimizer step uses, so the curve stays comparable
across runs and batch sizes. The SDK's own ``loss`` field is not used. When the SDK
splits a large request, it averages the per-chunk means by datum count, which gives a
different number.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from time import monotonic
from typing import Protocol

from baseten.loops.models import Datum, ForwardBackwardOutput

from loops_cookbook.sft.config import Config
from loops_cookbook.trainer import (
    LoopsTrainer,
    check_token_normalization,
    token_normalization,
    updated_version,
)

logger = logging.getLogger(__name__)

OPTIMIZER_METRICS = ("grad_norm", "num_tokens", "num_loss_tokens", "tps", "step_seconds")
"""The metrics that Loops reports about an optimizer step and that are worth plotting."""


@dataclass(frozen=True)
class Score:
    """Mean negative log-likelihood over the trained positions of some datums."""

    nll: float
    trained_tokens: int


@dataclass(frozen=True)
class Update:
    """One optimizer step, as a flat record."""

    step: int
    sequences: int
    tokens: int
    score: Score
    learning_rate: float
    seconds: float
    optimizer: dict[str, float] = field(default_factory=dict[str, float])
    """``OPTIMIZER_METRICS`` Loops returned for the step, verbatim."""


def score(datums: Sequence[Datum], output: ForwardBackwardOutput) -> Score:
    """Compute the token-mean NLL of ``datums`` from Loops' per-datum logprobs.

    Positions with weight 0 come back as null, because no logprob was computed for
    them. ``null * 0`` is not zero in any arithmetic that turns null into NaN, so these
    positions are skipped rather than multiplied away.
    """
    if len(output.loss_fn_outputs) != len(datums):
        raise RuntimeError(
            f"Loops returned {len(output.loss_fn_outputs)} outputs for {len(datums)} datums"
        )
    total, count = 0.0, 0
    for datum, outputs in zip(datums, output.loss_fn_outputs, strict=True):
        weights = datum.loss_fn_inputs["weights"].data
        logprobs = outputs["logprobs"].data
        for weight, logprob in zip(weights, logprobs, strict=True):
            if not weight:
                continue
            if logprob is None or not math.isfinite(logprob):
                raise RuntimeError("Loops returned no finite logprob for a trained position")
            total -= logprob
            count += 1
    if not count:
        raise ValueError("no trained positions to score")
    return Score(total / count, count)


class TrainerPort(Protocol):
    """What a run asks of its trainer. ``Trainer`` below is the Loops implementation."""

    @property
    def run_id(self) -> str | None: ...

    async def start(self) -> None: ...

    async def update(self, step: int, datums: Sequence[Datum], learning_rate: float) -> Update: ...

    async def evaluate(self, datums: Sequence[Datum]) -> Score: ...

    async def checkpoint(
        self, step: int, *, final: bool = False
    ) -> tuple[str | None, str | None]: ...

    async def checkpoint_state_only(self, step: int) -> str | None: ...

    async def run_ended(self) -> str | None: ...

    async def close(self) -> None: ...


class Trainer(LoopsTrainer):
    """A LoRA trainer with no sampler, because SFT trains on fixed data and serves nothing."""

    def __init__(self, config: Config, resume_from: str | None = None) -> None:
        """Configure the trainer from ``config``.

        ``resume_from`` is the remote trainer state that a checkpoint manifest names. A
        fresh run starts from the weights that ``[training] init_from`` names, if any.
        """
        super().__init__(
            model=config.model,
            compute=config.compute,
            output=config.output,
            resume_from=resume_from,
            init_from=config.training.init_from,
            init_from_optimizer=config.training.init_from_optimizer,
        )
        self._training = config.training

    async def start(self) -> None:
        """Provision the trainer, then restore a resume's state or load a fresh run's starting weights."""
        await self._provision(with_sampler=False, seed=self._training.seed)
        await self._restore()

    async def update(self, step: int, datums: Sequence[Datum], learning_rate: float) -> Update:
        """Accumulate the whole batch, then step the optimizer once over it."""
        client = self._require_client()
        started = monotonic()
        averaged = self._training.loss_aggregation == "token_mean"
        result = await (
            await client.forward_backward_async(
                list(datums), "cross_entropy", token_normalization(averaged)
            )
        ).result_async()
        check_token_normalization(result.metrics, averaged)
        measured = score(datums, result)
        self._applying()
        optimizer = await client.optim_step_async(self._training.adam_params(learning_rate))
        result = await optimizer.result_async()
        reported = result.metrics or {}
        # The update is recorded so that a failing run, on its way out, can tell whether
        # the trainer still holds it.
        self._applied(step, await updated_version(client, result))
        return Update(
            step=step,
            sequences=len(datums),
            tokens=sum(datum.model_input.length for datum in datums),
            score=measured,
            learning_rate=learning_rate,
            seconds=monotonic() - started,
            optimizer={
                name: float(reported[name])
                for name in OPTIMIZER_METRICS
                if isinstance(reported.get(name), (int, float)) and math.isfinite(reported[name])
            },
        )

    async def evaluate(self, datums: Sequence[Datum]) -> Score:
        """Score ``datums`` with the current weights, leaving no gradient behind."""
        client = self._require_client()
        result = await (await client.forward_async(list(datums), "cross_entropy")).result_async()
        return score(datums, result)

    async def checkpoint(self, step: int, *, final: bool = False) -> tuple[str | None, str | None]:
        """Save step ``step`` when the checkpoint cadences or ``final`` call for it.

        Return ``(sampler weights name, trainer state path)``.
        """
        name, state_path, _ = await self._checkpoint(step, final=final)
        return name, state_path
