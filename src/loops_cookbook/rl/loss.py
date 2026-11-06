"""A custom loss, run through Loops' ``forward_backward_custom``.

``[training] loss_fn = "custom"`` names a factory in ``[training.custom_loss]``: it
takes that table's ``config`` and returns the loss function. ``fn(datums, logprobs)``
receives the batch's datums and the trainer's logprobs, one tensor per datum, and
returns a scalar loss tensor and float metrics.

Loops scores every text position of a datum, the prompt and the environment's tokens
between turns included, so a datum's ``loss_fn_inputs`` hold:

- ``target_tokens``: the next token at every position, -100 at image positions, where
  there is none; the logprobs are zero where it is -100.
- ``actions``: 1.0 at the agent's own tokens, the ones the recipe trains.
- ``advantages`` and ``logprobs``: the rollout's advantage and the sampler's logprobs
  at the actions, as ``rl/grpo.py`` builds them.

Loops sums ``fn``'s gradients without token normalization, so ``fn`` normalizes as it
means to.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from baseten.loops.models import Datum, TensorData
from baseten.loops.training_client import TrainingClient

from loops_cookbook.rl.config import TrainingSettings
from loops_cookbook.rl.grpo import TrainingRow
from loops_cookbook.rl.mismatch import action_scores
from loops_cookbook.runtime.context import RuntimeConfigurationError, import_attribute

LossFn = Callable[[list[Datum], list[Any]], tuple[Any, dict[str, float]]]
"""``(datums, logprobs) -> (loss, metrics)``; the tensors are ``torch.Tensor``."""


def load_custom_loss(training: TrainingSettings) -> LossFn | None:
    """The loss function ``[training.custom_loss]`` makes, or None for a built-in loss."""
    if training.custom_loss is None:
        return None
    try:
        import torch  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except ImportError as error:
        raise RuntimeConfigurationError(
            'loss_fn = "custom" runs on PyTorch: install the custom-loss extra'
        ) from error
    path = training.custom_loss.factory
    factory = import_attribute(path, "custom loss factory")
    if not callable(factory):
        raise RuntimeConfigurationError(f"{path!r} is not a callable custom loss factory")
    fn = factory(deepcopy(training.custom_loss.config))
    if not callable(fn):
        raise RuntimeConfigurationError(f"{path!r} returned {type(fn).__name__}, not a function")
    return fn


async def forward_backward_custom(
    client: TrainingClient, rows: list[TrainingRow], fn: LossFn
) -> tuple[float, dict[str, float], list[list[list[float]]]]:
    """Accumulate ``fn``'s gradients; its own loss value and metrics, and the
    trainer's logprobs on each row's actions (``action_scores``)."""
    seen: list[tuple[float, dict[str, float], list[list[list[float]]]]] = []
    scored = [scored_datum(row) for row in rows]
    # The SDK's custom path takes only target_tokens in loss_fn_inputs; fn sees them all.
    bare = [
        datum.model_copy(
            update={"loss_fn_inputs": {"target_tokens": datum.loss_fn_inputs["target_tokens"]}}
        )
        for datum in scored
    ]

    def call(_: list[Datum], logprobs: list[Any]) -> tuple[Any, dict[str, float]]:
        # Only the action positions, for the mismatch gauges, and before fn's gradient
        # is applied, so an answer that does not fit the rows changes nothing.
        scores = [
            [action.detach().tolist() for action in actions]
            for actions in action_scores(rows, logprobs)
        ]
        value, metrics = fn(scored, logprobs)
        seen.append((value.item(), dict(metrics), scores))
        return value, metrics

    future = await client.forward_backward_custom_async(bare, call)
    # Its result is the surrogate pass that applies fn's gradient, not fn's loss.
    await future.result_async()
    return seen[0]


def scored_datum(row: TrainingRow) -> Datum:
    """The row's datum with its ``actions`` mask among the loss inputs."""
    inputs = row.datum.loss_fn_inputs
    length = len(inputs["target_tokens"].data)
    actions = [0.0] * length
    for start, stop in row.actions:
        actions[start:stop] = [1.0] * (stop - start)
    return row.datum.model_copy(
        update={
            "loss_fn_inputs": {
                **inputs,
                "actions": TensorData(data=actions, dtype="float32", shape=[length]),
            }
        }
    )
