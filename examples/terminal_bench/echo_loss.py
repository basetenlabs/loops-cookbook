"""The ECHO loss, which also teaches the agent to predict what its commands print.

The loss comes from arXiv:2605.24517.

    L = L_IS(actions) + observation_weight * L_CE(observations)

``L_IS`` is the recipe's importance-sampling loss on the agent's own tokens,
``-ratio * advantage`` averaged over every action token in the call. ``L_CE`` is the
next-token loss on the tokens the environment sent back, every scored position after
the agent's first action that is not an action, averaged within each datum, then over
datums. The task's prompt, before the first action, is not an observation. It takes no
advantage, because a failed rollout's terminal output is as true as a successful one's.

This loss differs from the paper in three ways. The paper's policy term clips the ratio,
as proximal policy optimization (PPO) does. The paper's advantage divides by the group's
spread, where this one uses the recipe's advantage. And the paper's observations leave
out the chat template around each tool result and a harness warning prefix, which here
are observations like any other token.
"""

from collections.abc import Mapping

import torch
from pydantic import BaseModel, ConfigDict, Field


class EchoOptions(BaseModel):
    """The ``[training.custom_loss.config]`` table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_weight: float = Field(default=0.05, ge=0)
    """λ, the paper's 0.05. It finds 0.01-0.05 useful; above that it competes with
    the policy."""


def create_loss(config: Mapping[str, object]):
    options = EchoOptions.model_validate(config)

    def echo(datums, logprobs):
        policy_sum, action_tokens, observation_losses = torch.zeros(()), 0, []
        for datum, current in zip(datums, logprobs, strict=True):
            inputs = datum.loss_fn_inputs
            actions = inputs["actions"].to_torch().bool()
            after_first = actions.cumsum(0) > 0
            observations = (inputs["target_tokens"].to_torch() >= 0) & ~actions & after_first
            ratio = torch.exp(current - inputs["logprobs"].to_torch())
            policy_sum = policy_sum + (ratio * inputs["advantages"].to_torch())[actions].sum()
            action_tokens += int(actions.sum())
            if observations.any():
                observation_losses.append(-current[observations].mean())
        policy = -policy_sum / max(action_tokens, 1)
        observation = (
            torch.stack(observation_losses).mean() if observation_losses else torch.zeros(())
        )
        loss = policy + options.observation_weight * observation
        return loss, {"policy": policy.item(), "observation_nll": observation.item()}

    return echo
