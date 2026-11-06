"""ECHO: importance sampling on the agent's tokens plus λ times the next-token loss on
what the environment printed, over the datums a custom loss receives."""

import pytest

torch = pytest.importorskip("torch")

import echo_loss
from support.builders import make_rollout, make_sample

from loops_cookbook.rl.grpo import build_rows
from loops_cookbook.rl.loss import scored_datum

LAMBDA = 0.05


def datum():
    # Prompt [1, 2]; the agent writes [3, 4]; the terminal prints [5, 6]; then [7].
    rollout = make_rollout(
        "task",
        "group",
        "rollout",
        (
            make_sample("a", (1, 2), (3, 4), (-0.5, -0.5), 0),
            make_sample("b", (1, 2, 3, 4, 5, 6), (7,), (-0.5,), 0),
        ),
    )
    (built,) = build_rows(rollout, 0.5)
    return scored_datum(built)


def test_echo_validates_its_options():
    with pytest.raises(ValueError, match="observation_wieght"):
        echo_loss.create_loss({"observation_wieght": 0.05})


def test_the_gradient_is_the_advantage_on_actions_and_lambda_on_observations():
    echo = echo_loss.create_loss({"observation_weight": LAMBDA})
    # Sampled and current logprobs agree on the actions, so every ratio is 1.
    logprobs = torch.tensor([0.0, -0.5, -0.5, -2.0, -3.0, -0.5], requires_grad=True)
    loss, metrics = echo([datum()], [logprobs])
    loss.backward()
    # Positions: prompt, 3 and 4 (actions), 5 and 6 (observations), 7 (action).
    action, observation = -0.5 / 3, -LAMBDA / 2
    assert logprobs.grad.tolist() == pytest.approx(
        [0.0, action, action, observation, observation, action]
    )
    assert metrics == pytest.approx({"policy": -0.5, "observation_nll": 2.5})
