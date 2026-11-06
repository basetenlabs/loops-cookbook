"""``[training.custom_loss]`` factories for the custom loss tests."""

NOT_CALLABLE = 3


def policy_gradient(datums, logprobs):
    """``-Σ advantage · logprob`` over the actions, so ``-dL/dlogprob`` is the advantage."""
    loss = sum(
        -(
            lp * d.loss_fn_inputs["actions"].to_torch() * d.loss_fn_inputs["advantages"].to_torch()
        ).sum()
        for d, lp in zip(datums, logprobs, strict=True)
    )
    return loss, {"datums": float(len(datums))}


def make_policy_gradient(config):
    """Keeps the config it was given, to show it arrived."""
    make_policy_gradient.received = config
    return policy_gradient


def not_a_function(config):
    return 3
