"""A custom loss: made by its factory at load, and given every position Loops scores."""

import pytest
from baseten.loops.models import EncodedTextChunk, ImageChunk, ModelInput
from support import losses
from support.builders import make_config, make_rollout, make_sample

from loops_cookbook.rl.grpo import build_rows
from loops_cookbook.rl.loss import load_custom_loss, scored_datum
from loops_cookbook.runtime.context import RuntimeConfigurationError


def custom(factory, config=None):
    return make_config(
        loss_fn="custom", custom_loss={"factory": factory, "config": config or {}}
    ).training


def test_a_built_in_loss_loads_no_function():
    assert load_custom_loss(make_config(loss_fn="cispo").training) is None


def test_the_factory_makes_the_loss_from_a_private_copy_of_its_config():
    pytest.importorskip("torch")
    config = {"observation_weight": 0.05, "nested": {"a": [1, 2]}}
    fn = load_custom_loss(custom("support.losses:make_policy_gradient", config))
    assert fn is losses.policy_gradient
    assert losses.make_policy_gradient.received == config
    assert losses.make_policy_gradient.received is not config


@pytest.mark.parametrize(
    ("factory", "error"),
    [
        ("no_such_module:create", "could not import"),
        ("support.losses:missing", "has no custom loss factory"),
        ("support.losses:NOT_CALLABLE", "not a callable"),
        ("support.losses:not_a_function", "returned int, not a function"),
    ],
)
def test_an_unusable_factory_is_refused_before_anything_runs(factory, error):
    pytest.importorskip("torch")
    with pytest.raises(RuntimeConfigurationError, match=error):
        load_custom_loss(custom(factory))


def test_every_text_position_is_scored_and_the_actions_are_marked():
    # Prompt [1, 2]; the agent writes [3, 4]; the environment adds [5, 6]; then [7].
    rollout = make_rollout(
        "task",
        "group",
        "rollout",
        (
            make_sample("a", (1, 2), (3, 4), (-0.1, -0.2), 0),
            make_sample("b", (1, 2, 3, 4, 5, 6), (7,), (-0.3,), 0),
        ),
    )
    (row,) = build_rows(rollout, 0.5)
    inputs = scored_datum(row).loss_fn_inputs
    assert inputs["target_tokens"].data == [2, 3, 4, 5, 6, 7]
    assert inputs["actions"].data == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    for name in ("advantages", "logprobs"):
        assert inputs[name] == row.datum.loss_fn_inputs[name]


def test_an_image_position_is_never_a_target():
    # [1, 2] prompt, an image of two placeholders, 8, then the action 9.
    prompt_input = ModelInput(
        chunks=[
            EncodedTextChunk(tokens=[1, 2]),
            ImageChunk(data=b"png", format="png", expected_tokens=2),
            EncodedTextChunk(tokens=[8]),
        ]
    )
    first = make_sample("a", (1, 2, 5, 5, 8), (9,), (-0.1,), 0, prompt_input=prompt_input)
    (row,) = build_rows(make_rollout("t", "g", "r", (first,)), 0.5)
    inputs = scored_datum(row).loss_fn_inputs
    assert inputs["target_tokens"].data == [2, -100, -100, 8, 9]
    assert inputs["actions"].data == [0.0, 0.0, 0.0, 0.0, 1.0]
