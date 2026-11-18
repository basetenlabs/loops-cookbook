"""DeepSeek-V4.1 renderer: image token counts and options, without a tokenizer."""

import pytest
from pydantic import ValidationError

from loops_cookbook.llm.renderer_extensions.deepseek_v41 import (
    DeepSeekV41RendererConfig,
    image_tokens,
)


@pytest.mark.parametrize(
    ("width", "height", "tokens"),
    [
        pytest.param(640, 480, 12 * 17 + 2, id="pads"),
        pytest.param(100, 100, 13 * 14 + 2, id="upscales"),
        pytest.param(4000, 3000, 27 * 37 + 2, id="shrinks"),
    ],
)
def test_image_tokens_match_the_trainer(width, height, tokens):
    # The cases of the trainer's test_deepseek_v41_sizes_an_image_like_the_reference.
    assert image_tokens(width, height) == tokens


@pytest.mark.parametrize(("width", "height"), [(1, 1), (30000, 20), (20, 30000)])
def test_image_tokens_stay_within_the_budget(width, height):
    assert 0 < image_tokens(width, height) <= 1024


@pytest.mark.parametrize(
    ("kwargs", "budget"),
    [
        ({}, 75),
        ({"reasoning_effort": "low"}, 50),
        ({"reasoning_effort": "max"}, 100),
        ({"reasoning_effort": 30}, 30),
    ],
)
def test_the_effort_budget_is_the_templates(kwargs, budget):
    assert DeepSeekV41RendererConfig(**kwargs).budget == budget


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reasoning_effort": "medium"},
        {"reasoning_effort": 0},
        {"reasoning_effort": True},
        {"thinking_mode": "chat"},
        {"drop_thinking": "yes"},
    ],
)
def test_options_outside_the_config_are_refused(kwargs):
    with pytest.raises(ValidationError):
        DeepSeekV41RendererConfig(**kwargs)
