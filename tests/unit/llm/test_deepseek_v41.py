"""DeepSeek-V4.1 renderer: image token counts and template options, without a tokenizer."""

import pytest

from loops_cookbook.llm.renderer_extensions.deepseek_v41 import Options, image_tokens, options


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
    ("variant", "kwargs", "expected"),
    [
        ("v4.1", {}, Options(thinking=True, drop_thinking=True, budget=75)),
        ("v4.1", {"enable_thinking": False}, Options(False, True, 75)),
        ("v4.1", {"reasoning_effort": "medium"}, Options(True, True, 62)),
        ("1017", {"reasoning_effort": "medium"}, Options(True, True, 75)),
        ("1017", {"reasoning_effort": "none"}, Options(False, True, 75)),
        ("1017", {"reasoning_effort": "none", "thinking": True}, Options(True, True, 75)),
        ("1017", {"thinking_mode": "chat", "drop_thinking": False}, Options(False, False, 75)),
        ("1017", {"reasoning_effort": 30}, Options(True, True, 30)),
    ],
)
def test_options_resolve_as_the_template_does(variant, kwargs, expected):
    assert options(variant, kwargs) == expected


@pytest.mark.parametrize(
    ("variant", "kwargs"),
    [
        ("v4.1", {"thinking": True}),
        ("v4.1", {"reasoning_effort": "none"}),
        ("1017", {"reasoning_effort": 0}),
        ("1017", {"reasoning_effort": True}),
        ("1017", {"thinking_mode": "fast"}),
        ("1017", {"drop_thinking": "yes"}),
    ],
)
def test_options_the_template_refuses_are_refused(variant, kwargs):
    with pytest.raises((ValueError, TypeError)):
        options(variant, kwargs)
