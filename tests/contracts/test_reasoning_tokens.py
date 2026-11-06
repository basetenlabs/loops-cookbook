"""A reply's reasoning tokens, counted on its sampled tokens by the real renderers."""

import pytest

from loops_cookbook.llm.factory import create_renderer

HISTORY = [{"role": "user", "content": "List the files."}]
REASONING = "The user wants the files listed, so I will run ls in the working directory."
GLM53 = "loops_cookbook.llm.renderer_extensions.glm53:renderer"


def _sampled(tool_call_model, text):
    tokenizer, renderer, _ = tool_call_model
    prompt = renderer.render(HISTORY)
    completion = tokenizer.encode(text, add_special_tokens=False)
    return renderer, completion, renderer.parse(prompt.ids, completion, [])


def test_reasoning_is_the_tokens_before_it_closed(tool_call_model):
    tokenizer, _, close = tool_call_model
    if not close:
        pytest.skip("this model's options turn reasoning off")
    renderer, completion, parsed = _sampled(tool_call_model, REASONING + close + "Done.<|im_end|>")
    expected = len(tokenizer.encode(REASONING, add_special_tokens=False))
    assert renderer.reasoning_tokens(completion, parsed) == expected < len(completion)


@pytest.mark.parametrize("closed", [True, False], ids=["closed", "cut"])
def test_glm_reasoning_is_the_tokens_before_it_closed(glm_tokenizer, closed):
    renderer = create_renderer("zai-org/GLM-5.3", renderer=GLM53, tokenizer=glm_tokenizer)
    prompt = renderer.render(HISTORY)
    rest = "</think>Done." if closed else ""
    completion = glm_tokenizer.encode(REASONING + rest, add_special_tokens=False)
    parsed = renderer.parse(prompt.ids, completion, [])
    expected = len(glm_tokenizer.encode(REASONING, add_special_tokens=False))
    assert renderer.reasoning_tokens(completion, parsed) == expected


def test_a_reply_cut_while_reasoning_is_all_reasoning(tool_call_model):
    if not tool_call_model[2]:
        pytest.skip("this model's options turn reasoning off")
    renderer, completion, parsed = _sampled(tool_call_model, REASONING)
    assert renderer.reasoning_tokens(completion, parsed) == len(completion)


@pytest.mark.parametrize("answer", ["Done.", "The delimiter is </think>."])
def test_a_reply_without_reasoning_has_none(tool_call_model, answer):
    """With reasoning off, a ``</think>`` the model writes is part of its answer."""
    renderer, completion, parsed = _sampled(
        tool_call_model, tool_call_model[2] + answer + "<|im_end|>"
    )
    assert renderer.reasoning_tokens(completion, parsed) == 0
