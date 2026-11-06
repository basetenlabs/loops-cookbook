"""PrimeRenderer adapter contract with fake renderers (no model files)."""

from types import SimpleNamespace

import pytest

from loops_cookbook.evidence import TokenSpan
from loops_cookbook.llm.prime import PrimeRenderer
from loops_cookbook.llm.rendering import ParsedReply, RenderingError, UnsupportedContextError


def test_full_messages_and_tools_are_delegated_without_mutating_history():
    seen = []

    class Native:
        def render(self, messages, **kwargs):
            seen.append((messages, kwargs))
            return SimpleNamespace(token_ids=[1, 2])

        def get_stop_token_ids(self):
            return [99]

    renderer = PrimeRenderer(Native())
    messages = [{"role": "assistant", "content": "authored replacement"}]
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    assert renderer.render(messages, tools).ids == (1, 2)
    assert renderer.stop_sequences == (99,)
    assert seen[0] == (messages, {"tools": tools, "add_generation_prompt": True})
    assert seen[0][0] is not messages


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16  # header only; passed through, never decoded


def test_text_only_renderer_rejects_images_instead_of_dropping_them():
    class TextOnly:
        def render(self, messages, **kwargs):
            return SimpleNamespace(token_ids=[1, 2], multi_modal_data=None)

    renderer = PrimeRenderer(TextOnly())
    assert not renderer.supports_images
    with pytest.raises(UnsupportedContextError, match="does not support image"):
        renderer.render([{"role": "user", "content": [{"type": "image", "image": PNG}]}])


@pytest.mark.parametrize(
    "part",
    [
        {"type": "video_url", "video_url": {"url": "x"}},
        {"type": "input_audio", "input_audio": {"data": "x"}},
        {"type": "mystery"},
    ],
)
def test_non_image_media_rejected_before_renderer_can_drop_it(part):
    with pytest.raises(UnsupportedContextError, match="text and image"):
        PrimeRenderer(SimpleNamespace(render=lambda *args, **kwargs: None)).render(
            [{"role": "user", "content": [part]}]
        )


def test_thinking_parts_are_rejected_instead_of_silently_dropped():
    with pytest.raises(UnsupportedContextError, match="reasoning_content"):
        PrimeRenderer(
            SimpleNamespace(render=lambda *args, **kwargs: pytest.fail("unexpected render"))
        ).render([{"role": "assistant", "content": [{"type": "thinking", "thinking": "keep me"}]}])


def test_none_content_and_wrapped_tools_are_not_normalized():
    seen = []

    class Native:
        def render(self, messages, **kwargs):
            seen.append((messages, kwargs["tools"]))
            return SimpleNamespace(token_ids=[1])

    messages = [{"role": "assistant", "content": None}]
    tools = [{"type": "function", "function": {"name": "f"}}]
    PrimeRenderer(Native()).render(messages, tools)
    assert seen == [(messages, tools)]


@pytest.mark.parametrize("bridged", [[1, 9, 3], [9, 2, 3], [1], [2, 1, 3], []])
def test_bridge_changes_to_prior_prefix_fall_back_to_full_render(bridged):
    native = SimpleNamespace(
        bridge_to_next_turn=lambda *args, **kwargs: SimpleNamespace(token_ids=bridged)
    )
    tool = [{"role": "tool", "content": "x"}]
    if not bridged:  # no tokens at all is the renderer's error, as before
        with pytest.raises(RenderingError, match="nonempty"):
            PrimeRenderer(native).bridge((1,), (2,), tool)
        return
    assert PrimeRenderer(native).bridge((1,), (2,), tool) is None
    assert PrimeRenderer(native).bridge(TokenSpan((1,)), (2,), tool) is None


def test_bridge_extends_a_span_prefix_without_copying_it_to_a_tuple():
    native = SimpleNamespace(
        bridge_to_next_turn=lambda prompt, completion, *a, **k: SimpleNamespace(
            token_ids=[*prompt, *completion, 3, 4]
        )
    )
    bridged = PrimeRenderer(native).bridge(TokenSpan((1, 2)), (5,), [{"role": "tool"}])
    assert isinstance(bridged.ids, TokenSpan) and bridged.ids == (1, 2, 5, 3, 4)
    assert bridged.prompt_input.to_ints() == [1, 2, 5, 3, 4]


def _reasoned(reasoning, complete=True):
    return ParsedReply("answer", reasoning, (), complete)


def test_reasoning_tokens_are_those_before_the_close_marker():
    renderer = PrimeRenderer(SimpleNamespace(_think_end=9))
    assert renderer.reasoning_tokens(TokenSpan([5, 6, 7, 9, 8]), _reasoned("abc")) == 3
    assert renderer.reasoning_tokens([5, 6], _reasoned("ab", complete=False)) == 2
    assert renderer.reasoning_tokens([8], _reasoned(None)) == 0
    # A ``</think>`` the model writes with no reasoning is part of its answer.
    assert renderer.reasoning_tokens([5, 9, 8], _reasoned(None)) == 0
    assert renderer.reasoning_tokens([5, 6], _reasoned(None, complete=False)) == 2


def test_reasoning_tokens_are_unknown_without_a_single_token_close_marker():
    """A renderer with no ``_think_end``, or a reasoning closed some other way (by a tool
    call), reports no count rather than a guess."""
    assert PrimeRenderer(SimpleNamespace()).reasoning_tokens([5, 9], _reasoned("a")) is None
    assert (
        PrimeRenderer(SimpleNamespace(_think_end=9)).reasoning_tokens([5], _reasoned("a")) is None
    )
