"""Who answers for a completion the renderer could not read, in both access modes.

A tool call the renderer reports as broken is the model's output: it reaches the
agent and trains. Every other failure to read a completion is the cookbook's: the
attempt fails whatever the rollout runtime does with the error, is retried as a fresh
attempt, and nothing of it trains or is graded. The driver, lease, gateway and
official client are the real ones; only the rollout runtime, sampler and renderer are fakes.
"""

import contextlib

import openai
import pytest
from support.harness import Runtime, rig

from loops_cookbook.llm.client import Renderers
from loops_cookbook.llm.rendering import ParsedReply, RenderedPrompt
from loops_cookbook.model_access.api import ModelAccessMode
from loops_cookbook.model_access.lease import DirectModelAccess
from loops_cookbook.model_access.openai.access import HttpModelAccess
from loops_cookbook.runtime.api import AttemptAbortReason, EpisodeResult
from loops_cookbook.runtime.driver import Discard

MODEL = "policy"


def _raises(n: int) -> ParsedReply:
    raise ValueError("the renderer cannot read its own template")


def _no_message(n: int) -> ParsedReply:
    return ParsedReply("plain text", None, (), True, None)


def _answer(n: int) -> ParsedReply:
    return ParsedReply("answer", None, (), True, {"role": "assistant", "content": "answer"})


class Renderer:
    """Text-only; each parse is scripted by call number, counted from 1."""

    stop_sequences: tuple[str, ...] = ()

    def __init__(self, *replies):
        self.replies = replies
        self.parses = 0

    def render(self, messages, tools):
        return RenderedPrompt.text([1 + len(str(m.get("content", ""))) for m in messages])

    def parse(self, prompt_ids, completion_ids, tools):
        self.parses += 1
        return self.replies[min(self.parses, len(self.replies)) - 1](self.parses)


def _mode(mode, renderer, gateway):
    """The ``rig`` options for ``mode``, and a model call that swallows nothing."""
    renderers = Renderers(renderer)
    if mode is ModelAccessMode.DIRECT:

        async def call(attempt):
            conversation = attempt.services.model_access.llm().conversation()
            await conversation.generate(messages=[{"role": "user", "content": "hi"}])

        return {"access": lambda lease: DirectModelAccess(lease, renderers)}, call

    async def call(attempt):
        connection = await attempt.services.model_access.http_connection()
        async with openai.AsyncOpenAI(
            base_url=connection.base_url,
            api_key=connection.api_key.get_secret_value(),
            max_retries=0,
        ) as agent:
            await agent.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": "hi"}]
            )

    options = {
        "access": lambda lease: HttpModelAccess(lease, renderers, gateway),
        "model_access": ModelAccessMode.OPENAI_HTTP,
    }
    return options, call


@pytest.mark.parametrize("swallowed", [False, True], ids=["propagated", "swallowed"])
@pytest.mark.parametrize("stop_reason", ["stop", "length"])
@pytest.mark.parametrize("fault", [_raises, _no_message], ids=["raises", "no_message"])
@pytest.mark.parametrize("mode", list(ModelAccessMode), ids=lambda mode: mode.value)
async def test_a_completion_the_renderer_cannot_read_fails_the_attempt_and_is_retried(
    gateway, mode, fault, stop_reason, swallowed
):
    renderer = Renderer(fault, _answer)
    options, call = _mode(mode, renderer, gateway)

    async def sample(attempt):
        with contextlib.suppress(Exception) if swallowed else contextlib.nullcontext():
            await call(attempt)
        return EpisodeResult()

    r = rig(Runtime(on_sample=sample), max_rollout_retries=1, **options)
    r.sampler.stop_reason = stop_reason
    graded = await r.rollout()

    first, second = (call[1] for call in r.runtime.named("sample"))
    assert first.spec.identity.attempt_id != second.spec.identity.attempt_id
    assert [call[2] for call in r.runtime.named("abort")] == [AttemptAbortReason.SAMPLE_FAILED]
    assert [call[1] for call in r.runtime.named("grade")] == [second]  # never the first
    assert len(graded.samples) == 1  # the retry's; nothing of the failed attempt trains
    assert r.driver.gauges()["retried/attempts/rollout_failed/renderer"] == 1.0


@pytest.mark.parametrize("mode", list(ModelAccessMode), ids=lambda mode: mode.value)
async def test_a_renderer_that_always_fails_is_discarded_with_its_cause(gateway, mode):
    options, call = _mode(mode, Renderer(_no_message), gateway)

    async def sample(attempt):
        await call(attempt)
        return EpisodeResult()

    r = rig(Runtime(on_sample=sample), max_rollout_retries=1, **options)
    assert await r.rollout() is Discard.ROLLOUT_FAILED
    assert not r.runtime.named("grade") and not r.runtime.named("finish")
    assert r.driver.gauges()["discard/rollouts/rollout_failed/renderer"] == 1.0
