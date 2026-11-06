"""Thread-safe renderers' calls on a pool's threads; any other's on the loop, as before."""

import asyncio
import threading
import time
from itertools import pairwise
from types import SimpleNamespace

import pytest
from support.fakes import ScriptedSampler
from support.leases import lease

from loops_cookbook.llm.client import Renderers
from loops_cookbook.llm.render_pool import (
    MAX_DEFAULT_WORKERS,
    RenderPool,
    RenderPoolClosed,
    default_workers,
)
from loops_cookbook.llm.rendering import ParsedReply, RenderedPrompt
from loops_cookbook.model_access.lease import DirectModelAccess


@pytest.fixture
def pool():
    pool = RenderPool(4)
    yield pool
    pool.close()


class Overlap:
    """Counts how many calls are inside at once."""

    def __init__(self):
        self._lock = threading.Lock()
        self.inside = self.most = 0
        self.order = []

    def __call__(self, index, seconds=0.02):
        with self._lock:
            self.inside += 1
            self.most = max(self.most, self.inside)
        time.sleep(seconds)
        with self._lock:
            self.inside -= 1
            self.order.append(index)
        return index


async def test_a_thread_safe_renderer_is_called_from_several_threads_at_once(pool):
    renderer = SimpleNamespace(thread_safe=True)
    overlap = Overlap()
    results = await asyncio.gather(*(pool.run(renderer, lambda i=i: overlap(i)) for i in range(8)))
    assert results == list(range(8))
    assert overlap.most == 4  # every worker, and no more


@pytest.mark.parametrize("renderer", [SimpleNamespace(), SimpleNamespace(thread_safe=False)])
async def test_any_other_renderer_is_called_on_the_loop_in_order(pool, renderer):
    threads = []

    def call(index):
        threads.append(threading.current_thread())
        return index

    results = await asyncio.gather(*(pool.run(renderer, lambda i=i: call(i)) for i in range(8)))
    assert results == list(range(8))
    assert threads == [threading.current_thread()] * 8


async def test_a_call_raises_what_the_renderer_raised(pool):
    error = ValueError("bad template")

    def fail():
        raise error

    for renderer in (SimpleNamespace(), SimpleNamespace(thread_safe=True)):
        with pytest.raises(ValueError) as raised:
            await pool.run(renderer, fail)
        assert raised.value is error


def test_a_pool_needs_a_worker():
    with pytest.raises(ValueError, match="at least one worker"):
        RenderPool(0)


def test_the_default_is_a_few_workers_however_many_cpus_the_os_reports():
    assert 1 <= default_workers() <= MAX_DEFAULT_WORKERS
    RenderPool().close()


async def test_a_closed_pool_refuses_every_call():
    """Thread-safe or not, nothing renders after close."""
    pool = RenderPool(2)
    pool.close()
    for renderer in (SimpleNamespace(thread_safe=True), SimpleNamespace()):
        with pytest.raises(RenderPoolClosed):
            await pool.run(renderer, lambda: "after")


class BlockingRenderer:
    """Renders only once the event loop has run a callback: on the loop, it never would."""

    stop_sequences = (99,)
    thread_safe = True

    def __init__(self, loop_ran: threading.Event):
        self.loop_ran = loop_ran
        self.entered = threading.Event()
        self.threads = []

    def render(self, messages, tools):
        self.threads.append(threading.current_thread().name)
        self.entered.set()
        if not self.loop_ran.wait(5):
            raise TimeoutError("the event loop did not run while this rendered")
        return RenderedPrompt.text((1, 2, 3))

    def parse(self, prompt, completion, tools):
        return ParsedReply("done", None, (), True, native_message={"role": "assistant"})


async def test_a_conversation_renders_on_the_pool_while_the_loop_runs(pool):
    loop_ran = threading.Event()
    renderer = BlockingRenderer(loop_ran)
    llm = DirectModelAccess(lease(ScriptedSampler()), Renderers(renderer, pool=pool)).llm()
    generation = asyncio.ensure_future(
        llm.conversation().generate(messages=[{"role": "user", "content": "hi"}])
    )
    # Awaiting the render's entry needs the loop to run while the render blocks.
    await asyncio.to_thread(renderer.entered.wait, 5)
    assert renderer.entered.is_set() and not generation.done()
    loop_ran.set()
    response = await generation
    assert response.sample.prompt_ids == (1, 2, 3)
    assert renderer.threads and all(name.startswith("render") for name in renderer.threads)


async def test_a_generation_cancelled_while_it_renders_samples_nothing(pool):
    loop_ran = threading.Event()
    renderer = BlockingRenderer(loop_ran)
    sampler = ScriptedSampler()
    conversation = (
        DirectModelAccess(lease(sampler), Renderers(renderer, pool=pool)).llm().conversation()
    )
    generation = asyncio.ensure_future(
        conversation.generate(messages=[{"role": "user", "content": "hi"}])
    )
    await asyncio.to_thread(renderer.entered.wait, 5)
    assert renderer.entered.is_set()  # cancelled mid-render, not before it
    generation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await generation
    assert sampler.prompts == []
    # The conversation is free again, and renders from scratch.
    loop_ran.set()
    response = await conversation.generate(messages=[{"role": "user", "content": "hi"}])
    assert response.sample.prompt_ids == (1, 2, 3)
    assert sampler.prompts == [[1, 2, 3]]


class ByteRenderer:
    """A deterministic text renderer with a bridge: each message is its UTF-8 bytes.

    It yields the GIL inside every call, so pooled calls interleave as much as they can.
    """

    stop_sequences = (0,)
    thread_safe = True

    @staticmethod
    def _encode(messages):
        ids = []
        for message in messages:
            time.sleep(0)
            ids += [1, *(byte + 2 for byte in str(message.get("content")).encode()), 1]
        return ids

    def render(self, messages, tools):
        return [*self._encode(messages), 0] if len(tools) else self._encode(messages)

    def bridge(self, prompt_ids, completion_ids, new_messages, tools, *, previous_messages):
        return [*prompt_ids, *completion_ids, *self._encode(new_messages)]

    def parse(self, prompt, completion, tools):
        return ParsedReply(
            "ok", None, (), True, native_message={"role": "assistant", "content": "ok"}
        )


async def _turns(renderers, seed):
    sampler = ScriptedSampler(tokens=lambda call: [seed % 7 + 3, call + 3])
    conversation = DirectModelAccess(lease(sampler), renderers).llm().conversation()
    history = [{"role": "user", "content": f"task {seed} " * (seed % 5 + 1)}]
    samples = []
    for turn in range(5):
        response = await conversation.generate(messages=history)
        samples.append((response.sample.prompt_ids, response.sample.completion_ids))
        history = [
            *history,
            response.history_message,
            {"role": "tool", "content": f"result {seed}.{turn} " * (turn * seed % 50)},
        ]
    return sampler.prompts, samples


async def test_pooled_conversations_sample_exactly_what_the_loop_would(pool):
    """The same prompts reach the sampler, bridged turn by turn, however calls interleave."""
    renderer = ByteRenderer()
    expected = [await _turns(Renderers(renderer), seed) for seed in range(40)]
    got = await asyncio.gather(*(_turns(Renderers(renderer, pool=pool), s) for s in range(40)))
    assert got == expected
    _prompts, samples = expected[3]
    for (prompt, completion), (following, _) in pairwise(samples):
        assert following[: len(prompt) + len(completion)] == (*prompt, *completion)


class CountingRenderer:
    """Renders and parses slowly, recording each call's thread and how many overlap."""

    stop_sequences = (99,)

    def __init__(self, *, thread_safe=False):
        self.thread_safe = thread_safe
        self.overlap = Overlap()
        self.rendered_on, self.parsed_on = [], []

    def render(self, messages, tools):
        self.rendered_on.append(threading.current_thread().name)
        self.overlap("render", 0.01)
        return [1, *range(2, 2 + len(messages))]

    def parse(self, prompt, completion, tools):
        self.parsed_on.append(threading.current_thread().name)
        self.overlap("parse", 0.005)
        return ParsedReply(
            "ok", None, (), True, native_message={"role": "assistant", "content": "ok"}
        )


async def _talk(renderers, turns=3):
    conversation = DirectModelAccess(lease(ScriptedSampler()), renderers).llm().conversation()
    history = [{"role": "user", "content": "go"}]
    for _ in range(turns):
        response = await conversation.generate(messages=history)
        history = [*history, response.history_message, {"role": "user", "content": "more"}]


async def test_any_other_renderer_renders_and_parses_on_the_loop_one_call_at_a_time(pool):
    """With a pool as without one: a renderer that is not thread-safe never leaves the loop."""
    renderer = CountingRenderer()
    await asyncio.gather(*(_talk(Renderers(renderer, pool=pool)) for _ in range(8)))
    assert renderer.overlap.most == 1
    assert len(renderer.rendered_on) == len(renderer.parsed_on) == 24
    assert set(renderer.rendered_on + renderer.parsed_on) == {threading.main_thread().name}


async def test_a_thread_safe_renderer_parses_on_the_loop(pool):
    """A short parse does not queue behind renders; only render and bridge leave the loop."""
    renderer = CountingRenderer(thread_safe=True)
    await asyncio.gather(*(_talk(Renderers(renderer, pool=pool), turns=1) for _ in range(4)))
    assert set(renderer.parsed_on) == {threading.main_thread().name}
    assert all(name.startswith("render") for name in renderer.rendered_on)
