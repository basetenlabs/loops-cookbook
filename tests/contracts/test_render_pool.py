"""Rendering on a pool's threads gives the sampler exactly what rendering on the loop does.

Real tokenizers and renderers: many conversations at once, each turn bridged from
the last sample through one shared renderer and tokenizer, against the same
conversations rendered one at a time on the event loop. Every prompt the sampler
receives, chunk structure included, and every sample must be byte-identical.
"""

import asyncio
import random
from types import SimpleNamespace

from support.leases import lease
from support.recording_tokenizer import record_writes

from loops_cookbook.llm.client import Renderers
from loops_cookbook.llm.render_pool import RenderPool
from loops_cookbook.model_access.lease import DirectModelAccess

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]
WORDS = [
    *[
        "the",
        "court",
        "held",
        "that",
        "plaintiff's",
        "motion",
        "§",
        "12(b)(6)",
        "indemnification",
        "€1,234.56",
    ],
    *[
        "naïve",
        "café",
        "数据",
        "таблица",
        "🚀",
        "def",
        "f(x):",
        "return",
        "x",
        "**",
        "2",
        "|",
        "col",
        "|",
        "col",
        "|",
    ],
    "\t",
    "\n",
    "    ",
]
CONVERSATIONS, TURNS = 24, 6


class Sampler:
    """Answers every turn with a call to ``bash``, and keeps every prompt it was sent."""

    def __init__(self, reply, seed):
        self.reply, self.rng, self.prompts = reply, random.Random(seed), []

    async def sample_async(self, prompt, **kwargs):
        self.prompts.append(prompt.model_dump_json())
        await asyncio.sleep(self.rng.uniform(0, 0.01))  # interleave the conversations
        return SimpleNamespace(
            prompt_token_ids=prompt.to_ints(),
            policy_version=0,
            sequences=[
                SimpleNamespace(
                    tokens=list(self.reply),
                    logprobs=[-0.1] * len(self.reply),
                    stop_reason="stop",
                    sequence_id=None,
                )
            ],
        )


def _text(rng, words):
    return " ".join(rng.choice(WORDS) for _ in range(words))


async def _conversation(renderers, reply, seed):
    """One agent's turns: a user task, then a tool result after every call."""
    rng = random.Random(seed)
    sampler = Sampler(reply, seed)
    conversation = DirectModelAccess(lease(sampler), renderers).llm().conversation()
    history = [{"role": "user", "content": _text(rng, rng.randint(20, 400))}]
    samples = []
    for turn in range(TURNS):
        response = await conversation.generate(messages=history, tools=TOOLS, max_tokens=64)
        samples.append((tuple(response.sample.prompt_ids), tuple(response.sample.completion_ids)))
        size = rng.choice([10, 200, 2_000, 12_000])
        result = {"role": "tool", "tool_call_id": f"call_{turn}", "content": _text(rng, size)}
        history = [*history, response.history_message, result]
    return sampler.prompts, samples


async def test_pooled_renders_match_the_loop_byte_for_byte(tool_call_model):
    tokenizer, renderer, prefix = tool_call_model
    call = "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
    reply = tokenizer.encode(prefix + call + "<|im_end|>", add_special_tokens=False)

    on_loop = Renderers(renderer)
    expected = [await _conversation(on_loop, reply, seed) for seed in range(CONVERSATIONS)]

    pool = RenderPool(4)
    try:
        pooled = Renderers(renderer, pool=pool)
        got = await asyncio.gather(
            *(_conversation(pooled, reply, seed) for seed in range(CONVERSATIONS))
        )
    finally:
        pool.close()
    assert got == expected


async def test_real_renders_bridges_and_parses_write_nothing_to_the_shared_tokenizer(
    tool_call_model,
):
    """Every call a conversation makes through the renderer only reads the backend."""
    tokenizer, renderer, prefix = tool_call_model
    call = "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
    reply = tokenizer.encode(prefix + call + "<|im_end|>", add_special_tokens=False)
    backend = tokenizer._tokenizer
    writes = record_writes(tokenizer)
    pool = RenderPool(4)
    try:
        pooled = Renderers(renderer, pool=pool)
        await asyncio.gather(*(_conversation(pooled, reply, seed) for seed in range(4)))
    finally:
        pool.close()
        tokenizer._tokenizer = backend
    assert writes == []
