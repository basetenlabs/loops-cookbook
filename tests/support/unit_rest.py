"""Helpers shared by the unit tests of the client, model access, resources and the repo scripts."""

import importlib.util
import io
from pathlib import Path
from types import ModuleType

from PIL import Image

from loops_cookbook.llm.rendering import ParsedReply
from support.fakes import ScriptedSampler
from support.leases import direct_llm

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` as a module, without running its ``__main__`` block."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jpeg(picture: bytes) -> bytes:
    """The same picture, encoded as JPEG."""
    buffer = io.BytesIO()
    Image.open(io.BytesIO(picture)).convert("RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


# One shell tool, as an agent declares it in a Chat Completions request.
BASH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


class ChainRenderer:
    """Renders a history to one token per message; bridges by appending a separator."""

    stop_sequences = (99,)

    def __init__(self):
        self.renders = 0

    def render(self, messages, tools):
        self.renders += 1
        return (1, *(10 + len(str(message.get("content"))) for message in messages))

    def bridge(self, prompt, completion, messages, tools, *, previous_messages=()):
        return (*prompt, *completion, 3, *(30 + len(m["content"]) for m in messages))

    def parse(self, prompt, completion, tools):
        return ParsedReply(
            "a", None, (), True, native_message={"role": "assistant", "content": "a"}
        )


def _completion(call):
    return [40 + call, 50 + call]


async def captured_chain():
    """Bridged turns, a discarded truncated turn and its retry, then a forked re-render."""
    sampler, renderer = ScriptedSampler(tokens=_completion), ChainRenderer()
    llm = direct_llm(sampler, renderer)
    chat = llm.conversation()
    history = [{"role": "user", "content": "start"}]
    prompts = []

    async def turn(messages):
        response = await chat.generate(messages=messages, max_tokens=2)
        prompts.append(tuple(response.sample.prompt_ids))
        return response

    first = await turn(history)
    history += [{"role": "assistant", "content": "a"}, {"role": "tool", "content": "r1"}]
    second = await turn(history)
    history += [{"role": "assistant", "content": "a"}, {"role": "tool", "content": "r22"}]
    truncated = await turn(history)
    chat.discard(truncated)
    retried = await turn(history)
    forked_history = [{"role": "user", "content": "edited start"}, *history[1:]]
    forked = await turn(forked_history)
    forked_history += [{"role": "assistant", "content": "a"}, {"role": "tool", "content": "r"}]
    after_fork = await turn(forked_history)
    assert renderer.renders == 2  # the first turn and the fork; the rest bridged
    responses = (first, second, truncated, retried, forked, after_fork)
    return llm, [response.sample for response in responses], prompts
