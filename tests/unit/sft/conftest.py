"""An offline chat renderer and Loops trainer, so SFT runs end to end in tests."""

import json
import re
from dataclasses import dataclass, field
from functools import partial

import pytest
from baseten.loops.models import ModelInput
from support.rl_sft import sft_config as build_sft_config

from loops_cookbook.llm.rendering import RenderedConversation, RenderedPrompt
from loops_cookbook.sft.trainer import Score, Update

HEADERS = {"system": 1, "user": 2, "assistant": 3, "tool": 4}
END = 5
"""The fake template's one stop token, closing every message."""


def text(value) -> list[int]:
    return [100 + ord(character) for character in str(value or "")]


class ChatRenderer:
    """``<role> content <end>`` per message; the model writes its content and the end."""

    stop_sequences = (END,)

    def render_conversation(self, messages, tools=()):
        ids, owners, generated = [], [], []
        for index, message in enumerate(messages):
            assistant = message["role"] == "assistant"
            body = text(message.get("reasoning_content")) + text(message.get("content"))
            for call in message.get("tool_calls") or []:
                body += text(json.dumps(call["function"]))
            for token, model in [
                (HEADERS[message["role"]], False),
                *((t, assistant) for t in body),
                (END, assistant),
            ]:
                ids.append(token)
                owners.append(index)
                generated.append(model)
        return RenderedConversation(
            tuple(ids), ModelInput.from_ints(ids), tuple(owners), tuple(generated)
        )

    def render(self, messages, tools=()):
        ids = [*self.render_conversation(messages, tools).ids, HEADERS["assistant"]]
        return RenderedPrompt.text(ids)

    def detokenize(self, ids):
        return "".join(chr(token - 100) if token >= 100 else f"<{token}>" for token in ids)

    def tokenize(self, value):
        return tuple(
            int(special) if special else 100 + ord(character)
            for special, character in re.findall(r"<(\d+)>|(.)", value, flags=re.DOTALL)
        )


PREFILL, MERGED = 7, 8
"""The generation prompt ends with a prefill ("~") that the tokenizer merges with the
turn's first character ("~a") when the turn is rendered whole, as Qwen's does."""


class Merging(ChatRenderer):
    def __init__(self, unwritten=False):
        self.unwritten = unwritten

    def render(self, messages, tools=()):
        return RenderedPrompt.text([*super().render(messages, tools).ids, PREFILL])

    def render_conversation(self, messages, tools=()):
        rendered = super().render_conversation(messages, tools)
        ids, generated = list(rendered.ids), list(rendered.generated)
        for position, token in enumerate(rendered.ids[:-1]):
            if token == HEADERS["assistant"]:
                ids[position + 1] = MERGED
                if self.unwritten:
                    # Template text inside the turn that the model does not write.
                    generated[position + 2] = False
        return type(rendered)(
            tuple(ids), ModelInput.from_ints(ids), rendered.owners, tuple(generated)
        )

    def detokenize(self, ids):
        names = {PREFILL: "~", MERGED: "~a"}
        return "".join(names.get(token) or ChatRenderer.detokenize(self, [token]) for token in ids)


@pytest.fixture
def sft_config(tmp_path):
    """``sft_config(rows=None, validation=None, **sections)``: a valid SFT ``Config`` over
    a small chat dataset."""
    return partial(build_sft_config, tmp_path)


@dataclass
class FakeTrainer:
    """Records what a run asks of Loops; ``fail_at`` raises during that update."""

    resume_from: str | None = None
    fail_at: int | None = None
    fail_keeping: int | None = None
    """Raise while keeping this step: after its update was applied."""
    keep_state_every: int = 0
    ended: str | None = None
    """What ``run_ended`` reports: why the Loops run ended, or ``None`` while it is live."""
    client: object = None
    updates: list = field(default_factory=list)
    evaluations: list = field(default_factory=list)
    kept: list = field(default_factory=list)
    last_gasp: list = field(default_factory=list)
    closed: bool = False

    @property
    def run_id(self):
        return getattr(self.client, "run_id", None)

    async def start(self):
        self.client = type("Client", (), {"run_id": "loops-run-1"})()

    async def run_ended(self):
        return self.ended

    async def update(self, step, datums, learning_rate):
        if step == self.fail_at:
            raise RuntimeError("trainer went away")
        self.updates.append((step, len(datums), learning_rate))
        return Update(
            step,
            len(datums),
            10 * len(datums),
            Score(1.0 / (step + 1), 3 * len(datums)),
            learning_rate,
            0.5,
        )

    async def evaluate(self, datums):
        self.evaluations.append(len(datums))
        return Score(2.0, 7)

    async def checkpoint(self, step, *, final=False):
        if step == self.fail_keeping:
            raise RuntimeError("checkpoint storage went away")
        due = final or (self.keep_state_every and not (step + 1) % self.keep_state_every)
        if not due:
            return None, None
        self.kept.append(step)
        return f"step-{step}", f"bt://state/step-{step}"

    async def checkpoint_state_only(self, step, *, timeout=30.0):
        self.last_gasp.append(step)
        return f"bt://state/step-{step}"

    async def close(self):
        self.closed = True
