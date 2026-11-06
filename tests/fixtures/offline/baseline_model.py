"""Deterministic stand-ins for the renderer, the Loops sampler, and the trainer.

The sampler's output depends only on the rollout (read from the affinity key), the
index of the call within that rollout, and the policy version it samples at. It is
therefore independent of scheduling order, so a run's evidence is the same however
its attempts interleave.
"""

from __future__ import annotations

import asyncio
import json
from hashlib import sha256
from types import SimpleNamespace

from baseten.loops.models import EncodedTextChunk, ImageChunk, ModelInput

from loops_cookbook.trainer import Capacity

LETTERS = {300: "A", 301: "B", 302: "C", 303: "D"}
LETTER_TOKENS = {letter: token for token, letter in LETTERS.items()}
END = 399
ROLE_TOKENS = {"system": 1, "user": 2, "assistant": 3, "tool": 4}
TOOLS_TOKEN = 5
IMAGE_TOKEN = 6
# Letters a tool-offering prompt answers through ``write_answer`` instead of text.
TOOL_CALL_LETTERS = frozenset({"A", "C"})

WRITE_ANSWER_TOOL = {
    "name": "write_answer",
    "description": "Write the chosen label to /app/answer.txt.",
    "parameters": {
        "type": "object",
        "properties": {"letter": {"type": "string", "enum": ["A", "B", "C", "D"]}},
        "required": ["letter"],
    },
}


def _digest(*parts: object) -> int:
    return int(sha256(":".join(map(str, parts)).encode()).hexdigest(), 16)


class BaselineRenderer:
    """Characters to tokens and back; four letter tokens and one end token.

    A plain-text message renders exactly as it always has. Tool declarations and
    image parts add their own marker tokens, so a prompt that uses neither is
    unchanged. An assistant message, tool calls included, renders back to the tokens
    it was sampled as, so a follow-up prompt extends the sampled prefix. With tools offered, a reply of ``A`` or ``C`` parses as a
    ``write_answer`` call carrying that letter.
    """

    stop_sequences: tuple[str, ...] = ()
    supports_images = True

    def render(self, messages, tools):
        from loops_cookbook.llm.images import model_input_from_placeholders
        from loops_cookbook.llm.rendering import RenderedPrompt

        ids: list[int] = []
        if tools:
            ids.append(TOOLS_TOKEN)
            ids.extend(_characters(json.dumps(list(tools), sort_keys=True)))
        placeholders: list[tuple[int, int]] = []
        images: list[tuple[bytes, str]] = []
        for message in messages:
            ids.append(ROLE_TOKENS[message["role"]])
            if message["role"] == "assistant":
                ids.extend(_reply_tokens(message))
                continue
            content = message["content"]
            if not isinstance(content, list):
                ids.extend(_characters(str(content or "")))
                continue
            for part in content:
                if part.get("type") == "image":
                    data = part["image"]
                    length = _image_tokens(data)
                    placeholders.append((len(ids), length))
                    images.append((data, "png"))
                    ids.extend([IMAGE_TOKEN] * length)
                else:
                    ids.extend(_characters(part["text"]))
        ids.append(ROLE_TOKENS["assistant"])
        if not images:
            return RenderedPrompt.text(ids)
        return RenderedPrompt(tuple(ids), model_input_from_placeholders(ids, placeholders, images))

    def parse(self, prompt_ids, completion_ids, tools):
        from renderers.base import ParsedToolCall

        from loops_cookbook.llm.rendering import ParsedReply

        text = "".join(LETTERS.get(token, "") for token in completion_ids)
        if tools and text in TOOL_CALL_LETTERS:
            call_id = f"call-{len(prompt_ids)}"
            arguments = {"letter": text}
            return ParsedReply(
                content="",
                reasoning_content=None,
                tool_calls=(
                    ParsedToolCall(
                        raw=text,
                        name=WRITE_ANSWER_TOOL["name"],
                        arguments=arguments,
                        token_span=(0, 1),
                        id=call_id,
                    ),
                ),
                reasoning_complete=None,
                native_message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": WRITE_ANSWER_TOOL["name"], "arguments": arguments},
                        }
                    ],
                },
            )
        return ParsedReply(
            content=text,
            reasoning_content=None,
            tool_calls=(),
            reasoning_complete=None,
            native_message={"role": "assistant", "content": text},
        )


def _characters(text: str) -> list[int]:
    return [10 + ord(character) % 200 for character in text]


def _reply_tokens(message) -> list[int]:
    """The tokens an assistant message was sampled as: letters, then the end token."""
    text = str(message.get("content") or "")
    for call in message.get("tool_calls") or ():
        text += call["function"]["arguments"]["letter"]
    tokens = [LETTER_TOKENS[character] for character in text if character in LETTER_TOKENS]
    return [*tokens, END]


def _image_tokens(data: bytes) -> int:
    """One placeholder per 16x16 patch, read from the PNG header."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("the baseline renderer only reads PNG images")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return max(1, (width // 16) * (height // 16))


def expanded_ids(prompt: ModelInput) -> list[int]:
    """The prompt as the sampler sees it: each image chunk becomes its placeholder run."""
    ids: list[int] = []
    for chunk in prompt.chunks:
        if isinstance(chunk, ImageChunk):
            ids.extend([IMAGE_TOKEN] * chunk.expected_tokens)
        elif isinstance(chunk, EncodedTextChunk):
            ids.extend(chunk.tokens)
        else:
            raise TypeError(f"the baseline sampler cannot read {type(chunk).__name__}")
    return ids


def create_renderer(tokenizer=None, chat_template_kwargs=None) -> BaselineRenderer:
    return BaselineRenderer()


class BaselineSampler:
    """Answers with one letter chosen by rollout, call index, and policy version."""

    def __init__(self, policy_version):
        self._policy_version = policy_version
        self._calls: dict[str, int] = {}

    async def sample_async(self, prompt: ModelInput, **kwargs):
        key = kwargs.get("affinity_key")
        if not key:
            raise ValueError("the baseline sampler needs sampler_replica_affinity")
        rollout_id = key.split(":", 1)[1]
        index = self._calls.get(rollout_id, 0)
        self._calls[rollout_id] = index + 1
        version = self._policy_version()
        await asyncio.sleep(0)
        value = _digest(rollout_id, index, version)
        letter = 300 + value % 4
        logprobs = [-((value >> 8) % 1000) / 1000 - 0.01, -((value >> 20) % 1000) / 1000 - 0.01]
        return SimpleNamespace(
            prompt_token_ids=expanded_ids(prompt),
            policy_version=version,
            sequences=[
                SimpleNamespace(
                    tokens=[letter, END],
                    logprobs=logprobs,
                    sequence_id=None,
                    stop_reason="stop",
                )
            ],
        )


class BaselineTrainer:
    """The trainer port with no network: counts versions and keeps every datum."""

    def __init__(self):
        self.policy_version = 0
        self.resumed_step = 0
        self.sampler = BaselineSampler(lambda: self.policy_version)
        self.sampler_version = None
        self.capacity = Capacity.DEDICATED
        self.steps: list[dict] = []
        self._rows: list = []

    async def start(self):
        return self.sampler

    async def accumulate(self, rows):
        self._rows = list(rows)
        await asyncio.sleep(0)

    async def commit(self, step, **stats):
        from loops_cookbook.rl.trainer import StepReport

        before = self.policy_version
        self.policy_version += 1
        self.steps.append(
            {
                "step": step,
                "policy_version_before": before,
                # Sorted: member order follows grading completion, which is timing.
                "rewards": sorted(stats["rewards"]),
                "advantages": sorted(round(value, 6) for value in stats["advantages_"]),
                "sampled_versions": sorted(stats["sampled_versions"]),
            }
        )
        return StepReport(
            step,
            before,
            self.policy_version,
            stats["groups"],
            stats["rollouts"],
            len(self._rows),
            sum(row.datum.model_input.length for row in self._rows),
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            min(stats["sampled_versions"]),
            max(stats["sampled_versions"]),
        )

    async def checkpoint(self, report, *, final=False, keep_sampler_weights=False):
        return report

    async def checkpoint_state_only(self, step):
        return None

    async def run_ended(self):
        return None

    async def close(self):
        return None
