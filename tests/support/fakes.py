"""In-memory stand-ins for the SDK client, the trainer, the sampler and a Prime renderer.

None of these talk to the network. They model the pinned SDK response shapes closely
enough for the runner's contracts; they are not live SDK contract checks.
"""

from __future__ import annotations

import asyncio
import dataclasses
from copy import deepcopy
from types import SimpleNamespace
from typing import NoReturn

import httpx
from baseten.loops.models import EncodedTextChunk, TensorData

from loops_cookbook.llm.rendering import ParsedReply
from loops_cookbook.rl.trainer import StepReport
from loops_cookbook.trainer import Capacity

PAD = 7  # stand-in for an image placeholder token such as <|image_pad|>


async def no_wait(seconds):
    """A ``sleep`` for a backoff or a poll: it yields to the loop and waits no time."""
    await asyncio.sleep(0)


class Future:
    def __init__(self, result):
        self._r = result

    async def result_async(self, timeout=None):
        return self._r


def answer(n: int) -> ParsedReply:
    """The ``n``-th reply, counted from 1: ``answer n``, as plain text."""
    return ParsedReply(
        f"answer {n}", None, (), True, {"role": "assistant", "content": f"answer {n}"}
    )


class ScriptedRenderer:
    """Deterministic tokens from the request's shape; each reply scripted by its call number.

    ``render`` gives a leading token and one more per message and per tool. ``reply(n)``
    is the ``n``-th parse's reply, counted from 1; a test may replace it. With
    ``bridging`` a next turn extends the prior tokens by one token instead of being
    rendered again; ``images`` declares image support. ``renders``, ``bridges`` and
    ``parses`` record what was asked.
    """

    stop_sequences = (99,)

    def __init__(self, name="base", *, reply=answer, bridging=False, images=False):
        self.name = name
        self.reply = reply
        self.supports_images = images
        self.renders, self.bridges = [], []
        self.parses = 0
        if bridging:
            self.bridge = self._bridge

    def render(self, messages, tools):
        self.renders.append((deepcopy(messages), deepcopy(tools)))
        return tuple(range(1, 2 + len(messages) + len(tools)))

    def _bridge(self, prompt, completion, messages, tools, *, previous_messages=()):
        self.bridges.append(deepcopy(messages))
        return (*prompt, *completion, 50 + len(messages))

    def parse(self, prompt, completion, tools):
        self.parses += 1
        return self.reply(self.parses)


class FakeSamplingClient:
    """The sampler half. ``ready_calls`` pins when readiness is settled, not just that."""

    def __init__(self):
        self.ready_calls = 0

    async def ensure_ready_async(self, ready_timeout=None):
        self.ready_calls += 1


CAPABILITIES_URL = "https://api.baseten.co/v1/loops/capabilities"
"""Where the SDK asks Loops what it serves, at its default base URL."""


def offline() -> httpx.ConnectError:
    """What ``ServiceClient.get_server_capabilities_async`` raises with Loops out of
    reach, as in the conftest's default: the model check reports it and does not refuse."""
    return httpx.ConnectError("tests are offline", request=httpx.Request("GET", CAPABILITIES_URL))


async def offline_capabilities() -> NoReturn:
    raise offline()


def scored_like_sampler(data):
    """Loops' per-datum logprobs: the sampler's where the datum has them (no
    trainer/sampler gap), else -1 (an SFT datum)."""
    return [
        {
            "logprobs": TensorData(
                data=(
                    list(datum.loss_fn_inputs["logprobs"].data)
                    if "logprobs" in datum.loss_fn_inputs
                    else [-1.0] * datum.model_input.length
                ),
                dtype="float32",
                shape=[datum.model_input.length],
            )
        }
        for datum in data
    ]


class FakeClient:
    run_id = "run-1"

    def __init__(self):
        self.fb_calls: list[int] = []
        self.fb_spans: list[list[object]] = []
        self.recorded: list[bool] = []
        self.loss_fns: list[tuple] = []
        self.optim_steps = 0
        self.published: list[str] = []
        self.sampler_checkpoints: list[tuple[str, object]] = []
        self.state_checkpoints: list[tuple[str, object]] = []
        self.policy_version = 0
        self.sampling_client = FakeSamplingClient()

    async def get_server_capabilities_async(self):
        # Some tests hand this client back as the ServiceClient too.
        raise offline()

    async def forward_backward_async(self, data, loss_fn, loss_fn_config=None):
        # Models the pinned SDK response; this is not a live SDK contract check.
        self.loss_fns.append((loss_fn, loss_fn_config))
        self.fb_calls.append(len(data))
        self.fb_spans.append([datum.model_input_spans for datum in data])
        summed = (loss_fn_config or {}).get("skip_token_normalization", True)
        metrics = {"mismatch_kl": 0.01, "skip_token_normalization": float(summed)}
        return Future(
            SimpleNamespace(loss=1.0, metrics=metrics, loss_fn_outputs=scored_like_sampler(data))
        )

    async def optim_step_async(self, params):
        self.adam_params = params
        self.optim_steps += 1
        version = self.policy_version = self.policy_version + 1
        # The trainer reports its step count with the step, as Loops does.
        return Future(
            SimpleNamespace(metrics={"grad_norm": 0.25, "step_seconds": 2.0, "step": version})
        )

    async def save_weights_and_get_sampling_client_async(
        self, name="default", *, record_stability_info=False
    ):
        self.published.append(name)
        self.recorded.append(record_stability_info)
        return self.sampling_client

    async def save_weights_for_sampler_async(self, name="default", ttl_seconds=None):
        self.sampler_checkpoints.append((name, ttl_seconds))
        return Future(
            SimpleNamespace(
                version=self.policy_version, path=f"bt://loops:run-1/sampler_weights/{name}"
            )
        )

    async def save_state_async(self, name, ttl_seconds=None):
        self.state_checkpoints.append((name, ttl_seconds))
        return Future(type("State", (), {"path": f"bt://loops:run-1/state/{name}"})())

    async def aclose(self):
        pass


class MemoryTrainer:
    def __init__(self):
        self.policy_version = self.resumed_step = 0
        self.sampler = self
        self.sampler_version = None
        self.capacity = Capacity.DEDICATED
        self.closed = False
        self.rows = []
        self.eval_samplers = []
        # The policy version eval samplers report: None (serverless), or 0 (dedicated).
        self.eval_sampler_reports = None
        self.serverless = False
        self.connected = self.run_active = False
        self.loaded = []
        # The rollout ID that each sampling request sent for Loops to trace.
        self.rollout_ids = []

    async def start(self):
        return self

    async def sample_async(self, prompt, **kwargs):
        self.rollout_ids.append(kwargs.get("rollout_id"))
        version = self.policy_version
        await asyncio.sleep(0)
        return SimpleNamespace(
            prompt_token_ids=None,
            policy_version=version,
            sequences=[
                SimpleNamespace(
                    tokens=[3, 4], logprobs=[-0.2, -0.3], sequence_id=None, stop_reason="stop"
                )
            ],
        )

    async def accumulate(self, rows):
        self.rows.append(rows)
        await asyncio.sleep(0)

    async def commit(self, step, **stats):
        await asyncio.sleep(0)
        before = self.policy_version
        self.policy_version += 1
        return StepReport(
            step,
            before,
            self.policy_version,
            stats["groups"],
            stats["rollouts"],
            len(self.rows[-1]),
            12,
            8,
            0.5,
            0.5,
            0.5,
            0.5,
            min(stats["sampled_versions"]),
            max(stats["sampled_versions"]),
        )

    async def checkpoint(self, report, *, final=False, keep_sampler_weights=False):
        if not keep_sampler_weights:
            return report
        return dataclasses.replace(
            report, sampler_weights=f"bt://loops:fake/sampler_weights/step-{report.step}"
        )

    async def checkpoint_state_only(self, step):
        return None

    async def run_ended(self):
        return None

    async def eval_sampler(self, model_path, *, ready_timeout):
        sampler = EvalSampler(model_path, reports=self.eval_sampler_reports)
        self.eval_samplers.append(sampler)
        return sampler

    async def deactivate_sampler(self, sampler):
        sampler.deactivated = True

    def connect(self):
        self.connected = True

    async def servable_weights(self, state_path, name):
        """``(path, version)``, as a trainer that loaded ``state_path`` saved it as
        ``name``; the state paths given are in ``loaded``."""
        self.loaded.append(state_path)
        self.run_active = True
        return f"bt://loops:eval/sampler_weights/{name}", 7

    async def deactivate_run(self):
        self.run_active = False

    async def close(self):
        self.closed = True


class EvalSampler:
    """A standalone eval sampler. ``reports`` is the policy version it answers with:
    None as a serverless one, its own counter (0) as a dedicated one."""

    def __init__(self, model_path, reports=None):
        self.model_path = model_path
        self.reports = reports
        self.deactivated = False
        self.calls = 0

    async def sample_async(self, prompt, **kwargs):
        self.calls += 1
        await asyncio.sleep(0)
        return SimpleNamespace(
            prompt_token_ids=None,
            policy_version=self.reports,
            sequences=[
                SimpleNamespace(
                    tokens=[3, 4], logprobs=[-0.2, -0.3], sequence_id=None, stop_reason="stop"
                )
            ],
        )


class FakePrime:
    """Emits ``<text> [pad]*N <text>`` per image like Prime's Qwen renderers."""

    is_multimodal = True
    tokens_per_image = 3

    def __init__(self):
        self.bridge_calls = []

    def get_stop_token_ids(self):
        return [99]

    def _render(self, messages):
        from renderers.base import MultiModalData, PlaceholderRange

        ids, placeholders = [], []
        for index, message in enumerate(messages):
            content = message.get("content")
            parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
            ids.append(100 + index)
            for part in parts:
                if part.get("type") == "text":
                    ids.extend([10] * len(part.get("text") or ""))
                else:
                    placeholders.append(PlaceholderRange(len(ids), self.tokens_per_image))
                    ids.extend([PAD] * self.tokens_per_image)
        data = None
        if placeholders:
            data = MultiModalData(
                mm_hashes={"image": ["h"] * len(placeholders)},
                mm_placeholders={"image": placeholders},
                mm_items={"image": [{}] * len(placeholders)},
            )
        return SimpleNamespace(token_ids=ids, multi_modal_data=data)

    def render(self, messages, **kwargs):
        return self._render(messages)

    def bridge_to_next_turn(self, prompt_ids, completion_ids, new_messages, **kwargs):
        from renderers.base import MultiModalData, PlaceholderRange

        self.bridge_calls.append(kwargs)
        prior = list(prompt_ids) + list(completion_ids)
        tail = self._render(new_messages)
        previous = kwargs.get("previous_multi_modal_data")
        merged = list(previous.mm_placeholders.get("image", [])) if previous else []
        if tail.multi_modal_data is not None:
            merged += [
                PlaceholderRange(p.offset + len(prior), p.length)
                for p in tail.multi_modal_data.mm_placeholders["image"]
            ]
        data = MultiModalData(mm_placeholders={"image": merged}) if merged else None
        return SimpleNamespace(token_ids=prior + tail.token_ids, multi_modal_data=data)

    def parse_response(self, completion, *, prompt_ids, tools):
        return SimpleNamespace(
            content="ok", reasoning_content=None, tool_calls=[], reasoning_complete=True
        )


class EchoSampler:
    """Echoes the expanded prompt like the Loops sampler does for image prompts."""

    def __init__(self):
        self.prompts = []

    async def sample_async(self, prompt, **kwargs):
        self.prompts.append(prompt)
        ids = []
        for chunk in prompt.chunks:
            ids.extend(
                chunk.tokens if isinstance(chunk, EncodedTextChunk) else [PAD] * chunk.length
            )
        return SimpleNamespace(
            prompt_token_ids=ids,
            policy_version=0,
            sequences=[
                SimpleNamespace(
                    tokens=[5, 6], logprobs=[-0.1, -0.2], stop_reason="stop", sequence_id=None
                )
            ],
        )


class ScriptedSampler:
    """One scripted completion per call; records prompts and sampling params.

    ``tokens`` may be a callable of the 1-based call index for per-call replies. Set
    ``release`` to an ``asyncio.Event`` to hold a call open after ``entered`` fires.
    """

    def __init__(
        self,
        tokens=(8, 9),
        logprobs=(-0.1, -0.2),
        *,
        stop_reason="length",
        policy_version=0,
        sequence_id=None,
    ):
        self.tokens, self.logprobs = tokens, logprobs
        self.stop_reason, self.policy_version = stop_reason, policy_version
        # A string, or a callable of the request's number, for a recording sampler.
        self.sequence_id = sequence_id
        self.prompts, self.params = [], []
        self.entered = asyncio.Event()
        self.release = None

    async def sample_async(self, prompt, **kwargs):
        self.prompts.append(prompt.to_ints())
        self.params.append(kwargs.get("sampling_params"))
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        tokens = self.tokens(len(self.prompts)) if callable(self.tokens) else list(self.tokens)
        return SimpleNamespace(
            prompt_token_ids=prompt.to_ints(),
            policy_version=self.policy_version,
            sequences=[
                SimpleNamespace(
                    tokens=tokens,
                    logprobs=list(self.logprobs),
                    stop_reason=self.stop_reason,
                    sequence_id=self.sequence_id(len(self.prompts))
                    if callable(self.sequence_id)
                    else self.sequence_id,
                )
            ],
        )
