"""A scripted runtime and a real driver around it, for lifecycle tests.

The capacity broker, attempt lease, and attempt driver are the real ones; only the
rollout runtime, the sampler, and the renderer are fakes. A test scripts the rollout runtime's
hooks as coroutines and reads back every call it received.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from loops_cookbook.evidence import AttemptIdentity, Rollout, TaskRef
from loops_cookbook.llm.client import Renderers, SamplingPolicy
from loops_cookbook.llm.rendering import ParsedReply, RenderedPrompt
from loops_cookbook.llm.transcript import SealedTranscript
from loops_cookbook.model_access.api import ModelAccessMode
from loops_cookbook.model_access.lease import (
    AttemptLease,
    DirectModelAccess,
    Freshness,
    LeaseWiring,
)
from loops_cookbook.model_access.producers import ProducerSettings
from loops_cookbook.runtime.api import (
    AttemptAbortReason,
    AttemptContext,
    EpisodeDisposition,
    EpisodeResult,
    GradeResult,
    Phase,
    RuntimeCapabilities,
)
from loops_cookbook.runtime.capacity import RunCapacity
from loops_cookbook.runtime.context import MetricsSink
from loops_cookbook.runtime.driver import AttemptDriver, DriverSettings, SystemicFailure
from loops_cookbook.trainer import Capacity
from support.resources import no_resources

TASK = TaskRef("task")


class Renderer:
    stop_sequences: tuple[str, ...] = ()

    def render(self, messages, tools):
        return RenderedPrompt.text([1 + len(str(m.get("content", ""))) for m in messages])

    def parse(self, prompt_ids, completion_ids, tools):
        return ParsedReply("ok", None, (), None, {"role": "assistant", "content": "ok"})


class Sampler:
    def __init__(self, version: Callable[[], int]):
        self.version = version
        self.calls = 0
        self.stop_reason = "stop"

    async def sample_async(self, prompt, **kwargs):
        self.calls += 1
        await asyncio.sleep(0)
        return SimpleNamespace(
            prompt_token_ids=prompt.to_ints(),
            policy_version=self.version(),
            sequences=[
                SimpleNamespace(
                    tokens=[5, 6],
                    logprobs=[-0.1, -0.2],
                    sequence_id=None,
                    stop_reason=self.stop_reason,
                )
            ],
        )


Hook = Callable[..., Awaitable[Any]]


async def generate(attempt: AttemptContext, producer: str | None = None) -> None:
    """One model call through the attempt's own client."""
    await (
        attempt.services.model_access.llm(producer)
        .conversation()
        .generate(messages=[{"role": "user", "content": "hi"}])
    )


async def default_sample(attempt: AttemptContext) -> EpisodeResult:
    await generate(attempt)
    return EpisodeResult(metadata={"answer": "episode"})


async def default_grade(attempt: AttemptContext, rollout: Rollout) -> GradeResult:
    number = int(attempt.spec.identity.rollout_id.rsplit("r", 1)[-1] or 0)
    return GradeResult(reward=float(number % 2), metadata={"verdict": "grade"})


@dataclass
class Runtime:
    """A rollout runtime whose hooks are scripted; every call is recorded."""

    capabilities: RuntimeCapabilities = field(
        default_factory=lambda: RuntimeCapabilities(
            model_access_modes=frozenset({ModelAccessMode.DIRECT})
        )
    )
    on_sample: Hook = default_sample
    on_grade: Hook = default_grade
    on_finish: Hook | None = None
    on_abort: Hook | None = None
    calls: list[tuple[Any, ...]] = field(default_factory=list[tuple[Any, ...]])

    async def prepare(self) -> list[TaskRef]:
        return [TASK]

    async def sample(self, attempt: AttemptContext) -> EpisodeResult:
        self.calls.append(("sample", attempt))
        return await self.on_sample(attempt)

    async def grade(self, attempt: AttemptContext, rollout: Rollout) -> GradeResult:
        self.calls.append(("grade", attempt, rollout))
        return await self.on_grade(attempt, rollout)

    async def finish(
        self, attempt: AttemptContext, rollout: Rollout, disposition: EpisodeDisposition
    ) -> None:
        self.calls.append(("finish", attempt, rollout, disposition))
        if self.on_finish is not None:
            await self.on_finish(attempt, rollout, disposition)

    async def abort(self, attempt: AttemptContext, reason: AttemptAbortReason) -> None:
        self.calls.append(("abort", attempt, reason))
        if self.on_abort is not None:
            await self.on_abort(attempt, reason)

    async def close(self) -> None:
        self.calls.append(("close",))

    def named(self, name: str) -> list[tuple[Any, ...]]:
        return [call for call in self.calls if call[0] == name]


class Admission:
    """A fixed start-admission target."""

    def __init__(self, target: int = 1000) -> None:
        self.value = target
        self.groups: list[str] = []
        self.orders: dict[str, int] = {}

    def target(self, group_id: str) -> int:
        return self.value

    def started(self, group_id: str) -> None:
        self.groups.append(group_id)

    def order(self, group_id: str) -> int:
        """Groups in the order this admission first sees them, as if dispatched so."""
        return self.orders.setdefault(group_id, len(self.orders))


@dataclass
class Rig:
    runtime: Any
    driver: AttemptDriver
    capacity: RunCapacity
    sampler: Sampler
    admission: Admission
    systemic: list[SystemicFailure]
    sleeps: list[float]
    version: list[int]
    metrics: MetricsSink

    async def rollout(self, rollout_id: str = "g-r1", task: TaskRef = TASK) -> Rollout | Any:
        """Sample and grade one rollout; waits for background finalization."""
        admitted = await self.driver.sample(task, "g", rollout_id)
        if not hasattr(admitted, "rollout"):
            return admitted
        result = await self.driver.grade(admitted)
        await self.driver.drain()
        return result


async def live() -> str | None:
    """A ``run_ended`` for a Loops run that has not ended."""
    return None


def rig(
    runtime: Any = None,
    *,
    phases: frozenset[Phase] = frozenset(),
    grading_retries: int = 0,
    pending_group_grade: bool = False,
    sampling_slots: int = 8,
    verification_slots: int = 8,
    resources: Any = None,
    producers: ProducerSettings | None = None,
    prestart: int = 0,
    keep_transcript: Callable[[SealedTranscript], Awaitable[None]] | None = None,
    access: Callable[[AttemptLease], Any] | None = None,
    model_access: ModelAccessMode = ModelAccessMode.DIRECT,
    run_ended: Callable[[], Awaitable[str | None]] | None = None,
    **settings: Any,
) -> Rig:
    """A driver over ``runtime``; ``keep_transcript`` records transcripts, kept by it.

    ``access`` builds each attempt's model access in place of direct access through
    the harness renderer; ``model_access`` is the mode the rollout runtime declares.
    """
    runtime = runtime or Runtime()
    if isinstance(runtime, Runtime):
        runtime.capabilities = RuntimeCapabilities(
            model_access_modes=frozenset({model_access}),
            internally_acquired_phases=phases,
            grading_retries=grading_retries,
            pending_group_grade=pending_group_grade,
        )
    version = [0]
    systemic: list[SystemicFailure] = []
    sleeps: list[float] = []
    sampler = Sampler(lambda: version[0])
    freshness = Freshness(
        max_steps_off_policy=settings.pop("max_steps_off_policy", 3),
        max_policy_delta_in_trajectory=settings.pop("max_policy_delta_in_trajectory", 1),
    )

    async def record(identity: AttemptIdentity, sample: Any) -> None:
        return None

    async def keep(identity: AttemptIdentity, transcript: SealedTranscript) -> None:
        assert keep_transcript is not None
        await keep_transcript(transcript)

    wiring = LeaseWiring(
        sampler=lambda: sampler,  # pyright: ignore[reportArgumentType]
        capacity=Capacity.DEDICATED,
        policy_version=lambda: version[0],
        producers=producers or ProducerSettings(),
        freshness=freshness,
        sampling_policy=SamplingPolicy(max_generation_tokens=2),
        on_sample=record,
        on_discard=lambda identity, sample: None,
        on_undelivered=lambda identity, sample: None,
        records_transcripts=keep_transcript is not None,
        on_transcript=None if keep_transcript is None else keep,
    )
    values: dict[str, Any] = {
        "run_id": "run",
        "rollout_timeout_seconds": 5.0,
        "setup_timeout_seconds": 5.0,
        "max_rollout_retries": 0,
        "grading_timeout_seconds": 5.0,
        "cleanup_timeout_seconds": 5.0,
        "cancellation_grace_seconds": 0.2,
        "process_kill_grace_seconds": 0.2,
        "max_steps_off_policy": freshness.max_steps_off_policy,
        "max_policy_delta_in_trajectory": freshness.max_policy_delta_in_trajectory,
        "max_consecutive_failures": 64,
        **settings,
    }
    admission = Admission()
    capacity = RunCapacity(
        sampling_slots=sampling_slots,
        verification_slots=verification_slots,
        admission=admission,
        on_change=lambda: None,
        prestart=prestart,
    )

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    metrics = MetricsSink()
    driver = AttemptDriver(
        runtime,
        runtime.capabilities,
        DriverSettings(**values),
        capacity=capacity,
        leases=lambda identity: AttemptLease(
            identity, wiring, affinity_key=None, rollout_id=identity.attempt_id
        ),
        access=access or (lambda lease: DirectModelAccess(lease, Renderers(Renderer()))),  # pyright: ignore[reportArgumentType]
        resources=resources or no_resources,
        metrics=metrics,
        policy_version=lambda: version[0],
        on_systemic=systemic.append,
        run_ended=run_ended or live,
        sleep=sleep,
    )
    return Rig(runtime, driver, capacity, sampler, admission, systemic, sleeps, version, metrics)
