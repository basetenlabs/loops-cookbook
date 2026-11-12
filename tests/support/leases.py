"""Real attempt leases over fake samplers, for tests that capture through ``LoopsLLM``."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loops_cookbook.evidence import AttemptIdentity, Sample
from loops_cookbook.llm.client import LoopsLLM, Renderers, SamplingPolicy
from loops_cookbook.llm.transcript import SealedTranscript
from loops_cookbook.model_access.lease import (
    AttemptLease,
    DirectModelAccess,
    Freshness,
    LeaseWiring,
)
from loops_cookbook.model_access.producers import ProducerSettings
from loops_cookbook.trainer import Capacity

IDENTITY = AttemptIdentity("run", "group", "rollout", "attempt")


def lease(
    sampler: Any,
    *,
    policy_version: Callable[[], int] | None = None,
    producers: ProducerSettings | None = None,
    sampling_policy: SamplingPolicy | None = None,
    max_steps_off_policy: int = 1_000_000,
    max_policy_delta_in_trajectory: int = 1_000_000,
    affinity_key: str | None = None,
    recorded: list[Sample] | None = None,
    discarded: list[Sample] | None = None,
    transcripts: list[SealedTranscript] | None = None,
    records_transcripts: bool | None = None,
    keep_transcript: Callable[[SealedTranscript], Awaitable[None]] | None = None,
    identity: AttemptIdentity = IDENTITY,
) -> AttemptLease:
    """An attempt lease whose sampler is ``sampler`` and whose trainer is at its version.

    By default the trainer's version is whatever the sampler reports, so a fake
    sampler never looks like it is serving an unpublished policy. Passing
    ``transcripts`` turns recording on, and each seal also appends its transcript;
    ``keep_transcript`` does too, and each seal awaits it with its transcript instead.
    """

    async def record(_identity: AttemptIdentity, sample: Sample) -> None:
        if recorded is not None:
            recorded.append(sample)

    def reject(_identity: AttemptIdentity, sample: Sample) -> None:
        if discarded is not None:
            discarded.append(sample)

    async def keep(_identity: AttemptIdentity, transcript: SealedTranscript) -> None:
        if keep_transcript is not None:
            await keep_transcript(transcript)
        elif transcripts is not None:
            transcripts.append(transcript)

    keeps = transcripts is not None or keep_transcript is not None

    wiring = LeaseWiring(
        sampler=lambda: sampler,
        capacity=Capacity.DEDICATED,
        policy_version=policy_version or (lambda: getattr(sampler, "policy_version", 0)),
        producers=producers or ProducerSettings(),
        freshness=Freshness(max_steps_off_policy, max_policy_delta_in_trajectory),
        sampling_policy=sampling_policy or SamplingPolicy(),
        on_sample=record,
        on_discard=reject,
        on_undelivered=reject,
        records_transcripts=(keeps if records_transcripts is None else records_transcripts),
        on_transcript=keep if keeps else None,
    )
    return AttemptLease(identity, wiring, affinity_key=affinity_key, rollout_id=identity.attempt_id)


def direct_llm(
    sampler: Any, renderer: Any, producer: str | None = None, **options: Any
) -> LoopsLLM:
    """A producer-locked ``LoopsLLM`` capturing into a fresh lease over ``sampler``."""
    return DirectModelAccess(lease(sampler, **options), Renderers(renderer)).llm(producer)
