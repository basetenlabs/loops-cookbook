"""One attempt's model-access lease: admission, attribution, capture, and sealing.

Every generation is admitted under a cookbook request ID bound to the attempt,
producer, and conversation, captured when the sampler returns, and settled when
the result reaches the caller. Attribution uses identities only; no prompt,
response, or tool-call content is ever inspected.

Lease lifecycle::

    OPEN --seal()--> SEALED
    OPEN --revoke()--> REVOKED

Request lifecycle::

    ADMITTED --capture--> SAMPLED --delivered--> DELIVERED
    ADMITTED | SAMPLED --failure--> ABORTED

A direct generation is delivered when it returns to its caller. An HTTP one stays
``SAMPLED`` until its transport writes the whole response, which a keyed retry
replaying it may still do after the first write failed.

Only delivered generations train. Whatever else was sampled when the attempt seals
(cut mid-write, abandoned by a client that timed out and retried, cut by a deadline,
or still running) is dropped from the evidence, and the rest of the episode trains. A
dropped generation the agent never received is a branch the episode never took; one
it may have partly seen only loses its own training signal, since its text reaches
later prompts as context either way. A cancelled generation is simply not evidence.

A generation that fails after admission poisons the lease, and the attempt's failure
keeps its cause: later requests and sealing raise ``ModelTransportFailure`` for a
transport or HTTP-status error, ``SamplingFailure`` for a sampler timeout,
termination, or unusable response (a completion longer than the ``max_tokens`` it
was sampled with among them), ``RendererFailure`` for a completion the renderer
could not read, and ``CaptureIntegrityFailure`` for anything else.
The Loops SDK has already retried what it treats as transient. A generation that
would exceed the freshness limits marks the lease stale instead, which the driver
turns into a discard whether or not the runtime propagates the error.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from uuid import uuid4

from baseten.loops.models import SamplingParams
from baseten.loops.sampling_client import (
    SamplerTerminatedError,
    SampleTimeoutError,
    SamplingClient,
)

from loops_cookbook.cancellation import Waited, outlast
from loops_cookbook.evidence import (
    AttemptIdentity,
    InternalPolicyDeltaError,
    Sample,
    StaleRolloutError,
    TokenBuffers,
    TokenSpan,
    TrajectoryError,
    compact_ids,
    compact_logprobs,
)
from loops_cookbook.llm.client import LoopsLLM, Renderers, SamplingPolicy
from loops_cookbook.llm.rendering import RenderedPrompt, RendererFailure
from loops_cookbook.llm.transcript import SealedTranscript, TranscriptRecorder, seal_transcript
from loops_cookbook.model_access.api import (
    CaptureIntegrityFailure,
    FuturePolicyVersion,
    ModelAccessError,
    ModelAccessMode,
    ModelConnection,
    ModelTransportFailure,
    SamplingFailure,
    UndeclaredProducer,
)
from loops_cookbook.model_access.producers import ProducerSettings
from loops_cookbook.trainer import Capacity

logger = logging.getLogger(__name__)


def affinity_key(namespace: str, identity: AttemptIdentity) -> str:
    """The sampler affinity key of ``identity``'s rollout, the same for each of its
    attempts: ``namespace`` (one per run) and a digest of the rollout ID. The rollout
    ID carries its task's ID, which is the user's and may not fit in a header."""
    return f"{namespace}:{hashlib.sha256(identity.rollout_id.encode()).hexdigest()[:32]}"


class RequestState(str, Enum):
    ADMITTED = "admitted"
    SAMPLED = "sampled"
    DELIVERED = "delivered"
    ABORTED = "aborted"


class LeaseState(str, Enum):
    OPEN = "open"
    SEALED = "sealed"
    REVOKED = "revoked"


# Told of a sample whose reply raised instead of reaching the agent, so its turn names it.
type FailedReply = Callable[[Sample, Exception], None]


@dataclass(slots=True)
class _Request:
    request_id: str
    producer: str
    conversation_id: str
    state: RequestState = RequestState.ADMITTED
    sample: Sample | None = None
    discarded: bool = False


@dataclass(frozen=True, slots=True)
class Freshness:
    """The limits a generation is checked against at capture."""

    max_steps_off_policy: int
    max_policy_delta_in_trajectory: int


@dataclass(frozen=True, slots=True)
class LeaseWiring:
    """Run-scoped collaborators every attempt lease shares."""

    sampler: Callable[[], SamplingClient]
    policy_version: Callable[[], int]
    # Decides how often an attempt reads ``sampler``; see ``AttemptLease._handle``.
    capacity: Capacity
    producers: ProducerSettings
    freshness: Freshness
    sampling_policy: SamplingPolicy
    on_sample: Callable[[AttemptIdentity, Sample], Awaitable[None]]
    # The agent discarded a generation it was delivered.
    on_discard: Callable[[AttemptIdentity, Sample], None]
    # Sealing dropped a generation that was sampled but never delivered.
    on_undelivered: Callable[[AttemptIdentity, Sample], None]
    # Record each attempt's transcript; its seal returns it.
    records_transcripts: bool = False
    # Also hand each sealed transcript here, before the seal settles anything else.
    on_transcript: Callable[[AttemptIdentity, SealedTranscript], Awaitable[None]] | None = None
    # The policy version the handle ``sampler`` returns was published at.
    sampler_version: Callable[[], int | None] = lambda: None
    # The sampler records every generation, for router replay; it records text only.
    records_generations: bool = False
    # A prompt with images reached a sampler that records generations.
    on_unrecordable_prompt: Callable[[], None] = lambda: None

    def __post_init__(self) -> None:
        if self.on_transcript is not None and not self.records_transcripts:
            raise ValueError("on_transcript needs records_transcripts")


@dataclass(frozen=True, slots=True)
class Sealed:
    """What a seal returns: the attempt's evidence, and its transcript when recorded.

    ``discarded_versions`` are the policy versions of the trainable generations the
    agent took back: none of them trains, but they say which policy the attempt ran.
    """

    samples: tuple[Sample, ...]
    transcript: SealedTranscript | None = None
    discarded_versions: tuple[int, ...] = ()


# How many tokens at each end of a text prompt the echo check compares.
_ECHO_EDGE = 64


def _echo_matches(returned: Sequence[int], ids: TokenSpan, has_images: bool) -> bool:
    """Whether the prompt the sampler reports is the one submitted.

    An image prompt is compared token by token, packed rather than as a tuple: the
    sampler expands each ImageChunk itself, and this is the guard that its expansion
    matches the renderer's placeholders. A text prompt reaches the sampler as exactly
    these IDs, and the SDK reports its own request IDs when the sampler echoes none,
    so the length and both ends suffice. Comparing all of it would walk a prompt of
    100k+ tokens on the event loop at every generation.
    """
    if has_images:
        return ids.same_tokens(returned)
    size = len(ids)
    if len(returned) != size:
        return False
    edge = min(size, _ECHO_EDGE)
    return ids[:edge] == tuple(returned[:edge]) and ids[size - edge :] == tuple(
        returned[size - edge :]
    )


class AttemptLease:
    def __init__(
        self, identity: AttemptIdentity, wiring: LeaseWiring, *, affinity_key: str | None
    ) -> None:
        if affinity_key == "":
            raise ValueError("affinity_key must not be empty")
        self.identity = identity
        self._wiring = wiring
        self._affinity_key = affinity_key
        self._sampler: SamplingClient | None = None
        self._sampler_version: int | None = None
        self._requests: dict[str, _Request] = {}
        self._captured: list[str] = []
        # Each held sample's prompt is a span of its conversation's one buffer, so
        # the attempt retains its tokens once rather than once per generation.
        self._tokens = TokenBuffers()
        self._state = LeaseState.OPEN
        self._poison: BaseException | None = None
        self._stale: StaleRolloutError | InternalPolicyDeltaError | None = None
        self._fatal: FuturePolicyVersion | UndeclaredProducer | None = None
        self._on_close: list[Callable[[], None]] = []
        self._transcript = TranscriptRecorder() if wiring.records_transcripts else None
        self._unsealed_kept = False

    @property
    def state(self) -> LeaseState:
        return self._state

    def on_close(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` once when the lease stops admitting, by sealing or revocation."""
        if self._state is not LeaseState.OPEN:
            callback()
            return
        self._on_close.append(callback)

    def _close(self, state: LeaseState) -> None:
        self._state = state
        callbacks, self._on_close = self._on_close, []
        for callback in callbacks:
            callback()

    @property
    def stale(self) -> StaleRolloutError | InternalPolicyDeltaError | None:
        """The freshness violation that made this attempt unusable, if any."""
        return self._stale

    @property
    def failure(
        self,
    ) -> SamplingFailure | ModelTransportFailure | RendererFailure | CaptureIntegrityFailure | None:
        """Why this attempt can no longer train, keeping the lost generation's cause."""
        if self._poison is None:
            return None
        if isinstance(self._poison, (SamplingFailure, ModelTransportFailure, RendererFailure)):
            kept = type(self._poison)(f"an earlier generation failed: {self._poison}")
        else:
            kept = CaptureIntegrityFailure("a generation in this attempt failed after admission")
        kept.__cause__ = self._poison
        return kept

    @property
    def fatal(self) -> FuturePolicyVersion | UndeclaredProducer | None:
        """What proves the run itself inconsistent or misconfigured, if anything: a
        capture from an unpublished policy, or a call naming an undeclared producer."""
        return self._fatal

    @property
    def sampling_policy(self) -> SamplingPolicy:
        return self._wiring.sampling_policy

    @property
    def transcript(self) -> TranscriptRecorder | None:
        """Where model access records this attempt's turns, when the run records them."""
        return self._transcript

    def producer(self, producer: str | None) -> str:
        """The declared producer a call names; an undeclared one is fatal to the run,
        whatever the runtime does with the refusal."""
        try:
            return self._wiring.producers.resolve(producer)
        except UndeclaredProducer as error:
            self._fatal = self._fatal or error
            raise

    def captured(self) -> tuple[Sample, ...]:
        """Evidence captured so far, in capture-return order, before sealing."""
        self._raise_unusable()
        return self._live_samples()

    def _live_samples(self) -> tuple[Sample, ...]:
        samples: list[Sample] = []
        for request_id in self._captured:
            request = self._requests[request_id]
            if request.sample is not None and not request.discarded:
                samples.append(request.sample)
        return tuple(samples)

    def _raise_unusable(self) -> None:
        if self._fatal is not None:
            raise self._fatal
        if self._stale is not None:
            raise self._stale
        if (failure := self.failure) is not None:
            raise failure

    def _admit(self, producer: str, conversation_id: str) -> _Request:
        if self._state is not LeaseState.OPEN:
            raise CaptureIntegrityFailure(f"model access for this attempt is {self._state.value}")
        self._raise_unusable()
        self.producer(producer)  # an undeclared producer never samples
        request = _Request(uuid4().hex, producer, conversation_id)
        self._requests[request.request_id] = request
        return request

    async def generate(
        self,
        producer: str,
        prompt: RenderedPrompt,
        sampling_params: SamplingParams,
        *,
        conversation_id: str,
        clamped_from: int | None = None,
        on_failed_reply: FailedReply | None = None,
    ) -> Sample:
        """Sample for a direct caller; returning the sample is its delivery."""
        request = self._admit(producer, conversation_id)
        sample = await self._capture(
            request, prompt, sampling_params, clamped_from, on_failed_reply
        )
        request.state = RequestState.DELIVERED
        return sample

    async def generate_for_delivery(
        self,
        producer: str,
        prompt: RenderedPrompt,
        sampling_params: SamplingParams,
        *,
        conversation_id: str,
        clamped_from: int | None = None,
        on_failed_reply: FailedReply | None = None,
    ) -> Sample:
        """Sample for a transport that settles delivery itself, by ``sample_id``."""
        request = self._admit(producer, conversation_id)
        return await self._capture(request, prompt, sampling_params, clamped_from, on_failed_reply)

    def delivered(self, sample_id: str) -> None:
        """The transport wrote the whole response, including a stream's termination.

        Does nothing once the lease has closed or the generation was aborted.
        """
        request = self._requests.get(sample_id)
        if request is None or request.sample is None:
            raise ModelAccessError(f"generation {sample_id!r} was not sampled by this attempt")
        if self._state is LeaseState.OPEN and request.state is not RequestState.ABORTED:
            request.state = RequestState.DELIVERED

    async def _capture(
        self,
        request: _Request,
        prompt: RenderedPrompt,
        sampling_params: SamplingParams,
        clamped_from: int | None,
        on_failed_reply: FailedReply | None,
    ) -> Sample:
        try:
            sample = await self._sample(request, prompt, sampling_params, clamped_from)
            if self._state is not LeaseState.OPEN:
                raise CaptureIntegrityFailure(
                    f"model access was {self._state.value} while this generation ran"
                )
            request.sample, request.state = sample, RequestState.SAMPLED
            self._captured.append(request.request_id)
            await self._wiring.on_sample(self.identity, sample)
            self._check_freshness()
        except (StaleRolloutError, InternalPolicyDeltaError) as error:
            request.state = RequestState.ABORTED
            self._stale = self._stale or error
            if request.sample is not None and on_failed_reply is not None:
                # Sampled, its row written, then found stale: its turn names it.
                on_failed_reply(request.sample, error)
            raise
        except FuturePolicyVersion as error:
            request.state = RequestState.ABORTED
            self._fatal = error
            raise
        except asyncio.CancelledError:
            # Whoever awaited it gave up on it: it is not evidence, and not a failure.
            request.state = RequestState.ABORTED
            raise
        except BaseException as error:
            request.state = RequestState.ABORTED
            if self._poison is None:
                self._poison = error
            raise
        return sample

    def _handle(self) -> tuple[SamplingClient, int | None]:
        """The sampler handle a generation uses, and the version it was published at.

        Every generation is served by the newest weights, by each capacity's means.
        A dedicated handle is bound once, at the attempt's first generation: its
        paired sampler hot-swaps to each publication and serves whatever live adapter
        meets the handle's ``min_policy_version`` floor, so later generations get new
        weights through it. A newer handle would raise the floor to a version no
        replica may have loaded yet, and park the generation until one has.

        A serverless handle names one publication's adapter, which serves as soon as
        it is named, so it is read at every generation. Its version is read with it:
        the trainer swaps both with no await between them, and a publication while a
        request is in flight does not relabel what that handle served.
        """
        if self._wiring.capacity is Capacity.SERVERLESS:
            return self._wiring.sampler(), self._wiring.sampler_version()
        if self._sampler is None:
            self._sampler = self._wiring.sampler()
            self._sampler_version = self._wiring.sampler_version()
        return self._sampler, self._sampler_version

    async def _sample(
        self,
        request: _Request,
        prompt: RenderedPrompt,
        sampling_params: SamplingParams,
        clamped_from: int | None,
    ) -> Sample:
        sampler, sampler_version = self._handle()
        has_images = any(chunk.type != "encoded_text" for chunk in prompt.prompt_input.chunks)
        if has_images and self._wiring.records_generations:
            self._wiring.on_unrecordable_prompt()
            raise SamplingFailure(
                "this prompt has images, and Loops cannot record an image prompt for router "
                "replay: set [runtime] has_images = true and [training] router_replay = false"
            )
        try:
            response = await sampler.sample_async(
                prompt=prompt.prompt_input,
                num_samples=1,
                sampling_params=sampling_params,
                affinity_key=self._affinity_key,
                # Loops traces it; the run's samples and transcript are filed under it.
                rollout_id=self.identity.attempt_id,
                # A recorded generation carries its routes, for router replay; any
                # other generation is told not to capture them.
                include_routed_experts=self._wiring.records_generations,
            )
        except (SampleTimeoutError, SamplerTerminatedError) as error:
            raise SamplingFailure(f"the Loops sampler failed: {error}") from error
        except Exception as error:
            raise ModelTransportFailure(f"Loops sampling request failed: {error}") from error
        returned = response.prompt_token_ids
        if returned is not None and not _echo_matches(returned, prompt.ids, has_images):
            if not returned and has_images:
                raise SamplingFailure(
                    "sampler did not echo the expanded prompt; image prompts require it"
                )
            raise SamplingFailure("sampler prompt differs from submitted token IDs")
        if len(response.sequences) != 1:
            raise SamplingFailure("expected exactly one sampled sequence")
        version = response.policy_version
        if version is None:
            # A pinned (serverless) sampler reports no version of its own; the one
            # its handle was published at is exactly the weights that served it.
            version = sampler_version
        if version is None:
            raise SamplingFailure("the sampler reported no policy version")
        if version > self._wiring.policy_version():
            raise FuturePolicyVersion(
                f"sampler reported policy {version}, newer than the "
                f"trainer's {self._wiring.policy_version()}"
            )
        sequence = response.sequences[0]
        if sequence.logprobs is None:
            raise SamplingFailure("the sampler returned no behavior logprobs")
        try:
            prompt_ids = self._tokens.span(request.conversation_id, prompt.ids)
            return Sample(
                sample_id=request.request_id,
                producer=request.producer,
                conversation_id=request.conversation_id,
                trainable=self._wiring.producers.trainable(request.producer),
                prompt_ids=prompt_ids,
                # Packed straight from the SDK's lists, never a tuple of them: a
                # graded rollout waits for training holding these.
                completion_ids=compact_ids(sequence.tokens),
                logprobs=compact_logprobs(sequence.logprobs),
                policy_version=version,
                # The evidence keeps the prompt's chunk structure, not the request's
                # ModelInput: its text is read back from the shared span when needed.
                prompt_chunks=self._tokens.chunks(prompt_ids, prompt.prompt_input),
                stop_reason=sequence.stop_reason,
                sequence_id=sequence.sequence_id,
                # Checked against the completion: one past it is not evidence.
                max_tokens=sampling_params.max_tokens,
                clamped_from=clamped_from,
            )
        except TrajectoryError as error:
            raise SamplingFailure(f"sampler returned unusable evidence: {error}") from error

    def _check_freshness(self) -> None:
        versions = [sample.policy_version for sample in self._live_samples() if sample.trainable]
        if not versions:
            return
        limits = self._wiring.freshness
        if max(versions) - min(versions) > limits.max_policy_delta_in_trajectory:
            raise InternalPolicyDeltaError("rollout exceeds internal policy-version spread")
        if self._wiring.policy_version() - min(versions) > limits.max_steps_off_policy:
            raise StaleRolloutError("rollout exceeds policy freshness limit")

    def reply_failed(self, sample_id: str, failure: RendererFailure) -> None:
        """The renderer could not read this generation's completion: the attempt fails.

        The sample is no longer delivered, and sealing raises ``failure``.
        """
        request = self._requests.get(sample_id)
        if request is None or request.sample is None:
            raise ModelAccessError(f"generation {sample_id!r} was not sampled by this attempt")
        request.state = RequestState.ABORTED
        if self._poison is None:
            self._poison = failure

    def discard(self, producer: str, sample_id: str) -> None:
        """Exclude a delivered generation from sealed evidence.

        Marking a sample untrainable is not enough. Chains cover samples by exact
        token prefix and are judged by their last member, so a kept-but-untrainable
        generation still splits the chain and still leaves a datum whose final
        target is a completion the agent rejected. Only removal undoes that.
        """
        if self._state is not LeaseState.OPEN:
            raise CaptureIntegrityFailure(f"model access for this attempt is {self._state.value}")
        request = self._requests.get(sample_id)
        if request is None or request.producer != producer or request.sample is None:
            raise ModelAccessError(f"generation {sample_id!r} does not belong to this client")
        if request.state is not RequestState.DELIVERED or request.discarded:
            raise ModelAccessError(f"generation {sample_id!r} cannot be discarded")
        request.discarded = True
        self._wiring.on_discard(self.identity, request.sample)

    async def seal(self) -> Sealed:
        """Close model access and return the attempt's evidence: its delivered generations.

        Raises the attempt's freshness violation or capture failure instead when
        there is one, once the transcript the run keeps is handed to it as far as the
        attempt got, and a ``CaptureIntegrityFailure`` when the generations and the
        recorded turns disagree. Every other sampled generation is dropped from the
        evidence, and reported only once the seal has succeeded.

        Keeping the transcript is awaited here, not left to run: a seal that returns
        has handed it over. Access is closed before the first await, so nothing is
        sampled meanwhile. A seal cancelled while it keeps the transcript still settles
        -- the write, in a thread, lands either way -- and then passes the cancellation
        on. A transcript that cannot be written is logged, never the attempt's failure:
        how an attempt is recorded does not decide whether it trains.
        """
        if self._state is not LeaseState.OPEN:
            raise CaptureIntegrityFailure(f"model access for this attempt is {self._state.value}")
        self._close(LeaseState.SEALED)
        discarded, dropped = self._settle()
        versions = tuple(
            request.sample.policy_version
            for key in discarded
            if (request := self._requests[key]).sample is not None and request.sample.trainable
        )
        try:
            self._raise_unusable()
        except Exception:
            await self._keep_unsealed(discarded, dropped)
            raise
        evidence = self._live_samples()
        if self._transcript is None:
            self._report_undelivered(dropped)
            return Sealed(evidence, discarded_versions=versions)
        try:
            transcript = seal_transcript(
                self._transcript.recorded(),
                evidence=evidence,
                discarded=discarded,
                undelivered={sample.sample_id for sample in dropped},
            )
        except ValueError as error:
            raise CaptureIntegrityFailure(f"the attempt's transcript is incomplete: {error}") from (
                error
            )
        waited = await self._hand_over(transcript)
        self._report_undelivered(dropped)
        if waited.interrupted:
            # The cancellation aborts the attempt, so what it leaves must not say it sealed.
            unsealed = replace(
                transcript,
                turns=tuple(replace(turn, trainable=False) for turn in transcript.turns),
                sealed=False,
            )
            await self._hand_over(unsealed)
        waited.propagate()
        return Sealed(evidence, transcript, versions)

    async def keep_unsealed(self) -> None:
        """Hand the run the transcript of an attempt revoked before it sealed, as far as
        it got; a no-op for a sealed attempt, whose seal kept it either way, and after
        the first call. Raises only a cancellation, once the write has settled: the attempt
        has already failed."""
        if self._state is LeaseState.REVOKED and self._keeps_unsealed():
            await self._keep_unsealed(*self._settle())

    def _settle(self) -> tuple[set[str], list[Sample]]:
        """What the agent discarded, and each generation sampled but never delivered,
        which this drops from the evidence."""
        discarded = {key for key, request in self._requests.items() if request.discarded}
        dropped: list[Sample] = []
        for request in self._requests.values():
            if request.state is RequestState.DELIVERED or request.discarded:
                continue
            if request.sample is not None:
                request.discarded = True
                dropped.append(request.sample)
        return discarded, dropped

    def _keeps_unsealed(self) -> bool:
        return (
            self._transcript is not None
            and self._wiring.on_transcript is not None
            and not self._unsealed_kept
        )

    async def _keep_unsealed(self, discarded: set[str], dropped: list[Sample]) -> None:
        if self._transcript is None or not self._keeps_unsealed():
            return
        self._unsealed_kept = True
        try:
            transcript = seal_transcript(
                self._transcript.recorded(),
                evidence=self._live_samples(),
                discarded=discarded,
                undelivered={sample.sample_id for sample in dropped},
                sealed=False,
            )
        except ValueError as error:
            logger.warning(
                "Attempt %s: its transcript is incomplete and is not kept: %s",
                self.identity.attempt_id,
                error,
            )
            return
        (await self._hand_over(transcript)).propagate()

    async def _hand_over(self, transcript: SealedTranscript) -> Waited:
        """Hand ``transcript`` to the run if it keeps transcripts, waiting out a
        cancellation of the caller, which the caller raises with ``propagate()``."""
        if self._wiring.on_transcript is None:
            return Waited(settled=True, interrupted=False)
        writing = asyncio.ensure_future(self._wiring.on_transcript(self.identity, transcript))
        waited = await outlast(writing)
        if writing.cancelled() or writing.exception() is not None:
            logger.warning(
                "Attempt %s: its transcript was not written",
                self.identity.attempt_id,
                exc_info=None if writing.cancelled() else writing.exception(),
            )
        return waited

    def _report_undelivered(self, dropped: list[Sample]) -> None:
        for sample in dropped:
            self._wiring.on_undelivered(self.identity, sample)

    def revoke(self) -> None:
        """End model access without evidence; idempotent, and a no-op once sealed."""
        if self._state is LeaseState.OPEN:
            self._close(LeaseState.REVOKED)


class _Channel:
    def __init__(self, lease: AttemptLease, producer: str) -> None:
        self._lease = lease
        self._producer = producer

    @property
    def producer(self) -> str:
        return self._producer

    @property
    def sampling_policy(self) -> SamplingPolicy:
        return self._lease.sampling_policy

    @property
    def samples(self) -> tuple[Sample, ...]:
        return self._lease.captured()

    @property
    def transcript(self) -> TranscriptRecorder | None:
        return self._lease.transcript

    async def generate(
        self,
        prompt: RenderedPrompt,
        sampling_params: SamplingParams,
        *,
        conversation_id: str,
        clamped_from: int | None = None,
        on_failed_reply: FailedReply | None = None,
    ) -> Sample:
        return await self._lease.generate(
            self._producer,
            prompt,
            sampling_params,
            conversation_id=conversation_id,
            clamped_from=clamped_from,
            on_failed_reply=on_failed_reply,
        )

    def discard(self, sample_id: str) -> None:
        self._lease.discard(self._producer, sample_id)

    def reply_failed(self, sample_id: str, failure: RendererFailure) -> None:
        self._lease.reply_failed(sample_id, failure)


class DirectModelAccess:
    """``ModelAccess`` for ``model_access = "direct"``: producer-locked ``LoopsLLM``s."""

    def __init__(self, lease: AttemptLease, renderers: Renderers) -> None:
        self._lease = lease
        self._renderers = renderers
        self._clients: dict[str, LoopsLLM] = {}

    @property
    def mode(self) -> ModelAccessMode:
        return ModelAccessMode.DIRECT

    async def http_connection(self, producer: str | None = None) -> ModelConnection:
        raise ModelAccessError(
            "the runtime declares direct model access, which llm() serves; "
            "http_connection() serves only a runtime that declares openai_http"
        )

    async def quiesce(self, grace_seconds: float) -> None:
        """Nothing to settle: a direct generation is delivered when it returns."""

    def llm(self, producer: str | None = None) -> LoopsLLM:
        name = self._lease.producer(producer)
        client = self._clients.get(name)
        if client is None:
            client = self._clients[name] = LoopsLLM(_Channel(self._lease, name), self._renderers)
        return client
