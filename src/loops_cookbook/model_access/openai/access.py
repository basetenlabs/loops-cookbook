"""One attempt's OpenAI-compatible model access: credentials, conversations, and delivery.

The run's gateway only authenticates and transports. Everything scoped to an attempt
lives here: the producer credentials it minted, the HTTP conversations and what they
returned, the ``Idempotency-Key`` ledger, and the sampling and handler tasks it owns.
All of it ends when the attempt's lease stops admitting.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from baseten.loops.models import SamplingParams
from pydantic import BaseModel, SecretStr

from loops_cookbook.cancellation import outlast, stop
from loops_cookbook.evidence import Sample
from loops_cookbook.llm.client import (
    ContextWindowExceeded,
    Conversation,
    ConversationBusy,
    LLMResponse,
    LoopsLLM,
    Renderers,
    SamplingPolicy,
    UnsupportedParameterError,
)
from loops_cookbook.llm.rendering import RenderedPrompt, RendererFailure, RenderingError
from loops_cookbook.llm.transcript import PendingTurn, TranscriptRecorder
from loops_cookbook.model_access.api import (
    ModelAccessError,
    ModelAccessMode,
    ModelConnection,
)
from loops_cookbook.model_access.lease import AttemptLease, FailedReply
from loops_cookbook.model_access.openai.policy import ChatRequest, InvalidRequestError
from loops_cookbook.model_access.openai.translate import (
    COMPLETION_ID_PREFIX,
    Completion,
    InvalidOutput,
    Returned,
    completion,
)
from loops_cookbook.model_access.openai.translate import renderer_input as translate_input


class IdempotencyConflict(ValueError):
    """An ``Idempotency-Key`` was reused with a different request."""


class AccessEnded(ModelAccessError):
    """The attempt sealed, aborted, or timed out; its credentials no longer admit."""


class UnknownGeneration(ModelAccessError):
    """No generation this credential's attempt and producer returned has that ID."""


# Rejections that leave nothing sampled: a retry under the same key is a new try.
_BEFORE_SAMPLING = (
    ConversationBusy,
    ContextWindowExceeded,
    UnsupportedParameterError,
    InvalidRequestError,
    RenderingError,
)


class GatewayPort(Protocol):
    """What an attempt needs from the run's gateway."""

    @property
    def model(self) -> str: ...

    async def base_url(self) -> str: ...

    def register(self, credential: str, access: HttpModelAccess, producer: str) -> None: ...

    def revoke(self, credential: str) -> None: ...


class _HttpChannel:
    """A producer's capture path for requests whose delivery the transport settles."""

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
        """Never here: the conversation sees translated messages, not the agent's."""
        return None

    async def generate(
        self,
        prompt: RenderedPrompt,
        sampling_params: SamplingParams,
        *,
        conversation_id: str,
        clamped_from: int | None = None,
        on_failed_reply: FailedReply | None = None,
    ) -> Sample:
        return await self._lease.generate_for_delivery(
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


@dataclass(slots=True)
class _Conversation:
    conversation: Conversation
    # The agent named it with the conversation header; otherwise it is one request.
    named: bool
    returned: Returned = field(default_factory=Returned)


@dataclass(slots=True)
class _Keyed:
    """One ``Idempotency-Key``: the request it names and the one generation serving it."""

    digest: str
    task: asyncio.Task[Completion | InvalidOutput]


def request_digest(request: BaseModel, conversation_id: str | None) -> str:
    """The canonical identity of a request, for ``Idempotency-Key`` comparison.

    Taken from the validated request, so two spellings of the same value, such as
    ``1`` and ``1.0``, are the same request, and an explicit ``null`` is the same as
    an omitted field.
    """
    body = request.model_dump(mode="json", exclude_unset=True, exclude_none=True)
    canonical = json.dumps(
        {"body": body, "conversation_id": conversation_id}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class HttpModelAccess:
    """``ModelAccess`` for ``model_access = "openai_http"``: attempt credentials on the gateway."""

    def __init__(self, lease: AttemptLease, renderers: Renderers, gateway: GatewayPort) -> None:
        self._lease = lease
        self._renderers = renderers
        self._gateway = gateway
        self._credentials: dict[str, str] = {}
        self._clients: dict[str, LoopsLLM] = {}
        self._conversations: dict[tuple[str, str], _Conversation] = {}
        self._keys: dict[tuple[str, str], _Keyed] = {}
        # Every generation returned, by producer and sample ID, for ``discard``.
        self._replies: dict[tuple[str, str], tuple[_Conversation, LLMResponse]] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._open = True
        lease.on_close(self._close)

    @property
    def mode(self) -> ModelAccessMode:
        return ModelAccessMode.OPENAI_HTTP

    def llm(self, producer: str | None = None) -> LoopsLLM:
        raise ModelAccessError(
            "the runtime declares openai_http model access, which http_connection() "
            "serves; llm() serves only a runtime that declares direct"
        )

    async def http_connection(self, producer: str | None = None) -> ModelConnection:
        """The endpoint and a credential bound to this attempt and producer.

        Binding the run's shared listener is the only I/O; nothing samples until the
        agent sends a request. The credential stays valid until the attempt seals,
        aborts, or times out, with no expiry of its own.
        """
        name = self._lease.producer(producer)
        if not self._open:
            raise AccessEnded("model access for this attempt has ended")
        base_url = await self._gateway.base_url()
        if not self._open:  # revoked while the listener was binding
            raise AccessEnded("model access for this attempt has ended")
        credential = self._credentials.get(name)
        if credential is None:
            credential = f"loops-{uuid.uuid4().hex}{uuid.uuid4().hex}"
            self._credentials[name] = credential
            self._gateway.register(credential, self, name)
        return ModelConnection(base_url, SecretStr(credential), self._gateway.model)

    def own(self, task: asyncio.Task[Any]) -> None:
        """Hold ``task`` until the attempt settles; handlers and generations alike."""
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def complete(
        self,
        producer: str,
        request: ChatRequest,
        *,
        sent: Mapping[str, Any],
        conversation_id: str | None,
        key: str | None,
        digest: str,
    ) -> asyncio.Task[Completion | InvalidOutput]:
        """The generation serving this request, shared with every retry of its key.

        ``sent`` is the request body as the agent sent it, for the transcript. It runs
        as a task the attempt owns, so a client that disconnects mid-generation cannot
        cancel the sample it asked for.
        """
        if not self._open:
            # A handler that authenticated before revocation admits nothing after it.
            raise AccessEnded("model access for this attempt has ended")
        if key is not None:
            keyed = self._keys.get((producer, key))
            if keyed is not None:
                if keyed.digest != digest:
                    raise IdempotencyConflict(
                        "Idempotency-Key was already used with a different request"
                    )
                return keyed.task
        conversation = self._conversation(producer, conversation_id)
        task = asyncio.create_task(self._generate(conversation, request, sent))
        self.own(task)
        if key is not None:
            self._keys[(producer, key)] = _Keyed(digest, task)
            task.add_done_callback(lambda done: self._forget_if_unsampled(producer, key, done))
        return task

    def _forget_if_unsampled(
        self, producer: str, key: str, done: asyncio.Task[Completion | InvalidOutput]
    ) -> None:
        """A key whose request was refused before sampling names nothing yet."""
        if done.cancelled() or not isinstance(done.exception(), _BEFORE_SAMPLING):
            return
        keyed = self._keys.get((producer, key))
        if keyed is not None and keyed.task is done:
            del self._keys[(producer, key)]

    def _conversation(self, producer: str, conversation_id: str | None) -> _Conversation:
        """The conversation a request names, or one of its own when it names none."""
        key = conversation_id or str(uuid.uuid4())
        slot = self._conversations.get((producer, key))
        if slot is None:
            client = self._clients.get(producer)
            if client is None:
                client = LoopsLLM(_HttpChannel(self._lease, producer), self._renderers)
                self._clients[producer] = client
            slot = _Conversation(
                client.conversation(conversation_id=key), conversation_id is not None
            )
            self._conversations[(producer, key)] = slot
        return slot

    async def _generate(
        self, slot: _Conversation, request: ChatRequest, sent: Mapping[str, Any]
    ) -> Completion | InvalidOutput:
        messages, tools = translate_input(request, slot.returned)
        # The transcript keeps the request as the agent sent it: its echoed assistant
        # messages, not the native ones translation substitutes for them, and its keys in
        # its own order, not the schema's.
        recorder = self._lease.transcript
        pending: PendingTurn | None = None

        def begin_turn() -> FailedReply:
            # Where a direct turn begins: a request refused before it leaves no turn.
            nonlocal pending
            assert recorder is not None
            pending = recorder.begin(
                sent["messages"],
                sent.get("tools") or (),
                rendered=messages,
                named=slot.named,
                producer=slot.conversation.producer,
                conversation_id=slot.conversation.conversation_id,
            )
            return pending.failed

        # A reply that raises after sampling is recorded as a direct turn records it.
        reply = await slot.conversation._generate(  # pyright: ignore[reportPrivateUsage]
            _sampling_params(request, self._lease.sampling_policy),
            messages=messages,
            tools=tools,
            # None is the run's trajectory budget; a request cannot name another.
            max_trajectory_tokens=None,
            reasoning_effort=request.reasoning_effort,
            begin_turn=begin_turn if recorder is not None else None,
        )
        done = completion(
            reply,
            model=self._gateway.model,
            created=int(time.time()),
            tools_offered=request.offer_tools and bool(request.tools),
        )
        if isinstance(done, Completion) and reply.native_message is not None:
            slot.returned.remember(done.message, reply.native_message)
        elif isinstance(done, InvalidOutput) and (written := reply.history_message) is not None:
            # Its echo stands for the sampled tokens, which the next turn bridges from.
            slot.returned.remember(done.message, written)
        if pending is not None:
            pending.completion(reply.sample, done.body())
        self._replies[(slot.conversation.producer, reply.response_id)] = (slot, reply)
        return done

    def discard(self, producer: str, completion_id: str) -> None:
        """Exclude a returned completion from training, as ``Conversation.discard`` does.

        The same rules as a direct discard: only a delivered generation, only the most
        recent one of its conversation, and only once. An ID this attempt and producer
        never returned, another attempt's included, is ``UnknownGeneration``.
        """
        if not self._open:
            raise AccessEnded("model access for this attempt has ended")
        found = None
        if completion_id.startswith(COMPLETION_ID_PREFIX):
            found = self._replies.get((producer, completion_id.removeprefix(COMPLETION_ID_PREFIX)))
        if found is None:
            raise UnknownGeneration(f"generation {completion_id!r} does not belong to this client")
        slot, reply = found
        slot.conversation.discard(reply)

    def delivered(self, sample_id: str) -> None:
        self._lease.delivered(sample_id)

    def _close(self) -> None:
        """The lease stopped admitting: revoke every credential and cancel owned work."""
        self._revoke()
        for task in list(self._tasks):
            task.cancel()

    def _revoke(self) -> None:
        self._open = False
        for credential in self._credentials.values():
            self._gateway.revoke(credential)

    async def quiesce(self, grace_seconds: float) -> None:
        """Before sealing: stop admitting, let in-flight work finish, then cancel the rest."""
        self._revoke()
        await _settle(self._tasks, grace_seconds)


def _sampling_params(request: ChatRequest, policy: SamplingPolicy) -> SamplingParams:
    """What the request supplied, and the run's defaults for what it left out.

    The same defaults a direct caller gets; the client joins the renderer's stops to a
    request's ``stop`` where they are the same kind.
    """
    sampling = request.sampling
    params = SamplingParams()
    temperature = sampling.temperature if sampling.temperature is not None else policy.temperature
    if temperature is not None:
        params.temperature = temperature
    max_tokens = sampling.max_tokens or policy.max_generation_tokens
    if max_tokens is not None:
        params.max_tokens = max_tokens
    if sampling.top_p is not None:
        params.top_p = sampling.top_p
    if sampling.seed is not None:
        params.seed = sampling.seed
    if sampling.stop is not None:
        params.stop = list(sampling.stop)
    if sampling.structured_outputs is not None:
        params.structured_outputs = dict(sampling.structured_outputs)
    return params


async def _settle(tasks: set[asyncio.Task[object]], grace_seconds: float) -> None:
    """The grace for ``tasks`` to finish, then a stop of each left, which no cancellation ends."""
    pending = {task for task in tasks if not task.done()}
    try:
        if pending:
            await asyncio.wait(pending, timeout=grace_seconds)
    finally:
        late = (task for task in pending if not task.done())
        stopped = await outlast(asyncio.gather(*(stop(task, grace_seconds) for task in late)))
    stopped.propagate()
