"""The typed boundary between a rollout runtime and the model it trains."""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple, Protocol

from pydantic import SecretStr

from loops_cookbook.llm.client import LoopsLLM


class ModelAccessMode(str, Enum):
    """How an agent reaches the model, which a rollout runtime declares.

    ``DIRECT`` is the cookbook's ``llm()`` client, which only an in-process agent can
    use. ``OPENAI_HTTP`` is an OpenAI-compatible endpoint that the run serves.
    """

    DIRECT = "direct"
    OPENAI_HTTP = "openai_http"


class ModelConnection(NamedTuple):
    """An attempt's OpenAI-compatible endpoint, with its base URL, bearer credential and model."""

    base_url: str
    api_key: SecretStr
    model: str


class ModelAccess(Protocol):
    """One attempt's way to the model, which a rollout runtime takes from ``AttemptServices``.

    Only the method that matches ``mode`` serves. The other raises
    ``ModelAccessError``. A producer that ``[rollout.producers]`` does not declare
    raises ``UndeclaredProducer`` and stops the run.
    """

    @property
    def mode(self) -> ModelAccessMode:
        """The attempt's model access mode."""
        ...

    def llm(self, producer: str | None = None) -> LoopsLLM:
        """Return the direct client that samples as ``producer``, or as the default producer."""
        ...

    async def http_connection(self, producer: str | None = None) -> ModelConnection:
        """Return the attempt's OpenAI-compatible endpoint and a credential for ``producer``.

        The credential is revoked when the attempt seals, aborts or times out.
        """
        ...


class AttemptModelAccess(ModelAccess, Protocol):
    """Model access as the driver holds it, which can settle before the lease seals."""

    async def quiesce(self, grace_seconds: float) -> None:
        """Stop admitting, then wait up to ``grace_seconds`` for in-flight work, then cancel it."""
        ...


class ModelAccessError(RuntimeError):
    """Raised when a rollout runtime uses model access outside its contract.

    The same call fails every time.
    """


class UndeclaredProducer(ModelAccessError):
    """Raised when a call names a producer that ``[rollout.producers]`` does not declare.

    The configuration is wrong for every attempt, so the run stops.
    """


class CaptureIntegrityFailure(RuntimeError):
    """Raised when a generation was lost, so the attempt's evidence cannot be trusted."""


class ModelTransportFailure(RuntimeError):
    """Raised when the Loops sampling request itself fails."""


class SamplingFailure(RuntimeError):
    """Raised when the sampler answers, but not with evidence the cookbook can capture."""


class FuturePolicyVersion(RuntimeError):
    """Raised when the sampler reports a policy version that the trainer has not published."""
