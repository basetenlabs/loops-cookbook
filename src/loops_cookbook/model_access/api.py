"""The typed boundary between a runtime and the model it trains."""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple, Protocol

from pydantic import SecretStr

from loops_cookbook.llm.client import LoopsLLM


class ModelAccessMode(str, Enum):
    DIRECT = "direct"
    OPENAI_HTTP = "openai_http"


class ModelConnection(NamedTuple):
    """An attempt's OpenAI-compatible endpoint: base URL, bearer credential, and model."""

    base_url: str
    api_key: SecretStr
    model: str


class ModelAccess(Protocol):
    @property
    def mode(self) -> ModelAccessMode: ...
    def llm(self, producer: str | None = None) -> LoopsLLM: ...
    async def http_connection(self, producer: str | None = None) -> ModelConnection: ...


class AttemptModelAccess(ModelAccess, Protocol):
    """Model access as the driver holds it: it can settle before the lease seals."""

    async def quiesce(self, grace_seconds: float) -> None:
        """Stop admitting, then wait up to ``grace_seconds`` for in-flight work, then cancel it."""
        ...


class ModelAccessError(RuntimeError):
    """A runtime used model access outside its contract: the same call fails every time."""


class UndeclaredProducer(ModelAccessError):
    """A call named a producer ``[rollout.producers]`` does not declare: the
    configuration is wrong for every attempt, so the run stops."""


class CaptureIntegrityFailure(RuntimeError):
    """The attempt's evidence cannot be trusted: a generation was lost."""


class ModelTransportFailure(RuntimeError):
    """The Loops sampling request itself failed."""


class SamplingFailure(RuntimeError):
    """The sampler answered, but not with evidence the cookbook can capture."""


class FuturePolicyVersion(RuntimeError):
    """The sampler reported a policy the trainer has not published."""
