"""Optional Harbor base class providing only Loops attempt binding."""

from typing import ClassVar

from harbor.agents.base import BaseAgent

from loops_cookbook.llm.client import Conversation, LoopsLLM
from loops_cookbook.model_access.api import ModelAccessMode, ModelConnection
from loops_cookbook.runtime.api import AttemptServices


class LoopsAgent(BaseAgent):
    """Implement Harbor's lifecycle; open conversations after the attempt binds.

    ``self.services`` is the attempt's full ``AttemptServices``, for an agent that
    reaches the model over HTTP or runs an agent program through ``processes``.

    The runtime binds the attempt's services before ``setup``. History, tools, and
    the agent loop remain entirely owned by the subclass. ``llm(producer)`` returns
    a client locked to that producer, so a multi-producer agent asks once per
    producer.
    """

    # No default: an agent states how it reaches the model, and check refuses one
    # that does not.
    model_access: ClassVar[ModelAccessMode]
    _loops_services: AttemptServices | None = None

    def bind_loops_services(self, services: AttemptServices) -> None:
        self._loops_services = services

    @property
    def services(self) -> AttemptServices:
        """The attempt's services: model access, processes, and the rest."""
        if self._loops_services is None:
            raise RuntimeError("Loops services are not bound; run this agent through `loops run`")
        return self._loops_services

    def llm(self, producer: str | None = None) -> LoopsLLM:
        return self.services.model_access.llm(producer)

    def conversation(
        self, *, producer: str | None = None, conversation_id: str | None = None
    ) -> Conversation:
        """Start a direct session with this attempt's producer binding and lifetime."""
        return self.llm(producer).conversation(conversation_id=conversation_id)

    async def http_connection(self, producer: str | None = None) -> ModelConnection:
        """Get this attempt's OpenAI-compatible endpoint and revocable credential."""
        return await self.services.model_access.http_connection(producer)
