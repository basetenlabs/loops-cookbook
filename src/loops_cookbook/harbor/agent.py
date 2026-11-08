"""An optional base class for Harbor agents that connects them to a Loops attempt."""

from typing import ClassVar

from harbor.agents.base import BaseAgent

from loops_cookbook.llm.client import Conversation, LoopsLLM
from loops_cookbook.model_access.api import ModelAccessMode, ModelConnection
from loops_cookbook.runtime.api import AttemptServices


class LoopsAgent(BaseAgent):
    """A Harbor agent that samples the model through the current Loops attempt.

    Subclass it and implement Harbor's agent lifecycle as usual. The Harbor rollout
    runtime binds the attempt's services before ``setup``, so open conversations from
    ``setup`` onward. The subclass owns the history, the tools and the agent loop.

    ``llm(producer)`` returns a client locked to that producer, so an agent with
    several producers calls it once for each. ``self.services`` is the attempt's full
    ``AttemptServices``. Use it to reach the model over HTTP or to run an agent program
    through ``processes``.
    """

    # This has no default. Each agent states how it reaches the model, and `loops check`
    # refuses an agent that does not.
    model_access: ClassVar[ModelAccessMode]
    _loops_services: AttemptServices | None = None

    def bind_loops_services(self, services: AttemptServices) -> None:
        """Keep the attempt's services, which the Harbor rollout runtime binds before ``setup``."""
        self._loops_services = services

    @property
    def services(self) -> AttemptServices:
        """Return the attempt's services, such as model access and processes."""
        if self._loops_services is None:
            raise RuntimeError("Loops services are not bound. Run this agent through `loops run`.")
        return self._loops_services

    def llm(self, producer: str | None = None) -> LoopsLLM:
        """Return the direct client that samples as ``producer``, or as the default producer."""
        return self.services.model_access.llm(producer)

    def conversation(
        self, *, producer: str | None = None, conversation_id: str | None = None
    ) -> Conversation:
        """Start a conversation that samples as ``producer`` for the life of this attempt."""
        return self.llm(producer).conversation(conversation_id=conversation_id)

    async def http_connection(self, producer: str | None = None) -> ModelConnection:
        """Return this attempt's OpenAI-compatible endpoint and its revocable credential."""
        return await self.services.model_access.http_connection(producer)
