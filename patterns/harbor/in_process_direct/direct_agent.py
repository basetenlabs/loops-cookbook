"""An in-process Harbor agent with direct model access: one ``llm()`` call.

The runtime calls this class inside the training process. It asks the model through
the cookbook's model client, which records the exact tokens it samples, then writes
its answer into the trial's sandbox for the task's verifier.
"""

import shlex

from harbor.agents.options import AgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from loops_cookbook.harbor import LoopsAgent
from loops_cookbook.model_access import ModelAccessMode


class DirectOptions(AgentOptions):
    """Settings supplied through ``agents[].kwargs`` in the Harbor job."""

    # Nothing yet: temperature and generation length are the run's decisions and
    # arrive from `[sampling]` in the training config.


class DirectAgent(LoopsAgent):
    model_access = ModelAccessMode.DIRECT
    options_model = DirectOptions
    options: DirectOptions

    @staticmethod
    def name() -> str:
        return "hill-climb-direct"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        conversation = self.conversation()
        response = await conversation.generate(
            messages=[{"role": "user", "content": instruction}],
        )
        answer = (response.content or "").strip()
        result = await environment.exec(
            "mkdir -p /app && printf %s " + shlex.quote(answer) + " > /app/answer.txt"
        )
        if result.return_code != 0:
            raise RuntimeError("could not write the answer into the sandbox")
