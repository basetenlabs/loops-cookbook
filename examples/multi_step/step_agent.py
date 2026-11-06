"""The hill-climb agent from patterns/harbor/in_process_direct, on a multi-step task.

The agent is unchanged: Harbor runs it once per step, with that step's instruction,
in the same sandbox, and verifies each step there.
"""

import shlex

from harbor.agents.options import AgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from loops_cookbook.harbor import LoopsAgent
from loops_cookbook.model_access import ModelAccessMode


class StepOptions(AgentOptions):
    """Settings supplied through ``agents[].kwargs`` in the Harbor job."""

    # Nothing yet: temperature and generation length are the run's decisions and
    # arrive from `[sampling]` in the training config.


class StepAgent(LoopsAgent):
    model_access = ModelAccessMode.DIRECT
    options_model = StepOptions
    options: StepOptions

    @staticmethod
    def name() -> str:
        return "hill-climb-steps"

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
