"""The hill-climb agent with a critic: one policy in two roles, both trained.

``main`` proposes a label. ``critic`` sees the proposal and gives the final answer,
keeping it or changing it. A producer is a label on a generation, not a model: both
roles sample the same Loops model and LoRA, at the same policy version. The verifier
scores only the final answer, and both producers are declared trainable in
``[rollout.producers]``, so both generations train the same weights on that one
reward: to propose the target, and to keep or fix a proposal.
"""

import shlex

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from loops_cookbook.harbor import LoopsAgent
from loops_cookbook.model_access import ModelAccessMode

REVIEW = (
    "Another model was asked the question below and proposed the label {proposal}.\n"
    "Give the final answer: reply with the single label only, keeping the proposal or "
    "replacing it.\n\n{instruction}"
)


class CriticLabelAgent(LoopsAgent):
    model_access = ModelAccessMode.DIRECT

    @staticmethod
    def name() -> str:
        return "hill-climb-critic"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        proposal = (
            await self.llm()
            .conversation()
            .generate(messages=[{"role": "user", "content": instruction}])
        )
        final = (
            await self.llm("critic")
            .conversation()
            .generate(
                messages=[
                    {
                        "role": "user",
                        "content": REVIEW.format(
                            proposal=(proposal.content or "").strip(), instruction=instruction
                        ),
                    }
                ]
            )
        )
        answer = (final.content or "").strip()
        context.metadata = {"proposal": (proposal.content or "").strip(), "final": answer}
        result = await environment.exec(
            "mkdir -p /app && printf %s " + shlex.quote(answer) + " > /app/answer.txt"
        )
        if result.return_code != 0:
            raise RuntimeError("could not persist the answer")
