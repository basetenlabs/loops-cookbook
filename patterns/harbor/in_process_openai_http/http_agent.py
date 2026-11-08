"""An in-process Harbor agent with openai_http model access, through an OpenAI client.

The rollout runtime calls this class inside the training process, as in the direct
pattern. But the agent reaches the model the way an agent built on any OpenAI client
would, through the attempt's OpenAI-compatible endpoint, which the run's gateway serves.
The gateway records the exact tokens behind every reply. The connection's credential is
the attempt's, so every sample is attributed to it, and it stops working when the
rollout is sealed."""

import shlex

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from openai import AsyncOpenAI

from loops_cookbook.harbor import LoopsAgent
from loops_cookbook.model_access import ModelAccessMode


class HttpAgent(LoopsAgent):
    model_access = ModelAccessMode.OPENAI_HTTP

    @staticmethod
    def name() -> str:
        return "hill-climb-http"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        model = await self.http_connection()
        async with AsyncOpenAI(
            base_url=model.base_url, api_key=model.api_key.get_secret_value()
        ) as client:
            completion = await client.chat.completions.create(
                model=model.model, messages=[{"role": "user", "content": instruction}]
            )
        answer = (completion.choices[0].message.content or "").strip()
        result = await environment.exec(
            "mkdir -p /app && printf %s " + shlex.quote(answer) + " > /app/answer.txt"
        )
        if result.return_code != 0:
            raise RuntimeError("could not write the answer into the sandbox")
