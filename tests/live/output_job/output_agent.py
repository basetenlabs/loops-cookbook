"""An agent that reads a long command output through Harbor, and says whether it was whole.

It asks the model for a label, so the rollout has a generation to train, then runs a
command whose 200 KB of output it can predict, and writes ``intact`` or ``cut``.
"""

import hashlib
import shlex

from harbor.agents.options import AgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from loops_cookbook.harbor import LoopsAgent
from loops_cookbook.model_access import ModelAccessMode

LINES = 10_000  # 200 KB of stdout, three times Blaxel's 64 KB process record
SCRIPT = f"import sys\nfor i in range({LINES}):\n    sys.stdout.write(f'line {{i:015d}}\\n')\n"
EXPECTED = hashlib.sha256("".join(f"line {i:015d}\n" for i in range(LINES)).encode()).hexdigest()


class OutputOptions(AgentOptions):
    """Nothing to configure."""


class OutputAgent(LoopsAgent):
    model_access = ModelAccessMode.DIRECT
    options_model = OutputOptions
    options: OutputOptions

    @staticmethod
    def name() -> str:
        return "live-output"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        response = (
            await self.llm()
            .conversation()
            .generate(
                messages=[{"role": "user", "content": instruction}],
            )
        )
        printed = await environment.exec(f"python3 -c {shlex.quote(SCRIPT)}")
        whole = hashlib.sha256(printed.stdout.encode()).hexdigest() == EXPECTED
        answer = (response.content or "").strip()
        result = await environment.exec(
            "mkdir -p /app && printf %s "
            + shlex.quote(answer)
            + " > /app/answer.txt && printf %s "
            + ("intact" if whole else "cut")
            + " > /app/output.txt"
        )
        if result.return_code != 0:
            raise RuntimeError("Could not persist the agent's answer")
        context.metadata = {**(context.metadata or {}), "output_bytes": len(printed.stdout)}
