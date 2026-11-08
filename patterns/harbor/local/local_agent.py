"""A local Harbor agent, whose loop is a separate program the cookbook does not own.

``LocalAgent`` runs ``command``, which is ``agents[].kwargs.command`` in the Harbor job,
as a process on the training machine, through the attempt's ``processes`` service. The
program gets two things in its environment: the attempt's OpenAI-compatible model
connection, served by the run's loopback gateway, and the trial environment's
``usage_environment()``, so it runs its tools in the trial's sandbox through the
provider's own API. The cookbook's Blaxel environment provides that usage environment.
The program never creates or deletes the sandbox.

Samples are attributed to the attempt through its credential alone, and the
credential is revoked when the rollout is sealed.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.options import AgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pydantic import Field, SecretStr

from loops_cookbook.harbor import LoopsAgent
from loops_cookbook.model_access import ModelAccessMode
from loops_cookbook.runtime import ProcessCommand

HERE = Path(__file__).resolve().parent


@runtime_checkable
class UsageEnvironment(Protocol):
    """A trial environment a separate process can reach, as the cookbook's Blaxel one is."""

    def usage_environment(self) -> Mapping[str, str | SecretStr]: ...


class LocalAgentOptions(AgentOptions):
    """The ``agents[].kwargs`` that give the program's argv. ``./`` paths resolve beside
    this file."""

    command: list[str] = Field(default_factory=lambda: ["python3", "./agent_program.py"])


class LocalAgent(LoopsAgent):
    model_access = ModelAccessMode.OPENAI_HTTP
    options_model = LocalAgentOptions
    options: LocalAgentOptions

    @staticmethod
    def name() -> str:
        return "local-agent"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        if not isinstance(environment, UsageEnvironment):
            raise TypeError(
                "LocalAgent runs its tools through the trial sandbox's usage environment; "
                "use loops_cookbook.harbor.blaxel:BlaxelEnvironment"
            )
        services = self.services
        model = await self.http_connection()
        result = await services.processes.run(
            ProcessCommand(
                argv=tuple(
                    str((HERE / part).resolve()) if part.startswith(("./", "../")) else part
                    for part in self.options.command
                ),
                cwd=self.logs_dir,
                env={
                    **environment.usage_environment(),
                    "OPENAI_BASE_URL": model.base_url,
                    "OPENAI_API_KEY": model.api_key,
                    "OPENAI_MODEL": model.model,
                    "AGENT_INSTRUCTION": instruction,
                },
            )
        )
        # The tails are redacted, so they are safe to keep beside the trial.
        (self.logs_dir / "process.log").write_text(result.stdout_tail + result.stderr_tail)
        ended = (
            f"signal_{result.signal}" if result.signal is not None else f"status_{result.exit_code}"
        )
        context.metadata = {"agent_exit": ended, "seconds": round(result.elapsed_seconds, 3)}
        if result.exit_code != 0 or result.signal is not None:
            raise NonZeroAgentExitCodeError(
                f"the agent program exited with "
                f"{f'signal {result.signal}' if result.signal is not None else result.exit_code}"
            )
