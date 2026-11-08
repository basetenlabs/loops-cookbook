"""Harbor agents for the Harbor rollout runtime tests, imported by Harbor through import_path."""

import asyncio
import contextlib
import shlex

from harbor.agents.base import BaseAgent
from harbor.agents.installed.base import ApiError

from loops_cookbook.harbor import LoopsAgent
from loops_cookbook.model_access import ModelAccessMode, ModelTransportFailure

RUNS: list[str] = []


class AnswerAgent(LoopsAgent):
    """One model call, then the answer the verifier grades, and a report about itself."""

    model_access = ModelAccessMode.DIRECT

    @staticmethod
    def name() -> str:
        return "test-answer"

    def version(self) -> str:
        return "1"

    async def setup(self, environment) -> None:
        pass

    async def run(self, instruction, environment, context) -> None:
        RUNS.append(instruction)
        response = (
            await self.llm()
            .conversation()
            .generate(messages=[{"role": "user", "content": instruction}])
        )
        context.metadata = {"turns": 1, "said": response.content}
        await environment.exec(
            "mkdir -p /app && printf %s " + shlex.quote("B") + " > /app/answer.txt"
        )


class FlakyAgent(AnswerAgent):
    """Fails the first time it runs; Harbor's own retry would run it again."""

    async def run(self, instruction, environment, context) -> None:
        RUNS.append("flaky")
        await self.llm().conversation().generate(messages=[{"role": "user", "content": "x"}])
        raise RuntimeError("agent crashed")


class AnswerThenHangAgent(AnswerAgent):
    """Answers, then makes a second model call that its deadline cuts mid-flight."""

    async def run(self, instruction, environment, context) -> None:
        await super().run(instruction, environment, context)
        await self.llm().conversation().generate(messages=[{"role": "user", "content": "more"}])


class SleepyAgent(AnswerAgent):
    """Generates once, then outlives its Harbor agent deadline between generations."""

    async def run(self, instruction, environment, context) -> None:
        await super().run(instruction, environment, context)
        await asyncio.sleep(30)


class UnboundAgent(BaseAgent):
    """A Harbor agent that never learned the Loops binding."""

    @staticmethod
    def name() -> str:
        return "test-unbound"

    def version(self) -> str:
        return "1"

    async def setup(self, environment) -> None:
        pass

    async def run(self, instruction, environment, context) -> None:
        pass


class UndeclaredAgent(UnboundAgent):
    """Has the binding setter, and declares no model access."""

    def bind_loops_services(self, services) -> None:
        pass


class AsynchronousAgent(UnboundAgent):
    """Declares its model access, with a setter Harbor would never await."""

    model_access = ModelAccessMode.DIRECT

    async def bind_loops_services(self, services) -> None:
        pass


class OldNameAgent(LoopsAgent):
    """Declares its model access under the attribute ``model_access`` replaced."""

    loops_model_access = ModelAccessMode.OPENAI_HTTP

    @staticmethod
    def name() -> str:
        return "test-old-name"

    def version(self) -> str:
        return "1"

    async def setup(self, environment) -> None:
        pass

    async def run(self, instruction, environment, context) -> None:
        pass


class AgentSafetyRefusalError(RuntimeError):
    """Named like Harbor's own refusal error, which its default retry excludes."""


class RefusingAgent(AnswerAgent):
    async def run(self, instruction, environment, context) -> None:
        RUNS.append("refused")
        await self.llm().conversation().generate(messages=[{"role": "user", "content": "x"}])
        raise AgentSafetyRefusalError("refused")


class HttpAgent(AnswerAgent):
    from loops_cookbook.model_access.api import ModelAccessMode as _Mode

    model_access = _Mode.OPENAI_HTTP


class CatchingAgent(AnswerAgent):
    """Swallows a failed generation, then crashes its trial with an unrelated error."""

    async def run(self, instruction, environment, context) -> None:
        RUNS.append("caught")
        with contextlib.suppress(ModelTransportFailure):
            await self.llm().conversation().generate(messages=[{"role": "user", "content": "x"}])
        raise RuntimeError("agent crashed after a sampler error")


class ExitingAgent(AnswerAgent):
    """Answers, then exits non-zero; Harbor records it and still verifies."""

    async def run(self, instruction, environment, context) -> None:
        from harbor.agents.installed.base import NonZeroAgentExitCodeError

        await super().run(instruction, environment, context)
        raise NonZeroAgentExitCodeError("the agent exited with status 1")


class ApiErrorAgent(AnswerAgent):
    """Answers, then fails as Harbor's installed agents do: a NonZeroAgentExitCodeError subclass."""

    async def run(self, instruction, environment, context) -> None:
        from harbor.agents.installed.base import ApiRateLimitError

        await super().run(instruction, environment, context)
        raise ApiRateLimitError("the provider rate-limited the agent")


class LostSandboxAgent(AnswerAgent):
    """Its sandbox is deleted under it, as the Blaxel environment reports."""

    async def run(self, instruction, environment, context) -> None:
        from loops_cookbook.harbor.blaxel import SandboxLost

        RUNS.append("lost")
        await super().run(instruction, environment, context)
        raise SandboxLost("a command's process vanished while it ran: the sandbox is gone")


class LocalApiError(ApiError):
    """An ApiError subclass Harbor never imports: it exists only once this module loads."""


class LocalApiErrorAgent(AnswerAgent):
    """Answers, then fails with an API error defined outside Harbor; Harbor still verifies."""

    async def run(self, instruction, environment, context) -> None:
        await super().run(instruction, environment, context)
        raise LocalApiError("the provider failed the agent")


class NonZeroAgentExitCodeError(RuntimeError):
    """Named like Harbor's non-zero agent exit, but not one: Harbor aborts the trial on it."""


class ImpostorExitAgent(AnswerAgent):
    async def run(self, instruction, environment, context) -> None:
        await super().run(instruction, environment, context)
        raise NonZeroAgentExitCodeError("not Harbor's exit error")
