"""A minimal terminal agent that the user owns. Harbor owns environments and grading."""

import asyncio
import json

from harbor.agents.options import AgentOptions
from pydantic import Field, create_model

from loops_cookbook.harbor import LoopsAgent
from loops_cookbook.llm import ContextWindowExceeded, ToolMessage, ToolSchema
from loops_cookbook.model_access import ModelAccessMode

SYSTEM_PROMPT = (
    "You are a skilled software engineer working in a sandboxed environment. "
    "You have access to a bash tool to execute commands. "
    "Complete the task described by the user."
)
BASH_TOOL: ToolSchema = {
    "name": "bash",
    "description": "Execute a bash command in the sandbox environment.\n\n"
    "Use this to run shell commands, install packages, edit files, etc.\n",
    "parameters": {
        "properties": {
            "command": {
                "description": "The bash command to execute.",
                "title": "Command",
                "type": "string",
            }
        },
        "required": ["command"],
        "title": "bash_params",
        "type": "object",
    },
}

BashParams = create_model(
    "bash_params", command=(str, Field(description="The bash command to execute."))
)
# A turn that comes back cut is re-sampled rather than abandoning the whole rollout.
# Bounded, because a task whose every generation runs long would otherwise spin here
# until the agent deadline expires and take the rollout with it anyway.
TURN_ATTEMPTS = 3


class TerminalOptions(AgentOptions):
    """This agent's own machinery. Temperature, generation length and the trajectory
    budget are the run's decisions and arrive from ``[sampling]`` in the training
    config, so an agent is not configured with the experiment."""

    max_turns: int = Field(ge=1)
    command_timeout: int = Field(gt=0)
    max_output_bytes: int = Field(ge=1)


class TerminalAgent(LoopsAgent):
    model_access = ModelAccessMode.DIRECT
    options_model = TerminalOptions
    options: TerminalOptions

    @staticmethod
    def name():
        return "terminal-loops"

    def version(self):
        return "1"

    async def setup(self, environment):
        pass

    async def _sample_turn(self, conversation, messages):
        """One turn, re-sampled while the generation keeps coming back cut.

        Returns the reply and the generations it cost, or ``None`` once every attempt
        was cut short. A reply the trajectory budget cut is returned as it is: every
        retry would get the same room.
        """
        for attempt in range(TURN_ATTEMPTS):
            reply = await conversation.generate(
                messages=messages,
                tools=[BASH_TOOL],
            )
            if not reply.truncated or reply.sample.clamped_from is not None:
                return reply, attempt + 1
            # Left in place a cut generation is trained anyway, and it forks the
            # chain this retry is about to extend.
            conversation.discard(reply)
        return None, TURN_ATTEMPTS

    async def run(self, instruction, environment, context):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        conversation = self.llm().conversation()
        status = "max_turns"
        generations = 0

        async def execute(call) -> ToolMessage:
            def error(message) -> ToolMessage:
                # The error is capped like any other tool output. An exception's text is
                # not ours to bound, and a gateway that answers with an HTML error page
                # would otherwise spend the model's context explaining itself.
                return {
                    "role": "tool",
                    "tool_call_id": call.id or "",
                    "name": call.name,
                    "content": json.dumps(
                        {
                            "error": message.encode("utf-8")[
                                : self.options.max_output_bytes
                            ].decode("utf-8", errors="ignore")
                        }
                    ),
                }

            if call.name != "bash":
                return error(f"Tool '{call.name}' not found")
            try:
                arguments = BashParams.model_validate(call.arguments)
            except ValueError as exc:
                return error(f"Parameter validation failed: {exc}")
            try:
                result = await environment.exec(
                    arguments.command, cwd="/", timeout_sec=self.options.command_timeout
                )
            except Exception as exc:  # noqa: BLE001 - any sandbox failure is the
                # model's problem to react to, not the rollout's to die from.
                # The error is handled, so no stack is logged. The model is told the
                # command failed and decides what to do next. The traceback is the same
                # frames into the sandbox client every time, and emitting it at ERROR
                # buries the run's real failures whenever the platform is flaky.
                # Report the type, not just the message, because a bare ConnectError or
                # timeout from the sandbox client stringifies to nothing at all.
                detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                self.logger.warning("Bash tool execution failed: %s", detail)
                return error(f"Tool execution failed: {detail}")
            return {
                "role": "tool",
                "tool_call_id": call.id or "",
                "name": "bash",
                "content": json.dumps(
                    {
                        "exit_code": result.return_code,
                        "stdout": (result.stdout or "")
                        .encode("utf-8")[: self.options.max_output_bytes]
                        .decode("utf-8", errors="ignore"),
                        "stderr": (result.stderr or "")
                        .encode("utf-8")[: self.options.max_output_bytes]
                        .decode("utf-8", errors="ignore"),
                    }
                ),
            }

        for _ in range(self.options.max_turns):
            try:
                response, attempts = await self._sample_turn(conversation, messages)
            except ContextWindowExceeded:
                status = "context_overflow"
                break
            generations += attempts
            # Every attempt at this turn was cut. There is no reply to append and
            # nothing the verifier should grade as an answer.
            if response is None:
                status = "truncated"
                break
            # The trajectory budget cut this reply. It is kept, so it trains, and the
            # episode is over.
            if response.truncated:
                status = "context_overflow"
                break
            if response.native_message is None:
                # There is no assistant turn to append. Either the completion did not
                # parse, or a tool call did not and the message cannot represent the
                # attempt.
                # Re-sampling might land a clean one, and telling the model what
                # `tool_calls[i].status` found would be more informative still, but
                # both need replacement history. This example stops instead.
                status = "unusable_reply"
                break
            messages.append(response.native_message)
            if not response.tool_calls:
                status = "completed"
                break
            messages.extend(await asyncio.gather(*(execute(c) for c in response.tool_calls)))
        context.metadata = {
            **(context.metadata or {}),
            "termination_reason": status,
            "generation_count": generations,
        }
