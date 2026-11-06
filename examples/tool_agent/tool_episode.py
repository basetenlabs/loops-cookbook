"""A multi-turn agent with tools, in the attempt's Blaxel sandbox, as an ``Episode``.

``setup()`` writes the task's files into ``/work``; ``run(sample)`` offers the policy a shell
and a file writer until it answers or the agent deadline cuts it; ``grade()`` runs the
task's check, which the agent never sees, on what ``run()`` left: 1 when it passes, else 0.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from loops_cookbook.episodes import Episode, Grading, Sampling, Setup
from loops_cookbook.llm import ContextWindowExceeded, Message, ToolSchema
from loops_cookbook.providers.blaxel import Blaxel
from loops_cookbook.runtime import JsonValue

if TYPE_CHECKING:
    from renderers.base import ParsedToolCall

WORKDIR = "/work"
SYSTEM_PROMPT = (
    f"You are working in a Linux sandbox. The current directory is {WORKDIR}, and it holds "
    "the files of the task. Use the tools to look at the files and change them. When the "
    "task is done, reply with one short sentence and no tool call."
)
# The path and content arrive as environment variables, so the shell never parses them.
WRITE = 'mkdir -p "$(dirname -- "$FILE")" && printf %s "$CONTENT" > "$FILE"'
WAIT_INTERVAL_MS = 200  # how often a wait asks Blaxel whether a command has ended
WAIT_MARGIN_SECONDS = 10  # how long past its timeout a wait allows for Blaxel's kill


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolTask(Strict):
    instruction: str = Field(min_length=1)
    files: dict[str, str]  # path relative to /work -> content
    check: str = Field(min_length=1)  # a shell command in /work; exit 0 when solved


class ToolOptions(Strict):
    max_turns: int = Field(ge=1)
    command_timeout_seconds: int = Field(gt=0)  # Blaxel refuses a float here
    max_output_chars: int = Field(gt=0)  # the model reads the last this many of each stream


class Shell(Strict):
    """Run a command with /bin/sh in /work, and see its output."""

    command: str


class WriteFile(Strict):
    """Write a file, relative to /work, replacing what it held."""

    path: str
    content: str


TOOLS: list[ToolSchema] = [
    {"name": name, "description": args.__doc__ or "", "parameters": args.model_json_schema()}
    for name, args in (("shell", Shell), ("write_file", WriteFile))
]


def ended(reason: str, turns: int, tool_calls: int) -> dict[str, JsonValue]:
    """The episode's metadata: why ``run()`` stopped, after how many turns and tool calls."""
    return {"termination_reason": reason, "turns": turns, "tool_calls": tool_calls}


class ToolEpisode(Episode[ToolTask, ToolOptions]):
    execution_kind = Blaxel
    grading_retries = 2  # the check leaves /work as it found it, so it can run again

    async def setup(self, setup: Setup) -> None:
        sandbox = setup.require_sandbox(Blaxel)
        files = self.task.value.files.items()
        box = await sandbox.client()
        await box.fs.write_tree([{"path": path, "content": text} for path, text in files], WORKDIR)

    async def run(self, sample: Sampling) -> Mapping[str, JsonValue]:
        sandbox = sample.require_sandbox(Blaxel)
        model = sample.model
        conversation = model.conversation()
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.task.value.instruction},
        ]
        calls = 0
        for turn in range(1, self.options.max_turns + 1):
            try:
                reply = await conversation.generate(messages=messages, tools=TOOLS)
            except ContextWindowExceeded:  # the conversation outgrew the trajectory budget
                return ended("context_overflow", turn - 1, calls)
            # A cut reply trains with the episode's reward. To drop one the trajectory
            # budget cut (`reply.sample.clamped_from is not None`), call
            # `conversation.discard(reply)` before ending.
            if reply.truncated or reply.native_message is None:  # cut, or won't parse
                return ended("truncated" if reply.truncated else "unusable_reply", turn, calls)
            messages.append(reply.native_message)
            if not reply.tool_calls:
                return ended("completed", turn, calls)
            for call in reply.tool_calls:  # in order: a later call may read an earlier one's file
                result = json.dumps(await self.call(sandbox, call))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id or "",
                        "name": call.name or "",
                        "content": result,
                    }
                )
                calls += 1
        return ended("max_turns", self.options.max_turns, calls)

    async def call(self, sandbox: Blaxel, call: ParsedToolCall) -> dict[str, JsonValue]:
        """One tool call's result, as the model will read it."""
        try:
            if call.name == "shell":
                return await self.sh(sandbox, Shell.model_validate(call.arguments).command)
            if call.name == "write_file":
                write = WriteFile.model_validate(call.arguments)
                return await self.sh(sandbox, WRITE, {"FILE": write.path, "CONTENT": write.content})
        except ValidationError as error:
            return {"error": f"bad arguments for {call.name}: {error.errors()[0]['msg']}"}
        return {"error": f"there is no tool {call.name!r}"}

    async def sh(
        self, sandbox: Blaxel, command: str, env: Mapping[str, str] | None = None
    ) -> dict[str, JsonValue]:
        """``command`` in /work, started, then waited for: Blaxel kills it at its timeout and
        the wait returns it ``killed``, where an exec that waited itself would raise."""
        box = await sandbox.client()
        timeout, limit = self.options.command_timeout_seconds, self.options.max_output_chars
        request = {"command": command, "working_dir": WORKDIR, "env": dict(env or {})}
        started = await box.process.exec({**request, "keep_alive": True, "timeout": timeout})
        wait_ms = (timeout + WAIT_MARGIN_SECONDS) * 1000
        done = await box.process.wait(started.pid, max_wait=wait_ms, interval=WAIT_INTERVAL_MS)
        # Blaxel's record keeps only the last 64 KB; the provider's wait refills it from the logs.
        stdout, stderr = done.stdout or "", done.stderr or ""
        return {
            "status": str(done.status),
            "exit_code": done.exit_code,
            "stdout": stdout[-limit:],
            "stderr": stderr[-limit:],
        }

    async def grade(self, grading: Grading) -> float:
        sandbox = grading.require_execution_sandbox(Blaxel)
        box = await sandbox.client()
        # A command the agent deadline cut may still be running: stop what the sandbox
        # lists as running before the check. A job the agent detached is not stopped.
        for process in await box.process.list():
            if process.status == "running":
                await box.process.kill(process.pid)
        # Only a check that ran to the end passes: Blaxel's wait hands back a record
        # still ``running`` (exit code 0) when the sandbox stops answering mid-check.
        checked = await self.sh(sandbox, self.task.value.check)
        if checked["status"] == "completed" and checked["exit_code"] == 0:
            return 1.0
        await sandbox.ensure_alive()  # a failed check measures only a live sandbox
        return 0.0
