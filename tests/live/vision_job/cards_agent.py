"""A multi-turn vision agent: tools return text and images, the final message answers."""

import asyncio
import base64
import binascii
import posixpath
import shlex

from harbor.agents.options import AgentOptions
from pydantic import Field

from loops_cookbook.harbor import LoopsAgent
from loops_cookbook.llm import ContextWindowExceeded
from loops_cookbook.model_access import ModelAccessMode

CARDS_DIR = "/app/cards"
SYSTEM_PROMPT = (
    "You can see images. Use the tools to inspect the cards, then answer the user "
    "with the letters only."
)
TOOLS = [
    {
        "name": "list_cards",
        "description": f"List the PNG cards in {CARDS_DIR}, in order.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "look",
        "description": f"Show one card. Returns the image at the given path under {CARDS_DIR}.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Card path, e.g. /app/cards/1.png"}
            },
            "required": ["path"],
        },
    },
]


class CardsOptions(AgentOptions):
    """Settings supplied through ``agents[].kwargs`` in the Harbor job."""

    max_turns: int = Field(default=6, ge=1)


class CardsAgent(LoopsAgent):
    model_access = ModelAccessMode.DIRECT
    options_model = CardsOptions
    options: CardsOptions

    @staticmethod
    def name() -> str:
        return "vision-tools"

    def version(self) -> str:
        return "1"

    async def setup(self, environment) -> None:
        pass

    async def run(self, instruction, environment, context) -> None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        conversation = self.llm().conversation()
        status, generations, answer = "max_turns", 0, ""

        def tool_message(call, content):
            return {
                "role": "tool",
                "tool_call_id": call.id or "",
                "name": call.name,
                "content": content,
            }

        async def execute(call):
            if call.name == "list_cards":
                listed = await environment.exec(f"ls {CARDS_DIR}")
                return tool_message(call, listed.stdout.strip() or listed.stderr.strip())
            if call.name != "look":
                return tool_message(call, f"error: unknown tool {call.name!r}")
            path = str((call.arguments or {}).get("path", ""))
            normalized = posixpath.normpath(path)
            if posixpath.dirname(normalized) != CARDS_DIR or not normalized.endswith(".png"):
                return tool_message(call, f"error: {path!r} is not a card under {CARDS_DIR}")
            # ``base64 < file | tr -d`` works on GNU and BusyBox alike (``-w0`` is GNU-only).
            shown = await environment.exec(f"base64 < {shlex.quote(normalized)} | tr -d '\\n'")
            if shown.return_code != 0:
                return tool_message(call, f"error: could not read {normalized}: {shown.stderr}")
            try:
                picture = base64.b64decode(shown.stdout.strip(), validate=True)
            except binascii.Error:
                return tool_message(call, f"error: {normalized} did not come back as base64")
            return tool_message(
                call,
                [
                    {"type": "image", "image": picture},
                    {"type": "text", "text": posixpath.basename(normalized)},
                ],
            )

        for _ in range(self.options.max_turns):
            try:
                response = await conversation.generate(
                    messages=messages,
                    tools=TOOLS,
                )
            except ContextWindowExceeded:
                status = "context_overflow"
                break
            generations += 1
            # A cut turn leaves no reply to append, and its empty content is not the
            # answer to write out -- grading it would score a blank deliverable.
            if response.truncated:
                status = "truncated"
                break
            if response.native_message is None:
                # No assistant turn to append: the completion did not parse, or a
                # tool call did not. A separate question from being cut short, and
                # answering it needs replacement history this example does not build.
                status = "unusable_reply"
                break
            messages.append(response.native_message)
            if not response.tool_calls:
                status, answer = "completed", (response.content or "").strip()
                break
            messages.extend(await asyncio.gather(*(execute(c) for c in response.tool_calls)))

        result = await environment.exec(
            "mkdir -p /app && printf %s " + shlex.quote(answer) + " > /app/answer.txt"
        )
        if result.return_code != 0:
            raise RuntimeError("Could not persist the example agent's answer")
        context.metadata = {
            **(context.metadata or {}),
            "termination_reason": status,
            "generation_count": generations,
        }
