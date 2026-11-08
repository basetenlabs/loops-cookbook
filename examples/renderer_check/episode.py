"""The live check of a renderer's tool-call path: an agent whose tools take every kind
of argument the chat templates write differently.

``run(sample)`` holds a small project in memory (files plus a settings store) and
offers tools whose schemas cover the cases a renderer must read as written:

- strings with leading or trailing whitespace and inner newlines (``edit_file``);
- integer, number, boolean and null (``configure``, ``set_setting``);
- an enum (``configure.level``), a union with a string branch (``set_setting.value``);
- an array of strings (``tag_file``) and an array of objects through ``$defs``
  (``apply_patch``);
- an undeclared tool the system prompt mentions on some tasks (``delete_file``), so the
  model calls a tool that is not in the schema and the episode sees ``unknown_tool``;
  the task then expects the file left in place, since no tool can remove it.

Every tool validates its arguments strictly: a string ``"3"`` for an integer or a
stripped ``old`` fails the call, and the model sees that in the result. ``grade()``
compares the whole of the files, settings and tags with the task's expected state, byte
for byte and type for type. The episode records what each call looked like from the agent's side so
``report.py`` can tabulate a run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from loops_cookbook.episodes import Episode, Grading, Sampling, Task
from loops_cookbook.llm import ContextWindowExceeded, Message, ToolSchema
from loops_cookbook.runtime import JsonValue

SYSTEM_PROMPT = (
    "You are editing a small project: some files and a settings store. Use read_file to "
    "look at a file and edit_file to change part of it: old must match the file exactly, "
    "including indentation and newlines, and must occur exactly once. Use configure and "
    "set_setting for settings, tag_file for tags, and apply_patch for several edits at "
    "once. Keep every change minimal. When the task is done, reply with one short "
    "sentence and no tool call."
)
DELETE_HINT = " There is also delete_file(path), which removes a file."


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CheckTask(Strict):
    name: str
    instruction: str = Field(min_length=1)
    files: dict[str, str]
    expected_files: dict[str, str]
    expected_settings: dict[str, JsonValue] = {}
    expected_tags: dict[str, list[str]] = {}
    mention_delete: bool = False


class CheckOptions(Strict):
    max_turns: int = Field(ge=1)


class ReadFile(Strict):
    """Read a file and see its whole content."""

    path: str


class EditFile(Strict):
    """Replace one exact occurrence of old with new in a file."""

    path: str
    old: str
    new: str


class WriteFile(Strict):
    """Write a file, replacing what it held."""

    path: str
    content: str


Level = Literal["debug", "info", "warning", "error"]


class Configure(Strict):
    """Set the project's main settings in one call."""

    retries: int = Field(ge=0)
    ratio: float
    enabled: bool
    level: Level
    proxy: str | None = None


class SetSetting(Strict):
    """Set one setting to a number, a boolean, null, or a string."""

    key: str
    value: int | float | bool | str | None


class TagFile(Strict):
    """Replace a file's tags."""

    path: str
    tags: list[str]


class PatchOp(Strict):
    path: str
    old: str
    new: str
    note: str | None = None


class ApplyPatch(Strict):
    """Apply several exact replacements, in order."""

    ops: list[PatchOp] = Field(min_length=1)


DECLARED: dict[str, type[Strict]] = {
    "read_file": ReadFile,
    "edit_file": EditFile,
    "write_file": WriteFile,
    "configure": Configure,
    "set_setting": SetSetting,
    "tag_file": TagFile,
    "apply_patch": ApplyPatch,
}
TOOLS: list[ToolSchema] = [
    {"name": name, "description": model.__doc__ or "", "parameters": model.model_json_schema()}
    for name, model in DECLARED.items()
]


def _kind(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


class RendererCheckEpisode(Episode[CheckTask, CheckOptions]):
    def __init__(self, task: Task[CheckTask], options: CheckOptions) -> None:
        super().__init__(task, options)
        self.files = dict(task.value.files)
        self.settings: dict[str, JsonValue] = {}
        self.tags: dict[str, list[str]] = {}
        self.calls: list[dict[str, JsonValue]] = []

    async def run(self, sample: Sampling) -> Mapping[str, JsonValue]:
        conversation = sample.model.conversation()
        system = SYSTEM_PROMPT + (DELETE_HINT if self.task.value.mention_delete else "")
        messages: list[Message] = [
            {"role": "system", "content": system},
            {"role": "user", "content": self.task.value.instruction},
        ]
        for turn in range(1, self.options.max_turns + 1):
            try:
                reply = await conversation.generate(messages=messages, tools=TOOLS)
            except ContextWindowExceeded:
                return self.ended("context_overflow", turn - 1)
            if reply.truncated:
                return self.ended("truncated", turn)
            if reply.native_message is None:
                self.calls.extend(self.record(call, native_kept=False) for call in reply.tool_calls)
                return self.ended("message_withheld", turn)
            messages.append(reply.native_message)
            if not reply.tool_calls:
                return self.ended("completed", turn)
            for call in reply.tool_calls:
                record = self.record(call, native_kept=True)
                result = self.call(call.name or "", call.arguments)
                record["tool_error"] = "error" in result
                self.calls.append(record)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id or "",
                        "name": call.name or "",
                        "content": json.dumps(result),
                    }
                )
        return self.ended("max_turns", self.options.max_turns)

    def record(self, call: object, *, native_kept: bool) -> dict[str, JsonValue]:
        arguments = getattr(call, "arguments", None)
        kinds: dict[str, JsonValue] = {}
        edge: list[str] = []
        if isinstance(arguments, dict):
            for key, value in arguments.items():
                kinds[str(key)] = _kind(value)
                if isinstance(value, str) and value != value.strip():
                    edge.append(str(key))
        return {
            "tool": getattr(call, "name", None) or "",
            "status": str(getattr(call, "status", "")),
            "native_kept": native_kept,
            "argument_kinds": kinds,
            "strings_with_edge_whitespace": edge,
            "arguments_mapping": isinstance(arguments, dict),
        }

    def ended(self, reason: str, turns: int) -> dict[str, JsonValue]:
        return {"termination_reason": reason, "turns": turns, "calls": self.calls}

    def call(self, name: str, arguments: object) -> dict[str, JsonValue]:
        model = DECLARED.get(name)
        if model is None:
            return {"error": f"there is no tool {name!r}"}
        try:
            args = model.model_validate(arguments)
        except ValidationError as error:
            first = error.errors()[0]
            where = ".".join(str(part) for part in first["loc"])
            return {"error": f"bad arguments for {name}: {where}: {first['msg']}"}
        if isinstance(args, ReadFile):
            if args.path not in self.files:
                return {"error": f"no such file: {args.path}"}
            return {"content": self.files[args.path]}
        if isinstance(args, EditFile):
            return self.replace(args.path, args.old, args.new)
        if isinstance(args, WriteFile):
            self.files[args.path] = args.content
            return {"ok": True}
        if isinstance(args, Configure):
            self.settings.update(args.model_dump())
            return {"ok": True, "settings": dict(self.settings)}
        if isinstance(args, SetSetting):
            self.settings[args.key] = args.value
            return {"ok": True, "settings": dict(self.settings)}
        if isinstance(args, TagFile):
            if args.path not in self.files:
                return {"error": f"no such file: {args.path}"}
            self.tags[args.path] = list(args.tags)
            return {"ok": True}
        if isinstance(args, ApplyPatch):
            for index, op in enumerate(args.ops):
                result = self.replace(op.path, op.old, op.new)
                if "error" in result:
                    return {"error": f"op {index}: {result['error']}"}
            return {"ok": True, "applied": len(args.ops)}
        return {"error": f"there is no tool {name!r}"}

    def replace(self, path: str, old: str, new: str) -> dict[str, JsonValue]:
        if path not in self.files:
            return {"error": f"no such file: {path}"}
        count = self.files[path].count(old)
        if count != 1:
            return {"error": f"old occurs {count} times in {path}; it must occur exactly once"}
        self.files[path] = self.files[path].replace(old, new, 1)
        return {"ok": True}

    async def grade(self, grading: Grading) -> float:
        # The whole of each store must equal the task's expected state: a file left in
        # place, a setting never set, or an extra tag all fail. Settings compare as JSON
        # so that 3 and "3", or true and 1, stay distinct.
        task = self.task.value
        return float(
            self.files == task.expected_files
            and json.dumps(self.settings, sort_keys=True)
            == json.dumps(task.expected_settings, sort_keys=True)
            and self.tags == task.expected_tags
        )
