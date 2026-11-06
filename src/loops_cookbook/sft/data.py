"""The SFT data contracts: two JSONL formats, each validated row by row.

A dataset is in one of them, named once by ``[data] format``: **chat**, a
conversation in OpenAI's chat format with per-turn ``weight``, re-rendered with the
trained model's template; or **tokens**, ``input_ids`` and 0/1 ``weights`` trained as
given. What each accepts, and why, is in docs/reference/sft.md; the models here are
the contract that page describes.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self, get_args

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictInt,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from loops_cookbook.evidence import JsonValue

Content = str | list[dict[str, JsonValue]] | None
"""Plain text, or content parts (text and images) for the renderer to check."""


class Record(BaseModel):
    """A row as exported: fields the contract does not read are kept aside, not refused.

    Real exports carry columns of their own -- an ``id``, a ``source``, OpenAI's
    ``parallel_tool_calls`` -- and refusing a dataset over them would make every user
    clean it first. They are never rendered or trained, and ``check`` lists them, so a
    misspelt field is visible rather than silently missing.
    """

    model_config = ConfigDict(extra="allow", frozen=True, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def _null_is_absent(cls, data: object) -> object:
        # A Hugging Face export gives every message every key any message has, null
        # where it has none: `"tool_calls": null` is a message that calls no tool.
        if not isinstance(data, dict):
            return data
        return {
            key: value
            for key, value in data.items()
            if value is not None
            or (key in cls.model_fields and cls.model_fields[key].is_required())
        }

    def ignored(self, at: str = "") -> set[str]:
        """Dotted paths of every field kept aside here and in nested records."""
        paths = {f"{at}{name}" for name in self.model_extra or {}}
        for name, value in self:
            items = value if isinstance(value, list) else [value]
            for item in items:
                if isinstance(item, Record):
                    suffix = "[]" if isinstance(value, list) else ""
                    paths |= item.ignored(f"{at}{name}{suffix}.")
        return paths


def _flag(value: object) -> object:
    # 1.0 is how a dataframe writes 1; true is not a weight.
    if isinstance(value, bool):
        raise ValueError("must be 0 or 1, not a boolean")  # noqa: TRY004 - pydantic reports ValueError
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    return value


Flag = Annotated[StrictInt, Field(ge=0, le=1), BeforeValidator(_flag)]
"""0 or 1; ``1.0`` is accepted as 1, ``true`` is refused."""


class Function(Record):
    name: str = Field(min_length=1)
    arguments: str
    """A JSON object, as OpenAI writes it; a mapping, as Hugging Face datasets do, is
    encoded to one."""

    @field_validator("arguments", mode="before")
    @classmethod
    def _encoded(cls, arguments: object) -> object:
        return json.dumps(arguments) if isinstance(arguments, dict) else arguments

    @field_validator("arguments")
    @classmethod
    def _json_object(cls, arguments: str) -> str:
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ValueError(f"arguments is not JSON: {error.msg}") from None
        if not isinstance(parsed, dict):
            raise ValueError("arguments must encode a JSON object")  # noqa: TRY004 - pydantic reports ValueError
        return arguments


class ToolCall(Record):
    id: str | None = Field(default=None, min_length=1)
    type: Literal["function"] = "function"
    function: Function


class SystemMessage(Record):
    role: Literal["system"]
    content: Content
    name: str | None = None


class UserMessage(Record):
    role: Literal["user"]
    content: Content
    name: str | None = None


class AssistantMessage(Record):
    role: Literal["assistant"]
    content: Content = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list[ToolCall])
    name: str | None = None
    weight: Flag = 1


class ToolMessage(Record):
    role: Literal["tool"]
    content: Content
    tool_call_id: str | None = Field(default=None, min_length=1)
    name: str | None = None


Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage, Field(discriminator="role")
]
_ROLES = frozenset(
    get_args(model.model_fields["role"].annotation)[0]
    for model in (SystemMessage, UserMessage, AssistantMessage, ToolMessage)
)


class ChatExample(Record):
    """The chat contract: one conversation, and which of its assistant turns train."""

    messages: list[Message] = Field(min_length=1)
    tools: list[dict[str, JsonValue]] = Field(default_factory=list[dict[str, JsonValue]])
    """OpenAI tool declarations; a JSON string of them, as Hub datasets often store
    them, is decoded."""
    metadata: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])

    @field_validator("tools", mode="before")
    @classmethod
    def _decoded(cls, tools: object) -> object:
        if not isinstance(tools, str):
            return tools
        try:
            return json.loads(tools)
        except json.JSONDecodeError as error:
            raise ValueError(f"tools is a string but not JSON: {error.msg}") from None

    @model_validator(mode="after")
    def _well_formed(self) -> Self:
        for index, message in enumerate(self.messages):
            if isinstance(message, SystemMessage) and index:
                raise ValueError(f"messages[{index}]: a system message must come first")
            if isinstance(message, ToolMessage):
                calls = _answered_calls(self.messages, index)
                if calls is None:
                    raise ValueError(
                        f"messages[{index}]: a tool message must follow an assistant "
                        "message with tool_calls, or another tool message"
                    )
                if message.tool_call_id is not None and message.tool_call_id not in {
                    call.id for call in calls
                }:
                    raise ValueError(
                        f"messages[{index}]: tool_call_id {message.tool_call_id!r} answers "
                        "no call of the assistant message it follows"
                    )
            if isinstance(message, AssistantMessage) and message.tool_calls:
                _check_answers(self.messages, index, message.tool_calls)
                following = self.messages[index + 1 : index + 2]
                if following and not isinstance(following[0], ToolMessage):
                    # The model ends a turn that calls tools by handing over to them; a
                    # turn followed by anything else never happened as written.
                    raise ValueError(
                        f"messages[{index}]: an assistant message with tool_calls must be "
                        "followed by its tool results, or end the conversation"
                    )
        if not self.trained_turns:
            raise ValueError("no assistant message is trained (every one has weight 0)")
        for index in self.trained_turns:
            message = self.messages[index]
            assert isinstance(message, AssistantMessage)
            if not (message.content or message.reasoning_content or message.tool_calls):
                # Usually a field the export dropped; trained, it teaches saying nothing.
                raise ValueError(
                    f"messages[{index}]: a trained assistant message with no content, "
                    "reasoning_content or tool_calls; give it weight 0 to keep it as context"
                )
        return self

    @property
    def trained_turns(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, message in enumerate(self.messages)
            if isinstance(message, AssistantMessage) and message.weight
        )

    def chat_messages(self) -> list[dict[str, object]]:
        """The messages as a renderer takes them: only the contract's own fields, no
        ``weight``, tool call arguments decoded (chat templates iterate them as a
        mapping), and calls given without ids numbered so their results name them in
        order."""
        result: list[dict[str, object]] = []
        unnamed: list[str] = []
        for index, message in enumerate(self.messages):
            rendered = _declared(message, exclude={"weight", "tool_calls"})
            if isinstance(message, AssistantMessage) and message.tool_calls:
                calls = []
                for position, call in enumerate(message.tool_calls):
                    identity = call.id or f"call_{index}_{position}"
                    calls.append(
                        {
                            **_declared(call, exclude={"function"}),
                            "id": identity,
                            "function": {
                                "name": call.function.name,
                                "arguments": json.loads(call.function.arguments),
                            },
                        }
                    )
                rendered["tool_calls"] = calls
                unnamed = [] if message.tool_calls[0].id else [call["id"] for call in calls]
            elif isinstance(message, ToolMessage) and message.tool_call_id is None:
                rendered["tool_call_id"] = unnamed.pop(0)
            result.append(rendered)
        return result


def _declared(record: Record, *, exclude: set[str]) -> dict[str, object]:
    return record.model_dump(exclude_none=True, exclude=exclude | set(record.model_extra or {}))


def _check_answers(messages: Sequence[object], index: int, calls: list[ToolCall]) -> None:
    """The tool block after ``messages[index]`` answers each of its calls exactly once,
    or is empty: a conversation may end on the calls. With ids, results name their
    calls; without any, the ``n``-th result answers the ``n``-th call."""
    results = []
    for following in messages[index + 1 :]:
        if not isinstance(following, ToolMessage):
            break
        results.append(following)
    if not results:
        return
    ids = [call.id for call in calls]
    answered = [result.tool_call_id for result in results]
    if all(name is None for name in [*ids, *answered]):
        if len(results) != len(calls):
            raise ValueError(
                f"messages[{index}]: {len(calls)} tool call(s) and {len(results)} result(s); "
                "without ids, results answer calls in order, so there must be one each"
            )
        return
    if None in ids or None in answered:
        raise ValueError(
            f"messages[{index}]: ids are given for some of its tool calls or results and "
            "not others; give every call an id and every result its tool_call_id, or none"
        )
    named = [name for name in ids if name is not None]
    answers = [name for name in answered if name is not None]
    if len(set(named)) != len(named):
        raise ValueError(f"messages[{index}]: two tool calls share an id")
    if not set(answers) <= set(named):
        # A result answering no call: that message reports it.
        return
    if twice := sorted({name for name in answers if answers.count(name) > 1}):
        raise ValueError(f"messages[{index}]: its tool results answer {', '.join(twice)} twice")
    if unanswered := [name for name in named if name not in answers]:
        raise ValueError(
            f"messages[{index}]: its tool results leave {', '.join(unanswered)} unanswered"
        )


def _answered_calls(messages: Sequence[object], index: int) -> list[ToolCall] | None:
    """The calls of the assistant message heading the tool block ``index`` is in."""
    while index > 0:
        index -= 1
        previous = messages[index]
        if isinstance(previous, AssistantMessage):
            return previous.tool_calls or None
        if not isinstance(previous, ToolMessage):
            return None
    return None


class TokenExample(Record):
    """The tokens contract: a sequence tokenized for the trained model, targets marked."""

    input_ids: list[Annotated[StrictInt, Field(ge=0)]] = Field(min_length=2)
    weights: list[Flag]
    metadata: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])

    @model_validator(mode="after")
    def _targets(self) -> Self:
        if len(self.weights) != len(self.input_ids):
            raise ValueError(
                f"weights has {len(self.weights)} entries for {len(self.input_ids)} input_ids"
            )
        if self.weights[0]:
            raise ValueError("weights[0] must be 0: the first token has nothing to predict it")
        if not any(self.weights):
            raise ValueError("no token is a target (every weight is 0)")
        return self


Format = Literal["chat", "tokens"]

CONTRACTS: dict[str, TypeAdapter[ChatExample] | TypeAdapter[TokenExample]] = {
    "chat": TypeAdapter(ChatExample),
    "tokens": TypeAdapter(TokenExample),
}


class DatasetError(ValueError):
    """Rows that do not meet the data contract, each named by file and line."""


@dataclass(frozen=True)
class Row:
    """A validated example and where it came from, for every report that names it."""

    source: str
    line: int
    example: ChatExample | TokenExample

    @property
    def name(self) -> str:
        return f"{self.source}:{self.line}"


_SHOWN_ERRORS = 20


def load_rows(path: str, format: Format) -> list[Row]:
    """Every row of a JSONL file in ``format``, or a ``DatasetError`` naming each one
    that does not meet it."""
    contract = CONTRACTS[format]
    source = Path(path)
    rows: list[Row] = []
    problems: list[str] = []
    with source.open(encoding="utf-8") as handle:
        for number, text in enumerate(handle, start=1):
            if not text.strip():
                continue
            try:
                rows.append(Row(source.name, number, contract.validate_json(text)))
            except ValidationError as error:
                problems.extend(
                    f"{source.name}:{number}: {_location(issue['loc'])}{issue['msg']}"
                    for issue in error.errors(include_url=False)
                )
    if problems:
        shown = "\n  ".join(problems[:_SHOWN_ERRORS])
        more = len(problems) - _SHOWN_ERRORS
        raise DatasetError(
            f"{len(problems)} problem(s) in {path}, read as the {format} format:\n  {shown}"
            + (f"\n  ... and {more} more" if more > 0 else "")
        )
    if not rows:
        raise DatasetError(f"{path} has no rows")
    return rows


def _location(loc: tuple[int | str, ...]) -> str:
    # The discriminated union adds the role as a path segment; it names no field.
    parts = [str(part) for part in loc if part not in _ROLES]
    return f"{'.'.join(parts)}: " if parts else ""


@dataclass(frozen=True)
class Dataset:
    train: list[Row]
    validation: list[Row]


def load_dataset(
    train: str,
    *,
    format: Format,
    validation: str | None,
    validation_fraction: float | None,
    seed: int,
) -> Dataset:
    """The training and validation rows: from two files, or one file split by ``seed``."""
    train_rows = load_rows(train, format)
    if validation is not None:
        held_out = load_rows(validation, format)
    elif validation_fraction is not None:
        order = list(range(len(train_rows)))
        random.Random(seed).shuffle(order)
        count = max(1, round(validation_fraction * len(train_rows)))
        if count >= len(train_rows):
            raise DatasetError(
                f"validation_fraction {validation_fraction} of {len(train_rows)} rows "
                "leaves nothing to train on"
            )
        chosen = set(order[:count])
        held_out = [row for index, row in enumerate(train_rows) if index in chosen]
        train_rows = [row for index, row in enumerate(train_rows) if index not in chosen]
    else:
        held_out = []
    return Dataset(train_rows, held_out)
