"""Tasks as data: one JSON object per line of ``[rollout_runtime.files] tasks``.

The cookbook loads the file once, when the runtime is built, and digests each task
by its canonical JSON, so the run record names exactly the data every rollout ran
on. A line's ``id`` names its task; a line without one is named by its position
among the file's tasks, counting from 0.

An episode that declares a task model has every line but its ``id`` validated with
it at load. The task's data, and so its digest, is then the validated model's JSON
dump with the line's ``id`` kept: defaults the model fills in are part of what ran,
and of what the record names.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticSerializationError

from loops_cookbook.evidence import JsonValue, json_mapping
from loops_cookbook.runtime.context import RuntimeConfigurationError


@dataclass(frozen=True, slots=True)
class Task[TaskT: BaseModel | None]:
    """One task: its ID and its line's data, a private read-only copy per attempt.

    ``value`` is the line as the episode's task model validated it, or ``None`` for
    an episode that declares none.
    """

    id: str
    data: Mapping[str, JsonValue]
    value: TaskT

    def copy(self) -> Task[TaskT]:
        """A copy no other attempt shares, its value included."""
        value = self.value
        return Task(
            self.id,
            json_mapping(self.data, where=f"task {self.id}"),
            value if value is None else value.model_copy(deep=True),
        )


def task_digest(data: Mapping[str, JsonValue]) -> str:
    """``sha256:`` of the task's canonical JSON: sorted keys, no whitespace."""
    canonical = json.dumps(dict(data), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{sha256(canonical.encode()).hexdigest()}"


def validated_dump(value: BaseModel, *, where: str) -> dict[str, JsonValue]:
    """The validated model's JSON dump, refused unless it is of the model's field types.

    It is the model's schema serializer's dump, so a ``model_dump`` the model
    overrides is never what is digested. A validator may return what the field's
    type is not; its dump would then not be what the model's types say (a
    ``SecretStr`` dumps as asterisks, so different values would share a digest), and
    is refused rather than digested.
    """
    try:
        return type(value).__pydantic_serializer__.to_python(value, mode="json", warnings="error")
    except PydanticSerializationError as error:
        raise RuntimeConfigurationError(
            f"{where}: the validated {type(value).__name__} does not dump as its field types: "
            f"{error}"
        ) from None


def load_tasks(path: Path, model: type[BaseModel] | None = None) -> list[Task[Any]]:
    """Every task in the JSONL file at ``path``, validated with ``model`` if one is given.

    Blank lines are skipped. The ``id`` names the line, so it is the cookbook's and
    never the model's: the model validates the rest of the line.
    """
    tasks: list[Task[Any]] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            data = json_mapping(json.loads(line), where=where)
        except ValueError as error:  # not JSON, or not an object of finite JSON values
            raise RuntimeConfigurationError(
                f"{where}: a task must be one JSON object: {error}"
            ) from None
        task_id = data.get("id", str(len(tasks)))
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeConfigurationError(f"{where}: a task's id must be a nonempty string")
        if task_id in seen:
            raise RuntimeConfigurationError(f"{where}: task id {task_id!r} is used twice")
        seen.add(task_id)
        tasks.append(_task(task_id, data, model, where))
    if not tasks:
        raise RuntimeConfigurationError(f"{path} has no tasks")
    return tasks


def _task(
    task_id: str, data: Mapping[str, JsonValue], model: type[BaseModel] | None, where: str
) -> Task[Any]:
    if model is None:
        return Task(task_id, data, None)
    try:
        value = model.model_validate({key: item for key, item in data.items() if key != "id"})
    except ValidationError as error:
        raise RuntimeConfigurationError(
            f"{where}: the task is not a valid {model.__name__}: {error}"
        ) from None
    dump = validated_dump(value, where=where)
    if "id" in data:
        dump["id"] = data["id"]
    return Task(task_id, json_mapping(dump, where=where), value)
