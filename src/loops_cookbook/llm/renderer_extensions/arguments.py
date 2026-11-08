"""Read a tool-call argument back as the chat template wrote it.

The chat templates the cookbook extends write a string argument as it is. GLM-5.x and
Qwen3.6 and later write any other value as JSON. Qwen3.5 and Nemotron 3 write a boolean,
null or number with Jinja's ``string`` filter, which gives ``True`` and ``None``.

The text alone therefore cannot say what the template wrote. The text ``7`` may be the
integer 7 or the string "7", and ``"x"`` may be a string that includes its quotes. The
tool's schema decides. A value is read as JSON only when the schema admits that JSON's
type. Every typed schema except a plain string admits null. A JSON string is never
used, because no template writes a string as JSON. Any other value is returned as the
text the template wrote. That covers an untyped value and text that its declared type
cannot read, such as ``30s`` for an integer. The tool then accepts or refuses it.
"""

from __future__ import annotations

import json
from typing import Any

# A schema with none of these keys says nothing about its value's type.
_TYPING_KEYS = ("type", "anyOf", "oneOf", "allOf", "enum", "const", "$ref")
_PYTHON_SCALARS: dict[str, Any] = {"True": True, "False": False, "None": None}
_UNREAD = object()


def read_argument(text: str, schema: Any, *, python_scalars: bool = False) -> Any:
    """Return the value the template wrote as ``text`` between its delimiters.

    Set ``python_scalars`` when the template may write booleans and null the way Python
    does, as ``True``, ``False`` and ``None``, as well as in JSON.
    """
    if not isinstance(schema, dict) or not any(key in schema for key in _TYPING_KEYS):
        return text
    types = _types(schema)
    if types == {"string"}:
        return text
    value = _UNREAD
    try:
        value = json.loads(text)
    except ValueError:
        if python_scalars and text.strip() in _PYTHON_SCALARS:
            value = _PYTHON_SCALARS[text.strip()]
    if value is _UNREAD or isinstance(value, str) or not _admits(types, value):
        return text
    return value


def _types(schema: dict[str, Any]) -> set[str] | None:
    """Return the JSON types that ``schema`` admits.

    The result is None when the types cannot be worked out, as with ``$ref`` or ``allOf``.
    """
    found: set[str] = set()
    declared = schema.get("type")
    if isinstance(declared, str):
        found.add(declared)
    elif isinstance(declared, list):
        found.update(t for t in declared if isinstance(t, str))
    for key in ("anyOf", "oneOf"):
        for branch in schema.get(key) or []:
            if not isinstance(branch, dict):
                continue
            inner = _types(branch)
            if inner is None:
                return None
            found |= inner
    for value in [*(schema.get("enum") or []), *([schema["const"]] if "const" in schema else [])]:
        found.add(_type_of(value))
    if not found and ("$ref" in schema or "allOf" in schema):
        return None
    return found


def _type_of(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return "array" if isinstance(value, list) else "object"


def _admits(types: set[str] | None, value: Any) -> bool:
    # A template writes a value that was left out as null, so ``days=None`` renders as
    # ``null``. Every schema except a plain string therefore admits null. For a plain
    # string, ``null`` is the text.
    if types is None or value is None:
        return True
    kind = _type_of(value)
    return kind in types or (kind == "integer" and "number" in types)
