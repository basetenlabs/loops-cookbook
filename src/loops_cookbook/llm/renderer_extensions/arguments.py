"""Read tool-call arguments back as the chat template wrote them.

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

A ``$ref`` is followed within the tool's own parameters, such as ``#/$defs/...``. That
is how pydantic writes an enum or a nested model. A ``$ref`` that leads nowhere admits
any type.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from renderers.base import ParsedToolCall, ToolCallParseStatus

# A schema with none of these keys says nothing about its value's type.
_TYPING_KEYS = ("type", "anyOf", "oneOf", "allOf", "enum", "const", "$ref")
_PYTHON_SCALARS: dict[str, Any] = {"True": True, "False": False, "None": None}
_UNREAD = object()

# A tool's parameter schemas by name, paired with the parameters object that their
# ``$ref``s point into.
Parameters = tuple[dict[str, Any], dict[str, Any]]


def tool_parameters(tools) -> dict[str, Parameters]:
    """Return each declared tool's parameters by tool name.

    Both flat tool specs and OpenAI-wrapped specs are read. A tool without parameters
    gets empty ones, so its name is still known.
    """
    declared: dict[str, Parameters] = {}
    for tool in tools or ():
        spec = tool.get("function", tool) if isinstance(tool, dict) else None
        if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
            continue
        root = spec.get("parameters")
        root = root if isinstance(root, dict) else {}
        properties = root.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        declared[spec["name"]] = (
            {key: schema for key, schema in properties.items() if isinstance(schema, dict)},
            root,
        )
    return declared


def read_argument(
    text: str, schema: Any, *, root: dict[str, Any] | None = None, python_scalars: bool = False
) -> Any:
    """Return the value the template wrote as ``text`` between its delimiters.

    ``root`` is the parameters object that a ``$ref`` in ``schema`` points into. Set
    ``python_scalars`` when the template may write booleans and null the way Python
    does, as ``True``, ``False`` and ``None``, as well as in JSON.
    """
    if not isinstance(schema, dict) or not any(key in schema for key in _TYPING_KEYS):
        return text
    types = _types(schema, root or {}, ())
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


def _types(schema: Any, root: dict[str, Any], seen: tuple[str, ...]) -> set[str] | None:
    """Return the JSON types that ``schema`` admits, or None when they can't be worked out."""
    if not isinstance(schema, dict):
        return None
    found: set[str] = set()
    declared = schema.get("type")
    if isinstance(declared, str):
        found.add(declared)
    elif isinstance(declared, list):
        found.update(t for t in declared if isinstance(t, str))
    for key in ("anyOf", "oneOf"):
        for branch in schema.get(key) or []:
            inner = _types(branch, root, seen)
            if inner is None:
                return None
            found |= inner
    for value in [*(schema.get("enum") or []), *([schema["const"]] if "const" in schema else [])]:
        found.add(_type_of(value))
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref not in seen:
        inner = _types(_resolve(ref, root), root, (*seen, ref))
        if inner is None:
            return None
        found |= inner
    if "allOf" in schema:
        known = [t for branch in schema["allOf"] or [] if (t := _types(branch, root, seen))]
        if known:
            found |= set.intersection(*known)
    if not found and ("$ref" in schema or "allOf" in schema):
        return None
    return found


def _resolve(ref: str, root: dict[str, Any]) -> Any:
    """Return what a local ``#/a/b`` reference names in ``root``, or None if nothing."""
    if not ref.startswith("#/"):
        return None
    node: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict):
            return None
        node = node.get(part.replace("~1", "/").replace("~0", "~"))
    return node


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


# --- the Qwen3.5 and Nemotron 3 XML form ----------------------------------------------

_FUNCTION = re.compile(r"\s*<function=([^>]+)>(.*?)</function>\s*", re.DOTALL)
# The template writes a newline on each side of a value. The vLLM Qwen3 parser also
# accepts whitespace after ``=``, which Prime puts in the parameter name.
_PARAMETER = re.compile(r"<parameter=\s*([^>]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
_MARKUP = ("<tool_call>", "<parameter=", "<function=", "</function>")


def xml_arguments_as_written(calls: list[ParsedToolCall], tools) -> list[ParsedToolCall]:
    """Read each ``<function=...>`` call's arguments as the template wrote them.

    Prime's parse strips every value. It also marks a call ``invalid_json`` when a value
    is not the JSON its schema asks for. This function reads each value from the call's
    raw text with ``read_argument``, which also accepts the Python spelling of booleans
    and null. A complete call is ``ok``. It is ``unknown_tool`` instead when ``tools``
    are declared and none of them has the call's name.

    Only complete, unambiguous function and parameter blocks are read. A duplicate key
    or markup inside a value makes the call ``malformed_structure``, and the call is
    never read in part.
    """
    declared = tool_parameters(tools)
    return [_xml_call(call, declared) for call in calls]


def _xml_call(call: ParsedToolCall, declared: dict[str, Parameters]) -> ParsedToolCall:
    if call.status not in (ToolCallParseStatus.OK, ToolCallParseStatus.INVALID_JSON):
        return call
    malformed = replace(call, status=ToolCallParseStatus.MALFORMED_STRUCTURE)
    function = _FUNCTION.fullmatch(call.raw)
    if function is None or not function[1].strip():
        return malformed
    body = function[2]
    schemas, root = declared.get(call.name, ({}, {}))
    arguments: dict[str, Any] = {}
    offset = 0
    for parameter in _PARAMETER.finditer(body):
        name, text = parameter[1], parameter[2]
        if (
            body[offset : parameter.start()].strip()
            or not name.strip()
            or name in arguments
            or any(marker in text for marker in _MARKUP)
        ):
            return malformed
        arguments[name] = read_argument(text, schemas.get(name), root=root, python_scalars=True)
        offset = parameter.end()
    if body[offset:].strip():
        return malformed
    status = ToolCallParseStatus.OK
    if declared and call.name not in declared:
        status = ToolCallParseStatus.UNKNOWN_TOOL
    return replace(call, arguments=arguments, status=status)
