"""Read tool-call arguments back as the chat template wrote them.

The chat templates the cookbook extends write a string argument as it is. GLM-5.x and
Qwen3.6 and later write any other value as JSON. Qwen3.5 and Nemotron 3 write a boolean,
null or number with Jinja's ``string`` filter, which gives ``True`` and ``None``.

The text alone therefore cannot say what the template wrote. The text ``7`` may be the
integer 7 or the string "7", and ``"x"`` may be a string that includes its quotes. The
tool's schema says which the template would have written. JSON that the schema accepts,
by its type, its literals and every branch, is that value. JSON the schema refuses is
still that value when the schema admits no string at all, because the template then
wrote it as JSON: ``[{"old": "a"}]`` for an array of objects that each need a ``path``
reaches the tool as a list, and ``7.5`` for an integer as a number, for the tool to
refuse. Where the schema admits a string, text that it refuses as JSON is the string,
so ``7`` for an enum of other strings is "7". A JSON string is never used, because no
template writes a string as JSON, and text with whitespace around it that the schema
accepts as a string is the string, because no template pads another value. Text that is
not JSON at all, such as ``30s`` for an integer, and the value of an untyped parameter,
come back as the text the template wrote. One exception: ``True``, ``False`` and
``None``, which a model writes under a JSON template too, are read where the schema
admits no string, since nothing else can be meant. The tool then accepts or refuses
every value.

A ``$ref`` is followed within the tool's own parameters, such as ``#/$defs/...``. That
is how pydantic writes an enum or a nested model. A ``$ref`` that leads nowhere admits
any type.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any
from urllib.parse import unquote

from renderers.base import ParsedToolCall, ToolCallParseStatus

# A schema with none of these keys says nothing about its value's type.
_TYPING_KEYS = (
    "type",
    "anyOf",
    "oneOf",
    "allOf",
    "enum",
    "const",
    "$ref",
    "properties",
    "items",
)
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
        # A parameters object may itself be a $ref into its own $defs; the properties
        # live there, and the parameters object stays the root the refs point into.
        schema = root
        seen: set[str] = set()
        while isinstance(schema, dict) and "properties" not in schema:
            ref = schema.get("$ref")
            if not isinstance(ref, str) or ref in seen:
                break
            seen.add(ref)
            schema = _resolve(ref, root)
        properties = schema.get("properties") if isinstance(schema, dict) else None
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
    value = _UNREAD
    try:
        value = json.loads(text)
    except ValueError:
        # A template that writes JSON never writes ``True`` or ``None``, but a model does
        # anyway. Where the schema admits no string the spelling can only mean that
        # value, so the tool receives it rather than text it would have to coerce.
        spelled = text.strip() in _PYTHON_SCALARS
        if spelled and (python_scalars or not _admits_string(schema, root or {}, ())):
            value = _PYTHON_SCALARS[text.strip()]
    if value is _UNREAD or isinstance(value, str):
        return text
    if not _accepts(value, schema, root or {}, ()):
        # JSON the schema refuses is the string where the schema admits one, since the
        # template writes a string as it is; elsewhere the template wrote JSON, so the
        # value is that JSON and the tool refuses its content.
        return text if _admits_string(schema, root or {}, ()) else value
    # No template pads a non-string value with whitespace, so text that carries some, and
    # that the schema accepts as a string, is the string: ``  True\n`` for bool-or-string.
    # A template that writes booleans and null as Python does never writes ``true`` or
    # ``null``, so under it those spellings are strings too, where the schema admits one.
    json_spelling = python_scalars and text.strip() in ("true", "false", "null")
    if (text != text.strip() or json_spelling) and _accepts(text, schema, root or {}, ()):
        return text
    return value


def _resolve(ref: str, root: dict[str, Any]) -> Any:
    """Return what a local ``#/a/b`` reference names in ``root``, or None if nothing."""
    if ref == "#":
        return root
    if not ref.startswith("#/"):
        return None
    node: Any = root
    for part in ref[2:].split("/"):
        # A fragment is URI-escaped, then a JSON Pointer with its own ~0 and ~1 escapes.
        part = unquote(part).replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            node = node.get(part)
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def _admits_string(schema: Any, root: dict[str, Any], seen: tuple[str, ...]) -> bool:
    """Whether ``schema`` could take a string, by what it declares a value to be.

    A ``type`` without ``string``, an ``enum`` or ``const`` with no string, a union
    without a branch that admits one, or an ``allOf`` with a branch that refuses one,
    rules a string out. Anything else, including a ``$ref`` that leads nowhere, admits
    one.
    """
    if not isinstance(schema, dict):
        return True
    declared = schema.get("type")
    declared = [declared] if isinstance(declared, str) else declared
    if isinstance(declared, list) and "string" not in declared:
        return False
    if "enum" in schema and not any(isinstance(literal, str) for literal in schema["enum"] or []):
        return False
    if "const" in schema and not isinstance(schema["const"], str):
        return False
    if "anyOf" in schema and not any(_admits_string(b, root, seen) for b in schema["anyOf"] or []):
        return False
    if "oneOf" in schema and not any(_admits_string(b, root, seen) for b in schema["oneOf"] or []):
        return False
    if not all(_admits_string(b, root, seen) for b in schema.get("allOf") or []):
        return False
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref not in seen:
        return _admits_string(_resolve(ref, root), root, (*seen, ref))
    return True


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


def _accepts(
    value: Any, schema: Any, root: dict[str, Any], seen: tuple[tuple[str, int], ...]
) -> bool:
    """Whether ``schema`` accepts the JSON ``value``.

    Every constraint that says what the text was written as holds at once: the
    ``type`` (a number admits an integer), the ``enum`` or ``const`` literals by JSON
    equality, an object's ``properties``, ``required`` keys and ``additionalProperties``,
    an array's ``items``, any ``anyOf`` branch, exactly one ``oneOf`` branch (how
    pydantic tells a discriminated union's members apart), every ``allOf`` branch, and
    the ``$ref`` it follows. A ``$ref`` that leads nowhere, or back to itself for the
    same value (a recursive schema recurs only through a child value), accepts.
    Bounds, lengths and patterns do not change what a value was written as; the tool
    checks those.
    """
    if not isinstance(schema, dict):
        return True
    kind = _type_of(value)
    declared = schema.get("type")
    declared = [declared] if isinstance(declared, str) else declared
    if isinstance(declared, list) and not (
        kind in declared
        or (kind == "integer" and "number" in declared)
        or (kind == "number" and "integer" in declared and float(value).is_integer())
    ):
        return False
    if "enum" in schema and not any(_same(value, literal) for literal in schema["enum"] or []):
        return False
    if "const" in schema and not _same(value, schema["const"]):
        return False
    if isinstance(value, dict):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        if any(key not in value for key in schema.get("required") or []):
            return False
        if any(
            key in properties and not _accepts(item, properties[key], root, seen)
            for key, item in value.items()
        ):
            return False
        extra = schema.get("additionalProperties")
        if extra is False and any(key not in properties for key in value):
            return False
        if isinstance(extra, dict) and not all(
            _accepts(item, extra, root, seen)
            for key, item in value.items()
            if key not in properties
        ):
            return False
    if (
        isinstance(value, list)
        and isinstance(schema.get("items"), dict)
        and not all(_accepts(item, schema["items"], root, seen) for item in value)
    ):
        return False
    if "anyOf" in schema and not any(_accepts(value, b, root, seen) for b in schema["anyOf"] or []):
        return False
    if "oneOf" in schema:
        matches = sum(_accepts(value, b, root, seen) for b in schema["oneOf"] or [])
        if matches != 1:
            return False
    if not all(_accepts(value, b, root, seen) for b in schema.get("allOf") or []):
        return False
    ref = schema.get("$ref")
    if isinstance(ref, str) and (ref, id(value)) not in seen:
        return _accepts(value, _resolve(ref, root), root, (*seen, (ref, id(value))))
    return True


def _same(value: Any, literal: Any) -> bool:
    """JSON equality: numbers by value, booleans apart from numbers, containers item by item."""
    if isinstance(value, bool) or isinstance(literal, bool):
        return isinstance(value, bool) and isinstance(literal, bool) and value is literal
    if isinstance(value, int | float) and isinstance(literal, int | float):
        return value == literal
    if isinstance(value, list) and isinstance(literal, list):
        return len(value) == len(literal) and all(map(_same, value, literal))
    if isinstance(value, dict) and isinstance(literal, dict):
        return value.keys() == literal.keys() and all(_same(value[k], literal[k]) for k in value)
    return type(value) is type(literal) and value == literal


# --- the Qwen3.5 and Nemotron 3 XML form ----------------------------------------------

_FUNCTION = re.compile(r"\s*<function=([^>]+)>(.*?)</function>\s*", re.DOTALL)
# The template writes a newline on each side of a value. The vLLM Qwen3 parser also
# accepts whitespace after ``=``, which Prime puts in the parameter name.
_PARAMETER = re.compile(r"<parameter=\s*([^>]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
# A value holding another parameter's opening tag cannot be told from two parameters,
# and one holding the ``<tool_call>`` token from a second call the model never closed.
# The text ``<function=`` is just text: a closing tag ends the match, so nothing nests.
_MARKUP = ("<tool_call>", "<parameter=")


def xml_arguments_as_written(
    calls: list[ParsedToolCall], tools, *, python_scalars: bool = True
) -> list[ParsedToolCall]:
    """Read each ``<function=...>`` call's arguments as the template wrote them.

    Prime's parse strips every value. It also marks a call ``invalid_json`` when a value
    is not the JSON its schema asks for. This function reads each value from the call's
    raw text with ``read_argument``. A complete call is ``ok``. It is ``unknown_tool``
    instead when ``tools`` are declared and none of them has the call's name.

    Set ``python_scalars`` to also accept booleans and null in their Python spelling, as
    Qwen3.5 and Nemotron 3 write them. Qwen3.6 and later write them as JSON.

    Only complete, unambiguous function and parameter blocks are read. A duplicate key,
    text between parameters, or a parameter's opening tag or a ``<tool_call>`` inside a
    value makes the call ``malformed_structure``, and the call is never read in part.
    """
    declared = tool_parameters(tools)
    return [_xml_call(call, declared, python_scalars) for call in calls]


def _xml_call(
    call: ParsedToolCall, declared: dict[str, Parameters], python_scalars: bool = True
) -> ParsedToolCall:
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
        arguments[name] = read_argument(
            text, schemas.get(name), root=root, python_scalars=python_scalars
        )
        offset = parameter.end()
    if body[offset:].strip():
        return malformed
    status = ToolCallParseStatus.OK
    if declared and call.name not in declared:
        status = ToolCallParseStatus.UNKNOWN_TOOL
    return replace(call, arguments=arguments, status=status)
