import json

import pytest

from loops_cookbook.llm.renderer_extensions.arguments import read_argument, tool_parameters

STRING = {"type": "string"}
INTEGER = {"type": "integer"}
NUMBER = {"type": "number"}
BOOLEAN = {"type": "boolean"}
OPTIONAL_STRING = {"type": ["string", "null"]}
OPTIONAL_INTEGER = {"type": ["integer", "null"]}
BOOL_OR_STRING = {"anyOf": [{"type": "boolean"}, {"type": "string"}]}
STRING_ENUM_WITH_A_NUMBER = {"type": "string", "enum": ["3", 3]}
NUMBER_AND_INTEGER = {"allOf": [{"type": "number"}, {"type": "integer"}]}
NUMBER_OR_STRING_NARROWED = {
    "type": ["number", "string"],
    "anyOf": [{"type": "integer"}, {"type": "string"}],
}
OVERLAPPING_ONE_OF = {"oneOf": [{"type": "integer"}, {"type": "number"}, {"type": "string"}]}
CONST_THREE = {"const": 3.0}
ENUM_OF_ONE_LIST = {"enum": [[1]]}
# What pydantic writes for a discriminated union: oneOf branches told apart by a const.
TAGGED_UNION = {
    "oneOf": [
        {
            "type": "object",
            "properties": {"kind": {"const": "read"}, "path": {"type": "string"}},
            "required": ["kind"],
        },
        {
            "type": "object",
            "properties": {"kind": {"const": "write"}, "path": {"type": "string"}},
            "required": ["kind"],
        },
    ]
}
ARRAYS_BY_ITEM = {
    "oneOf": [
        {"type": "array", "items": {"type": "integer"}},
        {"type": "array", "items": {"type": "string"}},
    ]
}
CLOSED_OBJECT = {
    "type": "object",
    "properties": {"x": {"type": "integer"}},
    "additionalProperties": False,
}
ROOT_REF = {"$ref": "#"}
ESCAPED_REF = {"$ref": "#/$defs/file%20text"}
# A recursive union, as pydantic writes a tree: a leaf, or a branch of children.
TREE_ROOT = {
    "$defs": {
        "Leaf": {"type": "object", "properties": {"kind": {"const": "leaf"}}, "required": ["kind"]},
        "Branch": {
            "type": "object",
            "properties": {
                "kind": {"const": "branch"},
                "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
            },
            "required": ["kind", "children"],
        },
        "Node": {"oneOf": [{"$ref": "#/$defs/Leaf"}, {"$ref": "#/$defs/Branch"}]},
    }
}
TREE = {"$ref": "#/$defs/Node"}
UNION = {"anyOf": [{"type": "integer"}, {"type": "string"}]}
ENUM = {"enum": ["low", 3]}
ARRAY = {"type": "array"}
REF = {"$ref": "#/$defs/thing"}
# These are the schemas pydantic writes for a ``str`` Enum, a nested model and an
# optional nested model.
DEFS = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "$defs": {
        "file text": {"type": "string"},
        "Mode": {"enum": ["read", "write"], "type": "string"},
        "Sub": {"type": "object", "properties": {"x": {"type": "integer"}}},
        "Loop": {"anyOf": [{"$ref": "#/$defs/Loop"}, {"type": "null"}]},
        "Choice": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
    },
}
MODE = {"$ref": "#/$defs/Mode"}
SUB = {"$ref": "#/$defs/Sub"}
OPTIONAL_SUB = {"anyOf": [{"$ref": "#/$defs/Sub"}, {"type": "null"}]}
LOOP = {"$ref": "#/$defs/Loop"}
BRANCH = {"$ref": "#/$defs/Choice/anyOf/0"}
NARROWED_REF = {"type": "string", "$ref": "#/$defs/Choice"}
ALL_OF = {"allOf": [{"$ref": "#/$defs/Mode"}], "default": "read"}
UNTYPED = {"description": "says nothing of its type"}


@pytest.mark.parametrize(
    "text,schema,value",
    [
        # A plain string is the text, whatever it looks like.
        ("  x = 1 \n", STRING, "  x = 1 \n"),
        ('"quoted"', STRING, '"quoted"'),
        ("null", STRING, "null"),
        ("7", STRING, "7"),
        ("", STRING, ""),
        # Another declared type is read as JSON when the schema admits the JSON's type.
        (" 3 ", INTEGER, 3),
        ("2.5", NUMBER, 2.5),
        ("3", NUMBER, 3),
        ("true", BOOLEAN, True),
        ("[1, 2]", ARRAY, [1, 2]),
        ("7", UNION, 7),
        # No template pads a non-string: padded text the schema accepts as a string is one.
        (" 7 ", UNION, " 7 "),
        (" null\n", OPTIONAL_STRING, " null\n"),
        (" 7 ", INTEGER, 7),
        ("3", ENUM, 3),
        ('{"a": 1}', REF, {"a": 1}),
        # Null means a value was left out, for any schema except a plain string.
        ("null", OPTIONAL_INTEGER, None),
        ("null", OPTIONAL_STRING, None),
        # A JSON string stays the text; so does JSON the schema refuses where it admits a string.
        ('"quoted"', OPTIONAL_STRING, '"quoted"'),
        ('"quoted"', UNION, '"quoted"'),
        ("[]", OPTIONAL_STRING, "[]"),
        ("true", UNION, "true"),
        # JSON the schema refuses, where no string fits, is still that value for the tool.
        ("2.5", INTEGER, 2.5),
        ("2.0", INTEGER, 2.0),  # an integer-valued number is an integer
        # Every constraint holds at once; refused JSON is text only where a string fits.
        ("null", INTEGER, None),
        ("null", UNION, "null"),
        ("3", STRING_ENUM_WITH_A_NUMBER, "3"),
        ("true", NUMBER_AND_INTEGER, True),
        ("3", NUMBER_AND_INTEGER, 3),
        ("3", NUMBER_OR_STRING_NARROWED, 3),
        ("3.0", INTEGER, 3.0),
        # oneOf wants exactly one branch; a literal is equal by JSON value, not Python type.
        ("7", OVERLAPPING_ONE_OF, "7"),
        ("3", CONST_THREE, 3),
        ("[true]", ENUM_OF_ONE_LIST, [True]),
        ("[1]", ENUM_OF_ONE_LIST, [1]),
        # Nested constraints count when telling branches apart.
        ('{"kind": "read", "path": "a"}', TAGGED_UNION, {"kind": "read", "path": "a"}),
        ('{"kind": "move"}', TAGGED_UNION, {"kind": "move"}),
        ('{"path": "a"}', TAGGED_UNION, {"path": "a"}),
        ("[1, 2]", ARRAYS_BY_ITEM, [1, 2]),
        ('["a"]', ARRAYS_BY_ITEM, ["a"]),
        ('[1, "a"]', ARRAYS_BY_ITEM, [1, "a"]),
        ('{"x": 1}', CLOSED_OBJECT, {"x": 1}),
        ('{"x": 1, "y": 2}', CLOSED_OBJECT, {"x": 1, "y": 2}),
        ("high", ENUM, "high"),
        # Text that the declared type cannot read stays the text.
        ("30s", INTEGER, "30s"),
        ("1, 20", ARRAY, "1, 20"),
        # Python spellings under a JSON template: the value where no string fits, else text.
        ("None", INTEGER, None),
        ("True", BOOLEAN, True),
        ("None", OPTIONAL_INTEGER, None),
        ("True", BOOL_OR_STRING, "True"),
        ("None", OPTIONAL_STRING, "None"),
        ("True", UNION, "True"),
        # Nothing says how an untyped value was written, so it stays the text.
        (" 600 ", UNTYPED, " 600 "),
        ("600", None, "600"),
        ("null", None, "null"),
    ],
)
def test_each_value_reads_back_as_written(text, schema, value):
    result = read_argument(text, schema)
    assert json.dumps(result) == json.dumps(value)


@pytest.mark.parametrize(
    "text,schema,value",
    [
        ("True", BOOLEAN, True),
        ("False", BOOLEAN, False),
        ("None", OPTIONAL_INTEGER, None),
        ("None", OPTIONAL_STRING, None),
        ("  None\n", OPTIONAL_STRING, "  None\n"),
        ("True", BOOL_OR_STRING, True),
        ("  True\n", BOOL_OR_STRING, "  True\n"),
        # These templates write a boolean as True and null as None: the JSON spellings
        # are strings wherever the schema admits one, and still read for a plain type.
        ("true", BOOL_OR_STRING, "true"),
        ("false", BOOL_OR_STRING, "false"),
        ("null", OPTIONAL_STRING, "null"),
        ("true", BOOLEAN, True),
        ("null", OPTIONAL_INTEGER, None),
        ("True", UNION, "True"),
        ("None", UNION, "None"),
        ("None", INTEGER, None),
        ("None", STRING, "None"),
        ("True", UNTYPED, "True"),
    ],
)
def test_python_spellings_read_where_the_template_writes_them(text, schema, value):
    result = read_argument(text, schema, python_scalars=True)
    assert json.dumps(result) == json.dumps(value)


@pytest.mark.parametrize(
    "text,schema,value",
    [
        # A ref is followed within the tool's parameters, so a string enum stays a string.
        ("read", MODE, "read"),
        ("1", MODE, "1"),
        ("null", MODE, "null"),
        ("1", ALL_OF, "1"),
        ('{"x": 1}', SUB, {"x": 1}),
        ("1", SUB, 1),
        ('{"x": 1}', OPTIONAL_SUB, {"x": 1}),
        # A pointer may step into an array; a ref and its siblings all hold.
        ("123", BRANCH, "123"),
        (" 7\n", NARROWED_REF, " 7\n"),
        # "#" is the parameters object itself: an object, so a number is the number for the tool.
        (" 123\n", ROOT_REF, 123),
        ('{"path": "a"}', ROOT_REF, {"path": "a"}),
        ("123", ESCAPED_REF, "123"),
        ("null", OPTIONAL_SUB, None),
        ("1", OPTIONAL_SUB, 1),
        # A ref that leads nowhere, or back to itself, admits any type.
        ("1", REF, 1),
        ("1", LOOP, 1),
    ],
)
def test_a_ref_is_read_within_the_tools_parameters(text, schema, value):
    result = read_argument(text, schema, root=DEFS)
    assert json.dumps(result) == json.dumps(value)


# --- the invariant, over schemas built from every constraint and values of every look -----

DEFS_ROOT = {
    "$defs": {
        "E": {"enum": ["read", "write"], "type": "string"},
        "S": {"type": "object", "properties": {"x": {"type": "integer"}}},
        "N": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
    }
}
LEAVES = [
    *({"type": t} for t in ("string", "integer", "number", "boolean", "null", "array", "object")),
    {"type": ["string", "null"]},
    {"enum": ["a", 1, True, None]},
    {"enum": ["3", "x"]},
    {"const": "k"},
    {"const": 2},
    {"$ref": "#/$defs/E"},
    {"$ref": "#/$defs/S"},
    {"$ref": "#/$defs/N"},
    {"$ref": "#/$defs/missing"},
    {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
    {"type": "array", "items": {"type": "string"}},
]
COMBINED = [
    *LEAVES,
    *({"anyOf": [a, b]} for a in LEAVES[:7] for b in LEAVES[7:]),
    *({"allOf": [a, b]} for a in LEAVES[:7] for b in LEAVES[7:]),
    *({**a, "enum": ["3", 3, None]} for a in LEAVES[:7]),
    *({**a, "const": "3"} for a in LEAVES[:7]),
]
LOOKS = [
    '{"x": 1}',
    '{"x": "1"}',
    '["a", 1]',
    "7",
    " 7 ",
    "-3.5",
    "1e3",
    "true",
    "false",
    "null",
    "None",
    "True",
    '"q"',
    "[1]",
    '{"x": 1}',
    "{}",
    "",
    "a b",
    "3",
    "7x",
    "read",
    "k",
]


def _json_equal(a, b) -> bool:
    """JSON equality: 3 equals 3.0, True never equals 1, containers item by item."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _satisfies(value, schema) -> bool:
    """A plain JSON Schema check of one value, written apart from the reader's type sets."""
    kind = {
        type(None): "null",
        bool: "boolean",
        int: "integer",
        float: "number",
        str: "string",
        list: "array",
        dict: "object",
    }[type(value)]
    declared = schema.get("type")
    if isinstance(declared, str):
        declared = [declared]
    if (
        declared
        and kind not in declared
        and not (kind == "integer" and "number" in declared)
        and not (kind == "number" and "integer" in declared and float(value).is_integer())
    ):
        return False
    if "enum" in schema and not any(_json_equal(v, value) for v in schema["enum"]):
        return False
    if "const" in schema and not _json_equal(schema["const"], value):
        return False
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        if any(k not in value for k in schema.get("required") or []):
            return False
        if any(k in props and not _satisfies(v, props[k]) for k, v in value.items()):
            return False
        if schema.get("additionalProperties") is False and any(k not in props for k in value):
            return False
    if (
        isinstance(value, list)
        and isinstance(schema.get("items"), dict)
        and not all(_satisfies(v, schema["items"]) for v in value)
    ):
        return False
    if "anyOf" in schema and not any(_satisfies(value, b) for b in schema["anyOf"]):
        return False
    if "oneOf" in schema and sum(_satisfies(value, b) for b in schema["oneOf"]) != 1:
        return False
    if "allOf" in schema and not all(_satisfies(value, b) for b in schema["allOf"]):
        return False
    if "$ref" in schema:
        target = DEFS_ROOT["$defs"].get(schema["$ref"].rsplit("/", 1)[-1])
        if target is not None and not _satisfies(value, target):
            return False
    return True


@pytest.mark.parametrize("python_scalars", [False, True])
@pytest.mark.parametrize("schema", COMBINED, ids=[json.dumps(s, sort_keys=True) for s in COMBINED])
def _could_be_string(schema) -> bool:
    """A plain reading of whether the schema admits a string, apart from the reader."""
    declared = schema.get("type")
    if isinstance(declared, str):
        declared = [declared]
    if declared and "string" not in declared:
        return False
    if "enum" in schema and not any(isinstance(v, str) for v in schema["enum"]):
        return False
    if "const" in schema and not isinstance(schema["const"], str):
        return False
    if "anyOf" in schema and not any(_could_be_string(b) for b in schema["anyOf"]):
        return False
    if "oneOf" in schema and not any(_could_be_string(b) for b in schema["oneOf"]):
        return False
    if "allOf" in schema and not all(_could_be_string(b) for b in schema["allOf"]):
        return False
    if "$ref" in schema:
        target = DEFS_ROOT["$defs"].get(schema["$ref"].rsplit("/", 1)[-1])
        if target is not None and not _could_be_string(target):
            return False
    return True


def _decoded(text, python_scalars, schema):
    try:
        return json.loads(text)
    except ValueError:
        spellings = {"True": True, "False": False, "None": None}
        if python_scalars or not _could_be_string(schema):
            return spellings.get(text.strip(), text)
        return text


@pytest.mark.parametrize("python_scalars", [False, True])
@pytest.mark.parametrize("schema", COMBINED, ids=[json.dumps(s, sort_keys=True) for s in COMBINED])
def test_the_reader_returns_what_the_template_wrote(schema, python_scalars):
    """Whatever the schema, a value is the text as written or the JSON it decodes to,
    and the schema only says which. Text that is not JSON, or decodes to a string, is
    text. JSON the schema accepts is the value. JSON it refuses is the text where the
    schema admits a string, since the template writes a string as it is, and the value
    where it admits none, since the template then wrote JSON and the tool judges the
    content. A Python spelling reads the same way where no string fits. (Where the schema accepts both readings, as ``7`` for an
    integer-or-string union, the template wrote them alike and JSON is the documented
    pick; padded text it accepts as a string is the string.)"""
    for text in LOOKS:
        result = read_argument(text, schema, root=DEFS_ROOT, python_scalars=python_scalars)
        decoded = _decoded(text, python_scalars, schema)
        if isinstance(decoded, str):
            assert result == text, (text, result)
        elif _satisfies(decoded, schema):
            padded = text != text.strip() or (
                python_scalars and text.strip() in ("true", "false", "null")
            )
            if padded and _satisfies(text, schema):
                assert result == text, (text, result)
            else:
                assert type(result) is not str and _json_equal(result, decoded), (text, result)
        elif _could_be_string(schema):
            assert result == text, (text, result)
        else:
            assert type(result) is not str and _json_equal(result, decoded), (text, result)


@pytest.mark.parametrize(
    "text,schema,value",
    [
        (
            '[{"old": "a", "new": "b"}]',
            {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "old": {}, "new": {}},
                    "required": ["path"],
                },
            },
            [{"old": "a", "new": "b"}],
        ),
        ('{"x": "1"}', {"type": "object", "properties": {"x": {"type": "integer"}}}, {"x": "1"}),
        ("7.5", {"type": "integer"}, 7.5),
        ("true", {"type": "integer"}, True),
        ("7", {"enum": ["a", "b"]}, "7"),
        ("7", {"anyOf": [{"type": "boolean"}, {"type": "string"}]}, "7"),
        ("30s", {"type": "integer"}, "30s"),
        (" [1] ", {"type": "array", "items": {"type": "string"}}, [1]),
    ],
)
def test_json_the_schema_refuses_is_the_value_where_no_string_fits(text, schema, value):
    """The template wrote JSON for a non-string parameter whatever the content, so the
    tool receives that value and refuses it itself. Where a string fits the schema the
    text is the string, and text that is not JSON is always text."""
    assert read_argument(text, schema, root=DEFS_ROOT) == value
    assert type(read_argument(text, schema, root=DEFS_ROOT)) is type(value)


@pytest.mark.parametrize(
    "text,value",
    [
        ('{"kind": "leaf"}', {"kind": "leaf"}),
        (
            '{"kind": "branch", "children": [{"kind": "leaf"}, {"kind": "branch", "children": []}]}',
            {"kind": "branch", "children": [{"kind": "leaf"}, {"kind": "branch", "children": []}]},
        ),
        (
            '{"kind": "branch", "children": [{"kind": "twig"}]}',
            {"kind": "branch", "children": [{"kind": "twig"}]},
        ),
        ("7", 7),
    ],
)
def test_a_recursive_union_reads_each_child_by_its_own_branch(text, value):
    """A ref recurs through child values: the cycle guard is per value, so a leaf inside
    a branch is matched against both branches afresh and exactly one accepts it."""
    assert json.dumps(read_argument(text, TREE, root=TREE_ROOT)) == json.dumps(value)


def test_a_parameters_object_that_is_a_ref_declares_the_referenced_properties():
    """The properties live in the definition the root refers to; the refs inside them
    still point into the parameters object."""
    tool = {
        "name": "count",
        "parameters": {
            "$ref": "#/$defs/Args",
            "$defs": {
                "Args": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}, "mode": {"$ref": "#/$defs/Mode"}},
                },
                "Mode": {"enum": ["read", "write"], "type": "string"},
            },
        },
    }
    [(schemas, root)] = tool_parameters([tool]).values()
    assert set(schemas) == {"count", "mode"} and root is tool["parameters"]
    assert read_argument("3", schemas["count"], root=root) == 3
    assert read_argument("1", schemas["mode"], root=root) == "1"
