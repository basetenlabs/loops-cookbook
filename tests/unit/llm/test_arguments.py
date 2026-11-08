import json

import pytest

from loops_cookbook.llm.renderer_extensions.arguments import read_argument

STRING = {"type": "string"}
INTEGER = {"type": "integer"}
NUMBER = {"type": "number"}
BOOLEAN = {"type": "boolean"}
OPTIONAL_STRING = {"type": ["string", "null"]}
UNION = {"anyOf": [{"type": "integer"}, {"type": "string"}]}
ENUM = {"enum": ["low", 3]}
ARRAY = {"type": "array"}
REF = {"$ref": "#/$defs/thing"}
# These are the schemas pydantic writes for a ``str`` Enum, a nested model and an
# optional nested model.
DEFS = {
    "$defs": {
        "Mode": {"enum": ["read", "write"], "type": "string"},
        "Sub": {"type": "object", "properties": {"x": {"type": "integer"}}},
        "Loop": {"anyOf": [{"$ref": "#/$defs/Loop"}, {"type": "null"}]},
    }
}
MODE = {"$ref": "#/$defs/Mode"}
SUB = {"$ref": "#/$defs/Sub"}
OPTIONAL_SUB = {"anyOf": [{"$ref": "#/$defs/Sub"}, {"type": "null"}]}
LOOP = {"$ref": "#/$defs/Loop"}
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
        ("3", ENUM, 3),
        ('{"a": 1}', REF, {"a": 1}),
        # Null means a value was left out, for any schema except a plain string.
        ("null", INTEGER, None),
        ("null", OPTIONAL_STRING, None),
        # A JSON string, or JSON of a type the schema does not admit, stays the text.
        ('"quoted"', OPTIONAL_STRING, '"quoted"'),
        ('"quoted"', UNION, '"quoted"'),
        ("[]", OPTIONAL_STRING, "[]"),
        ("true", UNION, "true"),
        ("2.5", INTEGER, "2.5"),
        ("2.0", INTEGER, "2.0"),
        ("high", ENUM, "high"),
        # Text that the declared type cannot read stays the text.
        ("30s", INTEGER, "30s"),
        ("1, 20", ARRAY, "1, 20"),
        ("None", INTEGER, "None"),
        ("True", BOOLEAN, "True"),
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
        ("None", INTEGER, None),
        ("None", OPTIONAL_STRING, None),
        ("True", UNION, "True"),
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
        ("1", SUB, "1"),
        ('{"x": 1}', OPTIONAL_SUB, {"x": 1}),
        ("null", OPTIONAL_SUB, None),
        ("1", OPTIONAL_SUB, "1"),
        # A ref that leads nowhere, or back to itself, admits any type.
        ("1", REF, 1),
        ("1", LOOP, 1),
    ],
)
def test_a_ref_is_read_within_the_tools_parameters(text, schema, value):
    result = read_argument(text, schema, root=DEFS)
    assert json.dumps(result) == json.dumps(value)
