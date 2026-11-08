"""Sanitized Qwen XML cases; no task data, downloads or sampler needed."""

from dataclasses import asdict

import pytest
from renderers.base import ParsedToolCall, ToolCallParseStatus

from loops_cookbook.llm.renderer_extensions.arguments import _xml_call

PARAMS = {"bash": ({"command": {"type": "string"}, "options": {"type": "object"}}, {})}


def call(key="command", value="echo ok"):
    return ParsedToolCall(
        raw=f"\n<function=bash>\n<parameter={key}>\n{value}\n</parameter>\n</function>\n",
        name="bash",
        arguments={key: value},
        token_span=(3, 42),
        status=ToolCallParseStatus.INVALID_JSON,
        id="saved-id",
    )


def parse(candidate):
    return _xml_call(candidate, PARAMS)


@pytest.mark.parametrize("key", ["command", "\ncommand", " \tcommand", "\r\ncommand"])
def test_parameter_header_whitespace_preserves_source_evidence(key):
    candidate = call(key, "printf 'a  b\\n'\ncat file")
    original = asdict(candidate)
    result = parse(candidate)
    assert result.status == "ok"
    assert result.arguments == {"command": "printf 'a  b\\n'\ncat file"}
    assert result.raw == candidate.raw and result.token_span == candidate.token_span
    assert result.id == "saved-id" and asdict(candidate) == original


@pytest.mark.parametrize(
    "raw",
    [
        "<function=bash><parameter=command>one</parameter><parameter=command>two</parameter></function>",
        "<function=bash><parameter=command>one</parameter><parameter=\ncommand>two</parameter></function>",
        "<function=bash><parameter=>one</parameter></function>",
        "<function=bash><parameter=command>one</function>",
        "<function=bash><parameter=command>one</parameter>ignored</function>",
        "<function=bash><parameter=command>one<parameter=other>two</parameter></function>",
        "<function=bash></function><function=bash></function>",
        "<function=bash><parameter=command>one</parameter></function>ignored",
    ],
)
def test_ambiguous_or_incomplete_structure_never_becomes_a_partial_call(raw):
    candidate = call()
    candidate.raw = raw
    assert parse(candidate).status == "malformed_structure"


def test_original_unclosed_status_is_not_recovered():
    candidate = call()
    candidate.status = ToolCallParseStatus.UNCLOSED_BLOCK
    assert parse(candidate) is candidate


def test_text_a_declared_type_cannot_read_reaches_the_tool():
    result = parse(call("options", "{broken"))
    assert result.status == "ok"
    assert result.arguments == {"options": "{broken"}


TYPED = {
    "s": {"type": "string"},
    "i": {"type": "integer"},
    "b": {"type": "boolean"},
    "n": {"type": ["integer", "null"]},
    "o": {"type": "object"},
    "a": {"type": "array"},
    "u": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
    "untyped": {"description": "says nothing of its type"},
}


@pytest.mark.parametrize(
    "key,text,value",
    [
        # A string comes back as written, whitespace and all.
        ("s", "  x = 1 \n", "  x = 1 \n"),
        ("s", "None", "None"),
        ("s", '"quoted"', '"quoted"'),
        # Another declared type is read as JSON, or as Jinja's string of a boolean or null.
        ("i", " 3 ", 3),
        ("b", "True", True),
        ("b", "false", False),
        ("n", "None", None),
        ("n", "null", None),
        ("o", '{"a": [1]}', {"a": [1]}),
        ("a", "[1, 20]", [1, 20]),
        ("u", "7", 7),
        ("u", "  later\n", "  later\n"),
        # Text that its type can't read, or that no type tells apart, stays the text.
        ("i", "30s", "30s"),
        ("a", "1, 20", "1, 20"),
        ("untyped", " 600 ", " 600 "),
        ("untyped", "None", "None"),
        ("undeclared", "600", "600"),
    ],
)
def test_each_value_reads_back_as_the_template_wrote_it(key, text, value):
    candidate = call(key, text)
    result = _xml_call(candidate, {"bash": (TYPED, {})})
    assert result.status == "ok"
    assert result.arguments == {key: value}
    assert type(result.arguments[key]) is type(value)


def test_valid_typed_arguments_keep_prime_coercion():
    assert parse(call("options", '{"x":1}')).arguments == {"options": {"x": 1}}


@pytest.mark.parametrize("key", ["parameter=command", "unknown"])
def test_unknown_argument_name_is_preserved_for_tool_validation(key):
    result = parse(call(key, "echo never repair this"))
    assert result.status == "ok"
    assert result.arguments == {key: "echo never repair this"}
    assert "command" not in result.arguments


def test_a_call_to_an_undeclared_tool_is_reported_with_its_arguments():
    candidate = call()
    candidate.name = "bsah"
    candidate.raw = candidate.raw.replace("function=bash", "function=bsah")
    result = parse(candidate)
    assert result.status == "unknown_tool"
    assert result.arguments == {"command": "echo ok"}
    # With no tools declared, any name is accepted.
    assert _xml_call(candidate, {}).status == "ok"


def test_a_function_tag_written_in_a_value_is_just_text():
    """Only another parameter's opening tag, or a second call's token, is ambiguous."""
    value = 'grep -n "<function=" agent.py'
    result = parse(call("command", value))
    assert result.status == "ok" and result.arguments == {"command": value}


def test_no_argument_call_is_representable_not_invented():
    candidate = call()
    candidate.raw = "<function=bash></function>"
    assert parse(candidate).arguments == {}
    assert parse(candidate).status == "ok"
