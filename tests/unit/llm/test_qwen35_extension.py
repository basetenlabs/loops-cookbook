"""Sanitized Qwen XML cases; no task data, downloads or sampler needed."""

from dataclasses import asdict

import pytest
from renderers.base import ParsedToolCall, ToolCallParseStatus

PARAMS = {"bash": {"command": {"type": "string"}, "options": {"type": "object"}}}


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
    from loops_cookbook.llm.renderer_extensions.qwen35 import _parse_call

    return _parse_call(candidate, PARAMS)


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


def test_declared_invalid_json_stays_invalid():
    candidate = call("options", "{broken")
    assert parse(candidate).status == "invalid_json"


def test_valid_typed_arguments_keep_prime_coercion():
    assert parse(call("options", '{"x":1}')).arguments == {"options": {"x": 1}}


@pytest.mark.parametrize("key", ["parameter=command", "unknown"])
def test_unknown_argument_name_is_preserved_for_tool_validation(key):
    result = parse(call(key, "echo never repair this"))
    assert result.status == "ok"
    assert result.arguments == {key: "echo never repair this"}
    assert "command" not in result.arguments


def test_no_argument_call_is_representable_not_invented():
    candidate = call()
    candidate.raw = "<function=bash></function>"
    assert parse(candidate).arguments == {}
    assert parse(candidate).status == "ok"
