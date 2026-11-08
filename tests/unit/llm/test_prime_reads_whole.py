"""Only a renderer that reads a block whole keeps the message of a call it did not mark ok."""

import pytest
from renderers.base import ParsedResponse, ParsedToolCall, ToolCallParseStatus

from loops_cookbook.llm.prime import PrimeRenderer


class _Parser:
    """A Prime-like renderer whose parse reports one call with the given status."""

    def __init__(self, status, *, whole):
        self.status = status
        if whole:
            self.reads_calls_whole = True

    def parse_response(self, token_ids, *, tools=None, prompt_ids=None):
        call = ParsedToolCall(raw="", name="f", arguments={"a": "1"}, status=self.status)
        return ParsedResponse(content="", reasoning_content=None, tool_calls=[call])


@pytest.mark.parametrize(
    "status,whole,kept",
    [
        (ToolCallParseStatus.OK, False, True),
        (ToolCallParseStatus.OK, True, True),
        (ToolCallParseStatus.INVALID_JSON, False, False),
        (ToolCallParseStatus.INVALID_JSON, True, True),
        (ToolCallParseStatus.UNKNOWN_TOOL, False, False),
        (ToolCallParseStatus.UNKNOWN_TOOL, True, True),
        (ToolCallParseStatus.MALFORMED_STRUCTURE, True, False),
        (ToolCallParseStatus.UNCLOSED_BLOCK, True, False),
    ],
)
def test_the_message_is_kept_only_when_the_renderer_vouches_for_the_call(status, whole, kept):
    reply = PrimeRenderer(_Parser(status, whole=whole)).parse([1], [2], [])
    [call] = reply.tool_calls
    assert call.status == status
    assert (reply.native_message is not None) is kept
