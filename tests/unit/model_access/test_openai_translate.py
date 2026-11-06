"""A captured generation reads as a Chat Completions response, as on ``develop``."""

import json
from dataclasses import replace

import pytest
from renderers.base import ParsedToolCall, ToolCallParseStatus
from support.builders import make_sample

from loops_cookbook.llm.client import LLMResponse
from loops_cookbook.llm.rendering import ParsedReply
from loops_cookbook.model_access.openai.policy import admit
from loops_cookbook.model_access.openai.schema.models import CreateChatCompletionRequest
from loops_cookbook.model_access.openai.translate import (
    InvalidOutput,
    Returned,
    completion,
    renderer_input,
)


def _reply(*, content="answer", reasoning=None, calls=(), native=True, stop="stop", complete=None):
    native_message = {"role": "assistant", "content": content} if native else None
    if native_message is not None and calls:
        native_message["tool_calls"] = [
            {"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in calls
        ]
    return LLMResponse(
        sample=make_sample("s1", (1, 2), (3, 4), (-0.1, -0.2), stop_reason=stop),
        parsed=ParsedReply(content, reasoning, tuple(calls), complete, native_message),
    )


def _call(name="lookup", arguments=None, status=ToolCallParseStatus.OK, id=None):
    return ParsedToolCall(raw="", name=name, arguments=arguments or {"q": 1}, status=status, id=id)


def _complete(reply, *, tools_offered=True):
    return completion(reply, model="m", created=1, tools_offered=tools_offered)


def test_a_plain_answer_stops():
    done = _complete(_reply())
    body = json.loads(done.body())
    assert (
        body["id"] == "chatcmpl-s1" and body["model"] == "m" and body["object"] == "chat.completion"
    )
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"] == {"role": "assistant", "content": "answer"}
    assert body["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}


def test_parsed_tool_calls_finish_with_tool_calls_and_get_derived_ids():
    done = _complete(_reply(calls=[_call(), _call("other")]))
    message = json.loads(done.body())["choices"][0]["message"]
    assert done.finish_reason == "tool_calls"
    assert [call["id"] for call in message["tool_calls"]] == ["call_s1_0", "call_s1_1"]
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"q": 1}


def test_a_native_tool_call_id_is_kept():
    done = _complete(_reply(calls=[_call(id="native-7")]))
    assert done.message.tool_calls[0].id == "native-7"


def test_a_tool_call_that_did_not_parse_is_the_models_invalid_output():
    broken = _call(status=ToolCallParseStatus.INVALID_JSON, arguments="{oops")
    done = _complete(_reply(content="partial", calls=[_call(), broken], native=False))
    assert isinstance(done, InvalidOutput) and done.code == "tool_call_parse_error"
    error = json.loads(done.body())["error"]
    assert error["type"] == "model_output_invalid" and error["code"] == done.code
    assert error["reply"] == {
        "id": "chatcmpl-s1",
        "content": "partial",
        "reasoning_content": None,
        "tool_calls": [
            {
                "id": "call_s1_0",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q": 1}'},
                "raw": "",
                "status": "ok",
            },
            {  # the arguments as written, unparsed
                "id": "call_s1_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{oops"},
                "raw": "",
                "status": "invalid_json",
            },
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    }


def test_a_generation_cut_by_its_limit_or_inside_reasoning_is_length():
    assert _complete(_reply(stop="length")).finish_reason == "length"
    cut = _reply(content="", reasoning="still thinking", complete=False)
    done = _complete(cut)
    assert done.finish_reason == "length"
    assert done.message.reasoning_content == "still thinking" and done.message.content == ""


@pytest.mark.parametrize(
    ("status", "stop"),
    [(ToolCallParseStatus.UNCLOSED_BLOCK, "stop"), (ToolCallParseStatus.INVALID_JSON, "length")],
    ids=["unclosed", "cut_by_limit"],
)
def test_a_tool_call_cut_short_is_invalid_output_that_says_so(status, stop):
    done = _complete(_reply(calls=[_call(status=status)], native=False, stop=stop))
    assert isinstance(done, InvalidOutput) and done.code == "tool_call_truncated"


def test_invalid_output_carries_the_generations_usage():
    """An agent tracking its context counts the call that ran out of room."""
    cut = _reply(
        calls=[_call(status=ToolCallParseStatus.INVALID_JSON)], native=False, stop="length"
    )
    done = _complete(cut)
    assert isinstance(done, InvalidOutput)
    usage = json.loads(done.body())["error"]["reply"]["usage"]
    assert usage == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}


def test_usage_counts_the_reasoning_tokens_a_renderer_found():
    reply = replace(_reply(reasoning="because"), reasoning_tokens=1)
    body = json.loads(completion(reply, model="m", created=1, tools_offered=True).body())
    assert body["usage"]["completion_tokens_details"]["reasoning_tokens"] == 1
    cut = replace(
        _reply(calls=[_call(status=ToolCallParseStatus.INVALID_JSON)], native=False, stop="length"),
        reasoning_tokens=2,
    )
    done = completion(cut, model="m", created=1, tools_offered=True)
    assert isinstance(done, InvalidOutput)
    usage = json.loads(done.body())["error"]["reply"]["usage"]
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 2


def test_reasoning_is_returned_apart_from_the_answer():
    done = _complete(_reply(reasoning="because"))
    message = json.loads(done.body())["choices"][0]["message"]
    assert message["reasoning_content"] == "because" and message["content"] == "answer"


def test_the_stream_carries_the_same_response_and_ends_with_done():
    done = _complete(_reply(reasoning="why", calls=[_call()]))
    events = done.events(include_usage=True)
    assert events[-1] == b"data: [DONE]\n\n"
    chunks = [json.loads(event.removeprefix(b"data: ")) for event in events[:-1]]
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks if chunk["choices"]]
    assert deltas[0] == {"role": "assistant"}
    assert {"reasoning_content": "why"} in deltas and {"content": "answer"} in deltas
    call = next(d["tool_calls"][0] for d in deltas if "tool_calls" in d)
    assert call["id"] == "call_s1_0" and call["index"] == 0
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["total_tokens"] == 4
    assert {chunk["id"] for chunk in chunks} == {"chatcmpl-s1"}


def test_the_stream_has_no_usage_chunk_unless_asked():
    events = _complete(_reply()).events(include_usage=False)
    assert all(b'"usage"' not in event for event in events)


def _returned(done, native):
    """What the gateway remembers of a returned completion, and the message an agent
    echoes back for it, with an empty answer as null as the official client sends it."""
    returned = Returned()
    returned.remember(done.message, native)
    echo = json.loads(done.message.model_dump_json(exclude_none=True, serialize_as_any=True))
    echo["content"] = echo["content"] or None
    return returned, echo


def _request(messages):
    return admit(
        CreateChatCompletionRequest.model_validate({"model": "m", "messages": messages}),
        model="m",
    )


def test_an_echoed_assistant_message_renders_as_the_native_message_it_was():
    done = _complete(_reply(reasoning="why", calls=[_call()]))
    native = {"role": "assistant", "content": "answer", "marker": "native"}
    returned, echo = _returned(done, native)
    variants = [
        echo,
        {k: v for k, v in echo.items() if k != "reasoning_content"},  # omitted reasoning
        {
            **echo,
            "tool_calls": [
                {
                    **echo["tool_calls"][0],
                    "function": {"name": "lookup", "arguments": '{ "q" : 1 }'},  # reformatted
                }
            ],
        },
    ]
    for message in variants:
        messages, _ = renderer_input(_request([message]), returned)
        assert messages == [native]


def test_an_edited_assistant_message_is_rendered_from_the_wire():
    done = _complete(_reply())
    returned = Returned()
    returned.remember(done.message, {"role": "assistant", "content": "answer", "marker": "native"})
    messages, _ = renderer_input(_request([{"role": "assistant", "content": "edited"}]), returned)
    assert messages == [{"role": "assistant", "content": "edited"}]


def test_content_null_equals_an_empty_answer():
    done = _complete(_reply(content="", calls=[_call()]))
    returned, echo = _returned(done, {"marker": "native"})
    assert echo["content"] is None
    assert renderer_input(_request([echo]), returned)[0] == [{"marker": "native"}]


@pytest.mark.parametrize("stop", ["stop", "length"])
def test_a_call_written_when_no_tools_were_offered_is_not_returned(stop):
    done = _complete(_reply(calls=[_call()], stop=stop), tools_offered=False)
    message = json.loads(done.body())["choices"][0]["message"]
    assert "tool_calls" not in message and done.message.tool_calls is None
    assert done.finish_reason == stop


def test_an_echo_whose_reasoning_differs_is_rendered_from_the_wire():
    done = _complete(_reply(reasoning="why"))
    native = {"role": "assistant", "content": "answer", "marker": "native"}
    returned, echo = _returned(done, native)
    echo["reasoning_content"] = "a different chain of thought"
    messages, _ = renderer_input(_request([echo]), returned)
    assert messages != [native] and "marker" not in messages[0]
