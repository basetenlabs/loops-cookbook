"""The capability matrix covers the reachable schema, and the policy enforces it."""

import base64
import typing

import pytest
from pydantic import BaseModel, RootModel

from loops_cookbook.llm.client import UnsupportedParameterError
from loops_cookbook.model_access.openai import policy
from loops_cookbook.model_access.openai.policy import (
    MATRIX,
    OPAQUE,
    InvalidRequestError,
    ModelNotFoundError,
    Support,
    admit,
)
from loops_cookbook.model_access.openai.schema import models
from loops_cookbook.model_access.openai.schema.models import CreateChatCompletionRequest
from loops_cookbook.model_access.openai.translate import Returned, image_bytes, renderer_input

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 8
EXTENSIONS = {
    ("CreateChatCompletionRequest", "chat_template_kwargs"),
    ("ChatCompletionRequestAssistantMessage", "reasoning_content"),
}


def _models(annotation: object) -> list[type[BaseModel]]:
    """The generated models an annotation can hold, through unions, lists, and roots."""
    found: list[type[BaseModel]] = []
    if isinstance(annotation, type) and issubclass(annotation, RootModel):
        return _models(annotation.model_fields["root"].annotation)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    for argument in typing.get_args(annotation):
        found.extend(_models(argument))
    return found


def _reachable() -> dict[str, type[BaseModel]]:
    """Every model a request can reach through fields the matrix does not reject."""
    reached: dict[str, type[BaseModel]] = {}
    pending: list[type[BaseModel]] = [CreateChatCompletionRequest]
    while pending:
        model = pending.pop()
        if model.__name__ in reached or model.__name__ in OPAQUE:
            continue
        reached[model.__name__] = model
        table = MATRIX.get(model.__name__, {})
        for name, info in model.model_fields.items():
            capability = table.get(name)
            if capability is not None and capability.support is Support.REJECTED:
                continue
            pending.extend(_models(info.annotation))
    return reached


def test_the_matrix_covers_every_field_the_request_schema_can_carry():
    reached = _reachable()
    missing = {
        (name, field)
        for name, model in reached.items()
        for field in model.model_fields
        if field not in MATRIX.get(name, {})
    }
    assert not missing
    listed = {(name, field) for name, table in MATRIX.items() for field in table}
    real = {(name, field) for name, model in reached.items() for field in model.model_fields}
    rejected_models = {
        name
        for name, table in MATRIX.items()
        if all(c.support is Support.REJECTED for c in table.values())
    }
    stale = {entry for entry in listed - real - EXTENSIONS if entry[0] not in rejected_models}
    assert not stale  # the matrix names only fields the pinned schema has


def _request(**fields):
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], **fields}
    return CreateChatCompletionRequest.model_validate(body)


# One schema-valid example per rejected top-level field.
REJECTED_EXAMPLES = {
    "logprobs": True,
    "top_logprobs": 2,
    "audio": {"voice": "alloy", "format": "wav"},
    "function_call": "auto",
    "functions": [{"name": "f"}],
    "metadata": {"k": "v"},
    "user": "u",
    "safety_identifier": "s",
    "prompt_cache_key": "k",
    "prompt_cache_retention": "24h",
    "prompt_cache_options": {"mode": "implicit"},
    "moderation": {"model": "omni-moderation-latest"},
    "prediction": {"type": "content", "content": "x"},
    "verbosity": "low",
    "web_search_options": {},
}


def test_every_rejected_request_field_has_an_example():
    rejected = {
        name
        for name, capability in MATRIX["CreateChatCompletionRequest"].items()
        if capability.support is Support.REJECTED
    }
    assert rejected == set(REJECTED_EXAMPLES)


@pytest.mark.parametrize("name", sorted(REJECTED_EXAMPLES))
def test_a_rejected_field_fails_naming_itself(name):
    try:
        request = _request(**{name: REJECTED_EXAMPLES[name]})
    except Exception as error:  # noqa: BLE001 - the example must be schema-valid
        pytest.fail(f"example for {name} is not schema-valid: {error}")
    with pytest.raises(UnsupportedParameterError) as raised:
        admit(request, model="m")
    assert raised.value.param == name


@pytest.mark.parametrize("name", ["logprobs", "top_logprobs"])
def test_logprobs_are_rejected_even_when_false_or_zero(name):
    with pytest.raises(UnsupportedParameterError) as raised:
        admit(_request(**{name: False if name == "logprobs" else 0}), model="m")
    assert raised.value.param == name


def test_an_explicit_null_is_omitted():
    admitted = admit(_request(logprobs=None, n=None, stop=None, user=None), model="m")
    assert admitted.sampling.stop is None


@pytest.mark.parametrize(
    ("name", "default", "other"),
    [
        ("n", 1, 2),
        ("parallel_tool_calls", True, False),
        ("modalities", ["text"], ["text", "audio"]),
        ("store", False, True),
        ("service_tier", "auto", "flex"),
        ("frequency_penalty", 0, 0.5),
        ("presence_penalty", 0, 0.5),
        ("logit_bias", {}, {"1": 5}),
        ("chat_template_kwargs", {}, {"enable_thinking": False}),
    ],
)
def test_a_default_only_field_accepts_only_its_no_op_value(name, default, other):
    admit(_request(**{name: default}), model="m")
    with pytest.raises(UnsupportedParameterError) as raised:
        admit(_request(**{name: other}), model="m")
    assert raised.value.param == name


def test_an_unknown_field_is_an_unsupported_parameter():
    with pytest.raises(UnsupportedParameterError) as raised:
        admit(_request(best_of=2), model="m")
    assert raised.value.param == "best_of"


def test_a_nested_unknown_field_is_named_by_its_path():
    request = _request(messages=[{"role": "user", "content": "hi", "cache": True}])
    with pytest.raises(UnsupportedParameterError) as raised:
        admit(request, model="m")
    assert raised.value.param == "messages[0].cache"


def test_the_model_must_be_the_advertised_one():
    with pytest.raises(ModelNotFoundError):
        admit(_request(), model="other")


def test_max_tokens_and_max_completion_tokens_are_exclusive_aliases():
    assert admit(_request(max_completion_tokens=7), model="m").sampling.max_tokens == 7
    assert admit(_request(max_tokens=5), model="m").sampling.max_tokens == 5
    with pytest.raises(UnsupportedParameterError):
        admit(_request(max_tokens=5, max_completion_tokens=7), model="m")
    with pytest.raises(InvalidRequestError):
        admit(_request(max_tokens=0), model="m")


def test_only_omitted_auto_or_none_tool_choice_is_accepted():
    tools = [{"type": "function", "function": {"name": "f"}}]
    assert admit(_request(tools=tools), model="m").offer_tools
    assert not admit(_request(tools=tools, tool_choice="none"), model="m").offer_tools
    for choice in ("required", {"type": "function", "function": {"name": "f"}}):
        with pytest.raises(UnsupportedParameterError) as raised:
            admit(_request(tools=tools, tool_choice=choice), model="m")
        assert raised.value.param == "tool_choice"


def test_strict_function_generation_is_rejected():
    tools = [{"type": "function", "function": {"name": "f", "strict": True}}]
    with pytest.raises(UnsupportedParameterError) as raised:
        admit(_request(tools=tools), model="m")
    assert raised.value.param == "tools[0].function.strict"


def test_stream_options_require_a_stream():
    with pytest.raises(InvalidRequestError):
        admit(_request(stream_options={"include_usage": True}), model="m")
    admitted = admit(_request(stream=True, stream_options={"include_usage": True}), model="m")
    assert admitted.stream and admitted.include_usage


SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


@pytest.mark.parametrize(
    ("response_format", "constraint"),
    [
        ({"type": "text"}, None),
        ({"type": "json_object"}, {"json_object": True}),
        (
            {"type": "json_schema", "json_schema": {"name": "a", "schema": SCHEMA}},
            {"json": SCHEMA},
        ),
        (
            {"type": "json_schema", "json_schema": {"name": "a", "schema": SCHEMA, "strict": True}},
            {"json": SCHEMA},
        ),
        (
            {
                "type": "json_schema",
                "json_schema": {"name": "a", "schema": SCHEMA, "strict": False},
            },
            {"json": SCHEMA},
        ),
    ],
)
def test_a_response_format_becomes_the_samplers_constraint(response_format, constraint):
    admitted = admit(_request(response_format=response_format), model="m")
    assert admitted.sampling.structured_outputs == constraint


def test_a_constrained_reply_cannot_also_offer_tools():
    """The constraint holds from the first token, so no tool call could ever be sampled."""
    tools = [{"type": "function", "function": {"name": "f"}}]
    with pytest.raises(UnsupportedParameterError) as raised:
        admit(_request(tools=tools, response_format={"type": "json_object"}), model="m")
    assert raised.value.param == "response_format"
    admit(
        _request(tools=tools, tool_choice="none", response_format={"type": "json_object"}),
        model="m",
    )
    admit(_request(tools=tools, response_format={"type": "text"}), model="m")


def test_a_json_schema_response_format_needs_its_schema():
    request = _request(response_format={"type": "json_schema", "json_schema": {"name": "a"}})
    with pytest.raises(InvalidRequestError) as raised:
        admit(request, model="m")
    assert raised.value.param == "response_format.json_schema.schema"


def test_a_schema_description_is_rejected_because_it_is_never_rendered():
    json_schema = {"name": "a", "schema": SCHEMA, "description": "an answer"}
    with pytest.raises(UnsupportedParameterError) as raised:
        admit(
            _request(response_format={"type": "json_schema", "json_schema": json_schema}), model="m"
        )
    assert raised.value.param == "response_format.json_schema.description"


def test_supplied_sampling_fields_are_kept_and_omitted_ones_left_to_the_run():
    admitted = admit(_request(temperature=0.3, top_p=0.9, seed=7, stop=["x", "y"]), model="m")
    assert admitted.sampling == policy.SamplingOptions(
        temperature=0.3, top_p=0.9, seed=7, max_tokens=None, stop=("x", "y")
    )
    assert admit(_request(), model="m").sampling == policy.SamplingOptions()


@pytest.mark.parametrize(
    ("part", "param"),
    [
        (
            {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
            "messages[0].content[0].type",
        ),
        ({"type": "file", "file": {"file_id": "f"}}, "messages[0].content[0].type"),
    ],
)
def test_audio_and_file_parts_are_rejected(part, param):
    request = _request(messages=[{"role": "user", "content": [part]}])
    with pytest.raises(UnsupportedParameterError) as raised:
        admit(request, model="m")
    assert raised.value.param == param


def _uri(kind, data):
    return f"data:{kind};base64,{base64.b64encode(data).decode()}"


def test_png_and_jpeg_data_uris_pass_through_unchanged():
    assert image_bytes(_uri("image/png", PNG), "p") == PNG
    assert image_bytes(_uri("image/jpeg", JPEG), "p") == JPEG


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/cat.png",
        "file:///etc/passwd",
        "/tmp/cat.png",
        "data:video/mp4;base64,AAAA",
        "data:image/gif;base64,R0lGODlh",
        "data:image/png,rawbytes",
    ],
)
def test_every_other_image_reference_is_refused_before_sampling(url):
    with pytest.raises(UnsupportedParameterError) as raised:
        image_bytes(url, "messages[0].content[0]")
    assert raised.value.param == "messages[0].content[0].image_url.url"


def test_a_payload_that_is_not_the_declared_image_type_is_invalid():
    with pytest.raises(InvalidRequestError):
        image_bytes(_uri("image/png", JPEG), "p")


def test_an_image_detail_other_than_auto_is_rejected():
    part = {"type": "image_url", "image_url": {"url": _uri("image/png", PNG), "detail": "high"}}
    with pytest.raises(UnsupportedParameterError) as raised:
        admit(_request(messages=[{"role": "user", "content": [part]}]), model="m")
    assert raised.value.param == "messages[0].content[0].image_url.detail"


def test_images_become_renderer_image_parts():
    part = {"type": "image_url", "image_url": {"url": _uri("image/png", PNG)}}
    request = admit(
        _request(messages=[{"role": "user", "content": [{"type": "text", "text": "see"}, part]}]),
        model="m",
    )
    messages, _ = renderer_input(request, Returned())
    assert messages == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "see"}, {"type": "image", "image": PNG}],
        }
    ]


def test_a_tool_message_gets_the_name_of_the_call_it_answers():
    history = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q": 1}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "found"},
    ]
    messages, _ = renderer_input(admit(_request(messages=history), model="m"), Returned())
    assert messages[1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": {"q": 1}},
            }
        ],
    }
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "lookup",
        "content": "found",
    }


def test_a_tool_result_for_no_known_call_is_invalid():
    history = [{"role": "tool", "tool_call_id": "nope", "content": "x"}]
    with pytest.raises(InvalidRequestError) as raised:
        renderer_input(admit(_request(messages=history), model="m"), Returned())
    assert raised.value.param == "messages[0].tool_call_id"


def test_developer_messages_render_as_system_messages():
    request = admit(_request(messages=[{"role": "developer", "content": "be brief"}]), model="m")
    assert renderer_input(request, Returned())[0] == [{"role": "system", "content": "be brief"}]


TEXT_PARTS = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
CALL = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}


@pytest.mark.parametrize(
    ("message", "rendered"),
    [
        ({"role": "system"}, {"role": "system", "content": TEXT_PARTS}),
        ({"role": "developer"}, {"role": "system", "content": TEXT_PARTS}),
        ({"role": "assistant"}, {"role": "assistant", "content": "ab"}),
        (
            {"role": "tool", "tool_call_id": "call_1"},
            {"role": "tool", "tool_call_id": "call_1", "name": "lookup", "content": TEXT_PARTS},
        ),
    ],
)
def test_text_content_sent_as_a_list_of_parts_renders_its_text(message, rendered):
    history = [{"role": "assistant", "content": None, "tool_calls": [CALL]}]
    request = _request(messages=[*history, {**message, "content": TEXT_PARTS}])
    messages, _ = renderer_input(admit(request, model="m"), Returned())
    assert messages[-1] == rendered


def test_function_role_messages_are_rejected():
    request = _request(messages=[{"role": "function", "name": "f", "content": "x"}])
    with pytest.raises(UnsupportedParameterError):
        admit(request, model="m")


def test_every_schema_type_named_in_the_matrix_exists():
    for name in MATRIX:
        assert hasattr(models, name), name


def test_include_obfuscation_is_accepted_only_off():
    admit(_request(stream=True, stream_options={"include_obfuscation": False}), model="m")
    with pytest.raises(UnsupportedParameterError) as raised:
        admit(_request(stream=True, stream_options={"include_obfuscation": True}), model="m")
    assert raised.value.param == "stream_options.include_obfuscation"


def test_an_assistant_refusal_is_accepted_only_as_null():
    admit(_request(messages=[{"role": "assistant", "content": "x", "refusal": None}]), model="m")
    with pytest.raises(UnsupportedParameterError) as raised:
        admit(
            _request(messages=[{"role": "assistant", "content": "x", "refusal": "no"}]), model="m"
        )
    assert raised.value.param == "messages[0].refusal"


def test_assistant_history_with_reasoning_is_passed_to_the_renderer():
    history = [{"role": "assistant", "content": "x", "reasoning_content": "earlier thought"}]
    messages, _ = renderer_input(admit(_request(messages=history), model="m"), Returned())
    assert messages == [
        {"role": "assistant", "content": "x", "reasoning_content": "earlier thought"}
    ]


def test_function_tools_are_declared_to_the_renderer_with_their_schema():
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "find a thing",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
    _, tools = renderer_input(admit(_request(tools=[tool]), model="m"), Returned())
    assert tools == [tool]
    _, offered = renderer_input(
        admit(_request(tools=[tool], tool_choice="none"), model="m"), Returned()
    )
    assert offered == []  # tool_choice none: rendered without the tools


GATEWAY = "test_gateway_serving"
TRANSLATE = "test_openai_translate"
DEFAULTS = "test_a_default_only_field_accepts_only_its_no_op_value"
# Each field that is not rejected, and the test that shows what it does.
SEMANTIC_TESTS = {
    **{
        ("CreateChatCompletionRequest", field): (GATEWAY, test)
        for field, test in {
            "model": "test_another_model_is_not_found",
            "messages": "test_the_official_client_completes_and_the_sample_is_the_attempts",
            "temperature": "test_omitted_sampling_fields_take_the_runs_defaults_and_stop_replaces_the_renderers",
            "top_p": "test_omitted_sampling_fields_take_the_runs_defaults_and_stop_replaces_the_renderers",
            "seed": "test_omitted_sampling_fields_take_the_runs_defaults_and_stop_replaces_the_renderers",
            "max_tokens": "test_omitted_sampling_fields_take_the_runs_defaults_and_stop_replaces_the_renderers",
            "stop": "test_omitted_sampling_fields_take_the_runs_defaults_and_stop_replaces_the_renderers",
            "stream": "test_a_streamed_response_arrives_whole_and_settles_after_its_termination",
            "stream_options": "test_a_streamed_response_arrives_whole_and_settles_after_its_termination",
            "reasoning_effort": "test_reasoning_effort_selects_a_declared_renderer_over_http",
            "chat_template_kwargs": "test_request_level_template_options_are_rejected_unless_empty",
        }.items()
    },
    ("CreateChatCompletionRequest", "max_completion_tokens"): (
        None,
        "test_max_tokens_and_max_completion_tokens_are_exclusive_aliases",
    ),
    ("CreateChatCompletionRequest", "tools"): (
        None,
        "test_function_tools_are_declared_to_the_renderer_with_their_schema",
    ),
    ("CreateChatCompletionRequest", "tool_choice"): (
        None,
        "test_only_omitted_auto_or_none_tool_choice_is_accepted",
    ),
    **{
        ("CreateChatCompletionRequest", field): (None, DEFAULTS)
        for field in (
            "n",
            "parallel_tool_calls",
            "modalities",
            "store",
            "service_tier",
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
        )
    },
    ("ChatCompletionStreamOptions1", "include_usage"): (
        GATEWAY,
        "test_a_streamed_response_arrives_whole_and_settles_after_its_termination",
    ),
    ("ChatCompletionStreamOptions1", "include_obfuscation"): (
        None,
        "test_include_obfuscation_is_accepted_only_off",
    ),
    ("ChatCompletionRequestDeveloperMessage", "role"): (
        None,
        "test_developer_messages_render_as_system_messages",
    ),
    ("ChatCompletionRequestDeveloperMessage", "content"): (
        None,
        "test_text_content_sent_as_a_list_of_parts_renders_its_text",
    ),
    ("ChatCompletionRequestSystemMessage", "role"): (
        None,
        "test_developer_messages_render_as_system_messages",
    ),
    ("ChatCompletionRequestSystemMessage", "content"): (
        None,
        "test_text_content_sent_as_a_list_of_parts_renders_its_text",
    ),
    ("ChatCompletionRequestUserMessage", "role"): (None, "test_images_become_renderer_image_parts"),
    ("ChatCompletionRequestUserMessage", "content"): (
        None,
        "test_images_become_renderer_image_parts",
    ),
    **{
        ("ChatCompletionRequestAssistantMessage", field): (
            None,
            "test_a_tool_message_gets_the_name_of_the_call_it_answers",
        )
        for field in ("role", "content", "tool_calls")
    },
    ("ChatCompletionRequestAssistantMessage", "refusal"): (
        None,
        "test_an_assistant_refusal_is_accepted_only_as_null",
    ),
    ("ChatCompletionRequestAssistantMessage", "reasoning_content"): (
        None,
        "test_assistant_history_with_reasoning_is_passed_to_the_renderer",
    ),
    **{
        ("ChatCompletionRequestToolMessage", field): (
            None,
            "test_a_tool_message_gets_the_name_of_the_call_it_answers",
        )
        for field in ("role", "tool_call_id")
    },
    ("ChatCompletionRequestToolMessage", "content"): (
        None,
        "test_text_content_sent_as_a_list_of_parts_renders_its_text",
    ),
    **{
        (model, field): (None, "test_images_become_renderer_image_parts")
        for model, field in (
            ("ChatCompletionRequestMessageContentPartText", "type"),
            ("ChatCompletionRequestMessageContentPartText", "text"),
            ("ChatCompletionRequestMessageContentPartImage", "type"),
            ("ChatCompletionRequestMessageContentPartImage", "image_url"),
        )
    },
    ("ImageUrl", "url"): (None, "test_png_and_jpeg_data_uris_pass_through_unchanged"),
    ("ImageUrl", "detail"): (None, "test_an_image_detail_other_than_auto_is_rejected"),
    **{
        (model, field): (None, "test_a_tool_message_gets_the_name_of_the_call_it_answers")
        for model, field in (
            ("ChatCompletionMessageToolCall", "id"),
            ("ChatCompletionMessageToolCall", "type"),
            ("ChatCompletionMessageToolCall", "function"),
            ("Function", "name"),
            ("Function", "arguments"),
        )
    },
    **{
        (model, field): (None, "test_function_tools_are_declared_to_the_renderer_with_their_schema")
        for model, field in (
            ("ChatCompletionTool", "type"),
            ("ChatCompletionTool", "function"),
            ("FunctionObject", "name"),
            ("FunctionObject", "description"),
            ("FunctionObject", "parameters"),
        )
    },
    ("FunctionObject", "strict"): (None, "test_strict_function_generation_is_rejected"),
    ("ResponseFormatText", "type"): (
        None,
        "test_a_response_format_becomes_the_samplers_constraint",
    ),
    ("CreateChatCompletionRequest", "response_format"): (
        GATEWAY,
        "test_a_response_format_constrains_the_sample",
    ),
    **{
        (model, field): (None, "test_a_response_format_becomes_the_samplers_constraint")
        for model, field in (
            ("ResponseFormatJsonObject", "type"),
            ("ResponseFormatJsonSchema", "type"),
            ("ResponseFormatJsonSchema", "json_schema"),
            ("JsonSchema", "name"),
            ("JsonSchema", "strict"),
        )
    },
    ("JsonSchema", "schema_"): (None, "test_a_json_schema_response_format_needs_its_schema"),
}


def test_every_accepted_field_has_a_semantic_test():
    import importlib
    import sys

    accepted = {
        (name, field)
        for name, table in MATRIX.items()
        for field, capability in table.items()
        if capability.support is not Support.REJECTED
    }
    assert accepted == set(SEMANTIC_TESTS)
    here = sys.modules[__name__]
    for (model, field), (module, test) in SEMANTIC_TESTS.items():
        owner = importlib.import_module(f"{__package__}.{module}") if module else here
        assert callable(getattr(owner, test, None)), (model, field, test)
