"""The two data contracts: what each accepts, and how a bad row is reported."""

import json

import pytest
from support.builders import write_jsonl
from support.rl_sft import chat_row

from loops_cookbook.sft.data import DatasetError, load_dataset, load_rows

CALL = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}


def chat(*messages):
    return {"messages": list(messages)}


USER = {"role": "user", "content": "q"}
ANSWER = {"role": "assistant", "content": "a"}
CALLING = {"role": "assistant", "content": "", "tool_calls": [CALL]}
RESULT = {"role": "tool", "tool_call_id": "c1", "content": "1"}
UNNAMED = {k: v for k, v in CALL.items() if k != "id"}
UNNAMED_RESULT = {"role": "tool", "content": "1"}

INVALID_CHAT = {
    "a system message after the first": (
        chat(USER, {"role": "system", "content": "s"}, ANSWER),
        "a system message must come first",
    ),
    "a tool result nothing asked for": (chat(USER, RESULT, ANSWER), "must follow an assistant"),
    "a tool result for another call": (
        chat(USER, CALLING, {**RESULT, "tool_call_id": "other"}, ANSWER),
        "answers no call",
    ),
    "a call left unanswered": (
        chat(USER, {**CALLING, "tool_calls": [CALL, {**CALL, "id": "c2"}]}, RESULT, ANSWER),
        "leave c2 unanswered",
    ),
    "a call answered twice": (chat(USER, CALLING, RESULT, RESULT, ANSWER), "answer c1 twice"),
    "two calls with one id": (
        chat(USER, {**CALLING, "tool_calls": [CALL, CALL]}, RESULT, ANSWER),
        "share an id",
    ),
    "tool calls nobody answered": (
        chat(USER, CALLING, USER, ANSWER),
        "followed by its tool results",
    ),
    "every turn context only": (
        chat(USER, {**ANSWER, "weight": 0}),
        "no assistant message is trained",
    ),
    "arguments that are not JSON": (
        chat(
            USER,
            {**CALLING, "tool_calls": [{**CALL, "function": {"name": "f", "arguments": "{x"}}]},
        ),
        "arguments is not JSON",
    ),
    "arguments that are not an object": (
        chat(
            USER,
            {**CALLING, "tool_calls": [{**CALL, "function": {"name": "f", "arguments": "[1]"}}]},
        ),
        "must encode a JSON object",
    ),
    "a weight other than 0 or 1": (chat(USER, {**ANSWER, "weight": 0.5}), "weight"),
    "a weight that is a boolean": (chat(USER, {**ANSWER, "weight": True}), "weight"),
    "a trained turn with nothing in it": (
        chat(USER, {"role": "assistant", "content": ""}),
        "no content, reasoning_content or tool_calls",
    ),
    "a call without an id answered by id": (
        chat(USER, {**CALLING, "tool_calls": [UNNAMED]}, RESULT, ANSWER),
        "ids are given for some",
    ),
    "a result without an id for a call with one": (
        chat(USER, CALLING, {"role": "tool", "content": "1"}, ANSWER),
        "ids are given for some",
    ),
    "calls without ids and too few results": (
        chat(USER, {**CALLING, "tool_calls": [UNNAMED, UNNAMED]}, UNNAMED_RESULT, ANSWER),
        "there must be one each",
    ),
    "a tokens row in a chat file": ({"input_ids": [1, 2], "weights": [0, 1]}, "messages"),
}

INVALID_TOKENS = {
    "weights of another length": ({"input_ids": [1, 2, 3], "weights": [0, 1]}, "2 entries for 3"),
    "a target with nothing before it": (
        {"input_ids": [1, 2], "weights": [1, 1]},
        "weights[0] must be 0",
    ),
    "no target": ({"input_ids": [1, 2], "weights": [0, 0]}, "no token is a target"),
    "a negative token": ({"input_ids": [1, -2], "weights": [0, 1]}, "input_ids"),
    "a boolean token": ({"input_ids": [1, True], "weights": [0, 1]}, "input_ids.1"),
    "a float token": ({"input_ids": [1, 2.0], "weights": [0, 1]}, "input_ids.1"),
    "a string token": ({"input_ids": [1, "2"], "weights": [0, 1]}, "input_ids.1"),
    "a boolean weight": ({"input_ids": [1, 2], "weights": [0, True]}, "weights.1"),
    "a fractional weight": ({"input_ids": [1, 2], "weights": [0, 0.5]}, "weights.1"),
    "a chat row in a tokens file": (chat(USER, ANSWER), "input_ids"),
}


@pytest.mark.parametrize(("row", "problem"), INVALID_CHAT.values(), ids=INVALID_CHAT.keys())
def test_a_chat_row_breaking_the_contract_is_named_by_file_and_line(tmp_path, row, problem):
    path = write_jsonl(tmp_path / "train.jsonl", [chat_row(("q", "a")), row])
    with pytest.raises(DatasetError, match="read as the chat format") as raised:
        load_rows(str(path), "chat")
    assert "train.jsonl:2:" in str(raised.value)
    assert problem in str(raised.value)


@pytest.mark.parametrize(("row", "problem"), INVALID_TOKENS.values(), ids=INVALID_TOKENS.keys())
def test_a_tokens_row_breaking_the_contract_is_named_by_file_and_line(tmp_path, row, problem):
    path = write_jsonl(tmp_path / "train.jsonl", [{"input_ids": [1, 2], "weights": [0, 1]}, row])
    with pytest.raises(DatasetError, match="read as the tokens format") as raised:
        load_rows(str(path), "tokens")
    assert "train.jsonl:2:" in str(raised.value)
    assert problem in str(raised.value)


def test_every_bad_row_is_reported_at_once(tmp_path):
    path = write_jsonl(tmp_path / "train.jsonl", [chat(USER, RESULT, ANSWER)] * 3)
    with pytest.raises(DatasetError, match="3 problem") as raised:
        load_rows(str(path), "chat")
    assert all(f"train.jsonl:{line}:" in str(raised.value) for line in (1, 2, 3))


def test_a_conversation_may_end_on_tool_calls(tmp_path):
    path = write_jsonl(tmp_path / "train.jsonl", [chat(USER, CALLING)])
    (row,) = load_rows(str(path), "chat")
    assert row.example.trained_turns == (1,)


def test_the_renderer_sees_decoded_arguments_and_no_weight(tmp_path):
    path = write_jsonl(
        tmp_path / "train.jsonl", [chat(USER, {**CALLING, "weight": 1}, RESULT, ANSWER)]
    )
    (row,) = load_rows(str(path), "chat")
    calling = row.example.chat_messages()[1]
    assert "weight" not in calling
    assert calling["tool_calls"][0]["function"]["arguments"] == {"x": 1}


def test_blank_lines_are_not_rows_but_keep_line_numbers(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("\n" + json.dumps(chat_row(("q", "a"))) + "\n\n")
    (row,) = load_rows(str(path), "chat")
    assert row.name == "train.jsonl:2"


def test_a_held_out_fraction_is_disjoint_from_training_and_reproducible(tmp_path):
    path = write_jsonl(tmp_path / "train.jsonl", [chat_row((f"q{i}", f"a{i}")) for i in range(20)])

    def split(seed):
        dataset = load_dataset(
            str(path), format="chat", validation=None, validation_fraction=0.25, seed=seed
        )
        return [row.line for row in dataset.train], [row.line for row in dataset.validation]

    train, held_out = split(seed=3)
    assert len(held_out) == 5 and not set(train) & set(held_out)
    assert sorted(train + held_out) == list(range(1, 21))
    assert split(seed=3) == (train, held_out)
    assert split(seed=4) != (train, held_out)


def test_a_held_out_fraction_must_leave_rows_to_train(tmp_path):
    path = write_jsonl(tmp_path / "train.jsonl", [chat_row(("q", "a"))])
    with pytest.raises(DatasetError, match="leaves nothing to train on"):
        load_dataset(str(path), format="chat", validation=None, validation_fraction=0.5, seed=0)


def test_a_real_world_export_loads_with_its_extras_kept_aside(tmp_path):
    """Columns of the export's own, OpenAI's optional fields, Hugging Face's mapping
    arguments, a dataframe's 1.0 and calls without ids are all real data, not errors."""
    row = {
        "id": "conv-1",
        "parallel_tool_calls": False,
        "messages": [
            {"role": "user", "content": "q", "name": "ada"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "f", "arguments": {"x": 1}}}
                ],
                "weight": 1.0,
            },
            {"role": "tool", "content": "1", "latency_ms": 12},
            {"role": "assistant", "content": "done"},
        ],
    }
    (loaded,) = load_rows(str(write_jsonl(tmp_path / "train.jsonl", [row])), "chat")
    assert loaded.example.ignored() == {"id", "parallel_tool_calls", "messages[].latency_ms"}
    messages = loaded.example.chat_messages()
    assert messages[0] == {"role": "user", "content": "q", "name": "ada"}
    (call,) = messages[1]["tool_calls"]
    assert call["function"] == {"name": "f", "arguments": {"x": 1}}
    # Without ids, the n-th result answers the n-th call.
    assert messages[2] == {"role": "tool", "content": "1", "tool_call_id": call["id"]}


def test_a_hugging_face_export_with_null_for_absent_fields_loads(tmp_path):
    """Arrow gives every message every key, null where a message has none, and Hub
    datasets often keep `tools` as a JSON string."""
    row = {
        "messages": [
            {
                "role": "user",
                "content": "q",
                "tool_calls": None,
                "tool_call_id": None,
                "weight": None,
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{**UNNAMED}],
                "tool_call_id": None,
                "weight": None,
            },
            {
                "role": "tool",
                "content": "1",
                "tool_calls": None,
                "tool_call_id": None,
                "weight": None,
            },
            {
                "role": "assistant",
                "content": "a",
                "tool_calls": None,
                "tool_call_id": None,
                "weight": None,
            },
        ],
        "tools": json.dumps([{"type": "function", "function": {"name": "f"}}]),
        "metadata": None,
    }
    (loaded,) = load_rows(str(write_jsonl(tmp_path / "train.jsonl", [row])), "chat")
    assert loaded.example.ignored() == set()
    assert loaded.example.tools == [{"type": "function", "function": {"name": "f"}}]
    assert loaded.example.trained_turns == (1, 3)


def test_results_without_ids_answer_calls_in_order_block_by_block(tmp_path):
    calls = {
        **CALLING,
        "tool_calls": [UNNAMED, {**UNNAMED, "function": {"name": "g", "arguments": "{}"}}],
    }
    row = chat(
        USER, calls, UNNAMED_RESULT, UNNAMED_RESULT, calls, UNNAMED_RESULT, UNNAMED_RESULT, ANSWER
    )
    (loaded,) = load_rows(str(write_jsonl(tmp_path / "train.jsonl", [row])), "chat")
    messages = loaded.example.chat_messages()
    for head in (1, 4):
        ids = [call["id"] for call in messages[head]["tool_calls"]]
        assert [messages[head + k]["tool_call_id"] for k in (1, 2)] == ids
    assert len({messages[k]["tool_call_id"] for k in (2, 3, 5, 6)}) == 4
