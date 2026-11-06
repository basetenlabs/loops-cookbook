"""Shared GLM-5.3 conversation shapes for the contract tests."""

TOOLS = [
    {"name": "f", "parameters": {"type": "object"}, "strict": True},
    {"name": "hidden", "parameters": {}, "defer_loading": True},
]
CALLS = [
    {"id": key, "function": {"name": "f", "arguments": {"value": value}}}
    for key, value in [("a", 1), ("b", "two")]
]
BASE = [{"role": "system", "content": "Help"}, {"role": "user", "content": "Task"}]
ASSISTANT = {
    "role": "assistant",
    "content": "",
    "reasoning_content": " plan\n",
    "tool_calls": CALLS,
}
HISTORIES = [
    BASE,
    BASE
    + [
        {"role": "assistant", "content": " answer ", "reasoning_content": "\n plan \n"},
        {"role": "user", "content": "Next"},
    ],
    BASE + [{"role": "assistant", "content": "answer"}, {"role": "user", "content": "Next"}],
    BASE + [{"role": "assistant", "content": "<think>\nplan\n</think> answer "}],
    BASE
    + [
        ASSISTANT,
        {"role": "tool", "tool_call_id": "b", "content": "TWO"},
        {"role": "tool", "tool_call_id": "a", "content": "ONE"},
    ],
    BASE + [ASSISTANT, {"role": "tool", "content": "TWO"}, {"role": "tool", "content": "ONE"}],
    BASE
    + [
        ASSISTANT,
        {"role": "tool", "tool_call_id": "a", "content": "ONE"},
        {"role": "tool", "tool_call_id": "a", "content": "duplicate"},
    ],
    BASE + [ASSISTANT, {"role": "tool", "tool_call_id": "unknown", "content": "unknown"}],
    BASE
    + [
        ASSISTANT,
        {
            "role": "tool",
            "content": [
                {"tool_call_id": "b", "output": "TWO"},
                {"tool_call_id": "a", "output": "ONE"},
            ],
        },
    ],
    BASE + [ASSISTANT, {"role": "tool", "content": [{"type": "tool_reference", "name": "hidden"}]}],
]
