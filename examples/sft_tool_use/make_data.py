"""Write train.jsonl and validation.jsonl for a unit-conversion assistant that uses a tool.

The script is deterministic, so the files beside it are what it produces. Each
conversation asks for a conversion. The assistant calls ``convert``, reads the result
and answers in one fixed style. A third of the conversations continue with a follow-up
that the assistant answers without the tool. Run it from any directory with
``python examples/sft_tool_use/make_data.py``.
"""

import json
import random
from pathlib import Path

HERE = Path(__file__).parent

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "convert",
            "description": "Convert a value between units.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    }
]

SYSTEM = "You convert units. Use the convert tool, then give the answer to two decimals."

# Each entry is (from, to, factor, plural names).
UNITS = [
    ("mi", "km", 1.609344, ("miles", "kilometres")),
    ("km", "mi", 1 / 1.609344, ("kilometres", "miles")),
    ("lb", "kg", 0.45359237, ("pounds", "kilograms")),
    ("kg", "lb", 1 / 0.45359237, ("kilograms", "pounds")),
    ("ft", "m", 0.3048, ("feet", "metres")),
    ("m", "ft", 1 / 0.3048, ("metres", "feet")),
    ("gal", "l", 3.785411784, ("gallons", "litres")),
    ("l", "gal", 1 / 3.785411784, ("litres", "gallons")),
]

SINGULAR = {
    "miles": "mile",
    "kilometres": "kilometre",
    "pounds": "pound",
    "kilograms": "kilogram",
    "feet": "foot",
    "metres": "metre",
    "gallons": "gallon",
    "litres": "litre",
}


def unit(name: str, amount: float) -> str:
    """Return ``name`` as said of ``amount``, such as one pound or two pounds."""
    return SINGULAR[name] if amount == 1 else name


QUESTIONS = [
    "How many {to} is {value} {frm}?",
    "Convert {value} {frm} to {to}.",
    "What's {value} {frm} in {to}?",
]


def conversation(rng: random.Random, index: int) -> dict:
    frm, to, factor, (frm_name, to_name) = rng.choice(UNITS)
    value = rng.choice([1, 2, 3, 5, 7, 10, 12, 15, 20, 26.2, 42, 100])
    result = value * factor
    call_id = f"call_{index}"
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": rng.choice(QUESTIONS).format(
                value=value, frm=unit(frm_name, value), to=to_name
            ),
        },
        {
            "role": "assistant",
            "reasoning_content": f"The user wants {frm_name} in {to_name}; call convert.",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "convert",
                        "arguments": json.dumps({"value": value, "from_unit": frm, "to_unit": to}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": f"{result:.6f}"},
        {
            "role": "assistant",
            "reasoning_content": "The tool returned the result; round it to two decimals.",
            "content": f"{value} {unit(frm_name, value)} is {result:.2f} {to_name}.",
        },
    ]
    if index % 3 == 0:
        messages += [
            {"role": "user", "content": "And rounded to a whole number?"},
            {
                "role": "assistant",
                "reasoning_content": "Round the previous result; no tool needed.",
                "content": f"About {round(result)} {unit(to_name, round(result))}.",
            },
        ]
    return {"messages": messages, "tools": TOOLS, "metadata": {"pair": f"{frm}->{to}"}}


def main() -> None:
    rng = random.Random(0)
    rows = [conversation(rng, index) for index in range(72)]
    for name, part in (("train.jsonl", rows[:64]), ("validation.jsonl", rows[64:])):
        (HERE / name).write_text("".join(json.dumps(row) + "\n" for row in part))


if __name__ == "__main__":
    main()
