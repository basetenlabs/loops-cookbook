"""SFT targets on the real renderers: one rule, checked per model family.

Each test states a property every supported template must give a supervised sequence,
over conversations that exercise tool calls, follow-ups and context-only turns.
"""

import json

import pytest

from loops_cookbook.llm.factory import create_renderer
from loops_cookbook.sft.data import ChatExample, Row
from loops_cookbook.sft.sequences import IGNORE_INDEX, Refused, supervise

GLM_RENDERER = "loops_cookbook.llm.renderer_extensions.glm53:renderer"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "convert",
            "parameters": {"type": "object", "properties": {"value": {"type": "number"}}},
        },
    }
]


def call(identity, value):
    return {
        "id": identity,
        "type": "function",
        "function": {"name": "convert", "arguments": json.dumps({"value": value})},
    }


TOOL_TURN = [
    {"role": "system", "content": "Convert units."},
    {"role": "user", "content": "12 miles in km?"},
    {
        "role": "assistant",
        "reasoning_content": "Call the tool.",
        "content": "",
        "tool_calls": [call("c1", 12)],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "19.312"},
    {"role": "assistant", "reasoning_content": "Round it.", "content": "About 19.31 km."},
]
FOLLOW_UP = [
    *TOOL_TURN,
    {"role": "user", "content": "Whole number?"},
    {"role": "assistant", "reasoning_content": "Round again.", "content": "About 19 km."},
]
CONTEXT_ONLY = [
    *TOOL_TURN[:4],
    {**TOOL_TURN[4], "content": "Context answer.", "weight": 0},
    {"role": "user", "content": "Whole number?"},
    {"role": "assistant", "reasoning_content": "Round again.", "content": "About 19 km."},
]
# Results answered out of call order: GLM-5.3's template sorts them back into it.
OUT_OF_ORDER = [
    TOOL_TURN[1],
    {
        "role": "assistant",
        "reasoning_content": "Both.",
        "content": "",
        "tool_calls": [call("a", 1), call("b", 2)],
    },
    {"role": "tool", "tool_call_id": "b", "content": "2"},
    {"role": "tool", "tool_call_id": "a", "content": "1"},
    {"role": "assistant", "reasoning_content": "Report.", "content": "Done."},
]
CONVERSATIONS = {
    "tool turn": TOOL_TURN,
    "follow-up": FOLLOW_UP,
    "context-only turn": CONTEXT_ONLY,
    "results out of call order": OUT_OF_ORDER,
}


def build(request, name, **chat_template_kwargs):
    if name == "qwen":
        tokenizer = request.getfixturevalue("qwen_tokenizer")
        model, extension = "Qwen/Qwen3.6-35B-A3B", None
    elif name == "glm":
        tokenizer = request.getfixturevalue("glm53_tokenizer")
        model, extension = "zai-org/GLM-5.3", GLM_RENDERER
    else:
        tokenizer = request.getfixturevalue("nemotron_tokenizer")
        model, extension = tokenizer.name_or_path, None
    renderer = create_renderer(
        model, tokenizer=tokenizer, renderer=extension, chat_template_kwargs=chat_template_kwargs
    )
    return name, renderer, tokenizer


@pytest.fixture(params=["qwen", "glm", "nemotron"])
def family(request):
    """``(name, renderer, tokenizer)`` for each supported model family."""
    return build(request, request.param)


def test_the_renderer_reaches_its_tokenizer(family):
    """`tokenize`/`detokenize` read Prime's private `_tokenizer`; a Prime bump that
    drops it fails here rather than in the middle of a dataset."""
    _, renderer, tokenizer = family
    text = "Hello there.<|im_end|>" if "<|im_end|>" in tokenizer.get_vocab() else "Hello there."
    assert renderer.detokenize(renderer.tokenize(text)) == text


def row(messages):
    return Row(
        "contract.jsonl", 1, ChatExample.model_validate({"messages": messages, "tools": TOOLS})
    )


def runs(trained):
    """``(start, end)`` of each maximal run of trained positions."""
    found, start = [], None
    for position, flag in enumerate([*trained, False]):
        if flag and start is None:
            start = position
        elif not flag and start is not None:
            found.append((start, position))
            start = None
    return found


def turns_per_sequence(example, supervised):
    turns = example.trained_turns
    return [(turn,) for turn in turns] if supervised.per_turn else [turns]


@pytest.mark.parametrize("messages", CONVERSATIONS.values(), ids=CONVERSATIONS.keys())
def test_every_trained_turn_ends_on_exactly_one_stop_token(family, messages):
    _, renderer, _ = family
    stops = set(renderer.stop_sequences)
    for sequence in supervise(row(messages), renderer, max_seq_len=8192).sequences:
        for start, end in runs(sequence.trained):
            span = sequence.ids[start:end]
            assert span[-1] in stops
            assert sum(token in stops for token in span) == 1


@pytest.mark.parametrize("messages", CONVERSATIONS.values(), ids=CONVERSATIONS.keys())
def test_each_trained_turn_reads_exactly_its_own_generation_prompt(family, messages):
    _, renderer, _ = family
    example = row(messages)
    supervised = supervise(example, renderer, max_seq_len=8192)
    chat = example.example.chat_messages()
    for sequence, turns in zip(
        supervised.sequences, turns_per_sequence(example.example, supervised), strict=True
    ):
        spans = runs(sequence.trained)
        assert len(spans) == len(turns)
        for (start, _), turn in zip(spans, turns, strict=True):
            assert sequence.ids[:start] == renderer.render(chat[:turn], TOOLS).ids


def test_a_context_only_turn_trains_nothing(family):
    _, renderer, tokenizer = family
    supervised = supervise(row(CONTEXT_ONLY), renderer, max_seq_len=8192)
    trained_text = "".join(
        tokenizer.decode([t for t, flag in zip(s.ids, s.trained, strict=True) if flag])
        for s in supervised.sequences
    )
    seen_text = "".join(tokenizer.decode(list(s.ids)) for s in supervised.sequences)
    assert "Context answer." in seen_text
    assert "Context answer." not in trained_text
    assert "About 19 km." in trained_text


def test_history_rendering_decides_whether_turns_train_together(family):
    """GLM-5.3 keeps reasoning in history; Qwen3.6 and Nemotron drop it before the
    latest user message, so their earlier turns each train as their own sequence."""
    name, renderer, _ = family
    supervised = supervise(row(FOLLOW_UP), renderer, max_seq_len=8192)
    assert supervised.per_turn is (name != "glm")
    assert len(supervised.sequences) == (1 if name == "glm" else 3)


@pytest.mark.parametrize("messages", CONVERSATIONS.values(), ids=CONVERSATIONS.keys())
def test_a_datum_predicts_each_trained_token_from_everything_before_it(family, messages):
    _, renderer, _ = family
    for sequence in supervise(row(messages), renderer, max_seq_len=8192).sequences:
        datum = sequence.datum()
        targets = datum.loss_fn_inputs["target_tokens"].data
        weights = datum.loss_fn_inputs["weights"].data
        assert datum.model_input.length == len(sequence.ids) - 1 == len(targets)
        for position, (token, trained) in enumerate(
            zip(sequence.ids[1:], sequence.trained[1:], strict=True)
        ):
            assert targets[position] == (token if trained else IGNORE_INDEX)
            assert weights[position] == float(trained)


def test_a_conversation_over_the_budget_is_refused(family):
    _, renderer, _ = family
    with pytest.raises(Refused, match="over max_seq_len 64"):
        supervise(row(TOOL_TURN), renderer, max_seq_len=64)


WITHOUT_REASONING = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello."},
    {"role": "user", "content": "Bye"},
    {"role": "assistant", "content": "Goodbye."},
]


@pytest.mark.parametrize(
    ("name", "options", "trains"),
    [
        ("qwen", {}, True),
        ("qwen", {"enable_thinking": False}, True),
        ("glm", {}, True),
        # Nemotron renders an empty reasoning as <think></think>, but prompts with
        # <think>\n: no sequence is what the model wrote, so only thinking-off trains it.
        ("nemotron", {}, False),
        ("nemotron", {"enable_thinking": False}, True),
    ],
)
def test_turns_without_reasoning_train_as_the_model_writes_them(request, name, options, trains):
    _, renderer, tokenizer = build(request, name, **options)
    if not trains:
        with pytest.raises(Refused, match="thinking-off option"):
            supervise(row(WITHOUT_REASONING), renderer, max_seq_len=8192)
        return
    supervised = supervise(row(WITHOUT_REASONING), renderer, max_seq_len=8192)
    chat = row(WITHOUT_REASONING).example.chat_messages()
    stops = set(renderer.stop_sequences)
    for sequence, turns in zip(
        supervised.sequences,
        turns_per_sequence(row(WITHOUT_REASONING).example, supervised),
        strict=True,
    ):
        for (start, end), turn in zip(runs(sequence.trained), turns, strict=True):
            assert sequence.ids[:start] == renderer.render(chat[:turn], TOOLS).ids
            assert sequence.ids[end - 1] in stops
            assert chat[turn]["content"] in tokenizer.decode(list(sequence.ids[start:end]))
