import asyncio
import contextlib
import inspect
from copy import deepcopy
from unittest.mock import Mock

import pytest
from baseten.loops.models import SamplingParams
from pydantic import BaseModel
from renderers.base import ParsedToolCall, ToolCallParseStatus
from support.builders import make_sample
from support.fakes import ScriptedSampler
from support.leases import direct_llm, lease

from loops_cookbook.llm.client import (
    ContextWindowExceeded,
    Conversation,
    ConversationBusy,
    LLMResponse,
    Renderers,
    SamplingPolicy,
    UnsupportedParameterError,
)
from loops_cookbook.llm.rendering import ParsedReply, RendererFailure, RenderingError
from loops_cookbook.model_access.lease import DirectModelAccess
from loops_cookbook.model_access.producers import ProducerSettings


class NativeToolCall(BaseModel):
    name: str
    arguments: dict[str, int]


class Renderer:
    stop_sequences = (99,)

    def __init__(self):
        self.calls = []
        self.parse_error = None
        self.native_message = {
            "role": "assistant",
            "tool_calls": [NativeToolCall(name="lookup", arguments={"first": 1, "second": 2})],
        }

    def render(self, messages, tools):
        self.calls.append(deepcopy((messages, tools)))
        return (1, 2) if len(messages) <= 1 else (1, 2, 8, 9, 3)

    def parse(self, prompt, completion, tools):
        if self.parse_error:
            raise self.parse_error
        return ParsedReply("answer", "thought", (), True, native_message=self.native_message)


@pytest.fixture
def client():
    sampler, renderer = ScriptedSampler(), Renderer()
    return direct_llm(sampler, renderer), renderer, sampler


def test_generation_keywords_cover_the_sdk_sampling_fields():
    controls = {
        "self",
        "sampling_params",
        "messages",
        "tools",
        "max_trajectory_tokens",
        "reasoning_effort",
    }
    keywords = set(inspect.signature(Conversation.generate).parameters) - controls
    assert keywords == set(SamplingParams.model_fields)


async def test_the_run_supplies_sampling_policy_an_agent_did_not_state():
    """Policy is the run's decision; an agent only states what it needs to differ on."""
    sampler = ScriptedSampler()
    policy = SamplingPolicy(temperature=0.3, max_generation_tokens=64, max_trajectory_tokens=4096)
    conversation = direct_llm(sampler, Renderer(), sampling_policy=policy).conversation()
    await conversation.generate(messages=[{"role": "user", "content": "hello"}])
    assert (sampler.params[0].temperature, sampler.params[0].max_tokens) == (0.3, 64)

    # An agent with a reason to differ still can.
    await conversation.generate(
        messages=[{"role": "user", "content": "hello"}], temperature=1.5, max_tokens=8
    )
    assert (sampler.params[1].temperature, sampler.params[1].max_tokens) == (1.5, 8)

    # Explicit None removes the generation default; zero is an override, not omission.
    await conversation.generate(
        messages=[],
        temperature=0,
        max_tokens=None,
        top_p=0.9,
        top_k=1,
        seed=0,
        stop=[77],
        structured_outputs={"regex": "[A-Z]+"},
    )
    explicit = sampler.params[2]
    assert (explicit.temperature, explicit.max_tokens) == (0, 4094)
    assert (explicit.top_p, explicit.top_k, explicit.seed) == (0.9, 1, 0)
    assert 77 in explicit.stop and 99 in explicit.stop
    assert explicit.structured_outputs == {"regex": "[A-Z]+"}


def one_token_client(**options):
    """A client whose sampler writes one token, so any room left is enough for it."""
    sampler, renderer = ScriptedSampler(tokens=(8,), logprobs=(-0.1,)), Renderer()
    return direct_llm(sampler, renderer, **options), renderer, sampler


async def test_a_generation_is_clamped_to_the_room_the_trajectory_budget_leaves(client):
    """Prompt (2) plus max_tokens (4) over a budget of 5 samples with the 3 left."""
    llm, _, sampler = client
    conversation = llm.conversation()
    hello = [{"role": "user", "content": "hello"}]
    clamped = await conversation.generate(messages=hello, max_tokens=4, max_trajectory_tokens=5)
    fits = await conversation.generate(messages=hello, max_tokens=4, max_trajectory_tokens=6)
    assert [params.max_tokens for params in sampler.params] == [3, 4]
    # What was asked for and what was given, so a "length" stop can be told apart.
    assert (clamped.sample.max_tokens, clamped.sample.clamped_from) == (3, 4)
    assert (fits.sample.max_tokens, fits.sample.clamped_from) == (4, None)


async def test_the_runs_default_max_tokens_is_clamped_by_the_runs_budget():
    sampler = ScriptedSampler()
    policy = SamplingPolicy(max_generation_tokens=64, max_trajectory_tokens=10)
    conversation = direct_llm(sampler, Renderer(), sampling_policy=policy).conversation()
    reply = await conversation.generate(messages=[{"role": "user", "content": "hello"}])
    assert sampler.params[0].max_tokens == 8
    assert (reply.sample.max_tokens, reply.sample.clamped_from) == (8, 64)


async def test_only_a_prompt_that_fills_the_budget_is_refused_before_sampling():
    """One token of room is still a generation; none is not."""
    llm, _, sampler = one_token_client()
    conversation = llm.conversation()
    hello = [{"role": "user", "content": "hello"}]
    for budget in (1, 2):  # the prompt is 2 tokens
        with pytest.raises(ContextWindowExceeded, match="no room"):
            await conversation.generate(messages=hello, max_tokens=4, max_trajectory_tokens=budget)
    assert sampler.params == []
    reply = await conversation.generate(messages=hello, max_tokens=4, max_trajectory_tokens=3)
    assert sampler.params[0].max_tokens == 1
    assert (reply.sample.prompt_ids, reply.sample.completion_ids) == ((1, 2), (8,))
    assert reply.stop_reason == "length" and reply.truncated


async def test_clamping_explicit_sampling_params_leaves_the_callers_unchanged(client):
    llm, _, sampler = client
    params = SamplingParams(max_tokens=4, stop=[77])
    reply = await llm.conversation().generate(
        params, messages=[{"role": "user", "content": "hello"}], max_trajectory_tokens=5
    )
    assert (params.max_tokens, params.stop) == (4, [77])
    assert sampler.params[0].max_tokens == 3
    assert reply.sample.clamped_from == 4


async def test_a_generation_that_names_no_max_tokens_gets_the_room_left(client):
    """As a model API given no max_tokens generates up to its context."""
    llm, _, sampler = client
    reply = await llm.conversation().generate(
        SamplingParams(), messages=[{"role": "user", "content": "hello"}], max_trajectory_tokens=5
    )
    assert sampler.params[0].max_tokens == 3
    assert (reply.sample.max_tokens, reply.sample.clamped_from) == (3, None)


async def test_a_bridged_turn_is_clamped_by_its_whole_prompt():
    """The bridge's prompt (the sampled prefix and the new turn) is what the budget
    counts, not the new messages alone."""
    llm, renderer, sampler = one_token_client()
    renderer.bridge = lambda prompt, completion, messages, tools, **_: (
        *prompt,
        *completion,
        3,
        3,
        3,
    )
    chat = llm.conversation()
    history = [{"role": "user", "content": "start"}]
    first = await chat.generate(messages=history, max_tokens=4, max_trajectory_tokens=8)
    history += [deepcopy(renderer.native_message), {"role": "tool", "content": "result"}]
    second = await chat.generate(messages=history, max_tokens=4, max_trajectory_tokens=8)
    assert second.sample.prompt_ids == (1, 2, 8, 3, 3, 3)  # bridged, not re-rendered
    assert [params.max_tokens for params in sampler.params] == [4, 2]
    assert (first.sample.clamped_from, second.sample.clamped_from) == (None, 4)


def test_a_conversation_and_its_client_expose_no_history_management(client):
    """The agent owns the history: neither keeps one it could append to or restore."""
    llm, _, _ = client
    chat = llm.conversation()
    assert {name for name in dir(chat) if not name.startswith("_")} == {
        "conversation_id",
        "discard",
        "generate",
        "producer",
    }
    assert {name for name in dir(llm) if not name.startswith("_")} == {
        "conversation",
        "producer",
        "renderer",
        "renderers",
        "samples",
    }


async def test_messages_required_and_explicit_sampling_options(client):
    llm, _, sampler = client
    chat = llm.conversation()
    with pytest.raises(TypeError):
        await chat.generate(max_tokens=2)
    with pytest.raises(TypeError, match="temprature"):
        await chat.generate(messages=[], temprature=0)
    assert not sampler.params
    params = SamplingParams(max_tokens=2, stop=[77])
    response = await chat.generate(params, messages=[{"role": "assistant", "content": "authored"}])
    assert params.stop == [77]  # the caller's params are not changed
    assert response.usage.total_tokens == 4
    with pytest.raises(ValueError, match="not both"):
        await chat.generate(params, messages=[], max_tokens=3)
    with pytest.raises(ValueError, match="not both"):
        await chat.generate(params, messages=[], structured_outputs=None)


@pytest.mark.parametrize("decline", [False, True])
async def test_agent_owned_full_history_bridges_or_falls_back(client, decline):
    llm, renderer, sampler = client
    bridges = []

    def bridge(prompt, completion, messages, tools, *, previous_messages=()):
        assert previous_messages == history[:-1]
        bridges.append(deepcopy((messages, tools)))
        return None if decline else (*prompt, *completion, 3)

    renderer.bridge = bridge
    chat = llm.conversation()
    history = [{"role": "user", "content": "start"}]
    tools = [{"name": "lookup"}]
    first = await chat.generate(messages=history, tools=tools, max_tokens=2)
    history += [deepcopy(renderer.native_message), {"role": "tool", "content": "result"}]
    assert isinstance(history[1]["tool_calls"][0], NativeToolCall)
    second = await chat.generate(messages=history, tools=tools, max_tokens=2)
    assert bridges == [([{"role": "tool", "content": "result"}], tools)]
    assert len(renderer.calls) == (2 if decline else 1)
    assert second.sample.prompt_ids == first.sample.prompt_ids + first.sample.completion_ids + (3,)
    assert sampler.params[0].stop == [99]
    assert len(llm.samples) == 2


@pytest.mark.parametrize(
    "rewrite",
    [
        lambda prompt, completion: (*prompt[:-1], 0, *completion, 3),  # a prompt token
        lambda prompt, completion: (*prompt, *completion[:-1], 0, 3),  # a completion token
        lambda prompt, completion: (*prompt, *completion[:-1]),  # shorter than the prefix
        lambda prompt, completion: (*completion, *prompt, 3),  # the parts swapped
    ],
    ids=["prompt", "completion", "short", "swapped"],
)
async def test_a_bridge_that_changes_the_sampled_prefix_is_a_rendering_error(client, rewrite):
    llm, renderer, sampler = client

    def rewrites(prompt, completion, messages, tools, *, previous_messages=()):
        return rewrite(prompt, completion)

    renderer.bridge = rewrites
    chat = llm.conversation()
    history = [{"role": "user", "content": "start"}]
    await chat.generate(messages=history, max_tokens=2)
    history += [deepcopy(renderer.native_message), {"role": "tool", "content": "result"}]
    with pytest.raises(RenderingError, match="changed the sampled prefix"):
        await chat.generate(messages=history, max_tokens=2)
    assert len(sampler.params) == 1  # rejected before the second generation sampled
    assert len(llm.samples) == 1


@pytest.mark.parametrize("edit", ["content", "argument_order", "tools", "tool_order", "assistant"])
async def test_changed_render_input_bypasses_bridge(client, edit):
    llm, renderer, _ = client

    def forbidden(*args):
        pytest.fail("edited history must render in full")

    renderer.bridge = forbidden
    chat = llm.conversation()
    history = [{"role": "user", "content": "start"}]
    tools = [{"name": "lookup", "description": "d"}]
    await chat.generate(messages=history, tools=tools, max_tokens=2)
    history += [deepcopy(renderer.native_message), {"role": "tool", "content": "result"}]
    if edit == "content":
        history[0]["content"] = "edited"
    elif edit == "argument_order":
        history[1]["tool_calls"][0].arguments = {"second": 2, "first": 1}
    elif edit == "tools":
        tools = [{"name": "different"}]
    elif edit == "tool_order":
        tools = [{"description": "d", "name": "lookup"}]
    else:
        history.append({"role": "assistant", "content": "authored continuation"})
    await chat.generate(messages=history, tools=tools, max_tokens=2)
    assert len(renderer.calls) == 2


@pytest.mark.parametrize("explicit_none", [False, True])
async def test_tools_are_per_call_not_retained(client, explicit_none):
    llm, renderer, _ = client
    chat = llm.conversation()
    history = [{"role": "user", "content": "start"}]
    await chat.generate(messages=history, tools=[{"name": "lookup"}], max_tokens=2)
    kwargs = {"tools": None} if explicit_none else {}
    await chat.generate(messages=history, max_tokens=2, **kwargs)
    assert renderer.calls[-1][1] == []


async def test_caller_and_native_message_mutations_do_not_alias_internal_bridge_state(client):
    llm, renderer, _ = client
    chat = llm.conversation()
    history = [{"role": "user", "content": "original"}]
    tools = [{"name": "lookup"}]
    native = deepcopy(renderer.native_message)
    await chat.generate(messages=history, tools=tools, max_tokens=2)
    history[0]["content"] = "mutated"
    tools[0]["name"] = "mutated"
    renderer.native_message["tool_calls"][0].arguments["first"] = 999
    bridges = []
    renderer.bridge = lambda p, c, m, t, **kw: bridges.append((m, t)) or (*p, *c, 3)
    await chat.generate(
        messages=[
            {"role": "user", "content": "original"},
            native,
            {"role": "tool", "content": "result"},
        ],
        tools=[{"name": "lookup"}],
        max_tokens=2,
    )
    assert len(bridges) == 1


@pytest.mark.parametrize("unread", ["raises", "count_raises", "no_native_message"])
async def test_a_completion_the_renderer_cannot_read_fails_the_attempt(client, unread):
    """Caught or not, the failure is the attempt's: nothing further samples or trains."""
    llm, renderer, sampler = client
    chat = llm.conversation()
    error = ValueError("invalid")
    if unread == "raises":
        renderer.parse_error = error
    elif unread == "count_raises":
        renderer.reasoning_tokens = Mock(side_effect=error)
    else:
        renderer.native_message = None
    with pytest.raises(RendererFailure) as failed:
        await chat.generate(messages=[{"role": "user", "content": "start"}], max_tokens=2)
    if unread != "no_native_message":
        assert failed.value.__cause__ is error
    with contextlib.suppress(RendererFailure):  # an agent that swallows it changes nothing
        await chat.generate(messages=[{"role": "user", "content": "again"}], max_tokens=2)
    assert len(sampler.params) == 1
    with pytest.raises(RendererFailure):
        _ = llm.samples


def _response(*, stop_reason=None, reasoning_complete=True, statuses=(), native_message=True):
    """A reply carrying one truncation signal at a time."""
    sample = make_sample("s1", (1,), (2,), (-0.1,), 0, stop_reason=stop_reason)
    calls = tuple(
        ParsedToolCall(name="t", arguments={}, status=status, raw="") for status in statuses
    )
    native = {"role": "assistant", "content": ""} if native_message else None
    parsed = ParsedReply("", None, calls, reasoning_complete, native_message=native)
    return LLMResponse(sample=sample, parsed=parsed)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"stop_reason": "length"}, True),
        ({"reasoning_complete": False}, True),
        # A tool-call block that never closed: the generation was still going.
        ({"statuses": (ToolCallParseStatus.UNCLOSED_BLOCK,), "native_message": False}, True),
        ({}, False),
        ({"stop_reason": "stop"}, False),
        # No reasoning state reported is not evidence of a cut.
        ({"reasoning_complete": None}, False),
    ],
)
def test_a_cut_generation_is_reported_however_it_was_cut(kwargs, expected):
    assert _response(**kwargs).truncated is expected


@pytest.mark.parametrize(
    "status",
    [
        ToolCallParseStatus.UNKNOWN_TOOL,
        ToolCallParseStatus.INVALID_JSON,
        ToolCallParseStatus.MISSING_NAME,
        ToolCallParseStatus.MALFORMED_STRUCTURE,
    ],
)
def test_a_tool_call_the_model_got_wrong_is_not_a_cut_generation(status):
    """It finished when it meant to. Re-sampling is not the remedy, so not truncation."""
    response = _response(statuses=(status,), native_message=False)
    assert response.truncated is False
    assert response.native_message is None


def test_an_unparsed_completion_is_unusable_without_being_cut():
    """`parse_error` and truncation are separate conditions with separate remedies."""
    sample = make_sample("s1", (1,), (2,), (-0.1,), 0)
    response = LLMResponse(sample=sample, parsed=None, parse_error=ValueError("bad"))
    assert response.truncated is False
    assert response.native_message is None


async def test_trainability_is_the_runs_and_each_client_samples_as_its_producer():
    access = DirectModelAccess(
        lease(ScriptedSampler(), producers=ProducerSettings({"main": True, "critic": False})),
        Renderers(Renderer()),
    )
    main, critic = access.llm(), access.llm("critic")
    await main.conversation().generate(messages=[], max_tokens=2)
    await critic.conversation().generate(messages=[], max_tokens=2)
    assert [(s.producer, s.trainable) for s in main.samples] == [
        ("main", True),
        ("critic", False),
    ]


async def test_discard_removes_the_generation_and_restores_the_bridge(client):
    llm, renderer, _ = client
    bridged = []
    renderer.bridge = lambda p, c, m, t, **kw: bridged.append(m) or (*p, *c, 3)
    chat = llm.conversation()
    history = [{"role": "user", "content": "start"}]
    first = await chat.generate(messages=history, max_tokens=2)
    followup = [*history, deepcopy(renderer.native_message), {"role": "tool", "content": "r"}]
    rejected = await chat.generate(messages=followup, max_tokens=2)
    chat.discard(rejected)
    assert [sample.sample_id for sample in llm.samples] == [first.sample.sample_id]
    again = await chat.generate(messages=followup, max_tokens=2)
    assert again.sample.prompt_ids == rejected.sample.prompt_ids
    assert [sample.sample_id for sample in llm.samples] == [
        first.sample.sample_id,
        again.sample.sample_id,
    ]
    with pytest.raises(ValueError, match="most recent"):
        chat.discard(first)


# -------------------------------------------------------- reasoning renderers


class EffortRenderer(Renderer):
    """Renders like the test renderer, recording which effort rendered each call."""

    def __init__(self, name):
        super().__init__()
        self.name = name

    def bridge(self, prompt, completion, messages, tools, *, previous_messages=()):
        self.calls.append(("bridge", self.name))
        return (*prompt, *completion, 3)


def effort_client():
    base, high = EffortRenderer("base"), EffortRenderer("high")
    llm = DirectModelAccess(lease(ScriptedSampler()), Renderers(base, {"high": high})).llm()
    return llm, base, high


async def test_reasoning_effort_selects_the_declared_renderer_for_one_generation():
    llm, base, high = effort_client()
    chat = llm.conversation()
    await chat.generate(messages=[{"role": "user", "content": "a"}], reasoning_effort="high")
    await llm.conversation().generate(messages=[{"role": "user", "content": "b"}])
    assert len(high.calls) == 1 and len(base.calls) == 1


async def test_an_undeclared_reasoning_effort_is_an_unsupported_parameter():
    llm, _, _ = effort_client()
    with pytest.raises(UnsupportedParameterError) as raised:
        await llm.conversation().generate(messages=[], reasoning_effort="low")
    assert raised.value.param == "reasoning_effort"
    assert llm.samples == ()  # rejected before anything was sampled


async def test_the_bridge_restarts_when_the_reasoning_effort_changes():
    llm, base, high = effort_client()
    chat = llm.conversation()
    history = [{"role": "user", "content": "start"}]
    await chat.generate(messages=history, reasoning_effort="high")
    history += [deepcopy(high.native_message), {"role": "tool", "content": "result"}]
    await chat.generate(messages=history)  # base renderer: rendered in full
    assert ("bridge", "high") not in high.calls and base.calls and base.calls[-1][0] != "bridge"
    history += [deepcopy(base.native_message), {"role": "tool", "content": "again"}]
    await chat.generate(messages=history)  # same renderer again: bridged
    assert base.calls[-1] == ("bridge", "base")


async def test_a_conversation_refuses_a_second_concurrent_generation():
    sampler = ScriptedSampler()
    sampler.release = asyncio.Event()
    chat = DirectModelAccess(lease(sampler), Renderers(Renderer())).llm().conversation()
    first = asyncio.create_task(chat.generate(messages=[{"role": "user", "content": "a"}]))
    await sampler.entered.wait()
    with pytest.raises(ConversationBusy):
        await chat.generate(messages=[{"role": "user", "content": "b"}])
    sampler.release.set()
    await first


@pytest.mark.parametrize(
    ("stop", "sent"),
    [([77], [77, 99]), ([99, 77], [99, 77]), (["\n"], ["\n"]), ("\n", ["\n"]), ([], [99])],
    ids=[
        "ids join the renderer's",
        "no id twice",
        "strings stay the caller's",
        "one string is a list of one",
        "an empty list asks for nothing",
    ],
)
async def test_a_caller_stop_keeps_the_renderers_end_of_turn_ids_when_it_can(client, stop, sent):
    """Ids merge; strings go as given, and the sampler still ends at the model's own EOS."""
    llm, _, sampler = client
    await llm.conversation().generate(messages=[{"role": "user", "content": "q"}], stop=stop)
    assert sampler.params[0].stop == sent


# ------------------------------------------------- a tool call that did not parse

UNPARSED = (
    ParsedToolCall(
        raw="<call>bsah ls</call>",
        name="bsah",
        arguments={"q": 1},
        status=ToolCallParseStatus.INVALID_JSON,
    ),
    ParsedToolCall(raw="<call>{oops</call>", status=ToolCallParseStatus.MALFORMED_STRUCTURE),
)


def _unparsed_then_native(renderer):
    """The first reply has calls that did not parse; every later one parses."""
    replies = iter([ParsedReply("let me look", "hmm", UNPARSED, True, None)])
    renderer.parse = lambda *_: next(
        replies, ParsedReply("answer", None, (), True, {"role": "assistant", "content": "answer"})
    )


def test_history_message_is_the_native_message_or_the_turn_as_written():
    sample = make_sample("s1", (1,), (2,), (-0.1,), 0)
    native = {"role": "assistant", "content": "hi"}
    assert LLMResponse(sample, ParsedReply("hi", None, (), True, native)).history_message == native
    unparsed = LLMResponse(sample, ParsedReply("let me look", "hmm", UNPARSED, True, None))
    assert unparsed.native_message is None
    assert unparsed.history_message == {
        "role": "assistant",
        "content": "let me look",
        "reasoning_content": "hmm",
        "tool_calls": [
            {
                "id": "call_s1_0",
                "type": "function",
                "function": {"name": "bsah", "arguments": '{"q": 1}'},
            },
            {  # nothing read: no name, and the raw call as its arguments
                "id": "call_s1_1",
                "type": "function",
                "function": {"name": "", "arguments": "<call>{oops</call>"},
            },
        ],
    }
    assert LLMResponse(sample, None, ValueError("bad")).history_message is None


async def test_after_a_call_that_did_not_parse_the_next_turn_extends_its_sampled_tokens(client):
    llm, renderer, _ = client
    _unparsed_then_native(renderer)
    renderer.bridge = lambda prompt, completion, messages, tools, **_: (*prompt, *completion, 3)
    chat = llm.conversation()
    history = [{"role": "user", "content": "start"}]
    first = await chat.generate(messages=history, max_tokens=2)
    history += [first.history_message, {"role": "tool", "content": "Error: unknown tool: bsah"}]
    second = await chat.generate(messages=history, max_tokens=2)
    assert second.sample.prompt_ids == first.sample.prompt_ids + first.sample.completion_ids + (3,)
    assert len(renderer.calls) == 1  # the turn as written was never rendered
    # Later turns bridge as usual.
    history += [second.history_message, {"role": "user", "content": "more"}]
    third = await chat.generate(messages=history, max_tokens=2)
    assert third.sample.prompt_ids[: len(second.sample.prompt_ids)] == second.sample.prompt_ids


@pytest.mark.parametrize("why", ["no_bridge", "declined", "tools_changed", "history_edited"])
async def test_a_call_that_did_not_parse_is_never_rendered_as_written(client, why):
    llm, renderer, sampler = client
    _unparsed_then_native(renderer)
    if why != "no_bridge":
        renderer.bridge = lambda p, c, m, t, **_: None if why == "declined" else (*p, *c, 3)
    chat = llm.conversation()
    history = [{"role": "user", "content": "start"}]
    first = await chat.generate(messages=history, tools=[{"name": "f"}], max_tokens=2)
    history += [first.history_message, {"role": "tool", "content": "error"}]
    tools = [{"name": "g"}] if why == "tools_changed" else [{"name": "f"}]
    if why == "history_edited":
        history[0] = {"role": "user", "content": "begin"}
    with pytest.raises(RenderingError, match="did not parse"):
        await chat.generate(messages=history, tools=tools, max_tokens=2)
    assert len(sampler.params) == 1  # refused before sampling


async def test_a_discarded_call_that_did_not_parse_cannot_be_sent_back(client):
    """Discarded, the turn did not happen: there are no sampled tokens left to continue from."""
    llm, renderer, sampler = client
    _unparsed_then_native(renderer)
    chat = llm.conversation()
    history = [{"role": "user", "content": "start"}]
    first = await chat.generate(messages=history, max_tokens=2)
    chat.discard(first)
    with pytest.raises(RenderingError, match="discarded"):
        await chat.generate(messages=[*history, first.history_message], max_tokens=2)
    assert len(sampler.params) == 1
    await chat.generate(messages=history, max_tokens=2)  # re-asking is the way on


def _reordered(message):
    return {key: message[key] for key in reversed(list(message))}


def _without_reasoning(message):
    return {key: value for key, value in message.items() if key != "reasoning_content"}


def _as_openai_models(message):
    from openai.types.chat import ChatCompletionMessageToolCall

    calls = [ChatCompletionMessageToolCall.model_validate(call) for call in message["tool_calls"]]
    return {**message, "tool_calls": calls}


def _arguments_parsed(message):
    calls = deepcopy(message["tool_calls"])
    calls[0]["function"]["arguments"] = {"q": 1}
    return {**message, "tool_calls": calls}


def _arguments_reserialized(message):
    calls = deepcopy(message["tool_calls"])
    calls[0]["function"]["arguments"] = '{ "q" : 1 }'
    return {**message, "tool_calls": calls}


@pytest.mark.parametrize(
    "edit",
    [_reordered, _without_reasoning, _as_openai_models, _arguments_reserialized, _arguments_parsed],
)
async def test_a_copy_of_a_call_that_did_not_parse_is_never_rendered_either(client, edit):
    """An agent that stores or trims its history still cannot render the turn as written."""
    llm, renderer, sampler = client
    _unparsed_then_native(renderer)
    chat = llm.conversation()
    history = [{"role": "user", "content": "start"}]
    first = await chat.generate(messages=history, max_tokens=2)
    history += [edit(first.history_message), {"role": "tool", "content": "error"}]
    with pytest.raises(RenderingError, match="did not parse"):
        await chat.generate(messages=history, max_tokens=2)
    assert len(sampler.params) == 1


async def test_a_native_message_with_a_call_that_did_not_parse_fails_the_renderer(client):
    """A renderer that keeps a native message for a call it could not parse breaks its contract."""
    llm, renderer, _ = client
    renderer.parse = lambda *_: ParsedReply(
        "let me look", None, UNPARSED, True, {"role": "assistant", "content": "let me look"}
    )
    with pytest.raises(RendererFailure, match="could not parse"):
        await llm.conversation().generate(messages=[{"role": "user", "content": "start"}])


def test_the_turn_as_written_keeps_the_arguments_characters():
    sample = make_sample("s1", (1,), (2,), (-0.1,), 0)
    call = ParsedToolCall(
        raw="<call>",
        name="find",
        arguments={"path": "café"},
        status=ToolCallParseStatus.INVALID_JSON,
    )
    reply = LLMResponse(sample, ParsedReply("", None, (call,), True, None))
    assert reply.history_message["tool_calls"][0]["function"]["arguments"] == '{"path": "café"}'


CUT = ParsedToolCall(
    raw="<call>",
    id="functions.bash:0",
    name="bash",
    arguments={"command": "ls"},
    status=ToolCallParseStatus.UNCLOSED_BLOCK,
)
NATIVE = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "functions.bash:0",
            "type": "function",
            "function": {"name": "bash", "arguments": {"command": "ls"}},
        }
    ],
}


@pytest.mark.parametrize(
    "reasoning", [("list them", "list them again"), (None, None)], ids=["differs", "identical"]
)
@pytest.mark.parametrize("discarded", [False, True])
async def test_a_retried_turn_that_parsed_is_rendered_even_with_the_same_call(
    client, reasoning, discarded
):
    """A model that names its calls (Kimi) can write the same turn again, and parse this time.

    The model wrote that turn whole, so rendering it puts no words in its mouth, even
    when nothing tells it apart from the reply that did not parse.
    """
    llm, renderer, _ = client
    cut_reasoning, retried_reasoning = reasoning
    native = {**NATIVE, "reasoning_content": retried_reasoning}
    replies = iter(
        [
            ParsedReply("", cut_reasoning, (CUT,), False, None),
            ParsedReply("", retried_reasoning, (), True, native),
        ]
    )
    renderer.parse = lambda *_: next(
        replies, ParsedReply("done", None, (), True, {"role": "assistant", "content": "done"})
    )
    chat = llm.conversation()
    history = [{"role": "user", "content": "start"}]
    first = await chat.generate(messages=history, max_tokens=2)
    assert first.truncated
    if discarded:
        chat.discard(first)
    retried = await chat.generate(messages=history, max_tokens=2)
    history += [retried.history_message, {"role": "tool", "content": "a b"}]
    await chat.generate(messages=history, max_tokens=2)
    assert renderer.calls[-1][0] == history


async def test_discarding_a_retry_that_parsed_keeps_the_turn_it_matched_unrendered(client):
    """Discarded, the retry did not happen: the model never wrote the turn whole."""
    llm, renderer, sampler = client
    replies = iter(
        [ParsedReply("", None, (CUT,), False, None), ParsedReply("", None, (), True, NATIVE)]
    )
    renderer.parse = lambda *_: next(replies)
    chat = llm.conversation()
    history = [{"role": "user", "content": "start"}]
    first = await chat.generate(messages=history, max_tokens=2)
    retried = await chat.generate(messages=history, max_tokens=2)
    chat.discard(retried)
    with pytest.raises(RenderingError, match="did not parse"):
        await chat.generate(messages=[*history, first.history_message], max_tokens=2)
    assert len(sampler.params) == 2


async def test_a_retry_that_parsed_does_not_release_a_turn_the_history_still_holds(client):
    """The turn was answered in history before the model wrote it whole: it stays unrendered."""
    llm, renderer, sampler = client
    replies = iter(
        [ParsedReply("", None, (CUT,), False, None), ParsedReply("", None, (), True, NATIVE)]
    )
    renderer.parse = lambda *_: next(replies)
    renderer.bridge = lambda p, c, m, t, **_: (*p, *c, 3)
    chat = llm.conversation()
    history = [{"role": "user", "content": "start"}]
    first = await chat.generate(messages=history, tools=[{"name": "bash"}], max_tokens=2)
    history += [first.history_message, {"role": "tool", "content": "a b"}]
    await chat.generate(messages=history, tools=[{"name": "bash"}], max_tokens=2)
    with pytest.raises(RenderingError, match="did not parse"):  # tools changed: no bridge
        await chat.generate(messages=history, tools=[{"name": "sh"}], max_tokens=2)
    assert len(sampler.params) == 2
