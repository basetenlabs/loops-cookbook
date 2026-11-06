"""The Terminal-Bench example's agent: its options, and how it treats each kind of reply.

``TerminalAgent`` runs as Harbor's agent with the options ``harbor.yaml`` gives it.
Its model is faked at the conversation, so each test scripts the replies.
"""

import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
import terminal_agent
import yaml
from support.builders import services_for
from terminal_agent import TerminalAgent

JOB = Path(__file__).resolve().parents[2] / "examples" / "terminal_bench" / "harbor.yaml"
OPTIONS = yaml.safe_load(JOB.read_text())["agents"][0]["kwargs"]


@pytest.mark.parametrize(
    "options",
    [
        {**OPTIONS, "max_turns": 0},
        {**OPTIONS, "unknown_option": True},
        {**OPTIONS, "command_timeout": 0},
        {},  # every limit is explicit
    ],
    ids=["no_turns", "unknown_option", "no_command_timeout", "no_limits"],
)
def test_native_options_reject_invalid_settings(options):
    with pytest.raises(ValueError):
        TerminalAgent.parse_options(options, {})


def _reply(*, truncated=False, usable=True, content="done", tool_calls=(), clamped_from=None):
    """Mirrors the facts a real reply reports, and whether the budget lowered its limit."""
    return NS(
        stop_reason="length" if truncated else "stop",
        native_message={"role": "assistant", "content": content} if usable else None,
        tool_calls=list(tool_calls),
        truncated=truncated,
        sample=NS(clamped_from=clamped_from),
    )


@pytest.fixture
def terminal(tmp_path):
    """``await terminal(replies, environment=None, **options)``: the agent, with ``options``
    over the shipped ones, run on one task whose model replies ``replies`` in turn.

    Returns the model's ``generate`` mock, the replies the agent ``discarded``, and the
    trial ``context`` it recorded its ending in.
    """

    async def run(replies, environment=None, **options):
        discarded = []
        generate = AsyncMock(side_effect=replies)
        conversation = NS(generate=generate, discard=discarded.append)
        agent = TerminalAgent(logs_dir=tmp_path, **{**OPTIONS, **options})
        agent.bind_loops_services(services_for(NS(conversation=lambda: conversation)))
        context = NS(metadata=None)
        await agent.run("Do the task", environment or NS(exec=AsyncMock()), context)
        return NS(generate=generate, discarded=discarded, context=context)

    return run


async def test_harbor_settings_and_tool_output_match_byte_limit(terminal):
    call = NS(name="bash", arguments={"command": "echo test"}, id="tool-1", status="ok")
    environment = NS(exec=AsyncMock(return_value=NS(stdout="éab", stderr="abcd", return_code=0)))
    ran = await terminal(
        [_reply(content="", tool_calls=[call]), _reply()], environment, max_output_bytes=3
    )
    environment.exec.assert_awaited_once_with(
        "echo test", cwd="/", timeout_sec=OPTIONS["command_timeout"]
    )
    # Blaxel's process API rejects a JSON float, even when its value is integral.
    assert type(environment.exec.await_args.kwargs["timeout_sec"]) is int
    prompt = ran.generate.await_args_list[1].kwargs["messages"]
    tool = next(message for message in prompt if message["role"] == "tool")
    assert json.loads(tool["content"]) == {"stdout": "éa", "stderr": "abc", "exit_code": 0}
    # The agent states neither: the run's [sampling] policy supplies both.
    assert "max_tokens" not in ran.generate.await_args.kwargs
    assert "temperature" not in ran.generate.await_args.kwargs
    assert ran.context.metadata == {"termination_reason": "completed", "generation_count": 2}


async def test_a_cut_turn_is_discarded_and_re_sampled(terminal):
    ran = await terminal([_reply(truncated=True), _reply(truncated=False)])
    assert len(ran.discarded) == 1, "the cut generation must leave the trajectory"
    assert ran.discarded[0].truncated is True
    assert ran.context.metadata == {"termination_reason": "completed", "generation_count": 2}


async def test_a_turn_cut_on_every_attempt_abandons_the_rollout(terminal):
    attempts = terminal_agent.TURN_ATTEMPTS
    ran = await terminal([_reply(truncated=True) for _ in range(attempts)])
    # Bounded: it does not keep re-sampling until the agent deadline expires.
    assert ran.generate.await_count == len(ran.discarded) == attempts
    assert ran.context.metadata == {"termination_reason": "truncated", "generation_count": attempts}


async def test_a_turn_the_trajectory_budget_cut_is_kept_and_ends_context_overflow(terminal):
    """Every retry would get the same room, so it is not re-sampled, and it trains."""
    ran = await terminal([_reply(truncated=True, clamped_from=16384)])
    assert ran.generate.await_count == 1 and ran.discarded == []
    assert ran.context.metadata == {"termination_reason": "context_overflow", "generation_count": 1}


async def test_a_reply_the_renderer_cannot_hand_back_is_not_an_answer(terminal):
    """A finished generation the renderer withheld: not appended, not graded as done."""
    ran = await terminal([_reply(usable=False)])
    assert ran.context.metadata["termination_reason"] == "unusable_reply"
    # Finished generation, so re-sampling would only repeat it.
    assert ran.generate.await_count == 1
    assert ran.discarded == []


async def test_a_reply_that_is_both_cut_and_unusable_is_treated_as_cut(terminal):
    """An unclosed tool block is always both; re-sampling is the response to it."""
    ran = await terminal([_reply(truncated=True, usable=False), _reply(truncated=False)])
    # Checking usability first would abandon the rollout instead of re-sampling.
    assert len(ran.discarded) == 1
    assert ran.context.metadata["termination_reason"] == "completed"


async def test_a_tool_error_is_capped_like_any_other_tool_output(terminal):
    """A gateway answering with an HTML page must not spend the model's context."""
    call = NS(name="bash", arguments={"command": "ls"}, id="c1", status="ok")
    environment = NS(exec=AsyncMock(side_effect=RuntimeError("<html>" + "x" * 5000)))
    ran = await terminal(
        [_reply(tool_calls=[call]), _reply(content="done")], environment, max_output_bytes=64
    )
    tool_message = next(
        m for m in ran.generate.await_args_list[-1].kwargs["messages"] if m.get("role") == "tool"
    )
    assert len(json.loads(tool_message["content"])["error"]) <= 64
