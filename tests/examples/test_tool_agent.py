"""The tool-agent example: a multi-turn agent with tools, in a Blaxel sandbox, as an Episode.

Every test goes through a real ``Run``, with the cookbook's own Blaxel adapter and
the real SDK over the tests' fake Blaxel (``support/fake_blaxel.py``), whose
sandbox API runs each command with ``/bin/sh`` in a directory of its own, so the
episode's tool calls and the task's check really execute. The policy is scripted: a renderer that answers each
prompt with the reply a script chooses, and the baseline sampler's first letter
decides which rollouts set out to solve the task.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import httpx
import pytest
from support.builders import task_file
from support.offline_run import offline_config, train_offline

from loops_cookbook.rl import cli

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "tool_agent"
CONFIG = EXAMPLE / "train_blaxel.toml"


SHIPPED = {
    task["id"]: task for task in map(json.loads, (EXAMPLE / "tasks.jsonl").read_text().splitlines())
}
PORT = SHIPPED["port"]  # the shipped task these runs train on, its check included
FIXED = "name=shop\nport=9090\ndebug=true\n"
SOLVERS = frozenset("AB")  # a first sampled letter that sets a rollout out to solve it
GROUP = 4


# --- the scripted policy ------------------------------------------------------------


def say(text: str):
    from loops_cookbook.llm import ParsedReply

    return ParsedReply(text, None, (), None, {"role": "assistant", "content": text})


def use(*calls: tuple[str, Any]):
    """A reply calling each ``(name, arguments)`` in order."""
    from renderers.base import ParsedToolCall

    from loops_cookbook.llm import ParsedReply

    parsed = tuple(
        ParsedToolCall(raw=json.dumps(arguments), name=name, arguments=arguments, id=f"c{index}")
        for index, (name, arguments) in enumerate(calls)
    )
    native = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "id": call.id, "function": {"name": call.name, "arguments": args}}
            for call, (_, args) in zip(parsed, calls, strict=True)
        ],
    }
    return ParsedReply("", None, parsed, None, native)


def malformed():
    """A tool call that won't parse: the renderer withholds the message, as Prime's does."""
    from renderers.base import ParsedToolCall, ToolCallParseStatus

    from loops_cookbook.llm import ParsedReply

    call = ParsedToolCall(raw='{"command": "cat app.co', status=ToolCallParseStatus.INVALID_JSON)
    return ParsedReply("", None, (call,), None, None)


Policy = Callable[[Sequence[Mapping[str, Any]], str], Any]


class PolicyRenderer:
    """Renders each message to a few tokens, and parses every reply from ``policy``.

    ``policy(messages, letter)`` gets the prompt's messages and the first letter the
    baseline sampler drew for this generation; ``first_letters`` keeps each first
    turn's letter, so a test knows how many rollouts set out to solve the task.
    """

    stop_sequences: tuple[str, ...] = ()
    supports_images = False

    def __init__(self, policy: Policy) -> None:
        self._policy = policy
        self._prompts: dict[tuple[int, ...], list[Mapping[str, Any]]] = {}
        self.first_letters: list[str] = []
        self.tool_results: list[dict[str, Any]] = []

    def render(self, messages, tools):
        from loops_cookbook.llm import RenderedPrompt

        ids = [1]
        for message in messages:
            digest = hashlib.sha256(json.dumps(message, sort_keys=True).encode()).digest()
            ids.extend(10 + byte for byte in digest[:4])
        self._prompts[tuple(ids)] = deepcopy(list(messages))
        return RenderedPrompt.text(ids)

    def parse(self, prompt_ids, completion_ids, tools):
        from baseline_model import LETTERS

        messages = self._prompts[tuple(prompt_ids)]
        letter = LETTERS[completion_ids[0]]
        if not any(message["role"] == "tool" for message in messages):
            self.first_letters.append(letter)
        last_turn = messages[-1]["role"] == "tool"
        for message in reversed(messages if last_turn else ()):  # the last turn's results
            if message["role"] != "tool":
                break
            self.tool_results.append(json.loads(message["content"]))
        return self._policy(messages, letter)

    @property
    def solvers(self) -> int:
        return sum(letter in SOLVERS for letter in self.first_letters)


def tool_turns(messages) -> int:
    return sum(message["role"] == "tool" for message in messages)


def fixes_the_port(messages, letter):
    """A solver reads app.conf, writes it fixed, and answers; the rest give up at once."""
    if tool_turns(messages) == 0:
        return use(("shell", {"command": "cat app.conf"})) if letter in SOLVERS else say("No.")
    if tool_turns(messages) == 1:
        return use(("write_file", {"path": "app.conf", "content": FIXED}))
    return say("Done.")


@pytest.fixture(autouse=True)
def quick_polls(monkeypatch):
    """The fake sandbox answers at once and kills a command at its timeout, so a
    command's wait need not pace itself or allow long for the kill."""
    import tool_episode

    monkeypatch.setattr(tool_episode, "WAIT_INTERVAL_MS", 10)
    monkeypatch.setattr(tool_episode, "WAIT_MARGIN_SECONDS", 1)


async def train(tmp_path, blaxel, policy: Policy, **settings):
    """One step of the shipped config on the shipped ``port`` task, with ``settings`` changed."""
    from loops_cookbook.providers.blaxel.settings import BlaxelSettings

    shipped = cli.load_training_config(CONFIG)
    config = offline_config(
        CONFIG,
        tmp_path,
        {
            "rollout_runtime.files": {"tasks": str(task_file(tmp_path, [PORT]))},
            "training.group_size": GROUP,
            "training.groups_per_step": 1,
            "training.max_steps": 1,
            **settings,
        },
    )
    adapter = blaxel.adapter(BlaxelSettings.model_validate(shipped.providers["blaxel"]))
    renderer = PolicyRenderer(policy)
    outcome, _, run = await train_offline(config, renderer=renderer, adapters={"blaxel": adapter})
    lines = (run.out / "rollouts.jsonl").read_text().splitlines()
    steps_file = run.out / "steps.jsonl"  # absent when the run stopped before a step
    steps = (
        [json.loads(line) for line in steps_file.read_text().splitlines()]
        if steps_file.exists()
        else []
    )
    return NS(
        outcome=outcome,
        records=[json.loads(line) for line in lines],
        steps=steps,
        renderer=renderer,
    )


def agent(options: Mapping[str, Any] | None = None, **changes: Any) -> dict[str, Any]:
    """The shipped ``[rollout_runtime.config]`` with ``changes``, and ``options`` changed in it."""
    config = cli.load_training_config(CONFIG).rollout_runtime.config
    return {
        "rollout_runtime.config": {
            **config,
            **changes,
            "options": {**config["options"], **(options or {})},
        }
    }


def rewards(trained) -> list[float]:
    return sorted(record["reward"] for record in trained.records)


def expected(trained, solved: int | None = None) -> list[float]:
    """Every rollout's reward, 1 for each that ``solved``: by default, each solver."""
    total = len(trained.renderer.first_letters)
    solved = trained.renderer.solvers if solved is None else solved
    return sorted([1.0] * solved + [0.0] * (total - solved))


def fixed(root: Path) -> bool:
    return (root / "work" / "app.conf").read_text() == FIXED


def on_check(blaxel, action: Callable[[str, dict[str, Any]], None]) -> None:
    """Run ``action`` before each of the task's checks, and before nothing else."""

    def before_exec(name: str, request: dict[str, Any]) -> None:
        if request["command"] == PORT["check"]:
            action(name, request)

    blaxel.before_exec = before_exec


# --- the tests ----------------------------------------------------------------------


async def test_a_solver_changes_the_sandbox_and_scores_1_where_the_rest_score_0(tmp_path, blaxel):
    trained = await train(tmp_path, blaxel, fixes_the_port)
    assert trained.outcome.ok
    assert 0 < trained.renderer.solvers < len(trained.records)  # the premise: mixed groups
    assert rewards(trained) == expected(trained)
    roots = list((tmp_path / "sandboxes").iterdir())
    assert len(roots) == len(trained.records)
    assert sum(map(fixed, roots)) == trained.renderer.solvers
    (step,) = trained.steps
    assert step["rollout/termination_reason/completed"] == GROUP  # the trained group's four
    assert all(client.closed for client in blaxel.clients)  # the layer closed every client
    assert blaxel.api.sandboxes == {}  # every sandbox released


async def test_an_agent_that_loops_until_its_deadline_is_graded_from_what_it_left(tmp_path, blaxel):
    def loops(messages, letter):
        """A solver fixes the file first; then every rollout lists files forever."""
        if tool_turns(messages) == 0 and letter in SOLVERS:
            return use(("write_file", {"path": "app.conf", "content": FIXED}))
        return use(("shell", {"command": "ls"}))

    trained = await train(
        tmp_path,
        blaxel,
        loops,
        **agent({"max_turns": 10_000}, agent_timeout_seconds=1),
        **{"rollout.rollout_timeout_seconds": 30},
    )
    assert trained.outcome.ok
    # Graded, every one: none discarded but a group whose rewards were all alike.
    assert all(record["reason"] in (None, "no_variance") for record in trained.records)
    # A rollout scores 1 where its own write_file ran before the cut, and nowhere else:
    # on a slow runner the deadline can cut a solver before its write.
    ran = [c for sandbox in blaxel.sandboxes.values() for c in sandbox.commands]
    writes = sum("FILE" in c.request.get("env", {}) and c.status == "completed" for c in ran)
    assert rewards(trained) == expected(trained, solved=writes)
    (step,) = trained.steps
    assert step["runtime/episodes/agent_timed_out/mean"] == 1.0  # every run() was cut
    assert any(len(commands) > 3 for commands in blaxel.commands.values())  # the premise


async def test_a_tool_error_goes_back_to_the_model_and_the_episode_goes_on(tmp_path, blaxel):
    def errs_then_fixes(messages, letter):
        """A solver first makes three bad calls, then fixes the file; the rest give up."""
        if tool_turns(messages) == 0:
            if letter not in SOLVERS:
                return say("No.")
            return use(
                ("delete_everything", {}),
                ("shell", {"cmd": "ls"}),
                ("shell", {"command": "cat missing.conf"}),
            )
        return fixes_the_port(messages[:2] + messages[6:], "A")  # as if the errors never were

    trained = await train(tmp_path, blaxel, errs_then_fixes)
    assert trained.outcome.ok and rewards(trained) == expected(trained)
    results = trained.renderer.tool_results
    assert {"error": "there is no tool 'delete_everything'"} in results
    assert any(r.get("error", "").startswith("bad arguments for shell") for r in results)
    assert any(r.get("exit_code", 0) != 0 and "missing.conf" in r["stderr"] for r in results)


async def test_a_malformed_tool_call_ends_the_episode_which_is_still_graded(tmp_path, blaxel):
    def malformed_or_fixes(messages, letter):
        """A solver's first reply won't parse; the rest fix the file."""
        if tool_turns(messages) == 0 and letter in SOLVERS:
            return malformed()
        return fixes_the_port(messages, "A")

    trained = await train(tmp_path, blaxel, malformed_or_fixes)
    assert trained.outcome.ok
    unusable = trained.renderer.solvers
    assert rewards(trained) == expected(trained, solved=len(trained.records) - unusable)
    (step,) = trained.steps
    trained_zeros = [
        r for r in trained.records if r["disposition"] == "trained" and not r["reward"]
    ]
    assert step["rollout/termination_reason/unusable_reply"] == len(trained_zeros)


WRITE_FIXED = ("write_file", {"path": "app.conf", "content": FIXED})
# The fixed file, written by a shell command instead of write_file.
PRINT_FIXED = f"printf '{FIXED.replace(chr(10), '\\n')}' > app.conf"


def trained_zeros(trained) -> int:
    return sum(r["disposition"] == "trained" and not r["reward"] for r in trained.records)


async def test_an_agent_that_never_answers_stops_at_max_turns(tmp_path, blaxel):
    def never_answers(messages, letter):
        """A solver fixes the file first; then every rollout lists files."""
        if tool_turns(messages) == 0 and letter in SOLVERS:
            return use(WRITE_FIXED)
        return use(("shell", {"command": "ls"}))

    trained = await train(tmp_path, blaxel, never_answers, **agent({"max_turns": 3}))
    assert trained.outcome.ok and rewards(trained) == expected(trained)
    (step,) = trained.steps
    assert step["rollout/termination_reason/max_turns"] == GROUP
    assert (step["rollout/turns"], step["rollout/tool_calls"]) == (3, 3)
    # Each sandbox ran three tool calls, then the check: no fourth turn.
    assert {len(commands) for commands in blaxel.commands.values()} == {4}


async def test_the_model_reads_the_last_max_output_chars_of_each_stream(tmp_path, blaxel):
    def prints(messages, letter):
        """Every rollout prints ten characters to each stream; a solver fixes the file."""
        if tool_turns(messages) == 0:
            loud = ("shell", {"command": "printf 0123456789; printf abcdefghij >&2"})
            return use(loud, WRITE_FIXED) if letter in SOLVERS else use(loud)
        return say("Done.")

    trained = await train(tmp_path, blaxel, prints, **agent({"max_output_chars": 4}))
    assert trained.outcome.ok and rewards(trained) == expected(trained)
    printed = [r for r in trained.renderer.tool_results if r.get("stdout")]
    assert len(printed) == len(trained.records)
    assert all((r["stdout"], r["stderr"]) == ("6789", "ghij") for r in printed)


async def test_a_command_past_its_timeout_is_killed_and_the_model_reads_that(tmp_path, blaxel):
    def hangs(messages, letter):
        """Every rollout runs a command that would take a minute; a solver fixes the file."""
        if tool_turns(messages) == 0:
            hang = ("shell", {"command": "echo started; sleep 60"})
            return use(WRITE_FIXED, hang) if letter in SOLVERS else use(hang)
        return say("Done.")

    trained = await train(
        tmp_path,
        blaxel,
        hangs,
        **agent({"command_timeout_seconds": 1}, agent_timeout_seconds=10),
    )
    assert trained.outcome.ok and rewards(trained) == expected(trained)
    (step,) = trained.steps
    assert (
        step["rollout/termination_reason/completed"] == GROUP
    )  # killed at 1 s, well inside the deadline
    killed = [r for r in trained.renderer.tool_results if r.get("status") == "killed"]
    assert len(killed) == len(trained.records)
    assert all((r["exit_code"], r["stdout"]) == (-1, "started\n") for r in killed)


async def test_a_check_past_its_timeout_is_killed_and_scores_0_not_graded_again(tmp_path, blaxel):
    def slow_check(name: str, request: dict[str, Any]) -> None:
        request["command"] = f"sleep 60; {request['command']}"

    on_check(blaxel, slow_check)
    trained = await train(
        tmp_path,
        blaxel,
        fixes_the_port,
        **agent({"command_timeout_seconds": 1}),
        **{"training.max_consecutive_unusable_groups": 1},
    )
    # A check that outlasts its timeout did not pass, in a sandbox still alive: 0, once.
    assert trained.records and all(
        (record["reward"], record["retries"]) == (0.0, 0) for record in trained.records
    )
    assert not any(
        command.request["command"].startswith("sleep 60") and command.status == "running"
        for sandbox in blaxel.sandboxes.values()
        for command in sandbox.commands
    )  # and it is killed, not left running


async def test_an_output_over_blaxels_64_kb_record_reaches_the_model_whole(tmp_path, blaxel):
    def floods(messages, letter):
        """Every rollout prints 1 to 20000 (over 64 KB); a solver fixes the file."""
        if tool_turns(messages) == 0:
            flood = ("shell", {"command": "seq 1 20000"})
            return use(flood, WRITE_FIXED) if letter in SOLVERS else use(flood)
        return say("Done.")

    whole = "".join(f"{n}\n" for n in range(1, 20001))
    assert len(whole.encode()) > 65_536  # the premise: the record keeps only its end
    trained = await train(tmp_path, blaxel, floods, **agent({"max_output_chars": len(whole)}))
    assert trained.outcome.ok and rewards(trained) == expected(trained)
    printed = [r for r in trained.renderer.tool_results if r.get("stdout")]
    assert len(printed) == len(trained.records)
    assert {r["stdout"] for r in printed} == {whole}


async def test_a_conversation_that_outgrows_the_trajectory_budget_ends_context_overflow(
    tmp_path, blaxel
):
    def never_answers(messages, letter):
        """One call a turn: a solver fixes the file first; then every rollout lists files."""
        if tool_turns(messages) == 0 and letter in SOLVERS:
            return use(WRITE_FIXED)
        return use(("shell", {"command": "ls"}))

    # The scripted renderer: 1 token, then 4 per message. Turn n's prompt is 9 + 8(n-1)
    # tokens; of a budget of 35, turn 4's prompt (33) leaves its reply 2 tokens, not
    # the 10 it asks for, and turn 5's (41) leaves none.
    trained = await train(
        tmp_path,
        blaxel,
        never_answers,
        **{"model.max_seq_len": 35, "sampling.max_generation_tokens": 10},
    )
    assert trained.outcome.ok and rewards(trained) == expected(trained)
    (step,) = trained.steps
    assert step["rollout/termination_reason/context_overflow"] == GROUP
    assert (step["rollout/turns"], step["rollout/tool_calls"]) == (4, 4)
    assert step["sampling/cut_at_budget"] == 0  # turn 4's 2 tokens of room were enough


async def test_a_cut_reply_ends_the_episode_truncated_and_its_tool_calls_never_run(
    tmp_path, blaxel
):
    def cut_or_fixes(messages, letter):
        """A solver fixes the file; the rest are cut mid-turn, asking to fix it."""
        if tool_turns(messages) == 0:
            reply = use(WRITE_FIXED)
            return reply if letter in SOLVERS else replace(reply, reasoning_complete=False)
        return say("Done.")

    trained = await train(tmp_path, blaxel, cut_or_fixes)
    assert trained.outcome.ok and rewards(trained) == expected(trained)
    (step,) = trained.steps
    assert step["rollout/termination_reason/truncated"] == trained_zeros(trained) > 0


async def test_a_command_the_deadline_cut_is_stopped_before_the_check(tmp_path, blaxel):
    """A solver fixes the file; then every rollout runs a command that fixes it once
    the check opens a gate, and the agent deadline cuts it while it waits. No clock
    orders the write and the check: only the gate does."""
    late_fix = f"until [ -e ../gate ]; do sleep 0.01; done; {PRINT_FIXED}"

    def fixes_late(messages, letter):
        if tool_turns(messages) == 0 and letter in SOLVERS:
            return use(WRITE_FIXED)
        return use(("shell", {"command": late_fix}))

    survivors: list[str] = []

    def opens_the_gate(name: str, request: dict[str, Any]) -> None:
        sandbox = blaxel.sandboxes[name]
        late = [c for c in sandbox.commands if c.request["command"] == late_fix]
        survivors.extend(name for c in late if not c.killed)
        sandbox.files["/gate"] = ""  # a late command still running would write now

    on_check(blaxel, opens_the_gate)
    trained = await train(
        tmp_path,
        blaxel,
        fixes_late,
        **agent(agent_timeout_seconds=1),
        **{"rollout.rollout_timeout_seconds": 30, "training.max_consecutive_unusable_groups": 2},
    )
    commands = [c for sandbox in blaxel.sandboxes.values() for c in sandbox.commands]
    assert any(c.request["command"] == late_fix for c in commands)  # the premise: one was cut
    # Every late command was stopped before its check opened the gate, so none wrote.
    assert survivors == []
    # A rollout scores 1 where its own write_file ran, and nowhere else.
    writes = sum("FILE" in c.request.get("env", {}) and c.status == "completed" for c in commands)
    assert rewards(trained) == expected(trained, solved=writes)
    roots = list((tmp_path / "sandboxes").iterdir())
    assert sum(map(fixed, roots)) == writes
    assert trained.outcome.ok


@pytest.mark.parametrize("gone", ["unreachable", "TERMINATED", "deleted"])
async def test_a_sandbox_gone_at_grade_is_graded_again_not_scored_0(tmp_path, blaxel, gone):
    """For one try the check cannot reach the sandbox, or fails and it reads as gone."""
    lost: set[str] = set()

    from support.blaxel_api import StatusError

    def goes_once(name: str, request: dict[str, Any]) -> None:
        if name in lost:
            return
        lost.add(name)
        if gone == "unreachable":
            raise httpx.ConnectError("the sandbox is unreachable")
        request["command"] = "exit 1"  # a check it could not finish
        if gone == "TERMINATED":  # the next lookup sees it gone, then it is back
            blaxel.api.sandboxes[name].statuses = ["TERMINATED", "DEPLOYED"]
        else:  # the next lookup finds no sandbox by that name
            blaxel.api.get_errors.append(StatusError(404, f"Sandbox '{name}' not found"))

    on_check(blaxel, goes_once)
    trained = await train(tmp_path, blaxel, fixes_the_port)
    assert trained.outcome.ok
    assert rewards(trained) == expected(trained)
    # Every rollout's first try failed as a grading failure, and was tried again.
    assert [record["retries"] for record in trained.records] == [1] * len(trained.records)


async def test_a_sandbox_gone_for_good_at_grade_is_discarded_never_scored_0(tmp_path, blaxel):
    def unreachable(name: str, request: dict[str, Any]) -> None:
        raise httpx.ConnectError("the sandbox is unreachable")

    on_check(blaxel, unreachable)
    trained = await train(
        tmp_path, blaxel, fixes_the_port, **{"rollout.max_consecutive_rollout_failures": GROUP}
    )
    assert not trained.outcome.ok
    assert trained.records and all(
        record["cause"] == "grading_failed/connect" and record["reward"] is None
        for record in trained.records
    )


async def test_a_sandbox_that_vanishes_mid_check_is_discarded_never_scored_1(tmp_path, blaxel):
    """A sandbox that stops answering mid-check: Blaxel's wait asks until its deadline,
    then gives up. A check that never finished is no pass."""

    def vanishes_mid_check(name: str, request: dict[str, Any]) -> None:
        request["command"] = f"sleep 30; {request['command']}"

        def vanish() -> None:
            blaxel.sandboxes[name].unreachable = True
            blaxel.lose(name, "deleted")

        asyncio.get_running_loop().call_later(0.1, vanish)  # after the wait's first poll

    on_check(blaxel, vanishes_mid_check)
    trained = await train(
        tmp_path,
        blaxel,
        fixes_the_port,
        **agent({"command_timeout_seconds": 1}),
        **{
            "rollout.max_consecutive_rollout_failures": GROUP,
            "training.max_consecutive_unusable_groups": 1,
        },
    )
    assert not trained.outcome.ok
    assert trained.records and all(
        record["reward"] is None and record["cause"].startswith("grading_failed/")
        for record in trained.records
    )


@pytest.mark.parametrize(
    ("options", "problem"),
    [({"max_turns": 0}, "max_turns"), ({"max_turn": 12}, "max_turn")],
    ids=["no_turns", "misspelt"],
)
def test_check_refuses_bad_agent_options(loops_check, options, problem):
    config = cli.load_training_config(CONFIG).rollout_runtime.config
    changed = json.dumps({**config, "options": {**config["options"], **options}})
    code, _, err = loops_check(CONFIG, "--rollout_runtime.config", changed)
    assert code == 2
    assert "is not a valid ToolOptions" in err and problem in err


def test_every_shipped_task_is_unsolved_as_written(tmp_path):
    """Each task's check fails on its files as given, so a 1 means the agent changed them."""
    import subprocess

    for task_id, task in SHIPPED.items():
        work = tmp_path / task_id
        for name, text in task["files"].items():
            (work / name).parent.mkdir(parents=True, exist_ok=True)
            (work / name).write_text(text)
        check = subprocess.run(
            ["/bin/sh", "-c", task["check"]], cwd=work, capture_output=True, check=False
        )
        assert check.returncode != 0, task_id
