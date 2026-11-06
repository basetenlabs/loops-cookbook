"""The custom local pattern, trained: an agent process, a run-owned sandbox, a pure verifier.

The episode is ``local_episode.LocalAgentEpisode``, named for a directory sandbox as
``DirectoryAgentEpisode`` and for Blaxel as ``BlaxelAgentEpisode``. These tests run it
through a real ``Run``: the agent is a separate Python program that reaches the model
over the run's gateway, and the sandbox is a directory the run's resource registry
creates and deletes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import local_episode
import pytest
from support.builders import task_file
from support.offline_run import offline_config, train_offline

from loops_cookbook.episodes import Closing, Grading, Sampling
from loops_cookbook.providers import create_adapters
from loops_cookbook.rl import cli

from .custom_local import EXAMPLE, SANDBOX, TASKS

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")

FACTORY = "loops_cookbook.episodes:create_runtime"
OPTIONS = cli.load_training_config(EXAMPLE / "train.toml").runtime.config["options"]


class Seen:
    """What the watched episodes observed, across every attempt of one test."""

    def __init__(self) -> None:
        self.connections: dict[str, object] = {}
        self.rollouts: list[object] = []
        self.after_seal: list[int] = []
        self.live_at_grade: list[bool] = []
        self.live_at_close: list[bool] = []


SEEN = Seen()


class Watched(local_episode.DirectoryAgentEpisode):
    """The example episode, observed from outside at each hook."""

    async def run(self, sample: Sampling):
        model = sample.model
        # The same producer's credential the agent receives.
        SEEN.connections[sample.attempt_id] = await model.http_connection()
        return await super().run(sample)

    async def grade(self, grading: Grading):
        rollout = grading.rollout
        import aiohttp

        connection = SEEN.connections[grading.attempt_id]
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                connection.base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {connection.api_key.get_secret_value()}"},
                json={"model": connection.model, "messages": [{"role": "user", "content": "x"}]},
            ) as response,
        ):
            SEEN.after_seal.append(response.status)
        SEEN.live_at_grade.append(self._live(grading.execution_sandbox))
        SEEN.rollouts.append(rollout)
        return await super().grade(grading)

    async def close(self, closing: Closing):
        SEEN.live_at_close.append(self._live(closing.execution_sandbox))

    def _live(self, sandbox) -> bool:
        return Path(str(sandbox.environment["SANDBOX_DIR"])).is_dir()


@pytest.fixture
def train(tmp_path, sandboxes):
    """Train one step of the example as the watched episode; returns what was observed."""
    SEEN.__init__()

    async def run(*, grade_against, agent=None, tasks=TASKS, training=None):
        options = {**OPTIONS, "grade_against": grade_against}
        if agent is not None:
            options["agent"] = agent
        settings = offline_config(
            EXAMPLE / "train.toml",
            tmp_path,
            {
                "runtime.config": {
                    "episode": f"{__name__}:Watched",
                    "agent_timeout_seconds": 90,
                    "sandbox": SANDBOX,
                    "options": options,
                },
                "runtime.files": {"tasks": str(task_file(tmp_path, tasks))},
                "providers.local_sandbox.root": str(sandboxes),
                "training.max_steps": 1,
                "training.group_size": 4,
                **(training or {}),
            },
        )
        adapters = await create_adapters(settings.providers)
        # A broken agent fails within the default timeout, not hangs.
        outcome, trainer, _ = await train_offline(settings, config_dir=EXAMPLE, adapters=adapters)
        return outcome, trainer

    yield run
    SEEN.__init__()


def test_the_example_config_selects_the_episode_and_its_provider():
    config = cli.load_training_config(EXAMPLE / "train.toml")
    assert config.runtime.factory == FACTORY
    assert config.runtime.model_access is None  # the class states it
    assert config.runtime.config["episode"] == "local_episode:DirectoryAgentEpisode"
    assert config.runtime.config["options"] == {
        "agent": ["python3", "./agent_program.py"],
        "verifier": ["python3", "./verifier.py"],
        "grade_against": "files",
    }
    assert config.providers["local_sandbox"]["adapter"] == "local_sandbox:create"


@pytest.mark.parametrize("grade_against", ["files", "sandbox"])
async def test_the_agent_trains_over_the_gateway_in_a_sandbox_the_run_owns(
    train, sandboxes, grade_against
):
    outcome, trainer = await train(grade_against=grade_against)
    assert outcome.ok and outcome.steps == 1
    rewards = [reward for step in trainer.steps for reward in step["rewards"]]
    assert rewards and all(reward in (0.0, 1.0) for reward in rewards)
    # Over the loopback gateway, one credential per attempt, revoked at sealing.
    connections = list(SEEN.connections.values())
    assert {urlparse(c.base_url).hostname for c in connections} == {"127.0.0.1"}
    assert len({c.api_key.get_secret_value() for c in connections}) == len(connections)
    assert SEEN.after_seal and set(SEEN.after_seal) == {401}
    # Every sample is attributed to its attempt, and the agent's exit is recorded.
    assert all(rollout.samples for rollout in SEEN.rollouts)
    assert {rollout.metadata["agent_exit"] for rollout in SEEN.rollouts} == {"status_0"}
    # The sandbox lives through grading and close(); the run releases every one after.
    assert set(SEEN.live_at_grade) == {True} and set(SEEN.live_at_close) == {True}
    assert [path.name for path in sandboxes.iterdir() if not path.name.startswith(".")] == []


FAILING_AGENT = """
import json, os, pathlib, signal, sys
sys.path.insert(0, {example!r})
import agent_program, sandbox_io

agent_program.ask(os.environ["AGENT_INSTRUCTION"])  # the episode's one model call
turn = 0
while True:  # this agent's turn, counted atomically across concurrent agents
    try:
        os.close(os.open(pathlib.Path({turns!r}, str(turn)), os.O_CREAT | os.O_EXCL))
        break
    except FileExistsError:
        turn += 1
if turn % 2 == 0:  # half of them answer correctly, from the instruction, before they fail
    instruction = os.environ["AGENT_INSTRUCTION"]
    answer = instruction.removeprefix("Say ") if instruction.startswith("Say ") else "?"
    sandbox_io.write("answer.txt", answer)
    pathlib.Path("result.json").write_text(json.dumps({{"answer": answer}}))
if {ends!r} == "signal":
    os.kill(os.getpid(), signal.SIGKILL)
sys.exit(1)
"""


@pytest.mark.parametrize(
    ("grade_against", "ends"),
    [("files", "status"), ("sandbox", "status"), ("files", "signal")],
)
async def test_a_failed_agent_program_is_graded_and_what_the_verifier_says_stands(
    train, tmp_path, grade_against, ends
):
    """Every agent fails; half left the right answer first. Each is scored, none discarded."""
    turns = tmp_path / "turns"
    turns.mkdir()
    script = tmp_path / "failing_agent.py"
    script.write_text(FAILING_AGENT.format(example=str(EXAMPLE), turns=str(turns), ends=ends))
    outcome, trainer = await train(
        grade_against=grade_against,
        agent=[sys.executable, str(script)],
        tasks=TASKS[1:],  # "Say C": graded against its own answer, C
        training={"training.groups_per_step": 1},
    )
    assert outcome.ok and outcome.steps == 1
    (step,) = trainer.steps
    assert sorted(step["rewards"]) == [0.0, 0.0, 1.0, 1.0]
    ended = {rollout.metadata["agent_exit"] for rollout in SEEN.rollouts}
    assert ended == {"status_1" if ends == "status" else "signal_9"}


async def test_an_agent_program_that_fails_before_calling_the_model_is_unversioned(train, tmp_path):
    """It left nothing to train: never graded, and a run of them stops the run."""
    script = tmp_path / "early_exit_agent.py"
    script.write_text("import sys; sys.exit(1)\n")
    outcome, trainer = await train(
        grade_against="files",
        agent=[sys.executable, str(script)],
        tasks=TASKS[:1],
        training={"training.groups_per_step": 1, "rollout.max_consecutive_rollout_failures": 3},
    )
    assert outcome.status == "aborted" and trainer.steps == [] and SEEN.rollouts == []
    # Those in flight when the run stops are cancelled; every other one was unversioned.
    assert set(outcome.discards) <= {"unversioned", "cancelled"}
    assert outcome.discards["unversioned"] >= 1


PROGRAMS = ("agent_program.py", "verifier.py", "sandbox_io.py")


@pytest.mark.parametrize("program", PROGRAMS)
def test_the_programs_need_nothing_from_the_cookbook(program):
    """A local agent is a program you run: what runs as one imports no cookbook module."""
    import ast

    tree = ast.parse((EXAMPLE / program).read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not {name for name in imported if name.split(".")[0] == "loops_cookbook"}


@pytest.mark.parametrize(
    ("options", "refusal"),
    [
        ({"grade_against": "both"}, "grade_against"),
        ({"verfier": ["python3"]}, "verfier"),
        ({"agent": []}, "agent"),
        ({"verifier": ["python3", 7]}, "verifier.1"),
    ],
    ids=["unknown_grade_against", "misspelt_option", "no_agent", "verifier_not_all_strings"],
)
def test_check_refuses_options_the_episode_cannot_run(loops_check, options, refusal):
    changed = json.dumps({"options": {**OPTIONS, **options}})
    code, _, err = loops_check(EXAMPLE / "train.toml", "--runtime.config", changed)
    assert code == 2
    assert "is not a valid LocalOptions" in err and refusal in err
