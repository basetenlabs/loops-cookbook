"""The custom local pattern's grading: one attempt through the real driver, with the
episode's own ``grade()`` and verifier, and the Blaxel episode's answer check."""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace as NS

import local_episode
import local_sandbox
import pytest
from pydantic import SecretStr
from support.builders import runtime_context, task_file

from loops_cookbook.episodes import Sampling
from loops_cookbook.model_access import ModelAccessMode
from loops_cookbook.resources import SandboxGone

from .custom_local import DEADLINES, EXAMPLE, SANDBOX, TASKS

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")


def counting(tmp_path: Path, name: str, body: str) -> tuple[str, ...]:
    """A program that logs each run to ``tmp_path/<name>.log``, then runs ``body``."""
    script = tmp_path / f"{name}.py"
    log = tmp_path / f"{name}.log"
    script.write_text(
        textwrap.dedent(
            f"""
            import json, os, pathlib, runpy, shutil, sys, time
            sys.path.insert(0, {str(EXAMPLE)!r})  # as if run from beside the example
            log = pathlib.Path({str(log)!r})
            runs = len(log.read_text().splitlines()) if log.exists() else 0
            log.write_text("run\\n" * (runs + 1))
            """
        )
        + textwrap.dedent(body)
    )
    return (sys.executable, str(script))


def leaves(answer=None, *, result=None, lose_sandbox=False, then_sleep=False) -> str:
    """An agent program's body: what it leaves in the sandbox and its result file."""
    lines = []
    if answer is not None:
        lines += [
            f"pathlib.Path(os.environ['SANDBOX_DIR'], 'answer.txt').write_text({answer!r})",
            f"pathlib.Path('result.json').write_text(json.dumps({{'answer': {answer!r}}}))",
        ]
    if isinstance(result, bytes):
        lines.append(f"pathlib.Path('result.json').write_bytes({result!r})")
    elif result is not None:
        lines.append(f"pathlib.Path('result.json').write_text({result!r})")
    if lose_sandbox:
        lines.append("shutil.rmtree(os.environ['SANDBOX_DIR'])")
    if then_sleep:
        lines.append("time.sleep(60)")
    return "\n".join(lines) + "\n"


VERIFIER = str(EXAMPLE / "verifier.py")
FLAKY = f"""
if runs == 0:
    sys.exit("the verifier's host went away")
runpy.run_path({VERIFIER!r}, run_name="__main__")
"""
GARBLED = "print('reward: most of it')"
HONEST = f"runpy.run_path({VERIFIER!r}, run_name='__main__')"
# The verifier against a Blaxel sandbox whose file read fails as the SDK fails it: a
# ResponseError with the status, or, for a body that is not JSON, the SDK's own
# json.loads error, raised before it checks the status.
BLAXEL_READ_FAILS = """
import json, httpx
import blaxel.core
from blaxel.core.sandbox.types import ResponseError

os.environ["BLAXEL_SANDBOX_NAME"] = "fake"

class Files:
    async def read(self, path):
        if {status!r} == "not_json":
            json.loads("<html>502 Bad Gateway</html>")
        raise ResponseError(httpx.Response({status!r}, json={{"error": "read failed"}}))

class Sandbox:
    fs = Files()

async def get(name):
    return Sandbox()

blaxel.core.SandboxInstance.get = get
runpy.run_path({verifier!r}, run_name="__main__")
"""
# The verifier against a Blaxel sandbox that holds the answer ``answer``.
BLAXEL_ANSWERS = """
import blaxel.core

os.environ["BLAXEL_SANDBOX_NAME"] = "fake"

class Files:
    async def read(self, path):
        return {answer!r}

class Sandbox:
    fs = Files()

async def get(name):
    return Sandbox()

blaxel.core.SandboxInstance.get = get
runpy.run_path({verifier!r}, run_name="__main__")
"""


class Scripted(local_episode.DirectoryAgentEpisode):
    """The example's own grade(), after an agent program that leaves a scripted result.

    The driver's rig hands out direct model access, so the one model call is direct
    and the program gets only the sandbox and the instruction.
    """

    model_access = ModelAccessMode.DIRECT

    async def run(self, sample: Sampling):
        model = sample.model
        sandbox = sample.sandbox
        assert sandbox is not None
        await model.llm().conversation().generate(messages=[{"role": "user", "content": "hi"}])
        ran = await sample.processes.run(
            local_episode.command(self.options.agent),
            env={**sandbox.environment, "AGENT_INSTRUCTION": self.task.value.instruction},
        )
        return {"agent_exit": local_episode.ended(ran)}


@pytest.fixture
async def attempt(tmp_path, sandboxes, monkeypatch):
    """Sample and grade one attempt through the real driver, on a directory sandbox."""
    from support.harness import rig
    from support.resources import episodes, local

    from loops_cookbook.episodes import create_runtime
    from loops_cookbook.resources.registry import ResourceRegistry
    from loops_cookbook.runtime.context import EvidenceRecord

    provider = local_sandbox.create({"root": str(sandboxes), **DEADLINES})
    registry = ResourceRegistry.open(
        tmp_path / "state", {"local_sandbox": provider}, mode="write", run_id="r", kind=local
    )

    async def run(
        grade_against="files",
        *,
        agent="",
        verifier=HONEST,
        tasks=TASKS[:1],
        agent_timeout=5.0,
    ):
        config = {
            "episode": f"{__name__}:Scripted",
            "agent_timeout_seconds": agent_timeout,
            "options": {
                "agent": counting(tmp_path, "agent", agent),
                "verifier": counting(tmp_path, "verifier", verifier),
                "grade_against": grade_against,
            },
            "sandbox": SANDBOX,
        }
        limits = NS(
            rollout_timeout_seconds=None,
            cancellation_grace_seconds=0.5,
            process_kill_grace_seconds=0.5,
            provider_deadlines={"local_sandbox": provider.deadlines},
            trajectory_tokens=None,
        )
        runtime = create_runtime(
            config,
            context=runtime_context(
                task_file(tmp_path, tasks),
                limits=limits,
                work_dir=tmp_path / "work",
                evidence=EvidenceRecord(),
                sandbox_providers={"local_sandbox": local_sandbox.Directory},
            ),
        )
        r = rig(
            runtime,
            max_rollout_retries=2,
            grading_timeout_seconds=30.0,  # the Blaxel SDK's first import is slow
            resources=episodes(registry),
            rollout_timeout_seconds=runtime.capabilities.rollout_timeout_seconds,
            cancellation_grace_seconds=limits.cancellation_grace_seconds,
            process_kill_grace_seconds=limits.process_kill_grace_seconds,
        )
        (task,) = await runtime.prepare()
        outcome = await asyncio.wait_for(r.rollout("g-r1", task), 60)  # a hang fails, not spins
        run.retried = {k: n for k, n in r.driver.gauges().items() if k.startswith("retried/")}
        log = tmp_path / "verifier.log"
        return outcome, len(log.read_text().splitlines()) if log.exists() else 0

    yield run
    await registry.close()


@pytest.mark.parametrize("grade_against", ["files", "sandbox"])
async def test_a_verifier_that_fails_is_retried_because_it_is_pure(attempt, grade_against):
    outcome, runs = await attempt(grade_against, agent=leaves("B"), verifier=FLAKY)
    assert outcome.reward == 1.0 and runs == 2


@pytest.mark.parametrize("grade_against", ["files", "sandbox"])
async def test_an_agent_that_left_no_answer_scores_zero(attempt, grade_against):
    outcome, runs = await attempt(grade_against, agent=leaves(None))
    assert outcome.reward == 0.0 and runs == 1


async def test_an_agent_cut_at_its_deadline_is_graded_on_what_it_left(attempt):
    outcome, runs = await attempt(agent=leaves("B", then_sleep=True), agent_timeout=1.0)
    assert outcome.reward == 1.0 and runs == 1
    assert "agent_exit" not in outcome.metadata  # run() was cut before it could say


# The example's agent program, killed by the kernel partway through writing its
# result file: the file-size limit leaves room for the answer in the sandbox, not
# for the whole result.
KILLED_MID_WRITE = """
import resource, signal
import agent_program

answer = "B" * 100_000
agent_program.ask = lambda instruction: answer
signal.signal(signal.SIGXFSZ, signal.SIG_DFL)  # the kernel's signal kills it
resource.setrlimit(resource.RLIMIT_FSIZE, (len(answer) + 5, len(answer) + 5))
agent_program.main()
"""


async def test_an_agent_killed_mid_write_left_no_answer_and_scores_zero(attempt):
    outcome, runs = await attempt(agent=KILLED_MID_WRITE)
    assert outcome.metadata["agent_exit"].startswith("signal_")  # it died writing
    assert outcome.reward == 0.0 and runs == 1  # scored, not discarded


async def test_a_sandbox_that_is_gone_is_not_a_missing_answer(attempt):
    from loops_cookbook.runtime.driver import Discard

    outcome, runs = await attempt("sandbox", agent=leaves("B", lose_sandbox=True))
    assert outcome is Discard.GRADING_FAILED and runs == 3  # retried, never scored


async def test_a_verifier_that_prints_no_reward_fails_grading_without_a_retry(attempt):
    from loops_cookbook.runtime.driver import Discard

    outcome, runs = await attempt(agent=leaves("B"), verifier=GARBLED)
    assert outcome is Discard.GRADING_FAILED and runs == 1


@pytest.mark.parametrize(
    "result",
    ["{not json", '{"answer": 7}', '["B"]', b'{"answer": "\xff"}'],
    ids=["not_json", "not_a_string", "no_answer", "not_utf8"],
)
async def test_a_result_file_that_is_not_a_result_fails_grading_without_a_retry(attempt, result):
    from loops_cookbook.runtime.driver import Discard

    outcome, runs = await attempt(agent=leaves(result=result))
    assert outcome is Discard.GRADING_FAILED and runs == 1


@pytest.mark.parametrize(
    ("status", "scored"),
    [(404, True), (503, False), ("not_json", False)],
    ids=["missing_file", "unavailable", "not_json_body"],
)
async def test_a_blaxel_read_reports_only_a_404_as_a_missing_answer(attempt, status, scored):
    """The verifier scores a 404 as a missing answer, 0; any other failed read is retried.

    The 404 may be the sandbox's, not the file's: the episode's alive() after it is
    locked below. This attempt's own sandbox is a directory, still there.
    """
    from loops_cookbook.runtime.driver import Discard

    body = BLAXEL_READ_FAILS.format(status=status, verifier=VERIFIER)
    outcome, runs = await attempt("sandbox", agent=leaves("B"), verifier=body)
    if scored:
        assert outcome.reward == 0.0 and runs == 1
    else:
        assert outcome is Discard.GRADING_FAILED and runs == 3  # retried, never scored


async def run_program(argv, *, env):
    """The episode's ``run_process``: ``argv`` run to its end, as a ``ProcessResult``."""
    plain = {
        key: value.get_secret_value() if isinstance(value, SecretStr) else value
        for key, value in env.items()
    }
    ran = await asyncio.create_subprocess_exec(
        *argv,
        env={**os.environ, **plain},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await ran.communicate()
    return NS(
        exit_code=ran.returncode, signal=None, stdout_tail=out.decode(), stderr_tail=err.decode()
    )


def on_sandbox(sandbox, tmp_path, verifier, grade_against="sandbox"):
    """The Blaxel episode, for the task answered C, graded by ``verifier``, on ``sandbox``."""
    from loops_cookbook.episodes import Task

    options = local_episode.LocalOptions(
        agent=["true"],
        verifier=list(counting(tmp_path, "verifier", verifier)),
        grade_against=grade_against,
    )
    task = Task("0", TASKS[1], local_episode.AnswerTask.model_validate(TASKS[1]))
    episode = local_episode.BlaxelAgentEpisode(task, options)
    grading = NS(execution_sandbox=sandbox, processes=NS(run=run_program))
    return episode, grading


def lookups(blaxel):
    return [name for call, name in blaxel.api.calls if call == "get"]


@pytest.mark.parametrize("how", ["alive", "deleted"])
async def test_an_answer_missing_from_a_blaxel_sandbox_scores_zero_only_while_it_is_alive(
    tmp_path, blaxel, blaxel_sandbox, how
):
    """The verifier's 404 is a missing answer only if the sandbox is still alive after
    it: a deleted one stays readable, so its 404 says nothing about the answer."""
    from loops_cookbook.runtime import GradeResult

    sandbox = await blaxel_sandbox()
    episode, grading = on_sandbox(
        sandbox, tmp_path, BLAXEL_READ_FAILS.format(status=404, verifier=VERIFIER)
    )
    if how != "alive":
        blaxel.lose(sandbox.name, how)
    asked = len(lookups(blaxel))
    if how == "alive":
        assert await episode.grade(grading) == GradeResult(reward=0.0)
    else:
        with pytest.raises(SandboxGone, match=sandbox.name):
            await episode.grade(grading)  # a failed grading, tried again
    assert lookups(blaxel)[asked:] == [sandbox.name]  # asked once, after the read


@pytest.mark.parametrize("grade_against", ["files", "sandbox"])
async def test_only_an_answer_missing_from_the_sandbox_asks_whether_it_is_alive(
    tmp_path, blaxel, blaxel_sandbox, grade_against
):
    """An answer the verifier read proves the sandbox, and a result file needs none."""
    from loops_cookbook.runtime import GradeResult

    sandbox = await blaxel_sandbox()
    verifier = (
        BLAXEL_ANSWERS.format(answer="C", verifier=VERIFIER)
        if grade_against == "sandbox"
        else HONEST
    )
    episode, grading = on_sandbox(sandbox, tmp_path, verifier, grade_against)
    blaxel.lose(sandbox.name, "deleted")  # asking now would fail grading
    asked = len(lookups(blaxel))
    expected = 1.0 if grade_against == "sandbox" else 0.0  # no result file: a missing answer
    assert await episode.grade(grading) == GradeResult(reward=expected)
    assert lookups(blaxel)[asked:] == []
