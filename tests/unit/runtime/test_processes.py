"""Agent processes: started apart, given exactly their environment, always killable.

Every test runs a real operating-system process from a small script and observes
it from outside: what it printed, what the supervisor returned, and whether any of
its process group is still alive.
"""

import asyncio
import json
import logging
import os
import shutil
import signal
import sys
import textwrap
import time

import pytest
from pydantic import SecretStr
from support.cancellation import (
    Landing,
    Response,
    Trigger,
    Work,
    WorkFailed,
    cancellation_matrix,
    settle_loop,
)
from support.harness import TASK, Runtime, generate, rig

from loops_cookbook.runtime import processes
from loops_cookbook.runtime.api import (
    EpisodeResult,
    GradeResult,
    Phase,
    ProcessCommand,
)
from loops_cookbook.runtime.driver import Discard
from loops_cookbook.runtime.processes import (
    REDACTED,
    TAIL_BYTES,
    AttemptProcessSupervisor,
    ProcessesClosed,
    ProcessLaunchFailure,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")


@pytest.fixture
def script(tmp_path):
    """Write a Python script and return the command that runs it in ``tmp_path``."""

    def write(body, *, env=None, interpreter=sys.executable):
        path = tmp_path / f"agent_{len(list(tmp_path.glob('agent_*.py')))}.py"
        path.write_text(textwrap.dedent(body))
        return ProcessCommand(argv=(interpreter, str(path)), cwd=tmp_path, env=env or {})

    return write


@pytest.fixture
def supervisor():
    return AttemptProcessSupervisor(0.2, environ={"PATH": os.environ["PATH"], "HOME": "/home/x"})


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def pids_from(tmp_path):
    return [int(line) for line in (tmp_path / "pids").read_text().split()]


async def started_both(tmp_path):
    """Until the agent and its child have both recorded their pids; bounded."""
    async with asyncio.timeout(10):
        while not (tmp_path / "pids").exists() or len(pids_from(tmp_path)) < 2:
            await asyncio.sleep(0.02)


SPAWNS_A_CHILD = """
    import os, signal, subprocess, sys, time
    stubborn = "stubborn" in sys.argv[1:]
    ignore = "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); " if stubborn else ""
    child = subprocess.Popen([sys.executable, "-c", ignore + "import time; time.sleep(60)"])
    if stubborn:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    open("pids", "w").write(f"{os.getpid()}\\n{child.pid}\\n")
    time.sleep(60)
"""


# ------------------------------------------------------------------- start


async def test_a_process_starts_alone_with_stdin_closed_in_its_directory(
    tmp_path, script, supervisor
):
    result = await supervisor.run(
        script(
            """
            import json, os, sys
            print(json.dumps({
                "own_group": os.getpgid(0) == os.getpid(),
                "stdin_is_devnull": os.path.samestat(os.fstat(0), os.stat(os.devnull)),
                "cwd": os.getcwd(),
            }))
            """
        )
    )
    seen = json.loads(result.stdout_tail)
    assert seen["own_group"] and seen["stdin_is_devnull"]
    assert os.path.samefile(seen["cwd"], tmp_path)
    assert result.exit_code == 0 and result.signal is None


async def test_a_process_runs_another_interpreter_without_the_cookbooks_import_path(
    script, supervisor
):
    other = shutil.which("python3", path="/usr/bin:/bin")
    if other is None or os.path.realpath(other) == os.path.realpath(sys.executable):
        pytest.skip("no interpreter other than the cookbook's")
    result = await supervisor.run(
        script(
            """
            import importlib.util, sys
            print(sys.executable, importlib.util.find_spec("loops_cookbook") is None)
            """,
            interpreter=other,
        )
    )
    executable, isolated = result.stdout_tail.split()
    assert os.path.realpath(executable) != os.path.realpath(sys.executable)
    assert isolated == "True"


async def test_the_environment_is_the_allowlist_plus_exactly_the_commands(script):
    ambient = {"PATH": os.environ["PATH"], "LC_ALL": "C", "AWS_SECRET_ACCESS_KEY": "ambient"}
    supervisor = AttemptProcessSupervisor(0.2, environ=ambient)
    command = script(
        "import json, os; print(json.dumps(dict(os.environ)))",
        env={"MODEL_URL": "http://127.0.0.1:1/v1", "MODEL_KEY": SecretStr("k-secret")},
    )
    env = json.loads((await supervisor.run(command)).stdout_tail.replace(REDACTED, "k-secret"))
    for platform_extra in ("__CF_USER_TEXT_ENCODING", "__PYVENV_LAUNCHER__"):
        env.pop(platform_extra, None)
    assert env == {
        "PATH": ambient["PATH"],
        "LC_ALL": "C",
        "MODEL_URL": "http://127.0.0.1:1/v1",
        "MODEL_KEY": "k-secret",
    }


async def test_a_secret_never_reaches_argv_output_or_logs(script, supervisor, caplog):
    caplog.set_level(logging.DEBUG)
    secret = SecretStr("tok-9f3e")
    result = await supervisor.run(
        script(
            """
            import os, sys
            print("key:", os.environ["KEY"]); print("key:", os.environ["KEY"], file=sys.stderr)
            """,
            env={"KEY": secret},
        )
    )
    assert "tok-9f3e" not in result.stdout_tail + result.stderr_tail + caplog.text
    assert f"key: {REDACTED}" in result.stdout_tail and f"key: {REDACTED}" in result.stderr_tail
    with pytest.raises(TypeError, match="argv"):
        await supervisor.run(ProcessCommand(argv=("echo", secret), cwd=script("pass").cwd))  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("offset", [0, 10, 29], ids=["whole", "cut_10", "cut_29"])
async def test_a_secret_the_tail_boundary_cuts_is_still_redacted(script, supervisor, offset):
    """The kept tail can start inside a secret; none of it may show."""
    secret = "SECRETSECRETSECRET-abcdef123456"  # 31 characters
    body = f"import os, sys; sys.stdout.write(os.environ['K'] + 'x' * ({TAIL_BYTES} - {offset}))"
    result = await supervisor.run(script(body, env={"K": SecretStr(secret)}))
    assert len(result.stdout_tail) <= TAIL_BYTES
    assert not any(secret[i:] in result.stdout_tail for i in range(len(secret) - 3))


async def test_argv_may_carry_empty_arguments_after_the_program(tmp_path, supervisor):
    command = ProcessCommand(
        argv=(sys.executable, "-c", "import sys; print(sys.argv[1:])", "", "x"), cwd=tmp_path
    )
    assert (await supervisor.run(command)).stdout_tail == "['', 'x']\n"


async def test_output_is_kept_as_bounded_tails(script, supervisor):
    result = await supervisor.run(script("import sys; sys.stdout.write('x' * 1_000_000 + 'END')"))
    assert len(result.stdout_tail) <= TAIL_BYTES and result.stdout_tail.endswith("END")


@pytest.mark.parametrize(
    ("body", "code", "sig"),
    [
        ("import sys; sys.exit(3)", 3, None),
        ("import os, signal; os.kill(os.getpid(), signal.SIGUSR1)", -1, signal.SIGUSR1),
    ],
)
async def test_an_exit_is_returned_not_raised(script, supervisor, body, code, sig):
    result = await supervisor.run(script(body))
    assert (result.exit_code, result.signal) == (code, sig)


@pytest.mark.parametrize("broken", ["missing_executable", "missing_directory"])
async def test_a_process_that_cannot_start_is_a_launch_failure(tmp_path, supervisor, broken):
    command = (
        ProcessCommand(argv=(str(tmp_path / "no-such-agent"),), cwd=tmp_path)
        if broken == "missing_executable"
        else ProcessCommand(argv=(sys.executable, "-c", "pass"), cwd=tmp_path / "gone")
    )
    with pytest.raises(ProcessLaunchFailure, match="could not start"):
        await supervisor.run(command)


# --------------------------------------------------------------- stopping


@pytest.mark.parametrize("cancels", [1, 2], ids=["cancelled", "cancelled_during_the_kill"])
@pytest.mark.parametrize("stubborn", [False, True], ids=["obeys_sigterm", "ignores_sigterm"])
async def test_cancelling_run_ends_the_whole_group_before_it_returns(
    tmp_path, script, supervisor, stubborn, cancels
):
    command = script(SPAWNS_A_CHILD)
    if stubborn:
        command = ProcessCommand(argv=(*command.argv, "stubborn"), cwd=command.cwd)
    running = asyncio.create_task(supervisor.run(command))
    await started_both(tmp_path)
    await asyncio.sleep(0.1)  # let the stubborn one install its handler
    started = time.monotonic()
    running.cancel()
    if cancels == 2:
        await asyncio.sleep(0.05)  # inside the stubborn one's kill grace
        running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, 10)
    assert not any(alive(pid) for pid in pids_from(tmp_path))
    if stubborn:
        assert time.monotonic() - started >= 0.2  # SIGKILL only after the grace


@cancellation_matrix(
    triggers=(Trigger.DEADLINE,),
    points={"stop": (Landing.NOTHING, Landing.CANCEL)},
    responses=(Response.STOPS, Response.FAILS),
)
async def test_tails_still_draining_at_their_deadline_are_stopped_whatever_cancels_run(
    script, supervisor, monkeypatch, case
):
    """The output tails get ``DRAIN_SECONDS`` after the group ends, then are stopped;
    a cancellation of ``run()`` while they stop still propagates."""
    work = Work(case.response)
    monkeypatch.setattr(processes, "DRAIN_SECONDS", 0.05)
    monkeypatch.setattr(processes._Tail, "drain", lambda self, stream: work())
    running = asyncio.create_task(supervisor.run(script('print("done")')))
    try:
        async with asyncio.timeout(10):
            await work.cancelled.wait()  # the deadline passed with the tails still draining
            if case.at("stop") is Landing.CANCEL:
                running.cancel()
                await settle_loop()
            assert not running.done()
            work.let_go.set()
            outcome = (await asyncio.gather(running, return_exceptions=True))[0]
        if case.cancelled:
            assert isinstance(outcome, asyncio.CancelledError)
        elif case.response is Response.FAILS:
            assert isinstance(outcome, WorkFailed)
        else:
            assert outcome.exit_code == 0
    finally:
        work.end()


async def test_a_leader_that_exits_returns_and_takes_what_it_left_with_it(
    tmp_path, script, supervisor
):
    body = """
        import subprocess, sys
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open("child", "w").write(str(child.pid))
        print("done")
    """
    result = await asyncio.wait_for(supervisor.run(script(body)), 10)
    assert result.exit_code == 0 and result.stdout_tail == "done\n"  # the child held stdout
    assert not alive(int((tmp_path / "child").read_text()))


def open_descriptors():
    return len(os.listdir("/dev/fd"))


async def test_no_pipe_outlives_its_run_even_if_a_descendant_escaped_the_group(script, supervisor):
    """A descendant in its own session keeps the pipes open; our ends are still closed."""
    body = """
        import os, sys
        if os.fork() == 0:
            os.setsid()
            import time; time.sleep(3)
            os._exit(0)
        print("leader done")
    """
    before = open_descriptors()
    result = await asyncio.wait_for(supervisor.run(script(body)), 10)
    assert result.stdout_tail == "leader done\n"
    assert open_descriptors() == before


async def test_a_blocking_agent_does_not_block_the_event_loop(script, supervisor):
    ticks = 0

    async def tick():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    ticker = asyncio.create_task(tick())
    await supervisor.run(script("import time; time.sleep(0.5)"))  # a blocking call
    ticker.cancel()
    assert ticks > 20


# ---------------------------------------------------------- through the driver


def agent_rig(tmp_path, runtime, **settings):
    return rig(
        runtime,
        phases=frozenset({Phase.SAMPLING}),
        process_kill_grace_seconds=0.2,
        cancellation_grace_seconds=0.2,
        **settings,
    )


def stubborn_command(tmp_path):
    path = tmp_path / "stubborn.py"
    path.write_text(textwrap.dedent(SPAWNS_A_CHILD))
    return ProcessCommand(argv=(sys.executable, str(path), "stubborn"), cwd=tmp_path)


async def test_a_rollout_deadline_kills_a_stubborn_agent_without_a_systemic_failure(tmp_path):
    async def sample(attempt):
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            await generate(attempt)
            await attempt.services.processes.run(stubborn_command(tmp_path))
        return EpisodeResult()

    r = agent_rig(tmp_path, Runtime(on_sample=sample), rollout_timeout_seconds=0.5)
    assert await r.rollout() is Discard.AGENT_TIMEOUT
    assert not r.systemic
    assert not any(alive(pid) for pid in pids_from(tmp_path))


async def held_sample(attempt):
    async with attempt.services.capacity.hold(Phase.SAMPLING):
        await generate(attempt)
    return EpisodeResult()


async def test_a_grading_deadline_kills_its_verifier_process(tmp_path):
    async def grade(attempt, rollout):
        await attempt.services.processes.run(stubborn_command(tmp_path))
        return GradeResult(reward=1.0)

    runtime = Runtime(on_sample=held_sample, on_grade=grade)
    r = agent_rig(tmp_path, runtime, grading_timeout_seconds=0.5)
    assert await r.rollout("g-r1") is Discard.GRADING_FAILED
    assert not r.systemic
    assert not any(alive(pid) for pid in pids_from(tmp_path))


@pytest.mark.parametrize("ending", ["sealed", "aborted"])
async def test_processes_a_runtime_did_not_await_end_with_their_phase(tmp_path, ending):
    seen_by_hooks = []

    async def sample(attempt):
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            await generate(attempt)
        asyncio.get_running_loop().create_task(
            attempt.services.processes.run(stubborn_command(tmp_path))
        )
        await started_both(tmp_path)
        if ending == "aborted":
            raise RuntimeError("the runtime failed after starting its agent")
        return EpisodeResult()

    async def grade(attempt, rollout):
        seen_by_hooks.append(any(alive(pid) for pid in pids_from(tmp_path)))
        return GradeResult(reward=1.0)

    async def abort(attempt, reason):
        seen_by_hooks.append(any(alive(pid) for pid in pids_from(tmp_path)))

    runtime = Runtime(on_sample=sample, on_grade=grade, on_abort=abort)
    await agent_rig(tmp_path, runtime).rollout("g-r1")
    assert seen_by_hooks == [False]  # gone before grade() after sealing, or abort()


def tokened(tmp_path, token):
    """A long-lived agent whose argv carries ``token``, so it can be found by any means."""
    return ProcessCommand(
        argv=(sys.executable, "-c", "import time; time.sleep(60)", token), cwd=tmp_path
    )


def running_with(token):
    import subprocess

    found = subprocess.run(["pgrep", "-f", token], capture_output=True, text=True, check=False)
    return found.stdout.split()


async def test_a_process_spawning_as_the_rollout_seals_is_ended_before_grade(tmp_path):
    """sample() leaves a run() inside its spawn and returns; sealing must still end it."""
    token = f"seal-straddle-{os.getpid()}"
    seen = []

    async def sample(attempt):
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            await generate(attempt)
        asyncio.get_running_loop().create_task(
            attempt.services.processes.run(tokened(tmp_path, token))
        )
        await asyncio.sleep(0)  # the run() is now inside its spawn
        return EpisodeResult()

    async def grade(attempt, rollout):
        seen.append(running_with(token))
        return GradeResult(reward=1.0)

    await agent_rig(tmp_path, Runtime(on_sample=sample, on_grade=grade)).rollout("g-r1")
    assert seen == [[]]


async def test_nothing_starts_once_the_attempts_last_hook_has_returned(tmp_path):
    """A task of the rollout runtime that starts a process late cannot outlive the attempt."""
    token = f"after-finish-{os.getpid()}"
    late = []

    async def sample(attempt):
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            await generate(attempt)
        return EpisodeResult()

    async def finish(attempt, rollout, disposition):
        processes = attempt.services.processes
        asyncio.get_running_loop().create_task(processes.run(tokened(tmp_path, token)))
        async with asyncio.timeout(10):
            while not running_with(token):
                await asyncio.sleep(0.02)
        late.append(processes)

    r = agent_rig(tmp_path, Runtime(on_sample=sample, on_finish=finish))
    await r.rollout("g-r1")
    assert running_with(token) == []  # started in finish(), ended with the attempt
    with pytest.raises(ProcessesClosed):
        await late[0].run(tokened(tmp_path, token))


async def test_a_second_cancellation_during_the_kill_still_cancels_the_hook_first(tmp_path):
    """sample() must be cancelled, and stop, before abort() runs."""
    events = []

    async def sample(attempt):
        async with attempt.services.capacity.hold(Phase.SAMPLING):
            await generate(attempt)
            try:
                await attempt.services.processes.run(stubborn_command(tmp_path))
                await asyncio.Event().wait()  # goes on after its process, until cancelled
            except asyncio.CancelledError:
                events.append("sample cancelled")
                raise
        return EpisodeResult()

    async def abort(attempt, reason):
        events.append("abort")

    r = agent_rig(tmp_path, Runtime(on_sample=sample, on_abort=abort))
    sampling = asyncio.create_task(r.driver.sample(TASK, "g", "g-r1"))
    await started_both(tmp_path)
    await asyncio.sleep(0.1)  # the stubborn agent ignores SIGTERM from here on
    sampling.cancel()
    await asyncio.sleep(0.05)  # inside the kill's grace
    sampling.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(sampling, 10)
    await r.driver.drain()
    assert events == ["sample cancelled", "abort"]
    assert not any(alive(pid) for pid in pids_from(tmp_path))
