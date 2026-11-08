"""Run agent code in a separate process that the cookbook supervises and can kill.

``attempt.services.processes.run(command)`` starts one operating-system process for
an attempt and supervises it to the end.

- The process starts in its own process group, with standard input closed, in
  ``command.cwd``, from ``command.argv``. It never inherits the cookbook's import
  path.
- Its environment is the non-secret allowlist below plus exactly ``command.env``. A
  ``SecretStr`` reaches it only through that environment, and is redacted from the
  output that the cookbook keeps.
- The cookbook keeps standard output and standard error as tails of bounded size. A
  nonzero exit is returned in ``ProcessResult``, not raised.
- Cancelling the caller terminates the whole group with ``SIGTERM``, and then with
  ``SIGKILL`` after ``process_kill_grace_seconds``. ``run()`` returns or raises only
  once the group has exited. A group that outlives ``SIGKILL`` by
  ``GROUP_EXIT_SECONDS`` raises ``ProcessKillFailure``.

The driver also terminates an attempt's processes itself. It terminates every
process when the rollout is sealed, every remaining one before ``finish()`` or
``abort()``, and all of them before a cancelled hook's own grace starts. An agent that
cannot stop cooperatively therefore never makes a hook look non-quiescent. A process
still being spawned counts as live. No process starts while a hook is being stopped,
or once the attempt's last hook has returned. ``run()`` then raises
``ProcessesClosed``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import SecretStr

from loops_cookbook.cancellation import outlast, stop
from loops_cookbook.runtime.api import ProcessCommand, ProcessResult

# Non-secret variables that a process needs to run at all. Nothing else is inherited.
ALLOWED_ENVIRONMENT = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LANGUAGE",
    "TERM",
    "TZ",
    "TMPDIR",
)
ALLOWED_PREFIXES = ("LC_",)
TAIL_BYTES = 64 * 1024
REDACTED = "<redacted>"
GROUP_EXIT_SECONDS = 5.0
# Something outside the group can hold the output pipes open. The tails then end where
# the group did.
DRAIN_SECONDS = 1.0


class ProcessLaunchFailure(RuntimeError):
    """Raised when an agent process could not be started, so nothing ran."""


class ProcessesClosed(RuntimeError):
    """Raised when no agent process may start, because its attempt ended or its hook is stopping."""


class ProcessKillFailure(RuntimeError):
    """Raised when an agent process group is still alive after ``SIGKILL`` and its time limit."""


def _environment(
    inherited: Mapping[str, str], extra: Mapping[str, str | SecretStr]
) -> dict[str, str]:
    env = {
        name: value
        for name, value in inherited.items()
        if name in ALLOWED_ENVIRONMENT or name.startswith(ALLOWED_PREFIXES)
    }
    for name, value in extra.items():
        env[name] = value.get_secret_value() if isinstance(value, SecretStr) else value
    return env


def _check(command: ProcessCommand) -> None:
    argv = command.argv
    if not argv or not isinstance(argv[0], str) or not argv[0]:
        raise TypeError("ProcessCommand.argv must start with the program to run")
    if not all(isinstance(part, str) for part in argv):
        raise TypeError("ProcessCommand.argv must be strings; pass secrets through env")
    for name, value in command.env.items():
        if not isinstance(name, str) or not isinstance(value, str | SecretStr):
            raise TypeError(f"ProcessCommand.env[{name!r}] must be a str or SecretStr")


class _Tail:
    """Keep the last ``TAIL_BYTES`` that a stream produced, with its secrets redacted.

    It keeps the longest secret's length beyond the tail. A secret that ends inside the
    tail is therefore whole when it is redacted, and only then trimmed.
    """

    def __init__(self, secrets: list[str]) -> None:
        self._secrets = sorted(secrets, key=len, reverse=True)
        self._keep = TAIL_BYTES + max((len(s.encode()) for s in secrets), default=0)
        self._data = bytearray()

    async def drain(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(65536):
            self._data.extend(chunk)
            if len(self._data) > self._keep:
                del self._data[: len(self._data) - self._keep]

    def text(self) -> str:
        text = self._data.decode("utf-8", errors="replace")
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        return text[-TAIL_BYTES:]


@dataclass(eq=False)
class _Running:
    """One ``run()``, registered before its spawn, so that no termination misses it."""

    process: asyncio.subprocess.Process | None = None
    spawned: asyncio.Event = field(default_factory=asyncio.Event)


class AttemptProcessSupervisor:
    """Implement ``AttemptProcesses`` for one attempt, supervising each process until it exits."""

    def __init__(
        self, kill_grace_seconds: float, *, environ: Mapping[str, str] | None = None
    ) -> None:
        self._kill_grace = kill_grace_seconds
        self._environ = os.environ if environ is None else environ
        self._live: set[_Running] = set()
        self._closed = False
        self._paused = 0

    async def run(self, command: ProcessCommand) -> ProcessResult:
        _check(command)
        if self._closed or self._paused:
            raise ProcessesClosed(
                "no agent process may start now: its attempt has ended or its hook is stopping"
            )
        secrets = [
            value.get_secret_value()
            for value in command.env.values()
            if isinstance(value, SecretStr) and value.get_secret_value()
        ]
        running = _Running()
        self._live.add(running)
        try:
            try:
                running.process = process = await asyncio.create_subprocess_exec(
                    *command.argv,
                    cwd=Path(command.cwd),
                    env=_environment(self._environ, command.env),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            except (OSError, ValueError) as error:
                raise ProcessLaunchFailure(
                    f"agent process {command.argv[0]!r} could not start: "
                    f"{type(error).__name__}: "
                    f"{error.strerror if isinstance(error, OSError) and error.strerror else error}"
                ) from None
            finally:
                running.spawned.set()
            return await self._supervise(running, process, secrets)
        finally:
            self._live.discard(running)

    async def _supervise(
        self, running: _Running, process: asyncio.subprocess.Process, secrets: list[str]
    ) -> ProcessResult:
        started = time.monotonic()
        stdout, stderr = _Tail(secrets), _Tail(secrets)
        drains = asyncio.create_task(_drain(process, stdout, stderr))
        try:
            await _exited(process)
            # End whatever the leader left behind in its group too, or its pipes stay open.
            await self._terminate(running)
            await asyncio.wait({drains}, timeout=DRAIN_SECONDS)
        except asyncio.CancelledError:
            # End the group before the cancellation goes on. Ending it has a time limit.
            ending = asyncio.ensure_future(self._terminate(running))
            await outlast(ending)
            ending.exception()  # Observe it, because the cancellation is the outcome.
            raise
        finally:
            drained = await stop(drains, grace=None)
            # Close our ends of the pipes, which a descendant outside the group may hold.
            process._transport.close()  # pyright: ignore[reportAttributeAccessIssue]
            await asyncio.sleep(0)  # The pipe transports close on the next iteration.
        drained.propagate()
        if not drains.cancelled():
            drains.result()  # Raise the failure of a tail that failed.
        code = process.returncode
        assert code is not None
        return ProcessResult(
            exit_code=code if code >= 0 else -1,
            signal=-code if code < 0 else None,
            stdout_tail=stdout.text(),
            stderr_tail=stderr.text(),
            elapsed_seconds=time.monotonic() - started,
        )

    async def _terminate(self, running: _Running) -> None:
        """Stop the whole group with SIGTERM, then SIGKILL after the grace, and wait it out.

        A process still being spawned is waited for first, so none escapes.
        """
        await running.spawned.wait()
        process = running.process
        if process is None:  # It never started.
            return
        group = process.pid
        if process.returncode is not None and not _group_alive(group):
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._kill_grace
        _signal(group, signal.SIGTERM)
        # asyncio reaps the leader, so once it has exited the group check sees only
        # what it left behind.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(_exited(process), self._kill_grace)
        while _group_alive(group) and loop.time() < deadline:
            await asyncio.sleep(0.05)
        if process.returncode is None or _group_alive(group):
            _signal(group, signal.SIGKILL)
            settle = loop.time() + GROUP_EXIT_SECONDS
            while (process.returncode is None or _group_alive(group)) and loop.time() < settle:
                await asyncio.sleep(0.02)
            if process.returncode is None or _group_alive(group):
                raise ProcessKillFailure(
                    f"agent process {process.pid}'s group outlived SIGKILL by "
                    f"{GROUP_EXIT_SECONDS:g}s"
                )

    async def terminate(self) -> None:
        """Terminate every live process, including those still spawning, and wait until each exits.

        At sealing every live process is a sampling one, because grading has not begun.
        """
        results = await asyncio.gather(
            *(self._terminate(running) for running in list(self._live)), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    def pause(self) -> None:
        """Refuse new processes until ``resume()``, as while a hook is being stopped."""
        self._paused += 1

    def resume(self) -> None:
        self._paused -= 1

    async def close(self) -> None:
        """End the attempt by refusing every later ``run()`` and ending what is still running."""
        self._closed = True
        await self.terminate()


async def _drain(process: asyncio.subprocess.Process, stdout: _Tail, stderr: _Tail) -> None:
    assert process.stdout is not None and process.stderr is not None
    await asyncio.gather(stdout.drain(process.stdout), stderr.drain(process.stderr))


async def _exited(process: asyncio.subprocess.Process) -> None:
    """Wait until the leader has exited.

    ``Process.wait()`` would also wait for its output pipes to close, which a
    descendant can hold open indefinitely. The exit status arrives first.
    """
    while process.returncode is None:
        await asyncio.sleep(0.02)


def _group_alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal(group: int, number: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(group, number)
