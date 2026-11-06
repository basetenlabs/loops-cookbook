"""Blaxel for the pattern and example tests: the real adapter and SDK over fake APIs.

An episode's sandbox is attached, closed and asked whether it is alive by the
cookbook's own Blaxel adapter, so a status reaches the episode only through the one
DEPLOYED rule. Behind it, ``FakeApi`` is Blaxel's control plane, and ``get`` stands
in for the SDK's ``SandboxInstance.get``: each read is a new real ``SandboxInstance``
whose HTTP clients answer from ``SandboxApi``, a fake of the sandbox's own API
behind an ``httpx.MockTransport``. So the SDK sends its real requests, parses
real-shaped answers and raises its real ``ResponseError``s. A sandbox's files live on
disk under ``root/<name>``, and its commands run there with ``/bin/sh``, whichever
client started them.

The sandbox API answers as Blaxel's was observed to, live, with SDK 0.4.9 on
2026-09-28:

- ``POST /process`` returns the command at once, ``running``. With
  ``waitForCompletion`` it answers when the command ends, or ``422 process timed out
  after N seconds`` at its ``timeout``.
- A ``keepAlive`` command is killed at its ``timeout`` (600 s by default): its record
  then reads ``killed``, exit code -1, with its output so far. Without ``keepAlive``
  the timeout only bounds a wait, and the command keeps running.
- A kill ends the command's process group; a process in a session of its own
  (``setsid``) survives it.
- A record keeps each stream whole up to 64 KB; past that, a truncation marker and
  the last 65,536 bytes. ``GET /process/{id}/logs`` has each stream whole.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import Callable, Iterator, MutableMapping
from pathlib import Path
from typing import Any

import httpx
from blaxel.core.client.models import Sandbox as SandboxModel
from blaxel.core.sandbox import SandboxInstance

from loops_cookbook.providers.blaxel.adapter import BlaxelAdapter
from loops_cookbook.providers.blaxel.settings import BlaxelSettings
from support.blaxel_api import CREDENTIALS, FakeApi
from support.fakes import no_wait

SETTINGS = BlaxelSettings(
    region="us-pdx-1",
    create_timeout_seconds=2,
    reconcile_timeout_seconds=1,
    image_build_timeout_seconds=3,
    release_timeout_seconds=1,
    confirm_absent_timeout_seconds=1,
    list_timeout_seconds=1,
    attach_timeout_seconds=1,
)
URL = "http://sandbox.test"
# ProcessRequest: "When keepAlive is true, [timeout] defaults to 600s"; 0 never kills.
KEEP_ALIVE_TIMEOUT_SECONDS = 600
RECORD_BYTES = 65_536
TRUNCATION = "[... truncated, see the process log file for the full output ...]\n"


def answer(status: int, body: Any) -> httpx.Response:
    return httpx.Response(status, json=body)


class Command:
    """One command, as the sandbox API runs it: on its own, whoever waits for it."""

    def __init__(self, pid: str, request: dict[str, Any], process: asyncio.subprocess.Process):
        self.pid, self.request, self.killed = pid, request, False
        self._process = process
        self._output = {"stdout": bytearray(), "stderr": bytearray()}
        self._drains = [
            asyncio.create_task(self._drain(process.stdout, self._output["stdout"])),
            asyncio.create_task(self._drain(process.stderr, self._output["stderr"])),
        ]
        self.done = asyncio.create_task(self._run())

    @staticmethod
    async def _drain(stream, into: bytearray) -> None:
        while chunk := await stream.read(65_536):
            into.extend(chunk)

    async def _run(self) -> None:
        finished = asyncio.gather(self._process.wait(), *self._drains)
        timeout = self.request.get("timeout", KEEP_ALIVE_TIMEOUT_SECONDS)
        if not self.request.get("keepAlive") or not timeout:  # it never kills
            await finished
            return
        try:
            await asyncio.wait_for(asyncio.shield(finished), timeout)
        except TimeoutError:
            await self.kill()
            await finished  # its output so far, drained before the record says killed

    async def kill(self) -> None:
        """Kill the command's process group: its shell and what the shell is running."""
        if self._process.returncode is None:
            self.killed = True
            self._signal_group()
        await self._process.wait()

    async def reap(self) -> None:
        """Kill everything the command started, as releasing the sandbox would."""
        self._signal_group()
        await asyncio.gather(self.done, *self._drains)

    def _signal_group(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self._process.pid, signal.SIGKILL)

    def stream(self, name: str) -> str:
        return self._output[name].decode(errors="replace")

    def _kept(self, name: str) -> str:
        output = self._output[name]
        if len(output) <= RECORD_BYTES:
            return output.decode(errors="replace")
        return TRUNCATION + output[-RECORD_BYTES:].decode(errors="replace")

    @property
    def _exit_code(self) -> int | None:
        """The exit code once the command has ended and both its streams are drained.

        The process's own return code is set as it exits, while the drains may still
        be reading its pipes; a record that said ``completed`` then would hand back
        output Blaxel's record never lacks.
        """
        return self._process.returncode if self.done.done() else None

    @property
    def status(self) -> str:
        code = self._exit_code
        if code is None:
            return "running"
        if self.killed:
            return "killed"
        return "completed" if code == 0 else "failed"

    def record(self) -> dict[str, Any]:
        """The ``ProcessResponse`` the API answers with."""
        code = self._exit_code
        return {
            "command": self.request["command"],
            "completedAt": "",
            "exitCode": -1 if self.killed else (code or 0),
            "logs": "",
            "name": self.request.get("name") or self.pid,
            "pid": self.pid,
            "startedAt": "",
            "status": self.status,
            "stdout": self._kept("stdout"),
            "stderr": self._kept("stderr"),
            "workingDir": self.request.get("workingDir", "/"),
            "keepAlive": bool(self.request.get("keepAlive")),
        }


class SandboxFiles(MutableMapping[str, str]):
    """A sandbox's files, by their path in the sandbox, as they are on disk."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _local(self, path: str) -> Path:
        return self._root / path.lstrip("/")

    def __getitem__(self, path: str) -> str:
        try:
            return self._local(path).read_text()
        except (FileNotFoundError, IsADirectoryError):
            raise KeyError(path) from None

    def __setitem__(self, path: str, text: str) -> None:
        self._local(path).parent.mkdir(parents=True, exist_ok=True)
        self._local(path).write_text(text)

    def __delitem__(self, path: str) -> None:
        try:
            self._local(path).unlink()
        except FileNotFoundError:
            raise KeyError(path) from None

    def __iter__(self) -> Iterator[str]:
        return (f"/{p.relative_to(self._root)}" for p in self._root.rglob("*") if p.is_file())

    def __len__(self) -> int:
        return sum(1 for _ in self)


class Sandbox:
    """One sandbox's side of its API: its disk, and every command it ran. Once
    ``unreachable``, every request to it fails to connect, as when it has vanished."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.files = SandboxFiles(root)
        self.commands: list[Command] = []
        self.unreachable = False

    def local(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    def command(self, identifier: str) -> Command | None:
        found = [c for c in self.commands if identifier in (c.pid, c.request.get("name"))]
        return found[0] if found else None


class SandboxApi:
    """The sandbox API one client reaches: its sandbox's, with this client's faults.

    ``failing`` answers every file read with that status; ``stuck`` holds every file
    write forever. The fake's ``before_exec(name, request)`` may raise, as the API
    would, or rewrite the request.
    """

    def __init__(self, name: str, sandbox: Sandbox, blaxel: FakeBlaxel) -> None:
        self.name, self.sandbox, self.blaxel = name, sandbox, blaxel
        self.failing: int | None = None
        self.stuck = blaxel.stuck

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.sandbox.unreachable:
            raise httpx.ConnectError("the sandbox is unreachable")
        method, path = request.method, request.url.path
        body = json_body(request)
        if path.startswith("/filesystem/tree/"):
            return self._write_tree(path.removeprefix("/filesystem/tree/"), body)
        if path.startswith("/filesystem/"):
            file = path.removeprefix("/filesystem/")
            return await (self._write(file, body) if method == "PUT" else self._read(file))
        if (method, path) == ("POST", "/process"):
            return await self._exec(body)
        if (method, path) == ("GET", "/process"):
            return answer(200, [c.record() for c in self.sandbox.commands])
        identifier, _, action = path.removeprefix("/process/").partition("/")
        command = self.sandbox.command(identifier)
        if command is None:
            return answer(404, {"error": f"process {identifier} not found"})
        if (method, action) == ("DELETE", "kill"):
            await command.kill()
            return answer(200, {"message": "process killed"})
        if (method, action) == ("GET", "logs"):
            out, err = command.stream("stdout"), command.stream("stderr")
            return answer(200, {"logs": out + err, "stdout": out, "stderr": err})
        return answer(200, command.record())

    def _write_tree(self, destination: str, body: dict[str, Any]) -> httpx.Response:
        for relative, text in body["files"].items():
            self.sandbox.files[f"{destination.rstrip('/')}/{relative}"] = text
        return answer(
            200, {"files": [], "name": destination, "path": destination, "subdirectories": []}
        )

    async def _write(self, path: str, body: dict[str, Any]) -> httpx.Response:
        if self.stuck:
            await asyncio.Event().wait()
        self.sandbox.files[path] = body["content"]
        return answer(200, {"message": "file written", "path": path})

    async def _read(self, path: str) -> httpx.Response:
        if self.failing is not None:
            return answer(self.failing, {"error": "read failed"})
        if path not in self.sandbox.files:
            return answer(404, {"error": f"{path} not found"})
        return answer(200, {"content": self.sandbox.files[path], "path": path})

    async def _exec(self, request: dict[str, Any]) -> httpx.Response:
        self.blaxel.commands.setdefault(self.name, []).append(request["command"])
        self.blaxel.before_exec(self.name, request)
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            request["command"],
            cwd=self.sandbox.local(request.get("workingDir", "/")),
            env={"PATH": os.environ["PATH"], **request.get("env", {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        command = Command(str(len(self.sandbox.commands) + 1), request, process)
        self.sandbox.commands.append(command)
        if not request.get("waitForCompletion"):
            return answer(200, command.record())
        timeout = request.get("timeout") or None
        try:  # a client that stops waiting leaves the command running
            await asyncio.wait_for(asyncio.shield(command.done), timeout)
        except TimeoutError:
            if request.get("keepAlive"):  # then the timeout kills it
                await command.done
            return answer(422, {"error": f"process timed out after {timeout} seconds"})
        return answer(200, command.record())


def json_body(request: httpx.Request) -> dict[str, Any]:
    import json

    return json.loads(request.content) if request.content else {}


class SandboxClient(SandboxInstance):
    """The SDK's client for one sandbox, whose HTTP clients reach ``api``; its API's
    faults and its sandbox's files are beside it, for a test to set and read."""

    def __init__(self, observed, api: SandboxApi) -> None:
        model = {
            "metadata": {"name": observed.name, "labels": dict(observed.labels)},
            "status": observed.status,
            "spec": {},
        }
        super().__init__(SandboxModel.from_dict(model), force_url=URL)
        transport = httpx.MockTransport(api)
        self.pools = [httpx.AsyncClient(base_url=URL, transport=transport) for _ in "ab"]
        self.fs._client, self.process._client = self.pools
        self.api = api

    @property
    def files(self) -> SandboxFiles:
        return self.api.sandbox.files

    @property
    def failing(self) -> int | None:
        return self.api.failing

    @failing.setter
    def failing(self, status: int | None) -> None:
        self.api.failing = status

    @property
    def stuck(self) -> bool:
        return self.api.stuck

    @stuck.setter
    def stuck(self, stuck: bool) -> None:
        self.api.stuck = stuck

    @property
    def closed(self) -> bool:
        return all(pool.is_closed for pool in self.pools)


class FakeBlaxel:
    """The fake control plane, each sandbox's API, and every SDK client read from them."""

    def __init__(self, root: Path) -> None:
        self.api = FakeApi()
        self.root = root
        self.sandboxes: dict[str, Sandbox] = {}
        self.clients: list[SandboxClient] = []
        self.commands: dict[str, list[str]] = {}  # every command each sandbox ran
        self.before_exec: Callable[[str, dict[str, Any]], None] = lambda name, request: None
        self.stuck = False  # every client read from now holds its writes forever

    @property
    def files(self) -> dict[str, SandboxFiles]:
        """Each attached sandbox's files, by its name."""
        return {name: sandbox.files for name, sandbox in self.sandboxes.items()}

    def adapter(self, settings=SETTINGS):
        return BlaxelAdapter(settings, CREDENTIALS, self.api, sleep=no_wait)

    async def get(self, name):
        observed = await self.api.get(name)
        sandbox = self.sandboxes.get(name) or self.sandboxes.setdefault(
            name, Sandbox(self.root / name)
        )
        client = SandboxClient(observed, SandboxApi(name, sandbox, self))
        self.clients.append(client)
        return client

    def attached(self):
        """The clients the adapter kept: those it did not close as soon as it read them."""
        return [client for client in self.clients if not client.closed]

    def lose(self, name, how):
        """Sandbox ``name`` went: deleted, or still readable in status ``how``."""
        if how == "deleted":
            del self.api.sandboxes[name]
        else:
            self.api.sandboxes[name].statuses = [how]

    async def reap(self):
        """End every command still running, and everything it started."""
        everything = [c for sandbox in self.sandboxes.values() for c in sandbox.commands]
        await asyncio.gather(*(command.reap() for command in everything))
