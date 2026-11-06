"""Blaxel sandboxes as Harbor's real Blaxel environment drives them, against a fake
Blaxel control plane. Import only once the Blaxel SDK is known to be installed.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import httpx
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from support.blaxel_api import FakeApi

from loops_cookbook.harbor.blaxel import BlaxelEnvironment
from loops_cookbook.providers.blaxel import usage

CUT = "[... truncated, see the process log file for the full output ...]\n"


class Sandbox:
    """A sandbox as Harbor drives it: every command and file operation is recorded."""

    def __init__(
        self,
        name,
        log,
        labels=None,
        fail_first=None,
        output=None,
        replies=None,
        vanishes=None,
    ):
        self.name, self.log, self.status = name, log, "DEPLOYED"
        self.vanishes = vanishes  # a command whose sandbox is deleted while it runs
        self.replies = dict(replies or {})  # command substring -> (exit code, stdout, stderr)
        self.output, self.command = output, None  # ``output``: a command's whole stdout
        self.metadata = NS(to_dict=lambda: {"name": name, "labels": dict(labels or {})})
        # The sandbox's own HTTP API, which only a process's logs are read from.
        client = httpx.AsyncClient(
            base_url="http://sandbox.test", transport=httpx.MockTransport(self._logs)
        )
        self.fs = NS(
            _client=client, get_client=lambda: client, ls=self._ls, write_binary=self._write
        )
        self.process = NS(
            _client=client,
            get_client=lambda: client,
            get_api_client=NS,
            exec=self._exec,
            wait=self._wait,
            kill=self._kill,
        )
        self._fail_first = list(fail_first or [])

    async def _ls(self, path):
        self.log.append(f"fs.ls {path}")

    async def _write(self, path, data):
        self.log.append("fs.write_binary " + ("<tar>" if path.endswith(".tar.gz") else path))

    async def _exec(self, request):
        if self._fail_first:
            raise self._fail_first.pop(0)
        self.log.append(f"process.exec cmd={request['command']}")
        self.command = request["command"]
        return NS(pid="p", name=None)

    async def _wait(self, pid, max_wait, interval):
        if self.vanishes is not None and self.vanishes in self.command:
            # Blaxel's wait, once the process record is gone: the last reading, running.
            return NS(exit_code=0, stdout="", stderr="", status="running")
        if self.output is not None and "print-output" in self.command:
            # As Blaxel's process record: the last 64 KB behind a truncation marker.
            cut = CUT + self.output[-65_536:]
            return NS(exit_code=0, stdout=cut, stderr="", status="completed")
        for fragment, (code, stdout, stderr) in self.replies.items():
            if fragment in self.command:
                return NS(exit_code=code, stdout=stdout, stderr=stderr, status="completed")
        return NS(exit_code=0, stdout="/app\n", stderr="", status="completed")

    def _logs(self, request: httpx.Request) -> httpx.Response:
        pid = request.url.path.removeprefix("/process/").removesuffix("/logs")
        self.log.append(f"process.logs {pid}")
        output = self.output or ""
        return httpx.Response(200, json={"logs": output, "stdout": output, "stderr": ""})

    async def _kill(self, pid):
        self.log.append(f"process.kill {pid}")


def blaxel_usage(monkeypatch, *apis):
    """Blaxel's usage client, as the SDK would give it for sandboxes in ``apis``."""

    async def get(name):
        (owner,) = [api for api in apis if name in api.sandboxes]
        return await owner.attach(name)

    monkeypatch.setattr(usage, "SandboxInstance", NS(get=get))


def attaching(api, log, cls=Sandbox, **options):
    """Attach ``api``'s sandboxes as ``cls`` with ``options``; the sandboxes it attached."""
    attached = []

    async def attach(name):
        api.calls.append(("attach", name))
        attached.append(cls(name, log, api.sandboxes[name].labels, **options))
        return attached[-1]

    api.attach = attach
    return attached


def blaxel_api(monkeypatch, log):
    """A fake Blaxel control plane whose sandboxes record what Harbor does in ``log``."""
    api = FakeApi()
    attaching(api, log)
    blaxel_usage(monkeypatch, api)
    return api


def environments(tmp_path):
    """``build(mode, session, attempt, cls, name, directory, files, docker_image)``: a Harbor
    Blaxel environment for ``attempt`` in ``tmp_path/directory``, holding ``files`` (relative
    path to text); ``mode`` picks its image source, ``docker_image`` the image of that mode."""

    def build(
        mode="dockerfile",
        session="trial__env",
        attempt="attempt-1",
        cls=BlaxelEnvironment,
        name="task",
        directory=None,
        files=None,
        docker_image="python:3.12",
    ):
        source = tmp_path / (directory or f"env-{mode}")
        source.mkdir(exist_ok=True)
        for relative, content in (files or {}).items():
            (source / relative).parent.mkdir(parents=True, exist_ok=True)
            (source / relative).write_text(content)
        if mode == "dockerfile":
            (source / "Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /app\n")
            task = EnvironmentConfig()
        else:
            (source / "data.txt").write_text("hello")
            task = EnvironmentConfig(docker_image=docker_image, workdir="/app")
        paths = TrialPaths(tmp_path / f"trial-{session}")
        paths.mkdir()
        options = {} if cls is not BlaxelEnvironment else {"loops_attempt_id": attempt}
        return cls(
            environment_dir=source,
            environment_name=name,
            session_id=session,
            trial_paths=paths,
            task_env_config=task,
            **options,
        )

    return build
