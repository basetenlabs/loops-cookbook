"""Daytona sandboxes as Harbor's real Daytona environment drives them, against a fake
Daytona. Import only once the Daytona SDK is known to be installed.

The provider's control plane (``FakeApi``) hands out the only client Harbor's
environment reaches its sandbox through, and every command and upload Harbor makes
is recorded.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace as NS

from daytona_api_client_async.models.sandbox_state import SandboxState
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from support.daytona_api import FakeApi

from loops_cookbook.harbor.daytona import DaytonaEnvironment

TASK = "FROM python:3.12-slim\nWORKDIR /app\nCOPY initial_workspace/ /app/\n"


def run(shell, command):
    """A command's exit code and stdout: one naming a ``failing`` part fails; one naming
    an ``answers`` part prints its answer; one that ``cat``s files prints the first of
    them in ``files``, as ``cat a || cat b`` does."""
    if any(part in command for part in shell.failing):
        return 1, ""
    if answer := next((out for part, out in shell.answers.items() if part in command), None):
        return 0, answer
    read = [path.strip("'\"") for path in re.findall(r"\bcat (\S+)", command)]
    if not read:
        return 0, ""
    return next(((0, shell.files[path]) for path in read if path in shell.files), (1, ""))


class HarborSandbox:
    """The SDK's sandbox as Harbor uses it: its process sessions and uploads, recorded."""

    def __init__(self, sandbox_id, labels, state, log, shell, name=None):
        self.id, self.name, self.labels, self.log = sandbox_id, name, dict(labels), log
        self.state = SandboxState(state)
        self._commands = {}

        async def create_session(session_id):
            pass

        async def execute_session_command(session_id, request, timeout=None):
            self.log.append(request.command)
            self._commands[session_id] = run(shell, request.command)
            return NS(cmd_id="c")

        async def get_session_command(session_id, command_id):
            return NS(id=command_id, exit_code=self._commands[session_id][0])

        async def get_session_command_logs(session_id, command_id):
            return NS(stdout=self._commands[session_id][1], stderr="")

        async def upload_file(source, target):
            self.log.append(f"upload {target}")
            shell.uploads[target] = Path(source).read_bytes()

        self.process = NS(
            create_session=create_session,
            execute_session_command=execute_session_command,
            get_session_command=get_session_command,
            get_session_command_logs=get_session_command_logs,
        )
        self.fs = NS(upload_file=upload_file)


def daytona_api(log, shell):
    """The provider's control plane; its client hands Harbor the sandboxes it runs in."""

    class Api(FakeApi):
        def client(self):
            fake = self

            class Client:
                async def get(self, sandbox_id):
                    fake.calls.append(("attach", sandbox_id))
                    box = fake.sandboxes[sandbox_id]
                    return HarborSandbox(sandbox_id, box.labels, box.states[-1], log, shell)

            return Client()

    return Api()


def environments(tmp_path):
    """``build(dockerfile, session, attempt, config, **kw)``: a Harbor Daytona
    environment for ``attempt``, over a task's Dockerfile."""

    def build(
        dockerfile=TASK,
        session="trial__env",
        attempt="attempt-1",
        config=None,
        **kw,
    ):
        source = tmp_path / f"task-{session}" / "environment"
        (source / "initial_workspace").mkdir(parents=True, exist_ok=True)
        (source / "initial_workspace" / "a.csv").write_text("1,2\n")
        (source / "Dockerfile").write_text(dockerfile)
        paths = TrialPaths(tmp_path / f"trial-{session}")
        paths.mkdir()
        return DaytonaEnvironment(
            environment_dir=source,
            environment_name="task",
            session_id=session,
            trial_paths=paths,
            task_env_config=config or EnvironmentConfig(cpus=2, memory_mb=4096, storage_mb=3072),
            loops_attempt_id=attempt,
            **kw,
        )

    return build
