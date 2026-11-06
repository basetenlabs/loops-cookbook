"""A Harbor environment backed by a local directory, for offline Harbor tests.

Container paths under a fixed set of roots map onto a per-session directory. The
same mapping rewrites shell commands and uploaded text files, so a task's own
scripts run unchanged against the local tree. No container, network, or billable
resource is involved.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

ROOTS = ("app", "logs", "tests", "solution", "installed-agent")
_PATTERN = re.compile(r"(?<![\w./-])/(" + "|".join(ROOTS) + r")(?=[/\s'\";&|)]|$)")


class LocalEnvironment(BaseEnvironment):
    @staticmethod
    def type() -> str:
        return "baseline-local"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True, dynamic_network_policy=True)

    def _validate_definition(self) -> None:
        return None

    def _root(self) -> Path:
        root = getattr(self, "_local_root", None)
        if root is None:
            root = Path(tempfile.mkdtemp(prefix="baseline-env-"))
            self._local_root = root
        return root

    def _map(self, text: str) -> str:
        return _PATTERN.sub(lambda match: f"{self._root()}/{match.group(1)}", text)

    def _host(self, path: str) -> Path:
        return Path(self._map(path))

    async def start(self, force_build: bool) -> None:
        for name in ROOTS:
            (self._root() / name).mkdir(parents=True, exist_ok=True)
        for target in (*self._mount_targets(), "/logs/agent", "/logs/verifier", "/logs/artifacts"):
            self._host(target).mkdir(parents=True, exist_ok=True)
        # The image build, reduced to the one instruction these tasks rely on.
        dockerfile = Path(self.environment_dir) / "Dockerfile"
        for line in dockerfile.read_text().splitlines() if dockerfile.is_file() else ():
            words = line.split()
            if len(words) == 3 and words[0] == "COPY":
                await self.upload_dir(Path(self.environment_dir) / words[1], words[2])

    async def stop(self, delete: bool) -> None:
        if delete:
            shutil.rmtree(self._root(), ignore_errors=True)

    async def _apply_network_policy(self, network_policy) -> None:
        return None

    async def upload_file(self, source_path, target_path) -> None:
        target = self._host(str(target_path))
        target.parent.mkdir(parents=True, exist_ok=True)
        self._copy(Path(source_path), target)

    async def upload_dir(self, source_dir, target_dir) -> None:
        source, target = Path(source_dir), self._host(str(target_dir))
        for path in sorted(source.rglob("*")):
            destination = target / path.relative_to(source)
            if path.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                self._copy(path, destination)

    def _copy(self, source: Path, target: Path) -> None:
        data = source.read_bytes()
        try:
            target.write_text(self._map(data.decode()))
        except UnicodeDecodeError:
            target.write_bytes(data)
        shutil.copymode(source, target)

    async def download_file(self, source_path, target_path) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._host(str(source_path)), target)

    async def download_dir(self, source_dir, target_dir) -> None:
        source = self._host(str(source_dir))
        if source.is_dir():
            shutil.copytree(source, Path(target_dir), dirs_exist_ok=True)

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None) -> ExecResult:
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            self._map(command),
            cwd=str(self._host(cwd)) if cwd else str(self._root()),
            env={"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return ExecResult(
            stdout=stdout.decode(), stderr=stderr.decode(), return_code=process.returncode or 0
        )
