"""The cookbook's Blaxel environment for Harbor jobs.

Harbor starts the environment, and the cookbook's resource registry owns its sandbox.
Select it in a Harbor job with
``environment: {import_path: loops_cookbook.harbor.blaxel:BlaxelEnvironment}``, and
configure ``[providers.blaxel]``. Harbor still decides when environments start and
stop, and it still runs commands, verification and artifacts. Every sandbox the
environment starts is provisioned through the attempt's ``EpisodeResources``. Every
stop is a cookbook release, so Harbor never creates or deletes a sandbox itself.

The Harbor adapter writes the attempt's ID into ``environment.kwargs`` as
``loops_attempt_id``. The constructor only looks that attempt up. So an environment
that Harbor constructs and never starts provisions nothing, such as the agent
environment inside a ``RegradeTrial``. ``start()`` provisions a sandbox keyed by
Harbor's ``session_id``, under Harbor's own name. It attaches to the sandbox through
Blaxel's usage code, with the attempt's usage environment and the run's attach
deadline. Then it repeats Harbor's post-create startup unchanged. The environment
never holds the provider's adapter, so it cannot create or delete behind the registry.

Blaxel's own rules live under ``providers/blaxel``, and this module holds only the
Harbor hooks. It reaches Harbor's private members, so it is pinned to Harbor 0.23.
Check it again on any Harbor upgrade. Only Harbor's direct mode is supported. Compose
mode needs its own evidence first.
"""

from __future__ import annotations

import functools
import hashlib
import json
import shlex
import stat
from collections.abc import Mapping, Sequence
from typing import Any

from harbor.environments.base import ExecResult
from harbor.environments.blaxel import BlaxelEnvironment as NativeBlaxelEnvironment
from harbor.environments.definition import (
    _CONTENT_HASH_IGNORE_NAMES,  # pyright: ignore[reportPrivateUsage]
    SNAPSHOT_HASH_LEN,
)
from harbor.models.trial.paths import EnvironmentPaths
from pydantic import SecretStr

from loops_cookbook.evidence import JsonValue
from loops_cookbook.harbor.attempts import attempt_for, release_sandbox
from loops_cookbook.harbor.oom import oom_kills
from loops_cookbook.providers.blaxel.api import image_build_specification
from loops_cookbook.providers.blaxel.images import blaxel_image
from loops_cookbook.providers.blaxel.incidents import (
    REPLAY_BUDGET_SECONDS,
    ReplayState,
    close_clients,
    never_dispatched,
    replay,
    workload_failed,
)
from loops_cookbook.providers.blaxel.usage import attach
from loops_cookbook.resources.api import ResourceHandle, ResourceRequest


class SandboxLost(RuntimeError):
    """Raised when the sandbox went away under a command.

    The sandbox may have been deleted, reaped, or kept past its time to live (TTL).
    """


def content_image_name(
    environment_id: str, build: Mapping[str, JsonValue], modes: Sequence[tuple[str, int]]
) -> str:
    """Return an image name derived from what is built.

    Tasks with identical environments then share one image and one build. Harbor's own
    image name includes the task, so it can't be shared.

    ``modes`` holds each entry's path and permission bits. Harbor's content hash reads
    only paths and bytes, but COPY keeps modes. The name leaves out builder memory and
    source paths, because neither changes the image.
    """
    files = build["local_files"]
    assert isinstance(files, list)
    context = sorted(
        f"{item['context_name']}:{item['destination_path']}"
        for item in files
        if isinstance(item, dict)
    )
    rest = {key: value for key, value in build.items() if key not in ("memory", "local_files")}
    identity = [environment_id, context, rest, [list(entry) for entry in sorted(modes)]]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    return f"harbor-img-{digest.hexdigest()[:SNAPSHOT_HASH_LEN]}"


class BlaxelEnvironment(NativeBlaxelEnvironment):
    """Harbor's Blaxel environment, with the sandbox provisioned through the attempt."""

    def __init__(self, *args: Any, loops_attempt_id: str, **kwargs: Any) -> None:
        attempt = attempt_for(loops_attempt_id)
        deadlines = attempt.provider_deadlines.get("blaxel")
        if deadlines is None:
            raise RuntimeError("the cookbook's Blaxel environment needs [providers.blaxel].")
        super().__init__(*args, **kwargs)
        if self._compose_mode:
            raise ValueError(
                "the cookbook does not support Harbor's Blaxel compose mode. Use a "
                "Dockerfile or docker_image task."
            )
        if self._imported_image() is None and self._content_hash_covers_build():
            self._image_name = content_image_name(
                self.environment_id, self._build_specification(), self._context_modes()
            )
        self._resources = attempt.resources
        self._attach_seconds = deadlines.attach_seconds
        self._handle: ResourceHandle | None = None
        self._replay = ReplayState()
        self._lost: str | None = None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Run one command, replaying it only when Blaxel provably never received it.

        Once the sandbox is known gone, every later command fails at once.
        """
        if self._lost is not None:
            raise SandboxLost(self._lost)
        call = functools.partial(
            NativeBlaxelEnvironment.exec,
            self,
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )
        try:
            result = await replay(call, state=self._replay, logger=self.logger)
        except Exception as error:
            # Replay gave up, because the sandbox took no command for its whole budget
            # or Blaxel says it is dead. Later commands would each wait the budget again.
            if workload_failed(error) or never_dispatched(error):
                self._lost = (
                    f"the sandbox took no command for {REPLAY_BUDGET_SECONDS:.0f}s "
                    f"({type(error).__name__}), so it is gone"
                )
                raise SandboxLost(self._lost) from error
            raise
        if self._lost is not None:
            raise SandboxLost(self._lost)
        return result

    async def oom_kills(self) -> int | None:
        """Return the sandbox kernel's OOM-kill count, or None. See ``harbor.oom``.

        A Blaxel sandbox is a micro-VM, so its kernel is its own.
        """
        return await oom_kills(self, "kernel")

    def _notice_lost_processes(self) -> None:
        """Record a command whose process vanished with its sandbox.

        Blaxel's ``process.wait`` stops polling when the process record can no longer
        be read. It then returns the last reading, which still says ``running``. A real
        timeout raises instead. Harbor then reports exit code 0 with no output. Without
        this record, a deleted sandbox would look like a command that succeeded.
        """
        assert self._sandbox is not None
        process = self._sandbox.process
        wait = process.wait

        async def waited(*args: Any, **kwargs: Any) -> Any:
            result = await wait(*args, **kwargs)
            if str(getattr(result, "status", "")).lower() == "running":
                self._lost = "a command's process vanished while it ran, so the sandbox is gone"
            return result

        process.wait = waited

    def usage_environment(self) -> Mapping[str, str | SecretStr]:
        """Return what a separate process needs to reach this environment's sandbox."""
        if self._handle is None:
            raise RuntimeError("the environment has no sandbox. Start it first.")
        return self._resources.usage_environment(self._handle.intent.resource_id)

    def _imported_image(self) -> str | None:
        """Return the Blaxel image that a ``docker_image`` of ``blaxel://<image>`` names.

        The sandbox starts from that image as it is, and nothing is built.
        """
        return blaxel_image(self.task_env_config.docker_image)

    def _content_hash_covers_build(self) -> bool:
        # The build uploads the whole directory. Harbor's content hash reads only regular
        # files, by path and bytes, and skips the names it ignores. So the hash does not
        # say what is built when the directory holds a symlink, an ignored name, or
        # anything else the hash cannot see. Examples are an empty directory, which COPY
        # still creates, and a FIFO.
        return not any(
            path.is_symlink()
            or path.name in _CONTENT_HASH_IGNORE_NAMES
            or not (path.is_file() or path.is_dir())
            or (path.is_dir() and next(path.iterdir(), None) is None)
            for path in self.environment_dir.rglob("*")
        )

    def _context_modes(self) -> list[tuple[str, int]]:
        """Return each entry of the build context, by path, with its permission bits."""
        return [
            (path.relative_to(self.environment_dir).as_posix(), stat.S_IMODE(path.lstat().st_mode))
            for path in self.environment_dir.rglob("*")
        ]

    def _build_specification(self) -> dict[str, JsonValue]:
        # This is a pure description of the task's image. The provider's build writes
        # the no-slim file that Harbor adds to the build context, not a trial.
        docker_image = self.task_env_config.docker_image
        built = (
            self._build_image_from_docker_image(docker_image)
            if docker_image
            else self._build_image_from_dockerfile()
        )
        return image_build_specification(
            built, memory=self._memory_mb, sandbox_version=self._sandbox_version, slim=False
        )

    def _request(self, force_build: bool) -> ResourceRequest:
        imported = self._imported_image()
        if imported is not None:
            # The image was built elsewhere and imported. The sandbox starts from it as it
            # is, with no build to wait on or force.
            return ResourceRequest(
                provider="blaxel",
                kind="sandbox",
                logical_key=self.session_id,
                specification={"sandbox": self._create_sandbox_config(imported)},
            )
        image = self._require_image_name()
        build = self._build_specification()
        return ResourceRequest(
            provider="blaxel",
            kind="sandbox",
            logical_key=self.session_id,
            specification={
                "sandbox": self._create_sandbox_config(f"sandbox/{image}:latest"),
                "image": {"name": image, "force_build": force_build, "build": build},
            },
        )

    async def _start_direct(self, force_build: bool) -> None:
        if self._sandbox is not None:
            return
        self._handle = await self._resources.provision(self._request(force_build))
        self._sandbox = await attach(
            self._handle.resource,
            timeout_seconds=self._attach_seconds,
        )
        self._notice_lost_processes()
        # This is Harbor's post-create startup, unchanged. The attach contract test pins
        # the upstream digest and compares the two paths' recorded actions.
        await self._wait_until_ready()
        dirs = " ".join(
            shlex.quote(str(path))
            for path in (
                self._workdir,
                EnvironmentPaths.agent_dir,
                EnvironmentPaths.verifier_dir,
                EnvironmentPaths.artifacts_dir,
                EnvironmentPaths.tests_dir,
                EnvironmentPaths.solution_dir,
            )
        )
        result = await self.exec(f"mkdir -p {dirs} && chmod 777 /logs /logs/*")
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to prepare Blaxel sandbox directories "
                f"(exit {result.return_code}): {result.stderr}"
            )
        await self._upload_environment_dir_after_start()

    async def _stop_direct(self, delete: bool) -> None:
        """Release the sandbox when Harbor deletes it.

        A sandbox that Harbor keeps is released with the episode.

        A release that does not prove the sandbox gone raises ``SandboxNotReleased``.
        """
        sandbox, self._sandbox = self._sandbox, None
        try:
            if delete and self._handle is not None:
                handle, self._handle = self._handle, None
                await release_sandbox(self._resources, handle)
        finally:
            if sandbox is not None:
                await close_clients(sandbox)
