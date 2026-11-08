"""The cookbook's Daytona environment for Harbor jobs.

Harbor starts the environment, and the cookbook's resource registry owns its sandbox.
Select it in a Harbor job with
``environment: {import_path: loops_cookbook.harbor.daytona:DaytonaEnvironment}``, and
configure ``[providers.daytona]``.

Harbor's own Daytona environment does nearly everything. It chooses the image, which
is a declarative build of the task's Dockerfile or a prebuilt image. It sizes the
sandbox, applies the network policy, and runs commands and file transfers. Only its
start changes. Harbor's start creates the sandbox through a process-wide client that
it builds from ``DAYTONA_*`` in the environment. It retries with tenacity, up to ten
times for a rate limit, minutes apart, and it creates a second time after an ambiguous
failure. Here the sandbox is one provision through the attempt's ``EpisodeResources``.
The provision journals the intent, holds the create window, and reconciles by name
instead of creating twice. Harbor then reaches the sandbox through the provider's own
client, configured from ``[providers.daytona]``, so its ``target`` holds for every
call. Harbor's client is never built. The environment refuses each kwarg that only
configures or uses that client, namely ``auto_snapshot``, ``snapshot_template_name``
and ``connection_pool_maxsize``. Harbor's delete, ``_stop_sandbox``, becomes a
cookbook release. So every sandbox is tracked, released with its attempt, and found
by ``loops cleanup``.

The environment also adds three things that Harbor's Daytona environment lacks for
training:

- Labels. The run's ownership labels, such as ``loops.run_id`` and
  ``loops.owner_id``, which names the attempt, are added to Harbor's ``harbor.*``
  labels and the job's ``labels``.
- Lifecycle. Harbor's defaults never stop a sandbox, because
  ``auto_stop_interval_mins`` is 0. A job that sets neither ``auto_stop_interval_mins``
  nor ``auto_delete_interval_mins`` gets ``[providers.daytona]``'s values. Those stop
  an idle sandbox and delete it as it stops.
- Threads. When ``thread_variables_from_vcpus`` is on, ``OMP_NUM_THREADS``,
  ``OPENBLAS_NUM_THREADS`` and ``MKL_NUM_THREADS`` are set to the sandbox's vCPUs for
  every command, unless the task sets them. This is needed because ``nproc`` counts
  the host's cores.

The constructor only looks the attempt up, so an environment that Harbor constructs
and never starts provisions nothing. Only Harbor's direct mode is supported. Compose
mode, which is Docker-in-Docker, needs its own evidence first. This module reaches
Harbor's private members, so it is pinned to Harbor 0.23. Check it again on any Harbor
upgrade.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from harbor.environments.daytona import DaytonaEnvironment as NativeDaytonaEnvironment
from harbor.environments.definition import should_use_prebuilt_docker_image
from pydantic import SecretStr

from loops_cookbook.evidence import JsonValue
from loops_cookbook.harbor.attempts import attempt_for, release_sandbox
from loops_cookbook.harbor.oom import oom_kills
from loops_cookbook.providers.daytona.api import BASE, dockerfile_image
from loops_cookbook.providers.daytona.settings import DaytonaSettings, lifecycle_minutes
from loops_cookbook.resources.api import ResourceHandle, ResourceRequest, SandboxResources

LIFECYCLE = {
    "auto_stop_interval_mins": "auto_stop_interval",
    "auto_delete_interval_mins": "auto_delete_interval",
}
_UNSET = object()
# Harbor's kwargs that act only through its own Daytona client, with their defaults
# and the reason each is refused.
HARBORS_CLIENT: dict[str, tuple[object, str]] = {
    "auto_snapshot": (False, "Harbor names and builds its snapshot with a client of its own"),
    "snapshot_template_name": (None, "Harbor resolves the snapshot with a client of its own"),
    "connection_pool_maxsize": (_UNSET, "it configures Harbor's own client"),
}
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def check_lifecycle(kwargs: Mapping[str, Any]) -> None:
    """Refuse a job's lifecycle interval that Daytona cannot take, before any sandbox is created."""
    for key in LIFECYCLE:
        if key in kwargs and not lifecycle_minutes(kwargs[key]):
            raise ValueError(
                f"environment.kwargs.{key} must be a whole number of minutes, at least 0"
            )


class DaytonaEnvironment(NativeDaytonaEnvironment):
    """Harbor's Daytona environment, with the sandbox provisioned through the attempt.

    Harbor reaches the sandbox through the provider's own client.
    """

    def __init__(self, *args: Any, loops_attempt_id: str, **kwargs: Any) -> None:
        attempt = attempt_for(loops_attempt_id)
        deadlines = attempt.provider_deadlines.get("daytona")
        settings = attempt.provider_settings.get("daytona")
        if deadlines is None or not isinstance(settings, DaytonaSettings):
            raise RuntimeError("the cookbook's Daytona environment needs [providers.daytona].")
        for key, (default, why) in HARBORS_CLIENT.items():
            if kwargs.get(key, default) != default:
                raise ValueError(
                    f"the cookbook's Daytona environment does not support "
                    f"environment.kwargs.{key}, because {why}. That client is configured "
                    "from the environment, not from [providers.daytona]."
                )
        # Harbor's Daytona defaults never stop a sandbox. When the job states no
        # lifecycle, the adapter fills in [providers.daytona]'s.
        self._lifecycle = {LIFECYCLE[key]: kwargs[key] for key in LIFECYCLE if key in kwargs}
        super().__init__(*args, **kwargs)
        if self._compose_mode:
            raise ValueError(
                "the cookbook does not support Harbor's Daytona compose mode, which is "
                "Docker-in-Docker. Use a Dockerfile or docker_image task."
            )
        if self._effective_gpus > 0:
            raise ValueError(
                "the cookbook's Daytona environment does not support GPU tasks. Run them "
                "on another provider."
            )
        self._resources = attempt.resources
        self._handle: ResourceHandle | None = None
        self._release_failure: BaseException | None = None
        # A sandbox's nproc is its host's core count, not its vCPUs, so thread pools
        # sized from it oversubscribe the cgroup's CPU. The table opts into this fix.
        # Harbor passes these variables with every command, over the image's ENV, and
        # the task's own values win. A task that names no CPUs gets the table's CPU
        # count, as its sandbox does.
        if settings.thread_variables_from_vcpus:
            cpus = self._effective_cpus or settings.cpu
            for name in THREAD_VARIABLES:
                self._persistent_env.setdefault(name, str(cpus))

    # ------------------------------------------------------- create and delete

    def _request(self, force_build: bool) -> ResourceRequest:
        """Return the sandbox that Harbor's direct start would create, as a registry request.

        Harbor chooses the image. It is the task's prebuilt ``docker_image`` when the
        task uses one, and a declarative build of its Dockerfile otherwise. Harbor
        passes the task's environment with every command, so the sandbox is created
        without it. That also keeps resolved secrets out of the journal.
        """
        docker_image = self.task_env_config.docker_image
        image = (
            {BASE: docker_image}
            if docker_image
            and should_use_prebuilt_docker_image(
                self.environment_dir, docker_image=docker_image, force_build=force_build
            )
            else dockerfile_image(self._dockerfile_path)
        )
        sandbox: dict[str, JsonValue] = {
            "labels": dict(self._sandbox_labels()),
            "image": image,
            **self._create_network_kwargs(),
            **self._lifecycle,
        }
        size = self.sandbox_size()
        sized: dict[str, JsonValue] = {
            key: value
            for key, value in (
                ("cpu", size.cpu),
                ("memory", size.memory_gib),
                ("disk", size.disk_gib),
            )
            if value is not None
        }
        if sized:
            sandbox["resources"] = sized
        if self._secrets:
            sandbox["secrets"] = dict(self._secrets)
        return ResourceRequest(
            provider="daytona",
            kind="sandbox",
            logical_key=self.session_id,
            specification={"sandbox": sandbox},
        )

    def sandbox_size(self) -> SandboxResources:
        """Return the size that Harbor's create would request.

        The size applies the job's overrides and enforcement policies. Memory and disk
        are in whole GiB, rounded down. A field is ``None`` where Harbor names nothing,
        and the provider fills it in.
        """
        resources = self._sandbox_resources()
        if resources is None:
            return SandboxResources()
        return SandboxResources(
            cpu=resources.cpu, memory_gib=resources.memory, disk_gib=resources.disk
        )

    async def start(self, force_build: bool) -> None:
        """Run Harbor's direct start, with its create replaced by one registry provision.

        Harbor's own start builds its process-wide client from ``DAYTONA_*`` in the
        environment and creates the sandbox with it. Here the sandbox comes from the
        provider's client instead. The rest is Harbor's post-create startup, unchanged.
        """
        self._validate_allowlist_targets()  # Harbor's own start does this first.
        self._handle = await self._resources.provision(self._request(force_build))
        sandbox: Any = await self._resources.sandbox_client(self._handle.intent.resource_id)
        self._sandbox = sandbox
        # This is Harbor's post-create startup, unchanged. A test pins the upstream digest.
        if self._expose_sandbox_id:
            await self._write_sandbox_id_file()
        workdir = self.task_env_config.workdir
        if workdir:
            await self._sandbox_exec(
                self._ensure_dirs_command([workdir], chmod=False), shell="sh -c"
            )
        await self.ensure_dirs(self._mount_targets(writable_only=True))
        await self._upload_environment_dir_after_start()

    async def _stop_sandbox(self) -> None:
        """Release the sandbox through the registry, so Harbor deletes nothing itself.

        Harbor logs a failed stop and carries on. So the environment keeps the failure
        and raises it from ``stop()``, and the cookbook learns that a sandbox may still
        be running.
        """
        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        try:
            await release_sandbox(self._resources, handle)
        except BaseException as error:
            self._release_failure = error
            raise

    async def stop(self, delete: bool) -> None:
        """Run Harbor's stop.

        A sandbox kept with ``delete=False`` is released with its episode.
        """
        self._release_failure = None
        await super().stop(delete)
        failure, self._release_failure = self._release_failure, None
        if failure is not None:
            raise failure

    # ------------------------------------------------------------- reporting

    def usage_environment(self) -> Mapping[str, str | SecretStr]:
        """Return what a separate process needs to reach this environment's sandbox."""
        if self._handle is None:
            raise RuntimeError("the environment has no sandbox. Start it first.")
        return self._resources.usage_environment(self._handle.intent.resource_id)

    async def oom_kills(self) -> int | None:
        """Return the sandbox cgroup's OOM-kill count, or None.

        A Daytona sandbox is a container on its host's kernel, and that kernel's
        ``/proc/vmstat`` counts the whole host. So the count comes from the cgroup.
        """
        return await oom_kills(self, "cgroup")
