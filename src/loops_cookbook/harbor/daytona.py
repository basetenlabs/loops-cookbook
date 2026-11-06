"""The cookbook's Harbor-facing Daytona environment: Harbor starts it, the registry owns it.

Select it in a Harbor job with
``environment: {import_path: loops_cookbook.harbor.daytona:DaytonaEnvironment}`` and
configure ``[providers.daytona]``. Harbor's own Daytona environment does nearly
everything: it chooses the image (a declarative build of the task's Dockerfile, or a
prebuilt image), sizes the sandbox, applies the network policy, and runs commands and
file transfers. Its start changes. Harbor's creates the sandbox with tenacity
retries (up to ten times for a rate limit, minutes apart, and a second create after
an ambiguous failure) through a process-wide client it builds from ``DAYTONA_*`` in
the environment. Here the sandbox is one provision through the attempt's
``EpisodeResources``, which journals the intent, holds the create window, and
reconciles by name instead of creating twice; Harbor then reaches it through the
provider's own client, configured from ``[providers.daytona]``, so its ``target``
holds for every call. Harbor's client is never built, and a kwarg that only
configures or uses it (``auto_snapshot``, ``snapshot_template_name``,
``connection_pool_maxsize``) is refused. Its delete (``_stop_sandbox``) becomes a
cookbook release. So every sandbox is tracked, released with its attempt, and found
by ``loops cleanup``.

The rest is what Harbor's Daytona environment lacks for training:

- labels: the run's ownership labels (``loops.run_id``, ``loops.owner_id`` = the
  attempt, ...) are added to Harbor's ``harbor.*`` labels and the job's ``labels``;
- lifecycle: Harbor's defaults never stop a sandbox (``auto_stop_interval_mins`` 0); a
  job that sets neither ``auto_stop_interval_mins`` nor ``auto_delete_interval_mins``
  gets ``[providers.daytona]``'s, which stop an idle sandbox and delete it as it stops;
- threads, opted into by ``thread_variables_from_vcpus``: ``OMP_NUM_THREADS``,
  ``OPENBLAS_NUM_THREADS``, and ``MKL_NUM_THREADS`` are the sandbox's vCPUs for every
  command, since ``nproc`` counts the host's cores, unless the task sets them.

The constructor only looks the attempt up, so an environment Harbor constructs and
never starts provisions nothing. Only Harbor's direct mode is supported; compose
(Docker-in-Docker) needs its own evidence first. This module reaches Harbor's private
members, so it is pinned to Harbor 0.23: revisit it on any upgrade.
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
# Harbor's kwargs that act only through its own Daytona client, with their defaults.
HARBORS_CLIENT: dict[str, tuple[object, str]] = {
    "auto_snapshot": (False, "Harbor names and builds its snapshot with a client of its own"),
    "snapshot_template_name": (None, "Harbor resolves the snapshot with a client of its own"),
    "connection_pool_maxsize": (_UNSET, "it configures Harbor's own client"),
}
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def check_lifecycle(kwargs: Mapping[str, Any]) -> None:
    """Refuse a job's lifecycle interval that Daytona cannot take, before any create."""
    for key in LIFECYCLE:
        if key in kwargs and not lifecycle_minutes(kwargs[key]):
            raise ValueError(
                f"environment.kwargs.{key} must be a whole number of minutes, at least 0"
            )


class DaytonaEnvironment(NativeDaytonaEnvironment):
    """Harbor's Daytona environment, with the sandbox provisioned by the attempt and
    reached through the provider's own client."""

    def __init__(self, *args: Any, loops_attempt_id: str, **kwargs: Any) -> None:
        attempt = attempt_for(loops_attempt_id)
        deadlines = attempt.provider_deadlines.get("daytona")
        settings = attempt.provider_settings.get("daytona")
        if deadlines is None or not isinstance(settings, DaytonaSettings):
            raise RuntimeError("the cookbook's Daytona environment needs [providers.daytona]")
        for key, (default, why) in HARBORS_CLIENT.items():
            if kwargs.get(key, default) != default:
                raise ValueError(
                    f"environment.kwargs.{key} is not supported by the cookbook's Daytona "
                    f"environment: {why}, configured from the environment rather than "
                    "[providers.daytona]"
                )
        # Harbor's Daytona defaults never stop a sandbox; a lifecycle the job does not
        # state is [providers.daytona]'s, filled in by the adapter.
        self._lifecycle = {LIFECYCLE[key]: kwargs[key] for key in LIFECYCLE if key in kwargs}
        super().__init__(*args, **kwargs)
        if self._compose_mode:
            raise ValueError(
                "Harbor's Daytona compose (Docker-in-Docker) mode is not supported by the "
                "cookbook; use a Dockerfile or docker_image task"
            )
        if self._effective_gpus > 0:
            raise ValueError(
                "GPU tasks are not supported by the cookbook's Daytona environment; "
                "run them on another provider"
            )
        self._resources = attempt.resources
        self._handle: ResourceHandle | None = None
        self._release_failure: BaseException | None = None
        # A sandbox's nproc is its host's core count, not its vCPUs, so thread pools
        # sized from it oversubscribe the cgroup's CPU. Opted into by the table; Harbor
        # passes these with every command, over the image's ENV, and the task's own win.
        # A task that names no CPUs gets the table's, as its sandbox does.
        if settings.thread_variables_from_vcpus:
            cpus = self._effective_cpus or settings.cpu
            for name in THREAD_VARIABLES:
                self._persistent_env.setdefault(name, str(cpus))

    # ------------------------------------------------------- create and delete

    def _request(self, force_build: bool) -> ResourceRequest:
        """The sandbox Harbor's direct start would create, as the registry's JSON request.

        The image is Harbor's choice: the task's prebuilt ``docker_image`` when it uses
        one, else a declarative build of its Dockerfile. Harbor passes the task's
        environment with every command, so the sandbox is created without it, which
        also keeps resolved secrets out of the journal.
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
        """The size Harbor's create would request, ``None`` where it names nothing: the
        job's overrides and enforcement policies applied, memory and disk in whole GiB,
        rounded down. The provider fills what it leaves out."""
        resources = self._sandbox_resources()
        if resources is None:
            return SandboxResources()
        return SandboxResources(
            cpu=resources.cpu, memory_gib=resources.memory, disk_gib=resources.disk
        )

    async def start(self, force_build: bool) -> None:
        """Harbor's direct start, its create one provision through the registry.

        Harbor's own start builds its process-wide client from ``DAYTONA_*`` in the
        environment and creates with it; here the sandbox comes from the provider's
        client instead, and the rest is Harbor's post-create startup, unchanged.
        """
        self._validate_allowlist_targets()  # Harbor's own start does this first
        self._handle = await self._resources.provision(self._request(force_build))
        sandbox: Any = await self._resources.sandbox_client(self._handle.intent.resource_id)
        self._sandbox = sandbox
        # Harbor's post-create startup, unchanged: a test pins the upstream digest.
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
        """Release the sandbox through the registry; Harbor deletes nothing itself.

        Harbor logs a failed stop and carries on; the failure is kept and raised from
        ``stop()``, so the cookbook learns a sandbox may still be running.
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
        """Harbor's stop; a kept sandbox (``delete=False``) is released with its episode."""
        self._release_failure = None
        await super().stop(delete)
        failure, self._release_failure = self._release_failure, None
        if failure is not None:
            raise failure

    # ------------------------------------------------------------- reporting

    def usage_environment(self) -> Mapping[str, str | SecretStr]:
        """What a separate process needs to reach this environment's sandbox."""
        if self._handle is None:
            raise RuntimeError("the environment has no sandbox; start it first")
        return self._resources.usage_environment(self._handle.intent.resource_id)

    async def oom_kills(self) -> int | None:
        """The sandbox's OOM-kill count, or None: its cgroup's, since a Daytona sandbox
        is a container on its host's kernel, whose /proc/vmstat counts the whole host."""
        return await oom_kills(self, "cgroup")
