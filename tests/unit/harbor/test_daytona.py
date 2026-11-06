"""The cookbook's Harbor Daytona environment: Harbor starts it, the registry owns it.

Harbor's real Daytona environment runs against a fake Daytona: the provider's control
plane (``FakeApi``), whose client is the only one Harbor's environment reaches its
sandbox through, and every command and upload Harbor makes is recorded. The tests observe what Daytona was asked
to create and what the registry did, never the subclass's internals.
"""

from __future__ import annotations

import re
from types import SimpleNamespace as NS

import pytest

hd = pytest.importorskip("harbor.environments.daytona.environment")

from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from support.daytona_api import CREDENTIALS, REQUIRED, SETTINGS, StatusError
from support.fakes import no_wait
from support.resources import episode, local

from loops_cookbook.harbor import attempts
from loops_cookbook.providers.daytona.adapter import DaytonaAdapter
from loops_cookbook.resources.api import CleanupStatus
from loops_cookbook.resources.registry import ResourceRegistry

from .daytona_sandboxes import TASK, HarborSandbox, daytona_api, environments


@pytest.fixture
def log():
    return []


@pytest.fixture
def shell():
    """What the sandboxes' commands read (``files``), answer, fail on, and were uploaded."""
    return NS(files={}, answers={}, failing=set(), uploads={})


@pytest.fixture
def api(log, shell):
    """The provider's control plane; its client hands Harbor the sandboxes it runs in."""
    return daytona_api(log, shell)


@pytest.fixture
def adapter(api):
    return DaytonaAdapter(SETTINGS, CREDENTIALS, api, sleep=no_wait)


@pytest.fixture
async def registry(tmp_path, adapter):
    opened = ResourceRegistry.open(
        tmp_path / "state", {"daytona": adapter}, mode="write", run_id="run-1", kind=local
    )
    yield opened
    await opened.close()


@pytest.fixture
def resources(registry, adapter):
    service = episode(registry, "attempt-1")
    attempts.bind(
        "attempt-1", service, {"daytona": adapter.deadlines}, {"daytona": adapter.settings}
    )
    yield service
    attempts.forget("attempt-1")


@pytest.fixture
def environment(tmp_path, resources):
    """A Harbor Daytona environment for the bound attempt, over a task's Dockerfile."""
    return environments(tmp_path)


def called(api, verb):
    return [subject for call, subject in api.calls if call == verb]


# ------------------------------------------------------------- construction


@pytest.mark.parametrize(
    "kwargs",
    [
        {"auto_snapshot": True},
        {"snapshot_template_name": "harbor__{name}__snapshot"},
        {"connection_pool_maxsize": 8},
    ],
    ids=["auto_snapshot", "snapshot_template_name", "connection_pool_maxsize"],
)
def test_a_path_through_harbors_own_daytona_client_is_refused_at_construction(environment, kwargs):
    """Harbor would build its own client from DAYTONA_* in the environment for these."""
    [key] = kwargs
    with pytest.raises(ValueError, match=key):
        environment(**kwargs)


# ----------------------------------------------------------- the one client


async def test_a_target_set_only_in_the_toml_reaches_every_client_harbor_uses(
    environment, bound, monkeypatch, log, shell
):
    """Every Daytona client built, the provider's or Harbor's, records its target; the
    only one is the provider's, with ``[providers.daytona] target``, and Harbor's
    commands run in a sandbox it gave."""
    from harbor.environments.daytona import snapshots

    from loops_cookbook.providers.daytona import api as facade
    from loops_cookbook.providers.daytona import create_adapter

    monkeypatch.delenv("DAYTONA_TARGET", raising=False)
    built: list[str | None] = []
    boxes: dict[str, HarborSandbox] = {}

    class Sdk:
        """The SDK's ``AsyncDaytona``."""

        def __init__(self, config=None):
            self.target = None if config is None else config.target
            built.append(self.target)

        async def create(self, params, timeout=None):
            box = HarborSandbox(
                f"id-{len(boxes)}", params.labels, "started", log, shell, params.name
            )

            async def delete():
                boxes.pop(box.id)

            box.delete = delete
            box.target = self.target
            boxes[box.id] = box
            return box

        async def get(self, id_or_name):
            if id_or_name not in boxes:
                raise StatusError(404)
            return boxes[id_or_name]

        async def close(self):
            pass

    for module in (facade, hd, snapshots):
        monkeypatch.setattr(module, "AsyncDaytona", Sdk)
    adapter = create_adapter(
        {"target": "eu", **REQUIRED}, environ={"DAYTONA_API_KEY": "dtn-key-123"}
    )
    env = environment(attempt=bound(adapter))
    await env.start(force_build=False)
    assert (await env.exec("echo hi")).return_code == 0
    await env.stop(delete=True)
    assert built == ["eu"]
    assert "echo hi" in " ".join(log) and boxes == {}


# -------------------------------------------------------------- lifecycle


async def test_start_provisions_through_the_registry_and_stop_releases(environment, api, resources):
    env = environment()
    await env.start(force_build=False)
    [config] = api.created
    assert called(api, "create") == [config["name"]]  # once; Harbor's retries never run
    labels = config["labels"]
    assert labels["harbor.session_id"] == "trial__env"  # Harbor's own labels
    assert labels["loops.owner_id"] == "attempt-1" and labels["loops.run_id"] == "run-1"
    assert config["resources"] == {"cpu": 2, "memory": 4, "disk": 3}  # the task's, independent
    # The job set no lifecycle, so [providers.daytona]'s: a lost sandbox stops and is deleted.
    assert (config["auto_stop_interval"], config["auto_delete_interval"]) == (120, 0)
    assert config["ttl_minutes"] == 1440
    assert "env_vars" not in config  # Harbor sends the task env with each command instead
    assert env.usage_environment()["DAYTONA_SANDBOX_ID"] in api.sandboxes
    await env.stop(delete=True)
    assert api.sandboxes == {}
    assert (await resources.close()).complete


async def test_a_lifecycle_the_job_states_is_kept(environment, api):
    env = environment(auto_stop_interval_mins=30, auto_delete_interval_mins=5)
    await env.start(force_build=False)
    [config] = api.created
    assert (config["auto_stop_interval"], config["auto_delete_interval"]) == (30, 5)
    await env.stop(delete=True)


async def test_a_job_lifecycle_that_never_ends_a_sandbox_creates_none(environment, api, bound):
    env = environment(attempt=bound(ttl_minutes=0), auto_stop_interval_mins=0)
    with pytest.raises(Exception, match="never stop"):
        await env.start(force_build=False)
    assert called(api, "create") == []


async def test_an_ambiguous_create_is_read_back_never_created_twice(environment, api):
    api.create_errors, api.land_before_error = [StatusError(502, "bad gateway")], True
    env = environment()
    await env.start(force_build=False)
    assert len(called(api, "create")) == 1 and len(api.sandboxes) == 1
    await env.stop(delete=True)


@pytest.fixture
async def bound(tmp_path, api):
    """An attempt bound to a Daytona provider with ``[providers.daytona]`` changed by
    ``update``, or to ``adapter``; its environments are built with
    ``environment(attempt=...)``."""
    registries = []

    def bind(adapter=None, **update):
        adapter = adapter or DaytonaAdapter(
            SETTINGS.model_copy(update=update), CREDENTIALS, api, sleep=no_wait
        )
        registry = ResourceRegistry.open(
            tmp_path / f"bound-{len(registries)}",
            {"daytona": adapter},
            mode="write",
            run_id="run-b",
            kind=local,
        )
        attempt = f"attempt-b{len(registries)}"
        registries.append((attempt, registry))
        attempts.bind(
            attempt,
            episode(registry, attempt),
            {"daytona": adapter.deadlines},
            {"daytona": adapter.settings},
        )
        return attempt

    yield bind
    for attempt, registry in registries:
        attempts.forget(attempt)
        await registry.close()


def thread_variables(log):
    """The thread variables Harbor's exec passed with the last command."""
    return dict(
        re.findall(r"\b(OMP_NUM_THREADS|OPENBLAS_NUM_THREADS|MKL_NUM_THREADS)=(\S+)", log[-1])
    )


def each(count):
    return {name: count for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}


@pytest.mark.parametrize(
    ("table", "cpus", "env", "expected"),
    [
        ({}, 2, {}, {}),
        ({"thread_variables_from_vcpus": True}, 2, {}, each("2")),
        ({"thread_variables_from_vcpus": True, "cpu": 3}, None, {}, each("3")),
        (
            {"thread_variables_from_vcpus": True},
            2,
            {"OMP_NUM_THREADS": "8"},
            {**each("2"), "OMP_NUM_THREADS": "8"},
        ),
    ],
    ids=["off_by_default", "task_cpus", "table_cpu", "task_env_wins"],
)
async def test_thread_variables_follow_the_sandboxs_vcpus_only_when_asked(
    environment, bound, log, table, cpus, env, expected
):
    """When on, every command gets them, over the image's ``ENV``; a task's own win."""
    config = EnvironmentConfig(cpus=cpus, memory_mb=4096, storage_mb=3072, env=env)
    harbor_env = environment(attempt=bound(**table), config=config)
    await harbor_env.start(force_build=False)
    await harbor_env.exec("python train.py")
    assert thread_variables(log) == expected
    await harbor_env.stop(delete=True)


async def test_a_bad_allowlist_for_a_later_phase_fails_before_any_sandbox(environment, api):
    mixed = ["example.com", "10.0.0.0/8"]  # Daytona takes domains or CIDRs, not both
    later = NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=mixed)
    env = environment(phase_network_policies=[later])
    with pytest.raises(ValueError):
        await env.start(force_build=False)
    assert api.created == []


# ------------------------------------------------------------------ the image


async def test_the_image_is_harbors_build_of_the_tasks_dockerfile(environment, api):
    env = environment()
    await env.start(force_build=False)
    [config] = api.created
    assert config["image"]["dockerfile"] == TASK  # the user's Dockerfile, as written
    assert config["image"]["context"]  # the workspace is in the build context
    await env.stop(delete=True)


def test_a_gpu_task_is_refused_at_construction(environment):
    with pytest.raises(ValueError, match="GPU"):
        environment(config=EnvironmentConfig(cpus=2, memory_mb=4096, gpus=1))


# ----------------------------------------------------------------- cleanup


async def test_a_label_sweep_finds_a_sandbox_harbor_left_running(environment, api, adapter):
    from loops_cookbook.resources.cleanup import sweep_labels

    env = environment()
    await env.start(force_build=False)  # the process dies here: Harbor never stops it
    report = await sweep_labels({"daytona": adapter}, "run-1")
    released = [r for r in report.results if r.status is CleanupStatus.RELEASED]
    assert len(released) == 1 and api.sandboxes == {}
    # A listing proves nothing about what it did not list, so the sweep never says "all".
    assert not report.complete


# As read in a Daytona sandbox on 4 Oct 2026, after one allocation past its 1 GiB limit.
MEMORY_EVENTS = "low 0\nhigh 0\nmax 23\noom 1\noom_kill 1\noom_group_kill 0\n"
HOST_VMSTAT = "pgfault 99\noom_kill 31560\n"


async def test_the_oom_count_is_the_sandboxs_cgroup_not_the_hosts_kernel(environment, shell):
    """A Daytona sandbox is a container: /proc/vmstat counts every OOM kill on its host."""
    shell.files.update({"/sys/fs/cgroup/memory.events": MEMORY_EVENTS, "/proc/vmstat": HOST_VMSTAT})
    env = environment()
    await env.start(force_build=False)
    assert await env.oom_kills() == 1
    await env.stop(delete=True)


async def test_without_cgroup_v2_the_oom_count_is_unknown(environment, shell):
    """Under cgroup v1 the memory controller's root may be a parent's, counting kills
    beyond the sandbox's; only v2's memory.events is the sandbox's own, as observed."""
    shell.files.update(
        {
            "/sys/fs/cgroup/memory/memory.oom_control": "oom_kill_disable 0\noom_kill 7\n",
            "/proc/vmstat": HOST_VMSTAT,
        }
    )
    env = environment()
    await env.start(force_build=False)
    assert await env.oom_kills() is None
    await env.stop(delete=True)
