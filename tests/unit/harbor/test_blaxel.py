"""The cookbook's Harbor Blaxel environment: Harbor starts it, the registry owns it.

Harbor's real environment class runs against a fake Blaxel control plane. Every
sandbox is recorded as Harbor-shaped actions, so the tests observe what Harbor
asked of the sandbox and what the registry did, never the subclass's internals.
"""

from __future__ import annotations

import asyncio
import tempfile
from types import SimpleNamespace as NS

import httpx
import pytest

hb = pytest.importorskip("harbor.environments.blaxel")

from support.blaxel_api import CREDENTIALS, SETTINGS, FakeApi, StatusError, not_dispatched
from support.fakes import MemoryTrainer, no_wait
from support.harness import Renderer
from support.resources import episode, local

from loops_cookbook.evidence import TaskRef
from loops_cookbook.harbor import attempts, blaxel
from loops_cookbook.harbor.attempts import SandboxNotReleased
from loops_cookbook.harbor.blaxel import BlaxelEnvironment
from loops_cookbook.model_access.api import ModelAccessMode
from loops_cookbook.providers.blaxel.adapter import BlaxelAdapter
from loops_cookbook.resources.registry import ResourceRegistry
from loops_cookbook.rl.run import Run
from loops_cookbook.runtime.api import EpisodeResult, GradeResult, RuntimeCapabilities

from .blaxel_sandboxes import Sandbox, attaching, blaxel_api, blaxel_usage, environments

CREATE_SIDE = ("create", "get", "update_ttl", "images", "build", "attach")


@pytest.fixture(autouse=True)
def quick_readiness(monkeypatch):
    monkeypatch.setattr(hb, "_SANDBOX_READY_INTERVAL_SEC", 0)


@pytest.fixture
def log():
    return []


@pytest.fixture
def api(monkeypatch, log):
    return blaxel_api(monkeypatch, log)


@pytest.fixture
def adapter(api):
    return BlaxelAdapter(SETTINGS, CREDENTIALS, api, sleep=no_wait)


@pytest.fixture
async def registry(tmp_path, adapter):
    opened = ResourceRegistry.open(
        tmp_path / "state", {"blaxel": adapter}, mode="write", run_id="run-1", kind=local
    )
    yield opened
    await opened.close()


@pytest.fixture
def resources(registry, adapter):
    service = episode(registry, "attempt-1")
    attempts.bind("attempt-1", service, {"blaxel": adapter.deadlines})
    yield service
    attempts.forget("attempt-1")


@pytest.fixture
def environment(tmp_path, resources):
    """A Harbor Blaxel environment for the bound attempt; ``mode`` picks its image source."""
    return environments(tmp_path)


def created(api, verb):
    return [name for call, name in api.calls if call == verb]


# ------------------------------------------------------------- construction


async def test_two_registries_in_one_process_do_not_interfere(
    monkeypatch, tmp_path, log, api, adapter, resources, environment
):
    """Two runs, each with its own Blaxel provider: an environment reaches only its own."""
    other_api = FakeApi()
    attaching(other_api, log)
    blaxel_usage(monkeypatch, api, other_api)
    other = BlaxelAdapter(SETTINGS, CREDENTIALS, other_api, sleep=no_wait)  # built last
    other_registry = ResourceRegistry.open(
        tmp_path / "other", {"blaxel": other}, mode="write", run_id="run-2", kind=local
    )
    other_resources = episode(other_registry, "attempt-2")
    attempts.bind("attempt-2", other_resources, {"blaxel": other.deadlines})
    try:
        mine = environment(session="a__env")
        theirs = environment(session="b__env", attempt="attempt-2")
        await asyncio.gather(mine.start(force_build=False), theirs.start(force_build=False))
        assert created(api, "create") == [mine._sandbox_name]
        assert created(other_api, "create") == [theirs._sandbox_name]
        await theirs.stop(delete=True)
        assert (await other_resources.close()).complete
        await other_registry.close()  # the other run ends; this one carries on
        await mine.exec("true")
        await mine.stop(delete=True)
    finally:
        attempts.forget("attempt-2")
    assert (await resources.close()).complete
    assert api.sandboxes == {} and other_api.sandboxes == {}


# -------------------------------------------------------------- lifecycle


@pytest.fixture
def provisioned(monkeypatch, resources):
    """Every handle the attempt's resources provisioned, in order."""
    handles = []
    provision = resources.provision

    async def recording(request):
        handles.append(await provision(request))
        return handles[-1]

    monkeypatch.setattr(resources, "provision", recording)
    return handles


@pytest.mark.parametrize("mode", ["dockerfile", "docker_image"])
async def test_start_provisions_harbors_sandbox_then_stop_releases_it(
    environment, api, resources, provisioned, mode
):
    env = environment(mode)
    await env.start(force_build=False)
    assert created(api, "create") == [env._sandbox_name]  # Harbor's own name
    assert created(api, "build") == [env._image_name]  # built once; not yet present
    (handle,) = provisioned
    assert env.usage_environment() == resources.usage_environment(handle.intent.resource_id)
    await env.stop(delete=True)
    assert env._sandbox_name not in api.sandboxes and env._image_name in api.images
    assert (await resources.close()).complete


@pytest.mark.parametrize("force_build", [False, True])
async def test_force_build_rebuilds_an_image_that_already_exists(environment, api, force_build):
    env = environment()
    api.images.add(env._image_name)
    await env.start(force_build=force_build)
    assert created(api, "build") == ([env._image_name] if force_build else [])
    await env.stop(delete=True)


async def test_tasks_with_identical_environments_share_one_image(environment, api):
    first = environment(session="a__env", name="task-a")
    second = environment(session="b__env", name="task-b")
    other = environment("docker_image", session="c__env", name="task-c")
    await asyncio.gather(*(env.start(force_build=False) for env in (first, second, other)))
    assert first._image_name == second._image_name != other._image_name
    assert sorted(created(api, "build")) == sorted({first._image_name, other._image_name})
    await asyncio.gather(*(env.stop(delete=True) for env in (first, second, other)))


@pytest.mark.parametrize("force_build", [False, True])
async def test_a_task_on_an_imported_blaxel_image_starts_from_it_with_no_build(
    monkeypatch, environment, api, force_build
):
    def describes_a_build(*args, **kwargs):
        raise AssertionError("an imported image's build was described")

    monkeypatch.setattr(blaxel, "content_image_name", describes_a_build)
    monkeypatch.setattr(BlaxelEnvironment, "_build_specification", describes_a_build)
    env = environment("docker_image", docker_image="blaxel://sandbox/my-image:v1")
    await env.start(force_build=force_build)
    assert created(api, "build") == []
    (name,) = created(api, "create")
    assert api.sandboxes[name].config["image"] == "sandbox/my-image:v1"  # as written
    await env.stop(delete=True)


async def test_a_docker_image_without_the_blaxel_scheme_is_a_base_the_run_builds_on(
    environment, api
):
    """Harbor's meaning: a Docker Hub image under a sandbox/ namespace is built on."""
    env = environment("docker_image", docker_image="sandbox/some-base:1")
    await env.start(force_build=False)
    assert created(api, "build") == [env._image_name]
    await env.stop(delete=True)


def test_identical_contexts_share_an_image_and_one_changed_file_does_not(environment):
    """Only the files differ, so only the content hash can tell the images apart."""
    first, second, changed = (
        environment(
            directory=directory,
            session=f"{directory}__env",
            name=f"task-{directory}",
            files={"app/data.txt": data},
        )
        for directory, data in (("a", "same"), ("b", "same"), ("c", "changed"))
    )
    assert first._image_name == second._image_name != changed._image_name


@pytest.mark.parametrize("mode", [0o755, 0o600])
def test_a_context_that_differs_only_in_a_files_mode_gets_its_own_image(
    tmp_path, environment, mode
):
    """Harbor's content hash reads paths and bytes, but COPY keeps each file's mode."""
    for directory, wanted in (("a", 0o644), ("b", mode)):
        script = tmp_path / directory / "app" / "run.sh"
        script.parent.mkdir(parents=True)
        script.write_text("echo hi\n")
        script.chmod(wanted)
    first, second = (
        environment(
            directory=directory,
            session=f"{directory}__env",
            name=f"task-{directory}",
            files={"app/run.sh": "echo hi\n"},
        )
        for directory in ("a", "b")
    )
    assert first._image_name.startswith("harbor-img-")
    assert first._image_name != second._image_name


@pytest.mark.parametrize(
    "skipped", [".git/HEAD", "app/__pycache__/m.pyc", "app/.DS_Store", "link", "empty/"]
)
async def test_an_environment_with_files_harbors_hash_skips_keeps_harbors_image_name(
    tmp_path, environment, api, skipped
):
    source = tmp_path / "env-dockerfile"
    source.mkdir()
    if skipped == "link":
        (source / "link").symlink_to(tmp_path)
    elif skipped.endswith("/"):
        (source / skipped).mkdir()
    else:
        (source / skipped).parent.mkdir(parents=True, exist_ok=True)
        (source / skipped).write_text("not in the hash")
    env = environment()
    assert env._image_name.startswith("harbor-img-task-")  # Harbor's per-task name
    await env.start(force_build=False)
    assert created(api, "build") == [env._image_name]
    await env.stop(delete=True)


@pytest.mark.parametrize("delete", [True, False])
async def test_stop_closes_the_sandboxs_clients(environment, api, log, delete):
    sandboxes = attaching(api, log)
    env = environment()
    await env.start(force_build=False)
    await env.stop(delete=delete)
    (sandbox,) = sandboxes
    assert sandbox.fs.get_client().is_closed and sandbox.process.get_client().is_closed


async def test_a_release_that_fails_still_closes_the_sandboxs_clients(environment, api, log):
    """Raised from stop, as every provider's environment does (the contract tests)."""
    sandboxes = attaching(api, log)
    env = environment()
    await env.start(force_build=False)
    api.delete_errors = [StatusError(500, "boom")] * 50
    with pytest.raises(SandboxNotReleased):
        await env.stop(delete=True)
    (sandbox,) = sandboxes
    assert sandbox.fs.get_client().is_closed  # the clients are closed all the same


async def test_a_command_blaxel_never_received_is_replayed(environment, api, log):
    env = environment()

    attaching(api, log, fail_first=[not_dispatched()])
    await env.start(force_build=False)  # its mkdir first fails undispatched
    assert sum("mkdir -p" in entry for entry in log) == 1  # then runs, once


async def test_a_commands_whole_output_reaches_harbor_past_blaxels_64_kb_cut(environment, api, log):
    env = environment()
    output = "".join(f"line {i:06d}\n" for i in range(20_000))  # 240 KB

    attaching(api, log, output=output)
    await env.start(force_build=False)
    result = await env.exec("print-output")
    assert result.stdout == output and result.return_code == 0
    assert log.count("process.logs p") == 1  # fetched only for the cut command
    await env.stop(delete=True)


async def test_the_same_harbor_name_twice_waits_for_the_first_sandbox_to_be_gone(
    environment, api, resources
):
    """A regrade retry reruns its trial under the same name."""
    first = environment(session="trial-grade__verifier")
    await first.start(force_build=False)
    api.linger = 2
    original = api.get

    async def get(name):
        api.tick()
        return await original(name)

    api.get = get
    stopping = asyncio.create_task(first.stop(delete=True))
    async with asyncio.timeout(2):  # the stop must reach Blaxel's delete
        while ("delete", first._sandbox_name) not in api.calls:
            await asyncio.sleep(0)
    second = environment(session="trial-grade__verifier")
    # Blaxel refuses a create while the name is taken, as the fake does: this start
    # succeeds only by waiting until the first sandbox is gone.
    await second.start(force_build=False)
    await stopping
    assert created(api, "create") == [first._sandbox_name] * 2
    await second.stop(delete=True)
    assert (await resources.close()).complete and api.sandboxes == {}


class OneSandboxEach:
    """A native runtime whose every attempt starts one Harbor Blaxel sandbox."""

    capabilities = RuntimeCapabilities(model_access_modes=frozenset({ModelAccessMode.DIRECT}))

    def __init__(self, environment):
        self.environment, self.starting, self.limits = environment, 0, None

    def built(self, settings, *, context):
        self.limits = context.limits  # the run's provider deadlines among them
        return self

    async def prepare(self):
        return [TaskRef("task-sandbox")]

    async def sample(self, attempt):
        attempt_id = attempt.spec.identity.attempt_id
        attempts.bind(attempt_id, attempt.services.resources, self.limits.provider_deadlines)
        try:
            env = self.environment(session=f"{attempt_id}__env", attempt=attempt_id)
            self.starting += 1
            await env.start(force_build=False)
            conversation = attempt.services.model_access.llm().conversation()
            await conversation.generate(messages=[{"role": "user", "content": "x"}])
            await env.stop(delete=True)
        finally:
            attempts.forget(attempt_id)
        return EpisodeResult()

    async def grade(self, attempt, rollout):
        return GradeResult(reward=float(rollout.identity.rollout_id.endswith("r1")))

    async def finish(self, attempt, rollout, disposition):
        pass

    async def abort(self, attempt, reason):
        pass

    async def close(self):
        pass


async def test_the_runs_provisioning_limit_admits_two_of_three_starts(
    tmp_path, config, adapter, api, environment
):
    runtime = OneSandboxEach(environment)
    run = Run(
        config(
            output_dir=str(tmp_path / "out"),
            max_steps=1,
            group_size=3,
            max_concurrent_rollouts=3,
            max_group_losses=0,
            **{"rollout.max_concurrent_provisioning": 2},
        ),
        MemoryTrainer(),
        runtime_factory=runtime.built,
        renderer_factory=lambda _config: Renderer(),
        adapters={"blaxel": adapter},
    )
    api.images.add(environment()._image_name)  # every trial runs one prebuilt image
    api.create_gate = asyncio.Event()
    running = asyncio.create_task(run.execute())
    async with asyncio.timeout(10):
        while runtime.starting < 3 or api.creating < 2:
            await asyncio.sleep(0.01)
    for _ in range(100):  # the third start has every chance to reach Blaxel
        await asyncio.sleep(0)
    assert api.creating == 2
    api.create_gate.set()
    assert (await asyncio.wait_for(running, 30)).ok
    assert len(created(api, "create")) == 3 and api.most_creating == 2
    assert not any(name.startswith("harbor-env") for name in api.sandboxes)


# --------------------------------------------------------- attach contract


def test_the_build_file_says_what_harbors_says_and_names_the_region():
    import tomllib

    from loops_cookbook.providers.blaxel.build_file import build_toml

    ours = tomllib.loads(build_toml("us-was-1", slim=False))
    assert ours == {
        "build": {**tomllib.loads(hb._NO_SLIM_BLAXEL_TOML)["build"], "region": "us-was-1"}
    }


@pytest.mark.parametrize("mode", ["dockerfile", "docker_image"])
async def test_a_trial_writes_no_files_to_describe_its_image(
    monkeypatch, tmp_path, environment, api, mode
):
    """Trials of a task share its image; a file one wrote could be read torn by another."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    builds = []
    build_image = api.build_image

    async def recording(name, build, labels, timeout, *, region, **sizes):
        builds.append(build)
        await build_image(name, build, labels, timeout, region=region, **sizes)

    api.build_image = recording
    env = environment(mode)
    source = sorted(env.environment_dir.rglob("*"))
    await env.start(force_build=False)
    assert list(scratch.iterdir()) == [] and sorted(env.environment_dir.rglob("*")) == source
    (build,) = builds
    assert build["slim"] is False  # the provider's build adds blaxel.toml
    await env.stop(delete=True)


@pytest.mark.parametrize("mode", ["dockerfile", "docker_image"])
async def test_attach_repeats_upstreams_post_create_startup(monkeypatch, environment, log, mode):
    """Upstream creates its own sandbox; the subclass attaches to the registry's."""

    class UpstreamSandboxes:
        @classmethod
        async def create(cls, config, safe=False):
            log.append("create")
            return Sandbox(config["name"], log)

        @classmethod
        async def update_ttl(cls, name, ttl):
            log.append("update_ttl")

    class BuiltImages:
        def get_async_httpx_client(self):
            async def get(path):
                log.append("images")
                payload = [
                    {
                        "metadata": {
                            "resourceType": "sandbox",
                            "name": upstream._image_name,
                            "status": "BUILT",
                        },
                        "spec": {"tags": ["latest"]},
                    }
                ]
                return NS(status_code=200, json=lambda: payload)

            return NS(get=get)

    monkeypatch.setattr(hb, "SandboxInstance", UpstreamSandboxes)
    monkeypatch.setattr(hb, "blaxel_client", BuiltImages())
    upstream = environment(mode, cls=hb.BlaxelEnvironment)
    await upstream.start(force_build=False)
    upstream_actions = [entry for entry in log if not entry.startswith(CREATE_SIDE)]
    log.clear()
    ours = environment(mode)
    await ours.start(force_build=False)
    assert [entry for entry in log if not entry.startswith(CREATE_SIDE)] == upstream_actions
    await ours.stop(delete=True)


# ------------------------------------------------------ a sandbox that went away


async def test_a_sandbox_deleted_under_a_command_is_lost_not_a_silent_success(
    environment, api, log
):
    """Harbor would report exit 0 and no output; the environment says the sandbox is gone."""
    from loops_cookbook.harbor.blaxel import SandboxLost

    env = environment()

    attaching(api, log, vanishes="sleep")
    await env.start(force_build=False)
    assert (await env.exec("echo fine")).return_code == 0
    with pytest.raises(SandboxLost, match="the sandbox is gone"):
        await env.exec("sleep 120")
    with pytest.raises(SandboxLost):
        await env.exec("python train.py")
    assert not any("python train.py" in entry for entry in log)  # never sent: it is gone
    await env.stop(delete=True)


async def test_a_sandbox_that_takes_no_command_for_the_replay_budget_is_lost(
    environment, api, log, monkeypatch
):
    """Deleted between commands: each later command would otherwise wait the budget again."""
    from loops_cookbook.harbor import blaxel
    from loops_cookbook.harbor.blaxel import SandboxLost

    monkeypatch.setattr(blaxel, "REPLAY_BUDGET_SECONDS", 0.05)
    monkeypatch.setattr("loops_cookbook.providers.blaxel.incidents.REPLAY_BUDGET_SECONDS", 0.05)
    env = environment()
    await env.start(force_build=False)
    env._sandbox._fail_first = [not_dispatched() for _ in range(1000)]  # deleted now
    with pytest.raises(SandboxLost, match="took no command"):
        await env.exec("ls")
    left = len(env._sandbox._fail_first)
    with pytest.raises(SandboxLost):
        await env.exec("python train.py")
    assert len(env._sandbox._fail_first) == left  # refused at once, nothing sent
    await env.stop(delete=True)


# ------------------------------------------------------ the OOM-kill report

VMSTAT = "nr_free_pages 81920\noom_kill 3\npgfault 12\n"


class Hangs(Sandbox):
    """A sandbox whose reads never return."""

    async def _wait(self, pid, max_wait, interval):
        if "vmstat" in self.command:
            await asyncio.Event().wait()
        return await super()._wait(pid, max_wait, interval)


class Refuses(Sandbox):
    """A sandbox whose connection drops under the read: not safe to replay."""

    async def _exec(self, request):
        if "vmstat" in request["command"]:
            raise httpx.ReadError("connection reset")
        return await super()._exec(request)


async def test_the_sandbox_kernels_oom_kill_count_is_read_through_exec(environment, api, log):
    env = environment()

    attaching(api, log, replies={"vmstat": (0, VMSTAT, "")})
    await env.start(force_build=False)
    assert await env.oom_kills() == 3
    assert "process.exec cmd=bash -c 'cat /proc/vmstat'" in log  # Harbor's own exec path
    await env.stop(delete=True)


@pytest.mark.parametrize(
    "sandbox",
    [
        lambda name, log, labels: Sandbox(name, log, labels, replies={"vmstat": (1, VMSTAT, "no")}),
        lambda name, log, labels: Sandbox(
            name, log, labels, replies={"vmstat": (0, "nr_free_pages 1\n", "")}
        ),
        lambda name, log, labels: Sandbox(
            name, log, labels, replies={"vmstat": (0, "oom_kill lots\n", "")}
        ),
        lambda name, log, labels: Refuses(name, log, labels),
        lambda name, log, labels: Hangs(name, log, labels),
    ],
    ids=["exit-nonzero", "no-counter", "garbled", "refused", "hangs"],
)
async def test_an_unreadable_oom_count_is_none_and_later_commands_still_run(
    environment, api, log, monkeypatch, sandbox
):
    from loops_cookbook.harbor import oom

    monkeypatch.setattr(oom, "OOM_READ_SECONDS", 0.05)
    env = environment()

    attaching(api, log, sandbox)
    await env.start(force_build=False)
    assert await env.oom_kills() is None
    assert (await env.exec("echo next")).return_code == 0
    await env.stop(delete=True)


async def test_a_lost_sandboxs_oom_count_is_none_not_an_error(environment, api, log):
    env = environment()

    attaching(api, log, vanishes="vmstat")
    await env.start(force_build=False)
    assert await env.oom_kills() is None  # its process vanished with the sandbox
    assert await env.oom_kills() is None  # and a sandbox known gone is not asked again
    assert sum("vmstat" in entry for entry in log) == 1
    await env.stop(delete=True)
