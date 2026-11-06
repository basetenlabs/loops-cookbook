"""Shared validated training configuration; no model or network fixtures."""

import sys
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest
from support.builders import make_config

# The episode rig's ``make`` and ``script``, for the episode tests in unit and examples.
pytest_plugins = ["support.episode_rig"]

# The tests that need Harbor; CI's without-harbor job runs everything else. The
# repository-wide checks import every package, Harbor's too.
if find_spec("harbor") is None:
    collect_ignore_glob = [
        "unit/harbor/*",
        "unit/test_architecture.py",
        "unit/test_contract.py",
        "unit/test_public_surface.py",
        "patterns/test_harbor_*",
        "examples/test_setting_examples.py",
        "examples/test_terminal_bench.py",
        "live/test_blaxel_output.py",
    ]

# Harbor agents live beside their training config; the CLI makes that directory
# importable at runtime, and the tests do the same here. Every module name is unique
# across these directories, so they can all be importable at once.
ROOT = Path(__file__).resolve().parents[1]
for directory in (
    "patterns/harbor/in_process_direct",
    "patterns/harbor/in_process_openai_http",
    "patterns/harbor/local",
    "examples/critic",
    "examples/multi_step",
    "examples/terminal_bench",
    "examples/episode_basics",
    # The offline stand-ins for the model and a Harbor environment: ``baseline_model``
    # and ``baseline_environment``, imported by tests and named by Harbor jobs.
    "tests/fixtures/offline",
    "examples/sat",
):
    sys.path.insert(0, str(ROOT / directory))


@pytest.fixture(autouse=True)
def providers_are_never_closed_under_a_call():
    """Fails a test in which a fake provider was closed while a call to it was still in
    flight: closing waits for every operation first."""
    from support import resources

    resources.LIVE.clear()
    yield
    closed_under = {p.name: p.closed_under for p in resources.LIVE if p.closed_under}
    resources.LIVE.clear()
    assert not closed_under, closed_under


class LoopsCapabilities:
    """``GET /v1/loops/capabilities`` as tests see it: out of reach until a test sets
    ``models`` (served with ``status``), and every request kept in ``requests``. The
    Loops session that ``ServiceClient()`` opens first is out of reach with it."""

    def __init__(self):
        self.models = None
        self.status = 200
        self.requests = []

    def answer(self, request):
        self.requests.append(request)
        if self.models is None:
            raise httpx.ConnectError("tests are offline", request=request)
        return httpx.Response(self.status, json={"supported_models": self.models})

    def open_session(self, request):
        """``POST /v1/loops/sessions``, which ``ServiceClient()`` sends: offline, or the
        ``status`` the capabilities get when it is a refusal."""
        if self.models is None:
            raise httpx.ConnectError("tests are offline", request=request)
        if self.status in (401, 403):
            return httpx.Response(self.status, json={"error": "rejected"})
        return httpx.Response(200, json={"session": {"id": "session"}})


@pytest.fixture(autouse=True)
def loops_capabilities(request, monkeypatch):
    """Every test but a live Loops one asks this fake, not Loops, what Loops serves.

    The fake answers at the HTTP transport, under the SDK's own client, so the SDK
    still sends the request and reads the response. Other requests go through as
    before.
    """
    capabilities = LoopsCapabilities()
    if request.node.get_closest_marker("live_loops") is None:
        send = httpx.AsyncHTTPTransport.handle_async_request
        send_sync = httpx.HTTPTransport.handle_request

        async def handle_async_request(transport, request):
            if request.url.path.endswith("/v1/loops/capabilities"):
                return capabilities.answer(request)
            return await send(transport, request)

        def handle_request(transport, request):
            if request.url.path.endswith("/v1/loops/sessions"):
                return capabilities.open_session(request)
            return send_sync(transport, request)

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async_request)
        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    return capabilities


@pytest.fixture
def config(tmp_path):
    """``config(**overrides)``: a valid nested ``Config`` with small test limits.

    Overrides name a setting by its leaf when that is unique (``mode="async"``) or
    by its dotted path (``**{"rollout.grading.timeout_seconds": 5}``). Run output
    goes under the test's own directory unless a test names another.
    """

    def build(**overrides):
        settings = {
            "group_size": 4,
            "groups_per_step": 1,
            "max_concurrent_rollouts": 4,
            "max_concurrent_verifiers": 2,
            "output_dir": str(tmp_path / "runs"),
            # Small, so a test that waits one out stays fast.
            "rollout.cleanup.timeout_seconds": 5,
            "rollout.cleanup.cancellation_grace_seconds": 1,
            "rollout.cleanup.process_kill_grace_seconds": 1,
            **overrides,
        }
        # These tests were written against `min_group_size = 2` -- two graded
        # rollouts are enough to train -- so the tolerance has to track group_size
        # for a test that overrides it.
        group_size = settings.get("group_size", settings.get("training.group_size"))
        if "max_group_losses" not in settings and "training.max_group_losses" not in settings:
            settings["max_group_losses"] = max(0, group_size - 2)
        return make_config(**settings)

    return build


@pytest.fixture
def baseline_renderer(monkeypatch):
    """``loops check`` and runs build the baseline model's renderer: no download.

    For tests that go through the CLI, which assembles its own run; a test that builds
    one passes ``renderer_factory`` to ``assemble`` instead.
    """
    from baseline_model import create_renderer

    monkeypatch.setattr(
        "loops_cookbook.rl.recipe.configured_renderer", lambda config: create_renderer()
    )


@pytest.fixture
def loops_check(monkeypatch, capsys, tmp_path, baseline_renderer):
    """``loops_check(path, *arguments, cwd=tmp_path) -> (code, out, err)``: ``loops check``
    of the config at ``path``, offline, run from ``cwd``.

    A config's output directory is relative to where it runs, and Harbor jobs name
    their tasks from the repository root. The provider credentials are check-only:
    check builds a provider's adapter and calls nothing, and asks the faked Loops
    what it serves.
    """
    from loops_cookbook import command

    monkeypatch.setenv("BL_WORKSPACE", "check-only")
    monkeypatch.setenv("BL_API_KEY", "check-only")
    monkeypatch.setenv("BASETEN_API_KEY", "check-only")  # also the LLM judge's key variable

    def check(path, *arguments, cwd=tmp_path):
        monkeypatch.chdir(cwd)
        try:
            code = command.main(["check", str(path), *arguments])
        except SystemExit as exit:
            code = exit.code
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return check


@pytest.fixture
async def blaxel(monkeypatch, tmp_path: Path):
    """Blaxel, faked at its two APIs; each sandbox's files and commands under
    ``tmp_path / "sandboxes" / <name>``. The SDK retries a failed read as often as it
    would, with no backoff: the fake answers at once."""
    pytest.importorskip("blaxel")
    from blaxel.core.sandbox import transient_retry
    from support.fake_blaxel import FakeBlaxel

    from loops_cookbook.providers.blaxel import usage

    fake = FakeBlaxel(tmp_path / "sandboxes")
    monkeypatch.setattr(usage, "SandboxInstance", NS(get=fake.get))
    monkeypatch.setattr(transient_retry, "_backoff_delay_seconds", lambda *args: 0)
    yield fake
    await fake.reap()


@pytest.fixture
async def blaxel_services(blaxel, tmp_path: Path):
    """``await blaxel_services(settings)``: one attempt's resource services, through a
    real registry and the real Blaxel adapter, with ``[providers.blaxel]`` ``settings``."""
    from support.fake_blaxel import SETTINGS
    from support.resources import episode, local

    from loops_cookbook.resources.registry import ResourceRegistry

    registries = []

    async def services(settings=SETTINGS):
        registry = ResourceRegistry.open(
            tmp_path / f"resources-{len(registries)}",
            {"blaxel": blaxel.adapter(settings)},
            mode="write",
            run_id="run-1",
            kind=local,
        )
        registries.append(registry)
        return episode(registry, f"attempt-{len(registries)}")

    yield services
    for registry in registries:
        await registry.close()


@pytest.fixture
async def blaxel_sandbox(blaxel_services):
    """``await blaxel_sandbox()``: a provisioned Blaxel sandbox, as an episode's ``ctx.sandbox``.

    Its session is the layer's, and its client is closed when the test ends, as the
    layer closes it after the episode's ``close()``.
    """
    from support.blaxel_api import sandbox_request

    from loops_cookbook.episodes.api import SandboxAttachment
    from loops_cookbook.providers.blaxel import Blaxel

    attachments = []

    async def provision():
        service = await blaxel_services()
        n = len(attachments)
        handle = await service.provision(sandbox_request(key=f"task__{n}", name=f"loops-r{n}"))
        resource_id = handle.intent.resource_id
        attachment = SandboxAttachment(resource_id, service)
        attachments.append(attachment)
        return Blaxel(handle.resource.name, service.usage_environment(resource_id), attachment)

    yield provision
    for attachment in attachments:
        await attachment.close()
