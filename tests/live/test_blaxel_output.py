"""A command's whole output reaches Harbor on real Blaxel, past the 64 KB process record.

One sandbox, provisioned by the cookbook's registry and attached by its Harbor
environment, so the commands run through Harbor's own ``exec`` and the cookbook's
``complete_output``. It pins what the offline tests assume about Blaxel: the marker
text, and that the process logs hold each stream whole.
"""

import contextlib
import shlex
import uuid

import pytest
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from loops_cookbook.harbor import attempts
from loops_cookbook.harbor.blaxel import BlaxelEnvironment
from loops_cookbook.providers import create_adapters
from loops_cookbook.providers.blaxel import incidents, usage
from loops_cookbook.resources.api import ResourceRequest
from loops_cookbook.resources.registry import ResourceRegistry
from loops_cookbook.resources.services import EpisodeResourceService

pytestmark = [pytest.mark.live_blaxel, pytest.mark.asyncio(loop_scope="module")]

PROVIDER = {
    "region": "us-was-1",
    "image_build_timeout_seconds": 420,
    "create_timeout_seconds": 120,
    "attach_timeout_seconds": 60,
    "reconcile_timeout_seconds": 60,
    "release_timeout_seconds": 120,
    "confirm_absent_timeout_seconds": 60,
    "list_timeout_seconds": 60,
}
ATTEMPT = "attempt-live-output"


def printing(lines: int, width: int = 15) -> str:
    """A command printing ``lines`` numbered lines, each its number zero-padded to
    ``width``, with every hundredth also on stderr."""
    script = (
        "import sys\n"
        f"for i in range({lines}):\n"
        f"    sys.stdout.write('line %0{width}d\\n' % i)\n"
        "    i % 100 or sys.stderr.write('err %d\\n' % i)\n"
    )
    return f"python3 -c {shlex.quote(script)}"


def expected(lines: int, width: int = 15) -> tuple[str, str]:
    stdout = "".join(f"line {i:0{width}d}\n" for i in range(lines))
    stderr = "".join(f"err {i}\n" for i in range(0, lines, 100))
    return stdout, stderr


@pytest.fixture(scope="module")
async def environment(tmp_path_factory):
    """The cookbook's Harbor Blaxel environment, attached to one real sandbox."""
    root = tmp_path_factory.mktemp("live-output")
    adapters = await create_adapters({"blaxel": PROVIDER})
    registry = ResourceRegistry.open(
        root / "state", adapters, mode="write", run_id=f"live-output-{uuid.uuid4().hex[:8]}"
    )
    service = EpisodeResourceService(registry, ATTEMPT, contextlib.nullcontext)
    attempts.bind(ATTEMPT, service, {"blaxel": adapters["blaxel"].deadlines})
    try:
        handle = await service.provision(
            ResourceRequest(
                provider="blaxel",
                kind="sandbox",
                logical_key="live-output",
                specification={
                    "sandbox": {"image": "blaxel/base-image:latest", "memory": 2048, "ttl": "1h"}
                },
            )
        )
        source = root / "env"
        source.mkdir()
        (source / "Dockerfile").write_text("FROM blaxel/base-image:latest\n")
        paths = TrialPaths(root / "trial")
        paths.mkdir()
        env = BlaxelEnvironment(
            environment_dir=source,
            environment_name="live-output",
            session_id="live-output",
            trial_paths=paths,
            task_env_config=EnvironmentConfig(),
            loops_attempt_id=ATTEMPT,
        )
        env._sandbox = await usage.attach(
            handle.resource,
            timeout_seconds=PROVIDER["attach_timeout_seconds"],
        )
        yield env
        await incidents.close_clients(env._sandbox)
        report = await service.close()
        assert report.complete, report
    finally:
        attempts.forget(ATTEMPT)
        await registry.close()


# Blaxel's process API spends about 2 ms on each captured line (22 Oct 2026: 2 MB as
# 100,000 lines took over 150 s, as one line 10 s), so 2 MB is sent in long lines.
@pytest.mark.parametrize(
    ("lines", "width"),
    [(50, 15), (10_000, 15), (20_000, 95)],
    ids=["1KB", "200KB", "2MB"],
)
async def test_every_byte_of_both_streams_comes_back(environment, lines, width):
    result = await environment.exec(printing(lines, width), timeout_sec=120)
    stdout, stderr = expected(lines, width)
    assert result.return_code == 0
    assert result.stdout == stdout, (len(result.stdout), len(stdout))
    assert result.stderr == stderr


async def test_blaxel_still_marks_a_cut_record_the_way_the_fix_looks_for(environment):
    """If Blaxel rewords its marker, ``complete_output`` would silently stop firing."""
    process = environment._sandbox.process
    started = await process.exec(
        {"command": printing(10_000), "working_dir": "/", "env": {}, "keep_alive": True}
    )
    identifier = started.pid or started.name
    await process.wait(identifier, max_wait=120_000, interval=1_000)
    record = await process.get(identifier)  # the raw record, which nothing completes
    assert record.stdout.startswith(incidents.TRUNCATED)
    assert len(record.stdout) < 70_000
