"""An image builds on real Blaxel through the cookbook, and its sandbox carries our labels.

The image's content is new each run, so the build is never skipped as already built:
it goes through the SDK's own ``ImageInstance.build`` and the cookbook's builder
payload, which an SDK release broke once (blaxel 0.4.19's ``build_labels``). The
sandbox started from it carries ``[providers.blaxel] labels`` and runs the image.
"""

import contextlib
import uuid

import pytest
from blaxel.core import ImageInstance
from blaxel.core.client.client import client

from loops_cookbook.providers import create_adapters
from loops_cookbook.providers.blaxel import incidents, usage
from loops_cookbook.providers.blaxel.api import BlaxelApi, image_build_specification
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
    "labels": {"team": "loops-live"},
}
ATTEMPT = "attempt-live-image-build"


async def test_a_new_image_builds_and_its_sandbox_runs_it_with_our_labels(tmp_path):
    marker = uuid.uuid4().hex
    image = f"loops-live-build-{marker[:12]}"
    built = ImageInstance.from_registry("python:3.12-slim").run_commands(
        f"echo {marker} > /loops-marker"
    )
    adapters = await create_adapters({"blaxel": PROVIDER})
    registry = ResourceRegistry.open(
        tmp_path / "state", adapters, mode="write", run_id=f"live-build-{marker[:8]}"
    )
    service = EpisodeResourceService(registry, ATTEMPT, contextlib.nullcontext)
    try:
        handle = await service.provision(
            ResourceRequest(
                provider="blaxel",
                kind="sandbox",
                logical_key="live-image-build",
                specification={
                    "sandbox": {"image": f"sandbox/{image}:latest", "memory": 2048, "ttl": "1h"},
                    "image": {
                        "name": image,
                        "force_build": False,
                        "build": image_build_specification(
                            built, memory=4096, sandbox_version="latest"
                        ),
                    },
                },
            )
        )
        assert await BlaxelApi().image_built(image)
        observed = await BlaxelApi().get(handle.resource.name)
        assert observed.labels["team"] == "loops-live"  # [providers.blaxel] labels
        assert observed.labels["loops.resource_id"] == handle.intent.resource_id
        sandbox = await usage.attach(handle.resource, timeout_seconds=60)
        try:
            started = await sandbox.process.exec(
                {"command": "cat /loops-marker", "working_dir": "/", "env": {}}
            )
            record = await sandbox.process.wait(
                started.pid or started.name, max_wait=60_000, interval=1_000
            )
            assert record.exit_code == 0 and record.stdout.strip() == marker
        finally:
            await incidents.close_clients(sandbox)
    finally:
        report = await service.close()  # released even when an assertion failed
        await registry.close()
        # The cookbook never deletes an image; this one is the test's own.
        deleted = await client.get_async_httpx_client().delete(f"/images/sandbox/{image}")
    assert report.complete, report
    assert deleted.status_code == 200, (image, deleted.status_code, deleted.text)
