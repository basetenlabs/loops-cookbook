"""What every cookbook Harbor environment owes the registry, whichever provider backs it.

Harbor constructs the environment and starts it; the attempt's registry owns its
sandbox. Each test runs against every provider's environment over that provider's
fake control plane.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
from support.fakes import no_wait
from support.resources import episode, local

from loops_cookbook.harbor import attempts
from loops_cookbook.harbor.attempts import SandboxNotReleased
from loops_cookbook.resources.registry import ResourceRegistry


def _blaxel(tmp_path, monkeypatch):
    hb = pytest.importorskip("harbor.environments.blaxel")
    from support.blaxel_api import CREDENTIALS, SETTINGS, StatusError

    from loops_cookbook.providers.blaxel.adapter import BlaxelAdapter

    from .blaxel_sandboxes import blaxel_api, environments

    monkeypatch.setattr(hb, "_SANDBOX_READY_INTERVAL_SEC", 0)
    api = blaxel_api(monkeypatch, [])
    adapter = BlaxelAdapter(SETTINGS, CREDENTIALS, api, sleep=no_wait)
    return NS(
        api=api,
        adapter=adapter,
        settings={},
        status_error=StatusError,
        environment=environments(tmp_path),
        source=tmp_path / "env-dockerfile",
    )


def _daytona(tmp_path, monkeypatch):
    pytest.importorskip("harbor.environments.daytona.environment")
    from support.daytona_api import CREDENTIALS, SETTINGS, StatusError

    from loops_cookbook.providers.daytona.adapter import DaytonaAdapter

    from .daytona_sandboxes import daytona_api, environments

    shell = NS(files={}, answers={}, failing=set(), uploads={})
    api = daytona_api([], shell)
    adapter = DaytonaAdapter(SETTINGS, CREDENTIALS, api, sleep=no_wait)
    return NS(
        api=api,
        adapter=adapter,
        settings={"daytona": adapter.settings},  # the table Daytona's environment reads
        status_error=StatusError,
        environment=environments(tmp_path),
        source=tmp_path / "task-trial__env" / "environment",
    )


@pytest.fixture(params=["blaxel", "daytona"])
async def provider(request, tmp_path, monkeypatch):
    """The provider's fake control plane and adapter, its run's registry, and
    ``environment(**options)`` building its Harbor environment for attempt-1, bound."""
    name = request.param
    rig = {"blaxel": _blaxel, "daytona": _daytona}[name](tmp_path, monkeypatch)
    rig.name = name
    rig.registry = ResourceRegistry.open(
        tmp_path / "state", {name: rig.adapter}, mode="write", run_id="run-1", kind=local
    )
    rig.resources = episode(rig.registry, "attempt-1")
    attempts.bind("attempt-1", rig.resources, {name: rig.adapter.deadlines}, rig.settings)
    yield rig
    attempts.forget("attempt-1")
    await rig.registry.close()


@pytest.mark.parametrize(
    ("attempt", "error"),
    [("", TypeError), ("unknown-attempt", RuntimeError), (7, TypeError)],
    ids=["empty", "unknown", "not a string"],
)
def test_construction_fails_without_a_live_attempt(provider, attempt, error):
    with pytest.raises(error):
        provider.environment(attempt=attempt)


def test_construction_needs_the_runs_provider_deadlines(provider):
    attempts.bind("no-provider", episode(provider.registry, "no-provider"), {})
    try:
        with pytest.raises(RuntimeError, match=rf"\[providers.{provider.name}\]"):
            provider.environment(attempt="no-provider")
    finally:
        attempts.forget("no-provider")


def test_construction_provisions_nothing(provider):
    provider.environment()  # as Harbor does for a RegradeTrial's agent environment
    assert provider.api.calls == []


def test_compose_mode_is_refused_at_construction(provider):
    provider.source.mkdir(parents=True, exist_ok=True)
    (provider.source / "docker-compose.yaml").write_text("services: {}\n")
    with pytest.raises(ValueError, match="compose"):
        provider.environment()


async def test_a_kept_sandbox_is_released_with_its_episode_not_early(provider):
    env = provider.environment()
    await env.start(force_build=False)
    await env.stop(delete=False)
    assert len(provider.api.sandboxes) == 1
    assert (await provider.resources.close()).complete and provider.api.sandboxes == {}


async def test_a_release_that_fails_is_raised_from_stop_not_swallowed(provider):
    env = provider.environment()
    await env.start(force_build=False)
    provider.api.delete_errors = [provider.status_error(500, "boom")] * 50
    with pytest.raises(SandboxNotReleased):  # Harbor's own stop would log and carry on
        await env.stop(delete=True)
