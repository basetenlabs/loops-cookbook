"""``train``: telemetry around an SFT run."""

import asyncio
import logging

import pytest

from loops_cookbook import telemetry
from loops_cookbook.cli import ConfigSource
from loops_cookbook.sft import recipe
from loops_cookbook.sft import run as run_module

from .conftest import ChatRenderer


class Interrupted:
    """Metrics whose start is interrupted, as by a Ctrl-C during `wandb.init`."""

    def __init__(self, config, group):
        pass

    async def start(self):
        raise asyncio.CancelledError

    async def close(self):
        pass


async def test_the_runs_log_names_where_its_config_came_from(
    sft_config, monkeypatch, tmp_path, caplog
):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(telemetry, "Metrics", Interrupted)
    monkeypatch.setattr(run_module, "build_renderer", lambda config: ChatRenderer())
    source = ConfigSource.of(tmp_path / "config.toml", (tmp_path / ".env",))
    with pytest.raises(asyncio.CancelledError):
        await recipe.train(sft_config(), source=source)
    (log,) = (tmp_path / "runs").glob("*/run.log")
    assert all(line in log.read_text() for line in source.describe())


async def test_metrics_interrupted_while_starting_are_still_closed(sft_config, monkeypatch):
    """`wandb.init` takes seconds; a Ctrl-C landing in it must not leave a remote run open."""
    closed = []

    class Metrics:
        def __init__(self, config, group):
            pass

        async def start(self):
            raise asyncio.CancelledError

        async def close(self):
            closed.append(True)

    monkeypatch.setattr(telemetry, "Metrics", Metrics)
    monkeypatch.setattr(run_module, "build_renderer", lambda config: ChatRenderer())
    with pytest.raises(asyncio.CancelledError):
        await recipe.train(sft_config())
    assert closed
