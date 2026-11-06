"""Fixtures for the model access tests."""

import pytest

from loops_cookbook.model_access.openai.gateway import Gateway, GatewaySettings

from .rig import MODEL, OPENED_CLIENTS


@pytest.fixture
async def gateway():
    served = Gateway(GatewaySettings("127.0.0.1:0", None, MODEL, shutdown_seconds=1.0))
    yield served
    await served.close()


@pytest.fixture(autouse=True)
async def _close_clients():
    yield
    for opened in OPENED_CLIENTS:
        await opened.close()
    OPENED_CLIENTS.clear()
