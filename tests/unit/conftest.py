"""Fixtures for the offline unit suite: in-memory fakes, no model files, no network."""

import pytest
from support.fakes import FakeClient, FakePrime
from support.rl_sft import serve_loops

from loops_cookbook.llm.prime import PrimeRenderer


@pytest.fixture
def fake_client():
    """Records forward/backward, optimizer and checkpoint calls like the SDK client."""
    return FakeClient()


@pytest.fixture
def loops_service(monkeypatch):
    """The Loops service a trainer's ``start`` provisions from (see ``serve_loops``)."""
    return serve_loops(monkeypatch)


@pytest.fixture
def fake_prime():
    """A Prime-shaped multimodal renderer emitting ``[PAD] * 3`` per image."""
    return FakePrime()


@pytest.fixture
def prime_renderer(fake_prime):
    return PrimeRenderer(fake_prime)
