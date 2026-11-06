"""Fixtures for the resource tests."""

import pytest
from support.resources import PROVIDERS, FakeProvider


@pytest.fixture
def registered():
    """``registered(instance, provider=None)``: a provider ``support.resources:create_fake``
    builds for that ``instance``, removed again after the test."""
    names = []

    def register(instance, provider=None):
        PROVIDERS[instance] = provider or FakeProvider()
        names.append(instance)
        return PROVIDERS[instance]

    yield register
    for name in names:
        PROVIDERS.pop(name, None)
