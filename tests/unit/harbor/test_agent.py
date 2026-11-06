"""The convenience base supplies a client, not an agent framework."""

import inspect
from unittest.mock import Mock

import pytest
from direct_agent import DirectAgent

from loops_cookbook.harbor.agent import LoopsAgent
from loops_cookbook.llm.client import Renderers
from loops_cookbook.model_access.api import CaptureIntegrityFailure, ModelAccess
from loops_cookbook.model_access.lease import DirectModelAccess
from loops_cookbook.runtime import AttemptServices
from unit.model_access.test_lease import lease


def test_base_retains_harbor_abstract_lifecycle():
    assert inspect.isabstract(LoopsAgent)
    assert {"name", "version", "setup", "run"} <= LoopsAgent.__abstractmethods__


def services():
    access = Mock(spec=ModelAccess)
    access.llm.side_effect = lambda producer=None: (access, producer)
    return AttemptServices(
        model_access=access, capacity=Mock(), resources=Mock(), processes=Mock(), metrics=Mock()
    )


async def test_binding_is_inherited_and_isolated_per_agent(tmp_path):
    first = DirectAgent(logs_dir=tmp_path / "first")
    second = DirectAgent(logs_dir=tmp_path / "second")
    with pytest.raises(RuntimeError, match="not bound"):
        first.llm()
    with pytest.raises(RuntimeError, match="not bound"):
        first.conversation()
    with pytest.raises(RuntimeError, match="not bound"):
        await first.http_connection()
    bound = services()
    first.bind_loops_services(bound)
    assert first.llm() == (bound.model_access, None)
    assert first.llm("critic") == (bound.model_access, "critic")
    with pytest.raises(RuntimeError, match="not bound"):
        second.llm()
    other = services()
    second.bind_loops_services(other)
    assert second.llm()[0] is other.model_access
    assert first.llm()[0] is bound.model_access


async def test_conversation_keeps_producer_identity_and_refuses_after_sealing(
    tmp_path, prime_renderer
):
    attempt, captured, _ = lease()
    agent = DirectAgent(logs_dir=tmp_path)
    agent.bind_loops_services(
        AttemptServices(
            model_access=DirectModelAccess(attempt, Renderers(prime_renderer)),
            capacity=Mock(),
            resources=Mock(),
            processes=Mock(),
            metrics=Mock(),
        )
    )
    conversation = agent.conversation(producer="critic", conversation_id="review")
    await conversation.generate(messages=[{"role": "user", "content": "Check this"}])
    sample = captured[0][1]
    assert (sample.producer, sample.conversation_id, sample.trainable) == (
        "critic",
        "review",
        False,
    )
    await attempt.seal()
    with pytest.raises(CaptureIntegrityFailure, match="sealed"):
        await conversation.generate(messages=[{"role": "user", "content": "Check again"}])
