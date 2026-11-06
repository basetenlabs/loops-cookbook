"""How a Harbor trial's result becomes a reward and episode metadata."""

from types import SimpleNamespace as NS

import pytest

from loops_cookbook.harbor.runtime import (
    HarborRolloutError,
    _agent_metadata,
    _check_trial,
    reward_of,
)


@pytest.mark.parametrize("value", [True, "1", float("nan")])
def test_invalid_reward_is_not_a_training_score(value):
    with pytest.raises(ValueError):
        reward_of(NS(verifier_result=NS(rewards={"reward": value})))


def test_multiple_rewards_require_explicit_training_key():
    result = NS(verifier_result=NS(rewards={"a": 1, "b": 0}))
    with pytest.raises(ValueError, match="reward_key"):
        reward_of(result)
    assert reward_of(result, key="b") == 0.0


def _entry(exception=None, rewards=None):
    info = NS(exception_type=exception, exception_message="boom") if exception else None
    verifier = NS(rewards=rewards) if rewards is not None else None
    return NS(exception_info=info, verifier_result=verifier, step_results=None)


def test_a_trial_trains_only_if_every_step_that_raised_still_has_a_reward():
    _check_trial(_entry("AgentTimeoutError", {"reward": 0.0}))
    with pytest.raises(HarborRolloutError, match="RuntimeError"):
        _check_trial(_entry("RuntimeError"))
    trial = _entry(rewards={"reward": 1.0})
    trial.step_results = [_entry(rewards={"reward": 1.0}), _entry("NonZeroAgentExitCodeError")]
    with pytest.raises(HarborRolloutError):
        _check_trial(trial)


def test_agent_metadata_comes_from_the_step_that_ran_last():
    first = NS(agent_result=NS(metadata={"step": 1}))
    last = NS(agent_result=NS(metadata={"step": 2}))
    assert _agent_metadata(NS(agent_result=None, step_results=[first, last])) == {"step": 2}
    assert _agent_metadata(NS(agent_result=NS(metadata={"a": (1, 2)}), step_results=None)) == {
        "a": [1, 2]
    }
    assert _agent_metadata(NS(agent_result=None, step_results=None)) == {}


def test_agent_metadata_that_json_cannot_carry_is_dropped_not_fatal():
    report = {"turns": 3, "latency": float("nan"), "at": object(), 7: "x"}
    assert _agent_metadata(NS(agent_result=NS(metadata=report), step_results=None)) == {"turns": 3}
