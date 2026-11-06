"""Building an EpisodeRuntime: its settings, and the episode class it names."""

from __future__ import annotations

import pytest

from loops_cookbook.runtime.context import RuntimeConfigurationError

from .declared import (
    ABSENT,
    DECLARED,
    build,
)


@pytest.mark.parametrize(
    ("config", "problem"),
    [
        ({"episode": f"{DECLARED}:NoGrade"}, r"does not define grade\(\)"),
        ({"episode": f"{DECLARED}:NoRun"}, r"does not define run\(\)"),
        ({"episode": f"{DECLARED}:SyncSetup"}, r"setup\(\) is not an async def"),
        ({"episode": f"{DECLARED}:SyncRun"}, r"run\(\) is not an async def"),
        ({"episode": f"{DECLARED}:SyncGrade"}, r"grade\(\) is not an async def"),
        ({"episode": f"{DECLARED}:SyncClose"}, r"close\(\) is not an async def"),
        ({"episode": f"{DECLARED}:InstanceGrader"}, r"grade_group\(\) as an async classmethod"),
        ({"episode": f"{DECLARED}:NotAnEpisode"}, "not a subclass of Episode"),
        ({"episode": f"{DECLARED}:Missing"}, "has no episode 'Missing'"),
        ({"episode": "no_such_module:Answer"}, "could not import episode module"),
        ({"episode": "just-a-name"}, "module:attribute"),
        ({"agent_timeout_seconds": 0}, "agent_timeout_seconds"),
        ({"agent_timeout_seconds": None}, "agent_timeout_seconds"),
        ({"agent_timeout_seconds": ABSENT}, "agent_timeout_seconds"),
        ({"agent_timout_seconds": 5}, "agent_timout_seconds"),  # a typo is refused, not ignored
        ({"sandbox": {"provider": "blaxel", "ttl": "1h"}}, "'blaxel' is not configured"),
        ({"sandbox": {"image": "x"}}, "sandbox.provider"),
    ],
)
def test_a_misconfigured_episode_runtime_is_refused_when_it_is_built(tmp_path, config, problem):
    with pytest.raises(RuntimeConfigurationError, match=problem):
        build(tmp_path, **config)


def test_an_unset_rollout_clock_is_the_agent_deadline_plus_every_wait_after_it(tmp_path):
    runtime = build(
        tmp_path,
        agent_timeout_seconds=10,
        limits={"rollout": None, "grace": 5.0, "kill_grace": 4.0},
    )
    # agent, kill grace, the wait for a killed process group, cancellation grace
    assert runtime.capabilities.rollout_timeout_seconds == 10 + 4 + 5 + 5


@pytest.mark.parametrize(
    ("agent", "ok"),
    [(46, True), (46.1, False), (50.9, False), (70, False)],
    ids=["at the minimum", "just below", "the minimum without the group exit", "far below"],
)
def test_a_rollout_clock_set_by_hand_may_not_be_below_the_derived_one(tmp_path, agent, ok):
    limits = {"rollout": 60.0, "grace": 5.0, "kill_grace": 4.0}
    if ok:
        runtime = build(tmp_path, agent_timeout_seconds=agent, limits=limits)
        assert runtime.capabilities.rollout_timeout_seconds == 60.0  # the one set is kept
        return
    with pytest.raises(
        RuntimeConfigurationError, match=r"\[rollout\] rollout_timeout_seconds \(60s\) is below"
    ):
        build(tmp_path, agent_timeout_seconds=agent, limits=limits)


@pytest.mark.parametrize(
    ("episode", "problem"),
    [
        ("StringAccess", "frozenset of ModelAccessMode"),
        ("NotACount", "grading_retries must be a nonnegative integer"),
    ],
)
def test_what_the_class_declares_is_checked_as_the_runtimes_capabilities(
    tmp_path, episode, problem
):
    from loops_cookbook.runtime.context import Preflight, check_capabilities

    runtime = build(tmp_path, episode=f"{DECLARED}:{episode}")
    with pytest.raises(RuntimeConfigurationError, match=problem):
        check_capabilities(
            runtime.capabilities,
            Preflight(
                model_access=None,
                rollout_timeout_seconds=None,
            ),
        )


@pytest.mark.parametrize(
    ("episode", "problem"),
    [
        ("NoPhaseSetup", "self and one phase argument"),
        ("ExtraRunArgument", "self and one phase argument"),
        ("OldConstructor", "constructor must accept task and options"),
    ],
)
def test_check_refuses_incompatible_signatures_before_any_attempt(tmp_path, episode, problem):
    with pytest.raises(RuntimeConfigurationError, match=problem):
        build(tmp_path, episode=f"{DECLARED}:{episode}")
