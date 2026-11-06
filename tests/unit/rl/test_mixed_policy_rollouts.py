import pytest
from support.builders import make_rollout, make_sample

from loops_cookbook.rl.run import mixed_policy_rollouts


@pytest.mark.parametrize(
    ("versions", "mixed"),
    [((3,), 0), ((3, 3), 0), ((2, 3), 1), ((3, None, 3), 0), ((2, None, 3), 1)],
    ids=["one turn", "one policy", "two policies", "untrained turn", "untrained between"],
)
def test_a_rollout_is_mixed_when_its_trained_turns_span_policy_versions(versions, mixed):
    """Only trained generations count: an untrained one is just its prompt."""
    samples, prompt = [], (1,)
    for i, version in enumerate(versions):
        samples.append(
            make_sample(
                f"s{i}",
                prompt,
                (2 + i,),
                (-0.5,),
                version if version is not None else 9,
                trainable=version is not None,
            )
        )
        prompt = (*prompt, 2 + i, 50)
    rollout = make_rollout("t", "g", "r", tuple(samples))
    assert mixed_policy_rollouts([rollout]) == mixed
