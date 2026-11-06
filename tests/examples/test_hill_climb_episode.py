"""The host-only hill-climb example through real sampling and grading admission."""

from pathlib import Path

import pytest
from hill_climb_episode import HillClimbEpisode, HillClimbTask
from support.harness import Renderer
from support.offline_run import offline_config, train_offline

from loops_cookbook.llm.rendering import ParsedReply


@pytest.mark.parametrize(
    ("reply", "reward"),
    [("A", 0), ("B", 1), ("C", 0), ("D", 0), (" B\n", 1), ("b", 0), ("B.", 0), (None, 0)],
)
async def test_letter_reward_is_host_only_and_does_not_reveal_the_target(
    make, monkeypatch, reply, reward
):
    prompts = []
    render = Renderer.render

    def record_prompt(self, messages, tools):
        prompts.append(messages)
        return render(self, messages, tools)

    def parse(self, prompt_ids, completion_ids, tools):
        return ParsedReply(reply, None, (), None, {"role": "assistant", "content": reply or ""})

    monkeypatch.setattr(Renderer, "render", record_prompt)
    monkeypatch.setattr(Renderer, "parse", parse)
    built = await make(episode=HillClimbEpisode, sandbox=False, tasks=({"id": "hill-climb"},))
    result = await built.rig.rollout("g-r1", built.task)
    await built.rig.driver.drain()
    assert result.reward == reward
    assert result.metadata == {"guess": (reply or "").strip()}
    assert prompts == [[{"role": "user", "content": HillClimbTask().prompt}]]
    assert built.provider.calls == []


async def test_shipped_hill_climb_config_trains_without_providers(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = offline_config(
        root / "examples/episode_basics/train_hill_climb.toml",
        tmp_path,
        {"training.max_steps": 1},
    )
    assert not config.providers
    outcome, trainer, _ = await train_offline(config)
    assert outcome.ok and outcome.steps == 1
    assert len(trainer.steps[0]["rewards"]) == config.training.group_size
    assert set(trainer.steps[0]["rewards"]) == {0.0, 1.0}
