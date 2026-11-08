"""The async-scheduling example: the custom in-process direct pattern, scheduled asynchronously.

Async is a setting of the run, not of the episode: ``tests/test_config_lineage.py``
locks each config as the pattern's with only the schedule changed, and the episode as
the pattern's own. Here the smoke trains, offline, within its staleness bound.
"""

from pathlib import Path

from support.builders import task_file
from support.offline_run import offline_config, train_offline

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "async_scheduling"


async def test_the_smoke_trains_offline_within_two_steps_of_the_policy(tmp_path):
    tasks = [{"name": "b", "instruction": "Say B", "answer": "B"}]
    config = offline_config(
        EXAMPLE / "train.toml",
        tmp_path,
        {"rollout_runtime.files": {"tasks": str(task_file(tmp_path, tasks))}},
    )
    outcome, trainer, _ = await train_offline(config)
    assert outcome.ok and outcome.steps == config.training.max_steps
    for step in trainer.steps:
        assert len(step["rewards"]) == 16
        assert all(
            step["policy_version_before"] - version <= config.training.steps_off_policy
            for version in step["sampled_versions"]
        )
