"""The custom in-process direct pattern: ``AnswerEpisode``, and on Blaxel ``SandboxAnswerEpisode``.

The episode asks the model once with ``llm()`` and scores the answer against the
task's. On Blaxel the answer goes into the attempt's sandbox and is graded from it.
Its hooks are locked beside its twins' in ``test_custom_in_process_twins.py``; here
it trains, on no backend and on the faked Blaxel.
"""

import json
from pathlib import Path

from direct_episode import ANSWER
from support.builders import task_file
from support.offline_run import offline_config, train_offline

from loops_cookbook.rl import cli

PATTERN = Path(__file__).resolve().parents[2] / "patterns" / "custom_runtime" / "in_process_direct"

TASKS = [
    {"name": "b", "instruction": "Say B", "answer": "B"},
    {"name": "c", "instruction": "Say C", "answer": "C"},
]


async def train(config_path, tmp_path, **run_options):
    overrides = {
        "runtime.files": {"tasks": str(task_file(tmp_path, TASKS))},
        "training.max_steps": 2,
    }
    return await train_offline(offline_config(config_path, tmp_path, overrides), **run_options)


async def test_the_episode_trains_through_the_shared_pipeline(tmp_path):
    outcome, trainer, _ = await train(PATTERN / "train.toml", tmp_path)
    assert outcome.ok and outcome.steps == 2
    assert all(len(step["rewards"]) == 16 for step in trainer.steps)
    assert {reward for step in trainer.steps for reward in step["rewards"]} <= {0.0, 1.0}


async def test_on_blaxel_each_attempt_answers_into_its_own_sandbox_which_the_run_releases(
    tmp_path, blaxel
):
    from loops_cookbook.providers.blaxel.settings import BlaxelSettings

    configured = cli.load_training_config(PATTERN / "train_blaxel.toml").providers["blaxel"]
    outcome, _, run = await train(
        PATTERN / "train_blaxel.toml",
        tmp_path,
        adapters={"blaxel": blaxel.adapter(BlaxelSettings.model_validate(configured))},
    )
    assert outcome.ok and outcome.steps == 2
    records = [json.loads(line) for line in (run.out / "rollouts.jsonl").read_text().splitlines()]
    creates = [name for operation, name in blaxel.api.calls if operation == "create"]
    assert len(set(creates)) == len(records) == 32
    assert all(files == {ANSWER: files[ANSWER]} for files in blaxel.files.values())
    assert set(blaxel.files) == set(creates)  # each attempt answered into its own sandbox
    # train_blaxel.toml's flat [runtime.config.sandbox], as Blaxel's create configuration
    asked = {(c["image"], c["memory"], c["ttl"]) for c in blaxel.api.created}
    assert asked == {("blaxel/base-image:latest", 2048, "1h")}
    assert all(client.closed for client in blaxel.clients)  # the layer closed every client
    assert blaxel.api.sandboxes == {}  # every sandbox released
