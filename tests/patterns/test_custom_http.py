"""The custom in-process openai_http pattern: an Episode whose agent is a stock OpenAI client.

``http_episode.py`` runs through ``loops_cookbook.episodes:create_runtime`` on no
backend or on Blaxel. Its hooks are locked beside the direct pattern's in
``test_custom_in_process_twins.py``; here it trains through the run's real gateway.
"""

from pathlib import Path

from support.builders import task_file
from support.offline_run import offline_config, train_offline

PATTERN = (
    Path(__file__).resolve().parents[2] / "patterns" / "custom_runtime" / "in_process_openai_http"
)
TASKS = [
    {"name": "b", "instruction": "Say B", "answer": "B"},
    {"name": "c", "instruction": "Say C", "answer": "C"},
]


async def test_it_trains_through_the_shared_pipeline_and_the_runs_gateway(tmp_path):
    config = offline_config(
        PATTERN / "train.toml",
        tmp_path,
        {"runtime.files": {"tasks": str(task_file(tmp_path, TASKS))}, "training.max_steps": 2},
    )
    outcome, trainer, _ = await train_offline(config)
    assert outcome.ok and outcome.steps == 2
    assert all(len(step["rewards"]) == 16 for step in trainer.steps)
