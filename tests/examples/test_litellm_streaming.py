"""The LiteLLM example: a real LiteLLM client, streaming, against the run's real gateway.

LiteLLM comes with the Harbor extras; these tests run where it is installed.
"""

from pathlib import Path

import pytest

litellm = pytest.importorskip("litellm")

from support.builders import task_file
from support.offline_run import offline_config, train_offline

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "litellm_streaming"

TASKS = [
    {"name": "b", "instruction": "Say B", "answer": "B"},
    {"name": "c", "instruction": "Say C", "answer": "C"},
]


async def test_a_streaming_litellm_agent_trains_through_the_gateway(tmp_path, monkeypatch):
    import litellm_episode

    calls, chunks = [], []
    real = litellm.acompletion

    async def acompletion(**kwargs):
        calls.append(kwargs)
        stream = await real(**kwargs)

        async def counted():
            async for chunk in stream:
                chunks.append(chunk)
                yield chunk

        return counted()

    monkeypatch.setattr(litellm_episode.litellm, "acompletion", acompletion)
    config = offline_config(
        EXAMPLE / "train.toml",
        tmp_path,
        {
            "rollout_runtime.files": {"tasks": str(task_file(tmp_path, TASKS))},
            "training.max_steps": 1,
        },
    )
    outcome, trainer, _ = await train_offline(config)
    assert outcome.ok and outcome.steps == 1
    assert len(trainer.steps[0]["rewards"]) == 16
    assert calls and all(call["stream"] is True for call in calls)
    assert all(call["model"].startswith("openai/") for call in calls)
    assert len(chunks) > len(calls)  # each reply arrived as a stream of chunks
