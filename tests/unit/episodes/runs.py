"""An episode class's run and its training config, for the tests that run or check one.

``episode`` is the class's path, ``module:Class``, as ``[rollout_runtime.config] episode``
names it.
"""

from __future__ import annotations

from pathlib import Path

from support.builders import task_file
from support.configs import Tables, episode_config, write_toml

TASKS = [
    {"id": "b", "instruction": "Say B", "answer": "B"},
    {"id": "c", "instruction": "Say C", "answer": "C"},
]
FACTORY = "loops_cookbook.episodes:create_runtime"


def settings(config, tmp_path, episode, *, options=None, tasks=TASKS, **overrides):
    """A one-step run of ``episode`` over ``tasks``, offline; ``overrides`` win."""
    return config(
        **{
            "rollout_runtime.factory": FACTORY,
            "rollout_runtime.config": {
                "episode": episode,
                "agent_timeout_seconds": 20,
                **({} if options is None else {"options": options}),
            },
            "rollout_runtime.files": {"tasks": str(task_file(tmp_path, tasks))},
            "compute.sampler_replica_affinity": True,
            "max_steps": 1,
            **overrides,
        }
    )


def write_config(tmp_path: Path, episode: str, *tables: Tables, tasks: list[dict] = TASKS) -> Path:
    """``episode``'s training config as ``loops check`` reads it, ``tables`` merged in."""
    return write_toml(tmp_path / "train.toml", episode_config(tmp_path, episode, tasks), *tables)
