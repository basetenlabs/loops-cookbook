"""Training configs the tests share, as the tables a TOML file holds.

A test states only the tables it changes: ``write_toml(path, MINIMAL, {"model": ...})``
merges them into a base and writes the file. Paths are written as TOML strings, so a
Windows path or one with a quote stays one value.
"""

import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from support.builders import task_file

type Tables = dict[str, Any]

# The checkpoint cadence and retention every recipe must state.
CHECKPOINTS: Tables = {
    "trainer_checkpoint_every": 5,
    "sampler_checkpoint_every": 5,
    "checkpoint_ttl_seconds": 0,
}

# The smallest RL config that validates: its recipe, a runtime factory, a model, the
# batch and optimizer, and the deadlines and concurrency a run must have.
MINIMAL: Tables = {
    "recipe": "rl",
    "runtime": {"factory": "m:create"},
    "model": {"base_model": "m", "lora_rank": 32},
    "sampling": {"max_generation_tokens": 4096},
    "training": {
        "max_steps": 0,
        "learning_rate": 1e-5,
        "group_size": 8,
        "max_group_losses": 2,
        "groups_per_step": 8,
        "loss_fn": "cispo",
    },
    "compute": {"scope": "dedicated"},
    "output": {"run_name": "run", **CHECKPOINTS},
    "rollout": {
        "max_concurrent_rollouts": 64,
        "max_concurrent_verifiers": 16,
        "rollout_timeout_seconds": 60,
        "setup_timeout_seconds": 60,
        "grading": {"timeout_seconds": 60},
        "cleanup": {
            "timeout_seconds": 5,
            "cancellation_grace_seconds": 1,
            "process_kill_grace_seconds": 1,
        },
    },
}

# The smallest SFT config that validates.
SFT_MINIMAL: Tables = {
    "recipe": "sft",
    "model": {"base_model": "m", "max_seq_len": 64, "lora_rank": 32},
    "data": {"train": "train.jsonl"},
    "training": {
        "max_steps": 0,
        "learning_rate": 1e-4,
        "lr_schedule": "constant",
        "warmup_steps": 0,
        "batch_size": 32,
        "epochs": 1,
    },
    "eval": {"every_steps": 0},
    "compute": {"scope": "dedicated"},
    "output": {"run_name": "sft", **CHECKPOINTS},
}

# The staleness limits mode = "async" must state.
ASYNC: Tables = {
    "training": {
        "mode": "async",
        "max_steps_off_policy": 3,
        "max_policy_delta_in_trajectory": 1,
    }
}


def episode_config(tmp_path: Path, episode: str, tasks: list[dict]) -> Tables:
    """MINIMAL for the episode class ``episode`` (``module:Class``) over ``tasks``, with
    a 20 s agent deadline and its runs under ``tmp_path / "runs"``."""
    return merged(
        MINIMAL,
        {
            "runtime": {
                "factory": "loops_cookbook.episodes:create_runtime",
                "config": {"episode": episode, "agent_timeout_seconds": 20},
                "files": {"tasks": task_file(tmp_path, tasks)},
            },
            "rollout": {
                "rollout_timeout_seconds": 30,
                "setup_timeout_seconds": 30,
                "grading": {"timeout_seconds": 10},
                "cleanup": {"cancellation_grace_seconds": 2, "process_kill_grace_seconds": 2},
            },
            "output": {"output_dir": tmp_path / "runs"},
        },
    )


def merged(*tables: Tables) -> Tables:
    """``tables`` merged left to right: a table merges into the table it meets, any other
    value replaces it, and ``None`` removes the key."""
    out: Tables = {}
    for table in tables:
        for key, value in table.items():
            if value is None:
                out.pop(key, None)
            elif isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = merged(out[key], value)
            elif isinstance(value, dict):
                out[key] = merged(value)
            else:
                out[key] = str(value) if isinstance(value, Path) else value
    return out


def toml(*tables: Tables) -> str:
    """``tables`` merged, as TOML text."""
    return tomli_w.dumps(merged(*tables))


def read_toml(path: Path) -> Tables:
    """The tables of the TOML file at ``path``."""
    return tomllib.loads(path.read_text())


def write_toml(path: Path, *tables: Tables) -> Path:
    """``tables`` merged and written to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(toml(*tables))
    return path
