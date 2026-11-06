"""Each shipped config derived from another differs from it only where its story says.

A pattern's variants and the examples built on a pattern are copies a reader compares
by eye: the same experiment with one thing changed. Each row names the base config,
the derived one, and every setting the derived one changes, with its value; any other
difference fails. The run's name and output directory always differ, so they are not
compared.
"""

from pathlib import Path
from typing import Any

import pytest

from loops_cookbook.rl.cli import load_training_config

ROOT = Path(__file__).resolve().parents[1]
ABSENT = "<absent>"
UNCOMPARED = {"output.run_name", "output.output_dir"}

HARBOR_DIRECT = "patterns/harbor/in_process_direct"
CUSTOM_DIRECT = "patterns/custom_runtime/in_process_direct"
CUSTOM_HTTP = "patterns/custom_runtime/in_process_openai_http"
BLAXEL = {"region": "us-was-1"}
SANDBOX = {
    "provider": "blaxel",
    "image": "blaxel/base-image:latest",
    "memory_mb": 2048,
    "ttl": "1h",
}
# What scheduling asynchronously changes, and nothing else may.
ASYNC = {
    "training.mode": "async",
    "training.max_steps_off_policy": 2,
    "training.max_policy_delta_in_trajectory": 2,
}


def differences(base: Any, derived: Any, path: str = "") -> dict[str, Any]:
    """``{dotted path: derived value}`` for every setting ``derived`` changes from ``base``;
    a table only one of them has is one entry, ``ABSENT`` on the side that lacks it."""
    if not (isinstance(base, dict) and isinstance(derived, dict)):
        return {} if base == derived else {path: derived}
    changed: dict[str, Any] = {}
    for key in base.keys() | derived.keys():
        at = f"{path}.{key}" if path else key
        if at not in UNCOMPARED:
            changed |= differences(base.get(key, ABSENT), derived.get(key, ABSENT), at)
    return changed


def job(path: str) -> str:
    return str(ROOT / path)


LINEAGE = [
    pytest.param(
        f"{HARBOR_DIRECT}/train.toml",
        f"{HARBOR_DIRECT}/train_blaxel.toml",
        {
            "providers.blaxel": BLAXEL,
            # A cloud sandbox can fail: its rollout gets fresh attempts and its group still trains.
            "rollout.max_rollout_retries": 2,
            "training.max_group_losses": 1,
            "runtime.files.job": job(f"{HARBOR_DIRECT}/job_blaxel.yaml"),
        },
        id="harbor_direct_on_blaxel",
    ),
    *(
        pytest.param(
            f"{HARBOR_DIRECT}/train{suffix}.toml",
            f"patterns/harbor/in_process_openai_http/train{suffix}.toml",
            {"runtime.files.job": job(f"patterns/harbor/in_process_openai_http/job{suffix}.yaml")},
            id=f"harbor_http{suffix}",
        )
        for suffix in ("", "_blaxel")
    ),
    pytest.param(
        f"{CUSTOM_DIRECT}/train.toml",
        f"{CUSTOM_DIRECT}/train_blaxel.toml",
        {
            "providers.blaxel": BLAXEL,
            "runtime.config.episode": "direct_episode:SandboxAnswerEpisode",
            "runtime.config.sandbox": SANDBOX,
            "rollout.grading.timeout_seconds": 30.0,
            "rollout.max_concurrent_rollouts": 8,
            "rollout.max_concurrent_verifiers": 8,
            "rollout.max_rollout_retries": 2,
        },
        id="custom_direct_on_blaxel",
    ),
    pytest.param(
        f"{CUSTOM_DIRECT}/train.toml",
        f"{CUSTOM_HTTP}/train.toml",
        {"runtime.config.episode": "http_episode:AnswerEpisode"},
        id="custom_http",
    ),
    pytest.param(
        f"{CUSTOM_DIRECT}/train_blaxel.toml",
        f"{CUSTOM_HTTP}/train_blaxel.toml",
        {"runtime.config.episode": "http_episode:SandboxAnswerEpisode"},
        id="custom_http_blaxel",
    ),
    pytest.param(
        "patterns/custom_runtime/local/train.toml",
        "patterns/custom_runtime/local/train_blaxel.toml",
        {
            "providers.local_sandbox": ABSENT,
            "providers.blaxel": BLAXEL,
            "runtime.config.episode": "local_episode:BlaxelAgentEpisode",
            "runtime.config.options.grade_against": "sandbox",
            # No sandbox name: the run names each sandbox.
            "runtime.config.sandbox.provider": "blaxel",
            "runtime.config.sandbox.image": "blaxel/base-image:latest",
            "runtime.config.sandbox.memory_mb": 2048,
            "runtime.config.sandbox.ttl": "1h",
        },
        id="custom_local_on_blaxel",
    ),
    pytest.param(
        f"{HARBOR_DIRECT}/train_blaxel.toml",
        "examples/critic/train_blaxel.toml",
        {
            "rollout.producers.critic": {"trainable": True},
            "runtime.files.job": job("examples/critic/job_blaxel.yaml"),
        },
        id="critic",
    ),
    pytest.param(
        f"{HARBOR_DIRECT}/train_blaxel.toml",
        "examples/multi_step/train_blaxel.toml",
        {"runtime.files.job": job("examples/multi_step/job_blaxel.yaml")},
        id="multi_step",
    ),
    pytest.param(
        f"{CUSTOM_DIRECT}/train.toml",
        "examples/async_scheduling/train.toml",
        {
            **ASYNC,
            "runtime.config.episode": "answer_episode:AnswerEpisode",
            # The smoke is short: every step is checkpointed.
            "output.trainer_checkpoint_every": 1,
            "output.sampler_checkpoint_every": 1,
        },
        id="async_scheduling",
    ),
    pytest.param(
        f"{CUSTOM_DIRECT}/train_blaxel.toml",
        "examples/async_scheduling/train_blaxel.toml",
        {**ASYNC, "runtime.config.episode": "answer_episode:SandboxAnswerEpisode"},
        id="async_scheduling_blaxel",
    ),
    pytest.param(
        f"{CUSTOM_HTTP}/train.toml",
        "examples/litellm_streaming/train.toml",
        {"runtime.config.episode": "litellm_episode:AnswerEpisode"},
        id="litellm_streaming",
    ),
    pytest.param(
        "examples/llm_judge/train.toml",
        "examples/llm_judge/train_blaxel.toml",
        {
            "providers.blaxel": BLAXEL,
            "runtime.config.episode": "judge_episode:SandboxedRankedEpisode",
            "runtime.config.sandbox": SANDBOX,
            "rollout.max_concurrent_rollouts": 8,
            "rollout.max_rollout_retries": 2,
            "rollout.setup_timeout_seconds": 90.0,
        },
        id="llm_judge_on_blaxel",
    ),
    pytest.param(
        "examples/terminal_bench/train.toml",
        "examples/terminal_bench/train_echo.toml",
        {
            "training.loss_fn": "custom",
            "training.custom_loss": {
                "factory": "echo_loss:create_loss",
                "config": {"observation_weight": 0.05},
            },
        },
        id="terminal_bench_echo",
    ),
]


@pytest.mark.parametrize(("base", "derived", "changes"), LINEAGE)
def test_the_config_differs_from_its_base_only_where_stated(monkeypatch, base, derived, changes):
    monkeypatch.setenv("BASETEN_API_KEY", "check-only")  # the LLM judge's key variable
    loaded = [load_training_config(ROOT / path).model_dump() for path in (base, derived)]
    assert differences(*loaded) == changes


@pytest.mark.parametrize(
    ("pattern", "copy", "renames"),
    [
        (f"{CUSTOM_DIRECT}/direct_episode.py", "examples/async_scheduling/answer_episode.py", {}),
        (
            f"{HARBOR_DIRECT}/direct_agent.py",
            "examples/multi_step/step_agent.py",
            {
                "StepOptions": "DirectOptions",
                "StepAgent": "DirectAgent",
                '"hill-climb-steps"': '"hill-climb-direct"',
            },
        ),
    ],
    ids=["async_scheduling_episode", "multi_step_agent"],
)
def test_an_examples_copied_code_is_the_patterns_own(pattern, copy, renames):
    """An example directory is copyable on its own, so it carries the code it runs:
    the pattern's, below the docstring, under its own names."""
    copied = (ROOT / copy).read_text().split('"""', 2)[2]
    for name, original in renames.items():
        copied = copied.replace(name, original)
    assert copied == (ROOT / pattern).read_text().split('"""', 2)[2]
