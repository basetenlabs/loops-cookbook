"""Builders for valid configurations and evidence, so tests state only what differs."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from baseten.loops.models import ModelInput

from loops_cookbook.evidence import AttemptIdentity, PromptChunks, Rollout, Sample, TaskRef

# The smallest configuration the model accepts: every mandatory setting, and the rollout
# clock a rollout runtime that derives none needs.
REQUIRED: dict[str, Any] = {
    "rollout_runtime.factory": "tests:unused",
    "model.base_model": "m",
    "model.lora_rank": 32,
    "sampling.max_generation_tokens": 4096,
    "training.max_steps": 0,
    "training.learning_rate": 1e-5,
    "training.group_size": 8,
    "training.max_group_losses": 2,
    "training.groups_per_step": 8,
    "training.loss_fn": "cispo",
    "rollout.max_concurrent_rollouts": 64,
    "rollout.max_concurrent_verifiers": 16,
    "compute.scope": "dedicated",
    "output.trainer_checkpoint_every": 5,
    "output.sampler_checkpoint_every": 5,
    "output.checkpoint_ttl_seconds": 0,
    "output.run_name": "run",
    "rollout.rollout_timeout_seconds": 60,
    "rollout.setup_timeout_seconds": 60,
    "rollout.grading.timeout_seconds": 60,
}

# The staleness limits mode = "async" requires, unless a test names its own.
ASYNC_REQUIRED: dict[str, Any] = {
    "training.max_steps_off_policy": 3,
    "training.max_policy_delta_in_trajectory": 1,
}


def _paths() -> dict[str, str | None]:
    """Leaf name to dotted path; ``None`` for a leaf that more than one section has."""
    from loops_cookbook.cli import settings
    from loops_cookbook.rl.config import Config

    paths: dict[str, str | None] = {}
    for path, _ in settings(Config):
        leaf = path.rsplit(".", 1)[-1]
        paths[leaf] = None if leaf in paths else path
    return paths


def config_values(**overrides: Any) -> dict[str, Any]:
    """Nested settings from ``REQUIRED`` plus ``overrides``.

    An override is named by its dotted path, or by its leaf name when that is
    unambiguous; ``rollout.grading.timeout_seconds`` needs the full path.
    """
    leaves = _paths()
    flat = dict(REQUIRED)
    for name, value in overrides.items():
        if "." not in name:
            if name not in leaves:
                raise KeyError(f"no setting named {name!r}")
            path = leaves[name]
            if path is None:
                raise KeyError(f"{name!r} is ambiguous; name it by its dotted path")
            name = path
        flat[name] = value
    if flat.get("training.mode") == "async":
        flat = {**ASYNC_REQUIRED, **flat}
    nested: dict[str, Any] = {}
    for path, value in flat.items():
        *sections, leaf = path.split(".")
        target = nested
        for section in sections:
            target = target.setdefault(section, {})
        target[leaf] = value
    return nested


def make_config(**overrides: Any):
    from loops_cookbook.rl.config import Config

    return Config.model_validate(config_values(**overrides))


def digest(text: str) -> str:
    return f"sha256:{sha256(text.encode()).hexdigest()}"


def task(task_id: str = "task") -> TaskRef:
    return TaskRef(task_id, digest(task_id))


def identity(
    rollout_id: str = "r0", group_id: str = "group", attempt_id: str | None = None
) -> AttemptIdentity:
    return AttemptIdentity("run", group_id, rollout_id, attempt_id or f"a-{rollout_id}")


def make_sample(
    sample_id: str,
    prompt_ids,
    completion_ids,
    logprobs,
    policy_version: int = 0,
    *,
    producer: str = "main",
    conversation_id: str = "main",
    trainable: bool = True,
    prompt_input: ModelInput | None = None,
    stop_reason: str | None = None,
    sequence_id=None,
) -> Sample:
    """A ``Sample`` with the text-only prompt chunks derived from ``prompt_ids``."""
    return Sample(
        sample_id=sample_id,
        producer=producer,
        conversation_id=conversation_id,
        trainable=trainable,
        prompt_ids=tuple(prompt_ids),
        completion_ids=tuple(completion_ids),
        logprobs=tuple(logprobs),
        policy_version=policy_version,
        prompt_chunks=PromptChunks.of(
            prompt_input or ModelInput.from_ints(list(prompt_ids)), tuple(prompt_ids)
        ),
        stop_reason=stop_reason,
        sequence_id=sequence_id,
    )


def make_rollout(
    task_id: str = "task",
    group_id: str = "group",
    rollout_id: str = "r0",
    samples=(),
    reward: float | None = None,
    *,
    metadata=None,
) -> Rollout:
    """Positional ``(task, group, rollout, samples, reward)``, as tests read best."""
    return Rollout(
        identity=identity(rollout_id, group_id),
        task=task(task_id),
        samples=tuple(samples),
        reward=reward,
        metadata=metadata or {},
    )


def services_for(llm: Any):
    """Attempt services whose model access hands out ``llm`` for every producer."""
    from types import SimpleNamespace

    from loops_cookbook.runtime import AttemptServices

    access = SimpleNamespace(mode="direct", llm=lambda producer=None: llm)
    return AttemptServices(
        model_access=access, capacity=None, resources=None, processes=None, metrics=None
    )  # pyright: ignore[reportArgumentType]


def write_jsonl(path, rows):
    """``rows`` written to ``path`` one JSON value per line, its directory made if need be."""
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def task_file(directory, tasks):
    """A custom rollout runtime's dataset: ``tasks`` written one JSON task per line."""
    from pathlib import Path

    return write_jsonl(Path(directory) / "tasks.jsonl", tasks)


def runtime_context(tasks_path, **fields):
    """A ``RuntimeContext`` stand-in whose ``files`` resolve ``tasks`` to ``tasks_path``."""
    from pathlib import Path
    from types import SimpleNamespace

    files = SimpleNamespace(path=lambda alias: Path(tasks_path) if alias == "tasks" else None)
    return SimpleNamespace(files=files, **fields)
