"""Test support: verify a run directory's recorded training boundary.

Not shipped and not a CLI; the runner's tests use it to check that committed steps
match their recorded samples, datums, masks and freshness. This checks saved
evidence, not sampler/trainer numerical agreement or remote execution. Only steps
with committed step records enter the audit.
"""

from __future__ import annotations

import base64
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from baseten.loops.models import Datum, ModelInput

from loops_cookbook.evidence import AttemptIdentity, PromptChunks, Rollout, Sample, TaskRef
from loops_cookbook.rl.aggregation import aggregate_loss
from loops_cookbook.rl.config import Config
from loops_cookbook.rl.grpo import advantages, build_rows, normalize_batch_provenance


def _restore_model_input(payload: dict, sizes: dict[str, int]) -> ModelInput:
    """Rebuild a recorded ``ModelInput`` whose image bytes were replaced by digests.

    The digest hex becomes the sentinel image bytes, so re-redacting an expected datum
    yields the same digest and byte count that the batch record carries.
    """
    chunks = []
    for chunk in payload.get("chunks", []):
        if chunk.get("type") == "image" and isinstance(chunk.get("data"), dict):
            digest, size = chunk["data"]["sha256"], chunk["data"]["bytes"]
            if sizes.setdefault(digest, size) != size:
                raise ValueError(f"image {digest} recorded with two different sizes")
            chunk = {**chunk, "data": base64.b64encode(bytes.fromhex(digest)).decode()}
        chunks.append(chunk)
    return ModelInput.model_validate({**payload, "chunks": chunks})


def _redact_expected(payload: dict, sizes: dict[str, int]) -> dict:
    chunks = []
    for chunk in payload.get("chunks", []):
        if chunk.get("type") == "image" and isinstance(chunk.get("data"), str):
            digest = base64.b64decode(chunk["data"]).hex()
            chunk = {**chunk, "data": {"sha256": digest, "bytes": sizes[digest]}}
        chunks.append(chunk)
    return {**payload, "chunks": chunks}


@dataclass(frozen=True)
class AcceptanceReport:
    steps: int
    rollouts: int
    samples: int
    datums: int
    uncommitted_batches: int = 0


class AcceptanceError(ValueError):
    """Recorded boundaries do not supply consistent acceptance evidence."""


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"missing or invalid evidence: {path}") from error


def _lines(path: Path) -> list[dict]:
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        raise ValueError(f"missing or invalid evidence: {path}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"evidence rows must be objects: {path}")
    return rows


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_run(run_dir: Path) -> AcceptanceReport:
    """Recompile every committed batch and compare exact SDK rows with its trace."""
    run_dir = Path(run_dir)
    try:
        return _audit(run_dir)
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise AcceptanceError(f"incomplete or inconsistent acceptance evidence: {error}") from error


def _audit(run_dir: Path) -> AcceptanceReport:
    config = Config.model_validate(_read(run_dir / "config.json"))
    steps = _lines(run_dir / "steps.jsonl")
    _require(bool(steps), "no committed steps to audit")
    step_ids = [row["step"] for row in steps]
    _require(
        all(type(value) is int and value >= 0 for value in step_ids),
        "invalid committed step identity",
    )
    _require(len(step_ids) == len(set(step_ids)), "duplicate committed step")
    _require(
        step_ids == list(range(step_ids[0], step_ids[0] + len(step_ids))),
        "committed steps are not contiguous and ordered",
    )
    # Discarded rollouts are recorded too; the evidence audited here is what trained.
    records = [
        record
        for record in _lines(run_dir / "rollouts.jsonl")
        if record["disposition"] == "trained"
    ]
    records_by_step: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        records_by_step[record["step"]].append(record)
    _require(set(records_by_step) <= set(step_ids), "rollout records lack a committed step")

    captures: dict[str, tuple[Sample, dict]] = {}
    attempt_samples: dict[str, list[str]] = defaultdict(list)
    image_sizes: dict[str, int] = {}
    sample_fields = {field.name for field in fields(Sample)}
    # `prompt_input` is written only when it carries something `prompt_ids` cannot
    # reconstruct -- an image chunk. For a text-only capture its absence is the
    # format, not a gap, and the audit rebuilds it from the ids. `prompt_chunks` is
    # never written: it is read back from `prompt_input`.
    required_fields = sample_fields - {"prompt_chunks"}
    for path in sorted((run_dir / "samples").glob("*.jsonl")):
        for record in _lines(path):
            _require(
                all(record.get(key) for key in ("attempt_id", "rollout_id", "group_id")),
                "sample capture lacks attempt/rollout attribution",
            )
            raw = {key: value for key, value in record.items() if key in sample_fields}
            _require(required_fields <= raw.keys(), "sample capture omits sampling evidence fields")
            if record.get("prompt_input") is not None:
                prompt_input = _restore_model_input(record["prompt_input"], image_sizes)
            else:
                prompt_input = ModelInput.from_ints(list(raw["prompt_ids"]))
            raw["prompt_chunks"] = PromptChunks.of(prompt_input, raw["prompt_ids"])
            sample = Sample(**raw)
            _require(sample.sample_id not in captures, "duplicate captured sample identity")
            _require(
                sample.producer in config.rollout.producers
                and sample.trainable == config.rollout.producers[sample.producer].trainable,
                "sample trainability disagrees with configured producers",
            )
            captures[sample.sample_id] = sample, record
            attempt_samples[record["attempt_id"]].append(sample.sample_id)

    used_samples: set[str] = set()
    used_rollouts: set[str] = set()
    datum_count = 0
    committed_paths = set()
    previous_policy = None
    for step in steps:
        step_id = step["step"]
        path = run_dir / "batches" / f"step-{step_id}.json"
        committed_paths.add(path)
        batch = _read(path)
        _require(batch["step"] == step_id, "batch step identity mismatch")
        version = batch["policy_version"]
        _require(type(version) is int and version >= 0, "invalid batch policy version")
        if previous_policy is not None:
            _require(version == previous_policy, "policy discontinuity between committed steps")
        _require(step["train/policy_version"] > version, "committed policy did not advance")
        previous_policy = step["train/policy_version"]
        expected_records = records_by_step[step_id]
        _require(
            [row["rollout_id"] for row in expected_records] == batch["rollout_ids"],
            "batch and committed rollout records disagree",
        )
        batch_records = expected_records
        rollouts = []
        groups: dict[str, list[Rollout]] = defaultdict(list)
        for record in batch_records:
            identity = record["rollout_id"]
            _require(identity not in used_rollouts, "rollout trained more than once")
            used_rollouts.add(identity)
            selected = []
            attempts = set()
            for sample_id in record["sample_ids"]:
                _require(sample_id in captures, f"missing sample capture: {sample_id}")
                _require(sample_id not in used_samples, "sample used by multiple trained rollouts")
                used_samples.add(sample_id)
                sample, attribution = captures[sample_id]
                _require(
                    all(
                        attribution[key] == record[key]
                        for key in ("rollout_id", "group_id", "attempt_id")
                    ),
                    "sample attribution disagrees with rollout",
                )
                attempts.add(attribution["attempt_id"])
                selected.append(sample)
            # A member whose agent discarded every generation keeps none: its record
            # names the attempt.
            _require(
                attempts <= {record["attempt_id"]} and record["attempt_id"] in attempt_samples,
                "rollout must use exactly one complete attempt",
            )
            attempt = record["attempt_id"]
            captured_ids = attempt_samples[attempt]
            # Every generation the attempt captured is this rollout's, kept or not: one
            # that kept none must not borrow another rollout's attempt to date it.
            _require(
                all(
                    captures[sample_id][1][key] == record[key]
                    for sample_id in captured_ids
                    for key in ("rollout_id", "group_id")
                ),
                "attempt attribution disagrees with rollout",
            )
            kept = set(record["sample_ids"])
            # What the attempt captured but did not keep is exactly what its record
            # says sealing dropped undelivered and the agent discarded.
            left_out = record.get("generations_undelivered", 0) + record.get(
                "generations_discarded_by_agent", 0
            )
            _require(
                kept <= set(captured_ids) and len(captured_ids) - len(kept) == left_out,
                "rollout omits captured generations",
            )
            _require(
                [sample_id for sample_id in captured_ids if sample_id in kept]
                == record["sample_ids"],
                "rollout reorders captured generations",
            )
            # What dates a member that trains nothing: the trainable generations its
            # agent discarded, which are what it captured and left out when sealing
            # dropped none undelivered.
            left_out_versions = tuple(
                captures[sample_id][0].policy_version
                for sample_id in captured_ids
                if sample_id not in kept and captures[sample_id][0].trainable
            )
            rollout = Rollout(
                identity=AttemptIdentity(run_dir.name, record["group_id"], identity, attempt),
                task=TaskRef(record["task_id"], record["task_digest"]),
                samples=tuple(selected),
                reward=record["reward"],
                metadata={},
                discarded_versions=left_out_versions,
            )
            minimum = rollout.minimum_policy_version
            _require(minimum is not None, "rollout lacks trainable generations")
            if not rollout.trains and record.get("generations_undelivered", 0):
                # Undelivered and discarded generations are both left out: its own
                # minimum is one of them.
                _require(
                    record["policy_version"] in left_out_versions,
                    "recorded rollout minimum policy disagrees with samples",
                )
                minimum = record["policy_version"]
            _require(
                record["policy_version"] == minimum,
                "recorded rollout minimum policy disagrees with samples",
            )
            _require(
                all(value <= version for value in rollout.policy_versions),
                "sample policy newer than batch",
            )
            _require(
                version - minimum <= config.training.steps_off_policy,
                "stale rollout entered batch",
            )
            _require(
                rollout.internal_delta <= config.training.policy_delta_in_trajectory,
                "rollout exceeds internal policy spread",
            )
            rollouts.append(rollout)
            groups[rollout.identity.group_id].append(rollout)
        training = config.training
        _require(len(groups) == training.groups_per_step, "wrong group count in committed batch")
        advantage_by_rollout = {}
        for members in groups.values():
            _require(
                training.min_group_size <= len(members) <= training.group_size, "invalid group size"
            )
            _require(len({member.task for member in members}) == 1, "group mixes tasks")
            _require(all(member.reward is not None for member in members), "missing reward")
            _require(
                len({member.reward for member in members}) > 1,
                "zero-variance group entered training",
            )
            for member, value in zip(members, advantages([m.reward for m in members]), strict=True):
                advantage_by_rollout[member.identity.rollout_id] = value
        prepared = aggregate_loss(
            [
                [
                    row
                    for rollout in members
                    if rollout.trains
                    for row in build_rows(
                        rollout, advantage_by_rollout[rollout.identity.rollout_id]
                    )
                ]
                for members in groups.values()
            ],
            training,
        )
        expected_rows = prepared.rows
        if "aggregation" in batch:
            expected_report = asdict(prepared.report) if prepared.report is not None else None
            _require(
                batch["aggregation"] == json.loads(json.dumps(expected_report)),
                "aggregation report differs from the optimizer batch",
            )
        expected = []
        for row in normalize_batch_provenance(expected_rows, enabled=training.router_replay):
            payload = row.datum.model_dump(mode="json")
            expected.append(
                {
                    "datum": {
                        **payload,
                        "model_input": _redact_expected(payload["model_input"], image_sizes),
                    },
                    "actions": [list(span) for span in row.actions],
                }
            )
        actual = []
        for raw in batch["rows"]:
            restored = Datum.model_validate(
                {
                    **raw["datum"],
                    "model_input": _restore_model_input(raw["datum"]["model_input"], image_sizes),
                }
            )
            payload = restored.model_dump(mode="json")
            actual.append(
                {
                    "datum": {
                        **payload,
                        "model_input": _redact_expected(payload["model_input"], image_sizes),
                    },
                    "actions": raw["actions"],
                }
            )
        _require(
            actual == expected, f"step {step_id}: submitted datums differ from captured evidence"
        )
        datum_count += len(actual)
    uncommitted = set((run_dir / "batches").glob("step-*.json")) - committed_paths
    return AcceptanceReport(
        len(steps), len(used_rollouts), len(used_samples), datum_count, len(uncommitted)
    )
