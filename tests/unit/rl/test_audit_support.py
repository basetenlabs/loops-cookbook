import json

import pytest
from support.audit import AcceptanceError, audit_run
from support.builders import make_config, make_rollout, make_sample, task, write_jsonl

from loops_cookbook.rl.aggregation import aggregate_loss
from loops_cookbook.rl.grpo import build_rows
from loops_cookbook.rl.run import row_record, sample_record


@pytest.fixture
def evidence(tmp_path):
    config = make_config(group_size=2, max_group_losses=0, groups_per_step=1)
    (tmp_path / "config.json").write_text(config.model_dump_json())
    rollouts = []
    datums = []
    for index, reward in enumerate((0.0, 1.0)):
        sample = make_sample(f"s{index}", (1, 2), (3 + index,), (-0.5,), 7)
        rollout = make_rollout("task", "group", f"r{index}", (sample,), reward)
        record = {
            "rollout_id": rollout.identity.rollout_id,
            "attempt_id": f"a{index}",
            "group_id": "group",
            "task_id": "task",
            "task_digest": task("task").digest,
            "reward": reward,
            "sample_ids": [sample.sample_id],
            "policy_version": 7,
            "step": 4,
            "disposition": "trained",
        }
        rollouts.append(record)
        write_jsonl(
            tmp_path / "samples" / f"a{index}.jsonl",
            [
                {
                    **sample_record(sample, tokens=True),
                    "rollout_id": rollout.identity.rollout_id,
                    "group_id": "group",
                    "attempt_id": f"a{index}",
                }
            ],
        )
        datums.extend(build_rows(rollout, reward - 0.5))
    rows = [row_record(row) for row in aggregate_loss([datums], config.training).rows]
    write_jsonl(tmp_path / "rollouts.jsonl", rollouts)
    write_jsonl(tmp_path / "steps.jsonl", [{"step": 4, "train/policy_version": 8}])
    (tmp_path / "batches").mkdir()
    (tmp_path / "batches" / "step-4.json").write_text(
        json.dumps({"step": 4, "policy_version": 7, "rollout_ids": ["r0", "r1"], "rows": rows})
    )
    return tmp_path


def test_audits_resumed_committed_steps_and_reports_uncommitted_preparation(evidence):
    (evidence / "batches" / "step-5.json").write_text("{}")
    report = audit_run(evidence)
    assert (report.steps, report.rollouts, report.samples, report.datums) == (1, 2, 2, 2)
    assert report.uncommitted_batches == 1


@pytest.mark.parametrize(
    "field,value", [("target_tokens", 99), ("logprobs", -9.0), ("advantages", 0.0)]
)
def test_rejects_modified_training_tensors(evidence, field, value):
    path = evidence / "batches" / "step-4.json"
    batch = json.loads(path.read_text())
    batch["rows"][0]["datum"]["loss_fn_inputs"][field]["data"][1] = value
    path.write_text(json.dumps(batch))
    with pytest.raises(AcceptanceError, match="datums differ"):
        audit_run(evidence)


def test_missing_sample_evidence_fails(evidence):
    (evidence / "samples" / "a0.jsonl").unlink()
    with pytest.raises(AcceptanceError, match="missing sample"):
        audit_run(evidence)


def test_omitted_generation_from_selected_attempt_fails(evidence):
    path = evidence / "samples" / "a0.jsonl"
    record = json.loads(path.read_text())
    write_jsonl(path, [record, {**record, "sample_id": "omitted"}])
    with pytest.raises(AcceptanceError, match="omits captured"):
        audit_run(evidence)


@pytest.mark.parametrize(
    "undelivered,discarded,accepted", [(1, 0, True), (0, 1, True), (0, 0, False), (1, 1, False)]
)
def test_a_generation_left_out_of_the_attempt_passes_only_when_its_record_counts_it(
    evidence, undelivered, discarded, accepted
):
    path = evidence / "samples" / "a0.jsonl"
    record = json.loads(path.read_text())
    write_jsonl(path, [record, {**record, "sample_id": "left-out"}])
    path = evidence / "rollouts.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0] |= {
        "generations_undelivered": undelivered,
        "generations_discarded_by_agent": discarded,
    }
    write_jsonl(path, records)
    if accepted:
        assert audit_run(evidence).rollouts == 2
    else:
        with pytest.raises(AcceptanceError, match="omits captured"):
            audit_run(evidence)


def test_a_rollout_that_kept_nothing_cannot_name_another_rollouts_attempt(evidence):
    path = evidence / "rollouts.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    # r0 kept no generation, and names r1's attempt, which captured one at the same
    # version: everything else about it, the batch included, is consistent.
    records[0] |= {"sample_ids": [], "attempt_id": "a1", "generations_discarded_by_agent": 1}
    write_jsonl(path, records)
    config = make_config(group_size=2, max_group_losses=0, groups_per_step=1)
    sample = make_sample("s1", (1, 2), (4,), (-0.5,), 7)
    rows = build_rows(make_rollout("task", "group", "r1", (sample,), 1.0), 0.5)
    batch_path = evidence / "batches" / "step-4.json"
    batch = json.loads(batch_path.read_text())
    batch["rows"] = [row_record(row) for row in aggregate_loss([rows], config.training).rows]
    batch_path.write_text(json.dumps(batch))
    with pytest.raises(AcceptanceError, match="attempt attribution disagrees"):
        audit_run(evidence)


def test_sync_policy_mismatch_fails_even_with_valid_datum_shapes(evidence):
    path = evidence / "samples" / "a0.jsonl"
    record = json.loads(path.read_text())
    record["policy_version"] = 6
    write_jsonl(path, [record])
    rollout_path = evidence / "rollouts.jsonl"
    rollouts = [json.loads(line) for line in rollout_path.read_text().splitlines()]
    rollouts[0]["policy_version"] = 6
    write_jsonl(rollout_path, rollouts)
    with pytest.raises(AcceptanceError, match="stale rollout entered batch"):
        audit_run(evidence)


def test_duplicate_capture_identity_fails(evidence):
    path = evidence / "samples" / "a0.jsonl"
    record = json.loads(path.read_text())
    write_jsonl(path, [record, record])
    with pytest.raises(AcceptanceError, match="duplicate captured"):
        audit_run(evidence)


def test_rejects_modified_actions(evidence):
    path = evidence / "batches" / "step-4.json"
    batch = json.loads(path.read_text())
    batch["rows"][0]["actions"] = [[0, 2]]
    path.write_text(json.dumps(batch))
    with pytest.raises(AcceptanceError, match="datums differ"):
        audit_run(evidence)


def test_no_committed_steps_is_not_acceptance(evidence):
    (evidence / "steps.jsonl").write_text("")
    with pytest.raises(AcceptanceError, match="no committed"):
        audit_run(evidence)


def test_replay_reference_without_compiler_evidence_fails(evidence):
    path = evidence / "batches" / "step-4.json"
    batch = json.loads(path.read_text())
    batch["rows"][0]["datum"]["model_input_spans"] = [
        {"type": "prompt", "sequence_id": "unverified", "offset": 0, "length": 2}
    ]
    batch["rows"][0]["datum"]["loss_fn_input_spans"] = [
        {"type": "prompt", "sequence_id": "unverified", "offset": 1, "length": 1},
        {"type": "sampled", "sequence_id": "unverified", "offset": 0, "length": 1},
    ]
    path.write_text(json.dumps(batch))
    with pytest.raises(AcceptanceError, match="datums differ"):
        audit_run(evidence)


def test_reordered_generations_fail_before_recompilation(evidence):
    path = evidence / "samples" / "a0.jsonl"
    first = json.loads(path.read_text())
    second = {**first, "sample_id": "later", "prompt_ids": [1, 2, 3, 4], "prompt_input": None}
    write_jsonl(path, [first, second])
    path = evidence / "rollouts.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["sample_ids"] = ["later", "s0"]
    write_jsonl(path, records)
    with pytest.raises(AcceptanceError, match="reorders captured"):
        audit_run(evidence)
