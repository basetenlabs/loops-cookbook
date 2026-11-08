"""The factory context: opaque configuration, declared files, metrics, evidence."""

import math

import pytest

from loops_cookbook.evidence import TaskRef
from loops_cookbook.model_access.api import ModelAccessMode
from loops_cookbook.runtime.api import Phase, RuntimeCapabilities
from loops_cookbook.runtime.context import (
    EvidenceRecord,
    MetricsSink,
    Preflight,
    RuntimeConfigurationError,
    RuntimeFileResolver,
    build_runtime,
    check_capabilities,
    check_tasks,
)

DIRECT = frozenset({ModelAccessMode.DIRECT})
BOTH = frozenset(ModelAccessMode)


def test_the_factory_receives_a_private_copy_of_exactly_the_users_values():
    config = {"path": "/bin/bash", "endpoint": "/v1/chat", "api_key": "k", "nested": {"a": [1]}}
    seen = []

    def factory(received, *, context):
        seen.append(received)
        received["nested"]["a"].append(2)
        return "runtime"

    assert build_runtime(factory, config, context=None) == "runtime"
    assert seen == [
        {"path": "/bin/bash", "endpoint": "/v1/chat", "api_key": "k", "nested": {"a": [1, 2]}}
    ]
    assert config["nested"]["a"] == [1]  # the rollout runtime's copy is private


def test_only_declared_file_aliases_resolve(tmp_path):
    files = RuntimeFileResolver({"job": tmp_path / "job.yaml"})
    assert files.path("job") == tmp_path / "job.yaml"
    with pytest.raises(RuntimeConfigurationError, match="declares no alias"):
        files.path("other")


def test_runtime_metrics_are_namespaced_finite_and_summarised():
    sink = MetricsSink()
    sink.set_gauge("harbor/live_trials", 3)
    sink.observe("harbor/trial_seconds", 1.0)
    sink.observe("harbor/trial_seconds", 3.0)
    assert sink.snapshot() == {
        "runtime/harbor/live_trials": 3.0,
        "runtime/harbor/trial_seconds/count": 2.0,
        "runtime/harbor/trial_seconds/mean": 2.0,
        "runtime/harbor/trial_seconds/max": 3.0,
    }
    with pytest.raises(ValueError, match="finite"):
        sink.set_gauge("x", math.nan)
    with pytest.raises(TypeError):
        sink.observe("x", True)


DIGEST = "sha256:" + "0" * 64


@pytest.mark.parametrize(
    "value", ["tok-5e3f9a", "sha256:abc", "SHA256:" + "0" * 64, DIGEST + "0", {"a": 1}]
)
def test_runtime_evidence_is_a_digest_never_a_value(value):
    evidence = EvidenceRecord()
    with pytest.raises(ValueError, match="must be a sha256: digest"):
        evidence.record_digest("options", value)
    assert evidence.digests == {}


def test_runtime_evidence_is_recorded_once_per_name():
    evidence = EvidenceRecord()
    evidence.record_digest("options", DIGEST)
    with pytest.raises(ValueError, match="already recorded"):
        evidence.record_digest("options", "sha256:" + "1" * 64)
    assert evidence.digests == {"options": DIGEST}


@pytest.mark.parametrize(
    "capabilities,preflight,match",
    [
        (RuntimeCapabilities(frozenset()), Preflight(None, 30), "supports \\[\\]"),
        (
            RuntimeCapabilities(DIRECT, frozenset({Phase.PROVISIONING})),
            Preflight(None, 30),
            "provisioning",
        ),
        (RuntimeCapabilities(DIRECT, grading_retries=-1), Preflight(None, 30), "nonnegative"),
        ("not capabilities", Preflight(None, 30), "RuntimeCapabilities"),
        (
            RuntimeCapabilities(DIRECT),
            Preflight(None, None),
            r"\[rollout\] rollout_timeout_seconds is required: this rollout runtime derives no rollout",
        ),
        (
            RuntimeCapabilities(BOTH),
            Preflight(None, 30),
            r"\[rollout_runtime\] model_access is required: the rollout runtime supports \['direct', 'openai_http'\]",
        ),
        (
            RuntimeCapabilities(DIRECT),
            Preflight(ModelAccessMode.DIRECT, 30),
            "the rollout runtime already declares 'direct'; the rollout runtime states it once",
        ),
        (
            RuntimeCapabilities(DIRECT, rollout_timeout_seconds=0),
            Preflight(None, None),
            "rollout_timeout_seconds must be finite and positive",
        ),
    ],
)
def test_capabilities_are_validated_at_preflight(capabilities, preflight, match):
    with pytest.raises(RuntimeConfigurationError, match=match):
        check_capabilities(capabilities, preflight)


def test_prepared_tasks_are_task_refs_an_id_repeated_as_the_runtime_lists_it():
    assert check_tasks([TaskRef("t"), TaskRef("t")]) == [TaskRef("t"), TaskRef("t")]
    with pytest.raises(RuntimeConfigurationError, match="not a TaskRef"):
        check_tasks(["t"])
    with pytest.raises(RuntimeConfigurationError, match="no tasks"):
        check_tasks([])


@pytest.mark.parametrize("digest", ["sha256:ABC", "md5:" + "a" * 32, 7])
def test_an_attempts_task_digest_must_be_a_sha256(digest):
    from loops_cookbook.evidence import TrajectoryError, check_task_digest

    with pytest.raises(TrajectoryError):
        check_task_digest(digest, where="rollout")
    assert check_task_digest(None, where="rollout") is None
    assert check_task_digest("sha256:" + "a" * 64, where="rollout")
