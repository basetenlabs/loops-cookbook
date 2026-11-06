"""The sandbox and grading_sandbox tables: the kind an episode is written for, the spec
it is checked against, and the run evidence they record.
"""

from __future__ import annotations

import pydantic
import pytest
from support.resources import FAST, Fake, FakeSpec

from loops_cookbook.episodes.tasks import task_digest
from loops_cookbook.providers.blaxel import Blaxel
from loops_cookbook.runtime.context import EvidenceRecord, RuntimeConfigurationError

from .declared import (
    DECLARED,
    FAKE_SANDBOX,
    KeyedSpec,
    LenientSpec,
    LooseSpec,
    NotASpec,
    RenamedSpec,
    SerializedSpec,
    build,
    fake_kind,
)


@pytest.mark.parametrize("episode", ["OnFake", "OnFakeKind"])
def test_a_sandbox_of_the_provider_the_episode_is_written_for_is_accepted(tmp_path, episode):
    build(
        tmp_path,
        episode=f"{DECLARED}:{episode}",
        sandbox=FAKE_SANDBOX,
        limits={"providers": {"fake": FAST}},
    )


@pytest.mark.parametrize(
    ("episode", "sandbox", "sandbox_providers", "problem"),
    [
        ("OnFake", None, None, "written for a 'fake' sandbox, and .* has no sandbox table"),
        ("Answer", FAKE_SANDBOX, None, "episode .* declares no sandbox kind to use it"),
        ("OnBlaxel", FAKE_SANDBOX, None, r"\[providers.fake\] adapter provides Fake .* for Blaxel"),
        ("OnFake", FAKE_SANDBOX, {}, r"\[providers.fake\] adapter is not a Sandbox"),
        ("OnUnnamed", FAKE_SANDBOX, None, "sandbox kind Unnamed names no provider"),
        ("OnNotAKind", FAKE_SANDBOX, None, "its sandbox kind <class 'str'> is not a SandboxKind"),
        ("KindLeftOpen", FAKE_SANDBOX, None, "leaves its sandbox kind open: ~KindT"),
    ],
    ids=[
        "missing",
        "unused",
        "another provider",
        "adapter without the protocol",
        "a kind naming no provider",
        "not a kind",
        "left open",
    ],
)
def test_the_sandbox_table_must_be_the_one_the_episode_is_written_for(
    tmp_path, episode, sandbox, sandbox_providers, problem
):
    config = {} if sandbox is None else {"sandbox": sandbox}
    limits = {"providers": {"fake": FAST}, "sandbox_providers": sandbox_providers}
    with pytest.raises(RuntimeConfigurationError, match=problem):
        build(tmp_path, episode=f"{DECLARED}:{episode}", limits=limits, **config)


@pytest.mark.parametrize(
    ("table", "problem"),
    [
        ({"ttl": "1h", "image": "img:1", "memory_mb": 2048}, None),
        ({"ttl": "1h", "memory_mb": 2048}, r"image\n  Field required"),
        ({"ttl": "1h", "image": "img:1"}, r"memory_mb\n  Field required"),
        ({}, r"ttl\n  Field required"),
        ({"ttl": "1h", "memroy_mb": 2048}, r"memroy_mb\n  Extra inputs are not permitted"),
        ({"ttl": ""}, r"ttl\n  String should have at least 1 character"),
        ({"ttl": "1h", "memory_mb": 0}, r"memory_mb\n  Input should be greater than 0"),
        ({"ttl": "1h", "image": ""}, r"image\n  String should have at least 1 character"),
        (
            {"ttl": "1h", "kind": "sandbox", "specification": {"sandbox": {"memory": 1}}},
            r"kind\n  Extra inputs are not permitted",
        ),
        ({"ttl": "1h", "memory_mb": True}, r"memory_mb\n  Input should be a valid integer"),
        ({"ttl": "1h", "memory_mb": "4096"}, r"memory_mb\n  Input should be a valid integer"),
        ({"ttl": "1h", "memory_mb": 4096.0}, r"memory_mb\n  Input should be a valid integer"),
        ({"ttl": "1h", "image": 7}, r"image\n  Input should be a valid string"),
        ({"ttl": 24}, r"ttl\n  Input should be a valid string"),
        ({"ttl": "banana"}, r"ttl\n  String should match pattern"),
        ({"ttl": "24"}, r"ttl\n  String should match pattern"),
        ({"ttl": "0h"}, r"ttl\n  String should match pattern"),
        ({"ttl": "1h30m"}, r"ttl\n  String should match pattern"),
        ({"ttl": "24H"}, r"ttl\n  String should match pattern"),
        ({"ttl": " 24h"}, r"ttl\n  String should match pattern"),
        ({"ttl": "24h\n"}, r"ttl\n  String should match pattern"),
        ({"ttl": "2w", "image": "img:1", "memory_mb": 2048}, None),
        ({"ttl": "90s", "image": "img:1", "memory_mb": 2048}, None),
    ],
    ids=[
        "every key",
        "no image",
        "no memory",
        "no ttl",
        "a typo",
        "an empty ttl",
        "memory of zero",
        "an empty image",
        "pass-through",
        "memory as a bool",
        "memory as a string",
        "memory as a float",
        "image as a number",
        "ttl as a number",
        "ttl as a word",
        "ttl without a unit",
        "ttl of zero",
        "ttl of two units",
        "ttl in capitals",
        "ttl with a space",
        "ttl with a newline",
        "ttl in weeks",
        "ttl in seconds",
    ],
)
def test_a_blaxel_sandbox_table_is_the_kinds_flat_spec_checked_when_it_is_built(
    tmp_path, table, problem
):
    config = {
        "episode": f"{DECLARED}:OnBlaxel",
        "sandbox": {"provider": "blaxel", **table},
    }
    limits = {"providers": {"blaxel": FAST}, "sandbox_providers": {"blaxel": Blaxel}}
    if problem is None:
        build(tmp_path, limits=limits, **config)
        return
    with pytest.raises(
        RuntimeConfigurationError,
        match=rf"(?s)\[runtime.config.sandbox\] is not a valid.*{problem}",
    ):
        build(tmp_path, limits=limits, **config)


@pytest.mark.parametrize(
    ("spec", "problem"),
    [
        (LenientSpec, r"LenientSpec reaches LenientSpec, which does not set .*extra='forbid'"),
        (KeyedSpec, r"KeyedSpec reaches KeyedSpec, whose field 'token' is typed with SecretStr"),
        (RenamedSpec, r"RenamedSpec, whose field 'image' is read or dumped as 'img'"),
        (SerializedSpec, r"SerializedSpec, whose field 'image' serializes its own way"),
        (LooseSpec, r"LooseSpec, whose field 'labels' is typed with dict"),
        (NotASpec, r"'NotASpec' is not a SandboxSpec"),
    ],
    ids=["extra keys", "a secret", "renamed", "serialized", "untyped", "not a spec"],
)
def test_a_sandbox_spec_the_allow_list_refuses_is_refused_when_the_runtime_is_built(
    tmp_path, spec, problem
):
    limits = {"providers": {"fake": FAST}, "sandbox_providers": {"fake": fake_kind(spec)}}
    with pytest.raises(RuntimeConfigurationError, match=rf"FakeKind sandbox spec .*{problem}"):
        build(tmp_path, episode=f"{DECLARED}:OnFake", sandbox={"provider": "fake"}, limits=limits)


@pytest.mark.parametrize(
    ("table", "ran"),
    [
        ({}, {"image": "fake-image"}),  # the default ran
        ({"image": "x"}, {"image": "x"}),
    ],
    ids=["defaults", "set"],
)
def test_the_validated_sandbox_spec_is_run_evidence_defaults_included(tmp_path, table, ran):
    evidence = EvidenceRecord()
    build(
        tmp_path,
        episode=f"{DECLARED}:OnFake",
        sandbox={"provider": "fake", **table},
        limits={"providers": {"fake": FAST}},
        evidence=evidence,
    )
    assert evidence.digests["sandbox"] == task_digest(ran)


def test_a_changed_sandbox_default_changes_the_run_evidence(tmp_path):
    def recorded(spec) -> dict[str, str]:
        evidence = EvidenceRecord()
        limits = {"providers": {"fake": FAST}, "sandbox_providers": {"fake": fake_kind(spec)}}
        build(
            tmp_path,
            episode=f"{DECLARED}:OnFake",
            sandbox={"provider": "fake"},
            limits=limits,
            evidence=evidence,
        )
        return evidence.digests

    moved = pydantic.create_model("MovedSpec", __base__=FakeSpec, image=(str, "another-image"))
    assert recorded(FakeSpec) != recorded(moved)


def test_a_blaxel_spec_asks_blaxel_for_its_image_memory_and_ttl():
    spec = Blaxel.spec.model_validate({"ttl": "1h", "image": "img:1", "memory_mb": 2048})
    assert spec.resource_kind == "sandbox"
    assert spec.specification() == {"sandbox": {"image": "img:1", "memory": 2048, "ttl": "1h"}}


def test_sandbox_lifetime_is_validated(tmp_path):
    with pytest.raises(RuntimeConfigurationError, match="sandbox_lifetime"):
        build(tmp_path, episode=f"{DECLARED}:InvalidLifetime")


@pytest.mark.parametrize(
    ("episode", "lifetime"), [("Answer", "attempt"), ("SamplingLifetime", "sampling")]
)
def test_effective_sandbox_lifetime_is_run_evidence(tmp_path, episode, lifetime):
    evidence = EvidenceRecord()
    build(tmp_path, episode=f"{DECLARED}:{episode}", evidence=evidence)
    assert evidence.digests["sandbox_lifetime"] == task_digest({"sandbox_lifetime": lifetime})


@pytest.mark.parametrize("episode", ["GradeSandboxOnly", "BoundVerifier"])
def test_grading_only_and_bound_generic_verifier_are_valid(tmp_path, episode):
    evidence = EvidenceRecord()
    build(
        tmp_path,
        episode=f"{DECLARED}:{episode}",
        grading_sandbox={"provider": "fake"},
        limits={"providers": {"fake": FAST}},
        evidence=evidence,
    )
    assert evidence.digests["grading_sandbox"] == task_digest({"image": "fake-image"})
    assert "sandbox" not in evidence.digests


def test_absent_grading_kind_does_not_require_or_record_a_verifier(tmp_path):
    evidence = EvidenceRecord()
    build(tmp_path, episode=f"{DECLARED}:Answer", evidence=evidence)
    assert evidence.digests == {"sandbox_lifetime": task_digest({"sandbox_lifetime": "attempt"})}


@pytest.mark.parametrize(
    ("episode", "table", "providers", "provided", "problem"),
    [
        ("GradeSandboxOnly", None, {"fake": FAST}, {"fake": Fake}, "no grading_sandbox table"),
        (
            "Answer",
            {"provider": "fake"},
            {"fake": FAST},
            {"fake": Fake},
            "declares no sandbox kind",
        ),
        ("GradeSandboxOnly", {"provider": "missing"}, {}, {}, "grading_sandbox.*not configured"),
        ("GradeSandboxOnly", {"provider": "fake"}, {"fake": FAST}, {}, "not a SandboxProvider"),
        (
            "GradeSandboxOnly",
            {"provider": "fake"},
            {"fake": FAST},
            {"fake": Blaxel},
            "adapter provides Blaxel",
        ),
        (
            "GradeSandboxOnly",
            {"provider": "fake", "image": True},
            {"fake": FAST},
            {"fake": Fake},
            "grading_sandbox.*not a valid",
        ),
        (
            "GradeSandboxOnly",
            {"provider": "fake", "typo": 1},
            {"fake": FAST},
            {"fake": Fake},
            "grading_sandbox.*not a valid",
        ),
        (
            "GroupOnlySandbox",
            {"provider": "fake"},
            {"fake": FAST},
            {"fake": Fake},
            "requires a per-attempt grade",
        ),
        ("UnboundVerifier", None, {}, {}, "leaves its sandbox kind open"),
    ],
)
def test_invalid_grading_role_is_refused_before_provisioning(
    tmp_path, episode, table, providers, provided, problem
):
    config = {} if table is None else {"grading_sandbox": table}
    with pytest.raises(RuntimeConfigurationError, match=problem):
        build(
            tmp_path,
            episode=f"{DECLARED}:{episode}",
            limits={"providers": providers, "sandbox_providers": provided},
            **config,
        )


def test_both_roles_have_distinct_effective_evidence(tmp_path):
    evidence = EvidenceRecord()
    build(
        tmp_path,
        episode=f"{DECLARED}:SeparateSandboxes",
        sandbox={"provider": "fake", "image": "agent"},
        grading_sandbox={"provider": "fake", "image": "verifier"},
        limits={"providers": {"fake": FAST}},
        evidence=evidence,
    )
    assert evidence.digests["sandbox"] == task_digest({"image": "agent"})
    assert evidence.digests["grading_sandbox"] == task_digest({"image": "verifier"})
