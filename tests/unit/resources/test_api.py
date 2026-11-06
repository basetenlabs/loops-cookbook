"""The resource contracts refuse what the registry could not rely on."""

import math

import pytest

from loops_cookbook.resources.api import (
    CleanupIncomplete,
    CleanupReport,
    CleanupResult,
    CleanupStatus,
    CreationDisposition,
    ProviderDeadlines,
    ProviderResource,
    ProvisioningFailure,
    ReconcileResult,
    ReconcileStatus,
    ResourceRequest,
)


@pytest.mark.parametrize("bad", [0, -1, math.inf])
def test_every_provider_deadline_is_positive_and_finite(bad):
    good = dict.fromkeys(
        (
            "create_seconds",
            "reconcile_seconds",
            "release_seconds",
            "confirm_absent_seconds",
            "list_seconds",
            "attach_seconds",
        ),
        1.0,
    )
    for name in good:
        with pytest.raises(ValueError, match="positive and finite"):
            ProviderDeadlines(**{**good, name: bad})
    with pytest.raises(ValueError, match="positive and finite"):
        ProviderDeadlines(**good, create_seconds_by_kind={"image_build": bad})
    deadlines = ProviderDeadlines(**good, create_seconds_by_kind={"image_build": 900})
    assert deadlines.create_for("image_build") == 900
    assert deadlines.create_for("sandbox") == 1.0


def test_only_found_carries_a_resource_and_absence_needs_evidence():
    resource = ProviderResource(provider_id="p", name="n", labels={})
    with pytest.raises(ValueError, match="only a FOUND"):
        ReconcileResult(ReconcileStatus.FOUND)
    with pytest.raises(ValueError, match="only a FOUND"):
        ReconcileResult(ReconcileStatus.UNKNOWN, resource)
    with pytest.raises(ValueError, match="evidence"):
        ReconcileResult(ReconcileStatus.PROVEN_ABSENT)
    assert ReconcileResult(ReconcileStatus.PROVEN_ABSENT, evidence="404").evidence == "404"
    with pytest.raises(ValueError, match="unusable"):
        ReconcileResult(ReconcileStatus.UNKNOWN, unusable="dead")
    assert ReconcileResult(ReconcileStatus.FOUND, resource, unusable="dead").unusable == "dead"


def test_a_request_needs_its_identity_and_cannot_be_changed_after():
    with pytest.raises(ValueError, match="logical_key"):
        ResourceRequest(provider="fake", kind="box", logical_key="")
    labels, specification = {"team": "a"}, {"memory": 1, "disk": {"gb": 10}}
    request = ResourceRequest(
        provider="fake", kind="box", logical_key="k", labels=labels, specification=specification
    )
    labels["team"], specification["memory"], specification["disk"]["gb"] = "b", 2, 20
    assert request.labels["team"] == "a"
    assert request.specification == {"memory": 1, "disk": {"gb": 10}}
    for field in (request.labels, request.specification):
        with pytest.raises(TypeError):
            field["x"] = "y"  # pyright: ignore[reportIndexIssue]


def test_a_report_is_complete_only_when_every_resource_is_proven_gone():
    done = CleanupResult("a", CleanupStatus.RELEASED)
    absent = CleanupResult("b", CleanupStatus.ALREADY_ABSENT)
    kept = CleanupResult("c", CleanupStatus.UNCONFIRMED, "release failed")
    deferred = CleanupResult("d", CleanupStatus.DEFERRED)
    assert CleanupReport(None, None, (done, absent)).complete
    report = CleanupReport(None, None, (done, kept, deferred))
    assert not report.complete and report.unresolved == (kept, deferred)
    assert "c (unconfirmed), d (deferred)" in str(CleanupIncomplete(report))


@pytest.mark.parametrize(
    "disposition", [CreationDisposition.AMBIGUOUS, CreationDisposition.DEFINITELY_CREATED]
)
def test_only_a_create_that_left_nothing_can_be_refused(disposition):
    with pytest.raises(ValueError, match="left nothing"):
        ProvisioningFailure("rate limited", disposition, refused=True)
