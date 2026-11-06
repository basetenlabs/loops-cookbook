"""Typed phase access selects the requested resource and explains unavailable ones."""

from dataclasses import replace

import pytest
from support.resources import Fake

from loops_cookbook.episodes import Closing, Grading, Sampling, Setup
from loops_cookbook.providers.blaxel import Blaxel
from loops_cookbook.resources import SandboxKind


@pytest.mark.parametrize(
    ("phase_type", "field", "method"),
    [
        (Setup, "sandbox", "require_sandbox"),
        (Sampling, "sandbox", "require_sandbox"),
        (Grading, "execution_sandbox", "require_execution_sandbox"),
        (Grading, "grading_sandbox", "require_grading_sandbox"),
        (Closing, "execution_sandbox", "require_execution_sandbox"),
        (Closing, "grading_sandbox", "require_grading_sandbox"),
    ],
)
def test_phase_access_selects_the_right_sandbox_and_checks_its_kind(phase_type, field, method):
    execution = Fake("execution", {}, None)
    verifier = Fake("verifier", {}, None)
    values = {"attempt_id": "attempt-1", "metrics": None, "processes": None}
    if phase_type in (Setup, Sampling):
        values["sandbox"] = execution
        if phase_type is Sampling:
            values["model"] = None
    else:
        values.update(execution_sandbox=execution, grading_sandbox=verifier, timed_out=False)
        if phase_type is Grading:
            values["rollout"] = None
    phase = phase_type(**values)
    expected = verifier if field == "grading_sandbox" else execution
    assert getattr(phase, method)(Fake) is expected
    assert getattr(phase, method)(SandboxKind) is expected
    with pytest.raises(TypeError, match=rf"{field}.*requires Blaxel, got Fake"):
        getattr(phase, method)(Blaxel)
    missing = replace(phase, **{field: None})
    with pytest.raises(RuntimeError, match=rf"{field}.*attempt-1.*unavailable"):
        getattr(missing, method)(Fake)
