"""Upstream Harbor code the cookbook repeats or relies on, pinned to the reviewed source.

A digest that changes means a Harbor upgrade changed that code: re-review what its
entry names before updating the digest.
"""

import hashlib
import inspect
import textwrap

import pytest

# (module, qualified name) -> (digest of its source, what a change needs re-reviewed)
PINS = {
    ("harbor.trial.single_step", "SingleStepTrial._run"): (
        "62058058105bd9388e34e9b0943b109cec32019e490dece175bc202424a0bc57",
        # It calls ``_run_verifier`` only after the agent phase ended in a way Harbor still
        # verifies, which is what ``integration.mark_verification_reached`` records.
        "the separate-verifier grading decision",
    ),
    ("harbor.environments.blaxel", "BlaxelEnvironment._start_direct"): (
        "6712271ed1eca12e453f82a9b7dadd4d31e2abfbc47b3de6c445d620b4355b2a",
        # The cookbook's Blaxel environment repeats its post-create tail.
        "the attach comparison",
    ),
    ("harbor.environments.blaxel", "BlaxelEnvironment._exec_on_sandbox_host"): (
        "f22656596bf4b0faa06d7bd8cafad67af00e38b1b0f2ef6a7a35583c940a5c5d",
        # It waits on the process and returns that record's streams, which is where
        # ``complete_output`` fixes Blaxel's 64 KB cut.
        "the full-output path",
    ),
    ("harbor.environments.daytona.environment", "_DaytonaDirect.start"): (
        "41466ad8576fdbd3be0df42950d8daa52523d76ac8757efbfb7ee561284c8c27",
        # The cookbook's Daytona start repeats the direct strategy's post-create tail
        # without Harbor's client.
        "the subclass's start",
    ),
    (
        "harbor.environments.daytona.environment",
        "DaytonaEnvironment._resolve_start_sandbox_params",
    ): (
        "a6a8309d6324679579d0ec1f94d7771afa8c28e229f38ce2c93471385d2ef59f",
        # The start's image choice, which the cookbook's Daytona start repeats.
        "the subclass's start",
    ),
}

# Private members of Harbor's Blaxel environment that the cookbook's subclass calls.
BLAXEL_MEMBERS = [
    "_create_sandbox_config",
    "_build_image_from_dockerfile",
    "_build_image_from_docker_image",
    "_require_image_name",
    "_wait_until_ready",
    "_upload_environment_dir_after_start",
    "_start_direct",
    "_stop_direct",
    "_memory_mb",
]


def _member(module: str, qualified: str):
    owner = pytest.importorskip(module)
    for part in qualified.split("."):
        owner = getattr(owner, part)
    return owner


@pytest.mark.parametrize(
    ("module", "qualified"), sorted(PINS), ids=[name for _, name in sorted(PINS)]
)
def test_upstream_code_is_the_reviewed_source(module, qualified):
    digest, review = PINS[module, qualified]
    source = textwrap.dedent(inspect.getsource(_member(module, qualified)))
    assert hashlib.sha256(source.encode()).hexdigest() == digest, f"re-review {review}"


@pytest.mark.parametrize("member", BLAXEL_MEMBERS)
def test_the_private_harbor_members_the_blaxel_subclass_uses_still_exist(member):
    assert hasattr(_member("harbor.environments.blaxel", "BlaxelEnvironment"), member)
