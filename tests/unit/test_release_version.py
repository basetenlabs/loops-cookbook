"""A release can only tag a new X.Y.Z above every earlier release."""

import pytest
from support.unit_rest import load_script

TAGS = ["v0.1.0-alpha.1", "v0.1.0-alpha.2", "v0.1.0", "v0.2.0-alpha.1"]


@pytest.fixture(scope="module")
def release():
    return load_script("release_version")


@pytest.mark.parametrize(
    "version",
    ["0.1.1", "0.2.0", "1.0.0"],
)
def test_a_new_version_above_the_latest_release_is_accepted(release, version):
    release.check(version, TAGS)


def test_the_first_release_is_accepted_over_only_pre_releases(release):
    release.check("0.1.0", ["v0.1.0-alpha.1", "v0.1.0-alpha.2"])


@pytest.mark.parametrize(
    ("version", "reason"),
    [
        ("0.1.0", "already tagged"),
        ("0.0.9", "not above the latest release, v0.1.0"),
        ("0.2.0-alpha.2", "is X.Y.Z"),
        ("0.2", "is X.Y.Z"),
    ],
)
def test_a_tagged_lower_or_malformed_version_is_refused(release, version, reason):
    with pytest.raises(ValueError, match=reason):
        release.check(version, TAGS)
