"""Package metadata declares compatible ranges; ``uv.lock`` records the tested versions."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
LOCKED = {
    package["name"]: package.get("version")
    for package in tomllib.loads((ROOT / "uv.lock").read_text())["package"]
}
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
DECLARED = [
    Requirement(line)
    for line in [
        *PROJECT["dependencies"],
        *(line for extra in PROJECT["optional-dependencies"].values() for line in extra),
    ]
]


@pytest.mark.parametrize("requirement", [r for r in DECLARED if r.specifier], ids=lambda r: str(r))
def test_every_declared_range_admits_the_locked_version(requirement):
    locked = LOCKED[requirement.name]
    assert locked and requirement.specifier.contains(locked, prereleases=True)


def test_metadata_declares_ranges_not_exact_pins():
    pinned = [str(r) for r in DECLARED if any(s.operator in ("==", "===") for s in r.specifier)]
    assert pinned == []


@pytest.mark.parametrize("package", ["harbor", "blaxel", "renderers"])
def test_packages_whose_private_members_are_used_have_a_bounded_range(package):
    ranges = [r for r in DECLARED if r.name == package]
    assert ranges and all(any(s.operator == "<" for s in r.specifier) for r in ranges)
