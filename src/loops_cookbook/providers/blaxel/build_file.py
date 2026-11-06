"""The ``blaxel.toml`` a build context carries; no SDK import, so a Harbor check can
read a task's file without the Blaxel extra."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

# The build-context file Blaxel reads its build settings from: the region, and
# Harbor's slimming off.
BUILD_TOML_NAME = "blaxel.toml"


def build_toml(region: str, *, slim: bool) -> str:
    """The ``blaxel.toml`` a build carries: its region, and slimming off if asked."""
    lines = ["[build]", f"region = {json.dumps(region)}"]
    if not slim:
        lines.append("slim = false")
    return "\n".join(lines) + "\n"


def check_task_build_region(path: Path, region: str) -> None:
    """Refuse a task's own ``blaxel.toml`` that builds in a region other than ``region``,
    since its image would be built away from its sandboxes."""
    named = tomllib.loads(path.read_text()).get("build", {}).get("region")
    if named is not None and named != region:
        raise ValueError(
            f"{path} builds in region {named!r}, but [providers.blaxel] region is {region!r}"
        )


def with_region(text: str, region: str) -> str:
    """A task's ``blaxel.toml`` with ``region`` added under ``[build]``, its keys kept."""
    if tomllib.loads(text).get("build", {}).get("region") == region:
        return text
    line = f"region = {json.dumps(region)}"
    lines = text.splitlines()
    for index, current in enumerate(lines):
        if current.strip() == "[build]":
            lines.insert(index + 1, line)
            break
    else:
        lines += ["", "[build]", line]
    merged = "\n".join(lines) + "\n"
    before, after = tomllib.loads(text), tomllib.loads(merged)
    expected = {**before, "build": {**before.get("build", {}), "region": region}}
    if after != expected:  # a layout the insertion cannot place: say so, never guess
        raise ValueError(f"cannot add region under [build] in this {BUILD_TOML_NAME}")
    return merged
