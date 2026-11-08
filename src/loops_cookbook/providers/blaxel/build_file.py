"""Write and check the ``blaxel.toml`` that a build context carries.

This module imports no SDK, so a Harbor check can read a task's file without the
Blaxel extra.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

# The build-context file that Blaxel reads its build settings from. The cookbook uses it
# to set the region and to turn off Harbor's slimming.
BUILD_TOML_NAME = "blaxel.toml"


def build_toml(region: str, *, slim: bool) -> str:
    """Return the ``blaxel.toml`` for a build in ``region``.

    It also turns slimming off when ``slim`` is false.
    """
    lines = ["[build]", f"region = {json.dumps(region)}"]
    if not slim:
        lines.append("slim = false")
    return "\n".join(lines) + "\n"


def check_task_build_region(path: Path, region: str) -> None:
    """Refuse a task's own ``blaxel.toml`` when it builds in a region other than ``region``.

    The image would otherwise be built away from its sandboxes.
    """
    named = tomllib.loads(path.read_text()).get("build", {}).get("region")
    if named is not None and named != region:
        raise ValueError(
            f"{path} builds in region {named!r}, but [providers.blaxel] region is {region!r}"
        )


def with_region(text: str, region: str) -> str:
    """Return a task's ``blaxel.toml`` with ``region`` added under ``[build]``.

    The file's other keys are kept.
    """
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
    # The insertion cannot place the region in every layout. Refuse such a file rather
    # than guess.
    if after != expected:
        raise ValueError(f"cannot add region under [build] in this {BUILD_TOML_NAME}")
    return merged
