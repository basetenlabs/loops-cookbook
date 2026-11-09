"""Check the version a release is about to tag, against the repository's tags.

    release_version.py 0.2.0    # the version typed into the release workflow

Prints the version on success; exits 1 with the reason otherwise. A release is
``X.Y.Z``, has no tag yet, and is above every earlier release (pre-releases, the
``-alpha.N`` tags, don't count). Reads tags from the git repository in the working
directory, so a checkout needs its tags (``fetch-depth: 0``).
"""

from __future__ import annotations

import re
import subprocess
import sys

_RELEASE = re.compile(r"\d+\.\d+\.\d+")
_TAG = re.compile(r"v(\d+)\.(\d+)\.(\d+)(-alpha\.\d+)?")


def check(version: str, tags: list[str]) -> None:
    if _RELEASE.fullmatch(version) is None:
        raise ValueError(f"a release version is X.Y.Z, not {version!r}")
    if f"v{version}" in tags:
        raise ValueError(f"v{version} is already tagged")
    releases = [m for m in map(_TAG.fullmatch, tags) if m is not None and m.group(4) is None]
    if releases:
        latest = max(tuple(int(part) for part in m.groups()[:3]) for m in releases)
        if tuple(int(part) for part in version.split(".")) <= latest:
            raise ValueError(
                f"v{version} is not above the latest release, v{'.'.join(map(str, latest))}"
            )


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    version = argv[0].strip().removeprefix("v")
    tags = subprocess.run(
        ["git", "tag", "--list", "v*"], capture_output=True, text=True, check=True
    ).stdout.split()
    try:
        check(version, tags)
    except ValueError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
