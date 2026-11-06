"""Check the version a release is about to tag, against the repository's tags.

    release_version.py title "Release v0.2.0"   # a release PR's title
    release_version.py version 0.2.0            # the version typed into "Prepare release"

Prints the version on success; exits 1 with the reason otherwise. A release is
``X.Y.Z``, has no tag yet, and is above every earlier release (pre-releases, the
``-alpha.N`` tags, don't count). Reads tags from the git repository in the working
directory, so a checkout needs its tags (``fetch-depth: 0``).
"""

from __future__ import annotations

import re
import subprocess
import sys

_TITLE = re.compile(r"Release v(\d+\.\d+\.\d+)")
_RELEASE = re.compile(r"\d+\.\d+\.\d+")
_TAG = re.compile(r"v(\d+)\.(\d+)\.(\d+)(-alpha\.\d+)?")


def version_from_title(title: str) -> str:
    match = _TITLE.fullmatch(title.strip())
    if match is None:
        raise ValueError(
            f'a PR into main is a release and is titled "Release vX.Y.Z", not {title!r}'
        )
    return match.group(1)


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
    if len(argv) != 2 or argv[0] not in ("title", "version"):
        print(__doc__, file=sys.stderr)
        return 2
    kind, value = argv
    tags = subprocess.run(
        ["git", "tag", "--list", "v*"], capture_output=True, text=True, check=True
    ).stdout.split()
    try:
        version = version_from_title(value) if kind == "title" else value.strip().removeprefix("v")
        check(version, tags)
    except ValueError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
