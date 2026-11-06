"""A task that starts from an image already in Blaxel; no SDK import, so a Harbor check can
read a task's image without the Blaxel extra.

A task states it once, in its ``docker_image``: ``blaxel://<image>``. Everything after the
scheme goes to Blaxel's sandbox create unchanged, so any image reference Blaxel accepts
works, and nothing is built. Any other ``docker_image`` keeps Harbor's meaning: a base the
image is built on. The scheme is no Docker reference, so it cannot be mistaken for one.
"""

from __future__ import annotations

BLAXEL_IMAGE_SCHEME = "blaxel://"
# The one form Blaxel can look up, as its CLI names a workspace image: sandbox/<name>:<tag>.
WORKSPACE_IMAGE_PREFIX = "sandbox/"


def blaxel_image(docker_image: str | None) -> str | None:
    """The Blaxel image reference a ``docker_image`` of ``blaxel://<image>`` names."""
    if docker_image is None or not docker_image.startswith(BLAXEL_IMAGE_SCHEME):
        return None
    return docker_image.removeprefix(BLAXEL_IMAGE_SCHEME)


def workspace_image(reference: str) -> tuple[str, str | None] | None:
    """The name and tag of ``sandbox/<name>:<tag>``, the form Blaxel can look up, the tag
    ``None`` when it names none; ``None`` for any other form, which Blaxel may accept but
    has no lookup for, such as a digest (``sandbox/<name>@sha256:...``)."""
    if not reference.startswith(WORKSPACE_IMAGE_PREFIX):
        return None
    rest = reference.removeprefix(WORKSPACE_IMAGE_PREFIX)
    if "@" in rest:
        return None
    name, _, tag = rest.partition(":")
    return (name, tag or None) if name and "/" not in name else None
