"""Read a task's ``docker_image`` that names an image already in Blaxel.

This module imports no SDK, so a Harbor check can read a task's image without the
Blaxel extra. A task states the image once, as ``docker_image = "blaxel://<image>"``.
Everything after the scheme goes to Blaxel's sandbox create unchanged, so any image
reference Blaxel accepts works, and nothing is built. Any other ``docker_image`` keeps
Harbor's meaning, which is a base image to build on. The scheme is not a Docker
reference, so it cannot be mistaken for one.
"""

from __future__ import annotations

BLAXEL_IMAGE_SCHEME = "blaxel://"
# The one form of reference that Blaxel can look up. Its CLI names a workspace image
# sandbox/<name>:<tag>.
WORKSPACE_IMAGE_PREFIX = "sandbox/"


def blaxel_image(docker_image: str | None) -> str | None:
    """Return the Blaxel image that a ``docker_image`` of ``blaxel://<image>`` names.

    Any other ``docker_image`` returns ``None``.
    """
    if docker_image is None or not docker_image.startswith(BLAXEL_IMAGE_SCHEME):
        return None
    return docker_image.removeprefix(BLAXEL_IMAGE_SCHEME)


def workspace_image(reference: str) -> tuple[str, str | None] | None:
    """Return the name and tag of a ``sandbox/<name>:<tag>`` reference.

    That is the one form Blaxel can look up. The tag is ``None`` when the reference
    names none. Any other form returns ``None``. Blaxel may accept such a form but has no
    lookup for it. A digest, such as ``sandbox/<name>@sha256:...``, is one example.
    """
    if not reference.startswith(WORKSPACE_IMAGE_PREFIX):
        return None
    rest = reference.removeprefix(WORKSPACE_IMAGE_PREFIX)
    if "@" in rest:
        return None
    name, _, tag = rest.partition(":")
    return (name, tag or None) if name and "/" not in name else None
