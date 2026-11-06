"""`[providers.blaxel]`: the region, every Blaxel deadline, and sandbox labels.

Credentials are not settings: `BL_WORKSPACE` and `BL_API_KEY` come from the
environment, loaded from `.env`. The region is required: where sandboxes run is
the user's. Every deadline has a default, by the rule for defaults in
docs/reference/KNOB_GLOSSARY.md: a wrong one only fails loudly, one value suits every workload,
and none is unbounded.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from loops_cookbook.resources.api import ProviderDeadlines

IMAGE_BUILD = "image_build"
SANDBOX = "sandbox"


def _deadline(default: float, description: str) -> Any:
    return Field(default=default, gt=0, allow_inf_nan=False, description=description)


class BlaxelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    region: str = Field(
        min_length=1,
        description="The region every sandbox is created and every image is built in. Required.",
    )
    create_timeout_seconds: float = _deadline(
        120, "Bound on creating one sandbox, until Blaxel reports it deployed."
    )
    reconcile_timeout_seconds: float = _deadline(
        60, "Bound on looking an ambiguous create up by name, and on deciding prerequisites."
    )
    image_build_timeout_seconds: float = _deadline(
        420, "Bound on building one sandbox image, builder sandbox included."
    )
    image_build_memory_mb: int | None = Field(
        default=None,
        ge=1,
        le=32768,
        description="Memory of the environment each image is built in, in MB. Unset: "
        "Blaxel's default, which a large image can run out of mid-build. It sizes the "
        "build alone; a sandbox's memory is its task's.",
    )
    image_build_volume_mb: int | None = Field(
        default=None,
        ge=0,
        le=131072,
        description="Disk for each build's intermediate layers, in MB; 0 builds in memory. "
        "Unset: Blaxel's default, which a large image can run out of mid-build.",
    )
    release_timeout_seconds: float = _deadline(120, "Bound on deleting one sandbox.")
    confirm_absent_timeout_seconds: float = _deadline(
        60, "Bound on confirming a deleted sandbox is gone."
    )
    list_timeout_seconds: float = _deadline(
        60,
        "Bound on each page of the workspace listing a label sweep reads; the whole "
        "listing is not bounded, since it grows with the workspace.",
    )
    attach_timeout_seconds: float = _deadline(
        30, "Bound on attaching a client to a created sandbox."
    )
    labels: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="Labels every sandbox, image builders included, carries besides the run's "
        "ownership labels, such as an owner to find a team's sandboxes by. "
        "They win over the same keys from a Harbor job's or [runtime.config.sandbox]'s "
        "labels.",
    )

    @property
    def deadlines(self) -> ProviderDeadlines:
        return ProviderDeadlines(
            create_seconds=self.create_timeout_seconds,
            reconcile_seconds=self.reconcile_timeout_seconds,
            release_seconds=self.release_timeout_seconds,
            confirm_absent_seconds=self.confirm_absent_timeout_seconds,
            list_seconds=self.list_timeout_seconds,
            attach_seconds=self.attach_timeout_seconds,
            create_seconds_by_kind={IMAGE_BUILD: self.image_build_timeout_seconds},
        )
