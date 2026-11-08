"""The `[providers.blaxel]` settings, which hold the region, the Blaxel deadlines and labels.

Credentials are not settings. `BL_WORKSPACE` and `BL_API_KEY` come from the
environment, loaded from `.env`. The region is required, because where sandboxes run
is the user's choice. Every deadline has a default, which follows the rule for
defaults in docs/reference/KNOB_GLOSSARY.md. A wrong deadline only fails loudly, one
value suits every workload, and no deadline is unbounded.
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
        description="The region where the run creates every sandbox and builds every "
        "image. It is required.",
    )
    create_timeout_seconds: float = _deadline(
        120, "The longest the run waits for Blaxel to create one sandbox and report it deployed."
    )
    reconcile_timeout_seconds: float = _deadline(
        60,
        "The longest the run spends looking up an ambiguous create by name. It also bounds "
        "working out what a sandbox needs created first, such as its image.",
    )
    image_build_timeout_seconds: float = _deadline(
        420,
        "The longest the run waits for one sandbox image to build. The time includes the "
        "builder sandbox.",
    )
    image_build_memory_mb: int | None = Field(
        default=None,
        ge=1,
        le=32768,
        description="The memory, in MB, of the environment that builds each image. When it "
        "is unset, Blaxel uses its default, which a large image can run out of mid-build. "
        "It sizes only the build. A sandbox's memory comes from its task.",
    )
    image_build_volume_mb: int | None = Field(
        default=None,
        ge=0,
        le=131072,
        description="The disk, in MB, for each build's intermediate layers. 0 builds in "
        "memory. When it is unset, Blaxel uses its default, which a large image can run "
        "out of mid-build.",
    )
    release_timeout_seconds: float = _deadline(
        120, "The longest the run waits for one sandbox to be deleted."
    )
    confirm_absent_timeout_seconds: float = _deadline(
        60, "The longest the run spends confirming that a deleted sandbox is gone."
    )
    list_timeout_seconds: float = _deadline(
        60,
        "The longest the run waits for each page of the workspace listing that a label "
        "sweep reads. The whole listing has no bound, because it grows with the workspace.",
    )
    attach_timeout_seconds: float = _deadline(
        30, "The longest the run spends attaching a client to a created sandbox."
    )
    labels: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="Labels that every sandbox carries in addition to the run's ownership "
        "labels. Image builders carry them too. For example, an owner label lets a team find "
        "its sandboxes. These labels override the same keys from a Harbor job's labels or "
        "from [rollout_runtime.config.sandbox] labels.",
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
