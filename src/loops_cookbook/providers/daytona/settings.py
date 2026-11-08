"""The `[providers.daytona]` settings, which hold the target, deadlines, sizes and limits.

Credentials are not settings. `DAYTONA_API_KEY` comes from the environment, loaded
from `.env`, and so does `DAYTONA_API_URL` when it is set. The sandbox sizes and
``ttl_minutes`` are required, by the rule for defaults in
docs/reference/KNOB_GLOSSARY.md, because a wrong value quietly fails or kills live
work. A wrong deadline only fails loudly, so each deadline has a default. The idle
lifecycle defaults make a leaked sandbox stop and delete itself.

Daytona sizes CPU, memory and disk independently, in whole vCPUs and GiB, with at
least 1 of each. Blaxel differs, because there memory sets all three. Daytona's sizes
are dedicated. Cgroups enforce the CPU and memory exactly, with no swap. This table
fills in the sizes a request leaves out, so every sandbox built from an image names
all three. Only a running sandbox counts against the organization's CPU and memory.
A stopped sandbox still holds its disk until it is deleted or archived. The
per-sandbox limits are optional. When they are set, the cookbook refuses a Harbor task
whose environment is larger than one sandbox may be, before the run starts.

Daytona's own lifecycle defaults stop a sandbox after 15 idle minutes and never delete
it. Harbor's defaults never stop it at all. The cookbook instead asks Daytona to stop a
sandbox after ``auto_stop_minutes`` without a command, and to delete it as it stops. A command run
through Daytona counts as activity, but a process running inside the sandbox does not.
Daytona also destroys the sandbox at ``ttl_minutes``, whatever it is doing. Together
these make sure that a sandbox the run loses always ends.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from loops_cookbook.resources.api import ProviderDeadlines

SANDBOX = "sandbox"


def lifecycle_minutes(value: object) -> bool:
    """Return whether Daytona accepts ``value`` as a lifecycle interval.

    An interval is a whole number of minutes, at least 0.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _deadline(default: float, description: str) -> Any:
    return Field(default=default, gt=0, allow_inf_nan=False, description=description)


def _limit(description: str) -> Any:
    return Field(default=None, gt=0, description=description)


class DaytonaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str | None = Field(
        default=None,
        min_length=1,
        description="The Daytona target, meaning the region, such as us or eu. Every sandbox "
        "and every Daytona call uses it, including Harbor's. When it is unset, the "
        "organization's default target applies. The cookbook refuses a DAYTONA_TARGET in the "
        "environment that differs from it.",
    )
    create_timeout_seconds: float = _deadline(
        600,
        "The longest the run waits for Daytona to create one sandbox and report it started. "
        "The time includes a first build of the sandbox's declarative image.",
    )
    reconcile_timeout_seconds: float = _deadline(
        60, "The longest the run spends looking up an ambiguous create by name."
    )
    release_timeout_seconds: float = _deadline(
        120, "The longest the run waits for one sandbox to be deleted."
    )
    confirm_absent_timeout_seconds: float = _deadline(
        60, "The longest the run spends confirming that a deleted sandbox is gone."
    )
    list_timeout_seconds: float = _deadline(
        60,
        "The longest the run waits for each page of the labelled listing that a label sweep reads.",
    )
    attach_timeout_seconds: float = _deadline(
        30, "The longest the run spends attaching a client to a created sandbox."
    )
    cpu: int = Field(gt=0, description="The vCPUs of a sandbox whose request names none.")
    memory_gib: int = Field(
        gt=0, description="The memory, in GiB, of a sandbox whose request names none."
    )
    disk_gib: int = Field(
        gt=0,
        description="The disk, in GiB, of a sandbox whose request names none. It covers only "
        "the sandbox's writable layer, because the image is not counted.",
    )
    thread_variables_from_vcpus: bool = Field(
        default=False,
        description="Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to a Harbor "
        "sandbox's vCPUs for every command. This matters because nproc in a sandbox counts "
        "its host's cores. Harbor passes these variables with each command, so they override "
        "the image's ENV. The task's [environment.env] still takes precedence.",
    )
    auto_stop_minutes: int = Field(
        default=120,
        ge=0,
        description="The minutes without a Daytona API call after which Daytona stops a "
        "sandbox whose request sets none. A command counts as a call, but a process inside "
        "the sandbox does not. With auto_delete_minutes 0, Daytona then deletes the sandbox. "
        "It must outlast an attempt's longest quiet spell, such as a long model turn or a "
        "wait for grading. 0 disables it, and then ttl_minutes must be set.",
    )
    auto_delete_minutes: int = Field(
        default=0,
        ge=0,
        description="The minutes Daytona keeps a stopped sandbox before deleting it. 0 "
        "deletes the sandbox as it stops.",
    )
    ttl_minutes: int = Field(
        ge=0,
        description="The wall-clock minutes after which Daytona destroys a sandbox, whatever "
        "its state. It is a backstop for a sandbox the run lost. It must outlast the longest "
        "attempt, because Daytona destroys a sandbox that is still working when the time "
        "passes. 0 sets no limit.",
    )
    labels: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="Labels that every sandbox carries in addition to the run's ownership "
        "labels. For example, an owner label lets a team find its sandboxes. These labels "
        "override the same keys from a Harbor job's labels or from "
        "[rollout_runtime.config.sandbox] labels.",
    )
    max_sandbox_cpu: int | None = _limit(
        "The most vCPUs Daytona gives one sandbox. The cookbook refuses a larger task before "
        "the run starts."
    )
    max_sandbox_memory_gib: int | None = _limit(
        "The most memory, in GiB, that one sandbox may have."
    )
    max_sandbox_disk_gib: int | None = _limit("The most disk, in GiB, that one sandbox may have.")

    @model_validator(mode="after")
    def _a_leak_must_die(self) -> DaytonaSettings:
        if self.auto_stop_minutes == 0 and self.ttl_minutes == 0:
            raise ValueError(
                "auto_stop_minutes 0 with ttl_minutes 0 lets a leaked sandbox run forever; "
                "set one of them"
            )
        return self

    @property
    def deadlines(self) -> ProviderDeadlines:
        return ProviderDeadlines(
            create_seconds=self.create_timeout_seconds,
            reconcile_seconds=self.reconcile_timeout_seconds,
            release_seconds=self.release_timeout_seconds,
            confirm_absent_seconds=self.confirm_absent_timeout_seconds,
            list_seconds=self.list_timeout_seconds,
            attach_seconds=self.attach_timeout_seconds,
        )
