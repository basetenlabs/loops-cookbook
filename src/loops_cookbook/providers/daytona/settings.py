"""`[providers.daytona]`: the target, every Daytona deadline, sandbox defaults, and limits.

Credentials are not settings: `DAYTONA_API_KEY` (and optionally `DAYTONA_API_URL`)
come from the environment, loaded from `.env`. By the rule for defaults in
docs/reference/KNOB_GLOSSARY.md, the sandbox sizes and ``ttl_minutes`` are required,
as a wrong one quietly fails or kills live work; a wrong deadline only fails loudly,
and the idle lifecycle defaults make a leaked sandbox stop and delete itself.

Daytona sizes CPU, memory, and disk independently, in whole vCPUs and GiB (at least
1 each), unlike Blaxel, where memory sets all three. They are dedicated: cgroups
enforce the CPU and memory exactly, with no swap. The sizes a request leaves out are
filled from here, so every sandbox built from an image names all three. Only a
running sandbox counts against the organization's CPU and memory; a stopped one still
holds its disk until it is deleted or archived. The per-sandbox limits are optional:
when set, a Harbor task with an environment larger than one sandbox may have is
refused before the run starts.

Daytona's own lifecycle defaults stop a sandbox after 15 idle minutes and never delete
it; Harbor's never stop it at all. Here a sandbox is stopped after ``auto_stop_minutes``
without a command (a command run through Daytona counts as activity; a process
running inside the sandbox does not), deleted as it stops, and destroyed at
``ttl_minutes`` whatever it is doing, so a sandbox the run loses always ends.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from loops_cookbook.resources.api import ProviderDeadlines

SANDBOX = "sandbox"


def lifecycle_minutes(value: object) -> bool:
    """Whether Daytona takes ``value`` as a lifecycle interval: whole minutes, at least 0."""
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
        description="The Daytona target (region), such as us or eu, of every sandbox and every "
        "Daytona call, Harbor's included. Unset, the organization's default target. A "
        "DAYTONA_TARGET in the environment that differs is refused.",
    )
    create_timeout_seconds: float = _deadline(
        600,
        "Bound on creating one sandbox until Daytona reports it started, a first build of "
        "its declarative image included.",
    )
    reconcile_timeout_seconds: float = _deadline(
        60, "Bound on looking an ambiguous create up by name."
    )
    release_timeout_seconds: float = _deadline(120, "Bound on deleting one sandbox.")
    confirm_absent_timeout_seconds: float = _deadline(
        60, "Bound on confirming a deleted sandbox is gone."
    )
    list_timeout_seconds: float = _deadline(
        60, "Bound on each page of the labelled listing a label sweep reads."
    )
    attach_timeout_seconds: float = _deadline(
        30, "Bound on attaching a client to a created sandbox."
    )
    cpu: int = Field(gt=0, description="vCPUs of a sandbox whose request names none.")
    memory_gib: int = Field(
        gt=0, description="Memory in GiB of a sandbox whose request names none."
    )
    disk_gib: int = Field(
        gt=0,
        description="Disk in GiB of a sandbox whose request names none: its writable layer "
        "only, as the image is not counted.",
    )
    thread_variables_from_vcpus: bool = Field(
        default=False,
        description="Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to a "
        "Harbor sandbox's vCPUs for every command, as nproc in a sandbox counts its host's "
        "cores. Harbor passes them with each command, so they override the image's ENV; the "
        "task's [environment.env] wins.",
    )
    auto_stop_minutes: int = Field(
        default=120,
        ge=0,
        description="Minutes without a Daytona API call (a command counts; a process inside "
        "does not) after which Daytona stops a sandbox whose request sets none; with "
        "auto_delete_minutes 0 it is then deleted. It must outlast an attempt's longest quiet "
        "spell, such as a long model turn or a wait for grading. 0 disables it, which needs "
        "ttl_minutes.",
    )
    auto_delete_minutes: int = Field(
        default=0,
        ge=0,
        description="Minutes a stopped sandbox is kept before Daytona deletes it; 0 deletes it "
        "as it stops.",
    )
    ttl_minutes: int = Field(
        ge=0,
        description="Wall-clock minutes after which Daytona destroys a sandbox whatever its "
        "state: a backstop for one the run lost. It must outlast the longest attempt, as a "
        "sandbox still working at it is destroyed; 0 sets none.",
    )
    labels: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="Labels every sandbox carries besides the run's ownership labels, such as "
        "an owner to find a team's sandboxes by. They win over the same keys from a Harbor "
        "job's or [rollout_runtime.config.sandbox]'s labels.",
    )
    max_sandbox_cpu: int | None = _limit(
        "The most vCPUs Daytona gives one sandbox; a larger task is refused before the run."
    )
    max_sandbox_memory_gib: int | None = _limit("The most memory one sandbox may have, in GiB.")
    max_sandbox_disk_gib: int | None = _limit("The most disk one sandbox may have, in GiB.")

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
