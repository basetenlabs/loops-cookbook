"""Runtime construction: the factory, its context, and preflight of what it declares."""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from loops_cookbook.evidence import JsonValue, TaskRef, json_value
from loops_cookbook.model_access.api import ModelAccessMode
from loops_cookbook.runtime.api import (
    GroupGrader,
    Phase,
    RolloutRuntime,
    RuntimeCapabilities,
    RuntimeContext,
    RuntimeFactory,
)


class RuntimeConfigurationError(ValueError):
    """The runtime or what it declares cannot run under this configuration."""


class RuntimeFileResolver:
    """Resolves exactly the aliases declared in ``[runtime.files]``."""

    def __init__(self, files: Mapping[str, Path]) -> None:
        self._files = dict(files)

    def path(self, alias: str) -> Path:
        try:
            return self._files[alias]
        except KeyError:
            raise RuntimeConfigurationError(
                f"[runtime.files] declares no alias {alias!r}; declared: {sorted(self._files)}"
            ) from None


class EvidenceRecord:
    """The run's ``RuntimeEvidence``: one ``sha256:`` digest per name."""

    DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

    def __init__(self) -> None:
        self._digests: dict[str, str] = {}

    def record_digest(self, name: str, digest: str) -> None:
        if type(name) is not str or not name:
            raise ValueError(f"runtime evidence names must be nonempty strings: {name!r}")
        if name in self._digests:
            raise ValueError(f"runtime evidence {name!r} is already recorded")
        if type(digest) is not str or not self.DIGEST.fullmatch(digest):
            raise ValueError(f"runtime evidence {name!r} must be a sha256: digest")
        self._digests[name] = digest

    @property
    def digests(self) -> dict[str, str]:
        return dict(self._digests)


class MetricsSink:
    """The typed runtime metrics sink: gauges and observations, namespaced.

    Metrics are reported, never read back by the cookbook; they have no effect on
    scheduling.
    """

    PREFIX = "runtime/"

    def __init__(self) -> None:
        self._gauges: dict[str, float] = {}
        self._observations: dict[str, tuple[int, float, float]] = {}

    def _name(self, name: str, value: float) -> str:
        if type(name) is not str or not name or name.startswith("/"):
            raise ValueError(f"runtime metric names must be nonempty relative names: {name!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"runtime metric {name!r} must be a number")
        if not math.isfinite(value):
            raise ValueError(f"runtime metric {name!r} must be finite")
        return self.PREFIX + name

    def set_gauge(self, name: str, value: float) -> None:
        self._gauges[self._name(name, value)] = float(value)

    def observe(self, name: str, value: float) -> None:
        key = self._name(name, value)
        count, total, maximum = self._observations.get(key, (0, 0.0, -math.inf))
        self._observations[key] = (count + 1, total + value, max(maximum, value))

    def snapshot(self) -> dict[str, float]:
        out = dict(self._gauges)
        for key, (count, total, maximum) in self._observations.items():
            out[f"{key}/count"] = float(count)
            out[f"{key}/mean"] = total / count
            out[f"{key}/max"] = maximum
        return out


def import_attribute(path: str, what: str) -> object:
    """The object ``module:attribute`` names; ``what`` says what it is, in errors."""
    module_name, _, attribute = path.partition(":")
    if not module_name or not attribute:
        raise RuntimeConfigurationError(f"{what} must be 'module:attribute', got {path!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeConfigurationError(
            f"could not import {what} module {module_name!r}: {error}"
        ) from error
    found = getattr(module, attribute, None)
    if found is None:
        raise RuntimeConfigurationError(f"module {module_name!r} has no {what} {attribute!r}")
    return found


def import_factory(path: str) -> RuntimeFactory:
    factory = import_attribute(path, "runtime factory")
    if not callable(factory):
        raise RuntimeConfigurationError(f"{path!r} is not a callable runtime factory")
    return factory  # pyright: ignore[reportReturnType]


def build_runtime(
    factory: RuntimeFactory, config: Mapping[str, JsonValue], context: RuntimeContext
) -> RolloutRuntime:
    """Construct the runtime from a private deep copy of exactly the user's values."""
    private = json_value(deepcopy(dict(config)), where="[runtime.config]")
    if not isinstance(private, dict):
        raise RuntimeConfigurationError("[runtime.config] must be a table")
    return factory(private, context=context)


@dataclass(frozen=True, slots=True)
class Preflight:
    # [runtime] model_access as configured; None when the runtime's declaration decides.
    model_access: ModelAccessMode | None
    rollout_timeout_seconds: float | None


def model_access_mode(
    configured: ModelAccessMode | None, capabilities: RuntimeCapabilities
) -> ModelAccessMode:
    """The run's model access: the runtime's one mode, or the one configured among several."""
    modes = sorted(capabilities.model_access_modes, key=lambda mode: mode.value)
    if configured is None:
        if len(modes) != 1:
            raise RuntimeConfigurationError(
                "[runtime] model_access is required: the runtime supports "
                f"{[mode.value for mode in modes]}"
            )
        return modes[0]
    if len(modes) == 1:
        raise RuntimeConfigurationError(
            f"[runtime] model_access is set, and the runtime already declares "
            f"{modes[0].value!r}; the runtime states it once, so remove the setting"
        )
    if configured not in modes:
        raise RuntimeConfigurationError(
            f"[runtime] model_access = {configured.value!r}, but the runtime supports "
            f"{[mode.value for mode in modes]}"
        )
    return configured


def rollout_clock(configured: float | None, capabilities: RuntimeCapabilities) -> float:
    """The rollout clock the run keeps: the one configured, else the runtime's own."""
    clock = configured if configured is not None else capabilities.rollout_timeout_seconds
    if clock is None:
        raise RuntimeConfigurationError(
            "[rollout] rollout_timeout_seconds is required: this runtime derives no rollout "
            "clock of its own"
        )
    return clock


def check_capabilities(capabilities: object, preflight: Preflight) -> RuntimeCapabilities:
    """Validate what the runtime declared, before any attempt is admitted."""
    if not isinstance(capabilities, RuntimeCapabilities):
        raise RuntimeConfigurationError(
            f"runtime capabilities must be RuntimeCapabilities, not {type(capabilities).__name__}"
        )
    modes = capabilities.model_access_modes
    phases = capabilities.internally_acquired_phases
    if not isinstance(modes, frozenset) or not all(  # pyright: ignore[reportUnnecessaryIsInstance]
        isinstance(mode, ModelAccessMode)  # pyright: ignore[reportUnnecessaryIsInstance]
        for mode in modes
    ):
        raise RuntimeConfigurationError("model_access_modes must be a frozenset of ModelAccessMode")
    if not isinstance(phases, frozenset) or not all(  # pyright: ignore[reportUnnecessaryIsInstance]
        isinstance(phase, Phase)  # pyright: ignore[reportUnnecessaryIsInstance]
        for phase in phases
    ):
        raise RuntimeConfigurationError("internally_acquired_phases must be a frozenset of Phase")
    retries = capabilities.grading_retries
    if type(retries) is not int or retries < 0:
        raise RuntimeConfigurationError("grading_retries must be a nonnegative integer")
    if type(capabilities.pending_group_grade) is not bool:
        raise RuntimeConfigurationError("pending_group_grade must be a boolean")
    derived = capabilities.rollout_timeout_seconds
    if derived is not None and (
        type(derived) not in (int, float) or not math.isfinite(derived) or derived <= 0
    ):
        raise RuntimeConfigurationError("rollout_timeout_seconds must be finite and positive")
    if Phase.PROVISIONING in phases:
        raise RuntimeConfigurationError(
            "provisioning capacity is acquired only by the cookbook's resource service"
        )
    model_access_mode(preflight.model_access, capabilities)
    rollout_clock(preflight.rollout_timeout_seconds, capabilities)
    return capabilities


def group_grader(runtime: RolloutRuntime, capabilities: RuntimeCapabilities) -> GroupGrader | None:
    """The runtime as its own group grader, when it declares ``pending_group_grade``."""
    if not capabilities.pending_group_grade:
        return None
    if not callable(getattr(runtime, "grade_group", None)):
        raise RuntimeConfigurationError(
            "the runtime declares pending_group_grade but defines no grade_group()"
        )
    return runtime  # pyright: ignore[reportReturnType]


def check_tasks(tasks: object) -> list[TaskRef]:
    """The prepared task list: TaskRefs whose repeated IDs agree on their digest."""
    if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
        raise RuntimeConfigurationError("prepare() must return a sequence of TaskRef")
    entries: Sequence[object] = tasks  # pyright: ignore[reportUnknownVariableType]
    result: list[TaskRef] = []
    digests: dict[str, str] = {}
    for task in entries:
        if not isinstance(task, TaskRef):
            raise RuntimeConfigurationError(
                f"prepare() returned a {type(task).__name__}, not a TaskRef"
            )
        if digests.setdefault(task.id, task.digest) != task.digest:
            raise RuntimeConfigurationError(f"task {task.id!r} was prepared with two digests")
        result.append(task)
    if not result:
        raise RuntimeConfigurationError("prepare() returned no tasks")
    return result
