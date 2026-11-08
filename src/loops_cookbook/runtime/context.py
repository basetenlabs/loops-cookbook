"""Build the rollout runtime from its factory and context, and check what it declares."""

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
    """Raised when the rollout runtime, or what it declares, cannot run under this configuration."""


class RuntimeFileResolver:
    """Resolve exactly the aliases declared in ``[rollout_runtime.files]``."""

    def __init__(self, files: Mapping[str, Path]) -> None:
        self._files = dict(files)

    def path(self, alias: str) -> Path:
        try:
            return self._files[alias]
        except KeyError:
            raise RuntimeConfigurationError(
                f"[rollout_runtime.files] declares no alias {alias!r}; declared: {sorted(self._files)}"
            ) from None


class EvidenceRecord:
    """Implement the run's ``RuntimeEvidence``, which holds one ``sha256:`` digest per name."""

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
    """Collect the rollout runtime's typed metrics, which are namespaced gauges and observations.

    The cookbook reports these metrics and never reads them back, so they have no effect
    on scheduling.
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
    """Return the object that ``module:attribute`` names.

    ``what`` says what the object is, for error messages.
    """
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
    factory = import_attribute(path, "rollout runtime factory")
    if not callable(factory):
        raise RuntimeConfigurationError(f"{path!r} is not a callable rollout runtime factory")
    return factory  # pyright: ignore[reportReturnType]


def build_runtime(
    factory: RuntimeFactory, config: Mapping[str, JsonValue], context: RuntimeContext
) -> RolloutRuntime:
    """Build the rollout runtime from a private deep copy of exactly the user's values."""
    private = json_value(deepcopy(dict(config)), where="[rollout_runtime.config]")
    if not isinstance(private, dict):
        raise RuntimeConfigurationError("[rollout_runtime.config] must be a table")
    return factory(private, context=context)


@dataclass(frozen=True, slots=True)
class Preflight:
    # [rollout_runtime] model_access as configured. It is None when the rollout runtime's
    # declaration decides.
    model_access: ModelAccessMode | None
    rollout_timeout_seconds: float | None


def model_access_mode(
    configured: ModelAccessMode | None, capabilities: RuntimeCapabilities
) -> ModelAccessMode:
    """Return the run's model access.

    It is the rollout runtime's one mode, or the configured one when the rollout runtime
    supports several.
    """
    modes = sorted(capabilities.model_access_modes, key=lambda mode: mode.value)
    if configured is None:
        if len(modes) != 1:
            raise RuntimeConfigurationError(
                "[rollout_runtime] model_access is required: the rollout runtime supports "
                f"{[mode.value for mode in modes]}"
            )
        return modes[0]
    if len(modes) == 1:
        raise RuntimeConfigurationError(
            f"[rollout_runtime] model_access is set, but the rollout runtime already declares "
            f"{modes[0].value!r}; the rollout runtime states it once, so remove the setting"
        )
    if configured not in modes:
        raise RuntimeConfigurationError(
            f"[rollout_runtime] model_access = {configured.value!r}, but the rollout runtime "
            "supports "
            f"{[mode.value for mode in modes]}"
        )
    return configured


def rollout_clock(configured: float | None, capabilities: RuntimeCapabilities) -> float:
    """Return the rollout clock the run keeps.

    It is the configured clock, or else the rollout runtime's own.
    """
    clock = configured if configured is not None else capabilities.rollout_timeout_seconds
    if clock is None:
        raise RuntimeConfigurationError(
            "[rollout] rollout_timeout_seconds is required: this rollout runtime derives no "
            "rollout clock of its own"
        )
    return clock


def check_capabilities(capabilities: object, preflight: Preflight) -> RuntimeCapabilities:
    """Validate what the rollout runtime declared, before any attempt is admitted."""
    if not isinstance(capabilities, RuntimeCapabilities):
        raise RuntimeConfigurationError(
            f"rollout runtime capabilities must be RuntimeCapabilities, not "
            f"{type(capabilities).__name__}"
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
    """Return the rollout runtime as its group grader, if it declares ``pending_group_grade``."""
    if not capabilities.pending_group_grade:
        return None
    if not callable(getattr(runtime, "grade_group", None)):
        raise RuntimeConfigurationError(
            "the rollout runtime declares pending_group_grade but defines no grade_group()"
        )
    return runtime  # pyright: ignore[reportReturnType]


def check_tasks(tasks: object) -> list[TaskRef]:
    """Return the prepared task list after checking it.

    It must hold TaskRefs. A task ID appears as often as the rollout runtime lists it.
    """
    if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
        raise RuntimeConfigurationError("prepare() must return a sequence of TaskRef")
    entries: Sequence[object] = tasks  # pyright: ignore[reportUnknownVariableType]
    result: list[TaskRef] = []
    for task in entries:
        if not isinstance(task, TaskRef):
            raise RuntimeConfigurationError(
                f"prepare() returned a {type(task).__name__}, not a TaskRef"
            )
        result.append(task)
    if not result:
        raise RuntimeConfigurationError("prepare() returned no tasks")
    return result
