"""The producers a run declares, and whether each one's generations train."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from loops_cookbook.model_access.api import UndeclaredProducer

DEFAULT_PRODUCER = "main"
"""The producer a call that names none uses."""


def _only_main() -> Mapping[str, bool]:
    return MappingProxyType({DEFAULT_PRODUCER: True})


@dataclass(frozen=True, slots=True)
class ProducerSettings:
    """Every producer a run declares, by name, and whether its generations train."""

    declared: Mapping[str, bool] = field(default_factory=_only_main)

    def __post_init__(self) -> None:
        if any(type(name) is not str or not name for name in self.declared):
            raise ValueError("producer names must be nonempty strings")
        if not any(self.declared.values()):
            raise ValueError("at least one declared producer must be trainable")
        object.__setattr__(self, "declared", MappingProxyType(dict(self.declared)))

    def resolve(self, producer: str | None) -> str:
        """The declared producer a call names, or the default when it names none."""
        name = DEFAULT_PRODUCER if producer is None else producer
        if type(name) is not str or name not in self.declared:
            raise UndeclaredProducer(
                f"producer {name!r} is not declared in [rollout.producers]; "
                f"declared: {sorted(self.declared)}"
            )
        return name

    def trainable(self, producer: str) -> bool:
        return self.declared[producer]
