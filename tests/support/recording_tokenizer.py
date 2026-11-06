"""A tokenizer whose backend records every change made to it, to prove shared use reads only."""

from __future__ import annotations

from typing import Any

# The backend's settings an encode may change for the length of a call.
SETTERS = frozenset({"no_truncation", "enable_truncation", "no_padding", "enable_padding"})


class RecordingBackend:
    """Stands in for a ``tokenizers.Tokenizer``: delegates everything, records writes."""

    def __init__(self, backend: Any) -> None:
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "writes", [])

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._backend, name)
        if name in SETTERS:

            def recorded(*args: Any, **kwargs: Any) -> Any:
                self.writes.append(name)
                return value(*args, **kwargs)

            return recorded
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self.writes.append(name)
        setattr(self._backend, name, value)


def record_writes(tokenizer: Any) -> list[str]:
    """Swap ``tokenizer``'s backend for a recording one; return its list of writes."""
    backend = RecordingBackend(tokenizer._tokenizer)
    tokenizer._tokenizer = backend
    return backend.writes
