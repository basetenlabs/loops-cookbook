"""A fake Daytona control plane behind the adapter's ``Api`` facade, and SDK-shaped errors.

It behaves as Daytona was observed to on 3 Oct 2026 (see ``providers/daytona/errors``):
names are unique, so a second create under a taken name answers 409; a deleted sandbox
reads ``destroying`` for a while, then 404 by id and by name, and its name is free.
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from types import SimpleNamespace as NS
from typing import Any

import aiohttp
from daytona.common.errors import DaytonaConnectionError, DaytonaError, create_daytona_error
from daytona_api_client_async.models.sandbox_state import SandboxState
from pydantic import SecretStr

from loops_cookbook.providers.daytona.adapter import Credentials
from loops_cookbook.providers.daytona.api import Observed
from loops_cookbook.providers.daytona.settings import DaytonaSettings

# The keys a [providers.daytona] table must state: the sandbox sizes and the ttl.
REQUIRED = {"cpu": 1, "memory_gib": 1, "disk_gib": 3, "ttl_minutes": 1440}
SETTINGS = DaytonaSettings(
    ttl_minutes=1440,
    create_timeout_seconds=2,
    reconcile_timeout_seconds=1,
    release_timeout_seconds=1,
    confirm_absent_timeout_seconds=1,
    list_timeout_seconds=1,
    attach_timeout_seconds=1,
    cpu=2,
    memory_gib=4,
    disk_gib=3,
    labels={"owner": "team-a"},
)
CREDENTIALS = Credentials(SecretStr("dtn-key-123"))
ENVIRON = {"DAYTONA_API_KEY": "dtn-key-123"}


def StatusError(status: int, message: str = "daytona said no") -> DaytonaError:
    """The ``DaytonaError`` subclass the SDK raises for an HTTP ``status``."""
    return create_daytona_error(message, status_code=status)


def never_connected(
    cause: type[aiohttp.ClientConnectorError] = aiohttp.ClientConnectorError,
) -> BaseException:
    """The SDK's transport error for a connection that was never made."""
    key = NS(host="app.daytona.io", port=443, ssl=True)
    error = DaytonaConnectionError("Failed to create sandbox: no route to host")
    error.__cause__ = cause(key, OSError("no route to host"))  # pyright: ignore[reportArgumentType]
    return error


@dataclass
class Sandbox:
    id: str
    name: str
    labels: dict[str, str]
    states: list[str]  # the state each later read reports; the last one sticks
    config: dict[str, Any] = field(default_factory=dict)

    def read(self) -> Observed:
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return Observed(self.id, self.name, dict(self.labels), state)


@dataclass
class FakeApi:
    sandboxes: dict[str, Sandbox] = field(default_factory=dict)  # by id
    calls: list[tuple[str, str]] = field(default_factory=list)
    start: list[str] = field(default_factory=lambda: ["started"])  # states after create
    create_errors: list[BaseException] = field(default_factory=list)
    land_before_error: bool = False  # a create error arrives after the sandbox exists
    get_errors: list[BaseException] = field(default_factory=list)
    delete_errors: list[BaseException] = field(default_factory=list)
    linger: int = 0  # reads that still see a deleted sandbox, as destroying
    create_gate: asyncio.Event | None = None  # creates wait for it before landing
    answer_gate: asyncio.Event | None = None  # creates land, then wait for it to answer
    created: list[dict[str, Any]] = field(default_factory=list)  # each landed create's config
    list_page_size: int = 0
    closed: bool = False
    ids: Any = field(default_factory=lambda: (f"id-{n}" for n in itertools.count(1)))

    def named(self, name: str) -> Sandbox | None:
        return next((box for box in self.sandboxes.values() if box.name == name), None)

    async def create(self, config, timeout):
        self.calls.append(("create", config["name"]))
        if self.create_gate is not None:
            await self.create_gate.wait()
        if self.named(config["name"]) is not None:
            raise StatusError(409, f"Sandbox with name {config['name']} already exists")
        if self.create_errors:
            error = self.create_errors.pop(0)
            if self.land_before_error:
                self._land(config)
            raise error
        landed = self._land(config)
        if self.answer_gate is not None:
            await self.answer_gate.wait()
        return landed.read()

    def _land(self, config):
        self.created.append(dict(config))
        box = Sandbox(next(self.ids), config["name"], dict(config["labels"]), list(self.start))
        box.config = dict(config)
        self.sandboxes[box.id] = box
        return box

    async def get(self, id_or_name):
        self.calls.append(("get", id_or_name))
        if self.get_errors:
            raise self.get_errors.pop(0)
        self.tick()
        box = self.sandboxes.get(id_or_name) or self.named(id_or_name)
        if box is None:
            raise StatusError(404, f"Sandbox with ID or name {id_or_name} not found")
        return box.read()

    def client(self):
        api = self

        class Client:
            async def get(self, id_or_name):
                observed = await api.get(id_or_name)
                state = SandboxState(observed.state)
                return NS(id=observed.id, state=state, labels=dict(observed.labels))

        return Client()

    async def delete(self, sandbox_id):
        self.calls.append(("delete", sandbox_id))
        if self.delete_errors:
            raise self.delete_errors.pop(0)
        if sandbox_id not in self.sandboxes:
            raise StatusError(404)
        box = self.sandboxes[sandbox_id]
        if self.linger:
            box.states = ["destroying"] * self.linger + ["GONE"]
            box.name = f"DESTROYED_{box.name}_1"  # as Daytona: the name is free at once
        else:
            del self.sandboxes[sandbox_id]

    async def list(self, labels):
        listed = [
            box.read()
            for box in self.sandboxes.values()
            if all(box.labels.get(k) == v for k, v in labels.items())
        ]
        size = self.list_page_size or max(len(listed), 1)
        for start in range(0, max(len(listed), 1), size):
            self.calls.append(("list", "*"))
            yield listed[start : start + size]

    def tick(self):
        for key, box in list(self.sandboxes.items()):
            if box.states == ["GONE"]:
                del self.sandboxes[key]

    async def close(self):
        self.closed = True


def sandbox_request(key="trial__env", name=None, **sandbox):
    from loops_cookbook.resources.api import ResourceRequest

    specification: dict[str, Any] = {
        "image": {"dockerfile": "FROM python:3.12-slim\n", "context": []},
        "labels": {"harbor.session_id": key},
        **sandbox,
    }
    specification = {key: value for key, value in specification.items() if value is not None}
    if name is not None:
        specification["name"] = name
    return ResourceRequest(
        provider="daytona",
        kind="sandbox",
        logical_key=key,
        specification={"sandbox": specification},
    )
