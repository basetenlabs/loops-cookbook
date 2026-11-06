"""The cancellation matrix: what lands at each await of a stop, and what must follow.

A caller that stops work -- the driver's hooks, a seal, the judge stage -- declares
its await points and what can land at each, and gets one ``Case`` per combination:

    @cancellation_matrix(
        triggers=(Trigger.CANCEL, Trigger.DEADLINE),
        points={"kill": ALL_LANDINGS, "grace": (Landing.NOTHING, Landing.CANCEL)},
        responses=tuple(Response),
    )
    async def test_...(case): ...

The test plays ``case`` against its caller, with a ``Work`` as the stopped work and a
``Gate`` at each await point it can stall, then asserts the case's properties:

- ``case.still_running``: the work ignored its cancellation. Everything it holds
  stays held and it is reported as unquiesced; otherwise nothing is kept.
- ``case.outcome``: an outside cancellation, the first or a second, always
  propagates, with the report of work still running as its cause when there is one;
  otherwise a caller whose work is still running reports that; otherwise the
  caller's own outcome for the trigger (its deadline, or the step's failure).
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pytest


class Trigger(Enum):
    """What starts the stop."""

    CANCEL = "cancelled"  # the caller is cancelled from outside, as on shutdown
    DEADLINE = "deadline"  # the caller's own deadline expires


class Landing(Enum):
    """What lands at one await point while the caller is stopping its work."""

    NOTHING = "clean"
    CANCEL = "recancelled"  # the caller is cancelled again
    FAILURE = "failed"  # the awaited step fails


ALL_LANDINGS = tuple(Landing)


class Response(Enum):
    """How the stopped work takes its cancellation."""

    STOPS = "stops"
    FAILS = "fails"  # raises an error of its own while stopping
    IGNORES = "ignores"  # keeps running past any grace


class Outcome(Enum):
    NON_QUIESCENT = "non-quiescent"
    CANCELLED = "cancelled"
    OWN = "own"  # the caller's own outcome for its trigger


@dataclass(frozen=True, slots=True)
class Case:
    trigger: Trigger
    landings: tuple[tuple[str, Landing], ...]
    response: Response

    def at(self, point: str) -> Landing:
        return dict(self.landings).get(point, Landing.NOTHING)

    @property
    def still_running(self) -> bool:
        return self.response is Response.IGNORES

    @property
    def cancelled(self) -> bool:
        """The caller was cancelled from outside, once or more."""
        return self.trigger is Trigger.CANCEL or any(
            landing is Landing.CANCEL for _, landing in self.landings
        )

    @property
    def outcome(self) -> Outcome:
        if self.cancelled:
            return Outcome.CANCELLED
        if self.still_running:
            return Outcome.NON_QUIESCENT
        return Outcome.OWN

    @property
    def id(self) -> str:
        parts = [self.trigger.value, *(f"{p}-{landing.value}" for p, landing in self.landings)]
        return "/".join([*parts, f"work-{self.response.value}"])


def cases(
    *,
    triggers: Sequence[Trigger],
    points: Mapping[str, Sequence[Landing]],
    responses: Sequence[Response],
) -> list[Case]:
    """Every combination of a trigger, one landing per await point, and a response."""
    names = list(points)
    return [
        Case(trigger, tuple(zip(names, landings, strict=True)), response)
        for trigger in triggers
        for landings in itertools.product(*(points[name] for name in names))
        for response in responses
    ]


def cancellation_matrix(
    *,
    triggers: Sequence[Trigger],
    points: Mapping[str, Sequence[Landing]],
    responses: Sequence[Response],
) -> pytest.MarkDecorator:
    """Parametrise a test's ``case`` over the matrix."""
    table = cases(triggers=triggers, points=points, responses=responses)
    return pytest.mark.parametrize("case", table, ids=[case.id for case in table])


class WorkFailed(RuntimeError):
    """What ``Work`` raises when its response is to fail."""


@dataclass
class Work:
    """Stopped work that runs until cancelled, then responds as its case says.

    It waits for ``let_go`` before stopping or failing, so a test can land a second
    cancellation while it is still stopping; one that ignores cancellation runs on
    until ``release``. ``cancellations`` counts every CancelledError it received.
    ``end()`` lets it finish whatever state it is in, for a test's teardown.
    """

    response: Response
    running: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    let_go: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    cancellations: int = 0
    finished: bool = False

    async def __call__(self, *args: object) -> None:
        self.running.set()
        try:
            try:
                await self.release.wait()
                return  # released at teardown without ever being cancelled
            except asyncio.CancelledError:
                self.cancellations += 1
                self.cancelled.set()
            if self.response is Response.IGNORES:
                while not self.release.is_set():
                    try:
                        await self.release.wait()
                    except asyncio.CancelledError:
                        self.cancellations += 1
                return
            await self.let_go.wait()
            if self.response is Response.FAILS:
                raise WorkFailed("failed while stopping")
            raise asyncio.CancelledError
        finally:
            self.finished = True

    def end(self) -> None:
        self.let_go.set()
        self.release.set()


@dataclass
class Gate:
    """An await point a test can stall, then pass or fail."""

    failure: BaseException
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    _open: asyncio.Event = field(default_factory=asyncio.Event)
    _fail: bool = False

    async def passage(self) -> None:
        """Await this in the step being gated."""
        self.entered.set()
        await self._open.wait()
        if self._fail:
            raise self.failure

    def open(self, landing: Landing) -> None:
        self._fail = landing is Landing.FAILURE
        self._open.set()


async def outcome_within(task: asyncio.Future[Any], seconds: float) -> object:
    """``task``'s result or raised error, or a test failure once ``seconds`` pass.

    Not ``await task`` under a timeout: the timeout's cancellation would reach the
    task, and a task that waits out cancellations, as the code under test does,
    would hang the test instead of failing it.
    """
    done, _ = await asyncio.wait({task}, timeout=seconds)
    if not done:
        pytest.fail(f"{task!r} did not finish within {seconds:g}s")
    try:
        return task.result()
    except BaseException as error:  # noqa: BLE001 - the outcome under test
        return error


async def settle_loop(rounds: int = 5) -> None:
    """Let every ready callback run: a cancellation just requested is delivered."""
    for _ in range(rounds):
        await asyncio.sleep(0)
