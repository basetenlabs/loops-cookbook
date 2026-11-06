"""Phase capacity: start admission, sampling and verification permits, and clocks.

The broker's holds are the pipeline's phase observations. Every hold is bound to
one attempt and attributed to whichever of that attempt's hooks is running when it
is taken; the attempt's rollout and setup clocks are derived from the same holds.
The one exception is a group grader's call, which grades a whole group rather than
an attempt: it takes a run-level verification hold.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import Protocol

from loops_cookbook.evidence import AttemptIdentity
from loops_cookbook.runtime.api import Phase


class RuntimeContractError(RuntimeError):
    """A runtime used a cookbook service outside its contract."""


class CapacityRevoked(RuntimeError):
    """Capacity asked for by a hook kept running past its grace: its rollout is discarded.

    Raised in the hook's own task; the run goes on.
    """


class Hook(str, Enum):
    SAMPLE = "sample"
    GRADE = "grade"


class StartAdmission(Protocol):
    """The pipeline's start rule, consulted when an attempt is admitted.

    ``order`` places a group among the others: lower is dispatched earlier.
    """

    def target(self, group_id: str) -> int: ...
    def started(self, group_id: str) -> None: ...
    def order(self, group_id: str) -> int: ...


@dataclass(slots=True)
class AttemptPhases:
    """One attempt's phase state: open holds, the running hook, and its clocks."""

    identity: AttemptIdentity
    declared: frozenset[Phase]
    clock: Callable[[], float] = time.monotonic
    hook: Hook | None = None
    sealed: bool = False
    admitted: bool = False
    sampling_entered: bool = False
    holding: dict[Phase, int] = field(default_factory=dict[Phase, int])
    waiting: dict[Phase, int] = field(default_factory=dict[Phase, int])
    open_holds: set[_Hold] = field(default_factory=set["_Hold"])
    contract_error: RuntimeContractError | None = None
    on_violation: Callable[[RuntimeContractError], None] | None = None
    kept: bool = False  # its hook ignored cancellation past its grace; its rollout is discarded
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    _sampling: bool = False
    _resource_depth: int = 0
    _rollout_total: float = 0.0
    _rollout_since: float | None = None
    _excluded_depth: int = 0
    _excluded_total: float = 0.0
    _excluded_since: float | None = None
    _wall_since: float | None = None

    def violate(self, message: str) -> RuntimeContractError:
        error = RuntimeContractError(message)
        if self.contract_error is None:
            self.contract_error = error
            if self.on_violation is not None:
                self.on_violation(error)
        self.changed.set()
        return error

    def refuse(self, what: str) -> CapacityRevoked:
        return CapacityRevoked(
            f"{what} refused for {self.identity.rollout_id}: its hook was kept running past "
            "its grace, and its rollout is discarded"
        )

    def ended(self, what: str, violation: str) -> RuntimeError:
        """A capacity call once the hook has ended: refused if it was kept, else a violation."""
        return self.refuse(what) if self.kept else self.violate(violation)

    def start_hook(self, hook: Hook) -> None:
        self.hook = hook
        self._wall_since = self.clock()
        self._excluded_total = 0.0
        self._excluded_since = self.clock() if self._excluded_depth else None

    def end_hook(self) -> None:
        self.hook = None

    def _exclude(self, delta: int) -> None:
        now = self.clock()
        before = self._excluded_depth
        self._excluded_depth += delta
        if before == 0 and self._excluded_depth > 0:
            self._excluded_since = now
        elif before > 0 and self._excluded_depth == 0 and self._excluded_since is not None:
            self._excluded_total += now - self._excluded_since
            self._excluded_since = None

    def _retime(self) -> None:
        """Run the rollout clock while a sampling hold is open and no resource operation is."""
        now = self.clock()
        running = self._sampling and self._resource_depth == 0
        if running and self._rollout_since is None:
            self._rollout_since = now
        elif not running and self._rollout_since is not None:
            self._rollout_total += now - self._rollout_since
            self._rollout_since = None

    def open_sampling(self) -> None:
        self._sampling = True
        self._retime()

    def close_sampling(self) -> None:
        self._sampling = False
        self._retime()

    @contextlib.contextmanager
    def resource_operation(self) -> Iterator[None]:
        """Time the resource service spends for this attempt, left out of its clocks.

        Inside a sampling hold too: a runner-owned hold spans all of ``sample()``,
        so a sandbox create there would otherwise count as the agent's time.
        """
        self._exclude(1)
        self._resource_depth += 1
        self._retime()
        self.changed.set()  # the clocks stopped: a deadline wait re-reads them
        try:
            yield
        finally:
            self._resource_depth -= 1
            self._retime()
            self._exclude(-1)
            self.changed.set()

    def excluded_seconds(self) -> float:
        """Time in holds, permit waits, and resource operations since the last hook started."""
        open_for = self.clock() - self._excluded_since if self._excluded_since is not None else 0.0
        return self._excluded_total + open_for

    def rollout_elapsed(self) -> float:
        """Seconds inside this attempt's sampling holds, outside resource operations."""
        open_for = self.clock() - self._rollout_since if self._rollout_since is not None else 0.0
        return self._rollout_total + open_for

    def setup_elapsed(self) -> float:
        """Seconds of the running hook outside holds, permit waits, and resource operations."""
        if self._wall_since is None:
            return 0.0
        return max(0.0, self.clock() - self._wall_since - self.excluded_seconds())

    @property
    def rollout_running(self) -> bool:
        return self._rollout_since is not None

    @property
    def outside_holds(self) -> bool:
        return self._excluded_depth == 0


class _Hold:
    """One capacity hold; exit is idempotent and may happen from another task."""

    def __init__(self, broker: RunCapacity, state: AttemptPhases, phase: Phase) -> None:
        self._broker, self._state, self._phase = broker, state, phase
        self._acquired = False
        self._released = False

    async def __aenter__(self) -> None:
        await self._broker._enter(self)  # pyright: ignore[reportPrivateUsage]

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def release(self) -> None:
        if self._acquired and not self._released:
            self._released = True
            self._broker._exit(self)  # pyright: ignore[reportPrivateUsage]

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def state(self) -> AttemptPhases:
        return self._state


class GroupHold:
    """A group grader call's verification slot, from ``RunCapacity.group_verification``."""

    def __init__(self) -> None:
        self.kept = False

    def keep(self) -> None:
        """The call ignored cancellation and is still running: the slot stays held."""
        self.kept = True


class AttemptCapacity:
    """The ``CapacityBroker`` a runtime receives: bound to one attempt."""

    def __init__(self, broker: RunCapacity, state: AttemptPhases) -> None:
        self._broker, self._state = broker, state

    def hold(self, phase: Phase) -> _Hold:
        if phase is Phase.PROVISIONING:
            raise self._state.violate("runtimes cannot hold provisioning capacity")
        if phase not in self._state.declared:
            raise self._state.violate(
                f"hold({phase.value}) needs {phase.value} in internally_acquired_phases"
            )
        if self._state.kept:
            raise self._state.refuse(f"hold({phase.value})")
        return _Hold(self._broker, self._state, phase)

    async def admit(self) -> None:
        """Admit this attempt now, before it starts an environment that exists to be sampled.

        Start admission happens once per attempt; its first sampling hold admits it
        otherwise. Admitting at environment start means backpressure stops
        environments starting, and the pre-start places let a few be ready ahead of
        a free sampling slot. The sampling permit, and the rollout clock, still wait
        for the first sampling hold.
        """
        state = self._state
        if Phase.SAMPLING not in state.declared:
            raise state.violate("admit() needs sampling in internally_acquired_phases")
        if state.kept:
            raise state.refuse("admit()")
        if state.admitted:
            return
        if state.hook is None:
            raise state.violate("admit() called while neither sample() nor grade() runs")
        state._exclude(1)  # pyright: ignore[reportPrivateUsage]
        self._broker.notify()
        try:
            await self._broker._admit(state)  # pyright: ignore[reportPrivateUsage]
        finally:
            state._exclude(-1)  # pyright: ignore[reportPrivateUsage]
            self._broker.notify()


class RunCapacity:
    """Run-wide permits for sampling and verification, and start admission."""

    def __init__(
        self,
        *,
        sampling_slots: int,
        verification_slots: int,
        admission: StartAdmission,
        on_change: Callable[[], None],
        prestart: int = 0,
    ) -> None:
        if sampling_slots < 1 or verification_slots < 1:
            raise ValueError("capacity limits must be positive")
        if prestart < 0:
            raise ValueError("the pre-start allowance must be nonnegative")
        self._prestart = prestart
        self._limits = {Phase.SAMPLING: sampling_slots, Phase.VERIFICATION: verification_slots}
        self._permits = {phase: asyncio.Semaphore(limit) for phase, limit in self._limits.items()}
        self._admission = admission
        self._on_change = on_change
        # Attempts waiting for start admission, in arrival order; served by group.
        self._queue: dict[str, tuple[AttemptPhases, asyncio.Future[None]]] = {}
        self._attempts: dict[str, AttemptPhases] = {}
        self._active: set[str] = set()
        # Run-level verification holds, taken for whole groups: held, and waiting.
        self._group_holding = 0
        self._group_waiting = 0

    def track(self, state: AttemptPhases) -> AttemptCapacity:
        self._attempts[state.identity.attempt_id] = state
        return AttemptCapacity(self, state)

    def forget(self, state: AttemptPhases) -> None:
        for hold in list(state.open_holds):
            hold.release()
        self._attempts.pop(state.identity.attempt_id, None)
        self._active.discard(state.identity.attempt_id)
        self.notify()

    def runner_hold(self, state: AttemptPhases, phase: Phase) -> _Hold:
        """A hold the runner takes around a whole hook, for a runner-owned phase."""
        return _Hold(self, state, phase)

    @contextlib.asynccontextmanager
    async def group_verification(self) -> AsyncIterator[GroupHold]:
        """One verification slot for one group grader call, bound to no attempt.

        It shares the verification permits with every attempt's ``grade()``, so a
        judge cannot run beside more graders than ``max_concurrent_verifiers``. The
        slot is given back on exit, unless the call was kept: it is still running.
        """
        permits = self._permits[Phase.VERIFICATION]
        self._group_waiting += 1
        self.notify()
        try:
            await permits.acquire()
        finally:
            self._group_waiting -= 1
        self._group_holding += 1
        self.notify()
        hold = GroupHold()
        try:
            yield hold
        finally:
            if not hold.kept:
                self._group_holding -= 1
                permits.release()
                self.notify()

    def notify(self) -> None:
        """Pressure or admission inputs changed; admit whoever now fits, in order."""
        self._serve()
        self._on_change()

    async def _admit(self, state: AttemptPhases) -> None:
        # A lower target never interrupts an admitted attempt; ``_serve`` decides
        # who is admitted, and where the pre-start places apply.
        attempt_id = state.identity.attempt_id
        admitted = asyncio.get_running_loop().create_future()
        self._queue[attempt_id] = (state, admitted)
        self._serve()
        try:
            await admitted
            if state.hook is None:
                # Queued during its grace and granted once its hook ended; nothing would
                # give the place back.
                raise state.ended(
                    "admission", "admission granted after its sample() or grade() ended"
                )
        except BaseException:
            if self._queue.pop(attempt_id, None) is None and state.admitted:
                # Admitted, then cancelled before it could run: the place is free again.
                state.admitted = False
                self._active.discard(attempt_id)
                self.notify()
            raise

    def _serve(self) -> None:
        """Admit waiting attempts oldest group first, each if it fits its own target.

        Groups go in the order the pipeline dispatched them, and a group's attempts
        in arrival order, so a retried attempt, which arrives after newer groups'
        first attempts, is not left behind them while its group waits for it.

        A started group's siblings fill up to the full rollout limit, so they may
        pass unstarted groups waiting ahead of them; every unstarted group shares one
        target, so among them nobody passes anybody, and a newcomer waits its turn.

        The pre-start places sit on top of the full limit only, never on top of a
        lowered target: an attempt admitted there starts its group, whose siblings
        then fill regardless of the target, so under backpressure a pre-start place
        would start a whole group the pipeline asked not to sample.
        """
        waiting = sorted(
            self._queue.items(),
            key=lambda item: self._admission.order(item[1][0].identity.group_id),
        )  # a stable sort: arrival order within a group
        for attempt_id, (state, admitted) in waiting:
            if admitted.done():  # cancelled; its waiter leaves the queue itself
                continue
            target = self._admission.target(state.identity.group_id)
            full = target >= self._limits[Phase.SAMPLING]
            if len(self._active) < target + (self._prestart if full else 0):
                # No await between checking the target and reserving the slot.
                del self._queue[attempt_id]
                self._active.add(attempt_id)
                self._admission.started(state.identity.group_id)
                state.admitted = True
                admitted.set_result(None)

    async def _enter(self, hold: _Hold) -> None:
        state, phase = hold.state, hold.phase
        if state.hook is None:
            raise state.ended(
                f"hold({phase.value})",
                f"{phase.value} held while neither sample() nor grade() runs",
            )
        if phase is Phase.SAMPLING and (state.holding.get(phase) or state.waiting.get(phase)):
            raise state.violate("sampling holds of one attempt must not overlap")
        state.waiting[phase] = state.waiting.get(phase, 0) + 1
        state._exclude(1)  # pyright: ignore[reportPrivateUsage]
        reserved = False
        acquired = False
        self.notify()
        try:
            if phase is Phase.SAMPLING:
                if not state.admitted:
                    await self._admit(state)
                else:
                    self._active.add(state.identity.attempt_id)
                reserved = True
            await self._permits[phase].acquire()
            if state.hook is None:
                # The hook ended while this waited, in a task its cancellation did not
                # reach; nothing would give the slot back.
                self._permits[phase].release()
                raise state.ended(
                    f"hold({phase.value})",
                    f"{phase.value} granted after its sample() or grade() ended",
                )
            acquired = True
        finally:
            state.waiting[phase] -= 1
            if not acquired:
                if reserved:
                    self._active.discard(state.identity.attempt_id)
                state._exclude(-1)  # pyright: ignore[reportPrivateUsage]
                self.notify()
                state.changed.set()
        hold._acquired = True  # pyright: ignore[reportPrivateUsage]
        state.holding[phase] = state.holding.get(phase, 0) + 1
        state.open_holds.add(hold)
        if phase is Phase.SAMPLING:
            state.sampling_entered = True
            state.open_sampling()
        self.notify()
        state.changed.set()

    def _exit(self, hold: _Hold) -> None:
        state, phase = hold.state, hold.phase
        self._permits[phase].release()
        state.holding[phase] -= 1
        state.open_holds.discard(hold)
        if phase is Phase.SAMPLING:
            state.close_sampling()
            self._active.discard(state.identity.attempt_id)
        state._exclude(-1)  # pyright: ignore[reportPrivateUsage]
        self.notify()
        state.changed.set()

    def admitted(self, state: AttemptPhases) -> bool:
        """Whether ``state`` holds one of the start admission places."""
        return state.identity.attempt_id in self._active

    def active(self) -> int:
        """Attempts admitted to start and not yet forgotten: what a target is compared with."""
        return len(self._active)

    def snapshot(self) -> dict[str, float]:
        states = list(self._attempts.values())
        sampling = sum(s.holding.get(Phase.SAMPLING, 0) for s in states)
        out: dict[str, float] = {
            "admission/active_rollouts": float(len(self._active)),
            # Waiting in the start-admission queue, sandbox not yet started.
            "admission/waiting": float(len(self._queue)),
            # Admitted and not yet sampling: for a runtime that admits at environment
            # start, environments starting or up and waiting for a sampling slot.
            "admission/prestarted": float(max(0, len(self._active) - sampling)),
            "admission/prestart_limit": float(self._prestart),
        }
        for phase in (Phase.SAMPLING, Phase.VERIFICATION):
            out[f"slots/{phase.value}"] = float(sum(s.holding.get(phase, 0) for s in states))
            out[f"queue/{phase.value}_waiting"] = float(
                sum(s.waiting.get(phase, 0) for s in states)
            )
            out[f"slots/{phase.value}_limit"] = float(self._limits[phase])
        out["slots/verification"] += self._group_holding
        out["queue/verification_waiting"] += self._group_waiting
        return out
