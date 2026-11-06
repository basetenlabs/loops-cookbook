"""One try of a group grader: one verification slot, under the grading deadline.

A group is not an attempt, so this is not the driver's attempt lifecycle. Each try of
``grade_group()`` holds one run-level verification slot and runs under
``[rollout.grading] timeout_seconds``. The pipeline's group stage decides whether to
try again, after rechecking its group, as the driver does for ``grade()``. A result
that is not one finite reward per member is a grader bug, a
``PermanentGradingFailure``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from loops_cookbook.cancellation import Stopped, run_until
from loops_cookbook.runtime.api import (
    AttemptAbortReason,
    GroupGrader,
    MemberEvidence,
    PermanentGradingFailure,
)
from loops_cookbook.runtime.capacity import GroupHold, RunCapacity
from loops_cookbook.runtime.driver import (
    HookDeadline,
    NonQuiescentHook,
    RuntimeExecutionFailure,
    finite_reward,
)


@dataclass(frozen=True, slots=True)
class GroupGradingSettings:
    timeout_seconds: float
    cancellation_grace_seconds: float


class GroupGrading:
    """Tries the runtime's ``grade_group()`` on a group's members."""

    def __init__(
        self,
        grader: GroupGrader,
        capacity: RunCapacity,
        settings: GroupGradingSettings,
        observe: Callable[[float], None],
        unquiesced: Callable[[str, NonQuiescentHook], None],
    ) -> None:
        self._grader = grader
        self._capacity = capacity
        self._settings = settings
        self._observe = observe
        self._unquiesced = unquiesced
        self._calls = 0

    @property
    def calls(self) -> int:
        """Calls on a slot now, not those waiting for one.

        A call still running after cancellation keeps its slot, so it still counts.
        """
        return self._calls

    async def score(self, group_id: str, members: Sequence[MemberEvidence]) -> list[float]:
        """One final reward per member, in order, or the try's failure raised.

        Each call's duration, from its slot to its end, goes to ``observe``. A call
        that ignores its cancellation keeps its verification slot, since the work it
        bounds is still happening, and goes to ``unquiesced``; it then raises
        ``NonQuiescentHook``, or the waiter's cancellation when one came.
        """
        async with self._capacity.group_verification() as slot:
            self._calls += 1
            started = time.monotonic()
            try:
                result = await self._call(members, slot)
            except NonQuiescentHook as still_running:
                self._unquiesced(group_id, still_running)
                still_running.reraise()
            finally:
                if not slot.kept:
                    self._calls -= 1
                self._observe(time.monotonic() - started)
        return _rewards(result, len(members))

    async def _call(self, members: Sequence[MemberEvidence], slot: GroupHold) -> object:
        """One try under the grading deadline, stopped with bounded quiescence.

        A call that ignores its stop keeps its slot and raises ``NonQuiescentHook``.
        """
        grace = self._settings.cancellation_grace_seconds
        ran = await run_until(
            asyncio.create_task(self._grade(members), name="grade_group"),
            asyncio.sleep(self._settings.timeout_seconds),
            grace,
            own_failure=lambda leaked: RuntimeExecutionFailure("grade_group", None, leaked),
        )
        if not isinstance(ran, Stopped):
            return ran
        if not ran.settled:
            slot.keep()
            raise NonQuiescentHook(
                f"runtime grade_group() ignored cancellation for {grace:g}s; it is still running",
                ran,
            )
        ran.propagate()
        raise HookDeadline(AttemptAbortReason.CANCELLED, "grade_group() exceeded its deadline")

    async def _grade(self, members: Sequence[MemberEvidence]) -> object:
        try:
            return await self._grader.grade_group(members)
        except PermanentGradingFailure:
            raise
        except Exception as error:
            raise RuntimeExecutionFailure("grade_group", None, error) from error


def _rewards(result: object, members: int) -> list[float]:
    """The grader's rewards, or a ``PermanentGradingFailure``: the same result every try."""
    if isinstance(result, (str, bytes)) or not isinstance(result, Sequence):
        raise PermanentGradingFailure(
            f"grade_group() returned a {type(result).__name__}, not one reward per member"
        )
    rewards: Sequence[object] = result  # pyright: ignore[reportUnknownVariableType]
    if len(rewards) != members:
        raise PermanentGradingFailure(
            f"grade_group() returned {len(rewards)} rewards for {members} members"
        )
    for reward in rewards:
        if not finite_reward(reward):
            raise PermanentGradingFailure(
                f"grade_group() returned a non-finite or non-numeric reward {reward!r}"
            )
    return [float(reward) for reward in rewards]  # pyright: ignore[reportArgumentType]
