"""Group admission, the freshness window, and batch selection over the attempt driver.

The pipeline decides which rollouts to dispatch and when a group can train; the
attempt driver owns every attempt's lifecycle, and the capacity broker owns the
sampling and verification permits and start admission.

For a runtime with a group grader, a complete group passes one more stage before it
is ready. Before every try of the grader::

    drop stale members -> size >= min_group_size? -> grade_group()

and once it has scored the group, the no-variance check. So the grader is never paid
to score a group about to be dropped. Every group dropped before training, here or at
selection, goes through ``_drop``.
"""

from __future__ import annotations

import asyncio
import collections
import heapq
import itertools
import logging
import math
import random
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from statistics import fmean
from typing import Any, Self

from loops_cookbook.cancellation import outlast
from loops_cookbook.evidence import JsonValue, Rollout, TaskRef
from loops_cookbook.llm.transcript import SealedTranscript
from loops_cookbook.rl.config import Config
from loops_cookbook.rl.metrics import DurationSummary
from loops_cookbook.runtime.api import (
    GroupGrader,
    MemberEvidence,
    PermanentGradingFailure,
    Phase,
)
from loops_cookbook.runtime.capacity import RunCapacity
from loops_cookbook.runtime.driver import (
    Admitted,
    AttemptDriver,
    Discard,
    SystemicFailure,
)
from loops_cookbook.runtime.groups import GroupGrading, GroupGradingSettings

logger = logging.getLogger(__name__)


class UnusableGroupsError(RuntimeError):
    """Repeated group rejection prevents training progress."""


def _unusable_streak(
    count: int, rewards: collections.Counter[float], undersized: int, untrainable: int = 0
) -> str:
    """What a streak of unusable groups was made of, and what would change it.

    A group whose rollouts all scored the same carries no advantage to learn from. The
    stop cannot tell a solved task from a stuck policy or a constant verifier, so it
    reports the rewards those groups shared rather than guessing.
    """
    parts = []
    if rewards:
        shared = ", ".join(
            f"{reward:g} in {n}"
            for reward, n in sorted(rewards.items(), key=lambda item: (-item[1], -item[0]))
        )
        parts.append(
            f"{sum(rewards.values())} gave every rollout one reward ({shared}), so they carry "
            "nothing to learn from. If the tasks are solved, training is done; otherwise raise "
            "[sampling] temperature or [training] group_size, or check the verifier"
        )
    if undersized:
        parts.append(
            f"{undersized} had too few usable rollouts (group_size - max_group_losses): check "
            "rollout failures and the freshness settings"
        )
    if untrainable:
        parts.append(
            f"{untrainable} had an advantage only on rollouts whose agent discarded every "
            "generation, so nothing in them trains"
        )
    return f"{count} consecutive unusable groups: " + "; ".join(parts) + "."


@dataclass
class Counters:
    admitted_groups: int = 0
    admitted_rollouts: int = 0
    dispatched: int = 0
    sampled: int = 0
    graded: int = 0
    trained: int = 0
    groups_trained: int = 0
    discarded: dict[str, int] = field(default_factory=dict)
    discarded_groups: dict[str, int] = field(default_factory=dict)

    def discard(self, reason: Discard, n: int = 1) -> None:
        """Count rollouts only; group rejection counts its surviving members separately."""
        if n:
            self.discarded[reason.value] = self.discarded.get(reason.value, 0) + n

    def discard_group(self, reason: Discard, surviving_rollouts: int) -> None:
        self.discarded_groups[reason.value] = self.discarded_groups.get(reason.value, 0) + 1
        self.discard(reason, surviving_rollouts)

    def gauges(self) -> dict[str, float]:
        out = {
            f"pipeline/{name}": float(getattr(self, name))
            for name in (
                "admitted_groups",
                "admitted_rollouts",
                "dispatched",
                "sampled",
                "graded",
                "trained",
                "groups_trained",
            )
        }
        for unit, counts in (("rollouts", self.discarded), ("groups", self.discarded_groups)):
            out.update({f"discard/{unit}/{key}": float(value) for key, value in counts.items()})
            out[f"discard/{unit}/total"] = float(sum(counts.values()))
        return out


@dataclass
class Group:
    """A task's rollouts, from admission until they train or are dropped.

    ``members`` are its sealed, graded rollouts. For a group grader, ``transcripts``
    holds each member's, by rollout id, until the grader scores the group; until then
    no member has its final reward.
    """

    group_id: str
    task: TaskRef
    expected: int
    members: list[Rollout] = field(default_factory=list)
    resolved: int = 0
    ready_at: float | None = None
    # When its first member, and then its last, was resolved (graded or discarded):
    # the time its first finishers waited for the rest, which they spend ageing.
    first_resolved_at: float | None = None
    completed_at: float | None = None
    transcripts: dict[str, SealedTranscript] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.resolved == self.expected

    def evidence(self) -> list[MemberEvidence]:
        """What its group grader reads: each member beside its transcript."""
        return [MemberEvidence(m, self.transcripts[m.identity.rollout_id]) for m in self.members]

    def recorded(self) -> list[Rollout]:
        """Every member as its record shows it: before its grader scores it, no reward."""
        if not self.transcripts:
            return list(self.members)
        return [replace(member, reward=None) for member in self.members]


# Dropped because the group cannot train; these extend the unusable-groups streak.
_UNUSABLE = frozenset({Discard.GROUP_TOO_SMALL, Discard.NO_VARIANCE, Discard.NOTHING_TO_TRAIN})
_STALE = frozenset({Discard.STALE_BEFORE_GRADING, Discard.STALE_BEFORE_TRAINING})
# Dropped without a verdict on the group: a resume samples it again.
_UNTRAINED = frozenset({Discard.TRAINING_ABORTED, Discard.CANCELLED})
# The async survival estimate: over the last this many batches' worth of groups, and
# never below this share, so a run of drops cannot make the expected losses unbounded.
_SURVIVAL_BATCHES = 4
_MIN_SURVIVAL = Fraction(1, 4)


def _admission_window(fresh: Fraction, survival: Fraction, batch: int) -> Fraction:
    """``fresh`` groups plus those ``survival`` says will drop, counted up to one ``batch``
    and rounded up to whole groups: a replacement is a whole group, and a fraction of one
    would hold back the replacement it was counted for.

    Drops rise as a policy degrades or saturates, so uncounted, the window widened most
    when the policy was least stable, each extra group training later. A batch short
    of replacements takes groups from the batch admitted after it, and a run dropping
    more than a batch is better slowed than fed staler groups.

    In exact arithmetic: float error would round some windows up a group.
    """
    return fresh + math.ceil(min(fresh / survival - fresh, Fraction(batch)))


def _admission_limit(needed: int, judged: int, survival: Fraction, batch: int) -> Fraction:
    """Live groups not training that the steps left can use, ``needed`` groups to find.

    The ``judged`` ones passed their check and can now drop only by staleness, so the
    allowance is for the rest alone: counted for them too, a batch already selected
    admitted a batch more that never trains.
    """
    return judged + _admission_window(Fraction(max(needed - judged, 0)), survival, batch)


class TaskStream:
    def __init__(self, tasks: list[TaskRef], *, group_size: int, seed: int) -> None:
        if not tasks:
            raise ValueError("the task pool is empty")
        self._tasks, self._size, self._seed = list(tasks), group_size, seed

    def groups(self) -> Iterator[Group]:
        rng, counter = random.Random(self._seed), itertools.count()
        while True:
            epoch = list(self._tasks)
            rng.shuffle(epoch)
            for task in epoch:
                yield Group(f"g{next(counter):07d}-{task.id}", task, self._size)


@dataclass(frozen=True)
class _Unit:
    group: Group
    rollout_id: str
    queued_at: float = field(default_factory=time.monotonic)


def _length(rollout: Rollout | None) -> dict[str, int | None]:
    """A rollout's length, for slicing dispositions by it; unknown before it sealed."""
    if rollout is None:
        return {"generations": None, "sampled_tokens": None}
    return {
        "generations": len(rollout.samples),
        "sampled_tokens": sum(len(sample.completion_ids) for sample in rollout.samples),
    }


class ReadyGroups:
    """Groups ready to train, the one closest to going stale first.

    A group goes stale with its oldest member: ``pop`` returns the group whose
    members' oldest policy version is lowest, then the one dispatched first, then the
    one made ready first. A group with no members sorts first; selection drops it.
    Completion order would let a long group, sampled early and finished late, wait
    behind newer, shorter ones until it aged out of the off-policy window.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, int, Group]] = []
        self._pushed = itertools.count()

    def push(self, group: Group, order: int) -> None:
        versions = [r.minimum_policy_version for r in group.members]
        oldest = min((v for v in versions if v is not None), default=-1)
        heapq.heappush(self._heap, (oldest, order, next(self._pushed), group))

    def pop(self) -> Group | None:
        return heapq.heappop(self._heap)[-1] if self._heap else None

    def clear(self) -> None:
        self._heap.clear()

    def __len__(self) -> int:
        return len(self._heap)


DriverFactory = Callable[
    [RunCapacity, Callable[[SystemicFailure], None], Callable[[], None]], AttemptDriver
]


class Pipeline:
    """Own admitted groups through selection until training commits or discards them."""

    def __init__(
        self,
        config: Config,
        *,
        tasks: list[TaskRef],
        driver: DriverFactory,
        initial_group_index: int = 0,
        initial_step: int = 0,
        replay_group_indices: Sequence[int] = (),
        on_rollout: Callable[[dict[str, Any]], None] | None = None,
        group_grader: GroupGrader | None = None,
    ) -> None:
        self.config, self.counters = config, Counters()
        # Receives one record per discarded rollout; the runner writes trained ones
        # from ``trained_record``.
        self._on_rollout = on_rollout
        self._training, self._rollout = config.training, config.rollout
        self._training_groups: set[str] = set()
        # Dispatched and not yet resolved, with the rollout once it sealed: what a
        # shutdown records as cancelled.
        self._in_flight: dict[str, tuple[Group, Rollout | None]] = {}
        self._stream = TaskStream(
            tasks, group_size=self._training.group_size, seed=self._rollout.shuffle_seed
        ).groups()
        if initial_group_index < 0:
            raise ValueError("initial_group_index must be nonnegative")
        replay = set(replay_group_indices)
        if len(replay) != len(replay_group_indices) or not all(
            0 <= index < initial_group_index for index in replay
        ):
            raise ValueError("replay_group_indices must be distinct and below initial_group_index")
        # Groups a checkpoint left untrained, in task order: dispatched before any new one.
        self._replay: collections.deque[tuple[int, Group]] = collections.deque()
        for index in range(initial_group_index):
            group = next(self._stream)
            if index in replay:
                self._replay.append((index, group))
        self._next_group_index = initial_group_index
        self._group_index: dict[str, int] = {}  # each live group's place in the stream
        # Dispatched groups that neither trained nor were dropped for good: a shutdown
        # cancels them, so a resume must sample them again.
        self._untrained: set[int] = set()
        self._started_groups: set[str] = set()
        # Whether each recent group passed its first size and variance check (True) or
        # was dropped as unusable (False): the share expected to survive, which sizes
        # the async window's drop allowance. Recorded once, when judged, so a good
        # group counts as soon as a bad one would; a later recheck can fail only by
        # staleness.
        self._outcomes: collections.deque[bool] = collections.deque(
            maxlen=_SURVIVAL_BATCHES * self._training.groups_per_step
        )
        # Their share, floored; 1 before any. Set when one is recorded, since every
        # admission check reads it.
        self._survival_share = Fraction(1)
        self._judged: set[str] = set()
        # Groups that lost a member to staleness before they were judged: their fate
        # says how the window was sized, not how the task goes, so it stays out.
        self._lost_to_staleness: set[str] = set()
        self._groups: dict[str, Group] = {}
        self._selected: set[str] = set()
        self._to_sample: asyncio.Queue[_Unit] = asyncio.Queue(maxsize=self._training.group_size)
        self._to_grade: asyncio.Queue[tuple[_Unit, Admitted]] = asyncio.Queue()
        self._ready = ReadyGroups()
        # Valid groups from a batch that needed replacements: they go first next time.
        self._retained: collections.deque[Group] = collections.deque()
        self._changed, self._ready_changed = asyncio.Event(), asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._sampling = self._grading = self._sync_demand = 0
        # Batches committed by this pipeline, from ``initial_step``: the runner trains
        # until the run has made ``max_steps`` updates, a resume's earlier ones included.
        self._initial_step = initial_step
        self._committed_batches = 0
        self._fatal: Exception | None = None
        self._stopping = False
        self._consecutive_unusable_groups = 0
        self._streak_rewards: collections.Counter[float] = collections.Counter()
        # Graded rewards of groups dropped as unusable since the last update took them,
        # and those members' metadata (only that: not their samples) for the same report.
        self._unusable_rewards: list[float] = []
        self._unusable_metadata: list[Mapping[str, JsonValue]] = []
        self._streak_undersized = 0
        self._streak_untrainable = 0
        self._started = time.monotonic()
        # Time the async window held new groups back while fewer than
        # max_concurrent_rollouts attempts were admitted: rollout capacity freshness
        # left unused. Admitted, not sampling: idle slots behind sandbox start-up are
        # not the window's doing. Since the start, and since the last update took it.
        self._held_since: float | None = None
        self._held_seconds = 0.0
        self._held_taken = (self._started, 0.0)
        # The last cycle, for choosing admission_steps_ahead: how long the trainer
        # waited for its batch, and the complete groups beyond a batch when it asked.
        # Measured from the runner's calls: it frees when it asks for a batch and has
        # it at start_training. An abort's cycle measures nothing.
        self._freed: tuple[float, int] | None = None  # when, spare groups
        self._last_wait: float | None = None
        self._last_spare: int | None = None
        self.capacity = RunCapacity(
            sampling_slots=self._rollout.max_concurrent_rollouts,
            verification_slots=self._rollout.max_concurrent_verifiers,
            admission=self,
            on_change=self._capacity_changed,
            prestart=self._rollout.prestart_places,
        )
        self.driver = driver(self.capacity, self._stop, self._sealed)
        self._durations = {
            name: DurationSummary()
            for name in (
                "sampling",
                "grading",
                "group_grading",
                "sampling_queue",
                "grading_queue",
                "ready_queue",
            )
        }
        if self.driver.capabilities.pending_group_grade != (group_grader is not None):
            raise ValueError("a group grader is passed exactly when the runtime declares one")
        self._group_grading = (
            None
            if group_grader is None
            else GroupGrading(
                group_grader,
                self.capacity,
                GroupGradingSettings(
                    timeout_seconds=self._rollout.grading.timeout_seconds,
                    cancellation_grace_seconds=self._rollout.cleanup.cancellation_grace_seconds,
                ),
                observe=self._durations["group_grading"].observe,
                unquiesced=lambda group_id, error: self.driver.group_unquiesced(group_id, error),
            )
        )
        self._to_group_grade: asyncio.Queue[Group] = asyncio.Queue()
        # Complete groups in the judge stage: queued for the group grader, or being judged.
        self._judging: set[str] = set()
        self.settled = True
        """False once a shutdown leaves an attempt unsettled; the run is then unclean."""

    def _stop(self, error: Exception) -> None:
        if self._fatal is not None:
            return
        logger.error("Stopping pipeline: %s", error)
        self._fatal = error
        self._stopping = True
        self._changed.set()
        self._ready_changed.set()

    def _sealed(self) -> None:
        # A rollout exists: counted before evidence checks can discard it.
        self.counters.sampled += 1

    def _capacity_changed(self) -> None:
        self._changed.set()
        self._account_hold()

    def _account_hold(self, now: float | None = None) -> float:
        """Start or stop the held clock on every change that can move it; the total so far."""
        now = time.monotonic() if now is None else now
        held = (
            self._training.mode == "async"
            and not self._room()
            and self.capacity.active() < self._rollout.max_concurrent_rollouts
        )
        if held and self._held_since is None:
            self._held_since = now
        elif not held and self._held_since is not None:
            self._held_seconds += now - self._held_since
            self._held_since = None
        return self._held_seconds + (0.0 if self._held_since is None else now - self._held_since)

    def take_window_held_share(self) -> float:
        """The share of wall time since the last call the window held rollout slots idle."""
        now = time.monotonic()  # one instant: the total and the interval end at it
        total = self._account_hold(now)
        since, before = self._held_taken
        self._held_taken = (now, total)
        return (total - before) / (now - since) if now > since else 0.0

    def _judge(self, group: Group, *, survived: bool) -> None:
        """Record a group's first size and variance check, unless staleness shaped it."""
        if group.group_id in self._judged or group.group_id in self._lost_to_staleness:
            return
        self._judged.add(group.group_id)
        self._outcomes.append(survived)
        self._survival_share = max(
            _MIN_SURVIVAL, Fraction(sum(self._outcomes), len(self._outcomes))
        )
        self._account_hold()  # survival sizes the window: this can close it

    def _survival(self) -> Fraction:
        """The share of recent groups judged usable, floored; 1 before any."""
        return self._survival_share

    def _committed(self) -> int:
        """Started groups not yet training: each will take a place in a later batch."""
        return len(self._started_groups - self._training_groups)

    def _committed_rollouts(self) -> int:
        """What the started groups not yet training hold of the window, in rollouts.

        A complete group holds its whole place, since a batch takes it however many
        members it has left; one still filling holds only its rollouts not yet
        resolved. Counted by groups, a group whose last member ran long kept its place
        while the rest sat finished, and a window of such groups held the sampler
        idle: the slots its finished members left go to new groups instead. They are
        lent, not freed: the group holds its whole place again once complete, so the
        count can pass the window by what filling groups had finished; the groups'
        bound in ``_room`` holds regardless.
        """
        held = 0
        for group_id in self._started_groups - self._training_groups:
            group = self._groups.get(group_id)
            if group is not None:
                held += group.expected if group.complete else group.expected - group.resolved
        return held

    def _window(self, steps: float | None = None) -> Fraction:
        """Started groups not yet training that async admits, drops included.

        Fresh are the ``steps`` batches after the one training, or, with none
        training, the batch being assembled and those after it: by default
        ``steps_admitted_ahead``, which is ``max_steps_off_policy`` unless
        ``admission_steps_ahead`` lowers it, by fractions of a batch. Counted in
        groups, since a batch takes ``groups_per_step`` of them however many members
        each has left, plus the groups recent drops say will not train, at most a batch
        of them. Exact: a fraction of a batch is a fraction of a group's rollouts.
        """
        ahead = self._training.steps_admitted_ahead if steps is None else steps
        batches = Fraction(ahead).limit_denominator(10**6) + (0 if self._training_groups else 1)
        fresh = batches * self._training.groups_per_step
        return _admission_window(fresh, self._survival(), self._training.groups_per_step)

    def _room(self) -> bool:
        """Room for one more group, by two bounds.

        The pace: its rollouts and those committed (``_committed_rollouts``) fit the
        window, so the finished members of groups still filling free their slots.
        The guarantee: the started groups fit the window ``max_steps_off_policy``
        sizes, since each takes a batch place however few members it has left, so
        none starts that could only train stale. With ``admission_steps_ahead``
        unset the two are one window and the guarantee binds first: admission is
        by groups alone.
        """
        size = self._training.group_size
        paced = self._committed_rollouts() + size <= self._window() * size
        return paced and self._committed() < self._window(self._training.steps_off_policy)

    def _needed(self) -> bool:
        """Whether the steps left to train can use one more group; always, unlimited.

        Groups still to find are the batches left, less the one training, which is
        certain. The live groups not training compete for them, plus the groups recent
        drops say will not train (``_admission_limit``), so a dropped group frees its
        place for a replacement. Group admission only: every admitted group is within
        it, so start admission need not ask.
        """
        if self._training.step_limit is None:
            return True
        left = self._training.step_limit - self._initial_step - self._committed_batches
        needed = left * self._training.groups_per_step - len(self._training_groups)
        if needed <= 0:
            return False
        candidates = len(self._groups) - len(self._training_groups)
        limit = _admission_limit(
            needed,
            len(self._judged - self._training_groups),
            self._survival(),
            self._training.groups_per_step,
        )
        return candidates < limit

    def target(self, group_id: str) -> int:
        """Start admission: a started group, and every group in sync mode, fills its slots.

        In async, an unstarted group may start while the window has room, and then at
        full concurrency.
        """
        if self._training.mode == "sync" or group_id in self._started_groups or self._room():
            return self._rollout.max_concurrent_rollouts
        return 0

    def started(self, group_id: str) -> None:
        self._started_groups.add(group_id)

    def order(self, group_id: str) -> int:
        """Admission order: groups in the order they were dispatched.

        A group no longer live has no attempt waiting; it would sort after them all.
        """
        return self._group_index.get(group_id, self._next_group_index)

    async def _dispatch(self) -> None:
        while not self._stopping:
            self._changed.clear()
            if self._training.mode == "sync" and len(self._groups) >= self._sync_demand:
                await self._changed.wait()
                continue
            if self._training.mode == "async" and not (self._room() and self._needed()):
                await self._changed.wait()
                continue
            if self._replay:
                index, group = self._replay.popleft()
            else:
                index, group = self._next_group_index, next(self._stream)
                self._next_group_index += 1
            self._group_index[group.group_id] = index
            self._untrained.add(index)
            self._groups[group.group_id] = group
            self.counters.admitted_groups += 1
            self.counters.admitted_rollouts += group.expected
            for index in range(group.expected):
                await self._to_sample.put(_Unit(group, f"{group.group_id}-r{index}"))
            await asyncio.sleep(0)

    async def _sampling_worker(self) -> None:
        while not self._stopping:
            unit = await self._to_sample.get()
            self._durations["sampling_queue"].observe(time.monotonic() - unit.queued_at)
            self.counters.dispatched += 1
            self._sampling += 1
            started = time.monotonic()
            self._in_flight[unit.rollout_id] = (unit.group, None)
            try:
                result = await self.driver.sample(
                    unit.group.task, unit.group.group_id, unit.rollout_id
                )
                if isinstance(result, Discard):
                    del self._in_flight[unit.rollout_id]
                    self.counters.discard(result)
                    self._discarded(unit.rollout_id, unit.group, result)
                    self._resolve(unit)
                else:
                    self._in_flight[unit.rollout_id] = (unit.group, result.rollout)
                    self._to_grade.put_nowait((replace(unit, queued_at=time.monotonic()), result))
                self._changed.set()
                await asyncio.sleep(0)
            except SystemicFailure as error:
                self._stop(error)
                return
            finally:
                self._durations["sampling"].observe(time.monotonic() - started)
                self._sampling -= 1
            # Handed on. Until the next rollout is sampled -- as long as a rollout
            # takes -- this worker must not keep this one's attempt alive.
            del unit, result

    async def _grading_worker(self) -> None:
        while not self._stopping:
            unit, admitted = await self._to_grade.get()
            self._durations["grading_queue"].observe(time.monotonic() - unit.queued_at)
            self._grading += 1
            started = time.monotonic()
            try:
                result = await self.driver.grade(admitted)
                del self._in_flight[unit.rollout_id]
                if isinstance(result, Discard):
                    self.counters.discard(result)
                    self._discarded(unit.rollout_id, unit.group, result, admitted.rollout)
                    self._resolve(unit)
                else:
                    self.counters.graded += 1
                    self._resolve(unit, result, admitted.transcript)
                await asyncio.sleep(0)
            except SystemicFailure as error:
                self._stop(error)
                return
            finally:
                self._durations["grading"].observe(time.monotonic() - started)
                self._grading -= 1
            # Graded, and its finish is under way: not held while the next is awaited.
            del unit, admitted, result

    def _resolve(
        self,
        unit: _Unit,
        rollout: Rollout | None = None,
        transcript: SealedTranscript | None = None,
    ) -> None:
        group = unit.group
        group.resolved += 1
        now = time.monotonic()
        if group.first_resolved_at is None:
            group.first_resolved_at = now
        if rollout is not None:
            if self._group_grading is not None:
                if transcript is None:
                    raise RuntimeError("a group grader's rollout was sealed without its transcript")
                group.transcripts[rollout.identity.rollout_id] = transcript
            group.members.append(rollout)
        if group.complete:
            group.completed_at = now
            if self._group_grading is None:
                self._make_ready(group)
            else:
                self._judging.add(group.group_id)
                self._to_group_grade.put_nowait(group)
        self.capacity.notify()

    def _make_ready(self, group: Group) -> None:
        self._judging.discard(group.group_id)
        group.ready_at = time.monotonic()
        self._ready.push(group, self.order(group.group_id))
        self._ready_changed.set()

    async def _group_grading_worker(self) -> None:
        while not self._stopping:
            group = await self._to_group_grade.get()
            try:
                await self._grade_group(group)
            except (SystemicFailure, UnusableGroupsError) as error:
                self._stop(error)
                return

    async def _grade_group(self, group: Group) -> None:
        """Before every try, stale members and size; then the grader; then variance.

        A try that fails is tried again, within the runtime's ``grading_retries``,
        on the group as it stands after the backoff: a member gone stale meanwhile
        is dropped first, and a group left too small is never scored.
        """
        grading = self._group_grading
        assert grading is not None
        tries = 1 + self.driver.capabilities.grading_retries
        for number in range(1, tries + 1):
            self._drop_stale(group)
            if len(group.members) < self._training.min_group_size:
                self._drop(group, Discard.GROUP_TOO_SMALL)
                return
            try:
                rewards = await grading.score(group.group_id, group.evidence())
            except SystemicFailure:
                raise
            except Exception as error:  # noqa: BLE001 - the grader's failure, recorded and budgeted
                if number < tries and not isinstance(error, PermanentGradingFailure):
                    self.driver.group_retrying(group.group_id, error)
                    await self.driver.back_off(number)
                    continue
                # A grading failure, like a failed grade(): budgeted, not an unusable group.
                # The driver notes the members' cause, which their records carry.
                ids = [member.identity.rollout_id for member in group.members]
                self.driver.group_failed(group.group_id, error, ids)
                self._drop(group, Discard.GRADING_FAILED)
                return
            self.driver.group_scored()
            # New rollouts, not changed ones: the evidence the grader read stays as it was.
            group.members = [
                replace(member, reward=reward)
                for member, reward in zip(group.members, rewards, strict=True)
            ]
            group.transcripts = {}
            reason = self._unusable(group)
            if reason is not None:
                self._drop(group, reason)
                return
            self._judge(group, survived=True)
            self._make_ready(group)
            return

    def _settle(self, group: Group) -> None:
        """The group trained, or was dropped for a reason a resume would not undo."""
        index = self._group_index.get(group.group_id)
        if index is not None:
            self._untrained.discard(index)

    def _release(self, group: Group) -> None:
        self._groups.pop(group.group_id, None)
        self._selected.discard(group.group_id)
        self._judging.discard(group.group_id)
        self._started_groups.discard(group.group_id)
        self._training_groups.discard(group.group_id)
        self._lost_to_staleness.discard(group.group_id)
        self._judged.discard(group.group_id)
        self._group_index.pop(group.group_id, None)
        self.capacity.notify()

    def _drop_stale(self, group: Group) -> None:
        stale = {r.identity.rollout_id for r in group.recorded() if self.driver.stale(r)}
        for rollout in group.recorded():
            if rollout.identity.rollout_id in stale:
                self._discarded_member(group, rollout, Discard.STALE_BEFORE_TRAINING)
        self.counters.discard(Discard.STALE_BEFORE_TRAINING, len(stale))
        group.members = [r for r in group.members if r.identity.rollout_id not in stale]

    def _unusable(self, group: Group) -> Discard | None:
        """Why a group of graded members cannot train, or ``None`` when it can."""
        if len(group.members) < self._training.min_group_size:
            return Discard.GROUP_TOO_SMALL
        rewards = [rollout.reward for rollout in group.members]
        if len(set(rewards)) == 1:
            return Discard.NO_VARIANCE
        # Rollouts whose agent discarded every generation count in the baseline but
        # train nothing; the group needs a member that trains with an advantage.
        # The baseline exactly as grpo.advantages computes it.
        mean = fmean(r for r in rewards if r is not None)
        if not any(r.trains and r.reward != mean for r in group.members):
            return Discard.NOTHING_TO_TRAIN
        return None

    def _trainable(self, group: Group) -> Discard | None:
        """Selection's recheck: members gone stale since, then size and variance."""
        self._drop_stale(group)
        return self._unusable(group)

    def _drop(self, group: Group, reason: Discard) -> None:
        """Drop a group before training: the one path for every such drop.

        Its members are counted and recorded as discarded, and the group is released. A group that cannot train, too small or without
        variance, also extends the streak ``max_consecutive_unusable_groups`` stops;
        reaching that raises ``UnusableGroupsError``, and stops the pipeline.
        """
        members = group.recorded()
        if reason not in _UNTRAINED:
            self._settle(group)
        self.counters.discard_group(reason, len(members))
        for rollout in members:
            self._discarded(rollout.identity.rollout_id, group, reason, rollout)
        if reason in _UNUSABLE:
            self._judge(group, survived=False)
        self._release(group)
        if reason in _UNUSABLE:
            graded = [r for r in members if r.reward is not None]
            self._unusable_rewards.extend(r.reward for r in graded if r.reward is not None)
            self._unusable_metadata.extend(r.metadata for r in graded)
            self._extend_streak(group.group_id, reason, members)

    def take_unusable_rewards(self) -> list[float]:
        """The graded rewards of groups dropped as unusable (no variance, too few members)
        since the last call; the update that calls next reports them."""
        rewards, self._unusable_rewards = self._unusable_rewards, []
        return rewards

    def take_unusable_metadata(self) -> list[Mapping[str, JsonValue]]:
        """The metadata of the same graded members as ``take_unusable_rewards``, since
        the last call; the update that calls next reports them."""
        metadata, self._unusable_metadata = self._unusable_metadata, []
        return metadata

    def _extend_streak(self, group_id: str, reason: Discard, members: list[Rollout]) -> None:
        self._consecutive_unusable_groups += 1
        count = self._consecutive_unusable_groups
        reward = members[0].reward if members else None
        if reason is Discard.NO_VARIANCE and reward is not None:  # graded: always set
            self._streak_rewards[reward] += 1
        elif reason is Discard.NOTHING_TO_TRAIN:
            self._streak_untrainable += 1
        else:
            self._streak_undersized += 1
        if count == 1 or count % 5 == 0:
            logger.warning(
                "%d consecutive unusable groups; latest group %s: %s",
                count,
                group_id,
                reason.value,
            )
        limit = self._training.max_consecutive_unusable_groups
        if limit and count >= limit:
            error = UnusableGroupsError(
                _unusable_streak(
                    count,
                    self._streak_rewards,
                    self._streak_undersized,
                    self._streak_untrainable,
                )
            )
            self._stop(error)
            raise error

    def trained_record(self, rollout: Rollout, step: int) -> dict[str, Any]:
        """The ``rollouts.jsonl`` record of a rollout the update at ``step`` trained on."""
        note = self.driver.note(rollout.identity.rollout_id)
        return {
            "rollout_id": rollout.identity.rollout_id,
            "attempt_id": rollout.identity.attempt_id,
            "group_id": rollout.identity.group_id,
            "task_id": rollout.task.id,
            "task_digest": rollout.task.digest,
            "reward": rollout.reward,
            "policy_version": rollout.minimum_policy_version,
            "sample_ids": [sample.sample_id for sample in rollout.samples],
            "step": step,
            "disposition": "trained",
            "reason": None,
            "cause": None,
            "retries": note.retries,
            **_length(rollout),
        }

    def _discarded_member(self, group: Group, rollout: Rollout, reason: Discard) -> None:
        self._discarded(rollout.identity.rollout_id, group, reason, rollout)

    def _discarded(
        self,
        rollout_id: str,
        group: Group,
        reason: Discard,
        rollout: Rollout | None = None,
    ) -> None:
        if reason in _STALE:
            self._lost_to_staleness.add(group.group_id)
        note = self.driver.note(rollout_id)
        if self._on_rollout is None:
            return
        self._on_rollout(
            {
                "rollout_id": rollout_id,
                "attempt_id": None if rollout is None else rollout.identity.attempt_id,
                "group_id": group.group_id,
                "task_id": group.task.id,
                "task_digest": group.task.digest,
                "reward": None if rollout is None else rollout.reward,
                # As in a trained record: the oldest version it sampled at, not the
                # trainer's when it was dropped; None when it sampled nothing.
                "policy_version": None if rollout is None else rollout.minimum_policy_version,
                "sample_ids": [] if rollout is None else [s.sample_id for s in rollout.samples],
                "step": None,
                "disposition": "discarded",
                "reason": reason.value,
                "cause": note.cause or reason.value,
                "retries": note.retries,
                **_length(rollout),
            }
        )

    async def _next_group(self) -> Group:
        while True:
            if self._fatal is not None:
                raise self._fatal
            if self._stopping:
                raise RuntimeError("pipeline is closed")
            self._ready_changed.clear()
            group = self._retained.popleft() if self._retained else self._ready.pop()
            if group is None:
                await self._ready_changed.wait()
                continue
            if group.ready_at is not None:
                self._durations["ready_queue"].observe(time.monotonic() - group.ready_at)
            reason = self._trainable(group)
            if reason is None:
                self._judge(group, survived=True)
                self._consecutive_unusable_groups = 0
                self._streak_rewards.clear()
                self._streak_undersized = 0
                self._streak_untrainable = 0
                self._selected.add(group.group_id)
                return group
            self._drop(group, reason)

    async def next_batch(self) -> list[Group]:
        """Select one batch; sync admits only its groups, including replacements."""
        if self._selected:
            raise RuntimeError("commit or abort the selected batch first")
        if self._training.mode == "async":
            spare = len(self._ready) + len(self._retained) - self._training.groups_per_step
            self._freed = (time.monotonic(), spare)
        self._sync_demand = self._training.groups_per_step
        self._changed.set()
        groups: list[Group] = []
        try:
            while True:
                while len(groups) < self._training.groups_per_step:
                    groups.append(await self._next_group())
                if self.refresh_batch(groups):
                    self._sync_demand = 0
                    return groups
                groups = []
        except BaseException:
            self._sync_demand = 0
            self.abort_batch(groups)
            raise

    def refresh_batch(self, groups: list[Group]) -> bool:
        """Recheck before advantages; valid groups keep their turn if replacements are needed."""
        reasons = [self._trainable(group) for group in groups]
        if all(reason is None for reason in reasons):
            return True
        for group, reason in zip(groups, reasons, strict=True):
            self._selected.discard(group.group_id)
            if reason is None:
                self._retained.append(group)
                self._ready_changed.set()
        for group, reason in zip(groups, reasons, strict=True):
            if reason is not None:
                self._drop(group, reason)
        return False

    def start_training(self, groups: list[Group]) -> None:
        """Move a complete selected batch out of the window, retaining it until commit/abort."""
        if (
            self._training_groups
            or len(groups) != self._training.groups_per_step
            or len(self._selected) != self._training.groups_per_step
            or {group.group_id for group in groups} != self._selected
        ):
            raise ValueError("training needs one complete selected batch")
        self._training_groups = {group.group_id for group in groups}
        if self._freed is not None:
            freed, self._last_spare = self._freed
            self._last_wait = time.monotonic() - freed
            self._freed = None
        self.capacity.notify()

    def commit_batch(self, groups: list[Group]) -> None:
        """Release capacity after the runner commits training and publication."""
        if any(group.group_id not in self._selected for group in groups):
            raise ValueError("batch is not selected or already committed")
        self._sync_demand = 0
        self._committed_batches += 1
        self.counters.groups_trained += len(groups)
        self.counters.trained += sum(len(group.members) for group in groups)
        for group in groups:
            self._settle(group)
            self._release(group)

    def abort_batch(self, groups: list[Group]) -> None:
        self._sync_demand = 0
        self._freed = None
        for group in groups:
            if group.group_id in self._selected:
                self._drop(group, Discard.TRAINING_ABORTED)

    async def __aenter__(self) -> Self:
        # A runner-owned phase has one worker per slot; when the runtime acquires
        # its own phases the broker is the real limit, so workers only need never
        # to be the bottleneck. The group_size floor keeps the dispatcher from
        # blocking on its one-group queue.
        phases = self.driver.capabilities.internally_acquired_phases
        rollouts, verifiers = (
            self._rollout.max_concurrent_rollouts,
            self._rollout.max_concurrent_verifiers,
        )
        capacity = max(self._training.group_size, rollouts + verifiers)
        sampling = capacity if Phase.SAMPLING in phases else rollouts
        grading = capacity if Phase.VERIFICATION in phases else verifiers
        self._tasks = [asyncio.create_task(self._dispatch(), name="group-admission")]
        self._tasks += [
            asyncio.create_task(self._sampling_worker(), name=f"sample-{i}")
            for i in range(sampling)
        ]
        self._tasks += [
            asyncio.create_task(self._grading_worker(), name=f"grade-{i}") for i in range(grading)
        ]
        if self._group_grading is not None:
            # Each call holds a verification slot, so no more can run at once.
            self._tasks += [
                asyncio.create_task(self._group_grading_worker(), name=f"grade-group-{i}")
                for i in range(verifiers)
            ]
        for task in self._tasks:
            task.add_done_callback(self._worker_done)
        return self

    def _worker_done(self, task: asyncio.Task[None]) -> None:
        if self._stopping:
            return
        # Only shutdown cancels a worker, and it stops the pipeline first: any other
        # cancellation lost a worker, and whatever it held would wait forever.
        if task.cancelled():
            failure = f"pipeline worker {task.get_name()} was cancelled"
        elif (error := task.exception()) is not None:
            failure = f"pipeline worker failed: {error}"
        else:
            return
        self._fatal = RuntimeError(failure)
        self._stopping = True
        self._ready_changed.set()
        self._changed.set()

    @property
    def next_group_index(self) -> int:
        """The first place in the task stream no group has been dispatched from."""
        return self._next_group_index

    @property
    def pending_group_indices(self) -> list[int]:
        """Places below ``next_group_index`` whose groups have not trained, in order.

        A shutdown cancels the groups in flight, so a resume from a checkpoint replays
        these first: they include groups admitted for the next update, and groups
        whose update was aborted. Groups dropped as unusable, stale or failed in
        grading are not replayed.
        """
        return sorted(self._untrained | {index for index, _ in self._replay})

    async def __aexit__(
        self, _type: object = None, primary: BaseException | None = None, _tb: object = None
    ) -> None:
        self._stopping = True
        self._changed.set()
        self._ready_changed.set()
        for task in self._tasks:
            task.cancel()
        cleanup = asyncio.create_task(self._drain(), name="pipeline-cleanup")
        drain = self._rollout.shutdown_drain_seconds
        # A second cancellation, as on a repeated Ctrl-C, still gets the drain's
        # report: it is raised once the drain has settled or timed out.
        waited = await outlast(cleanup, drain)
        if not waited.settled:
            # Stop waiting, not stop cleaning: each attempt's abort() or finish()
            # still runs to its own cleanup deadline in the background.
            cleanup.cancel()
            logger.warning(
                "Attempts did not settle within shutdown_drain_seconds (%gs): %d live "
                "episode(s), %d worker(s) still running; runtime cleanup is incomplete",
                drain,
                self.driver.live_attempts,
                sum(not task.done() for task in self._tasks),
            )
        for hook in self.driver.unquiesced:
            logger.warning("Runtime hook %s ignored cancellation and still holds its permit", hook)
        if not waited.settled or self.driver.unquiesced:
            self.settled = False
        # Whether or not the drain settled: its records must not wait on attempts
        # that may never end.
        for rollout_id, (group, rollout) in self._in_flight.items():
            self._discarded(rollout_id, group, Discard.CANCELLED, rollout)
        self._in_flight.clear()
        if waited.settled:
            try:
                cleanup.result()
            except BaseException as failure:
                waited.propagate(failure)
                raise
        waited.propagate()

    async def _drain(self) -> None:
        # Queued sealed rollouts are finalized first: no worker holds them, and a
        # slow abort elsewhere must not keep them from their finish().
        while not self._to_grade.empty():
            _, admitted = self._to_grade.get_nowait()
            self.driver.discard_admitted(admitted)
        # A group in the judge stage is complete and will not be judged now: a
        # stopped judge never makes its group ready. Recorded before the wait, so a
        # drain that times out on a slow judge does not lose them.
        for group_id in list(self._judging):
            self._drop(self._groups[group_id], Discard.CANCELLED)
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.driver.drain()
        for group in self._groups.values():
            members = group.recorded()
            self.counters.discard_group(
                Discard.CANCELLED, group.expected - group.resolved + len(members)
            )
            for rollout in members:
                self._discarded_member(group, rollout, Discard.CANCELLED)
        self._groups.clear()
        self._group_index.clear()
        self._selected.clear()
        self._judging.clear()
        self._training_groups.clear()
        self._lost_to_staleness.clear()
        self._judged.clear()
        self._retained.clear()
        self._ready.clear()
        for queue in (self._to_sample, self._to_group_grade):
            while not queue.empty():
                queue.get_nowait()

    def _window_gauges(self) -> dict[str, float]:
        """The async window's inputs and hold time; sync has no window to report.

        The last cycle's wait and spare appear once a batch has started training.
        """
        if self._training.mode == "sync":
            return {}
        cycle = {
            "admission/batch_wait_seconds": self._last_wait,
            "admission/ready_spare_groups": self._last_spare,
        }
        # The pace's bound in whole rollouts, and the same bound in groups, fractions
        # of one included, so the two gauges always agree.
        rollouts = math.floor(self._window() * self._training.group_size)
        return {
            "admission/committed_groups": float(self._committed()),
            "admission/committed_rollouts": float(self._committed_rollouts()),
            "admission/window_groups": rollouts / self._training.group_size,
            "admission/window_rollouts": float(rollouts),
            "admission/group_survival": float(self._survival()),
            "admission/window_held_seconds": self._account_hold(),
            **{key: float(value) for key, value in cycle.items() if value is not None},
        }

    def snapshot(self) -> dict[str, float]:
        now = time.monotonic()
        return {
            "pipeline_calls/sample": float(self._sampling),
            "pipeline_calls/grade": float(self._grading),
            "pipeline_calls/grade_group": float(
                0 if self._group_grading is None else self._group_grading.calls
            ),
            "queue/sample_dispatch_rollouts": float(self._to_sample.qsize()),
            "queue/grade_dispatch_rollouts": float(self._to_grade.qsize()),
            "queue/group_grading_groups": float(self._to_group_grade.qsize()),
            "groups/ready": float(len(self._ready) + len(self._retained)),
            "groups/selected": float(len(self._selected)),
            "groups/outstanding": float(len(self._groups)),
            "groups/judging": float(len(self._judging)),
            # Still gathering members. `_groups` holds every live group, so the ones
            # in the judge stage, those complete but unselected (`ready`) and the
            # selected batch come out of it -- leaving the four disjoint and summing
            # to the total. Floored at zero: the reads are atomic here, but a
            # negative would be a lie rather than a diagnosis.
            "groups/filling": float(
                max(
                    0,
                    len(self._groups)
                    - len(self._judging)
                    - len(self._ready)
                    - len(self._retained)
                    - len(self._selected),
                )
            ),
            # Graded rollouts, split by what each is waiting for: more rollouts to
            # fill its group, or the optimizer. One number for both hid which half
            # was the holdup. Each falls as well as rising -- a group rejected for
            # training takes its members with it -- and neither is a fault.
            "queue/graded_rollouts": float(
                sum(
                    len(group.members)
                    for group_id, group in self._groups.items()
                    if group_id not in self._selected
                )
            ),
            "queue/trainable_rollouts": float(
                sum(
                    len(group.members)
                    for group_id, group in self._groups.items()
                    if group_id in self._selected
                )
            ),
            **self._window_gauges(),
            "attempts/live": float(self.driver.live_attempts),
            **self.capacity.snapshot(),
            "pipeline/elapsed_s": now - self._started,
            **self.counters.gauges(),
            **self.driver.gauges(),
            **{
                key: value
                for name, duration in self._durations.items()
                for key, value in duration.gauges(f"pipeline/{name}").items()
            },
        }
