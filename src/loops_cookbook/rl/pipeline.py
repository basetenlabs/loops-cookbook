"""Group admission, the freshness window and batch selection, on top of the attempt driver.

The pipeline decides which rollouts to dispatch and when a group can train. The
attempt driver owns every attempt's lifecycle. The capacity broker owns the sampling
and verification permits, and start admission.

For a rollout runtime with a group grader, a complete group passes one more stage
before it is ready. Before every try of the grader, the pipeline runs these checks::

    drop stale members -> size >= min_group_size? -> grade_group()

Once the grader has scored the group, the pipeline runs the no-variance check. So the
grader is never paid to score a group that is about to be dropped. Every group dropped
before training, here or at selection, goes through ``_drop``.
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
    """Groups were rejected for training so many times in a row that training can't progress."""


def _unusable_streak(
    count: int, rewards: collections.Counter[float], undersized: int, untrainable: int = 0
) -> str:
    """Describe what a streak of unusable groups was made of, and what would change it.

    A group whose rollouts all scored the same carries no advantage to learn from. The
    stop can't tell a solved task from a stuck policy or a constant verifier, so it
    reports the rewards those groups shared instead of guessing.
    """
    parts = []
    if rewards:
        shared = ", ".join(
            f"{reward:g} in {n}"
            for reward, n in sorted(rewards.items(), key=lambda item: (-item[1], -item[0]))
        )
        parts.append(
            f"{sum(rewards.values())} gave every rollout one reward ({shared}), so they carry "
            "nothing to learn from. If the tasks are solved, training is done. Otherwise, "
            "raise [sampling] temperature or [training] group_size, or check the verifier"
        )
    if undersized:
        parts.append(
            f"{undersized} had too few usable rollouts (group_size - max_group_losses). Check "
            "rollout failures and the freshness settings"
        )
    if untrainable:
        parts.append(
            f"{untrainable} had an advantage only on rollouts whose agent discarded every "
            "generation, so nothing in them trains"
        )
    return f"{count} consecutive unusable groups: " + ". ".join(parts) + "."


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
        """Count discarded rollouts. A group rejection counts its surviving members separately."""
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

    ``members`` are its sealed, graded rollouts. With a group grader, ``transcripts``
    holds each member's transcript, by rollout ID, until the grader scores the group.
    Until then, no member has its final reward.
    """

    group_id: str
    task: TaskRef
    expected: int
    members: list[Rollout] = field(default_factory=list)
    resolved: int = 0
    ready_at: float | None = None
    # When the group's first member and its last were resolved, which means graded or
    # discarded. The gap is how long the first finishers waited for the rest, and they
    # age while they wait.
    first_resolved_at: float | None = None
    completed_at: float | None = None
    transcripts: dict[str, SealedTranscript] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.resolved == self.expected

    def evidence(self) -> list[MemberEvidence]:
        """Return what the group grader reads, which is each member beside its transcript."""
        return [MemberEvidence(m, self.transcripts[m.identity.rollout_id]) for m in self.members]

    def recorded(self) -> list[Rollout]:
        """Return every member as its record shows it, with no reward until the grader scores."""
        if not self.transcripts:
            return list(self.members)
        return [replace(member, reward=None) for member in self.members]


# These drops mean the group can't train, and they extend the unusable-groups streak.
_UNUSABLE = frozenset({Discard.GROUP_TOO_SMALL, Discard.NO_VARIANCE, Discard.NOTHING_TO_TRAIN})
_STALE = frozenset({Discard.STALE_BEFORE_GRADING, Discard.STALE_BEFORE_TRAINING})
# These drops pass no verdict on the group, so a resume samples it again.
_UNTRAINED = frozenset({Discard.TRAINING_ABORTED, Discard.CANCELLED})
# The async survival estimate covers the last this many batches' worth of groups. It
# never falls below this share, so a run of drops can't make the expected losses
# unbounded.
_SURVIVAL_BATCHES = 4
_MIN_SURVIVAL = Fraction(1, 4)


def _admission_window(fresh: Fraction, survival: Fraction, batch: int) -> Fraction:
    """Return ``fresh`` groups plus the groups ``survival`` says will drop.

    The drops are counted up to one ``batch`` and rounded up to whole groups. A
    replacement is a whole group, and a fraction of one would hold back the
    replacement it was counted for.

    Drops rise as a policy degrades or saturates. Without this cap, the window would
    widen most when the policy is least stable, and each extra group would train later.
    A batch short of replacements takes groups from the batch admitted after it. A run
    that drops more than a batch is better slowed than fed staler groups.

    The arithmetic is exact, because float error would round some windows up a group.
    """
    return fresh + math.ceil(min(fresh / survival - fresh, Fraction(batch)))


def _admission_limit(needed: int, judged: int, survival: Fraction, batch: int) -> Fraction:
    """Return how many live groups not in training the steps left can use.

    ``needed`` is the number of groups still to find. The ``judged`` groups passed their
    check and can now drop only by staleness, so the drop allowance covers the rest
    alone. If it covered judged groups too, a batch already selected would admit one
    more batch that never trains.
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
    """Return a rollout's length, for slicing dispositions by it. It is unknown before sealing."""
    if rollout is None:
        return {"generations": None, "sampled_tokens": None}
    return {
        "generations": len(rollout.samples),
        "sampled_tokens": sum(len(sample.completion_ids) for sample in rollout.samples),
    }


class ReadyGroups:
    """Groups ready to train, with the one closest to going stale first.

    A group goes stale with its oldest member. ``pop`` returns the group whose
    members' oldest policy version is lowest, then the one dispatched first, then the
    one made ready first. A group with no members sorts first, and selection drops it.
    Completion order would let a long group, sampled early and finished late, wait
    behind newer, shorter groups until it aged out of the off-policy window.
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
    """Hold admitted groups through selection until training commits or discards them."""

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
        # This receives one record per discarded rollout. The runner writes the trained
        # ones from ``trained_record``.
        self._on_rollout = on_rollout
        self._training, self._rollout = config.training, config.rollout
        self._training_groups: set[str] = set()
        # Rollouts dispatched and not yet resolved, with each rollout once it sealed.
        # A shutdown records these as cancelled.
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
        # Groups a checkpoint left untrained, in task order. They are dispatched before
        # any new group.
        self._replay: collections.deque[tuple[int, Group]] = collections.deque()
        for index in range(initial_group_index):
            group = next(self._stream)
            if index in replay:
                self._replay.append((index, group))
        self._next_group_index = initial_group_index
        self._group_index: dict[str, int] = {}  # Each live group's place in the stream.
        # Dispatched groups that neither trained nor were dropped for good. A shutdown
        # cancels them, so a resume must sample them again.
        self._untrained: set[int] = set()
        self._started_groups: set[str] = set()
        # Whether each recent group passed its first size and variance check (True) or
        # was dropped as unusable (False). This gives the share expected to survive,
        # which sizes the async window's drop allowance. Each group is recorded once,
        # when judged, so a good group counts as soon as a bad one would. A later
        # recheck can fail only by staleness.
        self._outcomes: collections.deque[bool] = collections.deque(
            maxlen=_SURVIVAL_BATCHES * self._training.groups_per_step
        )
        # The floored share of those outcomes, or 1 before any. It is set when an
        # outcome is recorded, because every admission check reads it.
        self._survival_share = Fraction(1)
        self._judged: set[str] = set()
        # Groups that lost a member to staleness before they were judged. Their fate
        # reflects how the window was sized, not how the task goes, so it stays out of
        # the estimate.
        self._lost_to_staleness: set[str] = set()
        self._groups: dict[str, Group] = {}
        self._selected: set[str] = set()
        self._to_sample: asyncio.Queue[_Unit] = asyncio.Queue(maxsize=self._training.group_size)
        self._to_grade: asyncio.Queue[tuple[_Unit, Admitted]] = asyncio.Queue()
        self._ready = ReadyGroups()
        # Valid groups from a batch that needed replacements. They go first next time.
        self._retained: collections.deque[Group] = collections.deque()
        self._changed, self._ready_changed = asyncio.Event(), asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._sampling = self._grading = self._sync_demand = 0
        # Batches this pipeline committed, counted from ``initial_step``. The runner
        # trains until the run has made ``max_steps`` updates, including a resume's
        # earlier ones.
        self._initial_step = initial_step
        self._committed_batches = 0
        self._fatal: Exception | None = None
        self._stopping = False
        self._consecutive_unusable_groups = 0
        self._streak_rewards: collections.Counter[float] = collections.Counter()
        # Graded rewards of groups dropped as unusable since the last update took them.
        # The same report also takes those members' metadata, but not their samples.
        self._unusable_rewards: list[float] = []
        self._unusable_metadata: list[Mapping[str, JsonValue]] = []
        self._streak_undersized = 0
        self._streak_untrainable = 0
        self._started = time.monotonic()
        # Time the async window held new groups back while fewer than
        # max_concurrent_rollouts attempts were admitted. That is rollout capacity left
        # unused for the sake of freshness. It counts admitted attempts, not sampling
        # ones, because idle slots behind sandbox start-up are not the window's doing.
        # It is kept since the start, and since the last update took it.
        self._held_since: float | None = None
        self._held_seconds = 0.0
        self._held_taken = (self._started, 0.0)
        # The last cycle, for choosing admission_steps_ahead. It records how long the
        # trainer waited for its batch, and how many complete groups beyond a batch
        # there were when it asked. It is measured from the runner's calls. The trainer
        # is free when the runner asks for a batch, and it has the batch at
        # start_training. An aborted cycle measures nothing.
        self._freed: tuple[float, int] | None = None  # When, and how many spare groups.
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
            raise ValueError(
                "a group grader is passed exactly when the rollout runtime declares one"
            )
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
        # Complete groups in the judge stage, queued for the group grader or being judged.
        self._judging: set[str] = set()
        self.settled = True
        """False once a shutdown leaves an attempt unsettled. The run is then unclean."""

    def _stop(self, error: Exception) -> None:
        if self._fatal is not None:
            return
        logger.error("Stopping pipeline: %s", error)
        self._fatal = error
        self._stopping = True
        self._changed.set()
        self._ready_changed.set()

    def _sealed(self) -> None:
        # A rollout now exists. It is counted before evidence checks can discard it.
        self.counters.sampled += 1

    def _capacity_changed(self) -> None:
        self._changed.set()
        self._account_hold()

    def _account_hold(self, now: float | None = None) -> float:
        """Start or stop the held clock on every change that can move it, and return the total."""
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
        """Return the share of wall time since the last call that the window held slots idle."""
        now = time.monotonic()  # The total and the interval both end at this instant.
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
        self._account_hold()  # Survival sizes the window, so this can close it.

    def _survival(self) -> Fraction:
        """Return the floored share of recent groups judged usable, or 1 before any."""
        return self._survival_share

    def _committed(self) -> int:
        """Count started groups not yet training. Each will take a place in a later batch."""
        return len(self._started_groups - self._training_groups)

    def _committed_rollouts(self) -> int:
        """Return how much of the window the started groups not yet training hold, in rollouts.

        A complete group holds its whole place, because a batch takes it however many
        members it has left. A group still filling holds only its rollouts not yet
        resolved. If the count were by groups, a group whose last member ran long would
        keep its whole place while the rest sat finished, and a window of such groups
        would hold the sampler idle. Instead, the slots its finished members leave go
        to new groups. They are lent, not freed. The group holds its whole place again
        once complete, so the count can pass the window by what filling groups had
        finished. The bound on groups in ``_room`` holds regardless.
        """
        held = 0
        for group_id in self._started_groups - self._training_groups:
            group = self._groups.get(group_id)
            if group is not None:
                held += group.expected if group.complete else group.expected - group.resolved
        return held

    def _window(self, steps: float | None = None) -> Fraction:
        """Return how many started groups not yet training async admits, drops included.

        The fresh groups are the ``steps`` batches after the one in training. With no
        batch in training, they are the batch being assembled and the batches after
        it. ``steps`` defaults to ``steps_admitted_ahead``, which is
        ``max_steps_off_policy`` unless ``admission_steps_ahead`` lowers it by
        fractions of a batch. The count is in groups, because a batch takes
        ``groups_per_step`` of them however many members each has left. It adds the
        groups that recent drops say won't train, at most a batch of them. The result
        is exact, because a fraction of a batch is a fraction of a group's rollouts.
        """
        ahead = self._training.steps_admitted_ahead if steps is None else steps
        batches = Fraction(ahead).limit_denominator(10**6) + (0 if self._training_groups else 1)
        fresh = batches * self._training.groups_per_step
        return _admission_window(fresh, self._survival(), self._training.groups_per_step)

    def _room(self) -> bool:
        """Whether there is room for one more group, by two bounds.

        The first bound is the pace. The new group's rollouts and the committed ones
        (``_committed_rollouts``) must fit the window, so the finished members of
        groups still filling free their slots. The second bound is the guarantee. The
        started groups must fit the window that ``max_steps_off_policy`` sizes,
        because each takes a batch place however few members it has left. So no group
        starts that could only train stale. With ``admission_steps_ahead`` unset, the
        two windows are one, and the guarantee binds first, so admission is by groups
        alone.
        """
        size = self._training.group_size
        paced = self._committed_rollouts() + size <= self._window() * size
        return paced and self._committed() < self._window(self._training.steps_off_policy)

    def _needed(self) -> bool:
        """Whether the steps left to train can use one more group.

        Without a step limit, they always can.

        The groups still to find are the batches left, less the batch in training,
        which is certain to train. The live groups not in training compete for those
        places, with an allowance for the groups recent drops say won't train
        (``_admission_limit``). So a dropped group frees its place for a replacement.
        Only group admission asks this. Every admitted group is within the limit, so
        start admission doesn't need to ask.
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
        """Return how many slots a group may fill under start admission.

        A started group, and every group in sync mode, fills its slots. In async mode,
        an unstarted group may start while the window has room, and then it starts at
        full concurrency.
        """
        if self._training.mode == "sync" or group_id in self._started_groups or self._room():
            return self._rollout.max_concurrent_rollouts
        return 0

    def started(self, group_id: str) -> None:
        """Record that one of a group's attempts took a sampling slot.

        The capacity broker calls this on each admission. A started group fills its
        slots from then on. It holds a place in the async window until its batch starts
        training or the group is dropped.
        """
        self._started_groups.add(group_id)

    def order(self, group_id: str) -> int:
        """Return a group's admission order, which is the order groups were dispatched in.

        A group that is no longer live has no attempt waiting, and it would sort after
        all the others.
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
            # The rollout has been handed on. Sampling the next one can take as long as
            # a rollout takes, and this worker must not keep this attempt alive meanwhile.
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
            # The rollout is graded and its finish is under way, so don't hold it while
            # the next one is awaited.
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
        """Check stale members and size before every try, then run the grader, then check variance.

        A failed try is tried again, within the rollout runtime's ``grading_retries``,
        on the group as it stands after the backoff. A member that went stale in the
        meantime is dropped first, and a group left too small is never scored.
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
                # This is a grading failure, like a failed grade(). It counts against the
                # failure budget, not as an unusable group. The driver notes the members'
                # cause, which their records carry.
                ids = [member.identity.rollout_id for member in group.members]
                self.driver.group_failed(group.group_id, error, ids)
                self._drop(group, Discard.GRADING_FAILED)
                return
            self.driver.group_scored()
            # Make new rollouts instead of changing the old ones, so the evidence the
            # grader read stays as it was.
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
        """Mark a group that trained, or was dropped for a reason a resume wouldn't undo."""
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
        """Return why a group of graded members can't train, or ``None`` when it can."""
        if len(group.members) < self._training.min_group_size:
            return Discard.GROUP_TOO_SMALL
        rewards = [rollout.reward for rollout in group.members]
        if len(set(rewards)) == 1:
            return Discard.NO_VARIANCE
        # Rollouts whose agent discarded every generation count in the baseline, but
        # they train nothing. The group needs a member that trains with an advantage.
        # This baseline is computed exactly as grpo.advantages computes it.
        mean = fmean(r for r in rewards if r is not None)
        if not any(r.trains and r.reward != mean for r in group.members):
            return Discard.NOTHING_TO_TRAIN
        return None

    def _trainable(self, group: Group) -> Discard | None:
        """Recheck a group at selection for members gone stale, then for size and variance."""
        self._drop_stale(group)
        return self._unusable(group)

    def _drop(self, group: Group, reason: Discard) -> None:
        """Drop a group before training. Every such drop takes this one path.

        The group's members are counted and recorded as discarded, and the group is
        released. A group that can't train, because it is too small or has no
        variance, also extends the streak that ``max_consecutive_unusable_groups``
        stops. Reaching that limit raises ``UnusableGroupsError`` and stops the
        pipeline.
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
        """Return the graded rewards of groups dropped as unusable since the last call.

        A group is unusable when it has no variance or too few members. The update
        that calls next reports these rewards.
        """
        rewards, self._unusable_rewards = self._unusable_rewards, []
        return rewards

    def take_unusable_metadata(self) -> list[Mapping[str, JsonValue]]:
        """Return the metadata of the same graded members as ``take_unusable_rewards``.

        It covers the time since the last call, and the update that calls next reports it.
        """
        metadata, self._unusable_metadata = self._unusable_metadata, []
        return metadata

    def _extend_streak(self, group_id: str, reason: Discard, members: list[Rollout]) -> None:
        self._consecutive_unusable_groups += 1
        count = self._consecutive_unusable_groups
        reward = members[0].reward if members else None
        if reason is Discard.NO_VARIANCE and reward is not None:  # A graded member always has one.
            self._streak_rewards[reward] += 1
        elif reason is Discard.NOTHING_TO_TRAIN:
            self._streak_untrainable += 1
        else:
            self._streak_undersized += 1
        if count == 1 or count % 5 == 0:
            logger.warning(
                "%d consecutive unusable groups. The latest, group %s, was dropped as %s",
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
        """Return the ``rollouts.jsonl`` record of a rollout the update at ``step`` trained on."""
        note = self.driver.note(rollout.identity.rollout_id)
        return {
            "rollout_id": rollout.identity.rollout_id,
            "attempt_id": rollout.identity.attempt_id,
            "group_id": rollout.identity.group_id,
            "task_id": rollout.task.id,
            "task_digest": rollout.task_digest,
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
                "task_digest": None if rollout is None else rollout.task_digest,
                "reward": None if rollout is None else rollout.reward,
                # As in a trained record, this is the oldest version it sampled at, not
                # the trainer's version when it was dropped. It is None when the rollout
                # sampled nothing.
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
        """Select one batch. In sync mode, only its groups are admitted, replacements included."""
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
        """Recheck a batch before advantages. Valid groups keep their turn if it needs new ones."""
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
        """Move a complete selected batch out of the window, and keep it until commit or abort."""
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
        """Discard a selected batch whose training failed, as ``training_aborted``."""
        self._sync_demand = 0
        self._freed = None
        for group in groups:
            if group.group_id in self._selected:
                self._drop(group, Discard.TRAINING_ABORTED)

    async def __aenter__(self) -> Self:
        # A phase the runner owns has one worker per slot. When the rollout runtime
        # acquires its own phases, the broker is the real limit, so the workers only
        # need never to be the bottleneck. The group_size floor keeps the dispatcher
        # from blocking on its one-group queue.
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
            # Each call holds a verification slot, so no more calls than this can run.
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
        # Only shutdown cancels a worker, and it stops the pipeline first. Any other
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
        """The first place in the task stream that no group has been dispatched from."""
        return self._next_group_index

    @property
    def pending_group_indices(self) -> list[int]:
        """Places below ``next_group_index`` whose groups have not trained, in order.

        A shutdown cancels the groups in flight, so a resume from a checkpoint replays
        these first. They include groups admitted for the next update, and groups whose
        update was aborted. Groups dropped as unusable, as stale or for failed grading
        are not replayed.
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
        # report. It is raised once the drain has settled or timed out.
        waited = await outlast(cleanup, drain)
        if not waited.settled:
            # This stops the waiting, not the cleanup. Each attempt's abort() or
            # finish() still runs to its own cleanup deadline in the background.
            cleanup.cancel()
            logger.warning(
                "Attempts did not settle within shutdown_drain_seconds (%gs). %d live "
                "episode(s) and %d worker(s) are still running, so rollout runtime cleanup "
                "is incomplete",
                drain,
                self.driver.live_attempts,
                sum(not task.done() for task in self._tasks),
            )
        for hook in self.driver.unquiesced:
            logger.warning(
                "Rollout runtime hook %s ignored cancellation and still holds its permit", hook
            )
        if not waited.settled or self.driver.unquiesced:
            self.settled = False
        # Write these records whether or not the drain settled. They must not wait on
        # attempts that may never end.
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
        # Queued sealed rollouts are finalized first. No worker holds them, and a slow
        # abort elsewhere must not keep them from their finish().
        while not self._to_grade.empty():
            _, admitted = self._to_grade.get_nowait()
            self.driver.discard_admitted(admitted)
        # A group in the judge stage is complete and won't be judged now, because a
        # stopped judge never makes its group ready. These groups are recorded before
        # the wait, so a drain that times out on a slow judge doesn't lose them.
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
        """Return the async window's inputs and hold time. Sync has no window to report.

        The last cycle's wait and spare groups appear once a batch has started training.
        """
        if self._training.mode == "sync":
            return {}
        cycle = {
            "admission/batch_wait_seconds": self._last_wait,
            "admission/ready_spare_groups": self._last_spare,
        }
        # The pace's bound in whole rollouts, and the same bound in groups, including
        # fractions of a group, so the two gauges always agree.
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
        """Return the pipeline's live gauges, such as queue lengths, group counts and capacity."""
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
            # Groups still gathering members. `_groups` holds every live group. The
            # groups in the judge stage, the complete but unselected ones (`ready`) and
            # the selected batch come out of it, so the four counts are disjoint and sum
            # to the total. The count is floored at zero. The reads are atomic here,
            # but a negative count would be a lie, not a diagnosis.
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
            # Graded rollouts, split by what each is waiting for, which is more
            # rollouts to fill its group or the optimizer. One number for both would
            # hide which half was the holdup. Each count falls as well as rises,
            # because a group rejected for training takes its members with it. Neither
            # is a fault.
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
