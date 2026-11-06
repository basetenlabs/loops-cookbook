"""Harbor as a rollout runtime: supplied agents run in Harbor trials.

Harbor owns trial execution; this adapter binds each trial's agent to its attempt,
takes the attempt's sampling and verification holds from Harbor's trial hooks, and
parks each trial at its final verification until ``grade()`` is admitted, so the
final verifier never runs before verification capacity is held or before the
stale-before-grading check.
"""

from __future__ import annotations

import asyncio
import collections
import inspect
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

import yaml
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import JobConfig
from harbor.models.task.verifier_mode import (
    VerifierEnvironmentMode,
    resolve_step_verifier_mode,
    resolve_task_verifier_mode,
)
from harbor.models.trial.config import SourceTrialConfig, TrialConfig
from harbor.models.verifier.result import VerifierResult
from harbor.trial.hooks import TrialEvent
from harbor.trial.queue import TrialQueue
from harbor.trial.trial import Trial
from harbor.utils.import_path import import_symbol
from pydantic import BaseModel

from loops_cookbook.cancellation import Waited, outlast
from loops_cookbook.evidence import JsonValue, Rollout, TaskRef, TrajectoryError, json_value
from loops_cookbook.harbor import attempts, integration
from loops_cookbook.harbor.oom import ReportsOomKills
from loops_cookbook.model_access.api import ModelAccessMode
from loops_cookbook.providers.blaxel.build_file import BUILD_TOML_NAME, check_task_build_region
from loops_cookbook.providers.blaxel.images import blaxel_image, workspace_image
from loops_cookbook.resources.api import ProviderDeadlines, SandboxCapacity, SandboxResources
from loops_cookbook.resources.capacity import oversized
from loops_cookbook.runtime.api import (
    AttemptAbortReason,
    AttemptContext,
    EpisodeDisposition,
    EpisodeResult,
    ExcludedReward,
    GradeResult,
    ImageLookup,
    NamedFailure,
    PermanentGradingFailure,
    PermanentRolloutFailure,
    Phase,
    RunLimits,
    RuntimeCapabilities,
    RuntimeContext,
    RuntimeMetrics,
)
from loops_cookbook.runtime.context import import_attribute

logger = logging.getLogger(__name__)


# The runtime metric for a trial's sandbox OOM kills.
OOM_KILLS = "harbor/sandbox_oom_kills"
# Blaxel image lookups at once when prepare checks the images tasks start from.
IMAGE_LOOKUPS = 16

# grading_retries stays 0: a Harbor job states its regrades once, in its own retry:
# block, which the runtime applies to a separate verifier.
CAPABILITIES = RuntimeCapabilities(
    model_access_modes=frozenset({ModelAccessMode.DIRECT}),
    internally_acquired_phases=frozenset({Phase.SAMPLING, Phase.VERIFICATION}),
)


class HarborRolloutError(NamedFailure, RuntimeError):
    """Harbor could not produce a usable trial result.

    Its discard metric names Harbor's exception: ``grading_failed/verifier_timeout``.
    """

    def __init__(
        self,
        message: str,
        *,
        exception_type: str | None = None,
        trial_name: str | None = None,
        trial_uri: str | None = None,
    ):
        self.exception_type = exception_type
        context = f"; trial={trial_name}; evidence={trial_uri}" if trial_name or trial_uri else ""
        super().__init__(message + context, name=exception_type)


class AgentBindingError(ValueError):
    """The supplied Harbor agent does not implement the Loops binding."""


class SandboxStopFailure(RuntimeError):
    """A Harbor environment failed to stop; the attempt's cleanup failed with it."""


# The trial and its final verifier result, before the environments still running stop (a
# separate verifier's agent sandbox is already gone): a reason to drop the reward, or None.
RewardGate = Callable[[Trial, VerifierResult], Awaitable[str | None]]


class RewardRejected(NamedFailure, ExcludedReward):
    """The job's reward gate dropped a reward: left out of training, never a zero.

    Its discard metric names the gate's reason: ``grading_failed/<reason>``.
    """


class RewardGateError(RuntimeError):
    """The reward gate raised: ``grading_failed/reward_gate``."""


class RewardGateResultError(TypeError):
    """A reward gate returned something other than a reason or None."""


# A reason names its discard metric, grading_failed/<reason>, so it is one already.
_REASON = re.compile(r"[a-z0-9_]+")


class RewardUngated(NamedFailure, PermanentGradingFailure):
    """The reward gate raised, so the reward never passed it; a failure, not a rejection."""


@dataclass
class _GateRequest:
    """A reward awaiting the gate, which grade() runs; ``answer`` tells the trial."""

    trial: Any
    result: VerifierResult
    trial_result: Any  # Harbor's result for the trial when it published
    answer: asyncio.Future[str | None]


@dataclass
class _Pending:
    attempt: AttemptContext
    config: TrialConfig
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    allow_grade: asyncio.Event = field(default_factory=asyncio.Event)
    separate: bool = False
    # Harbor carried a separate-verifier trial on to its verification.
    reached_verification: bool = False
    # Harbor has accepted the trial: its setup hook ran, so it holds a trial slot.
    accepted: bool = False
    source_dir: Path | None = None
    execution: asyncio.Task[Any] | None = None
    regrade: asyncio.Task[Any] | None = None
    discarded: asyncio.Event = field(default_factory=asyncio.Event)
    snapshot: Any = None
    rewarded: asyncio.Event = field(default_factory=asyncio.Event)
    reward: VerifierResult | None = None
    rewarded_trial: Any = None
    rejection: str | None = None
    # With a gate, each published reward waits here for grade() to run the gate on it.
    gated: bool = False
    gate_requests: asyncio.Queue[_GateRequest] = field(default_factory=asyncio.Queue[_GateRequest])
    # A reward published before grade() began, for sample()'s checks: a multi-step
    # early stop's, waiting for its gate.
    early: tuple[Any, VerifierResult] | None = None
    # Why the gate failed: Harbor's multi-step trial does not record it.
    gate_error: Exception | None = None
    stop_failures: list[BaseException] = field(default_factory=list[BaseException])
    # The sandbox kernel's OOM-kill count at the end of the last agent phase read.
    oom_kills: int | None = None
    holds: dict[Phase, AbstractAsyncContextManager[None]] = field(
        default_factory=dict[Phase, AbstractAsyncContextManager[None]]
    )

    async def admit(self) -> None:
        await self.attempt.services.capacity.admit()

    async def hold(self, phase: Phase) -> None:
        if phase in self.holds:
            return
        hold = self.attempt.services.capacity.hold(phase)
        await hold.__aenter__()
        self.holds[phase] = hold

    async def release(self, phase: Phase) -> None:
        hold = self.holds.pop(phase, None)
        if hold is not None:
            await hold.__aexit__(None, None, None)

    async def release_all(self) -> None:
        for phase in list(self.holds):
            await self.release(phase)

    @property
    def tasks(self) -> list[asyncio.Task[Any]]:
        return [task for task in (self.execution, self.regrade) if task is not None]

    async def publish(self, trial: Any, result: VerifierResult) -> None:
        """Harbor can no longer change this reward; hand it over, before any sandbox stops.

        Without a gate it is the outcome at once. With one, the trial waits here, its
        sandbox still up, while grade() runs the gate on it, inside grade()'s deadline;
        the gate's failure is raised here too, so Harbor records it, and retries it on a
        regrade. Everything after this is teardown, which ``finish()`` waits for.
        """
        if self.reward is not None:
            return
        if not self.gated:
            await self.hand_over(trial.result, result, None)
            return
        request = _GateRequest(
            trial, result, trial.result, asyncio.get_running_loop().create_future()
        )
        if not self.allow_grade.is_set():
            # A multi-step early stop publishes during sample(), before grade() is
            # admitted: the trial waits for it holding nothing, as a final step does.
            await self.release_all()
            self.early = (request.trial_result, result)
            self.ready.set()
        self.gate_requests.put_nowait(request)
        await request.answer

    async def hand_over(
        self, trial_result: Any, result: VerifierResult, rejection: str | None
    ) -> None:
        """Fix the trial's outcome and release its capacity; the rest is teardown."""
        await self.release_all()
        self.reward, self.rewarded_trial, self.rejection = result, trial_result, rejection
        self.rewarded.set()
        self.ready.set()

    def publisher(self, trial: Any) -> Callable[[VerifierResult], Awaitable[None]]:
        async def publish(result: VerifierResult) -> None:
            await self.publish(trial, result)

        return publish

    async def until_graded(self) -> None:
        """Park until grade() is admitted; a discard ends the trial instead."""
        grading = asyncio.create_task(self.allow_grade.wait())
        discarding = asyncio.create_task(self.discarded.wait())
        try:
            await asyncio.wait({grading, discarding}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            grading.cancel()
            discarding.cancel()
        if not self.allow_grade.is_set():
            raise asyncio.CancelledError


async def _next_request(pending: _Pending, running: asyncio.Task[Any]) -> _GateRequest | None:
    """The trial's next reward for the gate, or None once the trial has ended without one."""
    getting = asyncio.ensure_future(pending.gate_requests.get())
    try:
        await asyncio.wait({getting, asyncio.shield(running)}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        if not getting.done():
            getting.cancel()
    if getting.done() and not getting.cancelled():
        return getting.result()
    return None if pending.gate_requests.empty() else pending.gate_requests.get_nowait()


def _answer(request: _GateRequest, outcome: str | BaseException | None) -> None:
    """Tell the waiting trial the gate's outcome, unless it stopped waiting."""
    if request.answer.done():
        return
    if isinstance(outcome, BaseException):
        request.answer.set_exception(outcome)
    else:
        request.answer.set_result(outcome)


class HarborRuntime:
    def __init__(
        self,
        *,
        job_path: Path,
        reward_key: str | None,
        trials_dir: Path,
        reward_gate: RewardGate | None = None,
        limits: RunLimits,
        metrics: RuntimeMetrics,
        provider_regions: Mapping[str, str],
        provider_capacity: Mapping[str, SandboxCapacity] | None = None,
        provider_settings: Mapping[str, BaseModel] | None = None,
        provider_images: Mapping[str, ImageLookup] | None = None,
    ) -> None:
        self._job_path = job_path
        self._provider_images = provider_images or {}
        self._provider_regions = provider_regions
        self._provider_capacity = provider_capacity or {}
        self._provider_settings = provider_settings or {}
        self._reward_key = reward_key
        self._reward_gate = reward_gate
        # What Harbor recorded for each sampling trial, verbatim: reporting only.
        self._trial_exceptions: collections.Counter[str] = collections.Counter()
        self._sampling_trials = 0
        self._oom_killed_trials = 0
        self._trials_dir = trials_dir
        self._limits = limits
        self._metrics = metrics
        self._templates: dict[str, TrialConfig] = {}
        self._digests: dict[str, str] = {}
        self._pending: dict[str, _Pending] = {}
        self._by_trial_name: dict[str, _Pending] = {}
        self._queue: TrialQueue | None = None
        self._regrade_queue: TrialQueue | None = None
        self._excluded: frozenset[str] = frozenset()
        self._capabilities = CAPABILITIES
        # The provider whose sandboxes the job's environment provisions; None when local.
        self._provider: str | None = None

    async def _check_build_region(self, template: TrialConfig) -> None:
        """A task's own blaxel.toml may not build its image away from its sandboxes."""
        own = (await integration.task_environment_dir(template)) / BUILD_TOML_NAME
        region = self._provider_regions.get("blaxel")
        if region is not None and own.is_file():
            check_task_build_region(own, region)

    async def _check_blaxel_images(self, provider: str | None, images: Mapping[str, str]) -> None:
        """Refuse a task whose ``docker_image`` names a Blaxel image (``blaxel://<image>``)
        it cannot start from: on a job not backed by Blaxel, which would pull it as a Docker
        image, or one the Blaxel workspace does not hold under that tag, which would fail
        every trial. ``images`` maps each such task to its reference after the scheme.

        Blaxel looks images up only by name, so only ``sandbox/<name>:<tag>`` is checked:
        one lookup per image name, whatever tags name it, and none when no task names one.
        Any other form, and any image Blaxel cannot answer for, goes unchecked and says so."""
        if not images:
            return
        if provider != "blaxel":
            raise ValueError(
                "tasks name a Blaxel image (blaxel://) as their docker_image, which only a "
                "Blaxel-backed job can start from; this job's environment is "
                f"{provider or 'not a sandbox provider'}:\n  "
                + "\n  ".join(
                    f"task {task_id}: blaxel://{reference}" for task_id, reference in images.items()
                )
            )
        by_image: dict[str, list[str]] = {}
        for task_id, reference in images.items():
            by_image.setdefault(reference, []).append(task_id)
        lookup = self._provider_images.get("blaxel")
        limit = asyncio.Semaphore(IMAGE_LOOKUPS)

        async def tags(name: str) -> frozenset[str] | None | Exception:
            assert lookup is not None
            async with limit:
                try:
                    return await lookup(name)
                except Exception as error:  # noqa: BLE001 - unchecked, and reported so
                    return error

        refused: dict[str, str] = {}  # reference: why
        unchecked: dict[str, str] = {}
        wanted: dict[str, tuple[str, str]] = {}  # reference: its image's name and tag
        for reference in by_image:
            found = workspace_image(reference)
            if found is None:
                unchecked[reference] = "Blaxel has no lookup for this form"
            elif found[1] is None:
                refused[reference] = "names no tag"
            elif lookup is None:
                unchecked[reference] = "no Blaxel provider to ask"
            else:
                wanted[reference] = (found[0], found[1])
        # One lookup per image: its tags answer for every reference to it.
        async with asyncio.TaskGroup() as group:
            lookups = {
                name: group.create_task(tags(name)) for name in {n for n, _ in wanted.values()}
            }
        for reference, (name, tag) in wanted.items():
            held = lookups[name].result()
            if isinstance(held, Exception):
                unchecked[reference] = f"{type(held).__name__}: {held}"
            elif held is None:
                refused[reference] = f"the workspace has no image {name}"
            elif tag not in held:
                refused[reference] = f"image {name} has tags {', '.join(sorted(held)) or 'none'}"
        if unchecked:
            logger.warning(
                "Blaxel images not checked: %s",
                "; ".join(f"{reference} ({why})" for reference, why in unchecked.items()),
            )
        if refused:
            raise ValueError(
                "tasks name Blaxel images the workspace does not hold:\n  "
                + "\n  ".join(
                    f"blaxel://{reference}: {why} (task {by_image[reference][0]}"
                    + (
                        f" and {len(by_image[reference]) - 1} more)"
                        if len(by_image[reference]) > 1
                        else ")"
                    )
                    for reference, why in refused.items()
                )
            )

    @property
    def capabilities(self) -> RuntimeCapabilities:
        """Harbor acquires its own phases; its model access is the job's agents' mode."""
        return self._capabilities

    async def prepare(self) -> list[TaskRef]:
        """Delegate dataset resolution and task/agent expansion to Harbor."""
        if self._templates:
            return [TaskRef(task_id, digest) for task_id, digest in self._digests.items()]
        job = load_job_config(self._job_path)
        self._provider = provider = _provider_backed(job)
        if provider == "blaxel":
            _require_environment_settings(job)
        if provider == "daytona":
            from loops_cookbook.harbor.daytona import check_lifecycle  # needs the SDK

            check_lifecycle(job.environment.kwargs or {})
        if provider is not None and provider not in self._limits.provider_deadlines:
            raise ValueError(
                f"a {BACKINGS[provider].title}-backed Harbor job needs [providers.{provider}] "
                "in the training config"
            )
        modes = {_validate_agent_binding(agent) for agent in job.agents}
        if len(modes) != 1:
            raise ValueError(
                "every agent in a Harbor job must declare the same model_access; got "
                + ", ".join(sorted(mode.value for mode in modes))
            )
        self._capabilities = replace(
            CAPABILITIES,
            model_access_modes=frozenset(modes),
            model_access_source=", ".join(sorted(mode.value for mode in modes))
            + " (the job's agents' model_access)",
            # The cookbook's count is 0; these are Harbor's, of a separate verifier.
            grading_retries_source=f"{job.retry.max_retries} (Harbor job retry)",
        )
        self._excluded = frozenset(job.retry.exclude_exceptions or ())
        # Derived: the cap covers every live trial the cookbook can admit at once,
        # sampling, started ahead of a sampling slot, parked for grading, or
        # verifying in the trial. A finished trial frees its slot, including one
        # waiting for its regrade. Without the pre-start places, trials parked for a
        # slow verifier could fill the cap and leave none started ahead.
        limits = self._limits
        concurrent = (
            limits.max_concurrent_rollouts
            + limits.prestart_places
            + limits.max_concurrent_verifiers
        )
        logger.info(
            "Harbor trial cap set to %d (%d rollouts + %d pre-start + %d verifiers)",
            concurrent,
            limits.max_concurrent_rollouts,
            limits.prestart_places,
            limits.max_concurrent_verifiers,
        )
        self._metrics.set_gauge("harbor/trial_cap", concurrent)
        job = job.model_copy(update={"n_concurrent_trials": concurrent})
        config = job.model_copy(deep=True)
        config.job_name = f"prepare-{uuid4().hex}"
        config.jobs_dir = self._trials_dir / "preparation"
        templates: dict[str, TrialConfig] = {}
        digests: dict[str, str] = {}
        for index, template in enumerate(await integration.expand_trials(config)):
            if template.install_only or template.is_regrade or template.verifier.disable:
                raise ValueError("RL requires agent execution and an enabled Harbor verifier")
            if template.agent.n_concurrent is not None:
                # Harbor's agent pool would queue a trial inside its sampling hold.
                raise ValueError(
                    "agents[].n_concurrent is not supported: the cookbook's "
                    "[rollout].max_concurrent_rollouts bounds agent execution"
                )
            task_id = f"{index}:{template.task.get_task_id().get_name()}"
            templates[task_id] = template
            digests[task_id] = await integration.task_digest(template)
        if not templates:
            raise ValueError("the Harbor job resolved no trials")
        task_configs: dict[str, Any] = {}
        environments: dict[str, Any] = {}
        sized = provider is not None and provider in self._provider_capacity
        blaxel_images: dict[str, str] = {}
        for task_id, template in templates.items():
            timeouts = await self._timeouts(template, environments=sized)
            images = integration.declared_docker_images(timeouts.task)
            for where, image in images.items():
                if (reference := blaxel_image(image)) is not None:
                    blaxel_images[f"{task_id} ({where})" if where else task_id] = reference
            # An environment on a blaxel:// image starts from it: only the others build.
            builds = any(blaxel_image(image) is None for image in images.values())
            self._check_grading_bound(task_id, timeouts, job)
            if provider is not None:
                _check_build_bound(
                    task_id,
                    timeouts,
                    self._limits.provider_deadlines[provider],
                    provider,
                    image_build=builds,
                )
                task_configs[task_id] = timeouts.task
                for where, environment in timeouts.environments.items():
                    environments[f"{task_id} ({where})" if where else task_id] = environment
            if provider == "blaxel" and builds:
                await self._check_build_region(template)
        if provider == "blaxel":
            _report_sandbox_sizes(job, task_configs)
        await self._check_blaxel_images(provider, blaxel_images)
        if sized:
            assert provider is not None
            _check_sandbox_sizes(environments, provider, self._provider_capacity[provider])
        self._templates, self._digests = templates, digests
        # Sampling trials never retry inside Harbor: a retried trial's samples
        # cannot be separated from the failed try's inside one attempt lease.
        self._queue = integration.HookedTrialQueue(
            self._setup_sampling_trial,
            n_concurrent=job.n_concurrent_trials,
            retry_config=job.retry.model_copy(update={"max_retries": 0}),
        )
        # At least as many regrade slots as cookbook verification capacity, so the
        # queue adds no wait of its own inside grade().
        self._regrade_queue = integration.HookedTrialQueue(
            self._setup_regrade_trial,
            n_concurrent=self._limits.max_concurrent_verifiers,
            retry_config=job.retry,
        )
        logger.info(
            "Harbor regrades a separate verifier up to %d times (the job's retry: block); "
            "the cookbook adds no grading retries of its own",
            job.retry.max_retries,
        )
        return [TaskRef(task_id, digest) for task_id, digest in digests.items()]

    async def _timeouts(
        self, template: TrialConfig, *, environments: bool = False
    ) -> integration.Deadlines:
        probe = template.model_copy(deep=True)
        probe.trial_name = f"preflight-{uuid4().hex}"
        probe.trials_dir = self._trials_dir / "preflight"
        # Harbor constructs, and never starts, the probe's environments.
        _bind_environment(probe, probe.trial_name)
        attempts.bind(
            probe.trial_name,
            attempts.PREFLIGHT,
            self._limits.provider_deadlines,
            self._provider_settings,
        )
        try:
            return await integration.deadlines(probe, environments=environments)
        finally:
            attempts.forget(probe.trial_name)

    def _check_grading_bound(
        self, task_id: str, timeouts: integration.Deadlines, job: JobConfig
    ) -> None:
        """The grading deadline must exceed what Harbor itself allows one grade() call.

        For a separate-verifier regrade with R retries, effective build timeout B and
        verifier timeout V, that is (R + 1) x (B + V) plus Harbor's retry backoff; for
        shared verification, the final step's verifier timeout, plus the build timeout
        when that step's verifier has its own environment. Work Harbor gives no
        deadline, such as sandbox stops, is left to the cookbook deadline itself.
        """
        task = timeouts.task
        steps = task.steps or []
        separate = not steps and (
            resolve_task_verifier_mode(task) == VerifierEnvironmentMode.SEPARATE
        )
        if separate:
            retry = job.retry
            backoff = sum(
                min(retry.min_wait_sec * retry.wait_multiplier**attempt, retry.max_wait_sec)
                for attempt in range(retry.max_retries)
            )
            parts = (timeouts.environment_build, timeouts.verifier)
            bound = (
                None
                if None in parts
                else (retry.max_retries + 1) * sum(p for p in parts if p is not None) + backoff
            )
        elif steps:
            final_separate = (
                resolve_step_verifier_mode(task, steps[-1]) == VerifierEnvironmentMode.SEPARATE
            )
            parts = (timeouts.final_step_verifier,) + (
                (timeouts.environment_build,) if final_separate else ()
            )
            bound = None if None in parts else sum(p for p in parts if p is not None)
        else:
            bound = timeouts.verifier
        limit = self._limits.grading_timeout_seconds
        if bound is None or limit <= bound:
            shown = "unbounded" if bound is None else f"{bound:g}s"
            raise ValueError(
                f"[rollout.grading].timeout_seconds ({limit:g}s) must exceed Harbor's own "
                f"bound for grading task {task_id} ({shown}), with headroom for sandbox "
                "stops and uploads Harbor does not time"
            )

    def _setup_sampling_trial(self, trial: Any) -> None:
        """Bind a sampling trial to its attempt and take its holds from Harbor's hooks."""
        pending = self._by_trial_name[trial.config.trial_name]
        pending.accepted = True
        self._report_live()
        _binding_mode(type(trial.agent))
        bound = trial.agent.bind_loops_services(pending.attempt.services)
        if inspect.isawaitable(bound):
            if inspect.iscoroutine(bound):
                bound.close()
            raise AgentBindingError("bind_loops_services must be a synchronous setter")
        if integration.trial_task_digest(trial) != pending.attempt.spec.task.digest:
            raise PermanentRolloutFailure(
                f"task {pending.attempt.spec.task.id} changed after prepare(); its Harbor "
                "digest no longer matches"
            )
        pending.source_dir = trial.paths.trial_dir
        integration.expose_reward(trial, pending.publisher(trial))
        integration.observe_stops(trial, pending.stop_failures.append)
        steps = trial.task.config.steps or []
        pending.separate = not steps and (
            resolve_task_verifier_mode(trial.task.config) == VerifierEnvironmentMode.SEPARATE
        )
        if pending.separate:
            # Regrade later from Harbor's saved artifacts, if Harbor carries the
            # trial on to its verification. Multi-step trials stay intact because
            # intermediate rewards can control later steps.
            trial.config.verifier = trial.config.verifier.model_copy(update={"disable": True})
            integration.mark_verification_reached(
                trial, lambda: setattr(pending, "reached_verification", True)
            )

        async def environment_start(event: Any) -> None:
            # Admitted before the sandbox starts, so backpressure stops sandboxes
            # starting and the ones up and waiting for a slot stay within the
            # pre-start places; the sampling permit waits for the agent.
            await pending.admit()

        async def agent_start(event: Any) -> None:
            # Already admitted at environment start; the hold takes only the permit,
            # so it never blocks an already-started group or a later phase.
            await pending.release(Phase.VERIFICATION)
            await pending.hold(Phase.SAMPLING)

        async def agent_end(event: Any) -> None:
            await pending.release(Phase.SAMPLING)
            # Harbor emits this in a finally, with the sandbox still up. Report only:
            # the read never raises and never changes the trial.
            environment = trial.agent_environment
            if self._provider is not None and isinstance(environment, ReportsOomKills):
                reading = await environment.oom_kills()
                if reading is not None:
                    pending.oom_kills = reading

        async def verification_start(event: Any) -> None:
            final_step = not steps or len(event.result.step_results or []) == len(steps)
            # Harbor emits AGENT_END in a finally, but an agent phase that fails before
            # entering it (a user-agent prompt that will not render) skips it, and a
            # multi-step trial still verifies that step.
            await pending.release(Phase.SAMPLING)
            if final_step:
                pending.snapshot = event.result
                pending.ready.set()
                await pending.until_graded()
            await pending.hold(Phase.VERIFICATION)

        async def finish(event: Any) -> None:
            await pending.release_all()

        # Environment start is admission, then provisioning, bounded by the
        # provisioning limit; the sampling permit is held only while the agent runs.
        trial.add_hook(TrialEvent.ENVIRONMENT_START, environment_start)
        trial.add_hook(TrialEvent.AGENT_START, agent_start)
        trial.add_hook(TrialEvent.AGENT_END, agent_end)
        trial.add_hook(TrialEvent.VERIFICATION_START, verification_start)
        trial.add_hook(TrialEvent.END, finish)
        trial.add_hook(TrialEvent.CANCEL, finish)

    def _setup_regrade_trial(self, trial: Any) -> None:
        """Regrade trials publish their reward at the same exposure point as sampling ones."""
        pending = self._by_trial_name[trial.config.trial_name]
        integration.expose_reward(trial, pending.publisher(trial))
        integration.observe_stops(trial, pending.stop_failures.append)

    async def sample(self, attempt: AttemptContext) -> EpisodeResult:
        identity = attempt.spec.identity
        if identity.attempt_id in self._pending or self._queue is None:
            raise ValueError(f"attempt {identity.attempt_id!r} cannot start")
        queue = self._queue
        config = self._templates[attempt.spec.task.id].model_copy(deep=True)
        # Harbor also uses this name for sandbox resources, outside trials_dir.
        config.trial_name = f"lr-{uuid4().hex}"
        config.trials_dir = self._trials_dir / "trials"
        _bind_environment(config, identity.attempt_id)
        attempts.bind(
            identity.attempt_id,
            attempt.services.resources,
            self._limits.provider_deadlines,
            self._provider_settings,
        )
        pending = _Pending(attempt, config, gated=self._reward_gate is not None)
        self._pending[identity.attempt_id] = pending
        self._by_trial_name[config.trial_name] = pending
        self._report_live()

        async def execute() -> Any:
            try:
                result = await queue.submit(config)
                self._observe_result(result)
                return result
            finally:
                # Raw Harbor finalization errors can bypass END/CANCEL hooks.
                await pending.release_all()
                pending.ready.set()

        pending.execution = asyncio.create_task(execute(), name=f"harbor:{identity.rollout_id}")
        try:
            await pending.ready.wait()
            # A multi-step trial that stopped early fixed its reward here, gated or not.
            published = (
                (pending.rewarded_trial, pending.reward)
                if pending.reward is not None
                else pending.early
            )
            if published is not None:
                snapshot = published[0]
            elif pending.execution.done():
                snapshot = pending.execution.result()
            else:
                snapshot = pending.snapshot
            self._report_trial(snapshot)
            metadata = _agent_metadata(snapshot)  # the agent's own, untouched
            if pending.oom_kills is not None:
                self._report_oom_kills(pending.oom_kills)
            if published is not None:
                self._check(_rewarded(*published))
            elif pending.execution.done():
                self._check_sampled(pending, snapshot)
            return EpisodeResult(metadata=metadata)
        except BaseException:
            # A cancelled sample() returns promptly; abort() waits for the trial.
            self._cancel(identity.attempt_id)
            raise

    def _check_sampled(self, pending: _Pending, result: Any) -> None:
        """A sampling trial that ended in Harbor, before any regrade.

        A separate-verifier trial is graded exactly when Harbor carried it on to its
        verification, whatever exception it recorded on the way; one Harbor ended
        before then failed. Any other trial is checked as Harbor left it.
        """
        if pending.separate and pending.reached_verification:
            return
        self._check(result)
        if pending.separate:
            raise HarborRolloutError(
                "Harbor ended the trial before its verification",
                trial_name=getattr(result, "trial_name", None),
                trial_uri=getattr(result, "trial_uri", None),
            )

    def _check(
        self,
        result: Any,
        *,
        permanent: type[Exception] = PermanentRolloutFailure,
    ) -> None:
        """A trial that left no reward: ``permanent`` when Harbor would not retry it.

        The verification ``grade()`` runs passes ``PermanentGradingFailure``.
        """
        try:
            _check_trial(result)
        except HarborRolloutError as error:
            if error.exception_type in self._excluded:
                raise permanent(str(error)) from error
            raise

    async def grade(self, attempt: AttemptContext, rollout: Rollout) -> GradeResult:
        """Return the trial's reward once Harbor can no longer change it.

        The trial's teardown, including every sandbox stop, continues afterwards and
        is awaited by ``finish()``, outside the grading deadline.
        """
        pending = self._pending[attempt.spec.identity.attempt_id]
        if pending.execution is None or self._regrade_queue is None:
            raise RuntimeError("rollout has no Harbor execution")
        try:
            pending.allow_grade.set()
            if pending.reward is None and pending.separate:
                # The sampling trial ended with its sandbox; grade from its artifacts.
                sampled = await asyncio.shield(pending.execution)
                self._check_sampled(pending, sampled)
                if pending.source_dir is None:
                    raise RuntimeError("sampled Harbor trial has no source directory")
                pending.regrade = asyncio.create_task(
                    self._regrade(attempt, pending),
                    name=f"harbor-regrade:{attempt.spec.identity.rollout_id}",
                )
            running = pending.regrade or pending.execution
            if pending.gated:
                return await self._grade_gated(pending, running)
            if pending.reward is None:
                rewarded = asyncio.create_task(pending.rewarded.wait())
                try:
                    await asyncio.wait(
                        {rewarded, asyncio.shield(running)}, return_when=asyncio.FIRST_COMPLETED
                    )
                finally:
                    rewarded.cancel()
            if pending.reward is not None:
                self._check(
                    _rewarded(pending.rewarded_trial, pending.reward),
                    permanent=PermanentGradingFailure,
                )
                verified = NS(verifier_result=pending.reward)
            else:
                verified = running.result()
                self._check(verified, permanent=PermanentGradingFailure)
            return GradeResult(reward=reward_of(verified, key=self._reward_key))
        except asyncio.CancelledError:
            self._cancel(attempt.spec.identity.attempt_id)
            raise

    async def _grade_gated(self, pending: _Pending, running: asyncio.Task[Any]) -> GradeResult:
        """Run the gate on each reward the trial publishes, here inside grade(), so the
        driver bounds it as it bounds grade(): its deadline, its cancellation grace, and
        a gate that ignores cancellation kept as any grade() that does.

        A sampling trial's gate failure is its outcome; a regrade's is raised in the
        trial too, and Harbor may retry it, publishing again.
        """
        while (request := await _next_request(pending, running)) is not None:
            await pending.hold(Phase.VERIFICATION)  # the gate is verification work
            try:
                rejection = await self._call_gate(request)
            except Exception as error:
                pending.gate_error = error
                _answer(request, error)
                if pending.regrade is None:  # Harbor never retries a sampling trial
                    self._check(
                        _rewarded(request.trial_result, request.result),
                        permanent=PermanentGradingFailure,
                    )
                    raise RewardUngated(str(error), name=type(error).__name__) from error
                continue
            finally:
                # A regrade keeps its permit through Harbor's retries; _regrade() or the
                # handover releases it. Otherwise grade() may not return holding it.
                if pending.regrade is None:
                    await pending.release(Phase.VERIFICATION)
            _answer(request, rejection)
            pending.gate_error = None
            await pending.hand_over(request.trial_result, request.result, rejection)
            self._check(
                _rewarded(request.trial_result, request.result),
                permanent=PermanentGradingFailure,
            )
            if rejection is not None:
                raise RewardRejected(
                    f"the reward gate dropped the reward: {rejection}; "
                    f"trial={getattr(request.trial_result, 'trial_name', None)}",
                    name=rejection,
                )
            return GradeResult(
                reward=reward_of(NS(verifier_result=request.result), key=self._reward_key)
            )
        # The trial ended without a reward the gate passed.
        verified = running.result()
        self._check(verified, permanent=PermanentGradingFailure)
        if (failed := pending.gate_error) is not None:
            raise RewardUngated(str(failed), name=type(failed).__name__) from failed
        raise RewardUngated("the reward never passed the reward gate", name="ungated")

    async def _call_gate(self, request: _GateRequest) -> str | None:
        """The gate's reason to drop the reward, or None to keep it."""
        assert self._reward_gate is not None
        try:
            rejection = await self._reward_gate(request.trial, request.result)
        except Exception as error:  # also a gate that is not async, or raised before it awaited
            raise RewardGateError(f"the reward gate raised: {error!r}") from error
        if rejection is not None and not (
            isinstance(rejection, str) and _REASON.fullmatch(rejection)
        ):
            raise RewardGateResultError(
                "a reward gate returns None or a reason that names its discard metric "
                f"(lowercase letters, digits and underscores), not {rejection!r}"
            )
        return rejection

    def _report_live(self) -> None:
        """Trials this runtime holds, and those still waiting for Harbor to accept them."""
        self._metrics.set_gauge("harbor/live_trials", len(self._pending))
        self._metrics.set_gauge(
            "harbor/preparing_trials", sum(not p.accepted for p in self._pending.values())
        )

    def _report_trial(self, result: Any) -> None:
        """Count each sampling trial and the exceptions Harbor recorded in it, as recorded.

        A failed agent is graded as Harbor grades it; these say how often agents timed
        out or failed, by Harbor's own exception type, graded or not.
        """
        self._sampling_trials += 1
        self._metrics.set_gauge("harbor/sampling_trials", self._sampling_trials)
        for entry in [result, *(getattr(result, "step_results", None) or [])]:
            if entry.exception_info is not None:
                kind = entry.exception_info.exception_type
                self._trial_exceptions[kind] += 1
                self._metrics.set_gauge(
                    f"harbor/trial_exceptions/{kind}", self._trial_exceptions[kind]
                )

    def _report_oom_kills(self, count: int) -> None:
        """Report a sampling trial's sandbox OOM kills, as its kernel counted them.

        A fact for the user, never a judgement: Harbor's result stands either way.
        """
        self._metrics.observe(OOM_KILLS, count)
        if count > 0:
            self._oom_killed_trials += 1
            self._metrics.set_gauge("harbor/oom_killed_trials", self._oom_killed_trials)

    async def _regrade(self, attempt: AttemptContext, pending: _Pending) -> Any:
        if self._regrade_queue is None or pending.source_dir is None:
            raise RuntimeError("the Harbor runtime is not prepared")
        config = self._templates[attempt.spec.task.id].model_copy(deep=True)
        config.trial_name = f"{pending.config.trial_name}-grade"
        config.trials_dir = self._trials_dir / "grades"
        config.source_trial = SourceTrialConfig(
            action="regrade", type="local", path=pending.source_dir
        )
        _bind_environment(config, attempt.spec.identity.attempt_id)
        self._by_trial_name[config.trial_name] = pending
        await pending.hold(Phase.VERIFICATION)
        try:
            result = await self._regrade_queue.submit(config)
            self._observe_result(result)
            return result
        finally:
            await pending.release(Phase.VERIFICATION)

    async def finish(
        self, attempt: AttemptContext, rollout: Rollout, disposition: EpisodeDisposition
    ) -> None:
        """Teardown only: a graded trial finishes on its own; any other is cancelled."""
        if disposition is EpisodeDisposition.COMPLETED:
            await self._settle(attempt.spec.identity.attempt_id)
        else:
            await self._discard(attempt.spec.identity.attempt_id)

    async def abort(self, attempt: AttemptContext, reason: AttemptAbortReason) -> None:
        await self._discard(attempt.spec.identity.attempt_id)

    async def _settle(self, attempt_id: str) -> None:
        """Wait for a graded trial's sandbox stops and finalization, then forget it."""
        pending = self._pending.get(attempt_id)
        if pending is None:
            return
        try:
            if pending.tasks:
                await asyncio.wait(pending.tasks)
        finally:
            # A settle cut short by the cleanup deadline still owns the trial.
            await self._discard(attempt_id)

    def _cancel(self, attempt_id: str) -> None:
        pending = self._pending.get(attempt_id)
        if pending is None:
            return
        pending.discarded.set()
        for task in pending.tasks:
            if not task.done():
                task.cancel()

    async def _discard(self, attempt_id: str) -> None:
        """Cancel a retained trial and wait for Harbor to finish tearing it down.

        Harbor swallows a cancellation in some teardown paths and carries on, so
        the cancel is repeated until the trial ends. A cancellation of this wait,
        as at the cleanup deadline, does not end it: the trial is still Harbor's
        until it ends, so it stays registered, and the driver sees a hook that
        would not stop. Once it ends it is forgotten, then the cancellation raised.
        """
        pending = self._pending.get(attempt_id)
        if pending is None:
            return
        interrupted = False
        while running := [task for task in pending.tasks if not task.done()]:
            self._cancel(attempt_id)
            trial = asyncio.gather(*running, return_exceptions=True)
            interrupted = (await outlast(trial, timeout=1.0)).interrupted or interrupted
        for task in pending.tasks:
            if not task.cancelled():
                task.exception()  # observed; the attempt's outcome is decided
        self._pending.pop(attempt_id, None)
        attempts.forget(attempt_id)
        self._by_trial_name.pop(pending.config.trial_name, None)
        self._by_trial_name.pop(f"{pending.config.trial_name}-grade", None)
        self._report_live()
        waited = Waited(settled=True, interrupted=interrupted)
        if pending.stop_failures:
            # Harbor carried on past these; a sandbox may still be running.
            failure = SandboxStopFailure(
                f"Harbor could not stop {len(pending.stop_failures)} environment(s) of "
                f"trial {pending.config.trial_name}: "
                + "; ".join(f"{type(e).__name__}: {e}" for e in pending.stop_failures)
            )
            waited.propagate(failure)
            raise failure
        waited.propagate()

    def _observe_result(self, result: Any) -> None:
        """Report trial timings; separate grading is its own trial.

        Harbor's internal retry attempts are not included in these timestamps.
        """
        for phase in ("trial", "environment_setup", "agent_setup", "agent_execution", "verifier"):
            timing = result if phase == "trial" else getattr(result, phase, None)
            start = getattr(timing, "started_at", None)
            end = getattr(timing, "finished_at", None)
            if start is not None and end is not None:
                self._metrics.observe(f"harbor/{phase}_seconds", (end - start).total_seconds())

    async def close(self) -> None:
        """Cancel every retained trial and wait for it; a sandbox that would not stop fails it."""
        for attempt_id in list(self._pending):
            self._cancel(attempt_id)
        results = await asyncio.gather(
            *(self._discard(attempt_id) for attempt_id in list(self._pending)),
            return_exceptions=True,
        )
        failed = [result for result in results if isinstance(result, SandboxStopFailure)]
        if failed:
            raise SandboxStopFailure(
                f"{len(failed)} trial(s) left environments Harbor could not stop: "
                + "; ".join(str(failure) for failure in failed)
            )


def create_runtime(config: Mapping[str, JsonValue], *, context: RuntimeContext) -> HarborRuntime:
    """The Harbor runtime factory: ``[runtime].factory = "loops_cookbook.harbor:create_runtime"``.

    ``[runtime.files].job`` names the Harbor job. ``[runtime.config].reward_key``
    selects the verifier reward when it emits several; ``reward_gate`` names a
    ``RewardGate`` as ``module:attribute``. A trial whose agent timed out or failed is
    graded as Harbor grades it: its verifier's reward trains.
    """
    unknown = sorted(set(config) - {"reward_key", "reward_gate"})
    if unknown:
        raise ValueError(f"unknown Harbor runtime settings: {', '.join(unknown)}")
    reward_key = config.get("reward_key")
    if reward_key is not None and (type(reward_key) is not str or not reward_key):
        raise ValueError("reward_key must be a nonempty string")
    gate_path = config.get("reward_gate")
    reward_gate: RewardGate | None = None
    if gate_path is not None:
        if type(gate_path) is not str:
            raise ValueError("reward_gate must be 'module:attribute'")
        found = import_attribute(gate_path, "reward gate")
        if not callable(found):
            raise ValueError(f"reward_gate {gate_path!r} is not callable")
        reward_gate = found  # pyright: ignore[reportAssignmentType]
    return HarborRuntime(
        job_path=context.files.path("job"),
        reward_key=reward_key,
        reward_gate=reward_gate,
        trials_dir=context.work_dir / "trials",
        limits=context.limits,
        metrics=context.metrics,
        provider_regions=context.provider_regions,
        provider_capacity=context.provider_capacity,
        provider_settings=context.provider_settings,
        provider_images=context.provider_images,
    )


def _agent_metadata(result: Any) -> dict[str, JsonValue]:
    """What the agent recorded about its own episode, from the step that ran last.

    Harbor persists ``AgentContext.metadata`` untouched, so this is the agent's own
    vocabulary; the run passes it along without reading it.
    """
    # Reversed, so a multi-step trial reports its final step and a single-step trial
    # falls through to the trial itself.
    entries = [result, *(getattr(result, "step_results", None) or [])]
    for entry in reversed(entries):
        agent = getattr(entry, "agent_result", None)
        metadata = getattr(agent, "metadata", None)
        if isinstance(metadata, Mapping):
            return _json_entries(metadata)  # pyright: ignore[reportUnknownArgumentType]
    return {}


def _json_entries(metadata: Mapping[Any, Any]) -> dict[str, JsonValue]:
    """The agent's report as JSON, leaving out entries JSON cannot carry.

    An agent reporting a NaN latency or a timestamp still trains: those entries are
    dropped, with one warning per key.
    """
    kept: dict[str, JsonValue] = {}
    for key, value in metadata.items():
        try:
            if not isinstance(key, str):
                raise TrajectoryError(f"non-string key {key!r}")
            kept[key] = json_value(value, where=f"agent metadata {key!r}")
        except TrajectoryError as error:
            if key not in _DROPPED_KEYS:
                _DROPPED_KEYS.add(key)
                logger.warning("Dropping agent metadata %r, which is not JSON: %s", key, error)
    return kept


_DROPPED_KEYS: set[object] = set()


def _rewarded(trial_result: Any, reward: VerifierResult) -> Any:
    """The trial as it stood at its exposure point, carrying the published reward."""
    return NS(
        exception_info=trial_result.exception_info,
        verifier_result=reward,
        step_results=getattr(trial_result, "step_results", None),
        trial_name=getattr(trial_result, "trial_name", None),
        trial_uri=getattr(trial_result, "trial_uri", None),
    )


def _graded(entry: Any) -> bool:
    verifier = getattr(entry, "verifier_result", None)
    return bool(getattr(verifier, "rewards", None)) if verifier is not None else False


def _check_trial(result: Any) -> None:
    """Fail the rollout for an exception that stopped Harbor short of a reward.

    Harbor records two agent-side exceptions and carries on to the verifier anyway
    (``AgentTimeoutError`` and ``NonZeroAgentExitCodeError``), so those trials still
    hold a real trajectory and a real graded reward. Every other exception aborts
    the trial before grading, leaving nothing to train on. Whether a reward exists
    is therefore the signal, checked for every step that recorded an exception.
    """
    results = [result, *(getattr(result, "step_results", None) or [])]
    for entry in results:
        exception = entry.exception_info
        if exception is None or _graded(entry):
            continue
        raise HarborRolloutError(
            f"{exception.exception_type}: {exception.exception_message}",
            exception_type=exception.exception_type,
            trial_name=getattr(result, "trial_name", None),
            trial_uri=getattr(result, "trial_uri", None),
        )


def _binding_mode(agent_class: type) -> ModelAccessMode:
    """The model access an agent class declares; it must also bind its services."""
    bind = getattr(agent_class, "bind_loops_services", None)
    if not callable(bind):
        raise AgentBindingError(
            f"Harbor agent {agent_class.__name__} must implement "
            "bind_loops_services(services) before this job can train"
        )
    if inspect.iscoroutinefunction(bind):
        raise AgentBindingError("bind_loops_services must be a synchronous setter")
    mode = getattr(agent_class, "model_access", None)
    if not isinstance(mode, ModelAccessMode):
        raise AgentBindingError(
            f"Harbor agent {agent_class.__name__} must declare model_access as a ModelAccessMode"
        )
    return mode


def _validate_agent_binding(config: Any) -> ModelAccessMode:
    from harbor.agents.factory import AgentFactory

    return _binding_mode(AgentFactory.get_agent_class_from_config(config))


def reward_of(trial_result: object, *, key: str | None = None) -> float:
    verifier = getattr(trial_result, "verifier_result", None)
    rewards = getattr(verifier, "rewards", None) if verifier is not None else None
    if not rewards:
        raise ValueError("the verifier produced no rewards")
    if key is not None:
        if key not in rewards:
            raise ValueError(f"no reward key {key!r}; got {sorted(rewards)}")
    elif len(rewards) == 1:
        key = next(iter(rewards))
    else:
        raise ValueError(f"ambiguous reward keys {sorted(rewards)}; set reward_key")
    value = rewards[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"verifier reward {key!r} must be numeric")  # noqa: TRY004
    reward = float(value)
    if not math.isfinite(reward):
        raise ValueError(f"verifier reward {key!r} must be finite")
    return reward


def load_job_config(path: str | Path) -> JobConfig:
    """Read a Harbor JobConfig from JSON or YAML."""
    path = Path(path).expanduser().resolve()
    text = path.read_text()
    raw = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    return JobConfig.model_validate(raw)


BLAXEL_ENVIRONMENT = "loops_cookbook.harbor.blaxel:BlaxelEnvironment"
DAYTONA_ENVIRONMENT = "loops_cookbook.harbor.daytona:DaytonaEnvironment"
# The resource kind a provider builds images as, when it builds them apart from a create.
IMAGE_BUILD = "image_build"


@dataclass(frozen=True, slots=True)
class Backing:
    """A provider whose sandboxes a Harbor environment can provision through the run."""

    title: str
    environment: str  # the cookbook's subclass, as an import path
    native: str  # Harbor's own environment for the provider
    type: EnvironmentType


BACKINGS: Mapping[str, Backing] = {
    "blaxel": Backing(
        "Blaxel",
        BLAXEL_ENVIRONMENT,
        "harbor.environments.blaxel:BlaxelEnvironment",
        EnvironmentType.BLAXEL,
    ),
    "daytona": Backing(
        "Daytona",
        DAYTONA_ENVIRONMENT,
        "harbor.environments.daytona:DaytonaEnvironment",
        EnvironmentType.DAYTONA,
    ),
}


def _provider_backed(job: JobConfig) -> str | None:
    """The provider whose cookbook environment the job uses, or None for a local one.

    Harbor gives ``import_path`` precedence over ``type``. A provider's environment that
    is not the cookbook's subclass would drop ``loops_attempt_id`` and create sandboxes
    the run never owns, so it is refused, as is ``type: blaxel`` or ``type: daytona``
    with no ``import_path``. Local environments such as Docker run as they are, with no
    sandbox tracked.
    """
    environment = job.environment
    if environment.import_path:
        chosen = import_symbol(environment.import_path)
        if not isinstance(chosen, type):
            return None
        for provider, backing in BACKINGS.items():
            try:
                native = import_symbol(backing.native)
            except ImportError:
                continue  # that provider's Harbor extra is not installed
            if not issubclass(chosen, native):
                continue
            if not issubclass(chosen, import_symbol(backing.environment)):
                raise ValueError(  # noqa: TRY004 - a configuration error, as the rest
                    f"environment.import_path {environment.import_path} is a {backing.title} "
                    f"environment the cookbook does not own; use {backing.environment}"
                )
            return provider
        return None
    for backing in BACKINGS.values():
        if environment.type == backing.type:
            raise ValueError(
                f"a {backing.title}-backed job must set environment.import_path to "
                f"{backing.environment}, so every sandbox is provisioned and released by the run"
            )
    return None


def _bind_environment(config: TrialConfig, attempt_id: str) -> None:
    """Every environment the trial constructs receives the attempt's ID."""
    kwargs = dict(config.environment.kwargs or {})
    kwargs[attempts.ATTEMPT_ID] = attempt_id
    config.environment = config.environment.model_copy(update={"kwargs": kwargs})


def _check_build_bound(
    task_id: str,
    timeouts: integration.Deadlines,
    deadlines: ProviderDeadlines,
    provider: str,
    *,
    image_build: bool = True,
) -> None:
    """Harbor's build timeout bounds start(), so it must cover the provider's deadlines in
    it: resolving prerequisites (``reconcile_seconds``), a separate image build where the
    provider has one (Blaxel; Daytona builds inside its create) and the task builds one (not
    when every environment starts from a ``blaxel://`` image), the create, and the
    attach. The wait for a provisioning slot is not counted: it depends on how many
    starts overlap, which no deadline bounds. Nor are the steps of start() after the
    attach, which no provider deadline bounds, such as Daytona's workspace upload."""
    steps = ["reconcile (prerequisites)", "create", "attach"]
    needed = deadlines.reconcile_seconds + deadlines.create_seconds + deadlines.attach_seconds
    if image_build and IMAGE_BUILD in deadlines.create_seconds_by_kind:
        needed += deadlines.create_for(IMAGE_BUILD)
        steps.insert(1, "image build")
    build = timeouts.environment_build
    if build is None or build <= needed:
        shown = "unbounded" if build is None else f"{build:g}s"
        raise ValueError(
            f"task {task_id}'s effective Harbor build timeout ({shown}) must exceed the "
            f"{BACKINGS[provider].title} {', '.join(steps[:-1])}, and {steps[-1]} deadlines combined "
            f"({needed:g}s); the wait for a provisioning slot is not counted"
        )


def _require_environment_settings(job: JobConfig) -> None:
    """Refuse a Blaxel job without a TTL, or with a region of its own.

    A sandbox that outlives its TTL is reaped mid-trial, and nothing in the job can
    derive how long the longest trial runs. The region is the provider's: every
    sandbox is created in ``[providers.blaxel] region``, so the job does not state it.
    """
    kwargs = job.environment.kwargs or {}
    if "region" in kwargs:
        raise ValueError(
            "environment.kwargs.region is set; the Blaxel region is stated once, as "
            "[providers.blaxel] region, and every sandbox is created there"
        )
    if not kwargs.get("ttl"):
        raise ValueError(
            "environment.kwargs.ttl is required on Blaxel: it must outlast the "
            "longest single trial (agent, verifier and build on one sandbox, after "
            "every timeout multiplier), or sandboxes are reaped while still working"
        )


def _report_sandbox_sizes(job: JobConfig, tasks: Mapping[str, Any]) -> None:
    """Say how big each task's Blaxel sandbox is, and which cannot hold what it declares.

    Blaxel sizes a sandbox by memory alone: the job's override_memory_mb, else the
    task's memory_mb, else Harbor's default. Its disk is a tmpfs of half that memory,
    and a task's storage_mb and cpus are not applied: Harbor's Blaxel environment
    declares CPU not configurable, and Blaxel gives one core per 2048 MB of memory.
    """
    override = job.environment.override_memory_mb
    override_cpus = job.environment.override_cpus
    sizes: collections.Counter[int] = collections.Counter()
    for task_id, task in tasks.items():
        memory = override or task.environment.memory_mb or integration.blaxel_default_memory_mb()
        disk = memory // 2
        sizes[memory] += 1
        declared = task.environment.storage_mb
        if declared is not None and declared > disk:
            logger.warning(
                "Task %s declares storage_mb %d; its Blaxel sandbox has %d MB of memory "
                "and so %d MB of disk",
                task_id,
                declared,
                memory,
                disk,
            )
        cpus = override_cpus if override_cpus is not None else task.environment.cpus
        if cpus is not None:
            logger.warning(
                "Task %s declares %d cpus; Blaxel does not apply it, and its sandbox's "
                "CPU follows its %d MB of memory",
                task_id,
                cpus,
                memory,
            )
    logger.info(
        "Blaxel sandboxes (disk is half the memory): %s",
        ", ".join(f"{count} task(s) at {memory} MB" for memory, count in sorted(sizes.items())),
    )


@runtime_checkable
class SizedEnvironment(Protocol):
    """A Harbor environment that can say the size its create would request."""

    def sandbox_size(self) -> SandboxResources:
        """What the create names; ``None`` where it names nothing."""
        ...


def _check_sandbox_sizes(
    environments: Mapping[str, Any], provider: str, capacity: SandboxCapacity
) -> None:
    """Refuse a task with an environment no one sandbox of the provider can hold.

    Every environment the task declares is checked, the agent's, a separate
    verifier's, and each step's separate verifier's, each sized by the environment
    itself, as its create would size it; a dimension it leaves out is the provider's
    default.
    """
    sizes = {
        name: environment.sandbox_size()
        for name, environment in environments.items()
        if isinstance(environment, SizedEnvironment)
    }
    empty = [
        f"task {name}: {size}"
        for name, size in sizes.items()
        if any(
            value is not None and value < 1 for value in (size.cpu, size.memory_gib, size.disk_gib)
        )
    ]
    if empty:
        raise ValueError(
            f"{provider} needs at least 1 vCPU and 1 GiB of memory and disk, and Harbor "
            "rounds a task's MB down to whole GiB, so these would be created with none: "
            + "; ".join(empty)
        )
    if over := oversized(capacity, sizes):
        raise ValueError(
            f"{len(over)} task environment(s) need more than one {provider} sandbox may "
            f"have ([providers.{provider}] max_sandbox_*), and every create would be "
            "refused: " + "; ".join(over)
        )
