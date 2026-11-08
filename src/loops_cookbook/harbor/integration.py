"""The place where the Harbor rollout runtime reaches Harbor's private members.

Every such access the Harbor rollout runtime makes lives here. The cookbook's Blaxel
and Daytona environments, ``harbor.blaxel`` and ``harbor.daytona``, also reach the
members of the Harbor environments they subclass, and each says so. So a Harbor
upgrade has those three files to review. The package metadata bounds Harbor to the
minor version these accesses were checked against, and the contract tests pin their
behaviour.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import harbor.trial.trial as trial_module
from harbor.environments.factory import EnvironmentFactory
from harbor.job import Job
from harbor.models.job.config import JobConfig
from harbor.models.job.lock import (
    _build_lock_trial_task,  # pyright: ignore[reportPrivateUsage]
)
from harbor.models.task.verifier_mode import resolve_effective_verifier_env_config
from harbor.models.trial.config import TrialConfig
from harbor.models.verifier.result import VerifierResult
from harbor.trial.multi_step import MultiStepTrial
from harbor.trial.queue import TrialQueue
from harbor.trial.trial import Trial

Publish = Callable[[VerifierResult], Awaitable[None]]


async def expand_trials(config: JobConfig) -> list[TrialConfig]:
    """Run Harbor's own dataset resolution and task and agent expansion, without running a job."""
    job = await Job.create(config)
    try:
        return [trial.model_copy(deep=True) for trial in job._trial_configs]  # pyright: ignore[reportPrivateUsage]
    finally:
        job._close_logger_handlers()  # pyright: ignore[reportPrivateUsage]


async def task_environment_dir(config: TrialConfig) -> Path:
    """Return the directory that Harbor builds the task's environment image from."""
    task, _download = await Trial._load_task(config)  # pyright: ignore[reportPrivateUsage]
    return task.paths.environment_dir


async def task_digest(config: TrialConfig) -> str:
    """Return the task content digest that Harbor records in ``TrialLock.task.digest``."""
    _task, download = await Trial._load_task(config)  # pyright: ignore[reportPrivateUsage]
    return _build_lock_trial_task(config.task, download).digest


TrialSetup = Callable[[Any], None]


class HookedTrialQueue(TrialQueue):
    """Harbor's trial queue, calling ``setup`` for each trial it is about to run.

    ``TrialQueue`` has no public hook for each trial. ``_setup_hooks`` is where it sees
    a trial after ``Trial.create`` and before ``Trial.run``. ``setup`` runs after the
    queue's own hooks are registered, so it can bind the trial and add hooks of its own.
    """

    def __init__(self, setup: TrialSetup, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._setup = setup

    def _setup_hooks(self, trial: Any) -> None:
        try:
            super()._setup_hooks(trial)
            self._setup(trial)
        except BaseException:
            # Trial.create opened the trial's logger. Trial.run closes it, but Trial.run
            # never starts when setup fails.
            trial._close_logger_handler()
            raise


def trial_task_digest(trial: Any) -> str:
    """Return the digest of the task this trial actually loaded, as its lock records it."""
    return trial._trial_lock.task.digest


@dataclass(frozen=True, slots=True)
class Deadlines:
    """The Harbor-configured deadlines that bound one grading call, in seconds."""

    task: Any  # Harbor's TaskConfig.
    environment_build: float | None
    verifier: float | None
    final_step_verifier: float | None
    # When asked for, this holds each environment the trial declares, constructed and
    # never started. The agent's is under "". Each separate verifier's is under
    # "verifier" or "step <name> verifier".
    environments: Mapping[str, Any] = field(default_factory=dict[str, Any])


async def deadlines(config: TrialConfig, *, environments: bool = False) -> Deadlines:
    """Return Harbor's effective deadlines for a trial, as Harbor itself resolves them.

    With ``environments``, the result also holds every environment the trial declares.
    The trial is created, read and closed without being run. Its environments are
    constructed but never started. A user-agent bridge has no bearing on deadlines, so
    none is created.
    """
    trial = await Trial.create(config.model_copy(update={"user_agent": None}))
    try:
        steps = trial.task.config.steps or []
        final = trial._step_verifier_timeout_sec(steps[-1]) if steps else None  # pyright: ignore[reportPrivateUsage]
        return Deadlines(
            task=trial.task.config,
            environment_build=trial._environment_build_timeout_sec,  # pyright: ignore[reportPrivateUsage]
            verifier=trial._verifier_timeout_sec,  # pyright: ignore[reportPrivateUsage]
            final_step_verifier=final,
            environments=_declared_environments(trial) if environments else {},
        )
    finally:
        trial._close_logger_handler()  # pyright: ignore[reportPrivateUsage]


def declared_docker_images(task: Any) -> dict[str, str | None]:
    """Return each environment's ``docker_image`` in the Harbor TaskConfig ``task``.

    The keys match ``Deadlines.environments``. The agent's image is under "", followed
    by each separate verifier's, resolved as Harbor resolves it.
    """
    images: dict[str, str | None] = {"": task.environment.docker_image}
    for step in task.steps or [None]:
        config = resolve_effective_verifier_env_config(task, step)
        if config is not None:
            where = "verifier" if step is None else f"step {step.name} verifier"
            images[where] = config.docker_image
    return images


def _declared_environments(trial: Any) -> dict[str, Any]:
    """Return the agent's environment and each separate verifier's environment.

    Each verifier's environment is built the way Harbor's ``_separate_verifier_env``
    builds it when it verifies, from the same job environment and the verifier's own
    config. None of them is started. A verifier's environment is built without its
    network policy, which has no bearing on its size.
    """
    environments: dict[str, Any] = {"": trial.agent_environment}
    task = trial.task.config
    for step in task.steps or [None]:
        config = resolve_effective_verifier_env_config(task, step)
        if config is None:
            continue
        where = "verifier" if step is None else f"step {step.name} verifier"
        environments[where] = EnvironmentFactory.create_environment_from_config(
            config=trial.config.environment.model_copy(update={"extra_docker_compose": []}),
            environment_dir=trial._verifier_env_build_context(step),  # pyright: ignore[reportPrivateUsage]
            environment_name=trial.task.short_name,
            session_id=trial._separate_verifier_session_id(where),  # pyright: ignore[reportPrivateUsage]
            trial_paths=trial.paths,
            task_env_config=config,
            logger=trial.logger,
        )
    return environments


def blaxel_default_memory_mb() -> int:
    """Return the memory that Harbor's Blaxel environment gives a task that sets none.

    Harbor keeps this value private.
    """
    from harbor.environments import blaxel

    return blaxel._DEFAULT_MEMORY_MB  # pyright: ignore[reportPrivateUsage]


def mark_verification_reached(trial: Any, reached: Callable[[], None]) -> None:
    """Call ``reached`` where a single-step trial would verify, and verify nothing.

    ``SingleStepTrial._run`` calls ``_run_verifier`` only after two things. The agent
    phase ended in a way that Harbor still verifies, and its logs and artifacts were
    saved. Harbor still verifies when ``_run_agent`` recorded an agent timeout or a
    non-zero exit, or when the agent returned. Any other exception propagates past
    ``_run_verifier`` and aborts the trial. So ``reached`` is Harbor's own decision that
    this trial is graded. The verifier runs later, as a regrade from the saved trial
    directory.
    """

    async def mark() -> None:
        reached()

    trial._run_verifier = mark  # pyright: ignore[reportPrivateUsage]


# --------------------------------------------------------------- reward exposure
#
# Harbor 0.23 has no public hook between a verification and the sandbox stop that
# follows it. A return from ``verify()`` is not final either, because a later network
# policy restore can still turn a reward into a failed trial. So the reward is exposed
# at the points after which Harbor can no longer change the result:
#
#   shared single step      the return of ``_run_shared_verifier``, after the restore
#   multi-step              the return of ``_select_multi_step_reward``
#   separate verifier       the ``verify()`` result, once the verifier-environment
#   (in-trial or regrade)   body exits cleanly and before that environment's stop

_captured: ContextVar[list[VerifierResult] | None] = ContextVar("captured", default=None)
_original_factory: Any = trial_module.VerifierFactory


class _CapturingVerifierFactory:
    """Harbor's verifier factory, recording each ``verify()`` result where a caller asks."""

    @classmethod
    def create_verifier_from_config(cls, *args: Any, **kwargs: Any) -> Any:
        verifier = _original_factory.create_verifier_from_config(*args, **kwargs)
        verify = verifier.verify

        async def recorded() -> VerifierResult:
            result = await verify()
            if (sink := _captured.get()) is not None:
                sink.append(result)
            return result

        verifier.verify = recorded
        return verifier


def _install_capture() -> None:
    if trial_module.VerifierFactory is not _CapturingVerifierFactory:
        trial_module.VerifierFactory = _CapturingVerifierFactory  # pyright: ignore[reportAttributeAccessIssue]


def observe_stops(trial: Any, failed: Callable[[BaseException], None]) -> None:
    """Call ``failed`` for every environment stop of ``trial`` that raises.

    Harbor catches a failed stop, logs it and carries on. It does this for the agent's
    environment and for each separate verifier's alike. Without this hook, the cookbook
    would never learn that a sandbox may still be running.
    """

    def watch(environment: Any) -> None:
        stop = environment.stop

        async def stop_or_report(delete: bool) -> None:
            try:
                await stop(delete)
            except Exception as error:
                failed(error)
                raise

        environment.stop = stop_or_report

    watch(trial.agent_environment)
    separate_env = trial._separate_verifier_env  # pyright: ignore[reportPrivateUsage]

    @contextlib.asynccontextmanager
    async def watched(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async with separate_env(*args, **kwargs) as env:
            watch(env)
            yield env

    trial._separate_verifier_env = watched  # pyright: ignore[reportPrivateUsage]


def expose_reward(trial: Any, publish: Publish) -> None:
    """Call ``publish`` with the trial's authoritative verifier result, before any stop.

    Nothing is published when the verification fails, including a network-policy
    restore that fails after ``verify()`` returned.
    """
    if isinstance(trial, MultiStepTrial):
        select = trial._select_multi_step_reward  # pyright: ignore[reportPrivateUsage]
        stop = trial._stop_agent_environment  # pyright: ignore[reportPrivateUsage]
        selected: list[VerifierResult | None] = []

        def select_then_record() -> VerifierResult | None:
            result = select()
            selected.append(result)
            return result

        async def publish_then_stop() -> None:
            # The aggregate is selected just before the stop that follows it.
            try:
                result = selected.pop() if selected else None
                if result is not None:
                    await publish(result)
            finally:
                await stop()  # A publish that raised still stops the sandbox.

        trial._select_multi_step_reward = select_then_record  # pyright: ignore[reportPrivateUsage]
        trial._stop_agent_environment = publish_then_stop  # pyright: ignore[reportPrivateUsage]
        return
    shared = trial._run_shared_verifier  # pyright: ignore[reportPrivateUsage]
    separate_env = trial._separate_verifier_env  # pyright: ignore[reportPrivateUsage]

    async def shared_then_publish(**kwargs: Any) -> VerifierResult:
        result = await shared(**kwargs)
        await publish(result)
        return result

    @contextlib.asynccontextmanager
    async def env_then_publish(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        sink: list[VerifierResult] = []
        token = _captured.set(sink)
        try:
            async with separate_env(*args, **kwargs) as env:
                yield env
                if sink:
                    await publish(sink[-1])
        finally:
            _captured.reset(token)

    _install_capture()
    trial._run_shared_verifier = shared_then_publish  # pyright: ignore[reportPrivateUsage]
    trial._separate_verifier_env = env_then_publish  # pyright: ignore[reportPrivateUsage]
