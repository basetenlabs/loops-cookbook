"""The lifecycle of one Loops LoRA trainer, which every recipe shares.

Provisioning, restoring state, keeping checkpoints and closing are the same whatever a
recipe trains on. What a recipe sends to ``forward_backward``, and what it reports for
each step, belong to the recipe. A recipe's trainer subclasses ``LoopsTrainer`` for the
shared part and owns the rest.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from functools import partial
from typing import Any, NoReturn

import httpx
from baseten.loops import (
    ServiceClient,
    SupportedModel,
)
from baseten.loops.http_errors import raise_for_status_with_details
from baseten.loops.models import LoadWeightsResponse
from baseten.loops.sampling_client import SamplingClient
from baseten.loops.training_client import TrainingClient
from pydantic import ValidationError

from loops_cookbook.cancellation import outlast
from loops_cookbook.run_record import warn_still_up
from loops_cookbook.settings import ComputeSettings, ModelSettings, OutputSettings
from loops_cookbook.threads import in_thread

DEACTIVATE_RETRY_DELAYS = (1.0, 2.0, 4.0, 8.0)
"""Seconds between attempts to deactivate an eval sampler or a run, because one left up bills."""
_DEACTIVATE_RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

logger = logging.getLogger(__name__)

# How long a closing trainer waits for Loops to end its run, so a slow or unreachable
# control plane cannot hold the process at exit.
DEACTIVATE_TIMEOUT_SECONDS = 30.0


def token_normalization(averaged: bool) -> dict[str, Any]:
    """Return the loss settings that choose how Loops reduces gradients until ``optim_step``.

    Summing is Loops' default. Averaging divides the whole loss by the update's active
    tokens. Every built-in loss the cookbook sends states which one it wants. So a
    trainer older than summed defaults refuses the request rather than averaging a sum.
    """
    return {"skip_token_normalization": not averaged}


def check_token_normalization(metrics: dict[str, Any] | None, averaged: bool) -> None:
    """Refuse a ``forward_backward`` result whose trainer did not confirm the reduction."""
    if (metrics or {}).get("skip_token_normalization") != float(not averaged):
        reduction = "token-averaged" if averaged else "summed"
        raise RuntimeError(
            f"Loops did not confirm {reduction} gradients (skip_token_normalization)"
        )


async def ended_under(
    run_ended: Callable[[], Awaitable[str | None]], *, timeout: float = 10.0
) -> str | None:
    """Ask why the Loops run has ended, once a run has failed.

    The answer is the trainer's ``run_ended``. It is ``None`` while the run is live, or
    when Loops cannot say in time. Whichever call noticed the failure first, such as a
    sampling request, an update or a checkpoint, the run ended under all of them, and
    Loops is the one to say so. This function never raises. The caller is already
    unwinding, and a failure here must not replace the failure it classifies.
    """
    try:
        return await asyncio.wait_for(run_ended(), timeout)
    except Exception:
        logger.warning("Could not read whether the Loops run has ended", exc_info=True)
        return None


class CredentialsError(RuntimeError):
    """Raised when Loops rejects the API key, or the workspace lacks Loops access."""


_CAPABILITIES = "GET /v1/loops/capabilities"
_CAPABILITIES_TIMEOUT = 15.0
"""Seconds the capabilities request may take. Opening the session does not count toward it."""


async def check_served_model(model: ModelSettings, service: ServiceClient | None = None) -> str:
    """Ask Loops whether it serves ``model`` at its ``max_seq_len``, and return the line to report.

    The check is read-only, and it calls ``service.get_server_capabilities_async()``.
    Without ``service``, it makes a ``ServiceClient``, which opens a Loops session. A
    session is a record that holds no compute and needs no cleanup.

    A model that Loops does not list, has not enabled, or cannot hold at
    ``max_seq_len`` is refused with ``ValueError``. A key that Loops rejects is refused
    with ``CredentialsError``. ``run`` would fail on either after provisioning began.
    When Loops is out of reach, the line reports that and the check passes, so a config
    can be checked offline. Out of reach means no key, a network error, no answer in 15
    seconds, or any other status. A ``ServiceClient`` whose own settings are wrong, such
    as a ``LOOPS_TEAM`` that does not exist, raises its ``ValueError``, because that is
    the config's error, not Loops'.
    """
    name = model.base_model
    if service is None and not os.environ.get("BASETEN_API_KEY"):
        return f"Loops model: {name} not checked (BASETEN_API_KEY is not set)"
    try:
        if service is None:
            # This opens the session with a blocking request on the loop, as
            # provisioning does. It is outside the timeout, which covers only the
            # capabilities request.
            service = ServiceClient()
        capabilities = await asyncio.wait_for(
            service.get_server_capabilities_async(), _CAPABILITIES_TIMEOUT
        )
    except httpx.HTTPStatusError as error:
        status, url = error.response.status_code, error.request.url
        if status in (401, 403):
            raise CredentialsError(
                f"Loops rejected the credentials (HTTP {status} from {url}). Set "
                "BASETEN_API_KEY to a key for the workspace that has Loops enabled, or "
                "unset it to check the config without Loops."
            ) from error
        return f"Loops model: {name} not checked (HTTP {status} from {url})"
    except httpx.HTTPError as error:
        return f"Loops model: {name} not checked ({type(error).__name__} from {error.request.url})"
    except TimeoutError:
        return (
            f"Loops model: {name} not checked "
            f"(no answer to {_CAPABILITIES} in {_CAPABILITIES_TIMEOUT:g}s)"
        )
    except (json.JSONDecodeError, ValidationError):
        # The body is not JSON, or not the capabilities the SDK reads.
        return f"Loops model: {name} not checked (unrecognised response to {_CAPABILITIES})"
    listed = {entry.model_name: entry for entry in capabilities.supported_models}
    entry = listed.get(name)
    if entry is None:
        raise ValueError(
            f"[model] base_model {name!r} is not a model Loops serves; "
            + _closest(name, sorted(listed))
        )
    if not entry.enabled:
        raise ValueError(f"[model] base_model {name!r} is not enabled on Loops{_why(entry)}")
    # There are two ceilings. ``max_seq_len`` is what Loops supports, and
    # ``max_enabled_seq_len`` is what this workspace's approved hardware reaches. An
    # older server sends only the first. A length must fit under both.
    supported = entry.max_seq_len
    ceilings = [c for c in (supported, entry.max_enabled_seq_len) if c is not None]
    limit = min(ceilings) if ceilings else None
    if limit is None:
        return f"Loops model: {name}; max_seq_len {model.max_seq_len or 'model default'}"
    wanted = model.max_seq_len
    if wanted is not None and wanted > limit:
        if supported is not None and wanted <= supported:
            raise ValueError(
                f"[model] max_seq_len {wanted} is above the {limit} this workspace is "
                f"enabled for on {name}. Loops supports up to {supported} on hardware the "
                "workspace is not approved for: lower it, or contact Baseten to approve "
                "that hardware"
            )
        raise ValueError(
            f"[model] max_seq_len {wanted} is above the {limit} Loops enables for "
            f"{name}; lower it, or unset it for the model's default"
        )
    return (
        f"Loops model: {name}; max_seq_len {model.max_seq_len or 'model default'} "
        f"(enabled up to {limit})"
    )


def _why(entry: SupportedModel) -> str:
    """Return why Loops has not enabled ``entry`` and what to do, as a suffix for the refusal."""
    details = entry.enablement_details
    if details is None:
        return ""
    parts = (details.reason_detail or details.reason, details.remediation)
    return ": " + " ".join(part for part in parts if part)


def _closest(name: str, served: list[str]) -> str:
    """Return the served names nearest ``name``, by spelling or else from the same organisation."""
    close = difflib.get_close_matches(name, served, n=3, cutoff=0.6)
    if not close:
        owner = name.split("/")[0].lower() + "/"
        close = [listed for listed in served if listed.lower().startswith(owner)]
    if close:
        return "closest: " + ", ".join(close)
    return "it serves " + ", ".join(served)


async def deactivate_run(run_id: str) -> None:
    """End Loops run ``run_id`` and its paired sampler. Its checkpoints stay.

    This calls ``ServiceClient.deactivate_run_async``, which Loops answers the same way
    for a run that has already ended. The client first opens a Loops session, off the
    event loop. The session is a record that holds no compute and needs no cleanup. A
    key that Loops rejects raises ``CredentialsError``. Any other failure raises
    ``httpx.HTTPStatusError`` with what Loops said.
    """
    if not os.environ.get("BASETEN_API_KEY"):
        raise CredentialsError("BASETEN_API_KEY is not set. Put it in .env or the environment")
    await _deactivate_with(await asyncio.to_thread(ServiceClient), run_id)


async def _deactivate_with(service: ServiceClient, run_id: str) -> None:
    """Call ``service.deactivate_run_async``, raising a rejected key as ``CredentialsError``.

    The ``CredentialsError`` keeps what Loops said.
    """
    try:
        await service.deactivate_run_async(run_id)
    except httpx.HTTPStatusError as error:
        if error.response.status_code in (401, 403):
            raise CredentialsError(
                f"Loops rejected the credentials: {error}. Set BASETEN_API_KEY to a key for "
                "the workspace that owns the run."
            ) from error
        raise


def stepped_version(result: Any) -> int:
    """Return the policy version that an ``optim_step`` reports with its result.

    That version is the trainer's step count when the step ran.
    """
    step = (getattr(result, "metrics", None) or {}).get("step")
    if not isinstance(step, (int, float)) or isinstance(step, bool) or step < 0:
        raise RuntimeError(f"optim_step reported no step count (metrics: {result!r})")
    return int(step)


def restored_version(result: LoadWeightsResponse) -> int:
    """Return the policy version that a ``load_state`` left the trainer at, which is the step it restored.

    The SDK defaults a missing ``step`` to 0, which is also a real version. So a
    response that leaves the step out is refused rather than read as a fresh trainer.
    """
    if "step" not in result.model_fields_set:
        raise RuntimeError(f"load_state reported no restored step (result: {result!r})")
    return result.step


async def read_policy_version(client: TrainingClient) -> int:
    """Read the trainer's policy version off the event loop.

    The SDK's read is a blocking GET with no async form, and a timeout around it
    abandons the read rather than stopping it. The read runs in a daemon thread, not in
    the default executor. ``asyncio.run`` waits on the executor's threads, so an
    abandoned read there would hold up the process's exit for minutes. A serverless
    trainer serves no such read, and the SDK raises ``NotImplementedError``.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[int] = loop.create_future()

    def settle(outcome: Callable[[], None]) -> None:
        try:
            loop.call_soon_threadsafe(lambda: future.done() or outcome())
        except RuntimeError:
            pass  # The loop closed while the read was stalled; nobody is waiting.

    def read() -> None:
        try:
            version = client.policy_version
        except Exception as error:  # noqa: BLE001 - raised on the loop, to the awaiter
            settle(partial(future.set_exception, error))
        else:
            settle(partial(future.set_result, version))

    threading.Thread(target=read, name="policy-version-read", daemon=True).start()
    return await future


async def updated_version(client: TrainingClient, result: Any) -> int:
    """Return the policy version that an ``optim_step`` left the trainer at.

    A dedicated trainer serves the read, so the version is read from the trainer. A pod
    that restarted after the step is back at an earlier version, which the caller then
    refuses. A serverless trainer serves no read, so its version is the step count that
    the result reports.
    """
    try:
        return await read_policy_version(client)
    except NotImplementedError:
        return stepped_version(result)


class Capacity(StrEnum):
    """Where a Loops run trains and samples, as ``[compute] scope`` sets it.

    The run record must agree with it. A dedicated run has its own trainer deployment
    and paired sampler, and the sampler hot-swaps to each publication. A serverless run
    shares Loops capacity, and each publication is a separate adapter that a sampling
    client names.
    """

    DEDICATED = "dedicated"
    SERVERLESS = "serverless"


class LoopsTrainer:
    """A LoRA training client and its lifecycle.

    The trainer is provisioned, then optionally restored or started from another run's
    weights. It is then checkpointed and finally closed.
    """

    def __init__(
        self,
        *,
        model: ModelSettings,
        compute: ComputeSettings,
        output: OutputSettings,
        resume_from: str | None,
        init_from: str | None = None,
        init_from_optimizer: bool = False,
    ) -> None:
        self._model = model
        self._compute = compute
        self._output = output
        self._resume_from = resume_from
        self._init_from = init_from
        self._init_from_optimizer = init_from_optimizer
        self.client: TrainingClient | None = None
        self._service: ServiceClient | None = None
        # The SDK starts a sampler on a thread that it holds until the sampler is ready,
        # for up to the start timeout. So eval samplers start on threads of their own,
        # never on the default pool that training's calls run on.
        self._eval_starts = ThreadPoolExecutor(thread_name_prefix="eval-sampler-start")
        self.capacity = Capacity(compute.scope)
        """The run's capacity, as configured. Provisioning refuses a run that disagrees."""
        self._last_update: tuple[int, int] | None = None
        """The step of the last update this trainer applied, and the policy version it left."""
        self._started_at = 0
        """The policy version the trainer started this run at, before any update."""
        self._sent = False
        """Whether an optimizer step was sent and its version not yet recorded."""
        self._unaccounted = False
        """Whether the trainer restarted, or took an update whose version was never recorded.

        Once this is set, nothing the trainer holds is accounted for.
        """

    @property
    def run_id(self) -> str | None:
        """The Loops run's ID, once the trainer has started."""
        return self.client.run_id if self.client is not None else None

    @property
    def serverless(self) -> bool:
        """Whether the run trains on Loops' shared serverless capacity, as ``[compute] scope`` sets."""
        return self.capacity is Capacity.SERVERLESS

    def connect(self) -> ServiceClient:
        """Return the Loops client that starts this run's trainer and samplers, without starting any."""
        if self._service is not None:
            return self._service
        try:
            self._service = ServiceClient(
                reuse_from_run_id=self._compute.reuse_from_run_id,
                availability_model="spot" if self._compute.spot_job else "standard",
            )
        except httpx.HTTPStatusError as error:
            if error.response.status_code in (401, 403):
                raise CredentialsError(
                    "Loops rejected the credentials (HTTP "
                    f"{error.response.status_code}). Set BASETEN_API_KEY to a key for the "
                    "workspace that has Loops enabled, then re-run."
                ) from error
            raise
        return self._service

    async def servable_weights(self, state_path: str, name: str) -> tuple[str, int]:
        """Return a sampler path for the weights in trainer state ``state_path``, and their policy version.

        A trainer of its own loads the weights and saves them for a sampler as ``name``.
        That trainer's Loops run is left up, and ``deactivate_run`` ends it.
        """
        client = await self._provision(with_sampler=False)
        loaded = await client.load_state_async(state_path)
        version = restored_version(await loaded.result_async())
        saved = await client.save_weights_for_sampler_async(
            name=name, ttl_seconds=self._output.checkpoint_ttl
        )
        path = getattr(await saved.result_async(), "path", None)
        if not isinstance(path, str) or not path:
            raise RuntimeError(f"the trainer returned no sampler path for {state_path}")
        logger.info(
            "Saved the weights at %s for a sampler as %s (policy v%d)", state_path, path, version
        )
        return path, version

    async def _provision(
        self,
        *,
        with_sampler: bool,
        sampler_replicas: int | None = None,
        seed: int | None = None,
    ) -> TrainingClient:
        """Create the training client. A cancelled start still owns what it created."""
        service = self.connect()
        logger.info("%s", await check_served_model(self._model, service))

        async def provision() -> None:
            self.client = await in_thread(
                service.create_lora_training_client,
                base_model=self._model.base_model,
                rank=self._model.lora_rank,
                seed=seed,
                max_seq_len=self._model.max_seq_len,
                replicas=self._compute.trainer_replicas,
                with_sampler=with_sampler,
                sampler_num_replicas=sampler_replicas,
                name=self._output.run_name,
                scope=self.capacity.value,
            )

        provisioning = asyncio.create_task(provision(), name="loops-provisioning")
        try:
            await asyncio.shield(provisioning)
        except asyncio.CancelledError:
            logger.warning("Waiting for provisioning ownership before cancelling startup")
            # The wait is unbounded, because cancelling the await cannot stop the SDK's
            # worker thread.
            await outlast(provisioning)
            if (error := provisioning.exception()) is not None:
                logger.error(
                    "Provisioning failed while startup was being cancelled", exc_info=error
                )
            if self.client is not None:
                logger.warning(
                    "Cancelled startup retains ownership of Loops run %s", self.client.run_id
                )
            raise
        assert self.client is not None  # The completed provisioning task set it.
        run = await self._read_loops(f"runs/{self.client.run_id}", "run")
        # The run record says which capacity the run is on, and the SDK reads it too. A
        # serverless run has no trainer deployment. Loops fails a create that it cannot
        # place as asked, so a disagreement is Loops' error. Left alone, the run would
        # behave as the wrong capacity.
        reported = Capacity.SERVERLESS if run["deployment_id"] is None else Capacity.DEDICATED
        if reported is not self.capacity:
            await self._refuse(service, reported, run["deployment_id"])
        logger.info(
            "Loops run %s ready (rank=%d, max_seq_len=%s)",
            self.client.run_id,
            self._model.lora_rank,
            self._model.max_seq_len or "model default",
        )
        return self.client

    async def _refuse(
        self, service: ServiceClient, reported: Capacity, deployment_id: str | None
    ) -> NoReturn:
        """End a run that Loops placed on other capacity than asked for, and raise.

        The run is ended whatever ``keep_run`` says. Nothing it holds is this run's to
        keep, and its ID is not yet in the run record. If it is left up, the warning
        names the capacity Loops reports, because that is what it bills as.
        """
        client = self.client
        assert client is not None and client.run_id is not None
        run_id = client.run_id
        self.client = None  # The run ends here, so close() must not end or report it again.
        try:
            async with asyncio.timeout(DEACTIVATE_TIMEOUT_SECONDS):
                await _deactivate_with(service, run_id)
        except BaseException as error:  # A cancel also leaves the run up.
            logger.warning("Could not deactivate Loops run %s: %r", run_id, error)
            warn_still_up(run_id, serverless=reported is Capacity.SERVERLESS)
            if not isinstance(error, Exception):
                raise
        else:
            logger.info("Deactivated Loops run %s", run_id)
        finally:
            try:
                await client.aclose()
            except Exception:  # The refusal below is the outcome, not this failure.
                logger.warning("Could not close the client of Loops run %s", run_id, exc_info=True)
        raise RuntimeError(
            f"Loops run {run_id} was asked for {self.capacity} capacity but reports "
            f"{reported} (deployment_id {deployment_id!r})"
        )

    async def _restore(self) -> int:
        """Reload weights, optimizer state and step counter from a checkpoint, or start from another run's weights.

        A resume uses ``load_state_with_optimizer`` rather than ``load_state``, because
        ``load_state`` zeroes the Adam moments and the Adam step. For a mid-run resume,
        that would throw away the moments that the last N steps built. A fresh run
        started from other weights usually wants exactly that reset, unless
        ``init_from_optimizer`` asks for the source's optimizer too. A resume takes
        precedence, because its state already holds the weights the run started from.
        Loading them again would undo every update since.
        """
        if self.client is None:
            return 0
        if not self._resume_from:
            if self._init_from:
                self._started_at = await self._initialize(self.client, self._init_from)
            return 0
        future = await self.client.load_state_with_optimizer_async(self._resume_from)
        version = self._started_at = restored_version(await future.result_async())
        logger.info(
            "Resumed trainer state from %s at policy version %d. "
            "The runner restores its own position separately.",
            self._resume_from,
            version,
        )
        return version

    async def _initialize(self, client: TrainingClient, path: str) -> int:
        """Load the weights at ``path``, and return the policy version the trainer is left at.

        The weights load with fresh Adam moments, or with the source run's optimizer
        state when ``init_from_optimizer`` asks for it. Either way, Loops restores the
        trainer's step counter with the weights. So the policy version continues from
        the checkpoint's, which is the source run's version when it saved them, while
        this run's own steps count from 0. With the optimizer, the Adam moments and step
        are the source's. From a checkpoint saved without moments, Loops loads only the
        weights.

        Loops refuses weights of another base model or LoRA rank. The log names both, so
        the reader can compare that refusal with what was asked for.
        """
        logger.info(
            "Starting from the weights at %s (into %s at LoRA rank %d) with %s. The "
            "policy version continues from the checkpoint's",
            path,
            self._model.base_model,
            self._model.lora_rank,
            "its optimizer state" if self._init_from_optimizer else "fresh Adam moments",
        )
        if self._init_from_optimizer:
            future = await client.load_state_with_optimizer_async(path)
        else:
            future = await client.load_state_async(path)
        version = restored_version(await future.result_async())
        logger.info(
            "Loaded the weights%s at %s",
            " and optimizer state" if self._init_from_optimizer else "",
            path,
        )
        return version

    async def _checkpoint(
        self,
        step: int,
        *,
        final: bool = False,
        version: int | None = None,
        keep_sampler_weights: bool = False,
    ) -> tuple[str | None, str | None, str | None]:
        """Keep a durable, step-named copy of the weights and the trainer state.

        There are two artifacts, because they answer different questions. You serve the
        sampler checkpoint to evaluate step N after the fact. You reload the
        ``save_state`` checkpoint to resume training from step N.

        The weights are staged rather than published. ``save_weights_for_sampler`` does
        not make a paired sampler hot-swap, so keeping a checkpoint never changes what
        a recipe that samples is sampling from.
        """

        # The cadences are independent, because the artifacts answer different
        # questions and cost differently. Resumable state is cheap insurance that you
        # may want every step. Servable weights are for evaluating a step after the
        # fact. The final update is always kept for both, even when a cadence is 0. A
        # cadence of 0 asks not to checkpoint along the way, and a run with no artifact
        # at the end can be neither resumed nor served.
        def due(cadence: int) -> bool:
            return final or bool(cadence) and not (step + 1) % cadence

        weights_due = keep_sampler_weights or due(self._output.sampler_checkpoint_every)
        state_due = due(self._output.trainer_checkpoint_every)
        if not weights_due and not state_due:
            return None, None, None
        client = self._require_client()
        name = f"step-{step}"
        ttl = self._output.checkpoint_ttl
        weights_path = None
        if weights_due:
            weights = await client.save_weights_for_sampler_async(name=name, ttl_seconds=ttl)
            # Use the path to serve as the trainer published it. The trainer may rename
            # the checkpoint, so the path is never rebuilt from the name asked for.
            weights_path = getattr(await weights.result_async(), "path", None) or None
            if not isinstance(weights_path, str):
                # Only an eval reads the path, and it cannot serve weights it cannot
                # name. So that eval is skipped, and training goes on.
                if keep_sampler_weights:
                    logger.warning(
                        "step %d: the trainer returned no path for the sampler weights "
                        "kept for an eval, so that eval is skipped",
                        step,
                    )
                weights_path = None
        state_path = None
        if state_due:
            state = await client.save_state_async(name=name, ttl_seconds=ttl)
            state_path = getattr(await state.result_async(), "path", None)
            if not isinstance(state_path, str) or not state_path:
                raise RuntimeError("trainer checkpoint response has no resumable state path")
        logger.info(
            "step %d: kept %s as %r%s%s",
            step,
            " and ".join(
                filter(
                    None,
                    (
                        "sampler weights" if weights_due else "",
                        "trainer state" if state_due else "",
                    ),
                )
            ),
            name,
            f" (policy v{version})" if version is not None else "",
            f"; remote trainer state: {state_path}" if state_path else "",
        )
        return (name if weights_due else None), state_path, weights_path

    async def eval_sampler(self, model_path: str | None, *, ready_timeout: float) -> SamplingClient:
        """Start a standalone sampler on this run's ``ServiceClient``, never the paired sampler.

        The sampler serves ``model_path``, or the base model when ``model_path`` is
        ``None``. It shares the run's session, its scope, and its capacity choice of
        spot or standard. The SDK creates the sampler and waits for it in one call, so
        a wait that fails or is cancelled can leave the sampler up. The SDK logs the
        sampler's ID as it provisions it. The wait holds a thread of the trainer's own
        eval pool, not the default pool.
        """
        service = self._service
        if service is None:
            raise RuntimeError("the trainer has not started")
        requested = time.monotonic()

        def create() -> SamplingClient:
            # The pool may be busy, so the timeout bounds the whole start, including the
            # wait for a thread.
            remaining = ready_timeout - (time.monotonic() - requested)
            if remaining <= 0:
                raise TimeoutError(
                    f"no thread was free to start the eval sampler within {ready_timeout:g}s"
                )
            return service.create_sampling_client(
                base_model=self._model.base_model if model_path is None else None,
                model_path=model_path,
                max_seq_len=self._model.max_seq_len,
                ready_timeout=remaining,
                scope=self.capacity.value,
            )

        starting = self._eval_starts.submit(create)
        started = asyncio.wrap_future(starting)
        try:
            await asyncio.wait({started}, timeout=ready_timeout)
        except asyncio.CancelledError:
            starting.cancel()  # A start that is still queued never reaches Loops.
            raise
        # A start still queued at the deadline is dropped. A running start ends by its
        # own timeout.
        if not started.done() and starting.cancel():
            raise TimeoutError(
                f"no thread was free to start the eval sampler within {ready_timeout:g}s"
            )
        return await started

    async def deactivate_sampler(self, sampler: SamplingClient) -> None:
        """Stop an eval sampler. A failure is logged with the sampler's ID, never raised.

        This reads the sampler's private ``_sampler_id`` and posts the deactivation
        itself, with retries, as ``run_ended`` reads the SDK's private state.
        """
        sampler_id = getattr(sampler, "_sampler_id", None)
        if sampler_id is None and getattr(self._service, "_local_sampler_url", None) is not None:
            return  # A local sampler has nothing on Loops to deactivate.
        await self._deactivate("samplers", sampler_id, "Eval sampler")

    async def deactivate_run(self) -> None:
        """Stop the Loops run that ``servable_weights`` leaves up.

        A failure is logged with the run's ID, never raised.
        """
        if self.client is not None:
            await self._deactivate("runs", self.client.run_id, "Loops run")

    async def _deactivate(self, kind: str, resource_id: str | None, what: str) -> None:
        """POST ``/v1/loops/<kind>/<id>/deactivate``, retrying failures that may pass.

        A final failure is logged, never raised. The log line is loud, because a
        resource left up bills.
        """
        service = self._service
        try:
            if service is None or resource_id is None:
                raise RuntimeError(f"the {what.lower()}'s ID or the run's client is unknown")
            url = f"{service._base_url}/v1/loops/{kind}/{resource_id}/deactivate"
            async with httpx.AsyncClient(timeout=60.0, headers=service._headers()) as http:
                for delay in (*DEACTIVATE_RETRY_DELAYS, None):
                    try:
                        raise_for_status_with_details(await http.post(url))
                        break
                    except httpx.HTTPError as error:
                        retryable = not isinstance(error, httpx.HTTPStatusError) or (
                            error.response.status_code in _DEACTIVATE_RETRY_STATUSES
                        )
                        if delay is None or not retryable:
                            raise
                        logger.warning(
                            "Deactivating %s %s failed (%s). Retrying in %gs.",
                            what.lower(),
                            resource_id,
                            error,
                            delay,
                        )
                        await asyncio.sleep(delay)
        except (httpx.HTTPError, RuntimeError) as error:
            logger.error(
                "%s %s was NOT deactivated (%s). It bills until you deactivate it.",
                what,
                resource_id,
                error,
            )
            return
        logger.info("Deactivated %s %s.", what.lower(), resource_id)

    async def run_ended(self) -> str | None:
        """Return why the Loops run has ended, or ``None`` while it is live.

        This raises when Loops cannot say. It reads the control plane. The run is
        ``INACTIVE`` once it is shut down or replaced, and its trainer deployment ends
        as ``PREEMPTED``, ``FAILED`` or ``STOPPED``. The sampler's status is not read,
        because a sampler reports ``INACTIVE`` before it first deploys, and a cold
        sampler is not a dead run. Both reads are async, so cancelling this call stops
        them. The terminal states, base URL and headers are the SDK's own private ones.
        """
        from baseten.loops.service_client import _TRAINER_TERMINAL_FAILURE_STATES

        service, client = self._service, self.client
        if service is None or client is None or client.run_id is None:
            return None
        run = await self._read_loops(f"runs/{client.run_id}", "run")
        if run["status"]["name"] == "INACTIVE":
            return f"Loops reports run {client.run_id} INACTIVE"
        # The run may be live while its trainer is gone, because Loops marks the run
        # only later.
        deployment_id = run["deployment_id"]
        if deployment_id is None:
            # A serverless run trains on a shared trainer and has no deployment of its
            # own. Its run status is all that Loops reports.
            return None
        deployment = await self._read_loops(f"deployments/{deployment_id}", "deployment")
        trainer = deployment["status"]["name"]
        if trainer not in _TRAINER_TERMINAL_FAILURE_STATES:
            return None
        return f"Loops reports run {client.run_id}'s trainer deployment {deployment_id} {trainer}"

    async def _read_loops(self, path: str, key: str) -> dict[str, Any]:
        """Read one Loops control-plane record.

        The base URL and headers are the SDK's own private ones.
        """
        service = self._service
        assert service is not None  # It is set before provisioning.
        async with httpx.AsyncClient(timeout=30.0, headers=service._headers()) as http:
            response = await http.get(f"{service._base_url}/v1/loops/{path}")
        raise_for_status_with_details(response)
        return response.json()[key]

    def _applying(self) -> None:
        """Record that an optimizer step is being sent.

        A subclass calls this just before sending the step. Until ``_applied`` records
        the version the step left, the trainer may hold an update that this run cannot
        account for. Nothing is saved on the way out in that state. If another step is
        sent first, nothing the trainer holds is accounted for again.
        """
        if self._sent:
            self._unaccounted = True
            self._last_update = None
        self._sent = True

    def _applied(self, step: int, version: int) -> None:
        """Record that step ``step``'s optimizer update left the trainer at ``version``.

        A subclass calls this once the update is applied. The version is recorded rather
        than derived from the step. A run started from another run's weights, and every
        resume of it, is offset from its own step count by the version the trainer
        started at.

        This refuses a version that did not advance past the last update's version, or
        past the start's. Such a trainer restarted and no longer holds the training
        before this update, whatever this update applied to it. From then on it holds no
        update of this run, so none is recorded and none can be saved on the way out. A
        restarted trainer can come back to the very version an earlier step left. The
        same holds once an update's version went unrecorded, as ``_applying`` explains.
        """
        if self._unaccounted:
            raise RuntimeError(
                f"step {step}: the trainer took an update this run never recorded, so "
                "nothing it holds is accounted for"
            )
        previous = self._last_update[1] if self._last_update else self._started_at
        if version <= previous:
            self._unaccounted = True
            self._last_update = None
            raise RuntimeError(
                f"step {step}: policy version did not advance ({previous} -> {version}). "
                "The trainer no longer holds this run's earlier updates, which suggests "
                "it restarted"
            )
        self._last_update = (step, version)
        self._sent = False

    def _require_client(self) -> TrainingClient:
        if self.client is None:
            raise RuntimeError("the trainer has not been started")
        return self.client

    async def checkpoint_state_only(self, step: int, *, timeout: float = 30.0) -> str | None:
        """Save just enough to resume a step that is already committed. This never raises.

        A run calls this as it goes down. The sampler weights are deliberately skipped.
        Resuming needs the trainer state, and an artifact that only a later evaluation
        would read is not worth a dying run's last seconds. The call is bounded, because
        whatever is ending the run may also make this call hang.

        The state is saved only while the trainer still holds that step. Step ``step``
        must be the last update this trainer applied, and the trainer's policy version,
        its optimizer step count, must still be the one that update left. A restarted
        trainer fails that test. For example, an out-of-memory-killed pod comes back on
        base weights, a fresh optimizer and version 0. Without the test, its untrained
        state would be kept under a trained step's name, and a resume from it would
        silently drop that training.
        """
        client = self.client
        if client is None:
            return None
        if self._sent:
            logger.warning(
                "Not saving trainer state for step %d on the way out. An optimizer step "
                "was sent after it and its version was never read, so the trainer may "
                "hold an update this run did not record. This run has no resumable "
                "checkpoint past its last scheduled one",
                step,
            )
            return None
        if self._last_update is None or self._last_update[0] != step:
            logger.warning(
                "Not saving trainer state for step %d on the way out, because it is not "
                "the last update this trainer applied (%s). This run has no resumable "
                "checkpoint past its last scheduled one",
                step,
                f"step {self._last_update[0]}" if self._last_update else "none",
            )
            return None
        expected = self._last_update[1]
        try:
            async with asyncio.timeout(timeout):
                try:
                    version = await read_policy_version(client)
                except NotImplementedError:
                    # A serverless run's state is the broker's lineage, which outlives
                    # any one worker. So there is no restarted trainer to catch.
                    version = expected
                if version != expected:
                    logger.warning(
                        "Not saving trainer state for step %d on the way out. The trainer "
                        "is at policy version %d, not the %d that step left it at, so it "
                        "no longer holds that step and has probably restarted. This run "
                        "has no resumable checkpoint past its last scheduled one",
                        step,
                        version,
                        expected,
                    )
                    return None
                state = await client.save_state_async(
                    name=f"step-{step}", ttl_seconds=self._output.checkpoint_ttl
                )
                path = getattr(await state.result_async(), "path", None)
        except (Exception, asyncio.CancelledError):
            logger.warning(
                "Could not save trainer state for step %d on the way out. "
                "This run has no resumable checkpoint past its last scheduled one",
                step,
                exc_info=True,
            )
            return None
        return path if isinstance(path, str) and path else None

    async def close(self) -> None:
        """Close the client, then end the Loops run unless ``[compute].keep_run`` keeps it.

        A finished dedicated run holds its pods until it is deactivated, and a serverless
        run holds its workspace's slot. A failed deactivation never raises, because the
        run's outcome is already settled. A run left up is logged with how to end it.
        """
        run_id = self.run_id
        # A start that is still waiting holds its thread until the SDK gives up, and
        # nothing joins that thread.
        self._eval_starts.shutdown(wait=False, cancel_futures=True)
        try:
            if self.client is not None:
                await self.client.aclose()
        finally:
            if run_id is not None and self._service is not None:
                await self._end_run(self._service, run_id)

    async def _end_run(self, service: ServiceClient, run_id: str) -> None:
        if self._compute.keep_run:
            warn_still_up(run_id, serverless=self.serverless)
            return
        try:
            async with asyncio.timeout(DEACTIVATE_TIMEOUT_SECONDS):
                await _deactivate_with(service, run_id)
        except Exception as error:  # noqa: BLE001 - the run's outcome is already settled
            logger.warning("Could not deactivate Loops run %s: %s", run_id, error)
            warn_still_up(run_id, serverless=self.serverless)
        else:
            logger.info("Deactivated Loops run %s. Its checkpoints stay", run_id)
