"""Training runner: rollout runtime in, GRPO batches out, Loops effects applied.

``Run`` is the composition root for one training run. It builds the runtime from
its factory, validates what the runtime declares, wires each attempt's model-access
lease to the trainer's sampler, and drives the pipeline until the step limit.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import math
import os
import re
import statistics
import traceback
import urllib.request
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from hashlib import sha256
from pathlib import Path
from time import monotonic, time
from typing import Any, Literal, Self
from uuid import uuid4

from pydantic import Field, model_validator

from loops_cookbook import run_record
from loops_cookbook.cancellation import see_through, stop
from loops_cookbook.console import elapsed, first_sentence, note
from loops_cookbook.evidence import (
    AttemptIdentity,
    JsonValue,
    Logprobs,
    Rollout,
    Sample,
    TaskRef,
    TokenSpan,
)
from loops_cookbook.llm.client import Renderers, SamplingPolicy, probe_image_tokens
from loops_cookbook.llm.factory import describe, renderer_source
from loops_cookbook.llm.render_pool import RenderPool
from loops_cookbook.llm.rendering import Renderer
from loops_cookbook.llm.transcript import ParsedTurnReply, SealedTranscript, thaw
from loops_cookbook.model_access.api import AttemptModelAccess, ModelAccessMode
from loops_cookbook.model_access.lease import (
    AttemptLease,
    DirectModelAccess,
    Freshness,
    LeaseWiring,
    affinity_key,
)
from loops_cookbook.model_access.openai.access import HttpModelAccess
from loops_cookbook.model_access.openai.gateway import Gateway, GatewaySettings
from loops_cookbook.model_access.producers import ProducerSettings
from loops_cookbook.resources.api import (
    CapacityProvider,
    CleanupReport,
    ConfiguredProvider,
    ImageProvider,
    RegionalProvider,
    ResourceAdapter,
    SandboxProvider,
)
from loops_cookbook.resources.registry import ResourceRegistry, close_adapters
from loops_cookbook.resources.services import EpisodeResourceService, RunResourceService
from loops_cookbook.rl.aggregation import AggregationReport, aggregate_loss_in_slices
from loops_cookbook.rl.config import Config, OpenAISettings, TrainingSettings
from loops_cookbook.rl.evals import EvalResult, EvalSet, PinnedSampler, SetSlots, evaluate
from loops_cookbook.rl.grpo import (
    TrainingRow,
    advantages,
    build_rows_in_slices,
    normalize_batch_provenance,
)
from loops_cookbook.rl.metrics import metadata_means, rollout_metrics, trajectory_metrics
from loops_cookbook.rl.pipeline import Pipeline, UnusableGroupsError
from loops_cookbook.rl.slices import Sliced, in_slices
from loops_cookbook.rl.trainer import TrainerPort
from loops_cookbook.run_record import loops_url, record_failure, record_identity
from loops_cookbook.runtime.api import (
    GroupGrader,
    RolloutRuntime,
    RunLimits,
    RuntimeCapabilities,
    RuntimeContext,
    RuntimeFactory,
)
from loops_cookbook.runtime.capacity import RunCapacity
from loops_cookbook.runtime.context import (
    EvidenceRecord,
    MetricsSink,
    Preflight,
    RuntimeConfigurationError,
    RuntimeFileResolver,
    build_runtime,
    check_capabilities,
    check_tasks,
    group_grader,
    model_access_mode,
    rollout_clock,
)
from loops_cookbook.runtime.driver import (
    AttemptDriver,
    DriverSettings,
    LoopsRunEnded,
    RuntimeExecutionFailure,
    SystemicFailure,
)
from loops_cookbook.runtime.groups import GroupGradingSettings
from loops_cookbook.threads import in_thread
from loops_cookbook.trainer import Capacity, ended_under

logger = logging.getLogger(__name__)

# baseten-loops 0.32 creates a sampler and waits for it in one call, so one whose wait
# failed or was cancelled cannot be stopped from here.
_MAY_BE_UP = (
    "Its sampler may still be up and BILLING: find its ID in the SDK's 'Provisioned "
    "Loops sampler <id>' line and stop it with POST /v1/loops/samplers/<id>/deactivate."
)


def _later(step: int) -> str:
    """How to run an eval of ``step`` that did not run, from the weights it kept."""
    return f" Run it later with `loops eval <config> --step {'base' if step < 0 else step}`."


SAMPLER_WEIGHTS = "sampler_weights.jsonl"
"""In the run directory: each step whose sampler weights the run kept, with their path
and policy version, for ``loops eval``."""


def finished_hint(max_steps: int) -> tuple[str, str]:
    """The bound and the way to train more for a run that made all ``max_steps``
    updates (``run_record.finished_message``)."""
    return f"max_steps = {max_steps}", "Raise max_steps to train more"


class _NothingToTrain(Exception):
    """A resume with no update left: ``execute`` skips training and still cleans up."""


@dataclass(frozen=True)
class Outcome:
    status: str
    steps: int
    """Optimizer updates completed by this invocation, excluding restored steps."""
    discards: dict[str, int]
    reason: str | None = None
    next_step: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "completed"


def _append_json(path: Path, record: dict, *, after_failure: bool = False) -> None:
    """Append ``record`` as one line. ``after_failure``: an earlier append to ``path``
    failed and may have left half a row, so this one starts a line of its own, and
    only the row that failed is unreadable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        lead = b""
        if after_failure and handle.seek(0, os.SEEK_END) > 0:
            handle.seek(-1, os.SEEK_END)
            lead = b"" if handle.read(1) == b"\n" else b"\n"
        handle.write(lead + json.dumps(record, allow_nan=False).encode() + b"\n")


def redact_model_input(payload: dict) -> dict:
    """Replace inline image bytes in a serialized ``ModelInput`` with a digest.

    Run records must not embed images: they are large and potentially sensitive.
    The digest keeps image identity auditable across sample and batch records.
    """
    chunks = []
    for chunk in payload.get("chunks", []):
        if chunk.get("type") == "image" and isinstance(chunk.get("data"), str):
            raw = base64.b64decode(chunk["data"])
            chunk = {**chunk, "data": {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)}}
        chunks.append(chunk)
    return {**payload, "chunks": chunks}


BULK_SAMPLE_FIELDS = frozenset({"prompt_ids", "completion_ids", "logprobs"})


def _plain(value: object) -> object:
    """Packed tokens or logprobs as the tuple JSON encoders write as a list."""
    return tuple(value) if isinstance(value, (TokenSpan, Logprobs)) else value


def sample_record(sample: Sample, *, tokens: bool) -> dict:
    """One generation's evidence; ``tokens`` keeps the arrays that dominate the size.

    Without them the row still identifies the generation, the policy that produced it
    and how long it was -- which answers most questions about a finished run at about
    a hundredth of the bytes. Only replay needs the arrays themselves.
    """
    record = {
        field.name: _plain(getattr(sample, field.name))
        for field in fields(Sample)
        if field.name != "prompt_chunks" and (tokens or field.name not in BULK_SAMPLE_FIELDS)
    }
    if not tokens:
        record["prompt_tokens"] = len(sample.prompt_ids)
        record["completion_tokens"] = len(sample.completion_ids)
        return record
    # Text-only chunks are exactly `prompt_ids` -- a reader rebuilds them from it --
    # so writing them doubles the largest thing in the file to say it twice. Images
    # cannot be reconstructed that way, so those prompts are written as chunks.
    if not sample.prompt_chunks.text_only:
        record["prompt_input"] = redact_model_input(sample.prompt_input.model_dump(mode="json"))
    return record


def transcript_record(transcript: SealedTranscript) -> dict[str, Any]:
    """An attempt's transcript as written beside its samples.

    ``sealed`` is false for an attempt that failed instead, whose turns are as far as it
    got and none trainable. A turn with ``sample_id`` null is a request no generation
    answered, written whole. A turn's ``unrecordable`` says why its request or reply is
    missing, when it could not be recorded as JSON.

    Images are named by digest, with their format and size, and never embedded: a run
    record holds no image bytes. Tool definitions are written once, by digest. A
    non-finite number, which a policy can write in a tool call's arguments, is written
    as the string ``"NaN"``, ``"Infinity"`` or ``"-Infinity"``, so the file stays
    strict JSON that any reader can load.
    """
    turns: list[JsonValue] = []
    for sealed in transcript.turns:
        turn = sealed.turn
        reply = turn.reply
        turns.append(
            {
                "sample_id": turn.sample_id,
                "producer": turn.producer,
                "conversation_id": turn.conversation_id,
                "extends": turn.extends,
                "messages": thaw(turn.messages),
                "tools": list(turn.tools),
                "reply": (
                    None
                    if reply is None
                    else {
                        "parsed": None if reply.parsed is None else thaw(reply.parsed),
                        "parse_error": reply.parse_error,
                    }
                    if isinstance(reply, ParsedTurnReply)
                    else {"completion": thaw(reply.body)}
                ),
                "unrecordable": turn.unrecordable,
                "disposition": sealed.disposition.value,
                "trainable": sealed.trainable,
            }
        )
    return {
        "sealed": transcript.sealed,
        "turns": _strict(turns),
        "tools": _strict(
            {digest: thaw(definition) for digest, definition in transcript.tools.items()}
        ),
        "images": {
            digest: {"format": image.format, "bytes": len(image.data)}
            for digest, image in transcript.images.items()
        },
    }


def transcript_requests(record: Mapping[str, Any]) -> dict[str, list[JsonValue]]:
    """Each turn's whole request, by sample id, rebuilt from a written transcript.

    A turn stores only what its request adds to the turn it ``extends``, and that
    turn can have any disposition -- discarded, undelivered -- and, over HTTP without
    ``X-Loops-Conversation-Id``, any conversation of the same producer. So a request
    is rebuilt by following ``extends`` through every turn in the file, never only the
    trainable ones or one conversation's. A request no generation answered has no
    sample id: its turn, written whole, is the request.
    """
    turns: dict[str, Mapping[str, Any]] = {
        turn["sample_id"]: turn for turn in record["turns"] if turn["sample_id"] is not None
    }

    def whole(turn: Mapping[str, Any]) -> list[JsonValue]:
        parts = [turn["messages"]]
        while turn["extends"] is not None:
            turn = turns[turn["extends"]]
            parts.append(turn["messages"])
        return [message for messages in reversed(parts) for message in messages]

    return {sample_id: whole(turn) for sample_id, turn in turns.items()}


def _write_transcript(path: Path, identity: AttemptIdentity, transcript: SealedTranscript) -> None:
    """Whole or absent: a reader never finds half an attempt's transcript."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "attempt_id": identity.attempt_id,
        "rollout_id": identity.rollout_id,
        "group_id": identity.group_id,
        **transcript_record(transcript),
    }
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(record, allow_nan=False))
    partial.replace(path)


def _strict(value: JsonValue) -> JsonValue:
    """``value`` with each non-finite number spelled as the string JavaScript uses for it."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, list):
        return [_strict(item) for item in value]
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    return value


def datum_record(datum) -> dict:
    payload = datum.model_dump(mode="json")
    return {**payload, "model_input": redact_model_input(payload["model_input"])}


def row_record(row: TrainingRow) -> dict:
    """A training row as the batch record keeps it: the datum, and which of its target
    positions are actions, as half-open ranges."""
    return {"datum": datum_record(row.datum), "actions": [list(span) for span in row.actions]}


def _describe(error: BaseException) -> str:
    """Name first, message only when there is one.

    ``CancelledError`` and many transport errors stringify to nothing at all, so
    interpolating them alone leaves a dangling colon and says less than the class.
    """
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


@dataclass(frozen=True, slots=True)
class StepEvent:
    """One optimizer update, as an observer sees it.

    A value rather than an argument list, so a sink names what it reads and a
    later field costs nothing: the push half of observability exists to let sinks
    be added without touching the training loop, which three positional arguments
    would quietly undo the moment a fourth was needed.
    """

    step: int
    gauges: dict[str, float]
    step_scoped: frozenset[str]
    """Names in ``gauges`` an update alone produces, so only these can carry the
    step axis. Everything else is also reported between updates, by a heartbeat
    that has no step to plot against."""


StepSink = Callable[[StepEvent], None]


def _network_environment() -> dict[str, str]:
    """The proxy configuration every outbound call in this run will inherit.

    Worth a line in the log because a proxy failure is close to undiagnosable
    afterwards: httpx raises ``ProxyError`` naming neither the proxy nor the host,
    so a run that did not write down what was in its path cannot answer "what was
    proxying this?" once the shell that launched it is gone. Credentials in a proxy
    URL are redacted -- the host is what identifies it, the password never helps.
    """
    proxies = dict(urllib.request.getproxies())
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        if value := os.environ.get(name):
            proxies.setdefault(name.lower().removesuffix("_proxy"), value)
    return {
        scheme: re.sub(r"//[^/@]*@", "//<redacted>@", value)
        for scheme, value in sorted(proxies.items())
    }


def _write_eval(
    directory: Path,
    stem: str,
    rows: Sequence[Mapping[str, JsonValue]],
    transcripts: Sequence[tuple[str, SealedTranscript]],
) -> None:
    """One eval's files: ``<stem>.jsonl``, a JSON object per rollout, and
    ``<stem>.transcripts.jsonl``, each sealed rollout's conversation."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.transcripts.jsonl").write_text(
        "".join(
            json.dumps({"rollout_id": rollout_id, **transcript_record(transcript)}, allow_nan=False)
            + "\n"
            for rollout_id, transcript in transcripts
        )
    )
    (directory / f"{stem}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def _record_sample(
    path: Path, sample: Sample, identity: dict, *, tokens: bool, after_failure: bool = False
) -> None:
    """Record one generation, stamped with the wall clock it was captured at.

    The timestamp is the only way to ask a finished run *when* anything happened:
    without it the sampling rate over time can only be inferred from trace-file
    mtimes, which report each attempt's last write and so rewrite their own history
    as the run continues.
    """
    _append_json(
        path,
        {**identity, "recorded_at": time(), **sample_record(sample, tokens=tokens)},
        after_failure=after_failure,
    )


JSON_CHUNK = 8192
"""List items encoded as one unit."""


def _dump_in_slices(value: object, write: Callable[[str], object]) -> Sliced[None]:
    """``json.dumps(value, allow_nan=False)``, pausing between a long list's chunks."""
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        write("{")
        for index, (key, item) in enumerate(value.items()):
            write(f"{', ' if index else ''}{json.dumps(key)}: ")
            yield from _dump_in_slices(item, write)
        write("}")
    elif isinstance(value, (list, tuple)):
        write("[")
        for start in range(0, len(value), JSON_CHUNK):
            if start:
                write(", ")
            part = value[start : start + JSON_CHUNK]
            if any(isinstance(item, (dict, list, tuple)) for item in part):
                for index, item in enumerate(part):
                    if index:
                        write(", ")
                    yield from _dump_in_slices(item, write)
            else:
                write(json.dumps(part, allow_nan=False)[1:-1])
            yield
        write("]")
    else:
        write(json.dumps(value, allow_nan=False))


async def _record_batch(path, step, policy_version, groups, rows, aggregation) -> None:
    """Encoded in slices on the loop, since a thread would hold the GIL; written from a
    thread, since a write releases it. Whole or absent, like a transcript."""
    rollout_ids = [r.identity.rollout_id for group in groups for r in group.members]
    head = json.dumps(
        {"step": step, "policy_version": policy_version, "rollout_ids": rollout_ids},
        allow_nan=False,
    )
    tail = json.dumps(asdict(aggregation) if aggregation is not None else None, allow_nan=False)
    partial = path.with_suffix(".json.partial")
    try:
        await _in_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await _in_thread(_write, partial, f'{head[:-1]}, "rows": [', "w")
        for index, row in enumerate(rows):
            pieces = [", "] if index else []
            await in_slices(_dump_in_slices(row_record(row), pieces.append))
            await _in_thread(_write, partial, "".join(pieces), "a")
        await _in_thread(_write, partial, f'], "aggregation": {tail}}}', "a")
        await _in_thread(partial.replace, path)
    except BaseException:
        await _in_thread(partial.unlink, missing_ok=True)
        raise


def _write(path: Path, text: str, mode: str) -> None:
    with path.open(mode) as handle:
        handle.write(text)


async def _in_thread[T](function: Callable[..., T], /, *args, **kwargs) -> T:
    """``in_thread``, waited out when cancelled: the thread never outlives it."""
    return await see_through(in_thread(function, *args, **kwargs))


@dataclass(frozen=True)
class ScoredBatch:
    rows_by_group: list[list[TrainingRow]]
    rewards: list[float]
    advantages: list[float]
    versions: list[int]
    forked_rollouts: int  # rollouts whose generations form more than one chain


def score_batch(groups) -> Sliced[ScoredBatch]:
    """Derive all training rows only after rollout-atomic freshness selection, pausing
    after each row."""
    rows_by_group, rewards, advs, versions = [], [], [], []
    forked = 0
    for group in groups:
        group_rewards = [rollout.reward for rollout in group.members]
        if any(reward is None for reward in group_rewards):
            raise ValueError("cannot train an ungraded rollout")
        group_advs = advantages(group_rewards)
        rows = []
        for rollout, advantage in zip(group.members, group_advs, strict=True):
            if not rollout.trains:  # its agent discarded every generation: a baseline only
                continue
            chains = rollout.chains
            forked += len(chains) > 1
            rows.extend((yield from build_rows_in_slices(rollout, advantage, chains)))
            versions.extend(rollout.policy_versions)
        rows_by_group.append(rows)
        rewards.extend(group_rewards)
        advs.extend(group_advs)
    return ScoredBatch(rows_by_group, rewards, advs, versions, forked)


@dataclass(frozen=True)
class PreparedBatch:
    """One update's rows, ready for the trainer, and what its rollouts report.

    The rows are the batch's bulk -- Python objects per token -- and only the trainer
    and the batch record read them; everything the step reports afterwards is in the
    other fields. ``trained()`` is the same batch without its rows, to keep once they
    are submitted.
    """

    rows: list[TrainingRow]
    aggregation: AggregationReport | None
    rewards: list[float]
    advantages: list[float]
    versions: list[int]
    forked_rollouts: int
    routed_datums: int  # rows that name their samples' records, for replay
    unrecorded_datums: int  # rows with a generation the sampler did not record
    rollout_gauges: dict[str, float]

    def trained(self) -> PreparedBatch:
        """This batch's report without its rows, which the trainer has accumulated."""
        return replace(self, rows=[])


def prepare_batch(
    groups, training: TrainingSettings, *, metrics_limit: int = 32
) -> Sliced[PreparedBatch]:
    """Score, weight and summarize a selected batch: all of a step's work on its tokens.

    CPU-bound and linear in the batch's tokens -- seconds for a large batch of long
    rollouts -- so it pauses after each row it builds or weights, and the runner runs it on
    the event loop with ``in_slices``: attempts, heartbeats and logging wait at most
    a slice. Not in a worker thread: that holds the GIL just the same, and the loop
    then waits for it after every I/O wakeup, which slows hundreds of rollouts' HTTP
    traffic to a crawl. Selected groups are complete, and nothing changes them until
    the batch is committed or aborted, so the tasks that run between slices cannot
    change what is prepared.
    """
    scored = yield from score_batch(groups)
    batch = yield from aggregate_loss_in_slices(scored.rows_by_group, training)
    # Weighted rows share targets and inputs, but replace the advantage arrays. The
    # originals go now, before transport, a group at a time: freeing them is a pass
    # over every position too.
    # Under replay, a row without provenance takes the whole batch off replay.
    unrecorded = sum(not row.routed for row in batch.rows) if training.router_replay else 0
    for unweighted in scored.rows_by_group:
        unweighted.clear()
        yield
    rows = normalize_batch_provenance(batch.rows, enabled=training.router_replay)
    yield
    selected = [rollout for group in groups for rollout in group.members]
    return PreparedBatch(
        rows,
        batch.report,
        scored.rewards,
        scored.advantages,
        scored.versions,
        scored.forked_rollouts,
        sum(row.routed for row in rows),
        unrecorded,
        {**trajectory_metrics(selected), **rollout_metrics(selected, limit=metrics_limit)},
    )


def _lag_gauges(policy_version: int, rollouts) -> dict[str, float]:
    """How many updates behind the trainer each trained rollout's oldest generation was.

    ``freshness/lag_max`` reports only the worst; the spread says whether a batch is
    mostly fresh with one straggler, or old throughout.
    """
    lags = sorted(
        policy_version - rollout.minimum_policy_version
        for rollout in rollouts
        if rollout.minimum_policy_version is not None
    )
    if not lags:
        return {}
    return {
        "freshness/lag_mean": statistics.fmean(lags),
        "freshness/lag_p90": float(lags[max(0, math.ceil(0.9 * len(lags)) - 1)]),
    }


def _completion_spread_gauges(groups) -> dict[str, float]:
    """Seconds from each trained group's first member resolving to its last.

    Its first finishers wait that long for the rest, ageing towards the freshness
    limit; a wide spread means the run's rollouts are spread over many groups.
    """
    spreads = [
        group.completed_at - group.first_resolved_at
        for group in groups
        if group.completed_at is not None and group.first_resolved_at is not None
    ]
    if not spreads:
        return {}
    return {
        "groups/completion_spread_mean_seconds": statistics.fmean(spreads),
        "groups/completion_spread_max_seconds": max(spreads),
    }


def mixed_policy_rollouts(rollouts: Sequence[Rollout]) -> int:
    """Rollouts whose trainable generations came from more than one policy version."""
    return sum(len(set(rollout.policy_versions)) > 1 for rollout in rollouts)


def _update_gauges(
    report,
    groups,
    batch: PreparedBatch,
    unusable_rewards: Sequence[float] = (),
    unusable_metadata: Sequence[Mapping[str, JsonValue]] = (),
) -> dict[str, float]:
    """What only an optimizer update reports: the step, its batch, and its rows.

    What is linear in the batch's tokens was summarized with its rows, off the event
    loop, by ``prepare_batch``.

    ``unusable_rewards`` are the graded rewards of groups dropped as unusable since the
    previous update. ``reward/mean`` leaves them out, since they do not train;
    ``reward/mean_with_unusable`` counts them, so groups the policy solves or fails
    outright still show in the reward curve. ``unusable_metadata`` is the same members'
    metadata: each numeric ``rollout/<key>`` the batch reports is also reported as
    ``rollout/<key>_with_unusable``, so an all-pass rate counts the groups the policy
    solves outright too. Both are reported every update, with or without unusable
    groups, so their curves do not sample only the steps that dropped one.
    """
    selected = [rollout for group in groups for rollout in group.members]
    gauges = {
        **report.gauges(),
        **batch.rollout_gauges,
    }
    graded = [r.reward for r in selected if r.reward is not None] + list(unusable_rewards)
    if graded:
        gauges["reward/mean_with_unusable"] = sum(graded) / len(graded)
    gauges["reward/unusable_rollouts"] = float(len(unusable_rewards))
    means = metadata_means([r.metadata for r in selected] + list(unusable_metadata))
    for key, mean in means.items():
        if f"rollout/{key}" in batch.rollout_gauges:
            gauges[f"rollout/{key}_with_unusable"] = mean
    # The spread GRPO learns from is within each group; reward/std is across the
    # whole batch, so it reads high even when every group's members scored alike.
    spreads = [
        statistics.pstdev(rewards)
        for rewards in (
            [r.reward for r in group.members if r.reward is not None] for group in groups
        )
        if len(rewards) > 1
    ]
    if spreads:
        gauges["reward/group_std_mean"] = statistics.fmean(spreads)
        gauges["reward/group_std_min"] = min(spreads)
        gauges["reward/group_std_max"] = max(spreads)
    gauges.update(_lag_gauges(report.policy_version_before, selected))
    gauges.update(_completion_spread_gauges(groups))
    # What each trained group lost before training: GRPO's baseline is over its
    # survivors, so losses bias it.
    lost = [group.expected - len(group.members) for group in groups]
    gauges["groups/rollouts_lost"] = float(sum(lost))
    gauges["groups/rollouts_lost_max"] = float(max(lost, default=0))
    # Rollouts in the baseline that train nothing: their agent discarded every turn.
    gauges["trajectory/untrained_rollouts"] = float(sum(not r.trains for r in selected))
    gauges["trajectory/forked_rollouts"] = batch.forked_rollouts
    gauges["trajectory/mixed_policy_rollouts"] = mixed_policy_rollouts(selected)
    gauges["r3/routed_datums"] = batch.routed_datums
    gauges["r3/unrecorded_datums"] = batch.unrecorded_datums
    return gauges


def _digest(data: bytes) -> str:
    return f"sha256:{sha256(data).hexdigest()}"


def _content_digest(path: Path) -> str | None:
    """A digest of a file, or of every file under a directory by relative path."""
    if path.is_file():
        return _digest(path.read_bytes())
    if path.is_dir():
        entries = sorted(item for item in path.rglob("*") if item.is_file())
        listing = [
            [item.relative_to(path).as_posix(), _digest(item.read_bytes())] for item in entries
        ]
        return _digest(_canonical(listing))
    return None


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


class Manifest(run_record.Manifest):
    """An RL run's resume point: also where the task order was, and the policy version
    and training contract a resumed run is held to."""

    method: Literal["rl"] = "rl"
    next_group_index: int = Field(ge=0)
    pending_group_indices: list[int] = Field(default_factory=list)
    """Places below ``next_group_index`` whose groups had not trained when it was
    written: a resume samples them first, in this order. Empty in a manifest written
    before it had them, which resumes at ``next_group_index`` alone."""
    policy_version: int = Field(ge=0)
    shuffle_seed: int
    contract: dict[str, Any]

    @model_validator(mode="after")
    def _pending_before_next(self) -> Self:
        pending = self.pending_group_indices
        if pending != sorted(set(pending)) or not all(
            0 <= index < self.next_group_index for index in pending
        ):
            raise ValueError(
                "pending_group_indices must be distinct, ascending, and below next_group_index"
            )
        return self


class Run:
    def __init__(
        self,
        config: Config,
        trainer: TrainerPort,
        *,
        runtime_factory: RuntimeFactory,
        renderer_factory: Callable[[Config], Renderer],
        resume: Manifest | None = None,
        config_dir: Path | None = None,
        environ: Mapping[str, str] | None = None,
        adapters: Mapping[str, ResourceAdapter] | None = None,
        preflight_only: bool = False,
        run_directory: Path | None = None,
    ) -> None:
        """Everything a run uses is built by its composition root (``rl.recipe``) and
        handed in: the trainer, restoring ``resume``'s state when there is one; the
        runtime factory, which the run calls with its own context; and the renderer
        factory, which it calls once per reasoning effort.

        ``preflight_only`` is ``loops check``: it holds a resume to its contract
        alone, and never opens the run the manifest names. ``run_directory`` is the
        run ``evaluate_after`` evaluates.
        """
        if config.providers and adapters is None:
            raise ValueError(
                "[providers] is configured; build its adapters with "
                "loops_cookbook.providers.create_adapters and pass them"
            )
        self.config = config
        self.config_dir = (config_dir or Path.cwd()).resolve()
        self.out = run_directory or run_record.new_run_directory(
            config.output.output_dir, config.output.run_name
        )
        self.run_id = self.out.name
        self._adapters = dict(adapters or {})
        self._registry: ResourceRegistry | None = None
        self._run_resources: RunResourceService | None = None
        self._unclean = False
        self._affinity_namespace = uuid4().hex
        self._step_sinks: list[StepSink] = []
        self._notice: str | None = None
        self._training = False
        self._step = 0
        self._pipeline: Pipeline | None = None
        self._gateway: Gateway | None = None
        self._metrics = MetricsSink()
        self._discarded = {"generations": 0, "tokens": 0}
        self._undelivered = {"generations": 0, "tokens": 0}
        self._unwritten = {"samples": 0, "transcripts": 0}
        # Rollout ID -> attempt ID -> [undelivered, discarded by the agent], until the
        # rollout's rollouts.jsonl record takes the counts of the attempt it names.
        self._dropped: dict[str, dict[str, list[int]]] = {}
        self._last_commit = None
        # True from the optimizer step's start until its publication returns.
        self._committing = False
        self._resume_contract: dict[str, Any] = {}
        self._runtime_evidence = EvidenceRecord()
        self._resume = resume
        self._next_group_index = resume.next_group_index if resume else 0
        self._pending_group_indices = list(resume.pending_group_indices) if resume else []
        self._runtime_factory = runtime_factory
        self._renderer_factory = renderer_factory
        # Conversations render here, off the event loop, when the renderer is thread-safe.
        self._render_pool = RenderPool()
        self._environ = environ
        self.capabilities: RuntimeCapabilities | None = None
        # The rollout clock the run keeps, configured or the runtime's; set by preflight().
        self.rollout_timeout_seconds: float | None = None
        # How attempts reach the model, configured or the runtime's; set by preflight().
        self.model_access: ModelAccessMode | None = None
        self.group_grader: GroupGrader | None = None
        # [eval.sets], built by preflight(), each with its count of tasks it shares
        # with training.
        self.eval_sets: tuple[EvalSet, ...] = ()
        self.eval_shared: dict[str, int] = {}
        # One background task per eval step, by step (-1: the base model); and the
        # (set, step) evals that ended, evaluated or not, so shutdown can name the rest.
        self._evals: dict[int, asyncio.Future[Any]] = {}
        self._evals_ended: set[tuple[str, int]] = set()
        self._evals_written: set[str] = set()  # sets that wrote an eval's results
        # Each set's latest eval, and its step: a set has one eval at a time.
        self._set_evals: dict[str, tuple[int, asyncio.Task[None]]] = {}
        self._set_eval_runs: list[asyncio.Task[None]] = []  # every set eval started
        self._access: Callable[[AttemptLease], AttemptModelAccess] | None = None
        if resume is not None and not preflight_only:
            # The same run, so the same ownership labels and resource journal.
            self.out, self.run_id = Path(resume.run_directory), resume.run_id
        self._manifest_order = run_record.ManifestOrder(self.out)
        self.trainer = trainer

    def _runtime_context(
        self,
        evidence: EvidenceRecord,
        *,
        files: Mapping[str, str] | None = None,
        work_dir: Path | None = None,
        max_concurrent_rollouts: int | None = None,
        metrics: MetricsSink | None = None,
    ) -> RuntimeContext:
        """The training runtime's context; an eval set's names its own files, working
        directory, concurrency and metrics sink."""
        rollout = self.config.rollout
        return RuntimeContext(
            work_dir=work_dir or self.out / "runtime",
            project_root=Path.cwd().resolve(),
            config_dir=self.config_dir,
            files=RuntimeFileResolver(self._runtime_files(files)),
            run_resources=self._run_scope(),
            limits=RunLimits(
                max_concurrent_rollouts=max_concurrent_rollouts or rollout.max_concurrent_rollouts,
                max_concurrent_verifiers=rollout.max_concurrent_verifiers,
                rollout_timeout_seconds=rollout.rollout_timeout_seconds,
                grading_timeout_seconds=rollout.grading.timeout_seconds,
                cancellation_grace_seconds=rollout.cleanup.cancellation_grace_seconds,
                process_kill_grace_seconds=rollout.cleanup.process_kill_grace_seconds,
                prestart_places=rollout.prestart_places,
                trajectory_tokens=self.config.trajectory_tokens,
                provider_deadlines={
                    name: adapter.deadlines for name, adapter in sorted(self._adapters.items())
                },
            ),
            metrics=self._metrics if metrics is None else metrics,
            evidence=evidence,
            sandbox_providers={
                name: adapter.sandbox_kind
                for name, adapter in self._adapters.items()
                if isinstance(adapter, SandboxProvider)
            },
            provider_regions={
                name: adapter.region
                for name, adapter in self._adapters.items()
                if isinstance(adapter, RegionalProvider)
            },
            provider_capacity={
                name: adapter.capacity
                for name, adapter in self._adapters.items()
                if isinstance(adapter, CapacityProvider)
            },
            provider_settings={
                name: adapter.settings
                for name, adapter in self._adapters.items()
                if isinstance(adapter, ConfiguredProvider)
            },
            provider_images={
                name: adapter.image_tags
                for name, adapter in self._adapters.items()
                if isinstance(adapter, ImageProvider)
            },
        )

    def _runtime_files(self, overrides: Mapping[str, str] | None = None) -> dict[str, Path]:
        """``[runtime.files]`` with ``overrides`` on top, relative paths resolved against
        the config directory."""
        files = {**self.config.runtime.files, **(overrides or {})}
        return {
            alias: (Path(path) if Path(path).is_absolute() else self.config_dir / path)
            for alias, path in sorted(files.items())
        }

    def _lease_wiring(self) -> LeaseWiring:
        config = self.config
        keep_tokens = config.output.trace == "full"

        def sampler():
            if self.trainer.sampler is None:
                raise RuntimeError("trainer has no published sampler")
            return self.trainer.sampler

        cut_short: set[Path] = set()

        async def record(identity: AttemptIdentity, sample: Sample) -> None:
            if config.output.trace == "off":
                return
            attribution = {
                "attempt_id": identity.attempt_id,
                "rollout_id": identity.rollout_id,
                "group_id": identity.group_id,
            }
            path = self.out / "samples" / f"{identity.attempt_id}.jsonl"
            try:
                await in_thread(
                    _record_sample,
                    path,
                    sample,
                    attribution,
                    tokens=keep_tokens,
                    after_failure=path in cut_short,
                )
            except Exception:
                cut_short.add(path)
                # The trace is for reading the run, so a row it cannot write is
                # reported, never a reason for the generation's attempt to fail.
                self._unwritten["samples"] += 1
                logger.warning(
                    "Attempt %s: the sample row of generation %s was not written",
                    identity.attempt_id,
                    sample.sample_id,
                    exc_info=True,
                )

        def dropped(identity: AttemptIdentity, sample: Sample, *, delivered: bool) -> None:
            self._discarded["generations"] += 1
            self._discarded["tokens"] += len(sample.completion_ids)
            if not delivered:
                self._undelivered["generations"] += 1
                self._undelivered["tokens"] += len(sample.completion_ids)
            attempts = self._dropped.setdefault(identity.rollout_id, {})
            attempts.setdefault(identity.attempt_id, [0, 0])[1 if delivered else 0] += 1

        async def write_transcript(identity: AttemptIdentity, transcript: SealedTranscript) -> None:
            # A transcript runs to megabytes; encoding and writing it on the event loop
            # would stall every gateway handler and attempt. It is immutable, so a
            # thread can read it while the loop runs on. A cancelled write is over only
            # once its thread is, so a later write of the attempt never races it.
            try:
                await _in_thread(
                    _write_transcript,
                    self.out / "transcripts" / f"{identity.attempt_id}.json",
                    identity,
                    transcript,
                )
            except (Exception, asyncio.CancelledError) as error:
                # A cancelled write ran to its end: written, unless it failed, which
                # the cancellation carries as its cause.
                failure = error.__cause__ if isinstance(error, asyncio.CancelledError) else error
                if failure is not None:
                    self._unwritten["transcripts"] += 1
                    logger.warning(
                        "Attempt %s: its transcript was not written",
                        identity.attempt_id,
                        exc_info=failure,
                    )
                if isinstance(error, asyncio.CancelledError):
                    raise

        warned_images = False

        def unrecordable_prompt() -> None:
            # Each such generation fails, and its rollout is discarded as a sampling
            # failure; the cause is said once, where the config can fix it.
            nonlocal warned_images
            if not warned_images:
                warned_images = True
                logger.warning(
                    "A prompt has images, but [runtime] has_images = false and router "
                    "replay records every generation, which Loops cannot do for an image "
                    "prompt. Its rollout fails; set has_images = true and "
                    "router_replay = false."
                )

        # Recording copies every request, so it is on only for a reader: the run's
        # trace, or a group grader, which reads each member's transcript.
        keeps_transcripts = config.output.trace in {"transcripts", "full"}
        grouped = self.capabilities is not None and self.capabilities.pending_group_grade
        records_transcripts = keeps_transcripts or grouped
        rollout = config.rollout
        return LeaseWiring(
            sampler=sampler,
            capacity=self.trainer.capacity,
            sampler_version=lambda: self.trainer.sampler_version,
            policy_version=lambda: self.trainer.policy_version,
            producers=ProducerSettings(
                {name: spec.trainable for name, spec in rollout.producers.items()}
            ),
            freshness=Freshness(
                max_steps_off_policy=config.training.steps_off_policy,
                max_policy_delta_in_trajectory=config.training.policy_delta_in_trajectory,
            ),
            sampling_policy=SamplingPolicy(
                temperature=config.sampling.temperature,
                max_generation_tokens=config.sampling.max_generation_tokens,
                max_trajectory_tokens=config.trajectory_tokens,
            ),
            records_generations=config.training.router_replay,
            on_unrecordable_prompt=unrecordable_prompt,
            on_sample=record,
            on_discard=lambda identity, sample: dropped(identity, sample, delivered=True),
            on_undelivered=lambda identity, sample: dropped(identity, sample, delivered=False),
            records_transcripts=records_transcripts,
            on_transcript=write_transcript if keeps_transcripts else None,
        )

    def _lease(self, wiring: LeaseWiring) -> Callable[[AttemptIdentity], AttemptLease]:
        affinity = self.config.compute.sampler_replica_affinity

        def lease(identity: AttemptIdentity) -> AttemptLease:
            key = affinity_key(self._affinity_namespace, identity) if affinity else None
            return AttemptLease(identity, wiring, affinity_key=key)

        return lease

    def _driver_settings(self) -> DriverSettings:
        config = self.config
        rollout = config.rollout
        if self.rollout_timeout_seconds is None:
            raise RuntimeError("the rollout clock is resolved by preflight()")
        return DriverSettings(
            run_id=self.run_id,
            rollout_timeout_seconds=self.rollout_timeout_seconds,
            setup_timeout_seconds=rollout.setup_timeout_seconds,
            max_rollout_retries=rollout.max_rollout_retries,
            grading_timeout_seconds=rollout.grading.timeout_seconds,
            cleanup_timeout_seconds=rollout.cleanup.timeout_seconds,
            cancellation_grace_seconds=rollout.cleanup.cancellation_grace_seconds,
            process_kill_grace_seconds=rollout.cleanup.process_kill_grace_seconds,
            max_steps_off_policy=config.training.steps_off_policy,
            max_policy_delta_in_trajectory=config.training.policy_delta_in_trajectory,
            max_consecutive_failures=rollout.max_consecutive_rollout_failures,
        )

    async def _unless_ended(
        self, error: BaseException, status: str, reason: str
    ) -> tuple[str, str]:
        """``(status, reason)``, or ``loops_run_ended`` with what Loops reports when the
        Loops run ended under this one: whatever noticed first, that is the cause.

        Asked only once the failure is recorded, so a cancellation while Loops answers
        cannot lose the record; an ended run then records again, as such.
        """
        ended = await ended_under(self.trainer.run_ended)
        if not ended:
            return status, reason
        self._record_failure(error, "loops_run_ended", quiet=True)
        logger.error(
            "The Loops run had ended under the run (%s): recorded as loops_run_ended", ended
        )
        return "loops_run_ended", ended

    def _interrupted(self, error: BaseException) -> None:
        """Record a run stopped by a cancellation, and say how shutdown goes on."""
        self._notice = (
            "Shutdown requested -- draining attempts for up to "
            f"{self.config.rollout.shutdown_drain_seconds:g}s, then closing the runtime; "
            "any cleanup that is not confirmed is reported"
        )
        self._record_failure(error, "interrupted")

    def _record_failure(self, error: BaseException, status: str, *, quiet: bool = False) -> None:
        """Write what ended the run, before teardown can bury it. Never raises.

        ``quiet`` rewrites a record already announced, under a status found since.

        Cleanup runs for up to the shutdown drain and fills the log with cancelled
        trials and sandbox errors, so the original traceback surfaces thousands of
        lines later -- or not at all, if the process does not survive to print it.
        A run that died is exactly the run whose cause you need, so it is written
        first, next to the run's own artifacts.
        """
        record = {
            "status": status,
            "at": time(),
            "last_committed_step": getattr(self._last_commit, "step", None),
            "error": _describe(error),
            "traceback": "".join(traceback.format_exception(error)),
        }
        destination = record_failure(self.out, record)
        if destination is None or quiet:
            return
        if status == "interrupted":
            # Not an error -- somebody asked for this. What they need next is not the
            # cause but how long the shutdown takes, that its noise is expected, and
            # how cleanup that did not finish will show.
            logger.warning(
                "Interrupted; shutting down. In-flight attempts get up to %gs to settle, "
                "then the runtime closes; errors about resources that have already "
                "gone are expected, and not the cause. Any cleanup that is not "
                "confirmed is logged before exit. Recorded in %s.",
                self.config.rollout.shutdown_drain_seconds,
                destination,
            )
            return
        logger.error(
            "Run %s: %s. Cause recorded in %s; the cleanup output below is a "
            "consequence of it, not the cause.",
            status,
            record["error"],
            destination,
        )

    def reconcile_resume(self, tasks: list[TaskRef], renderer: Renderer) -> None:
        """Record this run's contract and hold a resume to the one it continues."""
        self._resume_contract = self._contract(tasks, renderer)
        if self._resume:
            self._reconcile_contract(self._resume.contract, self._resume_contract)

    def _reconcile_contract(self, stored: dict[str, Any], current: dict[str, Any]) -> None:
        """Refuse what cannot work, reset what cannot mean anything, report the rest.

        Only a little of a contract is unsound to change. Weights trained for another
        base model or LoRA rank will not load at all, so that is refused. A task
        cursor counts groups through one ordering of one task list, so against a
        different seed or task set it points nowhere and is restarted. Everything
        else -- batch shape, producers, tokenizer, renderer and its options, sampling policy, the
        runtime's configuration, files and evidence digests -- only makes the two halves of the run
        incomparable, and whether that matters is the caller's judgement. It is
        recorded so a later comparison knows the settings moved mid-run.
        """
        before, now = stored.get("training", {}), current["training"]
        # A contract recorded before runs kept where the renderer came from had no
        # cookbook selection: Prime chose unless the config named a renderer. The
        # class it built is unknown, so that is not compared.
        before = {
            "renderer_source": before.get("renderer") or renderer_source("", None),
            "renderer_class": now["renderer_class"],
            **before,
        }
        incompatible = [
            f"{name}: {before.get(name)!r} -> {now.get(name)!r}"
            for name in ("base_model", "lora_rank")
            if before.get(name) != now.get(name)
        ]
        if incompatible:
            raise ValueError(
                "this checkpoint's weights cannot load into this run ("
                + "; ".join(incompatible)
                + ")"
            )
        changed = sorted(
            name
            for name in set(before) | set(now)
            if name not in ("base_model", "lora_rank") and before.get(name) != now.get(name)
        )
        # A contract recorded before runs kept runtime evidence has none to compare.
        stored = {"runtime_evidence_digests": current["runtime_evidence_digests"], **stored}
        for name in (
            "runtime_config_digest",
            "runtime_file_digests",
            "runtime_evidence_digests",
            "producers",
        ):
            if stored.get(name) != current[name]:
                changed.append(name)
        if changed:
            logger.warning(
                "Resuming with changed settings (%s). Training continues from the "
                "checkpoint, but this run is not comparable with the one that produced it.",
                ", ".join(changed),
            )
        if (
            before.get("shuffle_seed") != now.get("shuffle_seed")
            or stored.get("tasks") != current["tasks"]
        ):
            logger.warning(
                "Task set or order changed, so the checkpoint's cursor at group %d "
                "means nothing here; starting the task order from the beginning.",
                self._next_group_index,
            )
            self._next_group_index = 0
            self._pending_group_indices = []

    def _contract(self, tasks: list[TaskRef], renderer: Renderer) -> dict[str, Any]:
        config = self.config
        rollout = config.rollout
        training = {
            "base_model": config.model.base_model,
            "tokenizer": config.model.tokenizer,
            "renderer": config.model.renderer,
            "renderer_source": renderer_source(
                config.model.tokenizer_source, config.model.renderer
            ),
            "renderer_class": describe(renderer),
            "renderer_chat_template_kwargs": config.model.renderer_chat_template_kwargs,
            "reasoning_efforts": config.model.reasoning_efforts,
            "lora_rank": config.model.lora_rank,
            "init_from": config.training.init_from,
            "max_seq_len": config.model.max_seq_len,
            "group_size": config.training.group_size,
            "max_group_losses": config.training.max_group_losses,
            "groups_per_step": config.training.groups_per_step,
            "shuffle_seed": rollout.shuffle_seed,
            "temperature": config.sampling.temperature,
            "max_generation_tokens": config.sampling.max_generation_tokens,
            "max_trajectory_tokens": config.sampling.max_trajectory_tokens,
        }
        files = {alias: _content_digest(path) for alias, path in self._runtime_files().items()}
        return {
            "training": json.loads(json.dumps(training)),
            "tasks": [[task.id, task.digest] for task in tasks],
            "runtime_config_digest": _digest(_canonical(config.runtime.config)),
            "runtime_file_digests": files,
            "runtime_evidence_digests": self._runtime_evidence.digests,
            "producers": {name: spec.trainable for name, spec in rollout.producers.items()},
        }

    async def _last_gasp(self, pipeline: Pipeline) -> None:
        """Try to leave the last committed step resumable as the run goes down.

        Only useful when that step fell between the checkpoint cadences -- if it
        was already checkpointed there is nothing to add. Best effort by design:
        when the trainer is what died, this cannot succeed, and saying so is more
        use than raising over it while the caller is already unwinding.
        """
        report = self._last_commit
        if report is None or report.checkpoint_state_path:
            return
        if self._committing:
            # Cut between the optimizer step and publication: the trainer's state is
            # neither the last committed step's nor a published one, so saving it
            # under any step's name would make a resume one update off.
            logger.warning(
                "A commit was cut short, so the trainer's state is not step %d's; no "
                "last-gasp checkpoint is saved",
                report.step,
            )
            return
        state_path = await self.trainer.checkpoint_state_only(report.step)
        if not state_path:
            return
        self._checkpoint_progress(report, pipeline, state_path=state_path)
        logger.warning(
            "Saved a last-gasp checkpoint for step %d; resume from %s",
            report.step,
            self.out / "checkpoints" / f"step-{report.step}.json",
        )

    def add_step_sink(self, sink: StepSink) -> Callable[[], None]:
        """Observe every optimizer update; returns the call that detaches this sink.

        A registry rather than a settable attribute, so attaching telemetry is not
        a thing observers can do to each other: two can watch the same run, either
        can leave without knowing about the other, and one attached after the run
        started still receives every update from then on.
        """
        self._step_sinks.append(sink)

        def detach() -> None:
            """Idempotent: a second call is a no-op, not a ``ValueError``."""
            if sink in self._step_sinks:
                self._step_sinks.remove(sink)

        return detach

    def _emit_step(self, event: StepEvent) -> None:
        """Offer one update to each observer. A failing one never ends the run.

        This is reached after the update has been applied, checkpointed and written
        to the run record, and there is no recovery past this point in the loop --
        so a sink raising here would discard a finished step, and with it a run that
        may be days old, to report a fault in something that was only watching.
        """
        for sink in list(self._step_sinks):
            try:
                sink(event)
            except Exception:
                logger.exception("A step observer failed; the update stands regardless")

    def _record_identity(self) -> None:
        """Write down which Loops run this is, and say so. Never raises.

        The id is assigned at start and, until now, existed only in the log -- so
        asking "which run do I reuse?" an hour later meant scrolling back through a
        teardown. It belongs beside the run's other artifacts: `config.json` is what
        was asked for, this is what was created, `failure.json` is how it ended.
        """
        run_id = self._loops_run_id()
        if not run_id:
            return
        if self.trainer.serverless:
            logger.info("Loops run %s:", run_id)
        else:
            logger.info("Loops run %s -- reuse it with reuse_from_run_id:", run_id)
        note(f"  {self.loops_url}")
        record_identity(self.out, run_id)

    @property
    def notice(self) -> str | None:
        """A line about the run's lifecycle, for pinning where it cannot scroll away.

        Words rather than gauges, which is why it is not in ``snapshot``: a teardown
        takes minutes and buries itself in Harbor's per-trial output, so "this is
        shutting down" has to stay on screen rather than be the line you scroll back
        to find.
        """
        return self._notice

    def _loops_run_id(self) -> str | None:
        """The trainer's Loops run id; None before it has one, or without a Loops client."""
        return getattr(getattr(self.trainer, "client", None), "run_id", None)

    @property
    def evals_running(self) -> int:
        """Set evals started and not yet ended, a final one waiting its turn included."""
        return sum(not run.done() for run in self._set_eval_runs)

    @property
    def resume_step(self) -> int | None:
        """The step this run resumes at, from the manifest it was given; ``None`` for a
        fresh run."""
        return self._resume.next_step if self._resume else None

    @property
    def loops_url(self) -> str | None:
        """This run's page on Baseten, or ``None`` until the trainer has an id.

        Only available once the trainer has started, which is exactly when it
        becomes worth showing -- before that there is nothing to look at.
        """
        return loops_url(self._loops_run_id())

    def snapshot(self) -> dict[str, float] | None:
        """Everything the run knows about itself right now, or ``None`` if nothing.

        The pull half of this run's observability, and the only definition of its
        current state: the heartbeat, the terminal region and the per-update record
        all read this, so none of them can drift from the others.

        ``None`` means there is no live pipeline to describe -- the run is still
        provisioning, or has stopped -- and is distinct from a reading that happens
        to be all zeroes, which is a real state and says the pipeline is idle. A
        live pipeline is the only thing this waits on, so it doubles as the flag.

        Safe to call from another task at any moment -- before the pipeline exists,
        between batches, while the run is unwinding. It reads counters and queue
        sizes, never awaits and never mutates, so an observer's timer cannot
        interleave with the orchestration it is watching.
        """
        if self._pipeline is None:
            return None
        return {
            # The caps travel with the counts so a reader never needs the config to
            # know whether 39 sampling is busy or nearly stalled.
            "progress/step": float(self._step),
            # Which half of the loop the run is in. A gauge rather than a notice, so
            # it reaches W&B and the progress file too -- on a long collection phase
            # "has it started training yet" is the question, and the answer belongs
            # beside the counts that explain the wait.
            "progress/training": float(self._training),
            "progress/policy_version": float(self.trainer.policy_version),
            "progress/groups_needed": float(self.config.training.groups_per_step),
            "progress/rollouts_needed": float(self.config.training.rollouts_per_step),
            "progress/rollout_slots": float(self.config.rollout.max_concurrent_rollouts),
            "progress/verifier_slots": float(self.config.rollout.max_concurrent_verifiers),
            **self._pipeline.snapshot(),
            **self._discard_gauges(),
            **self._trace_gauges(),
            **self._metrics.snapshot(),
        }

    def _discard_gauges(self) -> dict[str, float]:
        """Generations sampled and then left out of the evidence, since this process began.

        `discard/generations` counts them all: the ones the agent discarded, and the
        ones sealing dropped because they were never delivered, which
        `discard/undelivered_generations` counts on its own. Batch metrics describe
        what survived selection, so these would otherwise be invisible:
        `trajectory/sampled_tokens` would understate what the sampler actually
        produced, with nothing saying by how much.

        Not carried across a resume. The checkpoint manifest is a validated contract
        for what training needs to continue correctly, and a diagnostic counter is
        not that; a resumed run counts its own discards from zero.
        """
        return {
            "discard/generations": float(self._discarded["generations"]),
            "discard/generation_tokens": float(self._discarded["tokens"]),
            "discard/undelivered_generations": float(self._undelivered["generations"]),
            "discard/undelivered_generation_tokens": float(self._undelivered["tokens"]),
        }

    def _trace_gauges(self) -> dict[str, float]:
        """Trace records that could not be written, since this process began.

        A failed write never fails its attempt, so these are the only sign that the
        trace beside the run record is incomplete, and by how much.
        """
        return {
            "trace/unwritten_samples": float(self._unwritten["samples"]),
            "trace/unwritten_transcripts": float(self._unwritten["transcripts"]),
        }

    def _rollout_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """A rollouts.jsonl record, with what its attempt sampled that is not in its evidence.

        The counts are the recorded attempt's: generations sealing dropped because they
        were never delivered, and generations the agent discarded. A rollout recorded
        without an attempt sealed none, so both are zero.
        """
        attempts = self._dropped.pop(record["rollout_id"], {})
        undelivered, discarded = attempts.get(record["attempt_id"], (0, 0))
        return {
            **record,
            "generations_undelivered": undelivered,
            "generations_discarded_by_agent": discarded,
        }

    def _checkpoint_progress(
        self, report, pipeline: Pipeline, state_path: str | None = None
    ) -> None:
        path = state_path or report.checkpoint_state_path
        if not path:
            return
        Manifest(
            trainer_state_path=path,
            next_step=report.step + 1,
            next_group_index=pipeline.next_group_index,
            pending_group_indices=pipeline.pending_group_indices,
            policy_version=report.policy_version_after,
            shuffle_seed=self.config.rollout.shuffle_seed,
            contract=self._resume_contract,
            run_id=self.run_id,
            run_directory=str(self.out),
            seq=self._manifest_order.next(),
        ).write(self.out, report.step)

    def _record_step(self, pipeline: Pipeline, report, groups, batch: PreparedBatch) -> None:
        """Write one applied update to the run record, tell observers, and log it."""
        # Split by how often each source can report. The snapshot is also what
        # heartbeats carry, and those have no step, so only what an update alone
        # produces can be plotted per update. `or {}`: the accessor is nullable,
        # and the caller only guards against OSError, so a TypeError here would
        # discard an update that is already applied and checkpointed.
        shared = self.snapshot() or {}
        per_update = _update_gauges(
            report,
            groups,
            batch,
            pipeline.take_unusable_rewards(),
            pipeline.take_unusable_metadata(),
        )
        if self.config.training.mode == "async":
            per_update["admission/window_held_share"] = pipeline.take_window_held_share()
        # A name both sources report is a heartbeat name; it keeps the time axis
        # rather than changing meaning between records.
        gauges = {**shared, **per_update}
        step_scoped = per_update.keys() - shared.keys()
        for group in groups:
            for rollout in group.members:
                _append_json(
                    self.out / "rollouts.jsonl",
                    self._rollout_record(pipeline.trained_record(rollout, self._step)),
                )
        _append_json(
            self.out / "steps.jsonl",
            {
                "step": self._step,
                "recorded_at": time(),
                **gauges,
                "forward_backward_reports": [
                    asdict(call) for call in report.forward_backward_reports
                ],
            },
        )
        self._emit_step(StepEvent(self._step, gauges, frozenset(step_scoped)))
        # What the update produced, and nothing about where the pipeline is: the
        # region shows that live and the heartbeat records it. `lag` is how stale
        # the data trained on was, and `std` is the early warning for the
        # no-variance discards that stall a batch.
        logger.info(
            "step %d: loss %.4f  reward %.4f \u00b1%.3f  %d datums  lag %d  train %.1fs  total %s",
            self._step,
            report.loss,
            report.reward_mean,
            report.reward_std,
            report.datums,
            report.policy_version_before - report.sampled_version_min,
            sum(report.timings.values()),
            elapsed(gauges),
        )

    def _resources(self) -> ResourceRegistry:
        if self._registry is None:
            raise RuntimeError("resources are opened by preflight()")
        return self._registry

    def _run_scope(self) -> RunResourceService:
        """The one run-scope service, whose leases shutdown ends before releasing."""
        if self._run_resources is None:
            self._run_resources = RunResourceService(self._resources(), self.run_id)
        return self._run_resources

    async def _open_resources(self) -> ResourceRegistry:
        """The run's resource registry, after releasing what a previous process left.

        A resumed run reopens its own journal. Anything it cannot prove gone stays
        in the journal for ``loops cleanup``, and the run goes ahead.
        """
        if self._registry is not None:
            return self._registry
        permits = asyncio.Semaphore(self.config.rollout.max_concurrent_provisioning)

        @contextlib.asynccontextmanager
        async def permit() -> AsyncIterator[None]:
            async with permits:
                yield

        state_dir = self.out / self.config.resources.state_dir
        registry = ResourceRegistry.open(
            state_dir,
            self._adapters,
            mode="write",
            run_id=self.run_id,
            permit=permit,
            create_window=self.config.rollout.max_concurrent_provisioning,
        )
        self._registry = registry
        leftover = await registry.release_all()
        if leftover.results:
            logger.info(
                "Released %d resource(s) a previous process of this run left behind",
                len(leftover.results) - len(leftover.unresolved),
            )
        if leftover.unresolved:
            self._report_unresolved(leftover, state_dir)
        return registry

    async def close_resources(self) -> None:
        """Shutdown steps 6 and 7: release the run scope, then close adapters and journal.

        Provisioning stops first, and both steps share one bound,
        ``shutdown_drain_seconds``: what is still in flight then is cut off and
        reported, with the rest of what is unresolved, for ``loops cleanup``.
        """
        registry = self._registry
        if registry is None:
            # The journal never opened, so no registry took the adapters: they are
            # still the run's to close.
            adapters, self._adapters = self._adapters, {}
            await close_adapters(adapters)
            return
        scope = self._run_resources or RunResourceService(registry, self.run_id)
        self._registry = self._run_resources = None
        self._adapters = {}  # the registry closes them
        bound = self.config.rollout.shutdown_drain_seconds
        loop = asyncio.get_running_loop()
        deadline = loop.time() + bound
        registry.begin_shutdown()
        try:
            async with asyncio.timeout_at(deadline):
                await scope.close()
        except TimeoutError:
            logger.warning(
                "Releasing the run's resources did not finish within shutdown_drain_seconds "
                "(%gs); what is left is cut off and reported",
                bound,
            )
        finally:
            try:
                # Waits for creates and releases still in flight, up to the same deadline,
                # then gives what it cuts off, and the adapters, a short grace to stop.
                await registry.close(timeout=max(0.0, deadline - loop.time()))
            finally:
                # Reported however the close ended: failed, or cut short by an interrupt.
                unresolved = registry.unresolved
                if unresolved.results:
                    self._report_unresolved(unresolved, registry.journal.path.parents[2])

    def _report_unresolved(self, report: CleanupReport, state_dir: Path) -> None:
        self._unclean = True
        logger.error(
            "%d resource(s) are not proven released: %s. Run `loops cleanup "
            "<training.toml> --state-dir %s` to finish.",
            len(report.unresolved),
            "; ".join(
                f"{result.resource_id} ({result.status.value}: {result.detail})"
                for result in report.unresolved
            ),
            state_dir,
        )

    async def preflight(
        self, *, training: bool = True
    ) -> tuple[RolloutRuntime, list[TaskRef], Renderers]:
        """Build the runtime, discover its tasks, and validate what it declares.
        ``training``: whether the run trains, or only evaluates (``loops eval``)."""
        config = self.config
        check_open_file_limit(config, training=training)
        await self._open_resources()
        factory = self._runtime_factory
        self._runtime_evidence = EvidenceRecord()
        runtime = build_runtime(
            factory, config.runtime.config, self._runtime_context(self._runtime_evidence)
        )
        try:
            try:
                prepared = await runtime.prepare()
            except Exception as error:
                raise RuntimeExecutionFailure("prepare", None, error) from error
            tasks = check_tasks(prepared)
            capabilities = check_capabilities(
                runtime.capabilities,
                Preflight(
                    model_access=None
                    if config.runtime.model_access is None
                    else ModelAccessMode(config.runtime.model_access),
                    rollout_timeout_seconds=config.rollout.rollout_timeout_seconds,
                ),
            )
            self.rollout_timeout_seconds = rollout_clock(
                config.rollout.rollout_timeout_seconds, capabilities
            )
            self.model_access = model_access_mode(
                None
                if config.runtime.model_access is None
                else ModelAccessMode(config.runtime.model_access),
                capabilities,
            )
            if self.model_access is not ModelAccessMode.OPENAI_HTTP and config.runtime.openai:
                raise RuntimeConfigurationError(
                    f"[runtime.openai] is set, and the runtime's model access is "
                    f"{self.model_access.value!r}; it applies only to openai_http"
                )
            self.group_grader = group_grader(runtime, capabilities)
            self.eval_sets = await self._build_eval_sets(tasks)
            renderers = self._renderers()
        except BaseException:
            await self.close_runtimes(runtime)
            raise
        self.capabilities = capabilities
        return runtime, tasks, renderers

    async def _build_eval_sets(self, training: list[TaskRef]) -> tuple[EvalSet, ...]:
        """Each ``[eval.sets]`` entry: training's runtime factory and config on the set's
        files, prepared and checked as training's runtime is.

        A set whose runtime resolves another model access is refused, since one gateway
        serves both. Tasks it shares with training are counted and warned about.
        """
        config = self.config
        configured = (
            None
            if config.runtime.model_access is None
            else ModelAccessMode(config.runtime.model_access)
        )
        ids = {task.id for task in training}
        digests = {task.digest for task in training}
        built: list[EvalSet] = []
        try:
            for name, settings in config.eval.sets.items():
                metrics = MetricsSink()
                runtime = build_runtime(
                    self._runtime_factory,
                    config.runtime.config,
                    self._runtime_context(
                        EvidenceRecord(),
                        files=settings.runtime.files,
                        work_dir=self.out / "eval" / name / "runtime",
                        max_concurrent_rollouts=settings.max_concurrent_rollouts,
                        metrics=metrics,
                    ),
                )
                try:
                    try:
                        prepared = await runtime.prepare()
                    except Exception as error:
                        raise RuntimeExecutionFailure("prepare", None, error) from error
                    tasks = check_tasks(prepared)
                    capabilities = check_capabilities(
                        runtime.capabilities,
                        Preflight(
                            model_access=configured,
                            rollout_timeout_seconds=config.rollout.rollout_timeout_seconds,
                        ),
                    )
                    mode = model_access_mode(configured, capabilities)
                    if mode is not self.model_access:
                        raise RuntimeConfigurationError(
                            f"eval set {name!r} resolves model access {mode.value!r}, and "
                            f"training {getattr(self.model_access, 'value', None)!r}; both "
                            "must use the same"
                        )
                    grader = group_grader(runtime, capabilities)
                    clock = rollout_clock(config.rollout.rollout_timeout_seconds, capabilities)
                except BaseException:
                    await self.close_runtime(runtime)
                    raise
                evaluated = tasks[: settings.max_tasks]
                shared = [t.id for t in evaluated if t.id in ids or t.digest in digests]
                self.eval_shared[name] = len(shared)
                if shared:
                    logger.warning(
                        "eval set %r shares %d task(s) with training (%s); its scores "
                        "include tasks the policy trains on",
                        name,
                        len(shared),
                        ", ".join(shared[:5]) + (", ..." if len(shared) > 5 else ""),
                    )
                built.append(
                    EvalSet(
                        name=name,
                        settings=settings,
                        runtime=runtime,
                        capabilities=capabilities,
                        group_grader=grader,
                        tasks=tuple(evaluated),
                        rollout_timeout_seconds=clock,
                        metrics=metrics,
                        slots=SetSlots(settings.max_concurrent_rollouts),
                    )
                )
        except BaseException:
            await self._close_each(eval_set.runtime for eval_set in built)
            raise
        return tuple(built)

    def _open_gateway(self) -> Gateway | None:
        """The run's Chat Completions gateway, for ``model_access = "openai_http"``.

        It binds its listener when the first attempt asks for a connection.
        """
        runtime = self.config.runtime
        if self.model_access is not ModelAccessMode.OPENAI_HTTP:
            return None
        settings = runtime.openai or OpenAISettings()
        return Gateway(
            GatewaySettings(
                bind=settings.bind,
                advertised_base_url=settings.advertised_base_url,
                model=settings.model_name or self.config.model.base_model,
                shutdown_seconds=self.config.rollout.cleanup.cancellation_grace_seconds,
            )
        )

    def _renderers(self) -> Renderers:
        """The run renderer and one per ``[model.reasoning_efforts]`` value, each probed.

        Every renderer comes from the same factory: an effort only overrides the
        template options its table entry names.
        """
        model = self.config.model
        base = self._renderer_factory(self.config)
        _warn_without_images(base)
        by_effort: dict[str, Renderer] = {}
        for effort, overrides in sorted(model.reasoning_efforts.items()):
            options = {**model.renderer_chat_template_kwargs, **overrides}
            variant = self.config.model_copy(
                update={
                    "model": model.model_copy(update={"renderer_chat_template_kwargs": options})
                }
            )
            by_effort[effort] = self._renderer_factory(variant)
            _warn_without_images(by_effort[effort])
        return Renderers(base, by_effort, self._render_pool)

    def _start_eval(
        self, step: int, path: str | None, version: int, *, final: bool = False
    ) -> None:
        """Evaluate every eval set at ``step`` in the background, on a sampler pinned to
        ``path`` (the base model when None) at policy ``version``. On dedicated Loops a
        set whose last eval is still running skips this one, unless it is the ``final``
        one, which waits. On serverless a set's evals overlap, each with its own cap."""
        if not self.eval_sets:
            return
        if path is not None and not path.startswith("bt://loops:"):
            # Only a dev trainer's local weight sync publishes such a path.
            logger.warning(
                "eval at step %d skipped: the trainer returned %r, which a sampler cannot serve",
                step,
                path,
            )
            return
        runs = []
        serverless = self.trainer.serverless
        for eval_set in self.eval_sets:
            # A serverless sampler costs nothing while it waits, so nothing is saved by
            # running a set's evals one at a time.
            previous = None if serverless else self._running_eval(eval_set)
            if previous is not None and not final:
                self._skip_busy(eval_set, step, previous[0])
                continue
            run = asyncio.create_task(
                self._eval_set(
                    replace(eval_set, slots=None) if serverless else eval_set,
                    step,
                    path,
                    version,
                    after=previous,
                ),
                name=f"eval-{eval_set.name}-{step}",
            )
            self._set_evals[eval_set.name] = step, run
            self._set_eval_runs.append(run)
            runs.append(run)
        if runs:
            # The sets of one step at once, each on its own sampler. Cancelled, it ends
            # only once every set has cleaned up.
            self._evals[step] = asyncio.gather(*runs, return_exceptions=True)

    def _running_eval(self, eval_set: EvalSet) -> tuple[int, asyncio.Task[None]] | None:
        latest = self._set_evals.get(eval_set.name)
        return latest if latest is not None and not latest[1].done() else None

    def _skip_busy(self, eval_set: EvalSet, step: int, running: int) -> None:
        self._evals_ended.add((eval_set.name, step))
        logger.warning(
            "eval %r at step %d skipped: its eval %s is still running, and on dedicated "
            "Loops a set runs one eval at a time. Raise [eval] every_steps to evaluate "
            "every one.%s",
            eval_set.name,
            step,
            "before training" if running < 0 else f"at step {running}",
            _later(step),
        )

    async def _eval_set(
        self,
        eval_set: EvalSet,
        step: int,
        path: str | None,
        version: int,
        *,
        after: tuple[int, asyncio.Task[None]] | None = None,
    ) -> None:
        """One set's eval: its own sampler, always deactivated; its rows and gauges.
        ``after`` is the set's eval still running, which the final eval waits for.

        Nothing here reaches the run but a cancellation: a skipped or failed eval is
        logged, and training goes on.
        """
        config, name = self.config, eval_set.name
        where = "before training" if step < 0 else f"at step {step}"
        if after is not None:
            await asyncio.wait({after[1]})
        if eval_set.slots is not None and eval_set.slots.all_kept:
            self._evals_ended.add((name, step))
            logger.error(
                "eval %r %s skipped: hooks of its earlier evals ignored cancellation past "
                "their grace and hold all %d of its slots until they end. Training continues.",
                name,
                where,
                eval_set.slots.cap,
            )
            return
        started = monotonic()
        try:
            sampler = await self.trainer.eval_sampler(
                path, ready_timeout=config.eval.sampler_start_timeout_seconds
            )
        except asyncio.CancelledError:
            logger.error(
                "eval %r %s stopped while its sampler was starting. %s", name, where, _MAY_BE_UP
            )
            raise
        except Exception as error:  # noqa: BLE001 - a sampler that did not start skips this eval
            self._evals_ended.add((name, step))
            logger.error(
                "eval %r %s skipped: its sampler was not ready (%s: %s). On dedicated Loops "
                "each eval starts a sampler deployment, which needs free allocation for %s "
                "at max_seq_len %s; raise [eval] sampler_start_timeout_seconds or evaluate "
                "less often. Training continues.%s %s",
                name,
                where,
                type(error).__name__,
                error,
                config.model.base_model,
                config.model.max_seq_len,
                _later(step),
                _MAY_BE_UP,
            )
            return
        ready = monotonic() - started
        metrics = MetricsSink()  # this eval's alone: evals of other steps run beside it
        try:
            result = await evaluate(
                eval_set,
                step=step,
                verifiers=config.rollout.max_concurrent_verifiers,
                driver=self._eval_driver(eval_set, sampler, version, metrics),
                group_grading=GroupGradingSettings(
                    timeout_seconds=config.rollout.grading.timeout_seconds,
                    cancellation_grace_seconds=config.rollout.cleanup.cancellation_grace_seconds,
                ),
                drain_seconds=config.rollout.shutdown_drain_seconds,
                on_unsettled=self._eval_unsettled,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self._evals_ended.add((name, step))
            logger.exception("eval %r %s failed; training continues.%s", name, where, _later(step))
            return
        finally:
            await see_through(asyncio.ensure_future(self.trainer.deactivate_sampler(sampler)))
        try:
            stem = "base" if step < 0 else f"step-{step}"
            await _in_thread(
                _write_eval, self.out / "eval" / name, stem, result.rows, result.transcripts
            )
            self._record_eval(eval_set, step, result, metrics, ready, monotonic() - started)
            self._evals_written.add(name)
        except Exception:  # recording one eval never ends the others
            self._evals_ended.add((name, step))
            logger.exception("eval %r %s could not be recorded; training continues", name, where)
            return
        self._evals_ended.add((name, step))

    def _record_eval(
        self,
        eval_set: EvalSet,
        step: int,
        result: EvalResult,
        metrics: MetricsSink,
        ready: float,
        seconds: float,
    ) -> None:
        """One eval's gauges, and what its episodes reported, to the step sinks; its line
        to the log."""
        name = eval_set.name
        where = "before training" if step < 0 else f"at step {step}"
        gauges = {
            **result.gauges,
            **{f"eval/{name}/{key}": value for key, value in metrics.snapshot().items()},
            # What the set's runtime reports through its context, across its evals so far.
            **{
                f"eval/{name}/set/{key}": value
                for key, value in eval_set.metrics.snapshot().items()
            },
            f"eval/{name}/sampler_start_seconds": ready,
            f"eval/{name}/seconds": seconds,
        }
        self._emit_step(StepEvent(step, gauges, frozenset(gauges)))
        discarded = int(gauges[f"eval/{name}/rollouts"] - gauges[f"eval/{name}/graded"])
        logger.info(
            "eval %r %s: reward %s over %d rollouts (%d discarded), sampler ready in %.1fs, "
            "%.0fs in all",
            name,
            where,
            (
                f"{gauges[f'eval/{name}/reward_mean']:.4f} "
                f"\u00b1{gauges[f'eval/{name}/reward_std']:.3f}"
            )
            if f"eval/{name}/reward_mean" in gauges
            else "none",
            int(gauges[f"eval/{name}/rollouts"]),
            discarded,
            ready,
            seconds,
        )

    def _eval_unsettled(self) -> None:
        self._unclean = True

    def _eval_driver(self, eval_set: EvalSet, sampler: Any, version: int, metrics: MetricsSink):
        """The driver factory ``evaluate`` calls: training's settings but the set's own
        rollout clock, the eval's own ``metrics``, training's access and resources, over
        leases pinned to the eval's sampler and version."""
        access = self._access
        if access is None:
            raise RuntimeError("evals start once the run's model access is built")
        wiring = self._eval_wiring(eval_set, sampler, version)

        def driver(
            capacity: RunCapacity, on_systemic: Callable[[SystemicFailure], None]
        ) -> AttemptDriver:
            return AttemptDriver(
                eval_set.runtime,
                eval_set.capabilities,
                replace(
                    self._driver_settings(),
                    rollout_timeout_seconds=eval_set.rollout_timeout_seconds,
                ),
                capacity=capacity,
                # No affinity key: replica affinity is the paired sampler's prefix cache.
                leases=lambda identity: AttemptLease(identity, wiring, affinity_key=None),
                access=access,
                resources=lambda identity, phases: EpisodeResourceService(
                    self._resources(), identity.attempt_id, phases.resource_operation
                ),
                metrics=metrics,
                # Fixed: every sample is at the eval's version, so nothing is stale.
                policy_version=lambda: version,
                on_systemic=on_systemic,
                run_ended=self.trainer.run_ended,
            )

        return driver

    def _eval_wiring(self, eval_set: EvalSet, sampler: Any, version: int) -> LeaseWiring:
        """Training's lease wiring, pinned to the eval's sampler and version, recording
        nothing. Samples keep training's trainable flags, which is what lets the driver
        version them; none trains, since eval rollouts never reach the trainer."""
        config = self.config

        async def nothing(identity: AttemptIdentity, sample: Sample) -> None:
            return None

        temperature = eval_set.settings.sampling.temperature
        pinned = PinnedSampler(sampler)
        return LeaseWiring(
            sampler=lambda: pinned,  # pyright: ignore[reportReturnType]
            # One pinned sampler serves the whole eval, so an attempt binds it once,
            # whatever the run's capacity.
            capacity=Capacity.DEDICATED,
            sampler_version=lambda: version,
            policy_version=lambda: version,
            producers=ProducerSettings(
                {name: spec.trainable for name, spec in config.rollout.producers.items()}
            ),
            freshness=Freshness(
                max_steps_off_policy=config.training.steps_off_policy,
                max_policy_delta_in_trajectory=config.training.policy_delta_in_trajectory,
            ),
            sampling_policy=SamplingPolicy(
                temperature=config.sampling.temperature if temperature is None else temperature,
                max_generation_tokens=config.sampling.max_generation_tokens,
                max_trajectory_tokens=config.trajectory_tokens,
            ),
            on_sample=nothing,
            on_discard=lambda identity, sample: None,
            on_undelivered=lambda identity, sample: None,
            # An eval records every conversation, whatever [output] trace says: it is
            # small, and what the policy said is the point of evaluating it.
            records_transcripts=True,
        )

    async def _finish_evals(self, *, wait: bool) -> None:
        """Wait for the evals still running, or cancel them; either way each one
        deactivates its sampler before this returns. Then name the ones that did not
        finish."""
        tasks = list(self._evals.values())
        if tasks:
            gathered = asyncio.gather(*tasks, return_exceptions=True)
            if not wait:
                for task in tasks:
                    task.cancel()
            try:
                for outcomes in await asyncio.shield(gathered):
                    for outcome in outcomes if isinstance(outcomes, list) else [outcomes]:
                        if isinstance(outcome, Exception):
                            logger.error("an eval failed: %s", outcome, exc_info=outcome)
            except asyncio.CancelledError:
                for task in tasks:
                    task.cancel()
                await see_through(gathered)
                raise
            finally:
                for step in self._evals:
                    for eval_set in self.eval_sets:
                        if (eval_set.name, step) not in self._evals_ended:
                            where = "before training" if step < 0 else f"at step {step}"
                            logger.warning("eval %r %s did not finish", eval_set.name, where)

    async def close_runtimes(self, runtime: RolloutRuntime) -> None:
        """Close the training runtime and every eval set's, each as ``close_runtime``
        closes one; all are attempted whatever happens to the others."""
        try:
            await self.close_runtime(runtime)
        finally:
            eval_sets, self.eval_sets = self.eval_sets, ()
            await self._close_each(eval_set.runtime for eval_set in eval_sets)

    async def _close_each(self, runtimes: Iterable[RolloutRuntime]) -> None:
        """``close_runtime`` on each; a cancellation cuts one close short, not the
        closes after it, and is raised once every one has been attempted."""
        interrupted: asyncio.CancelledError | None = None
        for runtime in runtimes:
            try:
                await self.close_runtime(runtime)
            except asyncio.CancelledError as cancelled:
                interrupted = interrupted or cancelled
        if interrupted is not None:
            raise interrupted

    async def close_runtime(self, runtime: RolloutRuntime) -> None:
        """Shutdown step 4: ``close()`` under the cleanup deadline, then a bounded grace.

        A ``close()`` that still has not stopped is reported and the run goes on
        closing: shutdown is already under way, so this is not a systemic failure.
        """
        cleanup = self.config.rollout.cleanup
        task = asyncio.create_task(runtime.close(), name="runtime-close")
        try:
            await asyncio.wait({task}, timeout=cleanup.timeout_seconds)
        finally:
            stopped = await stop(task, cleanup.cancellation_grace_seconds)
        stopped.propagate()
        if not stopped.settled:
            logger.error(
                "The runtime's close() did not stop within %gs plus %gs of grace; its "
                "cleanup is not confirmed and the run does not end cleanly.",
                cleanup.timeout_seconds,
                cleanup.cancellation_grace_seconds,
            )
            self._unclean = True
            return
        if task.cancelled():
            logger.error("The runtime's close() was cancelled at its deadline")
            self._unclean = True
        elif (error := task.exception()) is not None:
            logger.error("The runtime's close() failed: %s", _describe(error), exc_info=error)
            self._unclean = True

    def _open_access(self, renderers: Renderers) -> Callable[[AttemptLease], AttemptModelAccess]:
        """How attempts reach the model: directly, or over the run's gateway, opened here."""
        gateway = self._gateway = self._open_gateway()

        def access(lease: AttemptLease) -> AttemptModelAccess:
            if gateway is None:
                return DirectModelAccess(lease, renderers)
            return HttpModelAccess(lease, renderers, gateway)

        self._access = access
        return access

    async def evaluate_after(
        self,
        step: int,
        *,
        sampler_weights: str | None = None,
        trainer_state: str | None = None,
        policy_version: int = 0,
    ) -> list[str]:
        """``loops eval``: evaluate update ``step`` of the run in ``self.out`` (-1: the
        base model) as an eval during the run would, writing the same files, with every
        eval set of the config; the sets that ran it.

        The weights are the ``sampler_weights`` path the run kept for ``step`` at
        ``policy_version``, or else its ``trainer_state``, which a trainer of its own
        saves for a sampler first; that trainer's Loops run is deactivated after.
        """
        runtime: RolloutRuntime | None = None
        made_weights = started = False
        try:
            runtime, _, renderers = await self.preflight(training=False)
            if not self.eval_sets:
                raise ValueError("the config has no [eval.sets] to evaluate with")
            wanted = {eval_set.name for eval_set in self.eval_sets}
            self.trainer.connect()
            if step >= 0 and sampler_weights is None:
                if trainer_state is None:
                    raise ValueError(f"step {step} has neither sampler weights nor trainer state")
                made_weights = True
                sampler_weights, policy_version = await self.trainer.servable_weights(
                    trainer_state, f"eval-step-{step}"
                )
            self._open_access(renderers)
            self._start_eval(step, sampler_weights if step >= 0 else None, policy_version)
            started = True
        finally:
            try:
                # Waits for the evals started; anything raised before cancels them.
                await self._finish_evals(wait=started)
            finally:
                try:
                    if runtime is not None:
                        await self.close_runtimes(runtime)
                finally:
                    try:
                        if self._gateway is not None:
                            await self._gateway.close()
                            self._gateway = None
                    finally:
                        try:
                            self._render_pool.close()
                            await self.close_resources()
                        finally:
                            try:
                                # Whatever else fails to close, the trainer made for
                                # the weights stops billing.
                                if made_weights:
                                    await self.trainer.deactivate_run()
                            finally:
                                await self.trainer.close()
        return sorted(self._evals_written & wanted)

    async def execute(self) -> Outcome:
        config = self.config
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_text(config.model_dump_json(indent=2))
        proxies = _network_environment()
        logger.info(
            "Outbound HTTP goes through: %s",
            ", ".join(f"{scheme}={value}" for scheme, value in proxies.items())
            if proxies
            else "no proxy configured",
        )
        self._unclean = False
        runtime: RolloutRuntime | None = None
        pipeline = None
        self._step = self._resume.next_step if self._resume else 0
        steps_completed = 0
        status, reason = "completed", None
        try:
            runtime, tasks, renderers = await self.preflight()
            active = runtime
            (self.out / "runtime_evidence_digests.json").write_text(
                json.dumps(self._runtime_evidence.digests, indent=2)
            )
            self.reconcile_resume(tasks, renderers.base)
            limit = config.training.step_limit
            if self._resume and run_record.finished(self._step, limit):
                # After the resume checks, so `loops run` refuses what `loops check`
                # refuses; before the trainer, so nothing is provisioned.
                assert limit is not None
                run_record.report_finished(limit, *finished_hint(limit))
                raise _NothingToTrain
            await self.trainer.start()
            self._record_identity()
            if self._resume and self.trainer.policy_version != self._resume.policy_version:
                raise ValueError("restored trainer policy does not match checkpoint manifest")
            self._step = self._resume.next_step if self._resume else self.trainer.resumed_step
            leases = self._lease(self._lease_wiring())
            settings = self._driver_settings()
            access = self._open_access(renderers)
            if self._resume is None and not config.training.init_from:
                # The base model's eval; a run that starts from other weights has none.
                self._start_eval(-1, None, self.trainer.policy_version)

            def driver(
                capacity: RunCapacity,
                on_systemic: Callable[[SystemicFailure], None],
                on_sealed: Callable[[], None],
            ) -> AttemptDriver:
                return AttemptDriver(
                    active,
                    self.capabilities,
                    settings,
                    capacity=capacity,
                    leases=leases,
                    access=access,
                    resources=lambda identity, phases: EpisodeResourceService(
                        self._resources(), identity.attempt_id, phases.resource_operation
                    ),
                    metrics=self._metrics,
                    policy_version=lambda: self.trainer.policy_version,
                    on_systemic=on_systemic,
                    on_sealed=on_sealed,
                    # Only from here: start() returned once the trainer and sampler
                    # were ready, so their readiness wait never meets a status check.
                    run_ended=self.trainer.run_ended,
                )

            pipeline = self._pipeline = Pipeline(
                config,
                tasks=tasks,
                driver=driver,
                initial_group_index=self._next_group_index,
                initial_step=self._step,
                replay_group_indices=self._pending_group_indices,
                # Every rollout's disposition, discarded ones too: sliced by task or
                # length from the file, rather than a metric per task.
                on_rollout=lambda record: _append_json(
                    self.out / "rollouts.jsonl", self._rollout_record(record)
                ),
                group_grader=self.group_grader,
            )
            async with pipeline:
                # This `finally` runs before `__aexit__`, which drains for up to
                # shutdown_drain_seconds. Observers poll throughout that, and a
                # stopped pipeline's frozen counts beside a still-climbing clock
                # read as a run that is working.
                try:
                    while limit is None or self._step < limit:
                        groups = await pipeline.next_batch()
                        self._training = True
                        try:
                            pipeline.start_training(groups)
                            # Seconds of CPU for a large batch, in slices: every
                            # attempt, heartbeat and log line waits at most one.
                            batch = await in_slices(
                                prepare_batch(
                                    groups,
                                    config.training,
                                    metrics_limit=config.output.rollout_metrics_limit,
                                )
                            )
                            # The exact training rows are evidence for replay, and the
                            # largest thing written after the traces themselves.
                            if config.output.trace == "full":
                                await _record_batch(
                                    self.out / "batches" / f"step-{self._step}.json",
                                    self._step,
                                    self.trainer.policy_version,
                                    groups,
                                    batch.rows,
                                    batch.aggregation,
                                )
                            # Only this runner updates the trainer, so its version cannot
                            # change between refresh, backward, and commit.
                            await self.trainer.accumulate(batch.rows)
                            # Accumulated, so the rows -- gigabytes for a large batch
                            # of long rollouts -- go now: the commit, checkpoint and
                            # step record need only what the batch reports.
                            batch = batch.trained()
                            self._committing = True
                            report = await self.trainer.commit(
                                self._step,
                                groups=len(groups),
                                rollouts=len(batch.rewards),
                                rewards=batch.rewards,
                                advantages_=batch.advantages,
                                sampled_versions=batch.versions,
                            )
                            report.aggregation = batch.aggregation
                            self._committing = False
                        except BaseException:
                            self._training = False
                            pipeline.abort_batch(groups)
                            raise
                        # Published, so applied: the batch is trained and the step
                        # counts whatever happens to its checkpoint, or a resume would
                        # repeat an update the sampler already serves.
                        pipeline.commit_batch(groups)
                        self._last_commit = report
                        try:
                            final = limit is not None and self._step + 1 >= limit
                            evaluated = config.eval.due(self._step, final=final)
                            report = await self.trainer.checkpoint(
                                report, final=final, keep_sampler_weights=evaluated
                            )
                            self._checkpoint_progress(report, pipeline)
                            self._last_commit = report
                            if report.sampler_weights is not None:
                                # What `loops eval` serves to evaluate this step later.
                                _append_json(
                                    self.out / SAMPLER_WEIGHTS,
                                    {
                                        "step": self._step,
                                        "path": report.sampler_weights,
                                        "policy_version": report.policy_version_after,
                                        "written_at": time(),
                                    },
                                )
                            # No weights path: the trainer said so, and the eval is skipped.
                            if evaluated and report.sampler_weights is not None:
                                self._start_eval(
                                    self._step,
                                    report.sampler_weights,
                                    report.policy_version_after,
                                    final=final,
                                )
                        finally:
                            self._training = False
                            # Bookkeeping about an update that is already applied; none
                            # of it can be allowed to discard that update.
                            try:
                                self._record_step(pipeline, report, groups, batch)
                            except OSError:
                                # Only the environment. A value that will not serialise, a
                                # key that is missing, a rollout with no samples -- those
                                # say a number or an invariant went wrong, and training on
                                # for days would produce a run whose own record the audit
                                # rejects.
                                logger.exception(
                                    "Could not record step %d; the update itself stands",
                                    self._step,
                                )
                            self._step += 1
                            steps_completed += 1
                        # Recorded. Collecting the next batch can take as long as a
                        # step; this one's rollouts must not be held through it.
                        del groups, batch
                finally:
                    self._pipeline = None
                    # However the run ends -- interrupted, aborted, failed, or a
                    # checkpoint that failed after its update was published -- the
                    # last applied update is kept if its cadence skipped it. Seen
                    # through: on Ctrl-C this task is already cancelling, and the save
                    # is bounded by its own timeout.
                    await see_through(asyncio.ensure_future(self._last_gasp(pipeline)))
        except _NothingToTrain:
            # Through the cleanup below, so a runtime that did not close is reported.
            pass
        except asyncio.CancelledError as error:
            status, reason = "interrupted", "run cancelled"
            self._interrupted(error)
            raise
        except (SystemicFailure, UnusableGroupsError) as error:
            if isinstance(error, LoopsRunEnded):
                status, reason = "loops_run_ended", error.ended
            else:
                status, reason = "aborted", str(error)
            self._record_failure(error, status)
            logger.error("%s", error)
            if status == "aborted":
                status, reason = await self._unless_ended(error, status, reason)
        except BaseException as error:
            status, reason = "failed", f"{type(error).__name__}: {error}"
            self._record_failure(error, status)
            if isinstance(error, Exception):
                status, reason = await self._unless_ended(error, status, reason)
            # A Loops run that ended under the run is the cause, and resumable: reported
            # as such, not raised.
            if status == "failed":
                raise
        finally:
            # Evals first: their samplers are deactivated through the trainer's client,
            # and their rollouts use the runtimes, gateway and resources closed below.
            try:
                await self._finish_evals(wait=status == "completed")
            except asyncio.CancelledError as error:
                # Stopped while a completed run waits for its final evals: recorded as
                # any other interruption.
                if status == "completed":
                    status, reason = "interrupted", "run cancelled"
                    self._interrupted(error)
                raise
            finally:
                # Whatever stops the eval wait, a Ctrl-C among them, the rest of
                # shutdown still runs.
                if pipeline is not None and not pipeline.settled:
                    self._unclean = True
                try:
                    try:
                        if runtime is not None:
                            await self.close_runtimes(runtime)
                    finally:
                        try:
                            # Shutdown step 5: stopping the gateway revokes every credential left.
                            if self._gateway is not None:
                                await self._gateway.close()
                                self._gateway = None
                        finally:
                            self._render_pool.close()
                            await self.close_resources()
                finally:
                    # Last: the run ends only once nothing else uses it.
                    await self.trainer.close()
        if self._unclean and status == "completed":
            status, reason = "unclean", "attempt or runtime cleanup is not confirmed"
        return Outcome(
            status,
            steps_completed,
            dict(pipeline.counters.discarded) if pipeline else {},
            reason,
            next_step=self._step,
        )


# Open files a run needs: a floor for the process itself, and an allowance for each
# attempt that can be live at once (its sandbox connections, trial logs, transcript and
# sample files). An estimate, not a measurement; on 1 Oct 2026 a run with ~120 live
# trials died at macOS's default soft limit of 256.
OPEN_FILES_FLOOR = 256
OPEN_FILES_PER_LIVE_ATTEMPT = 32


def check_open_file_limit(config: Config, *, training: bool = True, limits=None) -> None:
    """Refuse a run whose process may not open the files its live attempts need: its
    eval rollouts, and its training's when ``training`` (``loops eval`` trains none).

    Past the limit, sockets fail first: DNS lookups to the sampler, sandbox calls and
    journal writes all raise, minutes into a billed run. The limit is the caller's to
    raise (``ulimit -n``); this only says so before anything starts.
    """
    try:
        import resource
    except ImportError:  # not POSIX: no such limit to check
        return
    soft, _ = limits if limits is not None else resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY:
        return
    rollout = config.rollout
    # On dedicated Loops a set's cap spans all its evals, so evals add at most every
    # set's cap; on serverless they overlap, each with its own, so this is a floor. A
    # rollout frees its slot before its task's group grading, which takes one of the
    # eval's verifier slots: a set can have that many more live.
    live = sum(
        cap + min(cap, rollout.max_concurrent_verifiers)
        for cap in (eval_set.max_concurrent_rollouts for eval_set in config.eval.sets.values())
    )
    if training:
        live += (
            rollout.max_concurrent_rollouts
            + rollout.prestart_places
            + rollout.max_concurrent_verifiers
        )
    kinds = (
        "rollouts, pre-start places, verifiers and eval rollouts" if training else "eval rollouts"
    )
    needed = max(OPEN_FILES_FLOOR, OPEN_FILES_PER_LIVE_ATTEMPT * live)
    if soft < needed:
        raise RuntimeConfigurationError(
            f"this process may open {soft} files, and up to {live} attempts can be live at once "
            f"({kinds}), which needs about {needed}; raise the limit before starting, e.g. "
            f"`ulimit -n {max(needed, 65536)}`"
        )


def _warn_without_images(renderer: Renderer) -> None:
    """Probe image support before any Loops resource exists.

    A vision-capable model on a text-only task must keep working without the
    ``vision`` extra, so a failing probe is a warning; the first real image part
    still raises.
    """
    try:
        probe_image_tokens(renderer)
    except Exception as error:  # noqa: BLE001 - processor errors vary by backend
        logger.warning(
            "Image inputs are unavailable for this renderer (%s: %s). Text-only "
            "training is unaffected; run `uv sync --extra vision` if this job sends images.",
            type(error).__name__,
            first_sentence(str(error)),
        )
