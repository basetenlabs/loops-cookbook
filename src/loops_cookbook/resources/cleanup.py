"""The two ways to clean up after a training process dies, behind ``loops cleanup``.

With the run's state directory, ``reconcile_journal`` opens the journal in
reconcile mode and releases everything it still holds. When the journal is gone,
``sweep_labels`` finds the run's resources by their ownership labels instead. Both
release through the registry's one rule, ``release_resource``, and return a
``CleanupReport`` naming every identity whose absence is unproven.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

from loops_cookbook.resources.api import (
    CleanupReport,
    CleanupResult,
    CleanupStatus,
    ProviderResource,
    ResourceAdapter,
    ResourceIntent,
    ResourceRequest,
    ResourceScope,
)
from loops_cookbook.resources.journal import JOURNAL, JournalError, filesystem_type
from loops_cookbook.resources.registry import (
    OWNERSHIP_LABELS,
    ResourceRegistry,
    close_adapters,
    describe,
    release_resource,
    validate,
)

# The identity of a sweep's entry for a listing that succeeded but proves nothing
# about what it did not return. A listing that failed is ``<provider>:*`` instead.
LISTING = ":listing"
# How much a cleanup does at once: provider calls from the journal, resources by label.
# A run can leave hundreds of resources, and each release (validate, release, prove
# absent) takes seconds; an unbounded burst draws rate limits instead.
CLEANUP_CONCURRENCY = 32


def only_listings_unconfirmed(report: CleanupReport) -> bool:
    """Whether a sweep released everything it listed, leaving only its listings unproven."""
    unresolved = report.unresolved
    return bool(unresolved) and all(r.resource_id.endswith(LISTING) for r in unresolved)


async def reconcile_journal(
    state_dir: Path,
    adapters: Mapping[str, ResourceAdapter],
    *,
    kind: Callable[[Path], str] = filesystem_type,
    concurrency: int = CLEANUP_CONCURRENCY,
) -> CleanupReport:
    """Reconcile and release everything a run's journal holds, in dependency order,
    at most ``concurrency`` provider calls at a time across every provider.

    Fails before any provider call when ``state_dir`` is not a run's state directory,
    or while another process holds its journal. A run that opened its journal and
    recorded nothing created nothing, and its cleanup is complete. A damaged journal is cleaned from what it replayed,
    but whatever followed the damage is unknown, so its report is never complete.
    """
    journal = state_dir / JOURNAL
    try:
        _check_concurrency(concurrency)
        # A run writes its journal with its first intent, before any provider call, so
        # a run that opened the journal (its lock exists) and wrote none created
        # nothing: opening it takes the lock, which refuses a live run, and replays
        # nothing.
        if not journal.exists() and not journal.with_suffix(".lock").exists():
            raise JournalError(
                f"{state_dir} holds no resource journal ({JOURNAL}); pass the run's state "
                "directory as its unresolved-resource report names it"
            )
        permits = asyncio.Semaphore(concurrency)

        @contextlib.asynccontextmanager
        async def permit() -> AsyncIterator[None]:
            async with permits:
                yield

        registry = ResourceRegistry.open(
            state_dir, adapters, mode="reconcile", permit=permit, kind=kind
        )
    except BaseException:
        # No registry took the adapters, so nothing else will close them.
        await close_adapters(adapters)
        raise
    try:
        report = await registry.release_all()
        damage = registry.journal.damage
        if damage is None:
            return report
        unreadable = CleanupResult(
            str(registry.journal.path),
            CleanupStatus.UNCONFIRMED,
            f"{damage}; resources recorded after it are unknown. Once they are accounted "
            "for, move the journal aside by hand",
        )
        return CleanupReport(report.scope, report.owner_id, (*report.results, unreadable))
    finally:
        await registry.close()


async def sweep_labels(
    adapters: Mapping[str, ResourceAdapter], run_id: str, *, concurrency: int = CLEANUP_CONCURRENCY
) -> CleanupReport:
    """Release every resource each provider lists under ``run_id``'s ownership labels.

    Listing is eventually consistent, so this proves nothing about what it did not
    list, so each provider's report carries an ``UNCONFIRMED`` entry for the listing
    itself; a failed list, an ownership that does not validate, and unproven absence
    also end ``UNCONFIRMED``. Safe to rerun, but only once the run has stopped: it
    takes no lock and cannot tell a live run's resources from a dead one's. Each page
    of a listing is bounded by the provider's ``list_seconds``; the whole listing is
    not, since it grows with the provider's account rather than the run. A provider's
    resources are released ``concurrency`` at a time, each validated, released and
    proven absent before its slot is free, episode-scoped ones first, as a run's own
    release orders them.
    """
    results: list[CleanupResult] = []
    try:
        _check_concurrency(concurrency)
        for name, adapter in sorted(adapters.items()):
            results.extend(await _sweep(name, adapter, run_id, concurrency))
    finally:
        await close_adapters(adapters)
    return CleanupReport(None, None, tuple(results))


def _check_concurrency(concurrency: int) -> None:
    if concurrency < 1:  # no permit would ever be free, so the cleanup would hang
        raise ValueError(f"a cleanup's concurrency must be at least 1, not {concurrency}")


async def _list_pages(
    pages: AsyncGenerator[Sequence[ProviderResource], None], page_seconds: float
) -> list[ProviderResource]:
    """Every page of a listing, each bounded by ``page_seconds``. The total is not
    bounded: it grows with the provider's account, which the run does not size."""
    listed: list[ProviderResource] = []
    try:
        while True:
            async with asyncio.timeout(page_seconds):
                page = await anext(pages, None)
            if page is None:
                return listed
            listed.extend(page)
    finally:
        await pages.aclose()


async def _sweep(
    name: str, adapter: ResourceAdapter, run_id: str, concurrency: int
) -> list[CleanupResult]:
    try:
        listed = await _list_pages(
            adapter.list_owned({"loops.run_id": run_id}), adapter.deadlines.list_seconds
        )
    except (TimeoutError, Exception) as error:  # noqa: BLE001 - reported as unconfirmed
        return [
            CleanupResult(
                f"{name}:*",
                CleanupStatus.UNCONFIRMED,
                f"list failed: {describe(error, adapter.deadlines.list_seconds)}",
            )
        ]
    # A listing is eventually consistent: what it did not return is not proven gone.
    results = [
        CleanupResult(
            f"{name}{LISTING}",
            CleanupStatus.UNCONFIRMED,
            f"{len(listed)} resource(s) listed; a label listing cannot prove that no other "
            "resource of the run exists",
        )
    ]
    slots = asyncio.Semaphore(concurrency)
    # Episode-scoped resources first, each layer whole before the next, as a run releases.
    episodes = [r for r in listed if r.labels.get("loops.scope") == "episode"]
    others = [r for r in listed if r.labels.get("loops.scope") != "episode"]
    for layer in (episodes, others):
        # Every release settles before any failure or cancellation is raised, so the
        # adapter is never closed under a call still winding down.
        outcomes = await asyncio.gather(
            *(_release_listed(name, adapter, run_id, r, slots) for r in layer),
            return_exceptions=True,
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
            results.append(outcome)
    return results


async def _release_listed(
    name: str,
    adapter: ResourceAdapter,
    run_id: str,
    resource: ProviderResource,
    slots: asyncio.Semaphore,
) -> CleanupResult:
    """Release one listed resource once its ownership validates, holding a slot
    throughout, so a sweep stopped midway has finished the resources it started."""
    identity = resource.labels.get("loops.resource_id") or f"{name}:{resource.provider_id}"
    intent = _listed_intent(name, run_id, resource)
    async with slots:
        if intent is None or not await validate(adapter, intent, resource):
            return CleanupResult(
                identity,
                CleanupStatus.UNCONFIRMED,
                f"{resource.name}: ownership did not validate; left in place",
            )
        status, detail = await release_resource(adapter, resource)
    if status is CleanupStatus.RELEASED:
        return CleanupResult(identity, status, detail)
    return CleanupResult(identity, CleanupStatus.UNCONFIRMED, detail)


def _listed_intent(provider: str, run_id: str, resource: ProviderResource) -> ResourceIntent | None:
    """The ownership identity a listed resource's labels claim, for validation."""
    labels = resource.labels
    if any(not labels.get(label) for label in OWNERSHIP_LABELS):
        return None
    if labels["loops.run_id"] != run_id or labels["loops.scope"] not in ("episode", "run"):
        return None
    return ResourceIntent(
        resource_id=labels["loops.resource_id"],
        scope=ResourceScope(labels["loops.scope"]),
        owner_id=labels["loops.owner_id"],
        request=ResourceRequest(provider=provider, kind="listed", logical_key=resource.name),
        specification_digest="",
        desired_name=resource.name,
        ownership_labels=MappingProxyType({label: labels[label] for label in OWNERSHIP_LABELS}),
    )
