"""``loops cleanup``: recovery from a crashed run's journal, or from its labels alone."""

import asyncio

import pytest
from support.configs import MINIMAL, write_toml
from support.resources import (
    FACTORY_CALLS,
    FakeProvider,
    episode,
    local,
    registry_at,
    request,
    run_resources,
)

from loops_cookbook import command
from loops_cookbook.providers import create_adapters
from loops_cookbook.resources.api import (
    CleanupStatus,
    ProviderDeadlines,
    ProviderResource,
    ProvisioningFailure,
)
from loops_cookbook.resources.cleanup import reconcile_journal, sweep_labels
from loops_cookbook.resources.journal import JOURNAL, JournalError, JournalLocked
from loops_cookbook.runtime.context import RuntimeConfigurationError


def unlistable(provider):
    """``provider``, with a listing that fails before its first page."""

    async def fails(labels):
        raise ConnectionError("list unavailable")
        yield  # an async generator, as list_owned is

    provider.list_owned = fails
    return provider


async def crashed_run(state_dir, provider):
    """A journal a killed run left: released, owned with a dependant, ambiguous, unconfirmed."""
    registry = registry_at(state_dir, provider)
    run = run_resources(registry)
    attempt = episode(registry)
    released = await attempt.provision(request("released"))
    await attempt.release(released.intent.resource_id)
    base = await run.provision(request("base"))
    child = await attempt.provision(request("child", depends_on=(base.intent.resource_id,)))
    provider.hide_absence = True
    provider.create_errors.append(ConnectionError("reset"))
    with pytest.raises(ProvisioningFailure):
        await attempt.provision(request("ambiguous"))
    stuck = await attempt.provision(request("stuck"))
    provider.release_errors.append(ConnectionError("reset"))
    await attempt.release(stuck.intent.resource_id)
    provider.hide_absence = False
    registry.journal.close()  # the process dies here
    return base, child


@pytest.mark.parametrize("created", [True, False], ids=["created_one", "created_nothing_yet"])
async def test_cleanup_refuses_a_live_run_and_closes_its_adapters(tmp_path, created):
    """A live run holds the journal's lock, whether or not it has created anything yet."""
    registry = registry_at(tmp_path, FakeProvider())
    if created:
        await run_resources(registry).provision(request())
    cleaner = FakeProvider()
    try:
        with pytest.raises(JournalLocked):
            await reconcile_journal(tmp_path, {"fake": cleaner}, kind=local)
    finally:
        await registry.close()
    assert cleaner.calls == [] and cleaner.closed == 1


async def test_cleanup_releases_a_crashed_run_in_dependency_order_and_reruns_as_a_noop(tmp_path):
    provider = FakeProvider()
    base, child = await crashed_run(tmp_path, provider)
    provider.calls.clear()
    report = await reconcile_journal(tmp_path, {"fake": provider}, kind=local)
    assert report.complete
    releases = [subject for operation, subject in provider.calls if operation == "release"]
    assert releases.index("fake-child") < releases.index("fake-base")
    assert provider.store == {}
    statuses = {result.resource_id: result.status for result in report.results}
    assert statuses[base.intent.resource_id] is CleanupStatus.RELEASED
    assert statuses[child.intent.resource_id] is CleanupStatus.RELEASED
    assert CleanupStatus.ALREADY_ABSENT in statuses.values()  # the ambiguous create never landed

    provider.calls.clear()
    again = await reconcile_journal(tmp_path, {"fake": provider}, kind=local)
    assert again.complete and again.results == ()
    assert provider.calls == []


async def test_an_incomplete_cleanup_names_every_unresolved_identity_and_can_rerun(tmp_path):
    provider = FakeProvider()
    await crashed_run(tmp_path, provider)
    provider.hide_absence = True
    provider.release_errors.extend([ConnectionError("reset")] * 10)
    report = await reconcile_journal(tmp_path, {"fake": provider}, kind=local)
    assert not report.complete
    unresolved = {result.resource_id for result in report.unresolved}
    assert len(unresolved) == 4  # base, child, ambiguous, stuck
    provider.hide_absence = False
    provider.release_errors.clear()
    assert (await reconcile_journal(tmp_path, {"fake": provider}, kind=local)).complete
    assert provider.store == {}


@pytest.mark.parametrize("damage", [b'{"torn": ', b"garbage\n"], ids=["torn_tail", "garbage"])
async def test_a_damaged_journal_is_cleaned_from_what_it_replayed_and_never_complete(
    tmp_path, damage
):
    provider = FakeProvider()
    await crashed_run(tmp_path, provider)
    path = tmp_path / JOURNAL
    with path.open("ab") as stream:
        stream.write(damage)
    for _ in range(2):  # a rerun cannot make it complete either
        report = await reconcile_journal(tmp_path, {"fake": provider}, kind=local)
        assert not report.complete
        assert [result.resource_id for result in report.unresolved] == [str(path)]
    assert provider.store == {} and path.exists()  # everything replayed is released


async def test_cleanup_from_a_journal_makes_at_most_its_bound_of_provider_calls(tmp_path):
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    attempt = episode(registry)
    for index in range(20):
        await attempt.provision(request(f"s{index}"))
    registry.journal.close()  # the process dies here
    provider.delay = 0.01
    provider.most_in_flight = 0
    report = await reconcile_journal(tmp_path, {"fake": provider}, kind=local, concurrency=4)
    assert report.complete and provider.store == {}
    assert provider.most_in_flight == 4


async def test_a_cancelled_journal_cleanup_closes_its_adapters_only_once_calls_settle(tmp_path):
    provider = SlowToStop()
    registry = registry_at(tmp_path, provider)
    attempt = episode(registry)
    for index in range(3):
        await attempt.provision(request(f"s{index}"))
    registry.journal.close()  # the process dies here
    provider.blocked.add("release")
    cleanup = asyncio.create_task(
        reconcile_journal(tmp_path, {"fake": provider}, kind=local, concurrency=1)
    )
    while not any(operation == "release" for operation, _ in provider.calls):
        await asyncio.sleep(0)
    cleanup.cancel()
    provider.unblocked.set()  # the registry sees its releases through, cancelled or not
    with pytest.raises(asyncio.CancelledError):
        await cleanup
    assert provider.closed == 1 and provider.closed_under == []


@pytest.mark.parametrize("concurrency", [0, -1])
async def test_cleanup_from_a_journal_refuses_a_bound_that_releases_nothing(tmp_path, concurrency):
    provider = FakeProvider()
    await crashed_run(tmp_path, provider)
    provider.calls.clear()
    with pytest.raises(ValueError, match="concurrency"):
        await reconcile_journal(tmp_path, {"fake": provider}, kind=local, concurrency=concurrency)
    assert provider.calls == [] and provider.closed == 1


async def test_cleanup_refuses_a_state_directory_with_no_journal(tmp_path):
    empty = tmp_path / "typo"
    with pytest.raises(JournalError, match="no resource journal"):
        await reconcile_journal(empty, {"fake": FakeProvider()}, kind=local)
    assert not empty.exists()  # nothing was created there


async def test_a_run_that_created_nothing_cleans_up_complete(tmp_path):
    """It opened its journal and stopped before recording any intent, say before sampling."""
    provider = FakeProvider()
    registry = registry_at(tmp_path, provider)
    await registry.close()
    assert not (tmp_path / JOURNAL).exists()  # a journal is written with its first intent
    report = await reconcile_journal(tmp_path, {"fake": provider}, kind=local)
    assert report.complete and report.results == ()
    assert provider.calls == []


async def test_a_resumed_run_releases_what_its_previous_process_left(tmp_path):
    provider = FakeProvider()
    await crashed_run(tmp_path, provider)
    resumed = registry_at(tmp_path, provider)
    report = await resumed.release_all()
    assert report.complete
    assert provider.store == {}
    await run_resources(resumed).provision(request("fresh"))  # and provisions as usual
    await resumed.close()


# ------------------------------------------------------------- label sweep


def owned(provider, name, resource_id, *, run_id="run-1", scope="episode"):
    labels = {
        "loops.run_id": run_id,
        "loops.scope": scope,
        "loops.owner_id": "attempt-1",
        "loops.resource_id": resource_id,
    }
    resource = ProviderResource(provider_id=f"id-{name}", name=name, labels=labels)
    provider.store[resource.provider_id] = resource
    return resource


async def test_a_label_sweep_releases_what_it_lists_and_is_safe_to_rerun():
    provider = FakeProvider()
    owned(provider, "a", "r-a")
    owned(provider, "b", "r-b", scope="run")
    owned(provider, "other-run", "r-c", run_id="run-2")
    report = await sweep_labels({"fake": provider}, "run-1")
    statuses = {result.resource_id: result.status for result in report.results}
    assert statuses == {
        "fake:listing": CleanupStatus.UNCONFIRMED,  # the listing itself proves nothing
        "r-a": CleanupStatus.RELEASED,
        "r-b": CleanupStatus.RELEASED,
    }
    assert list(provider.store) == ["id-other-run"]
    assert provider.closed == 1
    again = await sweep_labels({"fake": provider}, "run-1")
    assert [result.resource_id for result in again.results] == ["fake:listing"]


async def test_a_label_sweep_releases_many_at_once_within_its_bound():
    provider = FakeProvider(delay=0.05)
    for index in range(40):
        owned(provider, f"s{index}", f"r-{index}")
    report = await sweep_labels({"fake": provider}, "run-1", concurrency=8)
    assert all(r.status is CleanupStatus.RELEASED for r in report.results[1:])
    assert provider.store == {}
    assert provider.most_in_flight == 8


async def test_a_label_sweep_finishes_each_resource_before_starting_another():
    """A sweep stopped midway has proven gone every resource it released."""
    provider = FakeProvider(delay=0.01)
    for index in range(4):
        owned(provider, f"s{index}", f"r-{index}")
    await sweep_labels({"fake": provider}, "run-1", concurrency=1)
    released = [call for call in provider.calls if call[0] != "list"]
    assert released == [(op, f"s{i}") for i in range(4) for op in ("release", "confirm_absent")]


async def test_a_label_sweep_releases_episode_resources_before_run_ones():
    provider = FakeProvider(delay=0.01)
    owned(provider, "base", "r-base", scope="run")  # listed first, released last
    for index in range(6):
        owned(provider, f"e{index}", f"r-e{index}")
    await sweep_labels({"fake": provider}, "run-1", concurrency=4)
    released = [subject for operation, subject in provider.calls if operation == "release"]
    assert released[-1] == "base" and sorted(released[:-1]) == [f"e{i}" for i in range(6)]


class SlowToStop(FakeProvider):
    """A release that, cancelled, takes a moment to wind down, as a client closing its
    connection does."""

    async def release(self, resource):
        try:
            await super().release(resource)
        except asyncio.CancelledError:
            self.busy += 1
            try:
                await asyncio.sleep(0.01)
            finally:
                self.busy -= 1
            raise


async def test_a_cancelled_label_sweep_closes_its_adapters_only_once_calls_settle():
    provider = SlowToStop(blocked={"release"})
    for index in range(3):
        owned(provider, f"s{index}", f"r-{index}")
    sweep = asyncio.create_task(sweep_labels({"fake": provider}, "run-1", concurrency=1))
    while ("release", "s0") not in provider.calls:
        await asyncio.sleep(0)
    sweep.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sweep
    assert provider.closed == 1 and provider.closed_under == []


@pytest.mark.parametrize("concurrency", [0, -1])
async def test_a_label_sweep_refuses_a_bound_that_releases_nothing(concurrency):
    provider = FakeProvider()
    owned(provider, "a", "r-a")
    async with asyncio.timeout(1):
        with pytest.raises(ValueError, match="concurrency"):
            await sweep_labels({"fake": provider}, "run-1", concurrency=concurrency)
    assert provider.calls == [] and provider.closed == 1


async def test_an_empty_label_sweep_is_never_complete():
    report = await sweep_labels({"fake": FakeProvider()}, "run-x")
    assert not report.complete
    [listing] = report.unresolved
    assert listing.resource_id == "fake:listing" and "0 resource(s) listed" in (
        listing.detail or ""
    )


async def test_a_label_sweep_reports_unproven_absence_and_foreign_ownership_as_unconfirmed():
    provider = FakeProvider(hide_absence=True)
    owned(provider, "a", "r-a")
    forged = owned(provider, "b", "r-b")
    provider.store[forged.provider_id] = ProviderResource(
        provider_id=forged.provider_id,
        name="b",
        labels={**forged.labels, "loops.scope": "elsewhere"},
    )
    provider.release_errors.append(ConnectionError("reset"))
    report = await sweep_labels({"fake": provider}, "run-1")
    assert {r.resource_id: r.status for r in report.results} == {
        "fake:listing": CleanupStatus.UNCONFIRMED,
        "r-a": CleanupStatus.UNCONFIRMED,
        "r-b": CleanupStatus.UNCONFIRMED,
    }
    assert "id-b" in provider.store  # never released without validated ownership


async def test_a_listing_is_bounded_per_page_so_a_large_one_completes():
    """Every page answers inside the bound; together they take longer than it."""
    provider = FakeProvider(deadlines=ProviderDeadlines(1, 1, 1, 1, 0.2, 1), list_page_size=1)
    for index in range(5):
        owned(provider, f"s{index}", f"r-{index}")
    provider.list_delay = 0.1  # 5 pages x 0.1 s, against a 0.2 s bound
    report = await sweep_labels({"fake": provider}, "run-1")
    assert {r.resource_id: r.status for r in report.results} == {
        "fake:listing": CleanupStatus.UNCONFIRMED,
        **{f"r-{index}": CleanupStatus.RELEASED for index in range(5)},
    }
    assert provider.store == {}


async def test_a_page_slower_than_the_bound_fails_the_listing_and_releases_nothing():
    provider = FakeProvider(deadlines=ProviderDeadlines(1, 1, 1, 1, 0.1, 1), list_page_size=1)
    owned(provider, "a", "r-a")
    owned(provider, "b", "r-b")
    provider.list_delay = 0.3
    [result] = (await sweep_labels({"fake": provider}, "run-1")).results
    assert result.resource_id == "fake:*" and result.status is CleanupStatus.UNCONFIRMED
    assert "TimeoutError" in (result.detail or "")
    assert sorted(provider.store) == ["id-a", "id-b"]


async def test_a_failed_listing_is_unconfirmed_not_empty():
    provider = unlistable(FakeProvider())
    report = await sweep_labels({"fake": provider}, "run-1")
    [result] = report.results
    assert result.resource_id == "fake:*" and result.status is CleanupStatus.UNCONFIRMED


# ----------------------------------------------------------------- the CLI


def config(tmp_path, instance, *tables):
    fake = {"adapter": "support.resources:create_fake", "instance": instance}
    return write_toml(tmp_path / "train.toml", {"providers": {"fake": fake}}, *tables)


FAKE = "support.resources:create_fake"


async def test_a_custom_adapter_factory_receives_its_settings_and_nothing_else(registered):
    provider = registered("factory")
    adapters = await create_adapters(
        {"fake": {"adapter": FAKE, "instance": "factory", "zone": "a"}}
    )
    assert adapters == {"fake": provider}
    assert FACTORY_CALLS[-1] == {"instance": "factory", "zone": "a"}


@pytest.mark.parametrize(
    ("broken", "error"),
    [
        ({}, "not a built-in provider"),
        ({"adapter": FAKE, "instance": "undeclared"}, "ProviderDeadlines"),
    ],
    ids=["no_factory", "no_deadlines"],
)
async def test_a_provider_that_cannot_be_built_fails_and_closes_those_already_built(
    registered, broken, error
):
    built = registered("built")
    registered("undeclared", FakeProvider(deadlines=None))
    with pytest.raises(RuntimeConfigurationError, match=error):
        await create_adapters({"a": {"adapter": FAKE, "instance": "built"}, "b": broken})
    assert built.closed == 1


@pytest.mark.parametrize(
    "path",
    [
        "no_colon",
        "support.missing_module:create",
        "support.resources:missing",
        "support.resources:FAST",
    ],
    ids=["malformed", "no_module", "no_attribute", "not_callable"],
)
async def test_an_adapter_factory_that_cannot_be_imported_names_its_provider(path):
    """The provider's table is where the fix goes, so the error names it, not a runtime."""
    with pytest.raises(RuntimeConfigurationError) as raised:
        await create_adapters({"fake": {"adapter": path}})
    assert "[providers.fake] adapter" in str(raised.value)
    assert "runtime" not in str(raised.value)


def test_cli_cleanup_uses_a_registered_provider_and_exits_by_completeness(
    tmp_path, capsys, registered
):
    provider = registered("cli")
    state = tmp_path / "state"
    asyncio.run(crashed_run(state, provider))
    path = config(tmp_path, "cli")
    provider.hide_absence = True
    provider.release_errors.extend([ConnectionError("reset")] * 10)
    assert command.main(["cleanup", str(path), "--state-dir", str(state)]) == 1
    assert "not proven released" in capsys.readouterr().err
    provider.hide_absence = False
    provider.release_errors.clear()
    assert command.main(["cleanup", str(path), "--state-dir", str(state)]) == 0
    assert "Cleanup complete" in capsys.readouterr().out
    assert provider.store == {}
    assert command.main(["cleanup", str(path), "--state-dir", str(state)]) == 0


def test_cli_cleanup_by_run_id_sweeps_labels(tmp_path, capsys, registered):
    provider = registered("sweep")
    owned(provider, "a", "r-a")
    path = config(tmp_path, "sweep")
    # A sweep releases what it lists but can never prove the run has nothing left.
    assert command.main(["cleanup", str(path), "--run-id", "run-1"]) == 3
    output = capsys.readouterr()
    assert "r-a released" in output.out and "Released everything listed" in output.err
    assert provider.store == {}


def test_cli_sweep_exits_1_while_a_listed_resource_or_a_listing_is_unresolved(tmp_path, registered):
    provider = registered("sweep-stuck")
    owned(provider, "a", "r-a")
    provider.stubborn.add("a")
    path = config(tmp_path, "sweep-stuck")
    assert command.main(["cleanup", str(path), "--run-id", "run-1"]) == 1
    assert "id-a" in provider.store

    unlistable(registered("sweep-down"))
    path = config(tmp_path, "sweep-down")
    assert command.main(["cleanup", str(path), "--run-id", "run-1"]) == 1


def test_cli_cleanup_exits_without_mutation_while_the_journal_is_locked(tmp_path, registered):
    provider = registered("locked")
    state = tmp_path / "state"
    registry = registry_at(state, provider)
    path = config(tmp_path, "locked")
    with pytest.raises(SystemExit) as exit:
        command.main(["cleanup", str(path), "--state-dir", str(state)])
    assert exit.value.code == 2
    assert provider.calls == []
    assert not (state / JOURNAL).exists()  # nothing was written
    registry.journal.close()


async def test_a_label_sweep_whose_listing_hangs_is_bounded_and_unconfirmed():
    provider = FakeProvider(deadlines=ProviderDeadlines(1, 1, 1, 1, 0.05, 1), blocked={"list"})
    [result] = (await sweep_labels({"fake": provider}, "run-1")).results
    assert result.status is CleanupStatus.UNCONFIRMED and "TimeoutError" in (result.detail or "")


@pytest.mark.parametrize("name", ["check", "run"])
def test_adapters_are_closed_when_the_run_cannot_be_constructed(tmp_path, registered, name):
    """The adapters are built before the run; a run that never exists cannot close them."""
    provider = registered(f"unbuilt-{name}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")  # not a checkpoint manifest, so Run() refuses it
    path = config(tmp_path, f"unbuilt-{name}", MINIMAL, {"training": {"resume_from": manifest}})
    with pytest.raises(SystemExit) as exited:
        command.main([name, str(path)])
    assert exited.value.code == 2
    assert provider.closed == 1


def test_adapters_are_closed_when_cleanup_finds_no_journal(tmp_path, registered):
    provider = registered("no-journal")
    path = config(tmp_path, "no-journal")
    with pytest.raises(SystemExit):
        command.main(["cleanup", str(path), "--state-dir", str(tmp_path / "empty")])
    assert provider.closed == 1


async def test_a_sweep_closes_every_adapter_even_when_one_fails_to_close(monkeypatch):
    failing, other = FakeProvider(), FakeProvider()

    async def refuses():
        raise ConnectionError("reset")

    monkeypatch.setattr(failing, "close", refuses)
    with pytest.raises(ConnectionError):
        await sweep_labels({"a": failing, "b": other}, "run-1")
    assert other.closed == 1


async def test_an_adapter_whose_listing_does_not_page_is_refused_when_it_is_created(registered):
    class Unpaged(FakeProvider):
        async def list_owned(self, labels):  # a list, as before listings paged
            return []

    registered("unpaged", Unpaged())
    with pytest.raises(RuntimeConfigurationError, match="list_owned must be an async generator"):
        await create_adapters({"fake": {"adapter": FAKE, "instance": "unpaged"}})
