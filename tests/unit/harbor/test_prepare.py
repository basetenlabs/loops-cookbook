"""What prepare() and loops check refuse or report about a Harbor job, before any trial."""

import pytest
import yaml
from support.configs import read_toml, write_toml

from loops_cookbook.harbor.runtime import (
    BLAXEL_ENVIRONMENT,
    CAPABILITIES,
    DAYTONA_ENVIRONMENT,
    AgentBindingError,
)
from loops_cookbook.model_access.api import ModelAccessMode
from loops_cookbook.runtime.api import (
    Phase,
)
from loops_cookbook.runtime.context import RuntimeConfigurationError

from .jobs import HILL_CLIMB, daytona_runtime, job, reported, runtime_for, shared_task, sized_task


def test_the_harbor_runtime_acquires_its_own_phases_and_never_retries_grading():
    assert CAPABILITIES.internally_acquired_phases == frozenset(
        {Phase.SAMPLING, Phase.VERIFICATION}
    )
    assert CAPABILITIES.grading_retries == 0  # regrades are the job's retry: block


def test_the_factory_accepts_only_reward_key_and_reward_gate_and_performs_no_io(tmp_path):
    missing = tmp_path / "missing.yaml"  # the job is not read
    runtime_for(missing, tmp_path, reward_key="score", reward_gate="harbor_test_gates:keep")
    with pytest.raises(ValueError, match="unknown Harbor runtime settings"):
        runtime_for(missing, tmp_path, harbor_config="job.yaml")
    with pytest.raises(ValueError, match="not callable"):
        runtime_for(missing, tmp_path, reward_gate="harbor_test_gates:NOT_CALLABLE")
    with pytest.raises(RuntimeConfigurationError, match="no reward gate"):
        runtime_for(missing, tmp_path, reward_gate="harbor_test_gates:absent")


async def test_prepare_states_the_jobs_regrades_as_the_jobs_own(tmp_path, caplog):
    task = shared_task(tmp_path)
    path = tmp_path / "job.yaml"
    body = yaml.safe_load(job(tmp_path, task, "AnswerAgent").read_text())
    body["retry"] = {"max_retries": 4}
    path.write_text(yaml.safe_dump(body))
    runtime = runtime_for(path, tmp_path / "work")
    with caplog.at_level("INFO", logger="loops_cookbook.harbor.runtime"):
        await runtime.prepare()
    assert "regrades a separate verifier up to 4 times (the job's retry: block)" in caplog.text
    capabilities = runtime.capabilities  # what check prints beside each count
    assert capabilities.grading_retries == 0
    assert capabilities.grading_retries_source == "4 (Harbor job retry)"
    assert capabilities.model_access_source == "direct (the job's agents' model_access)"


@pytest.mark.parametrize("field", [{"install_only": True}, {"verifier": {"disable": True}}])
async def test_prepare_rejects_templates_that_cannot_train(tmp_path, field):
    task = shared_task(tmp_path)
    path = tmp_path / "job.yaml"
    body = yaml.safe_load(job(tmp_path, task, "AnswerAgent").read_text())
    body.update(field)
    path.write_text(yaml.safe_dump(body))
    runtime = runtime_for(path, tmp_path / "work")
    with pytest.raises(ValueError, match="enabled Harbor verifier"):
        await runtime.prepare()


@pytest.mark.parametrize(
    ("agent", "problem"),
    [
        ("UnboundAgent", "must implement"),
        ("UndeclaredAgent", "model_access"),
        ("AsynchronousAgent", "synchronous"),
        ("OldNameAgent", "must declare model_access"),
    ],
    ids=["no setter", "no model access", "async setter", "old attribute"],
)
async def test_prepare_rejects_an_agent_it_cannot_bind_before_expanding_the_job(
    tmp_path, agent, problem
):
    runtime = runtime_for(job(tmp_path, shared_task(tmp_path), agent), tmp_path / "w")
    with pytest.raises(AgentBindingError, match=problem):
        await runtime.prepare()


async def test_prepare_keeps_develops_task_ids(tmp_path):
    runtime = runtime_for(job(tmp_path, HILL_CLIMB, "AnswerAgent"), tmp_path / "work")
    (task,) = await runtime.prepare()
    assert task.id == "0:hill_climb"  # develop's IDs: the index, then the directory name


async def test_model_access_is_computed_from_the_jobs_agents(tmp_path):
    runtime = runtime_for(job(tmp_path, shared_task(tmp_path), "HttpAgent"), tmp_path / "w")
    await runtime.prepare()
    assert runtime.capabilities.model_access_modes == frozenset({ModelAccessMode.OPENAI_HTTP})


async def test_agents_with_different_model_access_are_rejected(tmp_path):
    path = tmp_path / "mixed.yaml"
    body = yaml.safe_load(job(tmp_path, shared_task(tmp_path), "AnswerAgent").read_text())
    body["agents"].append({"import_path": "harbor_test_agents:HttpAgent"})
    path.write_text(yaml.safe_dump(body))
    with pytest.raises(ValueError, match="same model_access"):
        await runtime_for(path, tmp_path / "w").prepare()


async def test_per_agent_concurrency_is_rejected(tmp_path):
    path = job(tmp_path, shared_task(tmp_path), "AnswerAgent")
    raw = yaml.safe_load(path.read_text())
    raw["agents"][0]["n_concurrent"] = 1
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="n_concurrent"):
        await runtime_for(path, tmp_path / "w").prepare()


async def test_the_trial_cap_leaves_room_for_the_prestart_places(tmp_path):
    """Trials parked for a slow verifier must not use up the places started ahead."""
    runtime = runtime_for(
        job(tmp_path, shared_task(tmp_path), "AnswerAgent"),
        tmp_path / "work",
        limits={"prestart_places": 2},
    )
    await runtime.prepare()
    assert reported(runtime)["runtime/harbor/trial_cap"] == 4 + 2 + 4
    await runtime.close()


async def test_grading_preflight_rejects_a_deadline_under_harbors_own_bound(tmp_path):
    runtime = runtime_for(
        job(tmp_path, HILL_CLIMB, "AnswerAgent"),
        tmp_path / "work",
        # exactly build 600 + verifier 30: not enough
        limits={"grading_timeout_seconds": 630},
    )
    with pytest.raises(ValueError, match="must exceed Harbor's own bound.*630s"):
        await runtime.prepare()


@pytest.mark.parametrize(
    ("grading", "accepted"), [(3 * 630 + 1 + 2, False), (3 * 630 + 1 + 2 + 1, True)]
)
async def test_grading_preflight_counts_regrade_retries_and_backoff(tmp_path, grading, accepted):
    path = job(
        tmp_path,
        HILL_CLIMB,
        "AnswerAgent",
        retry={"max_retries": 2, "min_wait_sec": 1, "wait_multiplier": 2, "max_wait_sec": 60},
    )
    runtime = runtime_for(path, tmp_path / "work", limits={"grading_timeout_seconds": grading})
    if accepted:
        assert await runtime.prepare()
    else:
        with pytest.raises(ValueError, match="1893s"):
            await runtime.prepare()


@pytest.mark.parametrize(("grading", "accepted"), [(600, False), (601, True)])
async def test_grading_preflight_reads_the_final_steps_own_verifier_timeout(
    tmp_path, grading, accepted
):
    """Harbor gives each step a 600s verifier deadline whatever the task-level value."""
    task = shared_task(tmp_path, steps=2)  # task-level verifier timeout is 30s
    runtime = runtime_for(
        job(tmp_path, task, "AnswerAgent"),
        tmp_path / "work",
        limits={"grading_timeout_seconds": grading},
    )
    if accepted:
        assert await runtime.prepare()
    else:
        with pytest.raises(ValueError, match="600s"):
            await runtime.prepare()


@pytest.mark.parametrize(
    ("provider", "environment", "outcome"),
    [
        ("blaxel", {"type": "blaxel"}, "must set environment.import_path"),
        ("blaxel", {"import_path": "harbor.environments.blaxel:BlaxelEnvironment"}, "does not own"),
        ("blaxel", {"type": "blaxel", "import_path": BLAXEL_ENVIRONMENT}, "tracked"),
        ("blaxel", {"import_path": BLAXEL_ENVIRONMENT}, "tracked"),
        ("daytona", {"type": "daytona"}, "must set environment.import_path"),
        (
            "daytona",
            {"import_path": "harbor.environments.daytona:DaytonaEnvironment"},
            "does not own",
        ),
        ("daytona", {"type": "daytona", "import_path": DAYTONA_ENVIRONMENT}, "tracked"),
        ("daytona", {"import_path": DAYTONA_ENVIRONMENT}, "tracked"),
        (None, {"import_path": "baseline_environment:LocalEnvironment"}, None),
    ],
    ids=[
        "blaxel-type_only",
        "blaxel-harbors_own",
        "blaxel-type_and_ours",
        "blaxel-ours",
        "daytona-type_only",
        "daytona-harbors_own",
        "daytona-type_and_ours",
        "daytona-ours",
        "local",
    ],
)
async def test_a_provider_backed_job_must_use_the_cookbooks_environment_and_is_tracked(
    tmp_path, provider, environment, outcome
):
    """Harbor's own provider environments would create sandboxes no registry tracks; the
    cookbook's are tracked, so the job needs the run's provider table."""
    if provider is not None:
        pytest.importorskip(provider)
    if provider == "blaxel":
        environment = {**environment, "kwargs": {"ttl": "8h"}}
    runtime = runtime_for(
        job(tmp_path, HILL_CLIMB, "AnswerAgent", environment=environment), tmp_path / "work"
    )
    if outcome is None:
        assert await runtime.prepare()
        return
    tracked = rf"{provider.title()}-backed Harbor job needs \[providers.{provider}\]"
    with pytest.raises(ValueError, match=tracked if outcome == "tracked" else outcome):
        await runtime.prepare()


@pytest.mark.parametrize(
    ("create_seconds", "accepted"),
    [(60, True), (900, False), (None, False)],
    ids=["fits", "too_long", "no_blaxel_provider"],
)
async def test_a_tasks_build_timeout_must_cover_provisioning_and_attach(
    tmp_path, create_seconds, accepted
):
    pytest.importorskip("blaxel")
    from support.blaxel_api import SETTINGS

    deadlines = (
        {}
        if create_seconds is None
        else {
            "blaxel": SETTINGS.model_copy(
                update={"create_timeout_seconds": create_seconds}
            ).deadlines
        }
    )
    environment = {
        "import_path": BLAXEL_ENVIRONMENT,
        "kwargs": {"ttl": "8h"},
        "override_memory_mb": 8192,
    }
    runtime = runtime_for(
        job(tmp_path, HILL_CLIMB, "AnswerAgent", environment=environment),
        tmp_path / "work",
        provider_deadlines=deadlines,
    )
    if accepted:
        assert await runtime.prepare()
    else:
        with pytest.raises(
            ValueError, match="build timeout" if deadlines else r"\[providers.blaxel\]"
        ):
            await runtime.prepare()


@pytest.mark.parametrize(
    ("build", "raised", "accepted"),
    [
        (8.0, {}, True),  # prerequisites 1 + image build 3 + create 2 + attach 1 = 7 < 8
        (7.0, {}, False),  # equal is not enough: start() would time out at the edge
        (8.0, {"reconcile_timeout_seconds": 2}, False),
        (8.0, {"image_build_timeout_seconds": 4}, False),
        (8.0, {"create_timeout_seconds": 3}, False),
        (8.0, {"attach_timeout_seconds": 2}, False),
    ],
    ids=["exceeds", "equal", "prerequisites", "image_build", "create", "attach"],
)
async def test_the_build_bound_counts_every_provisioning_deadline(
    tmp_path, build, raised, accepted
):
    pytest.importorskip("blaxel")
    from support.blaxel_api import SETTINGS

    task = sized_task(tmp_path, cpus=1, memory_mb=256, build_timeout_sec=build)
    environment = {"import_path": BLAXEL_ENVIRONMENT, "kwargs": {"ttl": "8h"}}
    runtime = runtime_for(
        job(tmp_path, task, "AnswerAgent", environment=environment),
        tmp_path / "work",
        provider_deadlines={"blaxel": SETTINGS.model_copy(update=raised).deadlines},
    )
    if accepted:
        assert await runtime.prepare()
    else:
        with pytest.raises(ValueError, match="build timeout.*provisioning slot is not counted"):
            await runtime.prepare()


@pytest.mark.parametrize(("build", "accepted"), [(4.5, True), (4.0, False)])
async def test_daytonas_build_bound_has_no_separate_image_build(tmp_path, build, accepted):
    pytest.importorskip("daytona")
    from support.daytona_api import (  # reconcile 1 + create 2 + attach 1
        CREDENTIALS,
        SETTINGS,
        FakeApi,
    )

    from loops_cookbook.providers.daytona.adapter import DaytonaAdapter

    adapter = DaytonaAdapter(SETTINGS, CREDENTIALS, FakeApi())
    task = sized_task(tmp_path, cpus=1, memory_mb=1024, build_timeout_sec=build)
    runtime = runtime_for(
        job(tmp_path, task, "AnswerAgent", environment={"import_path": DAYTONA_ENVIRONMENT}),
        tmp_path / "work",
        provider_deadlines={"daytona": adapter.deadlines},
        provider_settings={"daytona": adapter.settings},
    )
    if accepted:
        assert await runtime.prepare()
    else:
        with pytest.raises(
            ValueError, match="Daytona reconcile .prerequisites., create, and attach"
        ):
            await runtime.prepare()


async def test_the_cookbooks_blaxel_environment_needs_a_ttl(tmp_path):
    pytest.importorskip("blaxel")
    runtime = runtime_for(
        job(
            tmp_path,
            HILL_CLIMB,
            "AnswerAgent",
            environment={"import_path": BLAXEL_ENVIRONMENT, "override_memory_mb": 8192},
        ),
        tmp_path / "work",
    )
    with pytest.raises(ValueError, match="ttl is required"):
        await runtime.prepare()


async def test_a_blaxel_job_may_not_state_the_region_providers_blaxel_states(tmp_path):
    pytest.importorskip("blaxel")
    environment = {"import_path": BLAXEL_ENVIRONMENT, "kwargs": {"ttl": "8h", "region": "us-was-1"}}
    runtime = runtime_for(
        job(tmp_path, HILL_CLIMB, "AnswerAgent", environment=environment), tmp_path / "work"
    )
    with pytest.raises(ValueError, match=r"stated once, as \[providers.blaxel\] region"):
        await runtime.prepare()


@pytest.mark.parametrize(
    ("task_sizes", "override", "sandbox", "warned"),
    [
        ({"memory_mb": 256}, None, 256, False),
        ({"memory_mb": 4096, "storage_mb": 1024}, None, 4096, False),
        ({"memory_mb": 4096, "storage_mb": 4096}, None, 4096, True),  # disk is 2048
        ({"memory_mb": 256, "storage_mb": 4096}, 8192, 8192, False),  # the job's override
        ({"storage_mb": 4096}, None, 4096, True),  # Harbor's default memory, disk 2048
    ],
    ids=["memory-only", "storage-fits", "storage-too-big", "override", "harbor-default"],
)
async def test_check_reports_each_tasks_blaxel_sandbox_and_what_cannot_fit(
    tmp_path, caplog, task_sizes, override, sandbox, warned
):
    pytest.importorskip("blaxel")
    from support.blaxel_api import SETTINGS

    environment = {"import_path": BLAXEL_ENVIRONMENT, "kwargs": {"ttl": "8h"}}
    if override is not None:
        environment["override_memory_mb"] = override
    runtime = runtime_for(
        job(tmp_path, sized_task(tmp_path, **task_sizes), "AnswerAgent", environment=environment),
        tmp_path / "work",
        provider_deadlines={"blaxel": SETTINGS.deadlines},
    )
    with caplog.at_level("INFO", logger="loops_cookbook.harbor.runtime"):
        assert await runtime.prepare()
    assert f"1 task(s) at {sandbox} MB" in caplog.text
    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "storage_mb" in r.message]
    assert bool(warnings) is warned
    if warned:
        assert (
            f"on {sandbox} MB of memory, which gives {sandbox // 2} MB of disk"
            in warnings[0].message
        )


@pytest.mark.parametrize(
    ("own", "refused"),
    [
        (None, False),
        ({"build": {"region": "us-was-1"}}, False),
        ({"build": {"region": "eu-lon-1"}}, True),
    ],
    ids=["no file", "the same region", "another region"],
)
async def test_check_refuses_a_task_whose_blaxel_toml_builds_in_another_region(
    tmp_path, own, refused
):
    pytest.importorskip("blaxel")
    from support.blaxel_api import SETTINGS

    task = sized_task(tmp_path, memory_mb=4096)
    if own is not None:
        write_toml(task / "environment" / "blaxel.toml", own)
    environment = {"import_path": BLAXEL_ENVIRONMENT, "kwargs": {"ttl": "8h"}}
    runtime = runtime_for(
        job(tmp_path, task, "AnswerAgent", environment=environment),
        tmp_path / "work",
        provider_deadlines={"blaxel": SETTINGS.deadlines},
        provider_regions={"blaxel": "us-was-1"},
    )
    if refused:
        with pytest.raises(ValueError, match="builds in region 'eu-lon-1'"):
            await runtime.prepare()
    else:
        assert await runtime.prepare()


@pytest.mark.parametrize(
    ("task_sizes", "override_cpus", "declared"),
    [
        ({"memory_mb": 4096}, None, None),
        ({"cpus": 4, "memory_mb": 4096}, None, 4),
        ({"memory_mb": 4096}, 8, 8),  # the job's override, as Harbor applies it
        ({"cpus": 4, "memory_mb": 4096}, 2, 2),
    ],
    ids=["none", "task", "job-override", "override-wins"],
)
async def test_check_says_blaxel_does_not_apply_a_declared_cpu_count(
    tmp_path, caplog, task_sizes, override_cpus, declared
):
    pytest.importorskip("blaxel")
    from support.blaxel_api import SETTINGS

    environment = {"import_path": BLAXEL_ENVIRONMENT, "kwargs": {"ttl": "8h"}}
    if override_cpus is not None:
        environment["override_cpus"] = override_cpus
    runtime = runtime_for(
        job(tmp_path, sized_task(tmp_path, **task_sizes), "AnswerAgent", environment=environment),
        tmp_path / "work",
        provider_deadlines={"blaxel": SETTINGS.deadlines},
    )
    with caplog.at_level("INFO", logger="loops_cookbook.harbor.runtime"):
        assert await runtime.prepare()
    warnings = [
        r.message for r in caplog.records if r.levelname == "WARNING" and "cpus" in r.message
    ]
    if declared is None:
        assert warnings == []
    else:
        assert warnings == [
            (
                "1 task(s) declare cpus, which Blaxel does not apply. A Blaxel sandbox's CPU "
                f"follows its memory instead. The groups are {declared} cpus on 4096 MB of "
                "memory, for 0:sized. Run with --verbose to name each task."
            )
        ]


async def test_check_says_once_what_many_tasks_declare_that_blaxel_cannot_give(tmp_path, caplog):
    pytest.importorskip("blaxel")
    from support.blaxel_api import SETTINGS

    sizes = [{"cpus": 1, "memory_mb": 4096, "storage_mb": 4096}] * 7 + [
        {"cpus": 2, "memory_mb": 8192, "storage_mb": 8192}
    ]
    tasks = [sized_task(tmp_path / str(i), **size) for i, size in enumerate(sizes)]
    path = job(
        tmp_path,
        tasks[0],
        "AnswerAgent",
        environment={"import_path": BLAXEL_ENVIRONMENT, "kwargs": {"ttl": "8h"}},
    )
    spec = yaml.safe_load(path.read_text())
    spec["tasks"] = [{"path": str(task)} for task in tasks]
    path.write_text(yaml.safe_dump(spec))
    runtime = runtime_for(
        path, tmp_path / "work", provider_deadlines={"blaxel": SETTINGS.deadlines}
    )
    with caplog.at_level("DEBUG", logger="loops_cookbook.harbor.runtime"):
        assert await runtime.prepare()
    detail = [r.message for r in caplog.records if r.levelname == "DEBUG" and "Blaxel" in r.message]
    assert sorted(detail) == sorted(
        [
            *(
                f"Task {i}:sized declares storage_mb 4096, but its Blaxel sandbox has 4096 MB of "
                "memory and so 2048 MB of disk"
                for i in range(7)
            ),
            *(
                f"Task {i}:sized declares 1 cpus; Blaxel does not apply it, and its sandbox's "
                "CPU follows its 4096 MB of memory"
                for i in range(7)
            ),
            (
                "Task 7:sized declares storage_mb 8192, but its Blaxel sandbox has 8192 MB of "
                "memory and so 4096 MB of disk"
            ),
            (
                "Task 7:sized declares 2 cpus; Blaxel does not apply it, and its sandbox's "
                "CPU follows its 8192 MB of memory"
            ),
        ]
    )
    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == [
        (
            "8 task(s) declare more storage_mb than their Blaxel sandbox's disk, which is half "
            "its memory. The groups are storage_mb 4096 on 4096 MB of memory, which gives "
            "2048 MB of disk, for 0:sized, 1:sized, 2:sized, 3:sized, 4:sized and 2 more; "
            "storage_mb 8192 on 8192 MB of memory, which gives 4096 MB of disk, for 7:sized. "
            "Run with --verbose to name each task."
        ),
        (
            "8 task(s) declare cpus, which Blaxel does not apply. A Blaxel sandbox's CPU "
            "follows its memory instead. The groups are 1 cpus on 4096 MB of memory, for "
            "0:sized, 1:sized, 2:sized, 3:sized, 4:sized and 2 more; 2 cpus on 8192 MB of "
            "memory, for 7:sized. Run with --verbose to name each task."
        ),
    ]


async def test_a_daytona_job_needs_providers_daytona(tmp_path):
    pytest.importorskip("daytona")
    environment = {"import_path": DAYTONA_ENVIRONMENT}
    runtime = runtime_for(
        job(tmp_path, HILL_CLIMB, "AnswerAgent", environment=environment), tmp_path / "work"
    )
    with pytest.raises(ValueError, match=r"Daytona-backed Harbor job needs \[providers.daytona\]"):
        await runtime.prepare()


@pytest.mark.parametrize(
    ("override", "verifier", "step_verifier", "table", "match"),
    [
        ({"override_cpus": 8}, None, None, {"max_sandbox_cpu": 4}, "8 vCPU > 4"),
        ({"override_memory_mb": 512}, None, None, {}, "at least 1"),
        (None, {"cpus": 8, "memory_mb": 4096}, None, {"max_sandbox_cpu": 4}, "8 vCPU > 4"),
        (None, {"cpus": 1, "memory_mb": 256}, None, {}, "at least 1"),
        (None, None, {"cpus": 8, "memory_mb": 4096}, {"max_sandbox_cpu": 4}, "8 vCPU > 4"),
        (None, None, {"cpus": 1, "memory_mb": 256}, {}, "at least 1"),
        (None, {"cpus": 0, "memory_mb": 1024}, None, {}, "at least 1"),
        (None, None, {"cpus": -1, "memory_mb": 1024}, {}, "at least 1"),
    ],
    ids=[
        "over_per_sandbox",
        "floored_to_zero",
        "verifier_over",
        "verifier_floored",
        "step_verifier_over",
        "step_verifier_floored",
        "verifier_no_cpu",
        "step_verifier_negative_cpu",
    ],
)
async def test_check_refuses_a_task_no_daytona_sandbox_can_hold(
    tmp_path, override, verifier, step_verifier, table, match
):
    """Every environment the task declares: the agent's, a separate verifier's, and
    each step verifier's."""
    pytest.importorskip("daytona")
    runtime = daytona_runtime(
        tmp_path, override=override, verifier=verifier, step_verifier=step_verifier, **table
    )
    with pytest.raises(ValueError, match=match):
        await runtime.prepare()


@pytest.mark.parametrize(
    ("override", "step_verifier"),
    [
        (None, {"cpus": 4, "memory_mb": 4096}),
        ({"override_cpus": 8, "cpu_enforcement_policy": "ignore"}, None),
    ],
    ids=["within_the_limit", "cpu_not_sent"],
)
async def test_check_passes_a_task_every_daytona_sandbox_can_hold(
    tmp_path, override, step_verifier
):
    """Sized as Harbor's create sizes it: a CPU Harbor does not send is the table's."""
    pytest.importorskip("daytona")
    runtime = daytona_runtime(
        tmp_path, override=override, step_verifier=step_verifier, max_sandbox_cpu=4
    )
    assert await runtime.prepare()


@pytest.mark.parametrize("key", ["auto_stop_interval_mins", "auto_delete_interval_mins"])
async def test_check_refuses_a_negative_job_lifecycle_interval(tmp_path, key):
    pytest.importorskip("daytona")
    runtime = daytona_runtime(tmp_path, override={"kwargs": {key: -1}})
    with pytest.raises(ValueError, match=f"environment.kwargs.{key} must be .* at least 0"):
        await runtime.prepare()


async def test_check_refuses_a_kwarg_for_harbors_own_daytona_client(tmp_path):
    """The environment refuses it at construction, which check's probe trial reaches."""
    pytest.importorskip("daytona")
    runtime = daytona_runtime(tmp_path, override={"kwargs": {"auto_snapshot": True}})
    with pytest.raises(ValueError, match="auto_snapshot"):
        await runtime.prepare()


def imported_task(root, docker_image):
    """A task whose task.toml names ``docker_image``."""
    task = shared_task(root)
    toml = task / "task.toml"
    tables = read_toml(toml)
    tables["environment"] = {"docker_image": docker_image}
    write_toml(toml, tables)
    return task


def blaxel_runtime(tmp_path, task, held=None, error=None, **options):
    """A Blaxel-backed runtime of ``task`` whose workspace holds ``held``, image to tags;
    the lookups it makes are recorded in ``asked``. ``options`` go to ``runtime_for``."""
    pytest.importorskip("blaxel")
    from support.blaxel_api import SETTINGS

    asked = []

    async def lookup(name):
        asked.append(name)
        if error is not None:
            raise error
        return (held or {}).get(name)

    environment = {"import_path": BLAXEL_ENVIRONMENT, "kwargs": {"ttl": "8h"}}
    runtime = runtime_for(
        job(tmp_path, task, "AnswerAgent", environment=environment),
        tmp_path / "work",
        provider_deadlines={"blaxel": SETTINGS.deadlines},
        provider_images={"blaxel": lookup},
        **options,
    )
    return runtime, asked


async def test_a_blaxel_image_the_workspace_holds_under_its_tag_is_accepted(tmp_path):
    task = imported_task(tmp_path, "blaxel://sandbox/my-image:v1")
    runtime, asked = blaxel_runtime(tmp_path, task, {"my-image": frozenset({"v1", "v0"})})
    assert await runtime.prepare()
    assert asked == ["my-image"]


@pytest.mark.parametrize(
    ("docker_image", "held", "refusal"),
    [
        ("blaxel://sandbox/my-image:v1", {}, "the workspace has no image my-image"),
        ("blaxel://sandbox/my-image:v1", {"my-image": frozenset({"v2"})}, "has tags v2"),
        ("blaxel://sandbox/my-image", {"my-image": frozenset({"v1"})}, "names no tag"),
    ],
    ids=["no_image", "wrong_tag", "no_tag"],
)
async def test_a_blaxel_image_the_workspace_does_not_hold_is_refused_before_the_run(
    tmp_path, docker_image, held, refusal
):
    runtime, asked = blaxel_runtime(tmp_path, imported_task(tmp_path, docker_image), held)
    with pytest.raises(ValueError, match=refusal):
        await runtime.prepare()
    assert len(asked) == (0 if refusal == "names no tag" else 1)


@pytest.mark.parametrize(
    ("docker_image", "error"),
    [
        ("blaxel://blaxel/base-image:latest", None),  # Blaxel has no lookup for this form
        ("blaxel://sandbox/my-image:v1", ConnectionError("unreachable")),
        ("blaxel://sandbox/my-image@sha256:" + "0" * 64, None),  # looked up by name, not digest
    ],
    ids=["other_form", "no_answer", "digest"],
)
async def test_a_blaxel_image_blaxel_cannot_answer_for_goes_unchecked_and_says_so(
    tmp_path, caplog, docker_image, error
):
    runtime, _ = blaxel_runtime(tmp_path, imported_task(tmp_path, docker_image), error=error)
    assert await runtime.prepare()
    assert "Blaxel images not checked" in caplog.text
    assert docker_image.removeprefix("blaxel://") in caplog.text


async def test_a_job_naming_no_blaxel_image_asks_blaxel_nothing(tmp_path):
    runtime, asked = blaxel_runtime(tmp_path, imported_task(tmp_path, "python:3.12"))
    assert await runtime.prepare()
    assert asked == []


async def test_a_blaxel_image_on_a_job_not_backed_by_blaxel_is_refused(tmp_path):
    """Harbor does not validate docker_image: Docker would only fail pulling it, per trial."""
    task = imported_task(tmp_path, "blaxel://sandbox/my-image:v1")
    runtime = runtime_for(job(tmp_path, task, "AnswerAgent"), tmp_path / "work")
    with pytest.raises(ValueError, match=r"only a Blaxel-backed job can start from"):
        await runtime.prepare()


async def test_a_job_not_backed_by_blaxel_names_every_task_naming_a_blaxel_image(tmp_path):
    tasks = [
        imported_task(tmp_path / name, f"blaxel://sandbox/{name}:v1") for name in ("one", "two")
    ]
    plain = imported_task(tmp_path / "plain", "python:3.12")
    path = job(
        tmp_path,
        tasks[0],
        "AnswerAgent",
        tasks=[{"path": str(task)} for task in (*tasks, plain)],
    )
    runtime = runtime_for(path, tmp_path / "work")
    with pytest.raises(ValueError) as refused:
        await runtime.prepare()
    message = str(refused.value)
    assert "blaxel://sandbox/one:v1" in message and "blaxel://sandbox/two:v1" in message
    assert "python:3.12" not in message


VERIFIER_IMAGE = "blaxel://sandbox/verifier-image:v1"


async def test_a_blaxel_image_on_a_separate_verifier_is_looked_up_too(tmp_path):
    task = sized_task(tmp_path, verifier={"docker_image": VERIFIER_IMAGE})
    runtime, asked = blaxel_runtime(tmp_path, task, {})
    with pytest.raises(ValueError, match="the workspace has no image verifier-image"):
        await runtime.prepare()
    assert asked == ["verifier-image"]


async def test_a_blaxel_image_on_a_separate_verifier_is_refused_off_blaxel(tmp_path):
    task = sized_task(tmp_path, verifier={"docker_image": VERIFIER_IMAGE})
    runtime = runtime_for(job(tmp_path, task, "AnswerAgent"), tmp_path / "work")
    with pytest.raises(ValueError, match=r"only a Blaxel-backed job") as refused:
        await runtime.prepare()
    assert f"(verifier): {VERIFIER_IMAGE}" in str(refused.value)


@pytest.mark.parametrize(
    ("verifier", "accepted"),
    [({"docker_image": VERIFIER_IMAGE}, True), ({"cpus": 1}, False)],
    ids=["nothing_builds", "the_verifier_builds"],
)
async def test_the_build_bound_counts_an_image_build_only_where_one_is_built(
    tmp_path, verifier, accepted
):
    """Prerequisites 1 + create 2 + attach 1 = 4 < 5; an image build of 3 makes it 7."""
    task = sized_task(
        tmp_path, verifier=verifier, docker_image=VERIFIER_IMAGE, build_timeout_sec=5.0
    )
    runtime, _ = blaxel_runtime(tmp_path, task, {"verifier-image": frozenset({"v1"})})
    if accepted:
        assert await runtime.prepare()
    else:
        with pytest.raises(ValueError, match="build timeout"):
            await runtime.prepare()


@pytest.mark.parametrize(
    ("verifier", "refused"),
    [({"docker_image": VERIFIER_IMAGE}, False), ({"cpus": 1}, True)],
    ids=["nothing_builds", "the_verifier_builds"],
)
async def test_a_blaxel_toml_in_another_region_is_refused_only_where_an_image_is_built(
    tmp_path, verifier, refused
):
    task = sized_task(tmp_path, verifier=verifier, docker_image=VERIFIER_IMAGE)
    write_toml(task / "environment" / "blaxel.toml", {"build": {"region": "eu-lon-1"}})
    runtime, _ = blaxel_runtime(
        tmp_path,
        task,
        {"verifier-image": frozenset({"v1"})},
        provider_regions={"blaxel": "us-was-1"},
    )
    if refused:
        with pytest.raises(ValueError, match="builds in region 'eu-lon-1'"):
            await runtime.prepare()
    else:
        assert await runtime.prepare()


async def test_one_image_named_under_several_tags_is_looked_up_once(tmp_path):
    """Its tags answer for every reference to it."""
    task = sized_task(
        tmp_path,
        verifier={"docker_image": "blaxel://sandbox/my-image:v2"},
        docker_image="blaxel://sandbox/my-image:v1",
    )
    runtime, asked = blaxel_runtime(tmp_path, task, {"my-image": frozenset({"v1", "v2"})})
    assert await runtime.prepare()
    assert asked == ["my-image"]


@pytest.mark.parametrize("backing", ["local", "blaxel"])
async def test_check_hashes_no_task(tmp_path, monkeypatch, backing):
    """A probe trial is read and never run, so it writes no lock and names no image.

    prepare() returns task IDs alone, so check never reads a task's whole content.
    """
    import harbor.environments.base as environment_base
    import harbor.trial.trial as trial_module

    def hashed(*args, **kwargs):
        raise AssertionError("check hashed a task")

    monkeypatch.setattr(trial_module, "build_trial_lock", hashed)
    monkeypatch.setattr(environment_base, "environment_content_hash", hashed)
    task = sized_task(tmp_path, cpus=1, memory_mb=256)
    if backing == "blaxel":
        runtime, _ = blaxel_runtime(tmp_path, task)
    else:
        runtime = runtime_for(job(tmp_path, task, "AnswerAgent"), tmp_path / "work")
    prepared = await runtime.prepare()
    assert [ref.id for ref in prepared] == ["0:sized"]
