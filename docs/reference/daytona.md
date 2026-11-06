# Daytona

The built-in Daytona provider: Daytona sandboxes as run-owned resources, for the
Harbor runtime and for an Episode's sandbox. Like Blaxel's, every sandbox is created
through the run's resource registry: its intent is journaled before Daytona is called,
its create holds a place in the provider's create window, it is released with its
attempt (or with the run), and a crashed run's sandboxes are found by
`loops cleanup`, from the journal or by label. See [resources.md](resources.md) for
the contract every provider meets.

```sh
uv sync --extra harbor-daytona   # Harbor on Daytona sandboxes
uv sync --extra daytona          # the provider alone, for a custom runtime
```

Put `DAYTONA_API_KEY` in `.env` (and `DAYTONA_API_URL` for a self-hosted Daytona). The
key needs `write:sandboxes` and `delete:sandboxes`. A `[providers.daytona]` table must
state the sandbox sizes and `ttl_minutes`; every other key has a default (see the
[knob glossary](KNOB_GLOSSARY.md#providersdaytona)):

```toml
[providers.daytona]
# target = "us"            # unset: the organization's default target
cpu = 1                    # sizes for a sandbox that names none, always sent
memory_gib = 1
disk_gib = 3
ttl_minutes = 1440         # wall-clock backstop, whatever it is doing
# thread_variables_from_vcpus = false  # true: OMP/OPENBLAS/MKL thread counts = vCPUs
# auto_stop_minutes = 120  # idle minutes before Daytona stops it; deleted as it stops
# labels = { owner = "my-team" }
# Optional per-sandbox limits, checked against every task before the run starts:
# max_sandbox_cpu = 4
# max_sandbox_memory_gib = 8
# max_sandbox_disk_gib = 10
```

## In a Harbor job

```yaml
environment:
  import_path: loops_cookbook.harbor.daytona:DaytonaEnvironment
  override_cpus: 1            # optional: one size for every task
  override_memory_mb: 4096
  override_storage_mb: 3072
  kwargs:
    labels: {owner: my-team}  # optional, besides the run's ownership labels
    # auto_stop_interval_mins: 60      # optional; else [providers.daytona]'s
```

The `import_path` is required: `type: daytona` on its own, or Harbor's own Daytona
class, is refused, because Harbor would create sandboxes no run owns. With the
cookbook's class, Harbor still chooses the image (a declarative build of the task's
Dockerfile, or a prebuilt `docker_image`), sizes the sandbox, applies the network
policy, and runs every command and file transfer. Its create and delete change: each
is one registry operation.

Harbor's own Daytona environment talks to Daytona through a process-wide client it
builds from `DAYTONA_*` in the environment. The cookbook's never builds it: the
sandbox comes from the provider's one client, configured from `[providers.daytona]`,
so `target` holds for every call, and a `DAYTONA_TARGET` in the environment that
differs from it is refused. The kwargs that act only through Harbor's client are
refused, by `loops check` too: `auto_snapshot` (Harbor names the snapshot after
`DAYTONA_TARGET` and may build it with a second client of its own),
`snapshot_template_name` (Harbor looks the snapshot up with its client), and
`connection_pool_maxsize` (it configures that client).

What Harbor's Daytona environment does on its own, and what the cookbook adds:

| | Harbor 0.23's Daytona environment | With the cookbook's |
| --- | --- | --- |
| Create | Retried by tenacity: 3 tries, or 10 for a rate limit or capacity error, waiting 60 s × the try; a timed-out or 5xx create is sent again | Sent once; a refusal (429, a request that never left) is sent again within the create deadline, from the provider's create window; an ambiguous one is looked up by name, never sent twice |
| Name | None: Daytona names it by its id | `loops-<resource id>`, unique in the organization, so a create is idempotent |
| Labels | `harbor.*` and the job's `labels` | Those, `[providers.daytona] labels`, and the run's ownership labels: `loops.run_id`, `loops.scope`, `loops.owner_id` (the attempt), `loops.resource_id` |
| Lifecycle | Never stops a sandbox (`auto_stop_interval_mins` 0); deleted as it stops | Stopped after `auto_stop_minutes` idle, deleted as it stops, destroyed at `ttl_minutes`, unless the job says otherwise; a lifecycle that would never end one is refused |
| Delete | On `stop(delete=True)`, two tries; a failure is logged and the trial carries on | A release, proven by a 404 for the sandbox's id; one not proven is raised from `stop()` and kept for the attempt's cleanup |
| Kept sandbox | `delete: false` leaves it running | Released when its attempt ends |
| Crash | Sandboxes run until something deletes them | Released by the resumed run, or `loops cleanup` (journal, or labels) |
| Client | A process-wide client built from `DAYTONA_*` in the environment | The provider's own, from `[providers.daytona]` |
| Sizes | CPU, memory, disk, and GPUs (with the task's `gpu_types`) from the task or the job's overrides | CPU, memory and disk the same; what Harbor sends none for comes from `[providers.daytona]`, so every create names all three, and `check` refuses one over `max_sandbox_*`. A GPU task is refused |
| Threads | `nproc` is the host's core count | The same by default. With `thread_variables_from_vcpus = true`, every command gets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` set to the sandbox's vCPUs, over the image's `ENV`; the task's `[environment.env]` wins |

A trial's OOM-kill count (`runtime/harbor/sandbox_oom_kills`) is the sandbox's
cgroup's `oom_kill`, from cgroup v2's `/sys/fs/cgroup/memory.events`; without v2 it
is not reported. A Daytona sandbox shares its host's kernel, so `/proc/vmstat`
counts every OOM kill on the host, tens of thousands, not the sandbox's.

Harbor sends the task's environment with every command, so the sandbox is created
without it; that also keeps resolved secrets out of the run's journal. Only Harbor's
direct mode is supported, not compose (Docker-in-Docker).

### Sizing

Daytona sizes CPU, memory, and disk independently, in whole vCPUs and GiB, at least 1
each. A size Harbor does not send (the task states none, or the job's
`cpu_enforcement_policy` or `memory_enforcement_policy` is `ignore`) is
`[providers.daytona]`'s `cpu`, `memory_gib` and `disk_gib`, which are required. They are dedicated:
cgroups enforce the CPU and memory exactly, with no swap. Disk is the writable layer only;
the image is not counted. Harbor gives a task's `memory_mb` and `storage_mb` to
Daytona in whole GiB, rounding down, so `check` refuses one under 1024 MB, and a task
with fewer than 1 `cpus` (Harbor accepts 0 or less). Only a
running sandbox counts against the organization's CPU and memory; a stopped one holds
its disk until it is deleted, which here is as it stops.

`loops check` refuses a task with an environment over a `max_sandbox_*` limit,
which Daytona would refuse at every create with a 400. It checks every environment the
task declares: the agent's, a separate verifier's (`[verifier.environment]`), and each
step's separate verifier's, each sized by Harbor's environment as its create would size
it. Daytona's limits are per organization; set `max_sandbox_*` to yours.

### Deadlines

Harbor bounds each start by the task's effective build timeout, which must cover the
provider's reconcile, create, and attach deadlines together; Daytona builds a
declarative image inside its create, so there is no separate image-build deadline, and
`create_timeout_seconds` (600 s) covers a cold build. With the defaults that is 690 s,
so a job keeping Harbor's 600 s build timeout needs
`environment_build_timeout_multiplier` of at least 1.2. The rest of a start, Harbor's
post-create setup, has no provider deadline and is not counted, so leave a little
margin past 1.2.

### Network

Daytona's network tier, an organization setting, decides a sandbox's egress: on some
tiers egress is restricted and a per-sandbox allowlist is refused. The cookbook
does not probe it, since that needs a sandbox; a task that needs the network fails at
its first command that does. Check the tier before a run whose tasks install packages
or call APIs, or run one trial first.

## For an Episode

```toml
[runtime.config.sandbox]
provider = "daytona"
image = "python:3.12-slim"   # unset: Daytona's default sandbox
cpu = 1
memory_gib = 2
```

Or `snapshot = "<name>"` in place of `image`: the sandbox starts from that Daytona
snapshot, which fixes its image and size, so it takes no `cpu`, `memory_gib` or
`disk_gib`. With neither set, the sandbox is Daytona's default, which fixes its own size too.

`ctx.sandbox` is a `loops_cookbook.providers.daytona.Daytona`; its `client()` is the
SDK's `AsyncSandbox`, and `read_file(path)` returns a file's text, or `None` when the
sandbox is alive without it. An agent process reaches the sandbox through its
`usage_environment()`: `DAYTONA_API_KEY` and `DAYTONA_SANDBOX_ID`. The key is
organization-wide, so run only agent programs you would trust with it.

## Cleanup

```sh
uv run loops cleanup train.toml --state-dir runs/<run>/state   # from the journal
uv run loops cleanup train.toml --run-id <run>                 # journal lost: by labels
```

Daytona filters a listing by label itself, so a sweep reads only the run's sandboxes.
A sandbox is released by its id, which Daytona never reuses, and is proven gone by a
404 for that id or a `destroyed` state. A deleted sandbox reads `destroying` for about
a second; Daytona renames it, freeing its name at once.

## Choosing a provider

| | Blaxel | Daytona |
| --- | --- | --- |
| Sizing | Memory alone: one vCPU per 2048 MB, disk a tmpfs of half the memory | CPU, memory, and disk independently, in whole vCPUs and GiB |
| A task's `cpus` and `storage_mb` | Not applied (`check` warns) | Applied |
| Image | Built once by a builder sandbox, named by its content, then reused by later runs and never deleted | Built inside the create and cached by content |
| Region | Required, `[providers.blaxel] region` | Optional, `[providers.daytona] target` |
| Leak backstop | The job's required `ttl` | Idle auto-stop, delete on stop, and a wall-clock `ttl_minutes` |
| Rate limit | A workspace-wide `429` on bursts of creates | `429` past the organization's create rate; an organization limit fails the create |
| Label sweep | Reads the whole workspace | Filtered by Daytona |

Choose Daytona when tasks need CPU or disk out of proportion to memory; Blaxel when tasks are memory-bound and the workspace is already
sized for them.
