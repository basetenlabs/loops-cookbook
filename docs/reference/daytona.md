# Daytona

Daytona is one of the cookbook's two built-in sandbox providers, beside Blaxel. The run
owns every Daytona sandbox it creates, both for the Harbor rollout runtime and for the
sandbox of an `Episode` in a custom rollout runtime.

As with Blaxel, the run creates every sandbox through its resource registry:

- The registry writes the sandbox's creation intent to the run's journal before it
  calls Daytona.
- The create holds a place in the provider's create window, which limits how many
  creates are in flight at once.
- The run releases the sandbox with its attempt, or with the run.
- After a crash, `loops cleanup` finds the run's sandboxes from the journal or by their
  labels.

[resources.md](resources.md) describes the contract that every provider meets.

## Choosing a provider

The table compares Daytona with Blaxel.

| | Blaxel | Daytona |
| --- | --- | --- |
| Sizing | Memory sets everything. A sandbox gets one vCPU per 2048 MB, and its disk is a tmpfs of half the memory. | CPU, memory and disk are set independently, in whole vCPUs and GiB. |
| A task's `cpus` and `storage_mb` | They aren't applied, and `loops check` warns about them. | They are applied. |
| Image | A builder sandbox builds it once and names it by its content. Later runs reuse it, and it is never deleted. A Harbor task can instead name an image already in Blaxel as `blaxel://<image>`, and then nothing is built. | Daytona builds it inside the create and caches it by content. |
| Region | It is required, as `[providers.blaxel].region`. | It is optional, as `[providers.daytona].target`. |
| What ends a leaked sandbox | The job's required `ttl` does. | An idle auto-stop, a delete on stop and a wall-clock `ttl_minutes` do. |
| Rate limit | Blaxel answers bursts of creates with a workspace-wide `429`. | Daytona answers `429` past the organization's create rate. An organization limit fails the create. |
| Label sweep | A sweep reads the whole workspace. | Daytona filters the listing by label. |

Choose Daytona when your tasks need CPU or disk out of proportion to their memory.
Choose Blaxel when your tasks are memory-bound and your workspace is already sized for
them.

## Set up the provider

Install the extra you need:

```sh
uv sync --extra harbor-daytona   # Harbor on Daytona sandboxes
uv sync --extra daytona          # the provider alone, for a custom rollout runtime
```

Put `DAYTONA_API_KEY` in `.env`. For a self-hosted Daytona, also set
`DAYTONA_API_URL`. The key needs `write:sandboxes` and `delete:sandboxes`.

A `[providers.daytona]` table must state the default sandbox sizes, `cpu`, `memory_gib`
and `disk_gib`, and the wall-clock limit, `ttl_minutes`. A wrong size or limit would
fail tasks or kill live work without saying why, so the cookbook doesn't guess them.
Every other key has a default, which the
[knob glossary](KNOB_GLOSSARY.md#providersdaytona) lists:

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

Harbor calls a sandbox an *environment*. To run a Harbor job's sandboxes on Daytona,
name the cookbook's `DaytonaEnvironment` class in the job's `environment:` block:

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

The `import_path` is required. `loops check` refuses `type: daytona` on its own, and it
refuses Harbor's own Daytona class, because with either one Harbor would create
sandboxes that no run owns.

The cookbook's class leaves most of the work to Harbor. Harbor still chooses the
image, which is either a declarative build of the task's Dockerfile or a prebuilt
`docker_image`. Harbor also sizes the sandbox, applies the network policy, and runs
every command and file transfer. Only the create and the delete change. Each one
becomes a single operation of the run's resource registry.

Harbor's own Daytona environment reaches Daytona through a process-wide client, which
it builds from the `DAYTONA_*` environment variables. The cookbook's class never builds
that client. Its sandboxes come from the provider's single client instead, which is
configured from `[providers.daytona]`. So `[providers.daytona].target` holds for every
call, and the run refuses a `DAYTONA_TARGET` environment variable that differs from it.

The run refuses the job kwargs that act only through Harbor's client, and so does
`loops check`:

- It refuses `auto_snapshot`, because Harbor names the snapshot after `DAYTONA_TARGET`
  and may build it with a second client of its own.
- It refuses `snapshot_template_name`, because Harbor looks the snapshot up with its
  client.
- It refuses `connection_pool_maxsize`, because that kwarg configures Harbor's client.

The table shows what Harbor's Daytona environment does on its own, and what the
cookbook's class changes.

| | Harbor 0.23's Daytona environment | With the cookbook's |
| --- | --- | --- |
| Create | Tenacity retries the create. It makes 3 tries, or 10 for a rate-limit or capacity error, and waits 60 s times the try number between them. A create that timed out or got a 5xx is sent again. | The create is sent once. A refusal, which is a 429 or a request that never left, is sent again within the create deadline, through the provider's create window. An ambiguous create is looked up by name and never sent twice. |
| Name | Harbor sends no name, so Daytona names the sandbox by its ID. | The name is `loops-<resource id>`, which is unique in the organization, so a create is idempotent. |
| Labels | The sandbox carries `harbor.*` labels and the job's `labels`. | The sandbox carries those labels, `[providers.daytona].labels`, and the run's ownership labels. The ownership labels are `loops.run_id`, `loops.scope`, `loops.owner_id` and `loops.resource_id`. `loops.owner_id` names the attempt. |
| Lifecycle | Harbor never stops a sandbox, because its `auto_stop_interval_mins` is 0. A sandbox is deleted as it stops. | Unless the job says otherwise, a sandbox is stopped after `auto_stop_minutes` idle, deleted as it stops, and destroyed at `ttl_minutes`. A lifecycle that would never end a sandbox is refused. |
| Delete | On `stop(delete=True)`, Harbor tries the delete twice. A failure is logged, and the trial carries on. | The delete is a release, which is proven by a 404 for the sandbox's ID. A release that isn't proven is raised from `stop()`, and the sandbox is kept for the attempt's cleanup. |
| Kept sandbox | With `delete: false`, the sandbox keeps running. | The sandbox is released when its attempt ends. |
| Crash | Sandboxes keep running until something deletes them. | The resumed run releases them, or `loops cleanup` does, from the journal or by labels. |
| Client | Harbor uses a process-wide client, which it builds from the `DAYTONA_*` environment variables. | The run uses the provider's own client, configured from `[providers.daytona]`. |
| Sizes | CPU, memory, disk and GPUs come from the task or from the job's overrides. GPUs use the task's `gpu_types`. | CPU, memory and disk come from the same place. When Harbor sends no value for one of them, it comes from `[providers.daytona]`, so every create names all three. `loops check` refuses a sandbox over `max_sandbox_*`. A GPU task is refused. |
| Threads | `nproc` in the sandbox returns the host's core count. | The same happens by default. With `thread_variables_from_vcpus = true`, every command gets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` set to the sandbox's vCPUs. These override the image's `ENV`, but the task's `[environment.env]` wins. |

A Daytona sandbox shares its host's kernel, so `/proc/vmstat` counts every
out-of-memory (OOM) kill on the host, tens of thousands of them, not the sandbox's
own. A trial's OOM-kill count, `runtime/harbor/sandbox_oom_kills`, is therefore the
`oom_kill` count of the sandbox's cgroup, from cgroup v2's
`/sys/fs/cgroup/memory.events`. Without cgroup v2, the count isn't reported.

Harbor sends the task's environment variables with every command, so the run creates
the sandbox without them. That also keeps resolved secrets out of the run's journal.

The cookbook supports only Harbor's direct mode. It doesn't support Harbor's compose
mode, which runs Docker-in-Docker.

### Sizing

Daytona sizes CPU, memory and disk independently, in whole vCPUs and GiB, with at
least 1 of each. Harbor sends no size when the task states none, or when the job's
`cpu_enforcement_policy` or `memory_enforcement_policy` is `ignore`. A size that
Harbor doesn't send comes from `cpu`, `memory_gib` or `disk_gib` in
`[providers.daytona]`, which you must set.

The sizes are dedicated. Cgroups enforce the CPU and memory exactly, with no swap. The
disk size counts only the writable layer, not the image.

Harbor passes a task's `memory_mb` and `storage_mb` to Daytona in whole GiB, rounding
down. So `loops check` refuses a task with less than 1024 MB of either. It also refuses
a task with fewer than 1 `cpus`, although Harbor itself accepts 0 or less.

Only a running sandbox counts against the organization's CPU and memory. A stopped
sandbox holds its disk until it is deleted, and here that happens as it stops.

`loops check` also refuses a task whose sandbox is over a `max_sandbox_*` limit,
because Daytona would refuse every create for it with a 400. It checks every sandbox
the task declares, and it sizes each one with Harbor's environment class, the same way
the sandbox's create would be sized. The sandboxes it checks are these:

- the agent's sandbox,
- a separate verifier's sandbox, from `[verifier.environment]`,
- each step's separate verifier's sandbox.

Daytona's limits are per organization, so set `max_sandbox_*` to your organization's
limits.

### Deadlines

Harbor bounds each sandbox start by the task's effective build timeout. `loops check`
refuses a task whose effective build timeout doesn't cover the provider's reconcile,
create and attach deadlines together. Daytona builds a declarative image inside its
create, so there is no separate image-build deadline. `create_timeout_seconds`, which
is 600 s by default, covers a cold build.

With the default deadlines, the three add up to 690 s. So a job that keeps Harbor's
600 s build timeout needs an `environment_build_timeout_multiplier` of at least 1.2.
The rest of a start is Harbor's post-create setup, which has no provider deadline and
isn't counted, so leave a little margin past 1.2.

### Network

Daytona's network tier is an organization setting, and it decides a sandbox's
outbound network access. On some tiers, outbound access is restricted, and a
per-sandbox allowlist is refused. The cookbook doesn't probe the tier, because that
needs a sandbox. A task that needs the network fails at its first command that uses
it. Before a run whose tasks install packages or call APIs, check the tier, or run one
trial first.

## For an Episode

An `Episode` in a custom rollout runtime can run each attempt in a Daytona sandbox.
Configure the sandbox in `[rollout_runtime.config.sandbox]`:

```toml
[rollout_runtime.config.sandbox]
provider = "daytona"
image = "python:3.12-slim"   # unset: Daytona's default sandbox
cpu = 1
memory_gib = 2
```

The sizes you leave out, here `disk_gib`, come from `[providers.daytona]`.

To start the sandbox from a Daytona snapshot, set `snapshot = "<name>"` in place of
`image`. A snapshot fixes the sandbox's image and size, so a snapshot sandbox takes no
`cpu`, `memory_gib` or `disk_gib`. With neither `image` nor `snapshot` set, the sandbox
is Daytona's default sandbox, which also fixes its own size.

In the episode, `sample.sandbox` is a `loops_cookbook.providers.daytona.Daytona`. Its
`client()` returns the SDK's `AsyncSandbox`. Its `write_file(path, content)` writes
UTF-8 text. Its `read_file(path)` returns a file's text, or `None` when the sandbox is
alive but the file isn't there.

An agent process reaches the sandbox through the sandbox's usage environment,
`sample.sandbox.environment`, which the provider's `usage_environment()` builds. It
holds `DAYTONA_API_KEY` and `DAYTONA_SANDBOX_ID`. It also holds `DAYTONA_API_URL` when
your environment sets it, and `DAYTONA_TARGET` when `[providers.daytona].target` is
set. The key is organization-wide, so run only agent
programs that you would trust with it.

## Cleanup

After a crash, `loops cleanup` releases the run's sandboxes. It works from the run's
journal, or by labels when the journal is lost:

```sh
uv run loops cleanup train.toml --state-dir runs/<run>/state   # from the journal
uv run loops cleanup train.toml --run-id <run>                 # journal lost: by labels
```

Daytona filters a listing by label itself, so a label sweep reads only the run's
sandboxes. The run releases a sandbox by its ID, which Daytona never reuses. A sandbox
is proven gone when Daytona answers 404 for that ID, or reports the sandbox as
`destroyed`. A deleted sandbox reads `destroying` for about a second. Daytona renames
it, which frees its name at once. [resources.md](resources.md#cleaning-up-after-a-crash)
explains both commands and their exit codes.
