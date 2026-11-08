# Resources and agent processes

This page covers two services that the cookbook gives each attempt. Provider resources
are the sandboxes and other billed things that a rollout runtime asks a
provider for. Agent processes are agent programs that the cookbook runs outside its own
process.

## Provider resources

A provider resource is anything a provider bills for, such as a sandbox or a volume.
A rollout runtime never asks the provider for one directly. It asks the run's
resource registry instead, through one of two services:

- `attempt.services.resources` provides resources that one attempt owns.
- `context.run_resources` provides resources that the whole run shares.

A request names a provider, a kind, a `logical_key` and a JSON specification. The
`logical_key` must be unique within the resource's owner, which is the attempt or the
run. The registry returns a handle, and it owns everything after that.

### Configuring a provider

Each provider has its own `[providers.<name>]` table in the training config. Blaxel
and Daytona are built in. A provider that isn't built in names its factory with
`adapter = "module:callable"`. The run calls the factory with the table's other keys,
and the factory returns an adapter that declares a finite deadline for every
operation.

No table holds a credential. Each provider reads its own credential from the
environment, which the run loads from `.env`. So a missing credential fails before the
run or `loops cleanup` calls the provider.

[daytona.md](daytona.md) describes Daytona's table and its Harbor environment.
Blaxel's table needs only the region. Every deadline has a default, which the
[knob glossary](KNOB_GLOSSARY.md#providersblaxel) lists, and a deadline you set in the
table replaces its default. Blaxel's credentials are `BL_WORKSPACE` and `BL_API_KEY`,
from the environment or `.env`:

```toml
[providers.blaxel]
region = "us-was-1"
# attach_timeout_seconds = 45   # only to change a default
```

When Blaxel already has a sandbox's image, the run adopts it. Otherwise the run builds
the image once for every sandbox in the run that needs it. The builder sandbox is
deleted once the build finishes, and the image itself is never deleted. A Harbor task
can also name an image that is already in Blaxel as `blaxel://<image>`. The run then
builds nothing, as [harbor.md](harbor.md#starting-from-an-image-already-in-blaxel)
explains.

### How the registry creates and releases a resource

The registry takes every resource through these steps:

1. **It records the intent.** Before it calls the provider, the registry writes an
   immutable creation intent to the run's resource journal and fsyncs it. The intent
   carries the exact name that the resource will be created with, and the ownership
   labels it will carry: `loops.run_id`, `loops.scope`, `loops.owner_id` and
   `loops.resource_id`.
2. **It sends the create.** The create runs under the provisioning limit and the
   provider's create deadline. The provisioning limit,
   `[rollout].max_concurrent_provisioning`, bounds how many provider operations run at
   once. A create that the provider turns away for now, such as
   Blaxel's `429`, is sent again within that deadline. The wait before each retry
   starts at 1 s and doubles up to 30 s, with jitter. A retry is sent only while half
   of the deadline is left. [Create windows](#create-windows) explains how many creates
   are in flight at once.
3. **It records the outcome.** A create can fail in a way that may still have created
   something. The registry then looks the resource up by its name and labels. The
   resource counts as absent only on authoritative evidence. An empty or failed lookup
   leaves the outcome unknown, and never counts as absent.
4. **It releases the resource.** The registry defers a release while a lease or a
   dependent resource still needs the resource. A lease is how an attempt borrows one
   of the run's resources. A release is complete only once the provider proves the
   resource gone. A release runs to its deadline even if the hook that asked for it is
   cancelled.

Some resources depend on a transient helper, such as the image build that a Blaxel
sandbox waits for. When a helper fails to create, every request for the same key fails
at once with that failure for the next 90 s (`FAILED_HELPER_COOLDOWN_SECONDS`). Without
that, each queued request would build the helper again.

Unrelated resources provision concurrently, and operations on one resource run one at
a time. A caller that waits for another caller's prerequisite holds none of the
provisioning limit.

### Create windows

The registry limits how many of one provider's creates are in flight at once, which
means sent and not yet answered. That limit is the provider's create window, and it
behaves like TCP's congestion window:

- The window starts at `[rollout].max_concurrent_provisioning`, which is also its cap.
- A refusal halves the window. A burst of creates that is refused together halves it
  only once.
- Each create that the provider answers without refusing widens the window a little,
  so the window grows by about one for each window's worth of such answers, back to
  its cap. A failure or a timeout counts as an answer that wasn't a refusal.
- A fractional size rounds up to the next whole slot.

Every create waits for a slot, whether it is a first send or a retry. It holds none of
the provisioning limit while it waits. Creates get slots in the order they first asked
for one, so a retry keeps its place ahead of newer creates. Image builds share their
provider's window. A wait before a create's first send costs none of its deadline.

The run logs one INFO line when a window first narrows, and one when it is back at its
cap, with the number of refusals in between.

### When an attempt ends

The run releases an attempt's resources when the attempt ends, whatever its hooks did.
The exception is a hook that ignores cancellation. Its attempt stays live, and its
resources are left for `loops cleanup`.

A resource that can't be proven gone fails the attempt's cleanup. The registry keeps
it, by its exact identity, for recovery.

Time spent in the resource services counts against neither the setup deadline nor the
cleanup deadline.

### The journal, resuming and shutdown

The journal lives at
`<run directory>/<[resources] state_dir>/resources/v1/journal.jsonl`. It must be on a
local filesystem, so the run refuses NFS, SMB and FUSE. Only one process writes the
journal at a time.

A resumed run reuses its run directory, its `run_id` and its journal. Before it admits
an attempt, it releases everything its previous process left.

The run reports whatever it can't prove released. It reports a previous process's
leftovers at startup, and its own at shutdown. A run that would otherwise complete
then ends `unclean`.

At shutdown, no create that hasn't been sent yet starts. The run's releases and the
creates already sent share one time limit, `[rollout].shutdown_drain_seconds`. Whatever
is still in flight when it passes is cut off and reported for `loops cleanup`.

A damaged journal refuses to start a run. Its cleanup is never reported complete. Once
its resources are accounted for, you have to move the journal aside by hand.

### Cleaning up after a crash

After a crash, `loops cleanup` finishes the job. It works from the journal, or by
labels when the journal is lost:

```sh
uv run loops cleanup train.toml --state-dir runs/<run>/state   # from the journal
uv run loops cleanup train.toml --run-id <run>                 # journal lost: by labels
```

With `--state-dir`, the command replays the journal and releases what the journal
holds, in dependency order. It refuses to run while the run still holds the journal.

With `--run-id`, the command sweeps by labels. It finds the run's resources by their
ownership labels. Run it only once the run has stopped. Provider listings lag, so a
sweep never proves that nothing is left, and it always reports its listing as
`unconfirmed`. A sweep reads the provider's listing one page at a time, and
`list_timeout_seconds` bounds each page, not the whole listing. Blaxel doesn't filter
by label, so a sweep reads every sandbox in the workspace and takes as long as the
workspace is large.

Both forms read only the config's `[providers]` tables, print every identity, and are
safe to rerun. Their exit codes are:

| Exit code | From the journal | By labels |
| --- | --- | --- |
| 0 | Every resource is proven released. | The config has no `[providers]` tables to sweep. A sweep never exits 0 otherwise, because the listing itself can't be proven complete. |
| 1 | Some resource isn't proven released. | A listed resource is unresolved, or a listing failed. |
| 2 | It means a configuration error, or that another process holds the journal. | It means a configuration error. |
| 3 | The journal cleanup doesn't use this code. | Everything the sweep listed is released. |

### Sandboxes for an Episode

An adapter that can back an `Episode`'s sandbox is also a `SandboxProvider`, which
`loops_cookbook.resources` exports. A `SandboxProvider` has four members:

- `sandbox_kind` is the sandbox kind it provides.
- `client(resource)` returns the provider's own client for the resource, within the
  attach deadline.
- `close_client(client)` closes that client.
- `alive(resource)` says whether the resource is still usable.

The provider ships its sandbox kind for Episodes to name, such as
`class Blaxel(SandboxKind[SandboxInstance]): provider = "blaxel"`. The kind types the
sandbox's client, and its `spec` types `[rollout_runtime.config.sandbox]`. For
Blaxel, the spec has three keys, which sit flat beside `provider`. All three are
required:

- `image` is the image the sandbox runs.
- `memory_mb` is the sandbox's memory.
- `ttl` is how long Blaxel keeps the sandbox. It is a whole number and one unit, `s`,
  `m`, `h`, `d` or `w`, as Blaxel takes it.

The spec refuses any other key, and it refuses a value of another type.

Blaxel's adapter is a `SandboxProvider`. `loops check` refuses an Episode sandbox from
an adapter that isn't one, or whose `sandbox_kind` isn't the episode's kind or a
subclass of it. The match is by type, so you can give a provider any name in
`[providers]`.

## Agent processes

Some agents are programs that the cookbook doesn't own, such as a program in another
language, another interpreter or another virtual environment. Such an agent runs
through `attempt.services.processes`:

```python
result = await attempt.services.processes.run(
    ProcessCommand(
        argv=("python3", "/abs/path/agent.py"),
        cwd=results_dir,
        env={"OPENAI_BASE_URL": model.base_url, "OPENAI_API_KEY": model.api_key, **usage},
    )
)
```

The process starts in its own process group, in `cwd`, with standard input closed. Its
environment is a short allowlist of non-secret variables, such as `PATH`, `HOME`,
`LANG` and `LC_*`, plus exactly `command.env`. Nothing else in the cookbook's
environment is inherited. A `SecretStr` in `env` reaches the process only through that
environment, and the cookbook redacts it from the output it keeps.

`run()` returns a `ProcessResult` with the exit status and bounded tails of standard
output and standard error. A nonzero exit is returned, not raised. A program that
can't start raises `ProcessLaunchFailure`. When the process group's leader exits, the
run also ends whatever the leader left in its group.

### Stopping a process

A process never outlives its phase. Each of these ends the whole process group:

- cancelling `run()`,
- the rollout deadline or the grading deadline,
- an abort.

The run sends `SIGTERM` first, then `SIGKILL` after
`[rollout.cleanup].process_kill_grace_seconds`. `run()` returns only once the group is
gone.

The driver, which is the part of the cookbook that runs each attempt, also ends
processes itself. It ends the attempt's sampling processes when the rollout is sealed,
which is when the run builds the rollout from what sampling captured. It ends every
remaining process before `finish()` or `abort()`. That includes processes a
rollout runtime started and never awaited. This kill happens before a cancelled
hook's own grace starts, so an agent that ignores `SIGTERM` isn't mistaken for a hook
that wouldn't stop.

A group that outlives `SIGKILL` raises `ProcessKillFailure`, which fails the attempt.
No process starts while a hook is being stopped, or once the attempt's last hook has
returned. In those cases, `run()` raises `ProcessesClosed`.

### Reaching the model and the sandbox

An agent program reaches the model through the attempt's HTTP connection. The
connection's credential is revoked when the rollout is sealed. The program reaches a
provisioned sandbox through the sandbox's `usage_environment()`.

The cookbook never has the program create or delete the sandbox, but it does trust the
program with the provider's credential. For Blaxel, the usage environment carries
`BL_API_KEY`, which is workspace-wide, so the program can act on every sandbox in the
workspace. Run only agent programs that you would trust with that key.

The two local patterns show agent processes:

- [patterns/custom_runtime/local](../../patterns/custom_runtime/local/) is a
  custom rollout runtime whose verifier is also a pure process. It uses a
  directory-backed provider, so it runs offline.
- [patterns/harbor/local](../../patterns/harbor/local/) is a
  Harbor agent that runs a program against the trial's sandbox.
