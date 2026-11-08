# Resources and agent processes

## Provider resources

A runtime that needs sandboxes, volumes, or anything else a provider bills for asks
for it through `attempt.services.resources` (one attempt's own) or
`context.run_resources` (shared by the run), never through the provider directly.
The request names a provider, a kind, a `logical_key` unique within its owner (the
attempt, or the run), and a
JSON specification; the registry returns a handle and owns everything after that:

1. an immutable creation intent is written to the run's resource journal and fsynced
   before the provider is called, carrying the exact name and ownership labels
   (`loops.run_id`, `loops.scope`, `loops.owner_id`, `loops.resource_id`) the
   resource will be created with;
2. the create runs under the provisioning limit and the provider's create deadline. A
   create the provider turns away for now (Blaxel's `429`) is sent again within that
   deadline, after a wait of 1 s doubling to 30 s with jitter, and only while half of
   it is left. How many of one provider's creates are in flight at once is a window:
   it starts at `max_concurrent_provisioning`, halves on a refusal (once for a burst
   refused together), and grows by about one for each window of creates the provider
   answers without refusing (a failure or timeout counts too), back to that cap; a
   fractional size rounds up to the next whole slot. A transient helper that fails to
   create (an image build) fails the requests for its key at once for 90 s
   (`FAILED_HELPER_COOLDOWN_SECONDS`), instead of each queued request building it again. Every create, first send or retry, waits for a slot holding no
   permit, in the order it first asked for one, so a retry keeps its place ahead of
   newer creates; image builds share their provider's window. A wait before a
   create's first send costs none of its deadline. One INFO line says when the window
   first narrows, and one when it is back at its cap, with the refusals in between;
3. its outcome is journaled. A create that failed in a way that may still have
   created something is looked up by that name and those labels, and counts as
   absent only on authoritative evidence; an empty or failed lookup is unknown,
   never absent;
4. a release is deferred while a lease or a dependent resource still needs it, and
   is complete only once the provider proves the resource gone. A release runs to
   its deadline even if the hook that asked for it is cancelled.

An attempt's resources are released when it ends, whatever its hooks did, unless a
hook ignores cancellation: its attempt stays live and its resources are left for
`loops cleanup`. One that
cannot be proven gone fails the attempt's cleanup and is kept, by exact identity,
for recovery. Resource-service time counts against neither the setup nor the
cleanup deadline. Unrelated resources provision concurrently, operations on one
resource are serialized, and `[rollout] max_concurrent_provisioning` bounds how many
provider operations run at once; a caller waiting for another's prerequisite holds
none of that limit.

Each provider is configured by its own `[providers.<name>]` table. A provider that
is not built in names its factory with `adapter = "module:callable"`; the factory is
called with the table's other keys and returns an adapter declaring a finite
deadline for every operation. No table holds a credential: each provider reads its
own from the environment, loaded from `.env`, so a missing one
fails before the run or `loops cleanup` calls the provider.

An adapter that can back an Episode's sandbox is also a `SandboxProvider`
(`loops_cookbook.resources`): `sandbox_kind` is the sandbox kind it provides,
`client(resource)` gives the provider's own client, bounded by the attach deadline,
`close_client(client)` closes it, and `alive(resource)` says whether the resource is
still usable. The provider ships its kind for Episodes to name, such as
`class Blaxel(SandboxKind[SandboxInstance]): provider = "blaxel"`, which types the
client, and whose `spec` types `[rollout_runtime.config.sandbox]`: for Blaxel, `image`,
`memory_mb` and `ttl`, flat beside `provider`, with any other key, and a value of
another type, refused; `ttl` is a whole number and one unit, `s`, `m`, `h`, `d` or
`w`, as Blaxel takes it. All three are required. Blaxel's adapter is one; `loops check` refuses an Episode sandbox from an
adapter that is not, or whose `sandbox_kind` is not the episode's kind or a subclass
of it. The match is by type, so a provider may be called anything in `[providers]`.

Blaxel and Daytona are built in; Daytona's table and its Harbor environment are in
[daytona.md](daytona.md). Blaxel's table needs only the region; every deadline has a default
(see the [knob glossary](KNOB_GLOSSARY.md)), and one set in the table replaces it.
Its credentials are `BL_WORKSPACE` and `BL_API_KEY`, from the environment or `.env`:

```toml
[providers.blaxel]
region = "us-was-1"
# attach_timeout_seconds = 45   # only to change a default
```

A sandbox's image is adopted when Blaxel already has it and is otherwise built once
per run for every sandbox that needs it; the builder sandbox is deleted once the
build finishes, and the image itself is never deleted. The journal lives at `<run directory>/<[resources] state_dir>/resources/v1/journal.jsonl`,
which must be on a local filesystem (NFS, SMB, and FUSE are refused); one process
writes it at a time.

A resumed run reuses its run directory, `run_id`, and journal, and releases
everything its previous process left before admitting an attempt. Whatever a run
cannot prove released is reported (at startup for a previous process's leftovers, and
at shutdown), and a run that would otherwise complete ends `unclean`. At shutdown no
create that has not been sent starts; the run's releases and the creates already sent
share one bound, `[rollout] shutdown_drain_seconds`, and what is still in flight then
is cut off and reported for `loops cleanup`. A damaged
journal refuses to start a run; its cleanup is never reported complete, and once its
resources are accounted for it has to be moved aside by hand. To finish
the job after a crash:

```sh
uv run loops cleanup train.toml --state-dir runs/<run>/state   # from the journal
uv run loops cleanup train.toml --run-id <run>                 # journal lost: by labels
```

The first replays the journal, releases what it holds in dependency order, and
refuses while the run still holds the journal. The second finds the run's resources
by their ownership labels, and must only be run once the run has stopped; because
listings lag, a sweep never proves that nothing is left, so it always reports its
listing `unconfirmed`. A sweep reads the provider's listing a page at a time, and
`list_timeout_seconds` bounds each page, not the whole listing: Blaxel filters no labels,
so the sweep reads every sandbox in the workspace and takes as long as the workspace
is large. Both read only the config's `[providers]` tables, print every
identity, and are safe to rerun. The journal cleanup exits 0 once every resource is
proven released and 1 while any is not. A sweep exits 1 while a listed resource is
unresolved or a listing failed, and 3 once everything it listed is released: never 0
unless the config has no `[providers]` tables to sweep, since the listing itself
cannot be proven complete. Exit 2 is a configuration error or
a journal another process holds.

## Agent processes

An agent the cookbook does not own, a program in another language, interpreter, or
virtual environment, runs through `attempt.services.processes`:

```python
result = await attempt.services.processes.run(
    ProcessCommand(
        argv=("python3", "/abs/path/agent.py"),
        cwd=results_dir,
        env={"OPENAI_BASE_URL": model.base_url, "OPENAI_API_KEY": model.api_key, **usage},
    )
)
```

The process starts in its own process group, with standard input closed, in `cwd`.
Its environment is a short non-secret allowlist (`PATH`, `HOME`, `LANG`, `LC_*`, and
a few others) plus exactly `command.env`, so nothing else in the cookbook's
environment is inherited. A `SecretStr` in `env` reaches the process only there, and
is redacted from the output the cookbook keeps. `run()` returns a `ProcessResult`
with the exit status and bounded tails of standard output and standard error; a
nonzero exit is returned, not raised. A program that cannot start raises
`ProcessLaunchFailure`. When the leader exits, whatever it left in its process group
is ended too.

A process never outlives its phase. Cancelling `run()`, the rollout or grading
deadline, and an abort each end the whole process group: `SIGTERM`, then `SIGKILL`
after `[rollout.cleanup] process_kill_grace_seconds`, and `run()` returns only once
the group is gone. The driver also ends the attempt's sampling processes when the
rollout is sealed, and every remaining one before `finish()` or `abort()`, including
processes a runtime started and never awaited. Because that kill happens before a
cancelled hook's own grace starts, an agent that ignores `SIGTERM` is not mistaken
for a hook that would not stop. A group that outlives `SIGKILL` raises
`ProcessKillFailure`, which fails the attempt. No process starts while a hook is being stopped or
once the attempt's last hook has returned: `run()` then raises `ProcessesClosed`.

An agent program reaches the model through the attempt's HTTP connection, whose
credential is revoked at sealing, and reaches a provisioned sandbox through its
`usage_environment()`. The cookbook never has it create or delete the sandbox, but
it trusts the program with the provider's credential: for Blaxel the usage
environment carries `BL_API_KEY`, which is workspace-wide, so the program can act on
every sandbox in the workspace. Run only agent programs you would trust with that
key. The two local
patterns show it: [patterns/custom_runtime/local](../../patterns/custom_runtime/local/),
a custom runtime with a pure verifier process and a directory-backed provider so it
runs offline, and [patterns/harbor/local](../../patterns/harbor/local/), a Harbor agent
that runs a program against the trial's sandbox.
