# Settings that interact

[`KNOB_GLOSSARY.md`](KNOB_GLOSSARY.md) defines each setting on its own. This page
covers the ones whose correct value depends on something else, the one Harbor
setting a Blaxel run refuses to start without (`ttl`), and the one it will not accept
(`region`).

There are five. Everything else that could couple is derived (Harbor's trial cap),
implied (route capture and sample recording follow `router_replay`), validated (the
trajectory budget against `max_seq_len`, and `router_replay` against `has_images`), or
scale-invariant (`max_group_losses`).

## Concurrency against sampler capacity

`max_concurrent_rollouts` does not create throughput; it divides whatever your
sampler replicas provide. Set far above what they can serve, every agent runs
slower against a fixed per-task deadline and they begin timing out together — a
throughput shortfall arrives as a wall of timeouts rather than as slowness.

On dedicated Loops, replica count is `[compute] sampler_replicas` (unset, the Loops
backend default), and nothing checks it against `max_concurrent_rollouts` for you.
Serverless capacity is Loops' to size.

*Looks like*: timeouts dominate the discard counters while `slots/sampling` stays full
and `pipeline/sampled` stays flat.

## Concurrency against grading throughput

A rollout holds one Harbor trial slot from the moment Harbor accepts it until its
verifier finishes — through waiting for a sampling slot, the sandbox starting, the
agent running, waiting for a verifier, and grading. Where the verifier runs in the
agent's own sandbox, which is the usual case, it holds that sandbox for the whole
wait too.

The trial cap is `max_concurrent_rollouts` + the pre-start places +
`max_concurrent_verifiers`, so a verifier slower than the agent fills those slots
with rollouts waiting to be graded, and sampling starves.

`max_concurrent_rollouts` covers the agent running: its permit is taken when the
agent starts and released when it ends. Starting the sandbox is provisioning,
bounded separately by `[rollout] max_concurrent_provisioning`, which counts provider
operations in flight rather than sandboxes, so it can be far smaller than the
sampling cap. Waiting for a verifier is not the same shape: that phase is entered
before the permit is taken, so `to-grade` holds none and `grading` carries its cap
alone.

That is backpressure rather than a fault. Throughput is the slower of the two
stages whatever the settings say, and sampling faster than you can grade buys
nothing but sandboxes sitting idle at full price. A larger trial cap would let you
pay for more of them to finish no sooner, which is why the cap is derived and not a
setting.

The lever is `max_concurrent_rollouts`, downward, until it matches what grading
sustains. `max_concurrent_verifiers` is the right lever only when the limit is yours
to raise: a verifier that calls an LLM behind a rate limit is not, and raising it
there moves the queue without draining it.

A verifier that can grade from saved artifacts rather than the live sandbox escapes
this entirely — Harbor then ends the agent's trial when sampling does, releasing
both the slot and the sandbox, and grading runs later against its own concurrency.
That is a property of the task, not a setting here.

*Looks like*: `trials` at its cap while `rollouts` sits below its own, `to-grade`
climbing, and `reward/mean` arriving no faster for it.

## Freshness across the batch against spread within a rollout

`max_steps_off_policy` bounds how stale a rollout's data may be relative to the
current policy. `max_policy_delta_in_trajectory` bounds how many versions a single
rollout may span internally.

Long agentic rollouts cross publishes, so the second binds first, and it aborts the
rollout mid-trajectory, discarding everything it had sampled.

*Looks like*: `discard/rollouts/internal_delta` climbing while mean sampling
time grows.

## The staleness cap against the steps admitted ahead

In async, `max_steps_off_policy` is the staleness cap, and by default it also sizes the
admission window: groups for that many batches beyond the one training. When the
trainer is the bottleneck, that window stays full and a finished group waits behind
the batches ahead of it, so most rollouts train at the full cap. Lowering the cap cuts
the lag but drops the longest groups, the ones that miss it.

`admission_steps_ahead` (at most the cap, fractions allowed) sizes the window on its
own. For long-tailed rollouts, keep the cap loose and admit about one step ahead: lag
falls, and a long group still trains up to the cap. Fewer groups in flight can idle the trainer between
steps; raise `groups_per_step` to keep it fed.

*Looks like*: with the window sized by the cap, `freshness/lag_mean` pinned near
`max_steps_off_policy` while `groups/ready` stays above `groups_per_step`; with the cap
lowered instead, `discard/rollouts/stale_before_grading` and `stale_before_training`
climbing, the longest rollouts among them.
See [scheduling](scheduling.md#admitting-fewer-steps-ahead-than-the-cap).

## The two abort counters

`max_consecutive_unusable_groups` counts groups rejected for training — no
variance, too small. `max_consecutive_rollout_failures` counts infrastructure
failures, and a group grader's failures, each stage on its own.

They are deadlock detectors, not quality gates: a high steady-state discard rate is
normal, a long unbroken run of them is not. An expiry of the rollout clock
(`rollout_timeout_seconds`) counts toward the second: the clock fires only when
`sample()` outlived every deadline of its own, so a run of expiries has a
systemic cause. An agent that runs out of its own deadline returns, and is graded.

## Tracing

The run record — `config.json`, `runtime_evidence_digests.json`, `steps.jsonl`,
`rollouts.jsonl`, the checkpoint manifests, and `run.json` and `failure.json` when they
apply — is always written. It is small (a few MB for a typical run), and it is the product of the run.
`trace` controls the per-generation evidence beside it, which is not.

| Level | Adds | Size, relative |
| --- | --- | --- |
| `off` *(default)* | nothing | — |
| `samples` | one row per generation: identity, policy version, timestamp, stop reason, token counts | ~1% |
| `transcripts` | those rows, and each attempt's transcript in `transcripts/` | between |
| `full` | token arrays and the exact training rows in `batches/` too | 100% |

`full` on a long agentic run is measured in gigabytes, because every turn records
its whole prompt and a trajectory's shared prefix is therefore stored once per
turn. Use it when you need replay or audit.

A transcript is the conversation at the model boundary, one turn per generation:
the request as the agent sent it (only what it adds to an earlier turn's request:
its conversation's latest, or over HTTP without `X-Loops-Conversation-Id` its
producer's latest that the request begins with), its tools, and the reply as the
agent received it — the parsed reply for direct access, a parse failure included,
and the Chat Completions body over HTTP. Each turn is marked `delivered`,
`discarded` or `undelivered`, or `unanswered` for a request no generation came back
for, which has no `sample_id` and no reply and is written whole. A request refused
before it reached the sampler, one too long for the trajectory budget or one that
does not render, leaves no turn; a generation found stale once sampled is
`undelivered`, its reply the error. A turn is `trainable` when
it was delivered by a trainable producer to an attempt that sealed. Trainable is not trained: the rollout can still be dropped after its seal,
by invalid metadata, a failed grade or an unusable group, and `rollouts.jsonl` says
which. Tool definitions are written once per attempt, and images are named by the
digest of the bytes the sampler received, never embedded. A non-finite number, which
a policy can write in a tool call's arguments, is kept as it is in memory and written
as the string `"NaN"`, `"Infinity"` or `"-Infinity"`, so each file stays strict JSON
that any reader loads.

To rebuild a turn's whole request from a file, follow `extends` through every turn
in it: the turn it names can be `discarded` or `undelivered`, and over HTTP without
`X-Loops-Conversation-Id` it can belong to another conversation of the same
producer. Keeping only trainable turns, or grouping by `conversation_id`, before
rebuilding gives wrong or empty requests.
`loops_cookbook.rl.run.transcript_requests(json.loads(path.read_text()))` returns
every turn's whole request by sample id.

Every attempt's transcript is written, however it ends. One that failed — the agent
raised, timed out or was cancelled, or its seal found it stale — is written as far as
it got, with `"sealed": false` and no turn trainable; `rollouts.jsonl` says why, by
`attempt_id`. Recording never changes what trains. A request or reply that cannot be
recorded as JSON — a direct agent whose history holds `bytes` or a `set`, which
renders and trains all the same — is written with its turn's `unrecordable` saying
why, and the turns around it as usual. A sample row or transcript that cannot be
written is logged and counted in `trace/unwritten_samples` or
`trace/unwritten_transcripts`, and its attempt trains. One exception: a transcript
that disagrees with the attempt's samples, a generation with no turn or a turn with no
generation, is a capture bug. It fails the attempt and is not written.

## Required in the Harbor job

One environment setting has no safe default and cannot be derived, so a Blaxel run
refuses to start without it. A Blaxel run also refuses `environment.kwargs.region`:
the region is stated once, as `[providers.blaxel] region`.

**`environment.kwargs.ttl`** must outlast the longest single trial — agent,
verifier and build on one sandbox, after every timeout multiplier. Too short and
sandboxes are reaped while still working. The safe value is usually set by one
outlier task rather than the median.

**Memory is the only Blaxel dial.** Disk is a tmpfs overlay of half the RAM and CPU
is derived from it, so a task's own `storage_mb` and `cpus` are not applied (`check`
warns about a declared `cpus` or `override_cpus`). Each
task's `memory_mb` sizes its sandbox (Harbor's default when it sets none), and
`environment.override_memory_mb`, when set, sizes every task alike. `loops check`
prints the resulting sandboxes and warns about any task whose declared
`storage_mb` does not fit. Under-sizing does not fail the trial: it scores what the
verifier gives, usually zero.

`n_concurrent_trials` is derived from `max_concurrent_rollouts` plus the pre-start
places plus `max_concurrent_verifiers`; setting it in the job file has no effect.
