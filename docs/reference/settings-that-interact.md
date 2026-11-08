# Settings that interact

[`KNOB_GLOSSARY.md`](KNOB_GLOSSARY.md) defines each setting on its own. This page
covers the settings whose correct value depends on something else. It also covers
tracing, and the two Harbor job settings that a Blaxel run checks: `ttl`, which a
Blaxel run refuses to start without, and `region`, which it refuses to accept.

Five pairs of settings interact. The cookbook handles every other setting that could
couple in one of four ways:

- It derives the setting, as it derives Harbor's trial cap.
- The setting is implied, as route capture and sample recording follow
  `router_replay`.
- It validates the setting, as it checks the trajectory budget against `max_seq_len`,
  and `router_replay` against `has_images`.
- The setting keeps its meaning at any scale, as `max_group_losses` is a tolerance of
  lost rollouts rather than a floor, so it means the same when `group_size` changes.

## Concurrency against sampler capacity

`max_concurrent_rollouts` doesn't create throughput. It divides whatever your sampler
replicas provide. If you set it far above what the replicas can serve, every agent runs
slower against a fixed per-task deadline, and the agents begin timing out together. A
shortfall in throughput then arrives as a wall of timeouts rather than as slowness.

On dedicated Loops, `[compute] sampler_replicas` sets the replica count. When it is
unset, the Loops backend default applies. Nothing checks the replica count against
`max_concurrent_rollouts` for you. On serverless Loops, Loops sizes the capacity.

*What it looks like:* timeouts dominate the discard counters, while `slots/sampling`
stays full and `pipeline/sampled` stays flat.

## Concurrency against grading throughput

With the Harbor rollout runtime, a rollout holds one Harbor trial slot from the moment
Harbor accepts it until its verifier finishes. A trial is Harbor's run of one task. The
slot covers waiting for a sampling slot, starting the sandbox, running the agent,
waiting for a verifier, and grading. In the usual case, the verifier runs in the
agent's own sandbox, so the rollout holds that sandbox for the whole wait too.

The trial cap is `max_concurrent_rollouts` plus the
[pre-start places](scheduling.md#pre-start-places) plus `max_concurrent_verifiers`. So a
verifier that is slower than the agent fills those slots with rollouts waiting to be
graded, and sampling starves.

`max_concurrent_rollouts` covers the agent running. Its permit is taken when the agent
starts and released when the agent ends. Starting the sandbox is provisioning, which
`[rollout] max_concurrent_provisioning` bounds separately. That setting counts provider
operations in flight rather than sandboxes, so it can be far smaller than the sampling
cap. Waiting for a verifier works differently. A rollout enters that phase before it
takes the verifier permit. So in the live console, the `to-grade` row holds no permit,
and the `grading` row carries the verifier cap alone.

This is backpressure, not a fault. Throughput is the speed of the slower of the two
stages, whatever the settings say. Sampling faster than you can grade buys nothing but
sandboxes that sit idle at full price. A larger trial cap would let you pay for more of
them to finish no sooner, which is why the cookbook derives the cap instead of making
it a setting.

The lever is `max_concurrent_rollouts`. Lower it until it matches what grading
sustains. `max_concurrent_verifiers` is the right lever only when the limit is yours to
raise. A verifier that calls a large language model (LLM) behind a rate limit is not
yours to raise. Raising `max_concurrent_verifiers` there moves the queue without
draining it.

A verifier that can grade from saved artifacts, rather than from the live sandbox,
avoids this problem entirely. Harbor then ends the agent's trial when sampling ends,
which releases both the slot and the sandbox, and grading runs later under its own
concurrency. That is a property of the task, not a setting here.

*What it looks like:* in the live console, the `trials` row sits at its cap while the
`rollouts` row sits below its own cap, `to-grade` climbs, and `reward/mean` arrives no
faster for it.

## Freshness across the batch against spread within a rollout

`max_steps_off_policy` bounds how stale a rollout's data may be, relative to the
current policy. `max_policy_delta_in_trajectory` bounds how many policy versions a
single rollout may span.

Long agentic rollouts cross several weight publishes, so the second limit binds first.
When it does, the cookbook discards the rollout, along with everything it had sampled.

*What it looks like:* `discard/rollouts/internal_delta` climbs while the mean sampling
time grows.

## The staleness cap against the steps admitted ahead

In async mode, `max_steps_off_policy` is the staleness cap. By default it also sizes
the admission window, which then holds groups for that many batches beyond the one
training. When the trainer is the bottleneck, that window stays full. A finished group
waits behind the batches ahead of it, so most rollouts train at the full cap. Lowering
the cap cuts the lag, but it drops the longest groups, because they are the ones that
miss the cap.

`admission_steps_ahead` sizes the window on its own. It must be at most the cap, and it
takes fractions. For long-tailed rollouts, keep the cap loose and admit about one step
ahead. The lag then falls, and a long group still trains up to the cap. Fewer groups in
flight can leave the trainer idle between steps. To keep it fed, raise
`groups_per_step`.

*What it looks like:* with the window sized by the cap, `freshness/lag_mean` stays
pinned near `max_steps_off_policy` while `groups/ready` stays above `groups_per_step`.
With the cap lowered instead, `discard/rollouts/stale_before_grading` and
`stale_before_training` climb, and the longest rollouts are among them. See
[scheduling](scheduling.md#admitting-fewer-steps-ahead-than-the-cap).

## The two abort counters

`max_consecutive_unusable_groups` counts groups rejected for training, such as groups
with no variance or too few members. `max_consecutive_rollout_failures` counts
infrastructure failures and a group grader's failures, with a separate count for each
stage.

Both counters detect a deadlock. They are not quality gates. A high, steady rate of
discards is normal, but a long unbroken run of them is not. An expiry of the rollout
clock, `rollout_timeout_seconds`, counts toward `max_consecutive_rollout_failures`. The
clock fires only when `sample()` outlived every deadline of its own, so a run of
expiries has a systemic cause. An agent that runs out of its own deadline returns, and
it is graded.

## Tracing

The run always writes its run record. The record is `config.json`,
`runtime_evidence_digests.json`, `steps.jsonl`, `rollouts.jsonl`, the checkpoint
manifests, and `run.json` and `failure.json` when they apply. It is small, a few MB for
a typical run, and it is the product of the run. `trace` controls the per-generation
evidence beside it, which is not part of the record.

| Level | What it adds | Size, relative to `full` |
| --- | --- | --- |
| `off` *(default)* | It adds nothing. | — |
| `samples` | It adds one row per generation, with its identity, policy version, timestamp, stop reason and token counts. | About 1% |
| `transcripts` | It adds those rows, and each attempt's transcript in `transcripts/`. | Between the two |
| `full` | It also adds the token arrays and the exact training rows in `batches/`. | 100% |

`full` on a long agentic run is measured in gigabytes. Every turn records its whole
prompt, so a trajectory's shared prefix is stored once per turn. Use `full` when you
need replay or an audit.

### What a transcript holds

A transcript is the conversation at the model boundary, with one turn per generation.
Each turn holds three things:

- The request as the agent sent it. The turn stores only what the request adds to an
  earlier turn's request. That earlier turn is its conversation's latest. Over HTTP
  without `X-Loops-Conversation-Id`, it is the latest turn of the same producer that
  the request begins with.
- The request's tools.
- The reply as the agent received it. For direct access, that is the parsed reply,
  including a parse failure. Over HTTP, it is the Chat Completions body.

Each turn is marked `delivered`, `discarded` or `undelivered`. A request that no
generation came back for is marked `unanswered`. It has no `sample_id` and no reply,
and it is written whole. A request that was refused before it reached the sampler
leaves no turn. Examples are a request too long for the trajectory budget and one that
doesn't render. A generation that is found stale once sampled is `undelivered`, and its
reply is the error.

A turn is `trainable` when a trainable producer delivered it to an attempt that sealed.
Trainable doesn't mean trained. The rollout can still be dropped after its seal,
because of invalid metadata, a failed grade or an unusable group, and `rollouts.jsonl`
says which.

The transcript writes tool definitions once per attempt. It names images by the digest
of the bytes the sampler received, and never embeds them. A policy can write a
non-finite number in a tool call's arguments. The transcript keeps it as it is in
memory and writes it as the string `"NaN"`, `"Infinity"` or `"-Infinity"`, so each file
stays strict JSON that any reader loads.

### Rebuilding a request

To rebuild a turn's whole request from a file, follow `extends` through every turn in
the file. The turn that `extends` names can be `discarded` or `undelivered`. Over HTTP
without `X-Loops-Conversation-Id`, it can belong to another conversation of the same
producer. If you keep only the trainable turns, or group by `conversation_id`, before
you rebuild, you get wrong or empty requests.
`loops_cookbook.rl.run.transcript_requests(json.loads(path.read_text()))` returns every
turn's whole request, keyed by sample ID.

### Failed attempts and unrecordable turns

The run writes every attempt's transcript, however the attempt ends. An attempt can
fail because the agent raised, timed out or was cancelled, or because its seal found it
stale. Its transcript is written as far as it got, with `"sealed": false` and no turn
trainable. `rollouts.jsonl` says why, by `attempt_id`.

Recording never changes what trains. A direct agent's history can hold `bytes` or a
`set`, which renders and trains all the same but can't be recorded as JSON. The turn
with such a request or reply is written with its `unrecordable` field saying why, and
the turns around it are written as usual. When a sample row or transcript can't be
written, the run logs it and counts it in `trace/unwritten_samples` or
`trace/unwritten_transcripts`, and its attempt trains.

There is one exception. A transcript that disagrees with the attempt's samples, such as
a generation with no turn or a turn with no generation, is a capture bug. It fails the
attempt, and the run doesn't write it.

## Required in the Harbor job

One environment setting has no safe default and can't be derived, so a Blaxel run
refuses to start without it. A Blaxel run also refuses `environment.kwargs.region`,
because the region is stated once, as `[providers.blaxel] region`.

**`environment.kwargs.ttl`**, the sandbox's time to live, must outlast the longest
single trial. That trial runs the
agent, the verifier and the build on one sandbox, after every timeout multiplier. If
`ttl` is too short, Blaxel reaps sandboxes while they are still working. The safe value
is usually set by one outlier task rather than by the median.

**Memory is the only Blaxel dial.** The disk is a tmpfs overlay of half the RAM, and
the CPU is derived from the memory. So the cookbook doesn't apply a task's own
`storage_mb` and `cpus`, and `loops check` warns about a declared `cpus` or
`override_cpus`. Each task's `memory_mb` sizes its sandbox. A task that sets none gets
Harbor's default. When `environment.override_memory_mb` is set, it sizes every task
alike. `loops check` prints the resulting sandboxes, and it warns about any task whose
declared `storage_mb` doesn't fit. Under-sizing doesn't fail the trial. The trial
scores what the verifier gives, which is usually zero.

The cookbook derives `n_concurrent_trials` from `max_concurrent_rollouts` plus the
pre-start places plus `max_concurrent_verifiers`. Setting it in the job file has no
effect.
