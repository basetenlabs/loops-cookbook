# Scheduling

This page explains how a run decides which rollouts run at once, and when it starts new
groups. Sampling and grading are separate stages, and each one has its own concurrency
limit. `max_concurrent_rollouts` limits sampling, and `max_concurrent_verifiers` limits
grading. Each limit is a number of slots.

A rollout holds a sampling slot only while its agent is running, and a grading slot
only while it is being graded. A rollout that is waiting to be graded holds neither.
The `Episode` rollout runtime and the Harbor rollout runtime take the sampling slot
themselves, when the agent starts. They declare this in `internally_acquired_phases`.
A rollout runtime written directly against the lower-level `RolloutRuntime` interface
that doesn't declare sampling there holds the sampling slot for all of `sample()`.

## Pre-start places

A rollout runtime can admit a rollout before its agent starts, so that the rollout's
environment is ready when a sampling slot frees. Harbor and the `Episode` rollout
runtime admit the rollout before its sandbox starts. Up to
`ceil([rollout] prestart_allowance × max_concurrent_rollouts)` admitted rollouts may
wait this way beyond the sampling limit. These are the *pre-start places*. They are
open only while admission is not throttled, as described [below](#async), so under
backpressure no environment starts ahead. `loops check` prints the number of pre-start
places, and `admission/prestarted` counts the rollouts that use them.

## Admission order

The run dispatches a group when it takes the group's task from the task stream.
Waiting rollouts are admitted oldest group first. The groups go in the order they were
dispatched, and a group's own rollouts go in the order they arrived. A retried rollout
is a fresh attempt, so it joins the queue behind every group dispatched since. It is
still admitted ahead of those groups, because its own group, and every batch that the
group belongs to, waits for it.

Once a rollout is admitted, its provider's create window paces the rollout's sandbox
create. The window narrows while the provider refuses creates. See
[resources](resources.md#provider-resources). The cookbook derives Harbor's own trial
cap from all three limits, as
`max_concurrent_rollouts + pre-start places + max_concurrent_verifiers`, so you have
nothing to keep in step by hand.

## Sync

In sync mode, `mode = "sync"`, the run admits exactly the groups that the next update
needs and waits for them. Then it trains, publishes the new weights, and admits the
next set. Every sample in a batch comes from the policy that the batch trains. If a
group is dropped while the batch collects, the run replaces it before training starts.

## Async

In async mode, `mode = "async"`, the run keeps sampling while the trainer works. A
group starts when the first of its rollouts is admitted. The run starts a group only
while that group can still train fresh. A group that has started
fills whatever happens, so the run bounds the groups that have started and are not
yet training. That bound is the admission window:

```
window = (steps ahead, or one more with no batch training) × groups_per_step
         + the groups expected to drop, up to a batch of them
steps ahead = admission_steps_ahead (fractions allowed), or max_steps_off_policy when unset
```

While a batch trains, the next `max_steps_off_policy` batches can still train fresh.
When no batch is training, the batch being assembled trains at the current version, so
one more batch fits. By default the window admits groups for all of these batches.
`admission_steps_ahead` admits fewer, as described
[below](#admitting-fewer-steps-ahead-than-the-cap).

The window counts groups, because a batch takes `groups_per_step` groups however many
members each one has left. Inside the window, sampling runs at the full
`max_concurrent_rollouts`. A closed window never cancels a running rollout. It only
stops new groups from starting. Committing or aborting a batch frees its places.

### Finished members lend their slots

Admission holds the window in rollouts. A complete group holds its whole place, which
is `group_size` rollouts. A group that is still filling holds only its members that
have not resolved yet. The finished members' slots go to new groups instead.

The reason is a group that waits on one long member. If the window held whole groups,
that group would keep its whole place while its other members sat finished. A window
full of such groups would leave the sampler idle while the trainer waited for them.

The finished members' slots are lent, not freed. A group holds its whole place again
once it is complete, so the count can pass the window by as many members as the
filling groups had finished. Underneath this, the started groups still fit the window
that `max_steps_off_policy` would size, with each group taking a whole batch place
however few members it has left. So no group starts that could only train stale. When
`admission_steps_ahead` is unset, the two windows are the same, and that bound is the
one that holds. Admission is then by groups.

### Replacements for expected drops

A group dropped before training, because it had no variance or too few members, takes
its place with it. If a replacement starts only then, the trainer waits for it. So the
window starts those replacements ahead of time.

The run measures the expected drops from recent groups. It takes the share of the last
four batches' worth of groups that passed the size and variance check. It counts each
group once, when the group is first judged, so a good group counts as soon as a bad
one. The share never goes below a quarter, and the run counts at most one batch,
`groups_per_step`, of expected drops. It rounds the expected drops up to whole groups,
because a replacement is a whole group.

The cap of one batch exists because drops rise as a policy degrades or saturates.
Without the cap, the window would widen most when the policy was least stable, and
each extra group would train later. A batch that is short of replacements takes groups
from the batch admitted after it. A run that drops more than a batch is better slowed
down than fed staler groups.

When fewer groups drop than expected, the surplus trains a batch later. Below the
staleness cap, that only raises its lag. With the window at the whole cap, the surplus
trains too late and is discarded as stale. Groups that lost a member to staleness
before they were judged stay out of the estimate, because otherwise stale drops would
widen the window that caused them.

These metrics report the window:

- `admission/committed_groups` counts the groups started and not yet training.
- `admission/window_groups` is the window in groups.
- `admission/group_survival` is the share of groups that survive the check.
- `admission/committed_rollouts` and `admission/window_rollouts` give the committed
  groups and the window in rollouts.

`window_groups` is `window_rollouts` divided by `group_size`, so it can be a fraction of
a group. Sync mode has no window, and it reports none of these metrics.

`admission_headroom` isn't a setting, and `loops check` refuses a config that sets it.
The window always counts the expected drops.

### The last steps of a run

With a nonzero `max_steps`, the window never reaches past the run's last step. Async
admits a group only while the steps that are left need it. The steps left run up to
`max_steps`, which also counts a resumed run's earlier updates. The run subtracts the
batch that is training, and adds the same allowance for expected drops. A dropped group
still frees its place for a replacement. Once the last batch is training, the run
admits nothing more. Sync already admits only the batch it is assembling.

### When the window holds sampling back

`admission/window_held_seconds` totals the time the window held new groups back while
fewer than `max_concurrent_rollouts` attempts were admitted. Each update reports
`admission/window_held_share`, which is that time as a share of the wall time since the
previous update. An attempt counts from admission, so an attempt that is still starting
its sandbox counts as busy. Slots that are idle behind sandbox start-up count as
start-up time, not as the window's time. The pre-start places are what hide start-up.

A high share means that sampling outruns what the trainer can use within the steps
admitted ahead, so sampler capacity is going unused. Raise `admission_steps_ahead`, up
to `max_steps_off_policy`, or raise `max_steps_off_policy`. You can also rebalance the
sampler replicas against the trainer step.

### Staleness

Rollouts sampled under an older policy train as long as they pass the freshness checks.
The loss uses the recorded behavior logprobs to correct for the mismatch. A rollout that
samples for longer than `max_steps_off_policy` updates is stale however you set
admission. To keep long rollouts, raise `max_steps_off_policy`.

### Which groups a batch takes

A batch first takes the groups held over from an earlier batch that needed
replacements. Then it takes the ready group that is closest to going stale. That is
the group whose oldest member sampled at the lowest policy version, and among equals,
the group dispatched first.

The run doesn't take groups in the order they complete. In completion order, a long
group that started early and finished late would wait behind newer, shorter groups
until it aged out of `max_steps_off_policy`. With this order, when sampling outpaces
training, the groups that are left to age are the newest ones. Selection never waits
for a particular group, so it costs no throughput.

### Settings async requires

Async requires `max_steps_off_policy`, which must be at least 1, and
`max_policy_delta_in_trajectory`. A rollout sampled while a step trains is already one
version behind when it could train. Sync refuses both settings and fixes them at 0,
because it trains each rollout on the policy that sampled it.

## Admitting fewer steps ahead than the cap

Unless `admission_steps_ahead` is set, `max_steps_off_policy` does two jobs. It is the
staleness cap, and it sizes the window. When the trainer is the bottleneck, a window of
the whole cap fills. Completed groups then wait in the ready queue for the batches
ahead of them, so nearly every rollout trains at a lag of the full cap. If you lower the
cap to cut that lag, the run drops the groups that take longest, because they are the
ones that miss the cap. That trains against long trajectories.

`admission_steps_ahead` sizes the window on its own. The run still judges staleness
against `max_steps_off_policy` everywhere it checks it. With long-tailed rollouts, keep
the cap loose enough that the slowest groups still train, and admit about one step
ahead. Groups then start closer to the time the trainer can take them, so they train
at a lower lag, and a group that runs long is kept up to the cap. Below the cap, a
group waiting on a long member lends its finished members' slots to new groups, up to
the number of groups the cap leaves room for. So the sampler keeps working while the
batch waits. Fewer groups in flight can still leave the trainer waiting between steps.

`admission_steps_ahead` takes fractions. With `0.5`, the run starts groups for half of
the next batch, in rollouts, while one batch trains. So a group waiting on a long
member still lends its finished members' slots. A group starts only once its whole
`group_size` fits in the window. So with one group per batch, a fraction below 1 starts
no group while a step trains, unless recent drops add a group to the window as an
expected drop.

The setting must be above 0 and at most `max_steps_off_policy`. A group admitted for a
batch further ahead would be stale by the time that batch trains. With 0, nothing would
start while a step trains, which is what `mode = "sync"` is for. Sync has no window,
and it refuses the setting.

### Choosing it

The right number depends on how many steps a rollout takes, plus how much margin
absorbs the noise in how long groups take. Only the run knows either. Start near
`time/pipeline/sampling/mean_seconds` divided by the step time, rounded up. Then read
these metrics at each update:

- `admission/batch_wait_seconds` is how long the trainer waited for its last batch. A
  trainer that keeps waiting wants more steps ahead. The exception is when
  `admission/window_held_share` is near 0. Then every sampling slot is busy, and
  sampling is the limit, not admission.
- `admission/ready_spare_groups` counts the complete groups beyond a batch when the
  trainer asked for one. It is negative when the trainer was short. If groups are to
  spare at every step, they trained later than they had to. Fewer steps ahead then
  lowers `freshness/lag_mean` at no cost in throughput.

You choose how much waiting to trade for lag. Fewer steps ahead trains closer to the
policy that sampled, and it leaves the trainer waiting more often. Async admission is
static, so nothing here changes as the run goes. If a run's rollouts lengthen as the
policy learns, look at these metrics again.
