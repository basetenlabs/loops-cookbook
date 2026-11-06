# Scheduling

Sampling and grading are separate stages with separate concurrency limits,
`max_concurrent_rollouts` and `max_concurrent_verifiers`. A rollout holds a sampling
slot only while its agent is running (for a custom runtime that does not declare
`internally_acquired_phases`, all of `sample()`), and a grading slot only while it is being
graded; waiting for grading holds neither.

A runtime can admit a rollout before its agent starts, so that its environment is
ready when a sampling slot frees; Harbor and the `Episode` runtime do so before the
sandbox starts. Up to `ceil([rollout] prestart_allowance × max_concurrent_rollouts)`
admitted rollouts, the *pre-start places*, may wait that way beyond the sampling
limit, and only while admission is not throttled (below), so under backpressure no
environment starts ahead. `loops check` prints the number, and
`admission/prestarted` counts them.

Waiting rollouts are admitted oldest group first: in the order their groups were
dispatched, and a group's own in arrival order. A retried rollout is a fresh attempt
that joins the queue behind every group dispatched since; it is still admitted ahead
of them, since its group, and every batch it belongs to, waits for it. An admitted
rollout's sandbox create is then paced by its provider's create window, which
narrows while the provider refuses creates
([resources](resources.md#provider-resources)). Harbor's own trial cap is derived from all three,
`max_concurrent_rollouts + pre-start places + max_concurrent_verifiers`, so there is
nothing to keep in step by hand.

**Sync** (`mode = "sync"`) admits exactly the groups the next update needs, waits
for them, trains, publishes, then admits the next set. Every sample in a batch comes
from the policy the batch trains. A group dropped while it collects is replaced before
training starts.

**Async** (`mode = "async"`) keeps sampling while the trainer works. It starts a group
only while that group can still train fresh. A started group fills regardless, so the
groups started and not yet training are what is bounded:

```
window = (steps ahead, or one more with no batch training) × groups_per_step
         + the groups expected to drop, up to a batch of them
steps ahead = admission_steps_ahead (fractions allowed), or max_steps_off_policy when unset
```

While a batch trains, the next `max_steps_off_policy` batches train fresh; with none
training, the batch being assembled trains at the current version, so one more fits.
By default the window admits groups for all of them;
`admission_steps_ahead` admits fewer ([below](#admitting-fewer-steps-ahead-than-the-cap)).
The window counts groups, since a batch takes `groups_per_step` of them however many
members each has left. Inside it sampling runs at full `max_concurrent_rollouts`, and
a closed window never cancels a running rollout: it only stops new groups starting.
Committing or aborting a batch frees its places.

Admission holds the window in rollouts: a complete group holds its whole place
(`group_size`), and one still filling holds only its members not yet resolved. Counted
by groups, a group waiting on one long member kept its place while the rest sat
finished, and a window of such groups left the sampler idle while the trainer waited
for them (run nwx9k5w: 26 to 394 rollouts in flight at a cap of 400, the trainer idle
~42%). The finished members' slots go to new groups instead. They are lent: a group
holds its whole place again once complete, so the count can pass the window by what
filling groups had finished. Under that, the started groups still fit the window
`max_steps_off_policy` would size, each a batch place however few members it has left,
so no group starts that could only train stale. With `admission_steps_ahead` unset
the two are one window and that bound is what holds: admission is by groups, as it
always was.

Groups dropped before training (no variance, too few members) take their places with
them, and a replacement starting only then leaves the trainer waiting. The window starts
those replacements ahead of time: the expected drops are measured from recent groups,
the share of the last four batches' worth that passed the size and variance check,
each counted once, when first judged, so a good group counts as soon as a bad one;
floored at a quarter, and counted up to one batch (`groups_per_step`): drops rise as a
policy degrades or saturates, so uncounted, the window widened most when the policy was
least stable, each extra group training later. A batch short of replacements takes
groups from the batch admitted after it, and a run dropping more than a batch is better
slowed than fed staler groups. When fewer drop than expected, the surplus trains a
batch later: below the cap that only raises its lag, but with the window at the whole
cap it trains too late and is discarded as stale. Groups that lost a
member to staleness before they were judged stay out of the estimate, or stale drops
would widen the window that made them. `admission/committed_groups`, `admission/window_groups` and
`admission/group_survival` report the three, and `admission/committed_rollouts` and
`admission/window_rollouts` the window in rollouts. `window_groups` is `window_rollouts`
over `group_size`, so it can be a fraction of a group; sync has no window, and reports
none of the window's gauges.

With a nonzero `max_steps`, the window also never reaches past the run's last step: async
admits a group only while the steps left (up to `max_steps`, which counts a resumed
run's earlier updates too) need it, less the batch training, plus the same
allowance for expected drops. A dropped group still frees its place for a replacement,
and once the last batch is training nothing more is admitted. Sync already admits only
the batch it is assembling.

`admission/window_held_seconds` totals the time the window held new groups back while
fewer than `max_concurrent_rollouts` attempts were admitted, and each update reports
`admission/window_held_share`, the share of wall time since the previous update. An
attempt counts from admission, so one still starting its sandbox counts as busy: slots
idle behind sandbox start-up are start-up time, not the window's, and the pre-start
places are what hide it. A high share says sampling outruns what the trainer can use
within the steps admitted ahead: sampler capacity is going unused.
Raise `admission_steps_ahead` (up to `max_steps_off_policy`) or `max_steps_off_policy`,
or rebalance sampler replicas against the trainer step.

Rollouts sampled under an older policy train as long as they pass the freshness
checks, and the loss uses the recorded behavior logprobs to correct for the mismatch.
A rollout that samples for longer than `max_steps_off_policy` updates is stale however
admission is set: raise `max_steps_off_policy` to keep long rollouts.

A batch takes the groups held over from a batch that needed replacements first, then
the ready group closest to going stale: the one whose oldest member sampled at the
lowest policy version, then the one dispatched first. Completion order would let a
long group, started early and finished late, wait behind newer, shorter ones until it
aged out of `max_steps_off_policy`; when sampling outpaces training, the groups this
leaves to age are the newest. Selection never waits for a particular group, so it
costs no throughput.

Async requires `max_steps_off_policy`, at least 1, and `max_policy_delta_in_trajectory`:
a rollout sampled while a step trains is one version behind when it could train. Sync
refuses both and fixes them at 0, as it trains each rollout on the policy that sampled it.

## Admitting fewer steps ahead than the cap

`max_steps_off_policy` does two jobs unless `admission_steps_ahead` separates them: it
is the staleness cap, and it sizes the window. When the trainer is the bottleneck, a
window of the whole cap fills, and completed groups wait in the ready queue for the
batches ahead of them: nearly every rollout trains at a lag of the full cap. Lowering
the cap to cut that lag drops the groups that take longest, since they are the ones that
miss it, and so trains against long trajectories.

`admission_steps_ahead` sizes the window alone; staleness is still judged against
`max_steps_off_policy` everywhere it is checked. With long-tailed rollouts, keep the cap
loose enough that the slowest groups still train, and admit about one step ahead: groups
start closer to when the trainer can take them, so they train at a lower lag, and one
that runs long is kept up to the cap. Below the cap, a group waiting on a long member
lends its finished members' slots to new groups, up to the groups the cap leaves room
for, so the sampler keeps working while the batch waits for it. Fewer groups in flight
can still leave the trainer waiting between steps.

It takes fractions: `0.5` starts groups for half the next batch while one trains, in
rollouts, so a group waiting on a long member still lends its finished members' slots;
a group starts only once its whole `group_size` fits, so with one group a batch any
fraction below 1 starts none while a step trains.
It must be above 0 and at most `max_steps_off_policy`: a group admitted for a batch
further ahead is stale by the time that batch trains, and with 0 nothing starts while a
step trains, which is what `mode = "sync"` is for. Sync has no window, and refuses the
setting.

### Choosing it

The right number is about how many steps a rollout takes, plus what absorbs the noise
in how long groups take, and only the run knows either. Start near
`time/sampling/mean_seconds` over the step time, rounded up, and read each update:

- `admission/batch_wait_seconds`: how long the trainer waited for its last batch. A
  trainer that keeps waiting wants more ahead, unless `admission/window_held_share` is
  near 0: then every sampling slot is busy and sampling is the limit, not admission.
- `admission/ready_spare_groups`: complete groups beyond a batch when the trainer asked
  for it, negative when it was short. Groups to spare at every step trained later than
  they had to: fewer ahead lowers `freshness/lag_mean` at no cost in throughput.

How much wait to trade for lag is yours to choose: less ahead trains closer to the
policy that sampled, and leaves the trainer waiting more often. Async admission is
static: nothing here changes as the run goes, so a run whose rollouts lengthen as the
policy learns wants a look at these again.

`admission_headroom` was removed: the window always counts the expected drops, and a
config that sets it is refused.
