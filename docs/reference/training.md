# Training

This page explains what happens between `loops run` and a new policy version, and what
each setting controls. It describes the code in `loops_cookbook.rl` and
`loops_cookbook.llm`. The Harbor rollout runtime changes only the environment side,
which [harbor.md](harbor.md) covers. The Loops trainer and sampler are explained in the
[Loops concepts](https://docs.baseten.co/loops/concepts).

## The loop

A reinforcement learning (RL) run trains on groups of rollouts. A rollout is one
attempt by your agent at one task, together with every model call it made and the
reward its grader gave it. A group is `group_size` rollouts of the same task. Each
optimizer update trains on `groups_per_step` groups.

```
                 ┌────────────────────────────────────────────────────────────┐
                 │  Pipeline: admit groups, schedule sampling and grading     │
                 └────────────────────────────────────────────────────────────┘
  task stream ──▶ group of N rollouts ──▶ agent runs ──▶ verifier ──▶ graded group
                                            │                              │
                                   generate() calls                fresh & varied?
                                            │                              │
                                     renderer → tokens                     ▼
                                     Loops sampler ◀──── publish ◀── optimizer step ◀── datums
                                     exact token capture                      ▲
                                                                     advantages = r − mean(r)
```

The cookbook computes advantages in the style of group relative policy optimization
(GRPO), within each group. The model therefore learns which of its attempts at a task
did better than the others, and it never learns from the absolute reward.

## From a rollout to training rows

Each model call in a rollout produces one sample. A sample holds the prompt tokens the
sampler received, the completion tokens it returned and their logprobs. Each call
names a producer, which is `main` unless the call names another, and
`[rollout.producers]` declares whether each producer's samples are trainable.

When grading finishes, the cookbook arranges the rollout's samples into **chains**. A
sample joins a chain when its prompt is exactly the previous sample's prompt plus its
completion, possibly followed by new tokens. A plain tool-calling agent produces one
chain. An agent that compacts or branches its history produces several.

Each chain that holds at least one trainable sample becomes one **training row**. A
training row is a Loops datum together with the positions it trains:

- `model_input` is the chain's final prompt plus its completion, without the last
  token. The last token is only ever a target, never an input.
- `target_tokens` holds the next token at every position, including the prompt and the
  tool results. An image position has no next token, so its target is `-100`. Router
  replay checks every target against the sampler's record, so the cookbook can't mask
  a position out through its target.
- The row's **actions** are the positions inside a trainable completion. They are the
  agent's own tokens, and they are the only positions a built-in loss trains. Prompt
  tokens, tool results and copied history carry no advantage, so the model never
  trains on them, unless a [custom loss](#the-loss) trains on tool results.
- `advantages` holds the rollout's advantage at the actions and 0 everywhere else.
- `logprobs` holds the behavior logprobs, which are the logprobs the sampler reported.
  The loss uses them for the importance ratio.

When `[model] max_seq_len` is set, it bounds the length of each datum. A longer chain
fails the run with a message that names its length, because truncating the chain
would misalign its targets. When `max_seq_len` is unset, the trainer's own limit
applies.

**The model check.** `loops check` asks Loops whether it serves `[model] base_model`.
It makes one read-only request, `GET /v1/loops/capabilities`. It refuses the config in
three cases:

- Loops doesn't list the model. The message names the closest model that Loops does
  list.
- Loops lists the model but hasn't enabled it.
- `max_seq_len` is above what Loops enables for the model.

When the model passes, `loops check` prints
`Loops model: <name>; max_seq_len <n> (enabled up to <m>)`. Sometimes the check can't
run, because `BASETEN_API_KEY` isn't set, Loops can't be reached in 15 seconds, or Loops
answers with anything other than 200 or a rejected key. Then `loops check` prints that
the model was not checked and why, and it passes, so you can still check a config
offline. A key that Loops rejects with 401 or 403 is refused, because `loops run` would
fail on it too. `loops run` makes the same check before it provisions the trainer.

**Router replay (R3).** For a mixture-of-experts model, router replay lets the trainer
replay the exact expert routes that the sampler used. It is off by default. Set
`router_replay = true` to turn it on. The sampler then records every generation, and
each datum names, for every position, the recorded generation that the position came
from. Loops calls this token provenance.

A generation's own tokens take that generation's routes. The sampler never ran the
tool results after a generation, or the generation's last token, through the model.
Those positions take the routes of the next generation's prompt instead. So a
multi-turn chain replays every turn, including turns sampled under different policies,
and each fork of a branching rollout replays its own chain. One position takes one
route, so an earlier turn keeps the routes it was sampled with, even where a later
policy would have prefilled it differently.

Within one optimizer step, replay is all or nothing. If any row in the batch was not
recorded, the cookbook strips the provenance from all of them, and
`r3/unrecorded_datums` counts the rows that weren't recorded.

Loops records text prompts only. A rollout runtime whose prompts carry images sets
`[rollout_runtime] has_images = true`. A run without that setting refuses every request
that has an image. `loops check` refuses `has_images = true` together with
`router_replay = true`.

## Groups, advantages, and what gets skipped

The run shuffles the tasks once per epoch, seeded by `shuffle_seed`, and starts a group
for each task in turn. A rollout resolves when it is graded, fails or is discarded.
When all `group_size` rollouts of a group have resolved, the cookbook checks the group
in this order:

1. It drops stale rollouts. This repeats the `max_steps_off_policy` check that also
   runs before grading.
2. It rejects the group as `group_too_small` if more than `max_group_losses` of its
   rollouts did not come back graded.
3. It rejects the group as `no_variance` if every remaining rollout has the same
   reward.
4. It rejects the group as `nothing_to_train` if every remaining rollout with a nonzero
   advantage trains nothing, because its agent discarded every generation. These
   rollouts still count in the group's mean reward, and
   `trajectory/untrained_rollouts` counts them at each update.

A rollout runtime can have a group grader, which scores a whole group in one call. An
`Episode` that defines `grade_group()` has one. The group grader runs between the size
check and the variance check. Once the stale rollouts are dropped and the group is
still large enough to train, one `grade_group()` call gives every member its final
reward. So the cookbook never pays a judge to score a group that is about to be
dropped. See [episode-api.md](episode-api.md#group-grading).

Each rollout in a surviving group gets `advantage = reward − mean(group rewards)`. The
cookbook doesn't divide the advantage by the group's standard deviation.

If `max_consecutive_unusable_groups` groups in a row are rejected, the run aborts. A
usable group resets the count. This setting exists to stop a run that is spending
sandboxes without learning. It is not a budget.

Rollouts that fail are counted separately from rejected groups. An agent that fails is
still graded, whichever rollout runtime runs it. The grader decides what the failure
scores, and the cookbook never does. Harbor still runs the trial's verifier, as Harbor
itself does. A custom rollout runtime returns from `sample()` for every episode that
happened, however badly the agent did, and `grade()` scores what the agent left.

When `sample()` raises, it means the episode didn't happen. The cookbook retries the
rollout up to `max_rollout_retries` times. After that, it discards the rollout as
`rollout_failed/<type>` and counts it toward `max_consecutive_rollout_failures`. A
`PermanentRolloutFailure` is discarded without a retry.

The cookbook decides one agent outcome itself, which is the rollout clock. A rollout
that runs out of `rollout_timeout_seconds` has no episode to grade. The cookbook
discards it as `agent_timeout/rollout_clock`, doesn't retry it, and counts it as a
sampling failure. A rollout that comes back with no trainable samples is treated as a
failure too, because it means the agent never called the policy.
[failures.md](failures.md) has the whole contract, with the retries, deadlines and
failure counts of each stage.

### Metrics for groups, freshness and mismatch

Rejected groups never reach `reward/mean`, which averages the rollouts that an update
trains on. So when the policy starts solving, or failing, whole groups outright,
`reward/mean` moves less than the real reward does. Each update therefore also logs
`reward/mean_with_unusable`. It averages the update's trained rollouts together with
every graded member of the groups rejected as `group_too_small`, `no_variance` or
`nothing_to_train` since the previous update. `reward/unusable_rollouts` counts the
members it added. Each numeric `rollout/<key>` metric is reported the same way, as
`rollout/<key>_with_unusable`. The run reports all of these at every update, whether or
not a group was rejected.

`reward/std` is the spread of rewards over the whole batch, so it reads high even when
the members of each group scored alike. GRPO learns from the spread within each group.
Each update reports that spread, over the groups it trains, as
`reward/group_std_mean`, `reward/group_std_min` and `reward/group_std_max`.

Two more sets of metrics say how fresh a batch is, and why:

- `freshness/lag_mean` and `freshness/lag_p90` describe how many updates the oldest
  generation of each trained rollout was behind the trainer. `freshness/lag_max` shows
  only the worst rollout.
- `groups/completion_spread_mean_seconds` and `groups/completion_spread_max_seconds`
  give, for each trained group, the time from its first member resolving to its last.
  The group's first finishers age for that long while they wait for the rest. A wide
  spread means that the run's rollouts are spread over many groups.

With a built-in loss, the trainer returns more metrics than `train/mismatch_kl`, and the
run reports each one as `train/<name>`. Examples are `train/mismatch_abs_logprob_diff`
and `train/mismatch_ess`, which is the normalized effective sample size of the
importance weights. The report of every `forward_backward` call in `steps.jsonl` keeps
all of them.

The `mismatch/*` metrics measure the same disagreement between the trainer and the
sampler on the tokens that train. The run computes them from the logprobs that the
trainer returns for every datum, so they exist for a built-in loss and for a custom
one. `mismatch/abs_logprob_diff` is the mean over tokens of |trainer − sampler|
logprob, and `mismatch/abs_logprob_diff_max` is the largest. The run also splits the
mean in two ways:

- By the action's place in its row, as `/turn_1`, `/turn_2`, `/turn_3-4`, `/turn_5-8`,
  `/turn_9-16` and `/turn_17+`. With router replay on, late turns should read like
  early ones.
- By how many updates the trainer's weights are past the policy that sampled the
  action, as `/lag_0`, `/lag_1` and `/lag_2+`. At lag 0, the gap comes from numerics
  and routing alone. From lag 1, it also includes the policy's own change.

A trainer logprob that isn't finite on a trained token is left out of those means and
counted in `mismatch/nonfinite_tokens`. If the trainer returns no logprobs, the run
leaves out `mismatch/*` and logs one warning. If the logprobs don't fit the rows,
because a datum's logprobs are missing or have the wrong length, the step fails. Loops
already holds that call's gradients, so the cookbook refuses the update.

## The training step

Once the run has selected `groups_per_step` usable groups, it rechecks their freshness
one last time. Then it trains on them in these steps:

1. With `[output] trace = "full"`, the run writes the rows, each a datum and its
   actions, to `batches/step-N.json` before it submits them.
2. `forward_backward` accumulates the gradients of the run's [loss](#the-loss). The SDK
   may split a large batch into several calls. Loops accumulates the gradients until
   the optimizer step and applies the update's reduction there, so the split makes no
   difference. See [loss aggregation](#loss-aggregation).
3. `optim_step` applies Adam with `learning_rate`, `grad_clip_norm`, `adam_epsilon`,
   `adam_beta1`, `adam_beta2` and `weight_decay`. Only `learning_rate` is required.
   When you leave one of the others unset, the request carries the Loops SDK's
   default. A `grad_clip_norm` of 0 turns clipping off. A non-finite loss aborts the
   step before this point. It also poisons the trainer, so nothing further is
   published from it.
4. The policy version advances with the optimizer step. If it doesn't advance, the
   trainer no longer holds the run's earlier updates, and the run refuses to
   continue. The run then publishes the weights to the paired sampler.
5. Every `sampler_checkpoint_every` updates, the run stages the sampler weights under
   the name `step-N`. Every `trainer_checkpoint_every` updates, it also saves the full
   trainer state, including the optimizer moments, under the same name. It then
   writes a resume manifest to `checkpoints/step-N.json`. The update at `max_steps`
   always writes both.
6. The run appends one record to `steps.jsonl` with everything the step reported.

A run can also end in other ways. It can be interrupted with Ctrl-C or by the SIGTERM
that a scheduler sends, it can abort or fail, or its Loops run can end. When that
happens, the run saves the trainer state of its last applied update on the way out, if
the cadence skipped that update. That manifest is what
[`resume_from`](#continue-or-fork-a-run) takes.

Don't let a second process drive the same trainer. Nothing in the cookbook stops it.

By default, the run leaves its Loops run up on the way out. On dedicated Loops, a
later run's `[compute] reuse_from_run_id` can then start warm on it. The run logs a
warning with the Loops run's ID. The Loops run stays up until you run
`loops deactivate <run id or run directory>`. A dedicated run holds its pods and bills
until then. A serverless run has no deployment of its own and costs nothing while it
waits, but it holds one of the workspace's serverless slots.

To end the Loops run on the way out, set `[compute] keep_run = false`. The run then
deactivates it however the run ends, whether it completed, was interrupted, aborted or
failed. The checkpoints stay. If Loops doesn't confirm the deactivation within 30
seconds, the run logs the same warning.

## Continue or fork a run

Two settings load trainer state that an earlier run saved. Both read the same
`bt://loops:<run>/weights/step-N` trainer state. They differ in what continues after
the load.

| | `resume_from` | `init_from` |
|---|---|---|
| What it's for | It continues a run that stopped, or gives a run more steps. | It starts a new run from trained weights. |
| What it takes | It takes a local `checkpoints/step-N.json` manifest, or the run directory. | It takes a `bt://loops:<run>/weights/step-N` path from any run you can read. |
| What it loads | It loads the weights, the optimizer state and the run's position. | It loads the weights with fresh Adam moments. `init_from_optimizer` adds the optimizer state. |
| Steps | The steps continue from the manifest's next step. | The steps start at 0. |
| Run directory | The run keeps the same run directory. | The run gets a new run directory. |
| Config changes | A changed `base_model` or `lora_rank` is refused. Other changes are logged as a warning. | You can change the data, the loss, the learning rate or the recipe. |

You can set each one as a TOML value in `[training]` or as a flag on `loops run`.
`--resume` is the short form of `resume_from` for the latest run of the same config:

```sh
# Continue the latest run of train.toml, from the manifest it wrote last.
uv run loops run train.toml --resume
# Continue it until the run has made 30 updates in all.
uv run loops run train.toml --resume --training.max_steps 30
# Continue one run from one manifest: an older run, or an earlier step.
uv run loops run train.toml \
  --training.resume_from runs/<run_name>-<time>-<id>/checkpoints/step-9.json
# Fork: RL from the weights an SFT run kept.
uv run loops run train.toml --training.init_from bt://loops:<run>/weights/step-7
```

`--resume` takes the latest run, which is the `<output_dir>/<run_name>-<time>-<id>`
directory that trained most recently. The cookbook judges this by the times that the
run recorded for its last update and its manifests. It doesn't use the time the run
started, or file times, which a copy resets. So a run that was resumed today is later
than one that was started yesterday. `--resume` logs which run it took. When no run has
trained, or the latest run kept no manifest, it exits with code 2 and starts nothing.
It never takes an older run instead. You can't combine `--resume` with `resume_from`.
`[output] run_name` is required, so that two configs in one `output_dir` never take
each other's runs.

The log names each trainer state as the run keeps it, in the form
`remote trainer state: bt://loops:<run>/weights/step-N`. A manifest holds the same
path as `trainer_state_path`. An `init_from` source must have the run's `base_model`
and `lora_rank`. It can come from the other recipe, so an RL run can start from the
weights of a supervised fine-tuning (SFT) run.

### `resume_from`

A resume restores the trainer state, the step counter and the task cursor.
`max_steps` counts all of the run's updates, including the ones made before the
resume. So a resume from step 10 with `max_steps = 30` makes 20 more updates. This is
the meaning Tinker gives `max_steps`. A resume that has no update left is checked like
any other resume, and then it completes with exit code 0. It logs that the run already
made every update, and it starts no trainer. To train more, raise `max_steps`.
`loops check` reports the same thing.

The resumed run writes into the manifest's run directory, or into the run directory
you gave. It refuses a changed `base_model` or `lora_rank`, because the saved weights
can't load into a different base model or rank. Any other change to the settings that
shape training, such as the batch shape, the producers, the renderer, the sampling
settings or the rollout runtime's configuration and files, is allowed. The run logs a
warning that names each changed setting, because the two halves of the run are no
longer comparable.

The task cursor counts groups through one ordering of one task list. If you change
`shuffle_seed` or the task set, the cursor points nowhere, so the run logs a warning
and starts the task order again from the beginning. Otherwise, the resume first runs
the groups that the checkpoint left untrained, in task order. These are the groups
that were in flight for the next update when the run stopped, and the groups whose
update was aborted. They start again from scratch, because the rollouts that were in
flight are not kept. Groups that were dropped as unusable, as stale, or because their
grading failed don't run again. After those groups, the resume continues from the
cursor.

You can resume from an older manifest, one from before the last manifest the run
wrote. If that resume stops before it writes its own first manifest, the later
manifests are still the ones written last. So `--resume`, or a `resume_from` that names
the run directory, takes the latest of them. To stay on the earlier step, resume from
that manifest file again.

### `init_from`

`init_from` loads the low-rank adaptation (LoRA) adapter at step 0 of a new run
directory. It doesn't reset the trainer's policy version. The version continues from
the checkpoint's, so it is offset from the run's step count. If you later resume that
run, the resume restores the run's own state and doesn't load the `init_from` weights
again.

`init_from_optimizer = true` also loads the source's optimizer state, which is its Adam
moments and its Adam step. Use it for a fork that keeps the source run's optimizer
under a new config. Those moments reflect the source run's gradient scale. A fork that
changes what produces the gradients, such as the loss, the data or the batch size,
starts with moments sized for the old gradients. From a checkpoint that was saved
without moments, Loops loads only the weights. `init_from_optimizer` requires
`init_from`. A resume ignores it, as it ignores `init_from`.

`init_from` with `init_from_optimizer = true` loads the same state as a resume. Unlike
a resume, it doesn't keep the step count, the task cursor or the run directory, and it
runs none of the resume checks. Use `resume_from` to continue a run, and `init_from`
to start a new one.

## Evals

An eval scores the policy on held-out tasks while it trains. Each eval set runs
training's own rollout runtime, with the same factory and `[rollout_runtime.config]`,
on other files. The agent, the grader and the sandboxes are training's. The only thing
an eval set requires is its tasks:

```toml
[eval]
every_steps = 10

[eval.sets.val]
samples_per_task = 4              # optional; 1 by default
max_concurrent_rollouts = 16      # this set's rollouts at once

[eval.sets.val.rollout_runtime.files]     # replaces these [rollout_runtime.files] entries
tasks = "val.jsonl"

[eval.sets.val.sampling]          # optional; overrides [sampling] for this set
temperature = 0.6
```

A key at the path of a training setting overrides that setting for the set. In this
example those keys are `rollout_runtime.files` and `sampling.temperature`. The other
keys belong to the set alone.

The run evaluates at three points:

| Eval | When | Weights | Logged at `train/update` |
|---|---|---|---|
| Base | After the trainer starts, on a run with neither `resume_from` nor `init_from` | The base model. | `-1` |
| Step | After update N, when `(N + 1) % every_steps == 0` | The update's sampler weights. | N |
| Final | After the last update, which needs a nonzero `max_steps` | The update's sampler weights. | N |

With `every_steps = 0`, the run evaluates only the base model and the last update.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/eval-timeline-dark.svg">
  <img alt="When evals run, with every_steps = 2 and max_steps = 6. Training runs updates 0 to 5 without waiting; steps 1, 3 and 5, which an eval is due for, save their weights at the end of the update. On dedicated Loops each eval starts a sampler deployment, then runs its rollouts: the base eval is still running when step 1's eval comes due, so step 1 is skipped; step 3's eval runs; the final step 5 eval comes due while step 3's still runs, waits for it, then starts its own sampler; the run ends when the final eval does. Step 1 still keeps its weights. On serverless Loops the sampler is ready at once and a set's evals overlap: the base, step 1, step 3 and final evals each run when due, step 1's beside the base eval and the final beside step 3's, and the run ends sooner." src="../assets/eval-timeline.svg">
</picture>

Each eval set gets a sampler of its own. That sampler is pinned to the evaluated
weights, and it is never the run's paired sampler. Training doesn't wait for an eval.
The only part of an eval that sits on the training loop is the save of that step's
sampler weights, which shows in the step's `checkpoint` time. The sets of one step run
at the same time, each on its own sampler.

On dedicated Loops, each eval starts a sampler deployment, so a set runs one eval at a
time. If an eval comes due while the set's previous eval still runs, the run skips it
with a warning. The step keeps its weights all the same, so `loops eval` can run that
eval later. See [Evaluating after the fact](#evaluating-after-the-fact). The final eval
waits for the set's previous eval to finish, then runs.

On serverless Loops, a sampler costs nothing while it waits. So a set's evals overlap,
every eval that comes due runs, and each one has its own `max_concurrent_rollouts`.
Evals sample from the same serverless capacity as training, so in an RL run,
overlapping evals can slow the updates.

A set's rollouts keep `[rollout] rollout_timeout_seconds`. When that setting is unset,
they keep the clock that their own rollout runtime derives. A run that completes waits
for its evals, and its trainer and paired sampler stay up and keep billing until the
last eval ends. A run that stops cancels its evals. Either way, the run deactivates
every eval sampler. Nothing an eval samples trains.

Each set writes `eval/<set>/base.jsonl` and `eval/<set>/step-N.jsonl` in the run
directory, with one row per rollout. A row holds `task_id`, `task_digest`, `sample`,
`rollout_id`, `reward`, `discarded` and `metadata`. To compare steps, join the files on
the task. Beside each file is `base.transcripts.jsonl` or `step-N.transcripts.jsonl`.
It holds every rollout's conversation, meaning what the model was asked and what it
replied, and the run writes it whatever `[output] trace` says.

W&B gets these metrics for each set:

- `eval/<set>/reward_mean`, `reward_std`, `rollouts`, `graded`, `discarded/<reason>`,
  `sampler_start_seconds` and `seconds`.
- `eval/<set>/runtime/...`, which is what the set's episodes report during that eval.
  It is never mixed into training's `runtime/...` or into another eval's.
- `eval/<set>/set/runtime/...`, which is what the set's rollout runtime reports
  through its own context, outside any episode. It is one count per set, which adds
  up across the set's evals so far.

When something goes wrong, the eval reacts as this table shows:

| What happens | What the eval does |
|---|---|
| Its sampler isn't ready within `sampler_start_timeout_seconds`. | The run skips the eval, with a message and no file. Training continues. |
| The trainer names no path for the step's weights. | The run skips the eval, with a message and no file. Training continues. |
| A rollout fails past its retries. | The rollout's row has `discarded` set and no reward, and the mean leaves it out. |
| Failures in a row reach `max_consecutive_rollout_failures`. | That eval stops, with no file. Training continues. |
| Hooks that ignored cancellation hold every one of the set's `max_concurrent_rollouts` slots. | That eval stops, or is skipped before its sampler starts, with no file. Training continues. |
| The run stops in the middle of the eval. | The run deactivates the eval's sampler and writes no file, and the log says that the eval didn't finish. A resume doesn't run it again. |

`loops check` builds each eval set's rollout runtime and prints a line for it. It warns
about tasks that an eval set shares with training, matched by ID. It doesn't catch
the same task under another ID. The check refuses a set whose rollout runtime
reaches the model in a different way from training's, because one gateway serves both.
A rollout runtime with a group grader scores each task's samples together.

On serverless Loops, an eval's sampler shares replicas with training, and it is ready
at once. On dedicated Loops, each eval set starts a sampler deployment for every eval.
That deployment needs free allocation for the model at `max_seq_len`, takes time to
start, and bills until it is deactivated. If a rollout runtime's training and eval
instances lease the same run-scoped resource, the second lease fails.

Eval rollouts use training's sandbox providers and their quotas. A set's
`max_concurrent_rollouts` adds to training's. So during an eval, a provider can see
`[rollout] max_concurrent_rollouts` plus each set's limit at once. On serverless Loops,
it sees each set's limit again for every eval of that set that overlaps. Size these
limits to what the provider allows. Grading is bounded the same way. Each eval grades
at most `[rollout] max_concurrent_verifiers` rollouts at once, as training does, so a
judge's rate limit sees that many from training and that many from each eval that is
running.

### Evaluating after the fact

```bash
uv run loops eval train.toml --step 29             # update 29 of the latest run
uv run loops eval train.toml --step base --set val # the base model, one set
uv run loops eval train.toml --step 29 --run runs/sat/sat-1793877514-b36ce057
```

`loops eval` runs the config's eval sets on one update of a run, whether the run has
ended or is still training. It works as an eval during the run would. It writes
`eval/<set>/step-N.jsonl` and its transcripts into that run's directory, replacing any
eval of that step that is already there, and it logs the same line. Use it for an eval
that was skipped or failed, or for a new eval set on a finished run. It takes the latest
run of the config, as `--resume` does, or the run that `--run` names.

The weights are the ones the run kept for that update. The run keeps sampler weights
for every eval that comes due, including evals that were skipped or failed, and every
`sampler_checkpoint_every` updates. It records each set of weights in
`sampler_weights.jsonl`, and `loops eval` serves those. For an update that has only
trainer state, which the run keeps every `trainer_checkpoint_every` updates,
`loops eval` starts a trainer that loads that state and saves it for a sampler. It then
deactivates that trainer's Loops run. For any other update, it says which updates it
can evaluate. It exits with code 1 when a set didn't evaluate, and the log says why.

## The loss

`[training] loss_fn` chooses what `forward_backward` computes. It is required, and the
names are the SDK's. An action token is a token that the policy generated in a
trainable sample. The cookbook computes the group-centered advantages the same way
whatever the loss. [Loss aggregation](#loss-aggregation) then controls how much each
action token contributes to the optimizer update.

- `cispo`, `importance_sampling`, `ppo`, `dro` and `dppo` are Loops' policy losses.
  Each one runs over the recipe's datums as they are. The cookbook hands
  `loss_fn_config` to Loops unchanged, and a key you leave unset keeps Loops' default:

  | `loss_fn` | Its `loss_fn_config` keys, with Loops' defaults |
  | --- | --- |
  | `importance_sampling` | It takes no keys. The loss is `−ratio · advantage`. |
  | `ppo`, `cispo` | `clip_low_threshold` (0.8) and `clip_high_threshold` (1.2). The ratio is clipped to [low, high]. |
  | `dro` | `beta` (0.05) is the weight of the `log_ratio²` penalty. |
  | `dppo` | `dppo_mask_high` (0.2), `dppo_mask_low` (0.2), `adv_tau` (1.0) and `kl_tau` (0.001). |

  `loops check` holds each key to the bounds that the Loops trainer applies. It also
  refuses a key that the loss doesn't take, which the trainer would otherwise ignore.
  Loops also has `dpo`, which trains on pairs against a reference, and
  `cross_entropy`, which is for SFT. They train other kinds of data.
- `custom` runs your own loss through Loops' `forward_backward_custom`. In
  `[training.custom_loss]`, name a `factory` as `module:callable` and give its
  `config`. A module beside the config is importable. The cookbook hands `config` to
  the factory unchanged, and the factory returns the loss function. `loops check`
  calls the factory, so a bad factory or config fails there. A custom loss needs the
  `custom-loss` extra, which installs PyTorch.

```python
def create_loss(config):
    def fn(datums, logprobs):
        loss = sum(
            -(lp * d.loss_fn_inputs["actions"].to_torch()
              * d.loss_fn_inputs["advantages"].to_torch()).sum()
            for d, lp in zip(datums, logprobs, strict=True)
        )
        return loss, {}  # a scalar tensor, and float metrics logged as loss/<name>

    return fn
```

Loops scores every text position of a datum, including the prompt and the tool results
between turns. `fn` receives the trainer's logprobs at those positions, one tensor per
datum. Each datum's `target_tokens` is the next token at every position. At an image,
the target is `-100` and the logprob is zero. The `actions` input marks the agent's own
tokens, which are the ones that `advantages` and the sampler's `logprobs` describe. A
loss that trains on tool results, but not on the task's prompt, takes the positions
after the first action.

Loops sums the gradients without normalizing by token count, so `fn` must normalize as
you intend. The runner calls `fn` once per `forward_backward`, over the whole batch it
submits. For a custom loss, `train/loss` is the value `fn` returns, and
`train/mismatch_kl` is not reported. [ECHO](../../examples/terminal_bench/echo_loss.py)
is a worked example, which adds a next-token loss on the tool results.

### Loss aggregation

Loss aggregation chooses how the loss is reduced over one optimizer update. You choose
it independently of the loss function:

```toml
[training]
loss_fn = "cispo"
loss_aggregation = "sum"
```

`cispo` uses Loops' clipping defaults, 0.8 and 1.2, unless you set `loss_fn_config`. For
the plain importance-sampling objective, set `loss_fn = "importance_sampling"`.

In the table below, `S_g` is the sum of the policy loss over the action tokens of task
group `g`. A task group holds the repeated episodes of one task, and a task admitted
again forms a new group. Prompt and tool-result tokens are not part of `S_g`.

| `loss_aggregation` | Optimizer objective | Meaning |
| --- | --- | --- |
| `sum` (default) | `sum(S_g)` | This is Loops' own default. Nothing is divided, as when you call the SDK directly. |
| `token_mean` | `sum(S_g) / N` | `N` is the update's count of action tokens **with nonzero float32 advantages**. Loops divides the whole loss by it, penalty terms included. |
| `fixed_budget` | `sum(S_g) / (group_size * groups_per_step * loss_token_budget)` | The denominator is a fixed number from the config. It doesn't depend on how long the episodes were or on how many group members were lost. |
| `task_mean` | `mean_g(S_g / T_g)` | Each selected task group gets equal weight. `T_g` counts all of the group's action tokens, including those with zero advantage. |

For a fixed denominator, give an explicit budget that covers the **whole episode**:

```toml
[training]
loss_fn = "cispo"
loss_aggregation = "fixed_budget"
loss_token_budget = 8192
```

The budget is a normalization constant, not a generation limit. It covers every turn
and every trainable producer in an episode. The cookbook never infers it from
`max_generation_tokens` or from the context window. A larger denominator makes the
gradient smaller, so compare learning rates when you change the aggregation.

An agent that compacts or branches its history splits a rollout into several
[chains](#from-a-rollout-to-training-rows), and each chain becomes its own datum. That
adds rows but not trained tokens. Each generated token is a target in exactly one
chain, and the history that a new chain copies is prompt in that chain. So the budget
counts each generated token once, however many chains the rollout has. To choose a
budget, run a few steps and read `trajectory/sampled_tokens_median` and
`trajectory/sampled_tokens_max` in `steps.jsonl`. They count the tokens each rollout
generated across all its chains, including tokens from producers that don't train.

The reduction happens in `rl.aggregation.aggregate_loss(datums_by_group, training)`. It
runs after freshness selection and the group advantage calculation, and before the
rows are recorded or submitted.

- `sum` and `token_mean` are Loops' own reductions. The rows go to Loops unchanged, and
  every built-in `forward_backward` call states which reduction it wants, through
  `skip_token_normalization`.
- `fixed_budget` and `task_mean` keep Loops' sum and multiply the advantages by
  `1 / D`. This works because the `importance_sampling`, `ppo` and `cispo` losses are
  linear in positively scaled advantages. For `fixed_budget`, `D` is the denominator in
  the table above. For `task_mean`, `D` is the number of groups in the update times the
  group's `T_g`, so it differs from group to group.

The prepared rows can be split across several `forward_backward` calls. All of them
must share one optimizer update, and they must not be mixed with rows from another
preparation. Splitting the rows for transport never changes the counts used for the
reduction. Before submission, the cookbook rejects any scaling that would change which
tokens have nonzero float32 advantages, or that would produce non-finite advantages.

`fixed_budget` and `task_mean` work with `importance_sampling`, `ppo` and `cispo`. The
`dro` and `dppo` losses have penalty terms that scaling the advantages wouldn't
reweight. So they take `sum` or `token_mean`, which divide the whole loss, penalty terms
included. A custom loss receives the original advantages and reduces them inside its own
function. Loops sums what the custom loss returns, so a custom loss keeps `sum`, and the
cookbook refuses any other mode. Loss aggregation doesn't change how advantages are
computed. Each advantage is still the reward minus the group mean, with no division
by the standard deviation.

With `trace = "full"`, the batch record holds the aggregation mode, the counts, the
denominators and the scales, beside the exact weighted rows sent to Loops. Step
metrics under `aggregation/` appear only for the modes in which the cookbook divides.
`fixed_budget` reports `aggregation/denominator`, and `task_mean` reports
`aggregation/group_denominator_min` and `aggregation/group_denominator_max`. The token
counts are `train/trained_positions` and Loops' own `optim/num_loss_tokens`.
`train/loss` is still the loss the SDK reports. It is not a reconstruction of the
optimizer objective after the global reduction.

### Trainer normalization contract

The built-in aggregation modes depend on one behavior of the Loops trainer, which the
cookbook names `loops.rl.sum_or_active_token_mean.v2`. Loops sums gradients across
backward calls. With `skip_token_normalization = false`, it divides once, at the
optimizer step, by the targeted tokens with nonzero float32 advantages. The cookbook
states the setting on every built-in `forward_backward` call, and it refuses a result
whose trainer doesn't echo the setting back. A trainer from before Loops summed by
default rejects the setting. The name is the cookbook's identifier for this
requirement, and the trainer doesn't advertise it as a capability. With
`trace = "full"`, the batch record includes `backend_normalization` beside the expected
count, and the optimizer reports its actual count as `optim/num_loss_tokens`.

Before a release, or after the trainer or the SDK changes, trigger the **trainer
normalization contract** workflow in GitHub Actions. It is a billable check that runs
only on demand. Ordinary PR CI and publishing don't launch it, and they don't require
it to pass.

The check keeps the weights fixed by running optimizer steps with a learning rate of
zero. It compares the built-in `cispo` loss against an explicitly reduced custom
objective for all four modes. It also compares the built-in `dro` and `dppo` losses
under `token_mean`, with each penalty divided like its advantage term. It checks the
gradient norms, and it checks that the logprobs don't change.

Split `forward_backward` calls move the trainer's logprobs, because a row's logprobs
depend on its neighbours in the batch. The check records how far they move. It also
holds two calls of one batch to within 0.1% of a single call of the doubled objective,
so it checks accumulation separately from that drift. It measures the norms before
Adam, so the optimizer moments don't affect the comparison.

With matching transport chunks, the custom objective and the built-in loss must agree
within 0.01%, for both on-policy and clipped importance weights. Where Loops averages,
the custom objective runs undivided, and the check divides its norm afterwards, as
Loops divides the accumulated gradient. For clipping, the check holds the importance
weights fixed, to isolate differences in forward scoring. Zero-advantage tokens,
ignored targets and split backward calls test the denominator and the accumulation.
The check runs on one trainer, with no sampler or sandbox, and it deactivates the run
it owns during teardown.

The workflow needs the `BASETEN_API_KEY` secret in GitHub Actions, and it fails when
the secret is missing. You can also run the check locally, with credentials in `.env`:

```sh
LOOPS_LIVE_LOOPS=1 uv run --env-file .env --extra custom-loss pytest -q -s \
  tests/live/test_trainer_normalization.py --basetemp=trainer-contract-results
```

`LOOPS_CONTRACT_MODEL` selects another supported Loops model. The default is
`Qwen/Qwen3.5-4B`. The workflow keeps the numerical receipt and the cleanup result as
an artifact. A passing receipt qualifies only the model and trainer deployment it
tested. Run the check again after backend changes, and qualify other backends and
model families before you assume that they behave the same.

## Outputs

Each run writes its record to its own run directory:

```
runs/<run_name>-<time>-<id>/
  config.json               settings as run
  runtime_evidence_digests.json
                            digests of what the runtime made of its settings
                            (an episode's validated options, defaults included)
  samples/*.jsonl           with any trace but "off": one file per attempt,
                            one line per generation
  batches/step-N.json       with trace = "full": rows as submitted, each a datum
                            and its actions
  transcripts/*.json        with trace = "transcripts" or "full": one file per
                            attempt, failed ones as far as they got, one turn per
                            request, as the agent sent and received it
  rollouts.jsonl            every rollout's disposition; trained ones → their step;
                            what its attempt sampled but left out (below)
  steps.jsonl               per-update metrics
  checkpoints/step-N.json   resume manifests
  sampler_weights.jsonl     each update's kept sampler weights, for `loops eval`
  eval/<set>/               each eval's rows and transcripts (see Evals)
  run.json                  the Loops run ID and URL
  failure.json              for a run that failed: how it ended, with the traceback
  run.log, progress.jsonl   the run's log and its progress records
  state/resources/v1/       provider resource journal (see Provider resources)
  runtime/                  the runtime's work dir: Harbor trials under runtime/trials/,
                            Episode attempts under runtime/attempts/
```

Every update writes its metrics to `steps.jsonl`. With the `wandb` extra, they also go
to Weights & Biases (W&B). The metrics come in these families:

| Family | What it reports |
| --- | --- |
| `reward/*`, `advantage/*` | The rewards and advantages of each update. |
| `train/*` | The loss, the tokens, the trained positions and the policy version. |
| `freshness/*` | The lag and the version span of the batch. |
| `trajectory/*` | The generations and tokens per rollout. `trajectory/forked_rollouts` counts rollouts that forked. `trajectory/mixed_policy_rollouts` counts rollouts whose trainable generations came from more than one policy version. |
| `sampling/stop/<reason>`, `sampling/entropy_estimate` | Why generations stopped, and an estimate of the sampled tokens' entropy. |
| `optim/*` | The gradient norm, the step time and the throughput, when the SDK returns them. |
| `loss/*` | A custom loss's own metrics. |
| `mismatch/*` | The disagreement between the trainer and the sampler on the tokens that train, as described [above](#metrics-for-groups-freshness-and-mismatch). |
| `time/*` | How long each stage took. |
| `slots/*`, `queue/*` | How full the slots and queues are. |
| `admission/*` | The async window, the groups and rollouts committed to it, and the share of groups that survive. |
| `discard/rollouts/<reason>`, `discard/groups/<reason>` | The rollouts and groups that were discarded, by reason. |
| `discard/generations`, `discard/generation_tokens` | Every generation that was sampled and then left out of the evidence. Batch metrics don't count these, because batch metrics describe what survived. |
| `discard/undelivered_generations`, `discard/undelivered_generation_tokens` | The part of those generations that the cookbook dropped when it sealed the attempt, because the agent never received them. The agent discarded the rest itself. |
| `trace/unwritten_samples`, `trace/unwritten_transcripts` | Trace records that could not be written. The run logs a failed write, and it never fails the attempt. |
| `rollout/*` | Your agent's own metrics, as described below. |
| `r3/*` | Router replay. `r3/routed_datums` counts the rows replayed, and `r3/unrecorded_datums` counts the rows that took a batch off replay. |
| `pipeline/*`, `groups/*`, `progress/*`, `retried/*`, `attempts/live`, `pipeline_calls/*` | The state of the pipeline. |

`train/mismatch_kl` measures how much the sampler and the trainer disagree. The run
logs it and never uses it to gate anything.

`rollouts.jsonl` holds one record for every rollout, trained or discarded. Every record
has the same fields, and each field means the same thing in every record:

- `policy_version` is the oldest policy version among the rollout's trainable samples.
  It is `null` when the rollout sealed no attempt.
- `task_digest` is the rollout runtime's digest of the task as the attempt ran it. It
  is `null` when the rollout runtime reported none.
- `sample_ids` names what the rollout sampled. It is empty when the rollout sealed no
  attempt.
- `step` is the update that trained the rollout. It is `null` for a discarded rollout.
- `loops_rollout_id` is the ID that Loops traced each of the rollout's sampling
  requests under. Every attempt at the rollout sent it. Use it to find the record of a
  rollout that you saw in a Loops trace.
- `generations_undelivered` and `generations_discarded_by_agent` describe the attempt
  that the record names. The first counts the generations the cookbook dropped at
  sealing because the agent never received them, and the second counts the ones the
  agent discarded itself. A record that names no attempt sealed none, and both counts
  are zero.

The `rollout/*` family is yours. It comes from an `Episode`'s `run()` metadata, or from
whatever a Harbor agent puts in `context.metadata`, with the grader's metadata merged
over it. The run summarizes it over the step's selected rollouts. It averages numbers,
counts labels, and turns a boolean into the rate it describes. For example, an agent
that records

```python
context.metadata = {"termination_reason": status, "turns": turns, "hit_limit": capped}
```

gets `rollout/turns`, `rollout/hit_limit`, and one `rollout/termination_reason/<status>`
for each outcome, beside the reward that the step trained on. `train/rollouts` gives
the denominator those counts need, because a group can train with as few as
`group_size - max_group_losses` members.

The recipe doesn't interpret any of these keys, so they mean whatever your agent means
by them. Keep them stable across rollouts. Two rules drop keys, and each one logs its
own warning:

- A label whose value differs in every rollout produces a metric name rather than a
  metric, so the run drops it as an identifier.
- The family is capped at `[output] rollout_metrics_limit` names, which is 32 by
  default. Each label value counts as one name. Past the cap, the run drops the names
  that the fewest rollouts reported first, and breaks ties by name. So an agent that
  reports more names than the cap loses ordinary metrics until you raise the cap.

`rollout/dropped_keys` counts the keys that either rule left out.

## Settings by what they control

`loops run --help` lists every setting with its default, and the
[knob glossary](KNOB_GLOSSARY.md) describes each one. A setting that changes what
trains or what a run costs is required, such as `learning_rate`, `group_size`,
`loss_fn`, `max_steps` or the checkpoint cadences. `loops check` refuses a config that
leaves one out. The glossary marks each required setting.

This table groups the settings by what they control:

| You want to change | Settings |
| --- | --- |
| Batch shape | `group_size`, `max_group_losses`, `groups_per_step` |
| Task order | `shuffle_seed` |
| Optimizer | `learning_rate`, `grad_clip_norm`, `adam_epsilon`, `adam_beta1`, `adam_beta2`, `weight_decay`, `router_replay`, `lora_rank`, `max_seq_len`, `[training] seed` |
| The loss | `loss_fn`, `loss_fn_config`, `[training.custom_loss]` |
| Loss reduction | `loss_aggregation`, `loss_token_budget` |
| How long it runs | `max_steps`, `max_consecutive_unusable_groups`, `max_consecutive_rollout_failures` |
| Concurrency | `max_concurrent_rollouts`, `max_concurrent_verifiers`, `max_concurrent_provisioning`, `prestart_allowance` |
| Provider resources | `[providers.<name>]`, `[resources] state_dir` |
| Deadlines | `rollout_timeout_seconds`, `setup_timeout_seconds`, `[rollout.grading]`, `[rollout.cleanup]` |
| Retries | `max_rollout_retries`. Grading retries belong to the rollout runtime, as its `grading_retries`. |
| Staleness in async | `mode`, `max_steps_off_policy`, `admission_steps_ahead`, `max_policy_delta_in_trajectory` |
| Which generations train | `[rollout.producers]`. Setting this table replaces the default, which declares only `main`. If your table leaves out `main`, a call that names no producer is refused, and the run stops. |
| Model formatting | `tokenizer`, `renderer`, `renderer_chat_template_kwargs`, `[model.reasoning_efforts]`. [renderers.md](renderers.md) explains the renderer settings. |
| How agents reach the model | `[rollout_runtime] model_access`, and `bind`, `advertised_base_url` and `model_name` in `[rollout_runtime.openai]` |
| Infra | `scope`, `spot_job`, `trainer_replicas`, `sampler_replicas`, `sampler_replica_affinity`, `reuse_from_run_id`, `keep_run`, `shutdown_drain_seconds` |
| Evals | `[eval] every_steps`, `sampler_start_timeout_seconds`, `[eval.sets.<name>]` |
| Sampling defaults and the trajectory budget | `temperature`, `max_generation_tokens`, `max_trajectory_tokens` |
| Checkpoints and logs | `trainer_checkpoint_every`, `sampler_checkpoint_every`, `trace`, `checkpoint_ttl_seconds`, `resume_from`, `init_from`, `init_from_optimizer`, `output_dir`, `run_name`, `wandb_project`, `progress_every_seconds`, `rollout_metrics_limit`, `console` |
