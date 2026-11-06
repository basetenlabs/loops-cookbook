# Training

What happens between `loops run` and a new policy version, and what each setting
controls. Written from the code in `loops_cookbook.rl` and `loops_cookbook.llm`;
the Harbor runtime adds only the environment side, covered in
[harbor.md](harbor.md).

## The loop

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

Each optimizer update consumes `groups_per_step` groups. A group is `group_size`
rollouts of the same task. Advantages are computed within the group, so the model is
trained on "which of my attempts at this task did better," never on absolute reward.

## From a rollout to training rows

When grading finishes, the rollout's samples are arranged into **chains**: a sample
joins a chain if its prompt is exactly the previous sample's prompt plus completion,
possibly extended by new tokens. A straightforward tool-calling agent produces one
chain. An agent that compacts or branches its history produces several.

Each chain with at least one trainable sample becomes one **training row**: a datum,
and the positions it trains.

- `model_input` is the chain's final prompt plus completion, minus the last token
  (the last token is only ever a target).
- `target_tokens` is the next token at every position, prompt and tool results
  included; an image position has none and is `-100`. Router replay checks every
  target against the sampler's record, so no position can be masked out by its id.
- The row's **actions** are the positions inside a trainable completion: the agent's
  own tokens, and the only ones a built-in loss trains. Prompt tokens, tool results,
  and copied history carry no advantage, so they are never trained on, unless a
  [custom loss](#the-loss) trains on tool results.
- `advantages` carries the rollout's advantage at the actions and 0 elsewhere.
- `logprobs` carries the behavior logprobs from sampling, for the importance ratio.

When `[model] max_seq_len` is set, it bounds datum length; a longer chain fails the run
with a message naming the length, since silently truncating would misalign targets.
When it is unset, the trainer's own limit applies.

`loops check` asks Loops (`GET /v1/loops/capabilities`, read-only) whether it serves
`[model] base_model` and refuses a model it does not list, naming the closest it does,
one it has not enabled, and a `max_seq_len` above what it enables for the model.
It prints `Loops model: <name>; max_seq_len <n> (enabled up to <m>)`. Without
`BASETEN_API_KEY`, or when Loops cannot be reached or answers anything but 200 or a
rejected key, it prints that the model was not checked and why, and passes, so a
config still checks offline. A key Loops rejects (401 or 403) is refused: `run` would
fail on it too. `run` asks the same before it provisions the trainer.

**Router replay (R3).** Mixture-of-experts models let the trainer replay the exact
expert routes the sampler used. With `router_replay = true` the sampler records every
generation, and each datum names, for every position, the recorded generation it came
from (Loops' token provenance). A generation's own tokens take its own routes; the
tool results after it, and its last token, which the sampler never ran forward, take
the routes of the next generation's prompt. So a multi-turn chain replays every turn,
turns sampled under different policies included, and each fork of a branching
rollout replays its own chain. Within one optimizer step it is all or nothing: if any
row in the batch was not recorded, provenance is stripped from all of them
(`r3/unrecorded_datums` counts those rows). One position takes one route, so an
earlier turn keeps the routes it was sampled with even where a later policy would have
prefilled it differently. It is off by default; `router_replay = true` enables it.

Loops records text prompts only. A runtime whose prompts carry images sets
`[runtime] has_images = true`, and `loops check` refuses it with `router_replay = true`.
An image prompt that reaches a run with replay on fails its generation, and the run
logs once which settings to change.

## Groups, advantages, and what gets skipped

Tasks are shuffled once per epoch with `shuffle_seed` and emitted as groups. When
all `group_size` rollouts of a group have resolved (graded, failed, or discarded),
the group is checked:

- Stale rollouts are dropped (the `max_steps_off_policy` check, again).
- More than `max_group_losses` of the group did not come back graded: the group is
  rejected (`group_too_small`).
- All survivors have the same reward: rejected (`no_variance`).
- Every survivor with a nonzero advantage trains nothing, because its agent discarded
  every generation: rejected (`nothing_to_train`). Such rollouts still count in the
  group's mean; `trajectory/untrained_rollouts` counts them per update.

A runtime with a group grader, such as an `Episode` that defines `grade_group()`,
scores the group between the size and variance checks: once the stale are dropped and
the group is large enough to train, one `grade_group()` call gives every member its
final reward. The judge is never paid for a group about to be dropped. See
[episode-api.md](episode-api.md#group-grading).

Survivors get `advantage = reward − mean(group rewards)`, not divided by the group's
standard deviation. `max_consecutive_unusable_groups`
aborts the run after that many rejections in a row; a usable group resets the count.
It exists to stop a run that is burning sandboxes without learning. It is not a
budget.

Rejected groups never reach `reward/mean`, which averages the rollouts an update
trains. So a policy that starts solving, or failing, whole groups outright moves that
curve less than its real reward. Each update also logs `reward/mean_with_unusable`: its
trained rollouts plus every graded member of the groups rejected as `group_too_small`,
`no_variance` or `nothing_to_train` since the previous update. `reward/unusable_rollouts` counts the
members it added. Each numeric `rollout/<key>` is reported the same way, as
`rollout/<key>_with_unusable`. All of these are reported every update, whether or not
a group was rejected.

`reward/std` is the spread over the whole batch, so it reads high even when every
group's members scored alike. The spread GRPO learns from is reported per update as
`reward/group_std_mean`, `_min` and `_max`, over the groups it trains.

For a built-in loss the trainer returns more than `train/mismatch_kl`: each of its
metrics, such as `mismatch_abs_logprob_diff` and `mismatch_ess` (the importance
weights' normalised effective sample size), is reported as `train/<name>`, and every
`forward_backward` call's report in `steps.jsonl` keeps them all.

`mismatch/*` measures the same disagreement on the tokens that train, from the
logprobs the trainer returns for every datum, built-in loss or custom.
`mismatch/abs_logprob_diff` is the token mean of |trainer − sampler| logprob, and
`mismatch/abs_logprob_diff_max` the largest. The mean is also split two ways:
- `/turn_1`, `/turn_2`, `/turn_3-4`, `/turn_5-8`, `/turn_9-16` and `/turn_17+`: by the
  action's place in its row. With router replay, late turns should read like early ones.
- `/lag_0`, `/lag_1` and `/lag_2+`: by how many updates the trainer's weights are past
  the policy that sampled the action. At lag 0 the gap is numerics and routing alone;
  from lag 1 it includes the policy's own change.

A nonfinite trainer logprob on a trained token is left out of those means and
counted in `mismatch/nonfinite_tokens`. An answer with no logprobs leaves `mismatch/*`
out, with one warning. An answer that does not fit the rows (logprobs for a datum
missing, or of the wrong length) fails the step: Loops already holds that call's
gradients, so the update is refused.

Two more say how fresh a batch is and why. `freshness/lag_mean` and `freshness/lag_p90`
give the distribution of how many updates behind the trainer each trained rollout's
oldest generation was (`freshness/lag_max` alone shows only the worst). And
`groups/completion_spread_mean_seconds` and `_max_seconds` give, per trained group, the
time from its first member resolving to its last: its first finishers age that long
waiting for the rest, so a wide spread means rollouts are spread over many groups.

Failures are counted separately. An agent that fails is graded, whatever the runtime:
what it scores is the grader's call, never the cookbook's. Harbor still runs the
trial's verifier, as Harbor does. A custom runtime returns from `sample()` for every
episode that happened, however badly the agent did, and `grade()` scores what it
left. Raising from `sample()` means the episode didn't happen: it is retried up to
`max_rollout_retries`, then discarded as `rollout_failed/<type>` and counted toward
`max_consecutive_rollout_failures`; `PermanentRolloutFailure` is discarded without a
retry. The one agent outcome the cookbook decides itself is its rollout clock: a
rollout that runs out of `rollout_timeout_seconds` has no episode to grade, and is
discarded as `agent_timeout/rollout_clock` without a retry, and counted as a sampling
failure. A rollout that comes back with no
trainable samples is treated as a failure too, since it means the agent never called
the policy. [failures.md](failures.md) has the whole contract, with the
retries, deadlines and failure counts of each stage.

## The training step

Once `groups_per_step` usable groups are selected and pass a final freshness
recheck:

1. With `[output] trace = "full"`, the rows are written to `batches/step-N.json`
   before submission: each datum and its actions.
2. `forward_backward` with the run's [loss](#the-loss) accumulates gradients. The
   SDK may split a large batch into several calls; Loops accumulates the gradients
   until the optimizer step and applies the update's reduction there (see
   [loss aggregation](#loss-aggregation)), so the split is invisible.
3. `optim_step` with Adam: `learning_rate`, `grad_clip_norm` (0 disables),
   `adam_epsilon`. A non-finite loss aborts before this point and poisons the trainer
   so nothing further is published from it.
4. The policy version advances with the optimizer step, and the runner refuses to
   continue if it does not. Weights are then published to the paired sampler.
5. Every `sampler_checkpoint_every` updates the sampler weights are staged under
   `step-N`; every `trainer_checkpoint_every` updates the full trainer state
   (including optimizer moments) is saved alongside them and a manifest is written
   to `checkpoints/step-N.json`. The update at `max_steps` always writes both. A run that ends any other way --
   interrupted (Ctrl-C, or SIGTERM as a scheduler sends), aborted, failed, its Loops run ended -- saves the trainer state of its last applied update
   on the way out, if the cadence skipped it. That manifest is what
   [`resume_from`](#continue-or-fork-a-run) takes.
6. `steps.jsonl` gets one record with everything the step reported.

Nothing stops a second process from driving the same trainer; don't.

By default the run leaves its Loops run up on the way out, so that on dedicated Loops
a later run's `[compute] reuse_from_run_id` starts warm. The run then logs a warning with its ID.
The Loops run stays up until you run `loops deactivate <run id or run directory>`. A
dedicated run holds its pods and bills until then. A serverless run has no deployment
of its own and costs nothing while it waits, but it holds one of the workspace's
serverless slots.

To end the Loops run on the way out, set `[compute] keep_run = false`. The run then
deactivates it however the run ends: completed, interrupted, aborted or failed. The
checkpoints stay. If Loops does not confirm the deactivation within 30 seconds, the
run logs the same warning.

## Continue or fork a run

Two settings load trainer state that an earlier run saved. Both read the same
`bt://loops:<run>/weights/step-N` trainer state. They differ in what continues after
the load.

| | `resume_from` | `init_from` |
|---|---|---|
| Use it to | continue a run that stopped, or give it more steps | start a new run from trained weights |
| It takes | a local `checkpoints/step-N.json` manifest, or the run directory | a `bt://loops:<run>/weights/step-N` path, from any run you can read |
| It loads | the weights, the optimizer state, and the run's position | the weights, with fresh Adam moments (`init_from_optimizer` adds the optimizer state) |
| Steps | continue from the manifest's next step | start at 0 |
| Run directory | the same one | a new one |
| Config changes | refused when they change what an update means | free: new data, loss, learning rate, or recipe |

Each setting takes a TOML value in `[training]` or a flag on `loops run`. `--resume`
is the short form of `resume_from` for the latest run of the same config:

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

`--resume` takes the latest run: the `<output_dir>/<run_name>-<time>-<id>` directory
that trained most recently, by the times its last update and manifest record, not by
when it started or by file times, which a copy resets. A run resumed today is later
than one started yesterday. It logs which run. When no
run has trained, or the latest run kept no manifest, it exits 2 and starts nothing: it
never takes an older run instead. It cannot be combined with `resume_from`.
`[output] run_name` is required, so two configs in one `output_dir` never take each
other's runs.

The log names each trainer state as it is kept: `remote trainer state:
bt://loops:<run>/weights/step-N`. A manifest holds the same path as
`trainer_state_path`. An `init_from` source must have the run's `base_model` and
`lora_rank`. It can come from the other recipe: an RL run can start from SFT weights.

**`resume_from`** restores the trainer state, the step counter, and the task cursor.
`max_steps` counts the run's updates, the ones before the resume included: a resume
from step 10 with `max_steps = 30` makes 20 more. This is the meaning Tinker gives
`max_steps`. A resume with no update left is checked as any resume is, then completes
with exit 0: it logs that the run already made every update, and starts no trainer.
Raise `max_steps` to train more. `loops check` says the same.
Rollouts in flight when the run stopped are not replayed. The resumed run writes into
the manifest's run directory, or the run directory given. It refuses a changed
`base_model` or `lora_rank`. A resume from an older manifest that stops before it
writes its own first manifest leaves the later branch as the one written last. To stay
on the earlier step, resume that manifest file again.

**`init_from`** loads the LoRA adapter at step 0 of a new run directory. The trainer's
policy version is not reset: it continues from the checkpoint's, so it is offset from
the run's step count. A resume of that run restores its own state and does not load
the `init_from` weights again. `init_from_optimizer = true` also loads the source's
optimizer state (Adam moments and step), for a fork that keeps the source run's
optimizer under a new config. Those moments reflect the source run's gradient scale.
A fork that changes what produces the gradients (loss, data, batch size) starts with
moments sized for the old ones. From a checkpoint saved without moments, Loops loads
the weights alone. `init_from_optimizer` requires `init_from`, and a resume ignores it
with `init_from`.

`init_from` with `init_from_optimizer = true` loads the same state as a resume. It does
not keep the step count, the task cursor, or the run directory, and it does no resume
checks. Use `resume_from` to continue a run, and `init_from` to start a new one.

## Evals

An eval scores the policy on held-out tasks while it trains. Each eval set is training's
own runtime, with the same factory and `[runtime.config]`, given other files. The
agent, grader and sandboxes are training's. The only thing an eval set requires is
its tasks:

```toml
[eval]
every_steps = 10

[eval.sets.val]
samples_per_task = 4              # optional; 1 by default
max_concurrent_rollouts = 16      # this set's rollouts at once

[eval.sets.val.runtime.files]     # replaces these [runtime.files] entries
tasks = "val.jsonl"

[eval.sets.val.sampling]          # optional; overrides [sampling] for this set
temperature = 0.6
```

A key at the path of a training setting overrides it for the set: `runtime.files`
and `sampling.temperature`. The others are the set's own.

| Eval | When | Weights | Logged at `train/update` |
|---|---|---|---|
| Base | After the trainer starts, on a run with neither `resume_from` nor `init_from` | the base model | `-1` |
| Step | After update N, when `(N + 1) % every_steps == 0` | the update's sampler weights | N |
| Final | After the last update, which needs `max_steps` | the same | N |

`every_steps = 0` evaluates only the base model and the last update.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/eval-timeline-dark.svg">
  <img alt="When evals run, with every_steps = 2 and max_steps = 6. Training runs updates 0 to 5 without waiting; steps 1, 3 and 5, which an eval is due for, save their weights at the end of the update. On dedicated Loops each eval starts a sampler deployment, then runs its rollouts: the base eval is still running when step 1's eval comes due, so step 1 is skipped; step 3's eval runs; the final step 5 eval comes due while step 3's still runs, waits for it, then starts its own sampler; the run ends when the final eval does. Step 1 still keeps its weights. On serverless Loops the sampler is ready at once and a set's evals overlap: the base, step 1, step 3 and final evals each run when due, step 1's beside the base eval and the final beside step 3's, and the run ends sooner." src="../assets/eval-timeline.svg">
</picture>

Each eval set gets a sampler of its own, pinned to the evaluated weights and never the
run's paired one. Training doesn't wait for it: only the save of that step's sampler
weights sits on the training loop, in the step's `checkpoint` time. The sets of one
step run at once, each on its own sampler.

On dedicated Loops, where each eval starts a deployment, a set runs one eval at a time:
an eval due while the set's last one still runs is skipped with a warning. The step
keeps its weights all the same, so `loops eval` can run that eval after the fact (see
[Evaluating after the fact](#evaluating-after-the-fact)). The final eval waits for the
set's last one, then runs. On serverless Loops a sampler costs nothing while it waits,
so a set's evals overlap, every eval due runs, and each has its own
`max_concurrent_rollouts`. Evals sample from the same serverless capacity as training,
so in an RL run, overlapping evals can slow its updates.

A set's rollouts keep `[rollout] rollout_timeout_seconds`, or when that is
unset the clock its own runtime derives. A run that completes waits
for its evals, its trainer and paired sampler still up and billing until the last one
ends. A run that stops cancels them. Either way, every eval sampler is
deactivated. Nothing an eval samples trains.

Each set writes `eval/<set>/base.jsonl` and `eval/<set>/step-N.jsonl` in the run
directory, one row per rollout: `task_id`, `task_digest`, `sample`, `rollout_id`,
`reward`, `discarded` and `metadata`. To compare steps, join the files on task. Beside
each is `base.transcripts.jsonl` or `step-N.transcripts.jsonl`: every rollout's
conversation, what it was asked and what it replied, recorded whatever `[output]
trace` says. W&B
gets `eval/<set>/reward_mean`, `reward_std`, `rollouts`, `graded`,
`discarded/<reason>`, `sampler_start_seconds` and `seconds`, and what its episodes
report during that eval as `eval/<set>/runtime/...`, never mixed into training's or
another eval's `runtime/...`. What the set's runtime reports through its own context,
outside any episode, is `eval/<set>/set/runtime/...`: one count per set, adding up
across its evals so far.

| What happens | What the eval does |
|---|---|
| Its sampler isn't ready within `sampler_start_timeout_seconds` | It is skipped, with a message and no file. Training continues. |
| The trainer names no path for the step's weights | It is skipped, with a message and no file. Training continues. |
| A rollout fails past its retries | Its row has `discarded` set and no reward, and it is left out of the mean. |
| Failures in a row reach `max_consecutive_rollout_failures` | That eval stops, with no file. Training continues. |
| Hooks that ignored cancellation hold every one of the set's `max_concurrent_rollouts` slots | That eval stops, or is skipped before its sampler starts, with no file. Training continues. |
| The run stops mid-eval | Its sampler is deactivated, it has no file, and the log says it did not finish. A resume doesn't rerun it. |

`loops check` builds each eval set's runtime and prints a line for it. It warns about
tasks an eval set shares with training, by id or by the runtime's task digest. An
episode's digest covers the task's `id`, so the same task under another id is not
caught. It refuses a set whose runtime reaches the model another way than training's
does, since one gateway serves both. A runtime with a group grader scores each task's samples together.

On serverless Loops an eval's sampler shares replicas with training, and is ready at
once. On dedicated Loops each eval set starts a sampler deployment for every eval.
That deployment needs free allocation for the model at `max_seq_len`, takes time to
start, and bills until it is deactivated. A runtime whose training and eval instances
lease the same run-scoped resource fails on the second lease.

Eval rollouts use training's sandbox providers and their quotas. A set's
`max_concurrent_rollouts` adds to training's: at an eval, a provider can see
`[rollout] max_concurrent_rollouts` plus each set's at once, and on serverless Loops
each set's again for every eval of it that overlaps. Size them to what the provider
allows. Grading is bounded the same way: each eval grades at most
`[rollout] max_concurrent_verifiers` rollouts at once, as training does, so a judge's
rate limit sees that many from training and from each eval running.

### Evaluating after the fact

```bash
uv run loops eval train.toml --step 29             # update 29 of the latest run
uv run loops eval train.toml --step base --set val # the base model, one set
uv run loops eval train.toml --step 29 --run runs/sat/sat-1793877514-b36ce057
```

`loops eval` runs the config's eval sets on one update of a run that has ended or is
still training, as an eval during the run would: it writes `eval/<set>/step-N.jsonl`
and its transcripts into that run's directory, replacing an eval of that step already
there, and logs the same line. Use it for an eval that was skipped or failed, or for a
new eval set on a finished run. It takes the latest run of the config, as `--resume`
does, or the run `--run` names.

The weights are what the run kept for that update. The run keeps sampler weights for
every eval due, skipped or failed included, and every `sampler_checkpoint_every`
updates, and records each in `sampler_weights.jsonl`; `loops eval` serves those. For
an update with only trainer state, kept every `trainer_checkpoint_every` updates, it
starts a trainer that loads that state, saves it for a sampler, and then deactivates
that trainer's Loops run. For any other update it says which ones it can evaluate. It
exits 1 when a set did not evaluate, and the log says why.

## The loss

`[training] loss_fn` picks what `forward_backward` computes; the names are the SDK's.
Group-centered advantages are computed independently of the loss. Loss aggregation
controls how their action-token contributions are weighted for the optimizer update.

- `cispo`, `importance_sampling`, `ppo`, `dro` and `dppo` are Loops'
  policy losses, each over the recipe's datums as they are. `loss_fn_config` is
  handed to Loops as it is, and a key left unset keeps Loops' default:

  | `loss_fn` | `loss_fn_config` (default) |
  | --- | --- |
  | `importance_sampling` | none: `−ratio · advantage` |
  | `ppo`, `cispo` | `clip_low_threshold` (0.8), `clip_high_threshold` (1.2): the ratio is clipped to [low, high] |
  | `dro` | `beta` (0.05): weight of the `log_ratio²` penalty |
  | `dppo` | `dppo_mask_high`, `dppo_mask_low` (0.2), `adv_tau` (1.0), `kl_tau` (0.001) |

  `check` holds each to the bounds the Loops trainer applies, and refuses a key the
  loss does not take, which the trainer would otherwise ignore. Loops' `dpo` (pairs
  against a reference) and `cross_entropy` (SFT) train other kinds of data.
- `custom` is your own, run through Loops' `forward_backward_custom`.
  `[training.custom_loss]` names a `factory` as `module:callable` (a module beside the
  config is importable) and its `config`, handed to it unchanged; the factory returns
  the loss function. `check` calls it, so a bad factory or config fails there. Needs
  the `custom-loss` extra (PyTorch).

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

Loops scores every text position of a datum, the prompt and the tool results between
turns included, and `fn` gets the trainer's logprobs there, one tensor per datum. Each
datum's `target_tokens` is the next token at every position (`-100` at images, where
the logprob is zero); `actions` marks the agent's own tokens, the ones `advantages`
and the sampler's `logprobs` describe. A loss that trains on tool results, but not on
the task's prompt, takes the positions after the first action. Loops sums
the gradients without normalizing by token count, so `fn` normalizes as it means to.
`fn` is called once per `forward_backward`, over the whole batch the runner submits.
For a custom loss, `train/loss` is `fn`'s value and `train/mismatch_kl` is not
reported. [ECHO](../../examples/terminal_bench/echo_loss.py) is a worked example: it
adds a next-token loss on the tool results.

### Loss aggregation

Choose the objective and its reduction independently:

```toml
[training]
loss_fn = "cispo"
loss_aggregation = "sum"
```

CISPO uses Loops' clipping defaults (`0.8` and `1.2`) unless you set
`loss_fn_config`. Set `loss_fn = "importance_sampling"` for the plain
importance-sampling objective.

Let `S_g` be the sum of policy losses over action tokens in task group `g`.
Each group contains repeated episodes for one task; the same task admitted again
is a new group. Prompt and tool-result tokens are excluded.

| `loss_aggregation` | Optimizer objective | Meaning |
| --- | --- | --- |
| `sum` (default) | `sum(S_g)` | Loops' own default: no division, as with the SDK directly. |
| `token_mean` | `sum(S_g) / N` | `N` is the update's action-token count **with nonzero float32 advantages**; Loops divides the whole loss, penalty terms included. |
| `fixed_budget` | `sum(S_g) / (group_size * groups_per_step * loss_token_budget)` | A fixed configured denominator, independent of realized episode lengths or lost group members. |
| `task_mean` | `mean_g(S_g / T_g)` | Equal weight per selected task group; `T_g` counts all its action tokens, including zero-advantage actions. |

For a fixed denominator, supply an explicit budget across the **whole episode**:

```toml
[training]
loss_fn = "cispo"
loss_aggregation = "fixed_budget"
loss_token_budget = 8192
```

The budget is a normalization constant, not a generation limit. It covers all
turns and trainable producers in an episode; it is never inferred from
`max_generation_tokens` or the context window. A larger denominator reduces the
gradient scale, so compare learning rates when changing aggregation.

An agent that compacts or branches its history splits a rollout into several
[chains](#from-a-rollout-to-training-rows), one datum each. That adds rows, not
trained tokens: each generated token is a target in exactly one chain, and the history
a new chain copies is prompt there. So the budget counts each generated token once,
however many chains the rollout has. To choose it, run a few steps and read
`trajectory/sampled_tokens_median` and `trajectory/sampled_tokens_max` in
`steps.jsonl`: the tokens each rollout generated, across all its chains, including any
from producers that do not train.

The seam is `rl.aggregation.aggregate_loss(datums_by_group, training)`. It runs
after freshness selection and group advantage calculation, before recording or
submitting rows. `sum` and `token_mean` are Loops' own reductions: the rows go
unchanged, and every built-in `forward_backward` states which one it wants
(`skip_token_normalization`). `fixed_budget` and `task_mean` keep Loops' sum and
multiply the advantages by `1 / D`, which works because IS, PPO and CISPO are
linear in positively scaled advantages. For `task_mean`, `D` differs by group. The prepared rows can be
split across forward/backward calls; all of them must share one optimizer update,
without mixing rows from another preparation. Transport chunking never changes
the counts used for reduction. Scaling that changes the float32 active-token mask
or produces nonfinite advantages is rejected before submission.

`fixed_budget` and `task_mean` are supported for IS, PPO and CISPO. DRO and DPPO
have penalty terms that advantage scaling would not reweight, so they take `sum` or
`token_mean`, which divide the whole loss alike. Custom losses receive the original
advantages and reduce inside their own function; Loops sums what they return, so
they keep `sum` and any other mode is refused.
Advantage standard-deviation normalization is separate and remains unchanged:
advantages are reward minus the group mean.

Full batch traces record the aggregation mode, counts, denominators and scales
beside the exact weighted rows sent to Loops. Step metrics under `aggregation/`
appear only where the cookbook divides: `denominator` for `fixed_budget`,
`group_denominator_min` and `group_denominator_max` for `task_mean`. Token counts are
`train/trained_positions` and Loops' own `optim/num_loss_tokens`. `train/loss` remains SDK-reported loss telemetry;
it is not a reconstructed value of the globally reduced optimizer objective.

### Trainer normalization contract

Built-in aggregation requires `loops.rl.sum_or_active_token_mean.v2`: Loops sums
gradients across backward calls, and with `skip_token_normalization = false`
divides once at the optimizer step by targeted tokens with nonzero float32
advantages. The cookbook states the setting on every built-in `forward_backward`
and refuses a result whose trainer does not echo it back; a trainer older than
summed defaults rejects the setting. The name is the cookbook's identifier for
that requirement, not a capability advertised by the trainer. Full batch traces
include `backend_normalization` alongside the expected count; the optimizer
reports its actual count as `optim/num_loss_tokens`.

Before a release or after trainer/SDK changes, trigger the **trainer normalization
contract** Actions workflow. This is an on-demand, billable check; ordinary PR CI
and publishing do not launch it or require it to pass.

The check keeps weights fixed with zero-learning-rate optimizer steps and compares
built-in CISPO to an explicitly reduced custom objective for all four modes, and
built-in DRO and DPPO under `token_mean`, each with its penalty divided like its
advantage term. It checks gradient norms and unchanged logprobs. It records how far split
`forward_backward` calls move the trainer's logprobs, which depend on a row's batch
neighbours, and holds two calls of one batch to a single call of its doubled
objective within 0.1%, so accumulation is checked apart from that drift. Norms are measured before Adam, so
optimizer moments do not affect this comparison. With matching transport chunks,
the custom objective and the built-in loss must agree within 0.01% for both
on-policy and clipped importance weights; where Loops averages, the custom
objective runs undivided and its norm is divided afterwards, as Loops divides the
accumulated gradient. Clipping holds importance weights fixed
to isolate forward-scoring differences.
Zero-advantage tokens, ignored targets and split backward calls test the denominator
and accumulation behavior.
It runs on one trainer without a sampler or sandbox and deactivates its owned run
in teardown.

The manual workflow needs the `BASETEN_API_KEY` Actions secret and fails when it is
missing. Alternatively, run locally with credentials in `.env`:

```sh
LOOPS_LIVE_LOOPS=1 uv run --env-file .env --extra custom-loss pytest -q -s \
  tests/live/test_trainer_normalization.py --basetemp=trainer-contract-results
```

`LOOPS_CONTRACT_MODEL` can select another supported Loops model; the default is
`Qwen/Qwen3.5-4B`. Workflows retain the numerical receipt and cleanup result as
an artifact. A passing receipt qualifies the tested model/trainer deployment;
rerun after backend changes, and qualify other backend/model families before
assuming their behavior is identical.

## Outputs

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

Metric families in `steps.jsonl` (and W&B with the `wandb` extra):
`reward/*`, `advantage/*`, `train/*` (loss, tokens, trained positions, policy
version), `freshness/*` (lag and version span of the batch), `trajectory/*`
(generations and tokens per rollout), `sampling/stop/<reason>`, `sampling/entropy_estimate`,
`optim/*` (grad norm, step time, throughput when the SDK returns them), `loss/*` (a
custom loss's own metrics), `time/*`,
`slots/*` and `queue/*` (occupancy), `admission/*` (the async window, the groups and rollouts committed to it, and group survival),
`discard/rollouts/<reason>`, `discard/groups/<reason>`, `discard/generations` and
`discard/generation_tokens` (every generation sampled and then left out of the
evidence, which batch metrics do not count because they describe what survived),
`discard/undelivered_generations` and `discard/undelivered_generation_tokens` (the
part of those that sealing dropped because the agent never received them; the rest
the agent discarded), `trace/unwritten_samples` and `trace/unwritten_transcripts`
(trace records that could not be written; a failed write is logged and never fails
its attempt), `rollout/*`, `trajectory/*` (`trajectory/forked_rollouts`;
`trajectory/mixed_policy_rollouts`, rollouts whose trainable generations came from more
than one policy version), and `r3/*` (`r3/routed_datums`, the rows replayed;
`r3/unrecorded_datums`, the rows that took a batch off replay). Pipeline state is in `pipeline/*`,
`groups/*`, `progress/*`, `retried/*`, `attempts/live` and `pipeline_calls/*`.

Each `rollouts.jsonl` record, trained or discarded, carries the same two facts for
the attempt it names: `generations_undelivered` and `generations_discarded_by_agent`.
A rollout whose record names no attempt sealed none, and both are zero. Every record
has the same fields, meaning the same thing: `policy_version` is the oldest version
among the rollout's trainable samples (`null` when it sealed none), `task_digest` and
`sample_ids` (empty when it sealed none) name what it ran and sampled, and `step` is
the update that trained it, `null` for a discarded rollout.
`train/mismatch_kl` is the sampler-versus-trainer disagreement and is logged, never
used to gate.

`rollout/*` is yours: an `Episode`'s `run()` metadata, or whatever a Harbor agent puts in `context.metadata`,
with the grader's metadata merged over it, is summarized over the step's selected rollouts: numbers are averaged, labels counted,
and a boolean becomes the rate it describes. An agent that records

```python
context.metadata = {"termination_reason": status, "turns": turns, "hit_limit": capped}
```

gets `rollout/turns`, `rollout/hit_limit` and one `rollout/termination_reason/<status>`
per outcome, beside the reward that step trained on. `train/rollouts` is the
denominator those counts need, since each group can train with as few as
`group_size - max_group_losses` members. The recipe does not interpret
any of it, so the keys mean whatever your agent means by them — but keep them stable
across rollouts. Two rules drop keys, and each warns on its own. A label that differs
every rollout produces a metric name rather than a metric, so it is dropped as an
identifier. The family is also capped at `[output] rollout_metrics_limit` names (32 by
default, each label value counting as one): past it, the names the fewest rollouts
reported go first, then by name, so an agent that reports more than the cap loses
ordinary metrics until you raise it. `rollout/dropped_keys` counts what either rule
left out.

## Settings by what they control

`loops run --help` has every setting with its default. Grouped:

| You want to change | Settings |
| --- | --- |
| Batch shape | `group_size`, `max_group_losses`, `groups_per_step` |
| Task order | `shuffle_seed` |
| Optimizer | `learning_rate`, `grad_clip_norm`, `adam_epsilon`, `router_replay`, `lora_rank`, `max_seq_len`, `[training] seed` |
| The loss | `loss_fn`, `loss_fn_config`, `[training.custom_loss]` |
| Loss reduction | `loss_aggregation`, `loss_token_budget` |
| How long it runs | `max_steps`, `max_consecutive_unusable_groups`, `max_consecutive_rollout_failures` |
| Concurrency | `max_concurrent_rollouts`, `max_concurrent_verifiers`, `max_concurrent_provisioning`, `prestart_allowance` |
| Provider resources | `[providers.<name>]`, `[resources] state_dir` |
| Deadlines | `rollout_timeout_seconds`, `setup_timeout_seconds`, `[rollout.grading]`, `[rollout.cleanup]` |
| Retries | `max_rollout_retries`; grading retries are the runtime's (`grading_retries`) |
| Staleness in async | `mode`, `max_steps_off_policy`, `admission_steps_ahead`, `max_policy_delta_in_trajectory` |
| Which generations train | `[rollout.producers]`: it replaces the default, which declares only `main`, so with a table that omits `main`, a call that names no producer is refused and stops the run |
| Model formatting | `tokenizer`, `renderer`, `renderer_chat_template_kwargs` (see [renderers.md](renderers.md)), `[model.reasoning_efforts]` |
| How agents reach the model | `[runtime] model_access`, `[runtime.openai] bind`, `advertised_base_url`, `model_name` |
| Infra | `scope`, `spot_job`, `trainer_replicas`, `sampler_replicas`, `sampler_replica_affinity`, `reuse_from_run_id`, `keep_run`, `shutdown_drain_seconds` |
| Sampling defaults and the trajectory budget | `temperature`, `max_generation_tokens`, `max_trajectory_tokens` |
| Checkpoints and logs | `trainer_checkpoint_every`, `sampler_checkpoint_every`, `trace`, `checkpoint_ttl_seconds`, `resume_from`, `init_from`, `init_from_optimizer`, `output_dir`, `run_name`, `wandb_project`, `progress_every_seconds`, `rollout_metrics_limit`, `console` |
