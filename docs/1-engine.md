# 1 · The training engine

For engineers changing the cookbook. How one run works inside, the rules the code
keeps, and why. Using the cookbook is in the [README](../README.md) and
[patterns/](../patterns/).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/engine-overview-dark.svg">
  <img alt="The training engine. Run drives the Pipeline and the Trainer. The Pipeline calls the AttemptDriver's sample and grade. The AttemptDriver takes permits from RunCapacity, opens and seals the attempt's model-access lease, and calls the runtime's hooks. Two runtimes sit behind the runtime contract: EpisodeRuntime over your Episode class, and HarborRuntime over a Harbor job and your agent. Agents call Model access, which samples from the Loops sampler. The runtime provisions sandboxes through the ResourceRegistry, which creates them at Blaxel or Daytona. The Trainer runs forward/backward and the optimizer on the Loops trainer, which publishes new weights to the sampler every step." src="assets/engine-overview.svg">
</picture>

## The idea

A run is RL on agents: sample a group of attempts at each task against the live
policy, score them, turn each group's rewards into advantages, and train one step on
Loops. The engine is everything in blue. It schedules attempts, runs every clock and
retry, owns every billed sandbox, and — the part everything else serves — records the
exact tokens the sampler produced, so training never depends on text. A **runtime**
runs the episode itself; the engine calls it through a six-method contract (plus an optional group grader) and never
trusts it for evidence. What an episode *means* (its reward, what a failure scores) is
user code. Mechanics and evidence are ours.

## The parts

| Part | File | Owns |
| --- | --- | --- |
| Run | `rl/run.py`, `rl/recipe.py` | Lifecycle: preflight (`check` stops here), gateway, registry, the loop, the shutdown order, resume. `rl/recipe.py` (`assemble`) builds what it uses: the trainer, the runtime factory, the renderer. |
| Pipeline | `rl/pipeline.py` | Task stream → groups; sampling, grading and group-grading workers; sync/async gating; batch selection. Schedules only. |
| AttemptDriver | `runtime/driver.py` | One attempt's lifecycle around the runtime's hooks: clocks, retries, sealing the Rollout, cleanup, the systemic stop. |
| RunCapacity | `runtime/capacity.py` | Admission queue with pre-start places; sampling and verification permits; the clocks' bookkeeping. |
| Model access | `model_access/`, `llm/` | The per-attempt **lease** (the receipt book for every model call), the renderer, the direct client and the HTTP gateway. |
| Trainer | `rl/trainer.py`, `rl/grpo.py` | Advantages, token-exact datums, `forward_backward`, `optim_step`, publish, checkpoint. |
| ResourceRegistry | `resources/`, `providers/` | Every billed resource: journal first, create, release, prove absent, recover. |
| Runtime contract | `runtime/api.py` | `prepare · sample · grade · finish · abort · close` — the only seam user code sits behind. See [2 · Interfaces](2-interfaces.md). |

## Scheduling: from a task to a batch

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/scheduling-dark.svg">
  <img alt="Scheduling from a task to a batch. Top row: the task stream, shuffled per epoch; make a group of group_size rollouts when the gate allows; the admission queue, oldest group first, admitted when active is below the target plus pre-start places; setup (sandbox create, __init__, setup) bounded by max_concurrent_provisioning; the sampling slot bounded by max_concurrent_rollouts. After seal, bottom row: sealed and waiting holds no slot; the grading slot bounded by max_concurrent_verifiers; group complete, with the optional judge on the same verifier slots; pick a batch of groups_per_step usable groups, held-over groups first, then the oldest policy version; train one step. Below, the gates: sync starts groups only while fewer than groups_per_step are live; async starts a group while the started groups not yet training fit the freshness window, plus the groups expected to drop; pre-start places add ceil(prestart_allowance × max_concurrent_rollouts) admissions on top of the full target only." src="assets/scheduling.svg">
</picture>

The pipeline (`rl/pipeline.py`) decides *when a group starts*; `RunCapacity`
(`runtime/capacity.py`) decides *when each attempt may take a step*. Between them:

- **One gate starts groups.** Sync admits exactly the `groups_per_step` groups the next
  step needs, and nothing while a step trains. Async starts a group while the started groups not yet
  training fit the freshness window: `admission_steps_ahead` batches of groups, fractions
  allowed (`max_steps_off_policy` when unset), one more with no batch training, plus
  the groups expected to drop, up to a batch of them, held
  in rollouts so a group still filling holds only what it has left, and never more
  groups than `max_steps_off_policy` leaves batch places for.
  Staleness is bounded mainly here, by not starting work, rather than by discarding it
  later.
- **Three limits bound the work**, each a semaphore: provider operations
  (`max_concurrent_provisioning`), agents running (`max_concurrent_rollouts`), and
  graders running (`max_concurrent_verifiers`, shared by `grade()` and the group
  judge). A slot is held only while its work runs. A rollout waiting to be graded, or a
  group waiting for its batch, holds nothing.
- **Admission is one queue, oldest group first.** An attempt is admitted when the active
  count is under its group's target. Waiting attempts are served in the order their
  groups were dispatched, and a group's own in arrival order, so a retry, a fresh
  attempt that arrives behind newer groups, is not left waiting behind them while its
  group waits for it. A started group's siblings always fill to the full
  limit, so a group, once begun, finishes. **Pre-start places**,
  `ceil(prestart_allowance × max_concurrent_rollouts)` (7 at 64), let that many extra
  attempts create their sandbox and run `setup()` before a sampling slot frees — but
  only on top of the full target, so under backpressure no new group starts early.
- **Workers aren't the limit; the permits are.** The pipeline runs enough sampling,
  grading and group-grading workers never to be the bottleneck.
- **A group is ready when every rollout has resolved** (graded, failed or discarded).
  Batch selection takes held-over groups first, then the ready group closest to going
  stale: the lowest oldest policy version among its members, then dispatch order. It
  never waits for a particular group. It rechecks staleness, size
  (`group_size − max_group_losses`) and variance, and counts unusable groups toward
  `max_consecutive_unusable_groups` (16).

Locked by `tests/unit/runtime/test_capacity.py`
(`test_a_prestart_place_never_starts_a_group_the_target_holds_back`,
`test_a_lower_target_never_interrupts_an_active_rollout_and_started_groups_finish`,
`test_waiting_attempts_are_admitted_in_arrival_order_and_nobody_jumps_the_queue`,
`test_a_retried_attempt_of_an_older_group_is_admitted_before_newer_groups`,
`test_a_group_grader_call_takes_one_of_the_same_verification_slots_as_grade`) and
`tests/unit/rl/test_pipeline.py` (`test_sync_admission_stops_across_training_and_publish`,
`test_async_starts_only_groups_that_can_train_fresh`,
`test_the_window_admits_exactly_the_groups_expected_to_drop`,
`test_start_admission_orders_groups_as_they_were_dispatched`,
`test_the_ready_group_sampled_at_the_oldest_policy_trains_first_and_not_stale`). Detail: [reference/scheduling.md](reference/scheduling.md)
and [reference/settings-that-interact.md](reference/settings-that-interact.md).

## One attempt

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/story-1-attempt-dark.svg">
  <img alt="One attempt as a timeline: admit, setup, sampling, seal, grading, cleanup in the background, then group checks. For each phase: what the cookbook does, what user code runs, which clock is running, and what happens if it goes wrong. A retry loops back to admission as a fresh attempt." src="assets/story-1-attempt.svg">
</picture>

The driver, not the runtime, builds the `Rollout`. `sample()` returns only metadata
(`EpisodeResult`); at seal the driver kills the attempt's processes, lets model access
quiesce, seals the lease, and builds the Rollout from what the lease captured
(`runtime/driver.py`, `_seal`). Grading attaches a reward. `finish()` runs in the
background once the graded rollout has joined its group.

## A token's journey

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/story-2-token-dark.svg">
  <img alt="A model call from the agent, directly or over the HTTP gateway, through the Conversation's renderer to token IDs, through the attempt lease to the Loops sampler and back. At seal only delivered samples remain; they become exact-prefix chains, then advantages, then datums whose targets are completion tokens only. The transcript goes only to trace files and the group grader and never becomes training data." src="assets/story-2-token.svg">
</picture>

## One training step

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/story-3-training-step-dark.svg">
  <img alt="A training step: pick groups_per_step usable groups, compute advantages as reward minus group mean, build datums, forward_backward with the CISPO loss, optim_step, publish the weights to the sampler, checkpoint on cadence, record. Sync mode alternates collecting and training; async keeps sampling, bounded by max_steps_off_policy." src="assets/story-3-training-step.svg">
</picture>

`Trainer.commit` is optimizer step + publish; `Trainer.checkpoint` is separate
(`rl/trainer.py`). A step counts the moment its weights are published; a
checkpoint that fails afterwards loses only its artifacts. However the run ends, the
last applied update is saved if the cadence skipped it (`rl/run.py`, `_last_gasp`).

## The rules the code keeps

Break one of these and a test fails. Each names the check that holds it.

1. **Training data is the sampler's own tokens.** Datums are built from captured
   prompt/completion IDs and logprobs only; text is never re-tokenized. Targets exist
   only on trainable completion tokens. — `rl/grpo.py`; `tests/unit/rl/test_grpo.py`
   (`test_linear_rows_train_each_action_once_with_shifted_targets`,
   `test_fork_masks_copied_history_and_drops_all_routes`).
2. **Only delivered generations train.** A reply the agent never fully received
   (an HTTP write that failed, a request in flight at seal) is dropped at seal; the
   rest of the episode trains. — `model_access/lease.py`;
   `test_lease.py::test_a_request_still_in_flight_when_the_attempt_seals_stays_out_of_the_evidence`,
   `test_gateway_transcript.py::test_a_turn_whose_delivery_failed_mid_write_is_marked_undelivered`.
3. **The runtime never hands over evidence.** `sample()` returns metadata; the Rollout
   is built from the lease. The contract's signatures and fields are frozen. —
   `tests/unit/test_contract.py::test_every_runtime_has_exactly_the_normative_signatures_and_nothing_public_besides`.
4. **A cut agent is graded; only the rollout clock discards.** When the runtime's own
   agent deadline fires, the episode is sealed with what was delivered and graded
   (`grading.timed_out`). If the cut left no delivered generation there is nothing to train:
   it is discarded as `unversioned` and counts as a sampling failure, like any rollout
   without trainable evidence. The engine's rollout clock is a backstop: when it fires
   the rollout is discarded (`agent_timeout/rollout_clock`) and counts toward the
   sampling stop. — `tests/unit/harbor/test_trials.py::test_an_agent_deadline_that_cut_its_only_generation_leaves_nothing_to_train`,
   `test_driver.py::test_the_rollout_clock_cancels_sample_and_is_a_non_retried_agent_timeout`,
   `::test_rollout_clock_expiries_between_real_failures_count_toward_the_stop`.
5. **A retry is a fresh attempt, and only a failure that might pass again is
   retried.** New attempt id, runtime state, sandbox, lease and clocks; a failure that
   repeats on every try (non-JSON metadata, an undeclared producer) is tried once. —
   `test_driver.py::test_an_eligible_failure_is_retried_as_a_fresh_attempt_after_cleanup`;
   `tests/unit/episodes/test_runtime.py::test_episode_metadata_that_is_not_json_fails_the_rollout_without_a_retry`.
6. **A hook that ignores cancellation is kept, and the run goes on.** Python can't kill
   a coroutine; one that outlives its grace keeps its attempt, with its permits and
   resources, live until it ends. Its rollout is discarded or its cleanup counted as
   failed. —
   `test_driver.py::test_a_finish_that_ignores_cancellation_is_a_cleanup_failure_and_the_run_goes_on`.
7. **Staleness is bounded everywhere.** `max_steps_off_policy` is checked as samples
   arrive, before grading, before the group judge and at batch selection; a sample from
   a *newer* policy than the trainer's is fatal. — `test_lease.py::test_freshness_is_enforced_during_the_episode`,
   `::test_a_future_policy_version_is_fatal_to_the_run`,
   `test_pipeline.py::test_refresh_filters_stale_members_and_preserves_fresh_siblings`.
8. **A step counts once published.** — `tests/unit/rl/test_run.py::test_a_checkpoint_that_fails_after_publication_leaves_the_step_trained_and_resumable`;
   `tests/unit/rl/test_run.py::test_an_open_ended_run_that_stops_while_collecting_keeps_its_last_update`.
9. **Every billed resource is written down before it exists.** The registry fsyncs a
   creation intent before any provider call and records the outcome after; a crash is
   recovered from the journal. — `tests/unit/resources/test_registry_recovery.py::test_a_crash_after_intent_recovers_by_reconciling_the_identity`.
10. **A create the provider turns away is sent again; one that may exist never is.** A
    refusal (Blaxel's `429`, or a request that never reached it) left nothing, so the
    registry waits, from 1 s doubling to 30 s with jitter and holding no permit, and
    sends the same create within its one deadline. How many of a provider's creates
    are in flight at once is a window that halves on a refusal (once per refused burst)
    and grows by about one per window of creates answered without a refusal (a
    failure or timeout too), from `max_concurrent_provisioning` down to 1, a fractional
    size rounding up to the next whole slot; creates wait for a slot in the order they
    first asked, holding no permit. A wait before a create's first send costs none of
    its deadline. A stop during either wait ends it as never created. —
    `test_registry_backoff.py::test_a_refused_create_is_sent_again_after_a_growing_wait`,
    `::test_a_burst_into_a_shared_limit_converges_to_the_room_left`,
    `::test_the_window_narrows_as_others_load_rises_and_recovers_once_it_falls`,
    `::test_one_burst_of_refusals_halves_the_window_once`,
    `::test_a_halved_window_of_three_still_sends_two_creates_at_once`,
    `::test_answers_that_are_not_refusals_widen_a_narrowed_window_back_to_its_cap`,
    `::test_a_create_waiting_longest_is_sent_first_and_a_retry_keeps_its_place`,
    `::test_a_stop_during_a_wait_ends_it_at_once_as_never_created`,
    `::test_a_stop_while_a_create_waits_for_a_slot_sends_nothing`,
    `::test_every_try_of_a_refused_create_shares_its_one_deadline`,
    `::test_a_wait_for_a_slot_before_the_first_send_costs_none_of_the_deadline`,
    `tests/unit/providers/test_adapter_contract.py::test_a_create_the_provider_turns_away_for_now_is_sent_again`.
11. **Layering.** Core packages (`evidence`, `llm`, `model_access`, `runtime`, `rl`,
    `sft` and the shared modules) import no Harbor or provider code; only a recipe's
    composition roots (`rl/cli.py`, `rl/recipe.py` and SFT's) and the two that wire
    every recipe (`command.py`, `project.py`) do; the two recipes share
    only common modules; no package cycles; Harbor's private members only
    through `harbor/integration.py`, `harbor/blaxel.py`, and `harbor/daytona.py`; each
    provider's SDK only from `providers/<provider>`; no `Any` in public contracts.
    — `tests/unit/test_architecture.py`,
    `tests/unit/test_contract.py::test_the_public_contracts_contain_no_any`.

## Decisions, and why

| Decision | Why | Rejected |
| --- | --- | --- |
| **Meaning is user code; the engine reports facts.** No knobs for what a timeout, error or broken sandbox scores. | Each knob was a silent contract users couldn't see; the same failure means different things per task. | `on_agent_timeout` / `train_on_infrastructure` knobs (built, then deleted, #50). |
| **Evidence comes from the lease, not the runtime.** | A runtime (or a black-box agent over HTTP) can't be trusted to report tokens exactly; the engine can. | Runtime returns a `Rollout` (develop's port). |
| **Retries are opt-in and fresh.** `max_rollout_retries` and the runtime's `grading_retries` default to 0; no failure classifier; `Permanent*Failure` opts out. | A retried attempt must not mix samples with the failed one; classification by message or type name is guessing. | In-place retries; string-matched error classes. |
| **The rollout clock is a derived backstop.** For Episodes it is agent timeout + kill grace + 5 s group exit + cancel grace (`episodes/runtime.py`); Harbor and raw runtimes must set it. | The agent deadline is the meaningful one; a clock you have to keep above it by hand drifts. | A required hand-set clock. |
| **Mechanics deadlines have defaults; meaning-bearing ones don't.** Cleanup 600 s, cancel grace 5 s, kill grace 10 s; setup and the Blaxel TTL stay required. | A setting should exist only for what the cookbook itself does, and state its default. | Seven required deadlines. |
| **State it once.** `model_access` and grading retries come from the runtime; the Blaxel region only from `[providers.blaxel]`; `check` refuses a restatement. | A duplicated fact drifts, then fails at run time. | Class attribute + TOML key. |
| **The engine owns every billed sandbox**, journaled, with `loops cleanup` and a label sweep. | A crash must never lose track of something that bills; sandboxes also carry a required TTL as the last backstop. | Leaving sandboxes to Harbor or the user. |
| **Bound staleness at admission.** Async starts a group only while the groups started and not yet training fit `admission_steps_ahead` batches (at most, and by default, `max_steps_off_policy`), plus the share recent drops say will not train. | Not starting work is cheaper than sampling it and discarding it as stale; admitting fewer batches ahead than the cap lowers lag without dropping the long rollouts a tighter cap would. | Sampling flat out and filtering at the batch; one knob for both the cap and the window. |
| **A refused create is retried in the registry.** | Blaxel's `429` is a concurrent-create limit for the whole workspace, shared with every other client, so no cap of ours avoids it; only the registry knows nothing is in flight while it waits, so a stop can end the wait at once. | Retrying in the Blaxel adapter (a stop would wait out the backoff); leaving it to `max_rollout_retries` (0 by default, so the rollout was discarded). |
| **A create window per provider, TCP-style.** Creates in flight halve on a refusal and grow by ~1 per window of answers that are not refusals, capped at `max_concurrent_provisioning`; image builds share it (one Blaxel limit). | Blaxel's room changes with other teams' load and refusals are near-constant, so the rate has to follow them. Per-create backoffs let ~58 startup creates retry into the limit independently (9-12 refusals each, 18 gave up). | Each create backing off on its own (the herd); a shared gate that closes on a refusal (closed most of the time under that load, and the backlog rushes in when it opens). |
| **Slots measure work, not waiting.** Nothing holds a slot while it queues. | A rollout waiting to be graded would otherwise block agents from running. | A slot per rollout from start to finish. |
| **Pre-start places, only on the full target.** | Sandbox start-up overlaps the previous agent's run; under backpressure it must not start a group the target held back. | Unbounded pre-start; pre-start on a lowered target (an earlier bug). |
| **Weights publish every step; nothing blocking goes on the publish path.** | The sampler must see the new policy; any wait there stalls every rollout. | Readiness checks during weight sync. |
| **Advantage is reward − group mean; the loss is a setting.** | GRPO-style, no std division. `loss_fn` and `loss_fn_config` are the SDK's: one of Loops' policy losses (`cispo`, `importance_sampling`, `ppo`, `dro`, `dppo`), or `custom` through `forward_backward_custom`, scored at every text position so a loss can use tool results. | A loss as a new recipe; a datum flag per loss. |

## Where to look first

| Question | Start at |
| --- | --- |
| Why was this rollout discarded? | `rollouts.jsonl` `cause` = `<reason>/<sub_reason>`; reasons in `runtime/driver.py` (`Discard`), naming in `sub_reason`. |
| How many attempts run at once? | [Scheduling](#scheduling-from-a-task-to-a-batch); `RunCapacity` in `runtime/capacity.py`, `Pipeline._window` in `rl/pipeline.py`. |
| What exactly trained? | `rl/grpo.py` (`build_rows`), `batches/step-N.json` with `trace = "full"`. |
| How does a token get captured? | `model_access/lease.py`, `llm/client.py` (`Conversation`). |
| What does shutdown do, in what order? | `rl/run.py` (`execute`, `_last_gasp`). |

## Changing it safely

- The gates CI runs (`.github/workflows/ci.yml`):
  ```sh
  uv sync --locked --extra harbor-blaxel --extra harbor-daytona --extra wandb
  uv run python scripts/generate_openai_models.py --check
  uv run ruff check src tests patterns examples tasks scripts
  uv run ruff format --check src tests patterns examples tasks scripts
  uv run pyright
  uv run pytest -q
  ```
  plus the tokenizer test at the floor of the `transformers` range and the LiteLLM
  tests in the same job; a job with Harbor absent (`--extra blaxel --extra daytona`
  only), which proves the core runs without it; and a `contracts` job against real
  tokenizers and image processors (`--extra vision`).
- A behaviour change gets a test that fails first, and a mutation check: revert the fix
  and watch the test fail.
- Anything on the generation path or the publish path touches every rollout of every
  run. Look for the mechanism that should already cover the case before adding one.
- Adding a model: a caller's *string* `stop` replaces the renderer's end-of-turn stop
  IDs (one request carries IDs or strings, not both). That is safe only while the
  sampler still ends the turn at the model's own EOS — shown live for Qwen3.6
  (every generation ended on its end-of-turn token with a caller stop set). Probe a new
  model the same way; GLM-5.3 has not been.

## Known gaps

Not blocking; recorded so nobody rediscovers them. A clean Ctrl-C doesn't print whether
cleanup finished. Harbor environment starts can time out under heavy provisioning load
though `check` accepted the bound. A direct call that passes a trajectory budget above
`max_seq_len` can build a datum the trainer refuses, which fails the run.

---

Reference: [how the pieces behave in detail](reference/) ·
[every setting](reference/KNOB_GLOSSARY.md) · [2 · Interfaces](2-interfaces.md) ·
[3 · Patterns](3-patterns.md)
