# 4 · The SFT recipe

For engineers changing the cookbook. How `loops` turns fixed conversations into a
LoRA on Loops, the rules the code keeps, and why. The user-facing contracts, metrics,
run directory and resume are in [reference/sft.md](reference/sft.md); this page doesn't
repeat them.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/sft-flow-dark.svg">
  <img alt="The SFT recipe in two bands. Before anything is provisioned, which loops check runs and the start of loops run repeats: JSONL rows, in the chat or tokens format, are validated against their format's contract (load_rows), and every row that breaks it is counted, the first 20 named by file and line, and no trainer starts. Every row is then rendered with the model's chat template and its targets marked (supervise); a tokens row is taken as given. A row becomes one sequence, one sequence per trained turn when the template rewrites history, or a refusal with its reason. The report (plan) lists what trains, what was split and what was refused, with the digest and the schedule; check prints it and stops, and run stops there too if no row can train, or else writes it to data.json and goes on. Then, once per update: pick batch_size sequences in an order shuffled per epoch (Schedule.batch), build datums where token i predicts token i+1 and only targets are weighted (SupervisedSequence.datum), send them to the Loops trainer as one forward_backward with cross_entropy and one optim_step, with no sampler, then keep sampler weights and trainer state on cadence and always for the last update, with a resume manifest, and record steps.jsonl and W&B every update. train/nll is the token mean from the logprobs Loops returns, whichever reduction the gradient uses; val/nll is forward only. A run that fails or is stopped tries to save the last applied update's state, and keeps it only while the trainer still holds that update; a resume continues after the last update saved. A resume is held to the digest and the schedule." src="assets/sft-flow.svg">
</picture>

## The idea

SFT is the second recipe: no runtime, pattern, sandbox, model access or sampler, just
data and a trainer. It does all its judging before Loops is involved. `plan` loads
every row, holds it to its format's contract, renders it with the trained model's own
chat template and decides its targets, so `loops check` and the start of `loops run`
report exactly the same thing (`run` also writes it to `data.json`), and a run never
meets a bad row an hour in. No Loops trainer starts until every row is rendered,
though a `run` that stops there still leaves its run directory, with `failure.json`
(and a W&B run, if one is configured). The config's `recipe = "sft"` sends both
commands here.
Then the run steps through the epochs: one batch, one `forward_backward`, one
`optim_step`, NLL computed from the logprobs Loops returns. Which tokens train is one
rule over two facts the renderer reports per token (which message owns it, and whether
the template has the model generate it), so no model has code of its own. A renderer
that gives a token the model writes to a message that isn't an assistant turn, or to
none, is refused. A row that rule can't train is refused with its reason, never trained some
other way.

## The parts

| Part | File | Owns |
| --- | --- | --- |
| Data contracts | `sft/data.py` | The `chat` and `tokens` row models (`ChatExample`, `TokenExample`), `load_rows` (every bad row named by `file:line`), fields outside the contract kept aside (`Record.ignored`), the validation split (`load_dataset`). |
| Target rule | `sft/sequences.py` | `supervise`: a row to `SupervisedSequence`s or `Refused`; the per-turn split; the turn terminator; `check_renderer`; the compact sequence and its `datum()`; `digest`. |
| Plan and run | `sft/run.py` | `render_split` (in threads off the event loop, then spawned worker processes), `prepare` and `plan`, `Schedule` (batches, learning rate, `max_steps`, the resume contract), `check_resume`, `Run.execute` (the update loop, checkpoints, the last applied update on the way out), the SFT `Manifest`. |
| Trainer | `sft/trainer.py` | `Trainer`, a `LoopsTrainer` with no sampler: `update` (asking Loops for `loss_aggregation`'s reduction), `evaluate`, `checkpoint`; `score`, the token-mean NLL. It restores a resume's trainer state, or else starts from `[training] init_from`'s weights (and Adam moments with `init_from_optimizer`). `TrainerPort` is what `Run` asks of it. |
| Settings | `sft/config.py` | `Config`: `[data]` and `[eval]` are SFT's own; `[training]` is the shared one plus `lr_schedule`, `warmup_steps`, `batch_size`, `epochs` and `loss_aggregation`; `[model]`, `[compute]` and `[output]` are the shared ones, with `max_seq_len` required. |
| Composition roots | `sft/recipe.py`, `sft/cli.py` | `train` builds the renderer before the run starts; `assemble` builds the trainer (restoring a resume's state) and hands both to `Run`; `check` returns the plan, a resume checked against it, and the first sequence's trained text. `sft/cli.py` is SFT's `cli.Recipe`. |
| Shared with RL | `command.py`, `project.py`, `cli.py`, `trainer.py`, `settings.py`, `run_record.py`, `telemetry.py`, `threads.py`, `cancellation.py`, `llm/` | The `loops` command and the recipe key, config loading, flags and `--resume`, provisioning, checkpoint cadences, resume manifests, failure records, W&B and `run.log`, the cookbook's own threads, cooperative cancellation, the Prime renderers. |

## What a row becomes

`supervise` renders the whole conversation (`render_conversation`) and, for each
trained turn, its generation prompt (`render` of the messages before it). A token is a
target when it is generated, owned by a trained turn, and past that turn's prompt.
The role marker that ends a turn (GLM's `<|observation|>` and `<|user|>`) is generated,
and the renderer gives it to the turn it ends; `_conversation` refuses a render that
gives any generated token to a user, tool or system message, or to none.

- **One sequence**, when every trained turn's prompt is a token prefix of the whole
  render.
- **One sequence per trained turn**, when one is not (the template rewrites history,
  as Qwen3.6 and Nemotron drop earlier reasoning). Each turn is rendered as the last
  message; if even that does not start with its own prompt, because the tokenizer
  merged a token across the prompt's end, `_written_after` re-tokenizes the turn's
  text from where the prompt ends, as the model would write it.
- **A sequence ends at its last target**: whatever the template renders after it is
  never read, so it is cut.
- **A final turn with no stop token in it** gets the terminator the template renders
  for the message that would follow (`_terminator`); none at all is a refusal.
- **Refused**, with the reason, for anything else: over `max_seq_len`, no trained
  stop, a generated token given to a message that isn't an assistant turn, a turn
  that renders differently as the latest turn than after its own generation prompt
  (the template's thinking-off option often fixes this), template text the model
  doesn't write inside a re-tokenized turn, or a `RenderingError`. A `tokens`
  row is never rendered; it is checked only against the budget.

`render_split` renders here, a batch of rows at a time in a thread of the cookbook's
own (`BATCH_SECONDS`, 0.2 s, at least one row) so the event loop stays free. With more
than one worker (`render_workers`, by default the usable cores less one, at most 4), a
split still rendering after `SERIAL_SECONDS` (5 s) hands the rest to spawned worker
processes, each holding a copy of the renderer. If the renderer doesn't pickle, or the pool breaks, the rest renders
here. A cancellation or SIGTERM stops between batches and rows, lets active workers
finish, and joins them before the process exits. A sequence is held as packed ids, a
byte per token for whether it trains, and its text chunks as lengths; datums are built
only as each batch is sent.

## The rules the code keeps

Break one of these and a test fails. Unit tests use a template whose every token is
known (`tests/unit/sft/conftest.py`, `ChatRenderer`); the contract tests run the real
Qwen3.6, GLM-5.3 and Nemotron renderers.

1. **A turn trains its own tokens and its stop, never its header or its prompt; a
   token the model writes belongs to the turn that writes it, or the row is
   refused.** —
   `tests/unit/sft/test_sequences.py::test_a_turn_trains_its_content_and_stop_but_never_its_header`,
   `::test_a_turn_ending_marker_trains_with_its_turn_and_only_when_the_turn_owns_it`;
   `tests/contracts/test_sft_sequences.py::test_each_trained_turn_reads_exactly_its_own_generation_prompt`,
   `::test_a_context_only_turn_trains_nothing`.
2. **Every trained turn learns to end, or the row is refused.** —
   `test_sft_sequences.py::test_every_trained_turn_ends_on_exactly_one_stop_token`;
   `test_sequences.py::test_a_turn_that_could_never_learn_to_end_is_refused`.
3. **A turn trains in the context it was generated in, as the model writes it.** —
   `test_sft_sequences.py::test_history_rendering_decides_whether_turns_train_together`,
   `::test_turns_without_reasoning_train_as_the_model_writes_them`;
   `test_sequences.py::test_a_turn_merged_across_its_prompt_trains_what_the_model_writes`,
   `::test_template_text_the_model_does_not_write_is_never_trained_when_merged`,
   `::test_template_text_is_never_trained_when_the_turn_ends_with_the_next_message`.
4. **A row the rule can't train is reported, never trained some other way, and
   nothing trainable means no trainer.** — `test_sequences.py::test_a_row_over_the_budget_is_refused`;
   `tests/unit/sft/test_run.py::test_rows_that_cannot_train_are_reported_not_trained`,
   `::test_a_dataset_with_nothing_trainable_fails_before_the_trainer_starts`.
5. **Position `i` predicts token `i + 1`, weighted only where it is a target.** —
   `test_sft_sequences.py::test_a_datum_predicts_each_trained_token_from_everything_before_it`;
   `test_sequences.py::test_a_tokens_row_trains_exactly_its_weights_with_no_renderer`.
6. **Every bad row is counted at once, the first 20 named by file and line; extra
   fields are reported, not trained.** — `tests/unit/sft/test_data.py::test_a_chat_row_breaking_the_contract_is_named_by_file_and_line`,
   `::test_a_tokens_row_breaking_the_contract_is_named_by_file_and_line`,
   `::test_every_bad_row_is_reported_at_once`,
   `::test_a_real_world_export_loads_with_its_extras_kept_aside`;
   `test_run.py::test_fields_outside_the_contract_are_reported_not_trained`.
7. **Where a row renders never changes what it becomes, and a stop leaves no
   worker behind.** —
   `test_run.py::test_rows_rendered_in_worker_processes_become_what_they_would_here`,
   `::test_a_renderer_that_does_not_pickle_renders_in_this_process`,
   `::test_rows_the_workers_could_not_render_render_in_this_process`;
   `tests/unit/sft/test_project.py::test_check_keeps_the_loop_responsive_and_settles_preparation`,
   `::test_cancelled_check_finishes_active_rows_and_stops_before_the_rest`;
   `tests/unit/sft/test_cli.py::test_sigterm_during_preparation_joins_workers_before_exit`.
8. **The digest names exactly what trains, and doesn't drift.** It covers each row's
   file name and line too, so renaming the data file breaks a resume. Golden digests
   per case, recorded before sequences were held compactly; the datum hashes beside
   them were re-recorded for baseten-loops 0.33, the digests unchanged. —
   `test_run.py::test_what_trains_and_its_digest_are_what_runs_recorded`;
   `test_sequences.py::test_the_digest_tells_apart_inputs_with_the_same_tokens`,
   `::test_the_digest_tells_apart_a_held_out_row_moved_into_training`.
9. **One update is one `forward_backward` then one `optim_step`; NLL is a token
   mean.** The `forward_backward` states `loss_aggregation`'s reduction, and a result
   that doesn't echo it back is refused. Untrained positions come back null and are
   skipped; a trained one without a finite logprob is an error. Validation never takes
   a gradient. —
   `tests/unit/sft/test_trainer.py::test_an_update_accumulates_the_batch_then_steps_once`,
   `::test_an_update_sends_its_rows_and_asks_loops_for_its_reduction`,
   `tests/unit/test_loops_trainer.py::test_an_update_loops_does_not_confirm_the_reduction_of_is_refused`;
   `test_trainer.py::test_nll_is_the_mean_over_every_trained_token_not_over_datums`,
   `::test_an_untrained_position_returned_as_null_is_skipped`,
   `::test_a_trained_position_without_a_logprob_is_an_error`,
   `::test_validation_runs_forward_only`.
10. **Each epoch trains every sequence once in its own order, at its scheduled
    learning rate, and `max_steps` counts the run's updates, a resume's included.** —
    `test_run.py::test_each_epoch_trains_every_sequence_once_in_its_own_order`,
    `::test_learning_rate_warms_up_then_follows_its_schedule`,
    `::test_max_steps_stops_mid_epoch_and_0_runs_every_epoch`,
    `::test_a_resume_at_max_steps_completes_without_training`,
    `::test_a_constant_learning_rate_can_resume_for_longer`.
11. **A resume trains the same batches at the same learning rates, or is refused,
    without touching the first run's records; `check` refuses what `run` refuses, and
    a finished run trains nothing more.** Only what decides each update's batch and
    learning rate is held; the resume's own config is recorded beside the first. —
    `test_run.py::test_a_resumed_run_continues_where_its_manifest_left_off`,
    `::test_resume_refuses_data_that_changed`,
    `::test_resume_refuses_a_different_order_of_the_same_data`,
    `::test_resume_refuses_a_different_learning_rate_for_the_same_step`,
    `::test_a_refused_resume_leaves_the_first_runs_records_as_they_were`,
    `::test_a_resumed_run_keeps_the_config_it_started_with`,
    `::test_a_manifest_from_another_recipe_is_refused`,
    `::test_a_constant_learning_rate_can_resume_for_longer`,
    `::test_a_resume_after_the_last_epoch_completes_without_training`,
    `::test_a_resume_at_max_steps_completes_without_training`,
    `::test_a_finished_run_says_how_to_train_more`,
    `::test_a_resume_with_no_update_left_is_still_held_to_its_run`;
    `test_cli.py::test_check_holds_a_resume_to_what_its_run_trained`.
12. **No update is applied twice, and a failing run tries to keep its last one.** The
    save on the way out is best-effort: it is skipped when the trainer no longer holds
    that update (restarted, or its version never read) or fails or times out, and a
    resume then continues after the last update saved. A failure under a Loops run
    that ended is recorded as `loops_run_ended`. —
    `test_run.py::test_the_last_update_is_always_kept_with_a_resume_manifest`,
    `::test_a_failed_run_records_why_and_keeps_its_last_update`,
    `::test_a_run_failing_after_an_update_resumes_after_it_not_at_it`,
    `::test_a_failure_after_the_update_was_kept_needs_no_last_gasp`,
    `::test_a_failure_under_an_ended_loops_run_is_reported_as_such`,
    `::test_a_cancellation_while_loops_answers_keeps_the_record_and_the_last_update`;
    `test_loops_trainer.py::test_a_trainer_that_restarted_is_not_kept_on_the_way_out`,
    `::test_an_update_on_a_restarted_trainer_is_refused_and_not_kept`,
    `::test_an_update_whose_version_was_never_read_keeps_nothing_after_it`,
    `::test_a_trainer_whose_version_cannot_be_read_is_not_kept`,
    `::test_a_way_out_that_cannot_save_keeps_nothing_and_does_not_raise`,
    `::test_a_stalled_version_read_does_not_hold_up_the_way_out`.
13. **A renderer SFT can't train with is refused before any row, and before a run
    starts.** A template that writes tool call ids is one. —
    `test_sequences.py::test_a_template_that_writes_tool_call_ids_is_refused`,
    `::test_a_renderer_missing_what_training_needs_is_refused`;
    `test_cli.py::test_a_renderer_that_cannot_be_built_is_a_config_error_and_starts_no_run`.
14. **The recipes share only common modules.** `sft` imports nothing from `rl`, and
    only composition roots (each recipe's `cli` and `recipe`, plus `command` and
    `project`) import one. — `tests/unit/test_architecture.py::test_recipes_share_only_through_common_modules`,
    `::test_only_a_composition_root_imports_one`.

## Decisions, and why

| Decision | Why | Rejected |
| --- | --- | --- |
| **One target rule over the renderer's per-token facts.** | Every model's quirks (where a turn ends, what is prefilled, what history drops) show up as ownership and generation, so one rule covers them and a new model needs a contract test, not code. | A role filter on the renderer's mask; guessing which turn a generated token on a user or tool message ends (it also trained whatever a custom renderer mislabelled); code per model. |
| **Render the whole dataset before provisioning.** | `check` and `run` must agree, and a bad row found at hour three wastes the run. | Rendering lazily as batches are drawn. |
| **Refuse, never truncate or improvise.** | A truncated turn or one with no stop trains the model never to finish, and nothing in the loss shows it. | Cutting at `max_seq_len`; appending an EOS the template doesn't render. |
| **Split per turn when history is rewritten.** | In the full render an earlier turn sits in a context it never saw; training it there teaches the wrong thing. | Training the whole conversation as one sequence regardless. |
| **NLL from the returned logprobs, not the SDK's `loss`.** | A token mean is comparable across batches, runs and both reductions; the SDK's `loss` follows the reduction and averages per-chunk means by datum count when it splits a request. | Reading the SDK's `loss` field. |
| **`loss_aggregation` defaults to `sum`, Loops' own default, and every update states it.** | The gradient is what Loops does unless asked otherwise; `token_mean` asks it to divide by the update's trained tokens. Stating it, and refusing a result that doesn't echo it, means a trainer that reduces differently can't train silently. | Assuming Loops averages, which it no longer does by default. |
| **Two formats, named once by `[data] format`.** | `chat` is what exports already are; `tokens` covers what a template can't express (targets finer than a turn), with the responsibility stated. | Inferring the format from each row. |
| **Fields outside the contract are kept aside and listed.** | Real exports carry ids and columns of their own; refusing them makes every user clean the data first, and dropping them silently hides a misspelt field. | Refusing unknown fields; ignoring them without a word. |
| **A resume is held to the digest and the schedule's contract.** | `next_step` must mean the same batch at the same learning rate, or the resumed run is a different run with the old one's name. The update count joins the contract only when the rate decays towards it. | Resuming on changed data, order or schedule. |
| **Refuse templates that render tool call ids.** | Ids in data from elsewhere would train in a format the model's own parser can't read. | Training the ids as written. |
| **No sampler; the Loops run is left up unless `[compute] keep_run = false`.** | SFT trains on fixed data and serves nothing; on dedicated Loops a live run lets `reuse_from_run_id` start the next one warm. The log warns, with its ID, that it is still up and names `loops deactivate`; a dedicated run bills, a serverless one holds a workspace slot. With `keep_run = false` the trainer deactivates it on close, giving up after 30 s and warning instead. | Provisioning a sampler per run; always deactivating on exit. |
| **Render in spawned workers, after a few seconds.** | Long agent traces render for hours on one core; a small dataset isn't worth starting processes for. Spawned, because a fork would copy the tokenizer's and telemetry's threads mid-flight. | Forked workers; always serial; always parallel. |

## Where to look first

| Question | Start at |
| --- | --- |
| Why was this row refused? | `loops check`'s report, or `data.json` after `run` (`refusals`, with reasons); the `Refused` raises in `sft/sequences.py` (`supervise`, `_conversation`, `_sequence`, `_written_after`, `_terminator`). |
| What exactly trains? | `loops check` prints the first sequence's trained text, span by span; `SupervisedSequence.datum`. |
| Why was a conversation split per turn? | The prefix test in `supervise`; `check` counts the splits. |
| Why did a resume refuse, or train nothing? | `check_resume`, `Schedule.contract`, `digest`; `Schedule.total_steps` and `run_record.finished`. |
| Where do `train/nll` and `val/nll` come from? | `score` in `sft/trainer.py`. |
| Why is `check` slow, or using memory? | `render_split`, `[data] render_workers`. |

## Changing it safely

- The CI gates are doc 1's ([Changing it safely](1-engine.md#changing-it-safely)).
  The SFT unit tests run in the main job; the real-template tests in
  `tests/contracts/test_sft_sequences.py` skip without the model files, so run them
  with `LOOPS_DOWNLOAD_CONTRACT_MODELS=1` (as the `contracts` job does) after touching
  `sequences.py` or the renderers.
- Anything that changes the digest breaks every existing resume manifest.
  `test_what_trains_and_its_digest_are_what_runs_recorded` pins it; update its golden
  values only when what trains is meant to change.
- Adding a model: no SFT code. Pin its tokenizer in `tests/contracts/conftest.py`,
  add it to the contract tests' `build` and `family` fixture, and see the rules hold;
  a renderer that breaks them needs a fix or an extension in `llm/renderer_extensions/`, not SFT code.
- A run's statuses reach the exit code (`cli.py`: 0 completed, 75 `loops_run_ended`,
  1 otherwise; 143 on SIGTERM), and a resume writes its own `config-`, `data-`, `run-`
  and `failure-from-step-N.json` beside the first run's. Change either and check
  `run_record.py` and the reference.
- A new setting needs a description and lands in the knob glossary
  (`python -m loops_cookbook.docs`; `tests/unit/test_docs.py::test_every_setting_is_documented_exactly_once`).
- No test runs `loops check` on the shipped `examples/sft_*/sft.toml`;
  `tests/test_shipped_configs.py` covers `train*.toml` (RL) only. Run it by hand when a
  setting or the data contract changes.

## Known gaps

Not blocking; recorded so nobody rediscovers them. Trace importers (LangSmith,
Langfuse) are not built; the chat contract is the seam they would write to. DPO, in
the SFT recipe beside SFT, is not started. Evaluating a checkpoint on real rollouts is on
the roadmap; `examples/sft_toucan/evaluate.py` scores first-turn agreement with the
data meanwhile. Templates that render tool call ids (Kimi K2) are refused rather than
having the dataset's ids rewritten into the model's format. `loops check` on an SFT
config does not ask Loops whether it serves the model at `max_seq_len`, as it does for
RL (`check_served_model`); SFT finds out when `run` provisions.

---

Reference: [the SFT contracts, metrics and resume](reference/sft.md) ·
[every setting](reference/KNOB_GLOSSARY.md#sft-settings) ·
[1 · The training engine](1-engine.md) · [ROADMAP](../ROADMAP.md)
