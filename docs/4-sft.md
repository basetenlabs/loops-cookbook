# 4 · The SFT recipe

This page is for engineers who change the cookbook. It explains how `loops` turns fixed
conversations into a trained low-rank adaptation (LoRA) adapter on Loops, the rules the
code keeps, and why. [reference/sft.md](reference/sft.md) describes what a user sees,
which is the data contracts, the metrics, the run directory and resume. This page
doesn't repeat it.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/sft-flow-dark.svg">
  <img alt="The SFT recipe in two bands. Before anything is provisioned, which loops check runs and the start of loops run repeats: JSONL rows, in the chat or tokens format, are validated against their format's contract (load_rows), and every row that breaks it is counted, the first 20 named by file and line, and no trainer starts. Every row is then rendered with the model's chat template and its targets marked (supervise); a tokens row is taken as given. A row becomes one sequence, one sequence per trained turn when the template rewrites history, or a refusal with its reason. The report (plan) lists what trains, what was split and what was refused, with the digest and the schedule; check prints it and stops, and run stops there too if no row can train, or else writes it to data.json and goes on. Then, once per update: pick batch_size sequences in an order shuffled per epoch (Schedule.batch), build datums where token i predicts token i+1 and only targets are weighted (SupervisedSequence.datum), send them to the Loops trainer as one forward_backward with cross_entropy and one optim_step, with no sampler, then keep sampler weights and trainer state on cadence and always for the last update, with a resume manifest, and record steps.jsonl and W&B every update. train/nll is the token mean from the logprobs Loops returns, whichever reduction the gradient uses; val/nll is forward only. A run that fails or is stopped tries to save the last applied update's state, and keeps it only while the trainer still holds that update; a resume continues after the last update saved. A resume is held to the digest and the schedule." src="assets/sft-flow.svg">
</picture>

## The idea

Supervised fine-tuning (SFT) is the second recipe. A config chooses it with
`recipe = "sft"`, and both `loops check` and `loops run` then hand the config to it.
SFT has no rollout runtime, pattern, sandbox, model access or sampler. It has data and a
trainer.

SFT judges all of its data before Loops is involved. `plan` loads every row and holds it
to its format's contract. Then it renders the row with the trained model's own chat
template, and it decides which tokens of the row train. Those tokens are the row's
targets. `loops check` and the start of `loops run` both run `plan`, so they report
exactly the same thing. `run` also writes the report to `data.json`. A run therefore
never meets a bad row an hour in.

No Loops trainer starts until every row is rendered. A `run` that stops at that point
still leaves its run directory, with `failure.json`, and it leaves a W&B run if one is
configured.

After `plan`, the run steps through the epochs. Each update takes one batch, one
`forward_backward` and one `optim_step`. The run computes the negative log-likelihood
(NLL) from the log probabilities that Loops returns.

One rule decides which tokens train, for every model. It reads two facts that the
renderer reports for each token. The first is which message owns the token. The second
is whether the template has the model generate it. Because the rule needs only these
facts, no model has SFT code of its own. The rule refuses a renderer that gives a
generated token to a message that isn't an assistant turn, or to no message at all. It
also refuses, with a reason, any row that it can't train. It never trains such a row
some other way.

## The parts

| Part | File | What it does |
| --- | --- | --- |
| Data contracts | `sft/data.py` | It defines the `chat` and `tokens` row models, `ChatExample` and `TokenExample`. `load_rows` loads the rows and names every bad row by `file:line`. Fields outside the contract are kept aside in `Record.ignored`. `load_dataset` makes the validation split. |
| Target rule | `sft/sequences.py` | `supervise` turns a row into `SupervisedSequence` values, or into `Refused` with a reason. The module also owns the per-turn split, the turn terminator, `check_renderer`, the compact form of a sequence and its `datum()`, and `digest`. |
| Plan and run | `sft/run.py` | `render_split` renders the rows, first in threads off the event loop and then in spawned worker processes. The module also owns `prepare` and `plan`, and `Schedule`, which sets the batches, the learning rate, `max_steps` and the resume contract. It owns `check_resume`, the SFT `Manifest`, and `Run.execute`, which runs the update loop, saves checkpoints and keeps the last applied update on the way out. |
| Trainer | `sft/trainer.py` | `Trainer` is a `LoopsTrainer` with no sampler. Its `update` asks Loops for the reduction that `loss_aggregation` names, and it also has `evaluate` and `checkpoint`. `score` computes the token-mean NLL. The trainer restores a resume's trainer state. Otherwise it starts from the weights in `[training] init_from`, and also from that checkpoint's Adam moments when `init_from_optimizer` is set. `TrainerPort` is what `Run` asks of the trainer. |
| Settings | `sft/config.py` | `Config` holds the settings. `[data]` and `[eval]` are SFT's own. `[training]` is the shared table plus `lr_schedule`, `warmup_steps`, `batch_size`, `epochs` and `loss_aggregation`. `[model]`, `[compute]` and `[output]` are the shared tables, and SFT requires `max_seq_len`. |
| Composition roots | `sft/recipe.py`, `sft/cli.py` | `train` builds the renderer before the run starts. `assemble` builds the trainer, restoring a resume's state, and hands the renderer and the trainer to `Run`. `check` returns the plan, a resume checked against it, and the first sequence's trained text. `sft/cli.py` is SFT's `cli.Recipe`. |
| Shared with RL | `command.py`, `project.py`, `cli.py`, `trainer.py`, `settings.py`, `run_record.py`, `telemetry.py`, `threads.py`, `cancellation.py`, `llm/` | These modules own the `loops` command and the `recipe` key, config loading, flags and `--resume`, and provisioning. They also own checkpoint cadences, resume manifests, failure records, W&B and `run.log`, the cookbook's own threads, cooperative cancellation and the Prime renderers. |

## What a row becomes

`supervise` renders the whole conversation with `render_conversation`. For each trained
turn, it also renders that turn's generation prompt, which is `render` of the messages
before the turn. A token is a target when it meets three conditions:

- the template has the model generate it
- a trained turn owns it
- it comes after that turn's prompt

The role marker that ends a turn is generated, such as GLM's `<|observation|>` or
`<|user|>`, and the renderer gives it to the turn it ends. `_conversation` refuses a
render that gives any generated token to a user, tool or system message, or to no
message.

A row then becomes one of these:

- **One sequence**, when every trained turn's prompt is a token prefix of the whole
  render.
- **One sequence per trained turn**, when some prompt is not a prefix. This happens when
  the template rewrites history. For example, Qwen3.6 and Nemotron drop earlier
  reasoning. Each turn is then rendered as the last message. Sometimes even that render
  doesn't start with the turn's own prompt, because the tokenizer merged a token across
  the prompt's end. In that case, `_written_after` re-tokenizes the turn's text from
  where the prompt ends, as the model would write it.
- **A refusal, with its reason**, for anything else. The reasons are these:
  - the row is over `max_seq_len`
  - a trained turn has no stop token
  - a generated token belongs to a message that isn't an assistant turn
  - a turn renders differently as the latest turn than after its own generation
    prompt, which the template's option to turn thinking off often fixes
  - a re-tokenized turn contains template text that the model doesn't write
  - the renderer raised a `RenderingError`

Two more rules shape each sequence. A sequence ends at its last target, because nothing
reads what the template renders after that target, so the sequence is cut there. A final
turn that contains no stop token gets the terminator that the template renders for the
message that would follow it (`_terminator`). If the template renders no terminator, the
row is refused.

A `tokens` row is never rendered. It is checked only against the token budget.

`render_split` does the rendering. It renders a batch of rows at a time, in a thread of
the cookbook's own, so that the event loop stays free. A batch lasts `BATCH_SECONDS`,
which is 0.2 s, and holds at least one row. `[data] render_workers` sets how many
worker processes may help. By default it is the number of cores the process may use,
less one, and at most 4. With more than one worker, a split that is still rendering
after `SERIAL_SECONDS`, which is 5 s, hands the rest of its rows to spawned worker
processes. Each worker holds its own copy of the renderer. If the renderer can't be
pickled, or the pool breaks, the rest of the rows render in the main process.

A cancellation or a SIGTERM stops rendering between batches and between rows. It lets
active workers finish, and it joins them before the process exits.

A sequence is held compactly. Its token IDs are packed, one byte per token says whether
the token trains, and its text chunks are held as lengths. The run builds datums from a
batch's sequences only as it sends that batch.

## The rules the code keeps

A test holds each rule below. If you break a rule, its test fails. The unit tests use a
template whose every token is known, `ChatRenderer` in `tests/unit/sft/conftest.py`. The
contract tests run the real Qwen3.6, GLM-5.3 and Nemotron renderers.

1. **A turn trains its own tokens and its stop. It never trains its header or its
   prompt. A token that the model writes belongs to the turn that writes it, or the row
   is refused.** The tests are
   `tests/unit/sft/test_sequences.py::test_a_turn_trains_its_content_and_stop_but_never_its_header`,
   `test_sequences.py::test_a_turn_ending_marker_trains_with_its_turn_and_only_when_the_turn_owns_it`,
   `tests/contracts/test_sft_sequences.py::test_each_trained_turn_reads_exactly_its_own_generation_prompt`
   and
   `test_sft_sequences.py::test_a_context_only_turn_trains_nothing`.
2. **Every trained turn learns to end, or the row is refused.** The tests are
   `test_sft_sequences.py::test_every_trained_turn_ends_on_exactly_one_stop_token`
   and
   `test_sequences.py::test_a_turn_that_could_never_learn_to_end_is_refused`.
3. **A turn trains in the context it was generated in, as the model writes it.** The
   tests are
   `test_sft_sequences.py::test_history_rendering_decides_whether_turns_train_together`,
   `test_sft_sequences.py::test_turns_without_reasoning_train_as_the_model_writes_them`,
   `test_sequences.py::test_a_turn_merged_across_its_prompt_trains_what_the_model_writes`,
   `test_sequences.py::test_template_text_the_model_does_not_write_is_never_trained_when_merged`
   and
   `test_sequences.py::test_template_text_is_never_trained_when_the_turn_ends_with_the_next_message`.
4. **A row that the rule can't train is reported, and never trained some other way. When
   nothing can train, no trainer starts.** The tests are
   `test_sequences.py::test_a_row_over_the_budget_is_refused`,
   `tests/unit/sft/test_run.py::test_rows_that_cannot_train_are_reported_not_trained`
   and
   `test_run.py::test_a_dataset_with_nothing_trainable_fails_before_the_trainer_starts`.
5. **Position `i` predicts token `i + 1`, and it is weighted only where that token is a
   target.** The tests are
   `test_sft_sequences.py::test_a_datum_predicts_each_trained_token_from_everything_before_it`
   and
   `test_sequences.py::test_a_tokens_row_trains_exactly_its_weights_with_no_renderer`.
6. **Every bad row is counted at once, and the first 20 are named by file and line.
   Extra fields are reported, not trained.** The tests are
   `tests/unit/sft/test_data.py::test_a_chat_row_breaking_the_contract_is_named_by_file_and_line`,
   `test_data.py::test_a_tokens_row_breaking_the_contract_is_named_by_file_and_line`,
   `test_data.py::test_every_bad_row_is_reported_at_once`,
   `test_data.py::test_a_real_world_export_loads_with_its_extras_kept_aside`
   and
   `test_run.py::test_fields_outside_the_contract_are_reported_not_trained`.
7. **Where a row renders never changes what it becomes, and a stop leaves no worker
   behind.** The tests are
   `test_run.py::test_rows_rendered_in_worker_processes_become_what_they_would_here`,
   `test_run.py::test_a_renderer_that_does_not_pickle_renders_in_this_process`,
   `test_run.py::test_rows_the_workers_could_not_render_render_in_this_process`,
   `tests/unit/sft/test_project.py::test_check_keeps_the_loop_responsive_and_settles_preparation`,
   `test_project.py::test_cancelled_check_finishes_active_rows_and_stops_before_the_rest`
   and
   `tests/unit/sft/test_cli.py::test_sigterm_during_preparation_joins_workers_before_exit`.
8. **The digest names exactly what trains, and it doesn't drift.** The digest also
   covers each row's file name and line, so renaming the data file breaks a resume. The
   test holds golden digests for each case, and a hash of every datum sent beside them.
   The tests are
   `test_run.py::test_what_trains_and_its_digest_are_what_runs_recorded`,
   `test_sequences.py::test_the_digest_tells_apart_inputs_with_the_same_tokens`
   and
   `test_sequences.py::test_the_digest_tells_apart_a_held_out_row_moved_into_training`.
9. **One update is one `forward_backward` and then one `optim_step`, and NLL is a token
   mean.** The `forward_backward` call states the reduction that `loss_aggregation`
   names, and the trainer refuses a result that doesn't echo it back. Untrained
   positions come back null and are skipped. A trained position without a finite log
   probability is an error. Validation never takes a gradient. The tests are
   `tests/unit/sft/test_trainer.py::test_an_update_accumulates_the_batch_then_steps_once`,
   `test_trainer.py::test_an_update_sends_its_rows_and_asks_loops_for_its_reduction`,
   `tests/unit/test_loops_trainer.py::test_an_update_loops_does_not_confirm_the_reduction_of_is_refused`,
   `test_trainer.py::test_nll_is_the_mean_over_every_trained_token_not_over_datums`,
   `test_trainer.py::test_an_untrained_position_returned_as_null_is_skipped`,
   `test_trainer.py::test_a_trained_position_without_a_logprob_is_an_error`
   and
   `test_trainer.py::test_validation_runs_forward_only`.
10. **Each epoch trains every sequence once, in an order of its own, at the scheduled
    learning rate. `max_steps` counts the run's updates, including a resume's.** The
    tests are
    `test_run.py::test_each_epoch_trains_every_sequence_once_in_its_own_order`,
    `test_run.py::test_learning_rate_warms_up_then_follows_its_schedule`,
    `test_run.py::test_max_steps_stops_mid_epoch_and_0_runs_every_epoch`,
    `test_run.py::test_a_resume_at_max_steps_completes_without_training`
    and
    `test_run.py::test_a_constant_learning_rate_can_resume_for_longer`.
11. **A resume trains the same batches at the same learning rates, or it is refused
    without touching the first run's records. `check` refuses what `run` refuses, and a
    finished run trains nothing more.** The resume contract holds only what decides
    each update's batch and learning rate. The run records the resume's own config
    beside the first run's. The tests are
    `test_run.py::test_a_resumed_run_continues_where_its_manifest_left_off`,
    `test_run.py::test_resume_refuses_data_that_changed`,
    `test_run.py::test_resume_refuses_a_different_order_of_the_same_data`,
    `test_run.py::test_resume_refuses_a_different_learning_rate_for_the_same_step`,
    `test_run.py::test_a_refused_resume_leaves_the_first_runs_records_as_they_were`,
    `test_run.py::test_a_resumed_run_keeps_the_config_it_started_with`,
    `test_run.py::test_a_manifest_from_another_recipe_is_refused`,
    `test_run.py::test_a_constant_learning_rate_can_resume_for_longer`,
    `test_run.py::test_a_resume_after_the_last_epoch_completes_without_training`,
    `test_run.py::test_a_resume_at_max_steps_completes_without_training`,
    `test_run.py::test_a_finished_run_says_how_to_train_more`,
    `test_run.py::test_a_resume_with_no_update_left_is_still_held_to_its_run`
    and
    `test_cli.py::test_check_holds_a_resume_to_what_its_run_trained`.
12. **No update is applied twice, and a failing run tries to keep its last update.** The
    save on the way out is best-effort. The run skips it when the trainer no longer
    holds that update, because the trainer restarted or its version was never read. The
    save can also fail or time out. A resume then continues after the last update that
    was saved. A failure under a Loops run that has ended is recorded as
    `loops_run_ended`. The tests are
    `test_run.py::test_the_last_update_is_always_kept_with_a_resume_manifest`,
    `test_run.py::test_a_failed_run_records_why_and_keeps_its_last_update`,
    `test_run.py::test_a_run_failing_after_an_update_resumes_after_it_not_at_it`,
    `test_run.py::test_a_failure_after_the_update_was_kept_needs_no_last_gasp`,
    `test_run.py::test_a_failure_under_an_ended_loops_run_is_reported_as_such`,
    `test_run.py::test_a_cancellation_while_loops_answers_keeps_the_record_and_the_last_update`,
    `test_loops_trainer.py::test_a_trainer_that_restarted_is_not_kept_on_the_way_out`,
    `test_loops_trainer.py::test_an_update_on_a_restarted_trainer_is_refused_and_not_kept`,
    `test_loops_trainer.py::test_an_update_whose_version_was_never_read_keeps_nothing_after_it`,
    `test_loops_trainer.py::test_a_trainer_whose_version_cannot_be_read_is_not_kept`,
    `test_loops_trainer.py::test_a_way_out_that_cannot_save_keeps_nothing_and_does_not_raise`
    and
    `test_loops_trainer.py::test_a_stalled_version_read_does_not_hold_up_the_way_out`.
13. **A renderer that SFT can't train with is refused before any row is rendered, and
    before a run starts.** A template that writes tool call IDs is one example. The
    tests are
    `test_sequences.py::test_a_template_that_writes_tool_call_ids_is_refused`,
    `test_sequences.py::test_a_renderer_missing_what_training_needs_is_refused`
    and
    `test_cli.py::test_a_renderer_that_cannot_be_built_is_a_config_error_and_starts_no_run`.
14. **The recipes share only common modules.** `sft` imports nothing from `rl`. Only the
    composition roots import a recipe. These are each recipe's `cli` and `recipe`
    modules, plus `command` and `project`. The tests are
    `tests/unit/test_architecture.py::test_recipes_share_only_through_common_modules`
    and
    `test_architecture.py::test_only_a_composition_root_imports_one`.

## Decisions, and why

| Decision | Why | Rejected |
| --- | --- | --- |
| **One target rule reads the renderer's facts about each token.** | Every model's quirks show up as which message owns a token and whether the model generates it. Examples are where a turn ends, what the template prefills and what history it drops. So one rule covers every model, and a new model needs a contract test, not code. | A filter by role on the renderer's mask. Guessing which turn a generated token on a user or tool message ends, which also trained whatever a custom renderer mislabelled. Code for each model. |
| **The whole dataset is rendered before anything is provisioned.** | `check` and `run` must agree. A bad row found at hour three wastes the run. | Rendering lazily, as batches are drawn. |
| **A row is refused, never truncated or patched.** | A truncated turn, or a turn with no stop, trains the model never to finish, and nothing in the loss shows it. | Cutting at `max_seq_len`. Appending an end-of-sequence (EOS) token that the template doesn't render. |
| **A conversation is split per turn when its history is rewritten.** | In the full render, an earlier turn sits in a context it never saw, and training it there teaches the wrong thing. | Training the whole conversation as one sequence regardless. |
| **NLL comes from the returned log probabilities, not from the SDK's `loss`.** | A token mean is comparable across batches, across runs and across both reductions. The SDK's `loss` follows the reduction. When the SDK splits a request, it also averages the chunks' means by datum count. | Reading the SDK's `loss` field. |
| **`loss_aggregation` defaults to `sum`, which is Loops' own default, and every update states it.** | The gradient is what Loops computes unless the trainer asks for something else. `token_mean` asks Loops to divide by the update's trained tokens. The trainer states the reduction and refuses a result that doesn't echo it, so a trainer that reduces differently can't train silently. | Assuming that Loops averages. It doesn't by default. |
| **There are two data formats, and `[data] format` names the one a config uses.** | `chat` is the form that exports already have. `tokens` covers what a template can't express, such as targets finer than a turn, and the user takes responsibility for its tokens. | Inferring the format from each row. |
| **Fields outside the contract are kept aside and listed.** | Real exports carry IDs and columns of their own. Refusing them makes every user clean the data first. Dropping them without a word hides a misspelt field. | Refusing unknown fields. Ignoring them silently. |
| **A resume is held to the digest and to the schedule's contract.** | `next_step` must mean the same batch at the same learning rate. Otherwise the resumed run is a different run with the old run's name. The update count joins the contract only when the learning rate decays towards it. | Resuming on changed data, a changed order or a changed schedule. |
| **Templates that render tool call IDs are refused.** | IDs in data from elsewhere would train in a format that the model's own parser can't read. | Training the IDs as written. |
| **SFT runs no sampler. The Loops run stays up after the process exits, unless `[compute] keep_run = false`.** | SFT trains on fixed data and serves nothing. On dedicated Loops, a live run lets the next run's `reuse_from_run_id` start warm. The log warns that the run is still up, with its ID, and names `loops deactivate`. A dedicated run bills, and a serverless run holds a workspace slot. With `keep_run = false`, the trainer deactivates the run when it closes. It gives up after 30 s and warns instead. | Provisioning a sampler for each run. Always deactivating on exit. |
| **Rows render in spawned workers, after a few seconds.** | Long agent traces can take hours to render on one core, and a small dataset isn't worth starting processes for. The workers are spawned, because a fork would copy the tokenizer's and telemetry's threads while they run. | Forked workers. Always rendering in one process. Always rendering in parallel. |

## Where to look first

| Question | Start at |
| --- | --- |
| Why was this row refused? | Read the report from `loops check`, or `data.json` after `run`. Its `refusals` list the reasons. The `Refused` raises are in `sft/sequences.py`, in `supervise`, `_conversation`, `_sequence`, `_written_after` and `_terminator`. |
| What exactly trains? | `loops check` prints the first sequence's trained text, span by span. Then read `SupervisedSequence.datum`. |
| Why was a conversation split per turn? | Read the prefix test in `supervise`. `check` counts the splits. |
| Why did a resume refuse, or train nothing? | Read `check_resume`, `Schedule.contract` and `digest`, then `Schedule.total_steps` and `run_record.finished`. |
| Where do `train/nll` and `val/nll` come from? | Read `score` in `sft/trainer.py`. |
| Why is `check` slow, or using memory? | Read `render_split`, and the `[data] render_workers` setting. |

## Changing it safely

- SFT shares the CI gates in [1 · The training engine](1-engine.md#changing-it-safely).
  The SFT unit tests run in the main `test` job. The real-template tests in
  `tests/contracts/test_sft_sequences.py` skip without the model files. Run them with
  `LOOPS_DOWNLOAD_CONTRACT_MODELS=1`, as the `contracts` job does, after you change
  `sequences.py` or the renderers.
- Anything that changes the digest breaks every existing resume manifest.
  `test_what_trains_and_its_digest_are_what_runs_recorded` pins the digest. Update its
  golden values only when what trains is meant to change.
- Adding a model needs no SFT code. Pin the model's tokenizer in
  `tests/contracts/conftest.py`, and add the model to `build` and to the `family`
  fixture in `tests/contracts/test_sft_sequences.py`. Then check that the rules hold. A renderer that
  breaks them needs a fix or an extension in `llm/renderer_extensions/`, not SFT code.
- A run's status sets the exit code in `cli.py`. A completed run exits with 0, a run
  whose Loops run ended exits with 75, and any other run exits with 1. A SIGTERM exits
  with 143. A resume writes its own `config-`, `data-`, `run-` and
  `failure-from-step-N.json` files beside the first run's. If you change either
  behaviour, check `run_record.py` and the reference.
- A new setting needs a description, and the description lands in the knob glossary.
  Regenerate the glossary with `uv run python -m loops_cookbook.docs`.
  `tests/unit/test_docs.py::test_every_setting_is_documented_exactly_once` checks it.
- No test runs `loops check` on the shipped `examples/sft_*/sft.toml` configs.
  `tests/test_shipped_configs.py` covers only the RL `train*.toml` configs. Run
  `loops check` on the SFT configs by hand when a setting or the data contract changes.

## Known gaps

These gaps don't block anything. They are written down so that nobody has to
rediscover them.

- Trace importers for LangSmith and Langfuse aren't built. The `chat` contract is the
  seam they would write to.
- Direct preference optimization (DPO), in the SFT recipe beside SFT, isn't started.
- Evaluating an SFT checkpoint on real rollouts is on the roadmap. Until then,
  `examples/sft_toucan/evaluate.py` scores how well first turns agree with the data.
- Templates that render tool call IDs, such as Kimi K2's, are refused. The cookbook
  doesn't rewrite the dataset's IDs into the model's format.
- `loops check` on an SFT config doesn't ask Loops whether it serves the model at
  `max_seq_len`, as it does for RL with `check_served_model`. SFT finds out when `run`
  provisions the trainer.

## Read next

- [reference/sft.md](reference/sft.md) describes the SFT contracts, metrics and resume.
- [The knob glossary](reference/KNOB_GLOSSARY.md#sft-settings) lists every SFT setting.
- [1 · The training engine](1-engine.md) describes the RL core.
- [The roadmap](../ROADMAP.md) lists the large items that are planned.
