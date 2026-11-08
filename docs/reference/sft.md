# SFT

Supervised fine-tuning (SFT) trains a low-rank adaptation (LoRA) adapter on Loops from
fixed data. Conversations go in, and a trained adapter comes out. SFT shares the
trainer, the model and output settings, checkpoints and Weights & Biases (W&B)
reporting with the RL recipe. It shares nothing else, so an SFT run has no
rollout runtime, pattern, sandbox or sampler.

```sh
uv run loops check examples/sft_tool_use/sft.toml   # renders every row; launches nothing
uv run loops run   examples/sft_tool_use/sft.toml   # provisions a Loops trainer (billable)
```

Every setting is listed in the [knob glossary](KNOB_GLOSSARY.md#sft-settings), and
`loops run --help` lists them with their defaults. You can set any of them on the
command line by its dotted path, such as `--training.learning_rate 2e-4`.

## The two data contracts

A dataset is a JSONL file in one of two formats, `chat` or `tokens`. You name the format
once, in `[data] format`, and every row of the training and validation files must meet
it. When a row doesn't, the run reports it by file and line before it does anything
else.

### `chat` (the default)

Each line is one conversation in OpenAI's chat format. OpenAI fine-tuning files, Hugging
Face conversational datasets and most trace exports are already in this format, or are a
few lines of code away from it:

```json
{"messages": [
  {"role": "system", "content": "You convert units."},
  {"role": "user", "content": "12 miles in km?"},
  {"role": "assistant", "reasoning_content": "Call the tool.", "content": "",
   "tool_calls": [{"id": "c1", "type": "function",
                   "function": {"name": "convert", "arguments": "{\"value\": 12}"}}]},
  {"role": "tool", "tool_call_id": "c1", "content": "19.312"},
  {"role": "assistant", "content": "About 19.31 km.", "weight": 1}
 ],
 "tools": [{"type": "function", "function": {"name": "convert", "parameters": {...}}}],
 "metadata": {"source": "langsmith"}}
```

- `weight` on an assistant message decides whether that turn trains. A weight of 1, the
  default, trains the turn. A weight of 0 keeps the turn as context only. You decide
  which turns train, and this is your main control over what the model learns. A weight
  may also be written as `1.0`.
- `reasoning_content` on an assistant message holds the turn's reasoning. It is
  rendered where the model's chat template puts reasoning. You also decide whether your
  data carries reasoning. `check` reports what trains, so you can see when a dataset
  has none.
- Tool calls use OpenAI's format. `arguments` is a JSON object, written either as a
  string, as OpenAI writes it, or as a mapping, as Hugging Face datasets do. A tool
  message names the call it answers with `tool_call_id`. Data with no IDs at all is also
  accepted, and then the n-th result answers the n-th call. A message with `tool_calls`
  must be followed by its results, or it must end the conversation. When results follow,
  they must answer every call exactly once.
- `name` on a message is passed to the template.
- Any other field in a row, such as an `id` column, `parallel_tool_calls` or a field of
  your own, is kept aside. It is never rendered or trained, and `check` lists it, so a
  misspelt field shows up instead of going missing.
- `metadata` is carried along with the row and never read.

The conversations can come from any model, because the text is rendered again with the
trained model's own chat template. A renderer does this. It turns a conversation into
the token IDs that the trained model expects, and [renderers.md](renderers.md) explains
how the cookbook chooses one.

An importer is any script that writes this format, such as a LangSmith or Langfuse
export script, a filter over production logs, or a conversion of another model's
transcripts. A Hugging Face dataset that is already in this shape needs only a few
lines:

```python
import json
from datasets import load_dataset

rows = load_dataset("<org>/<dataset>", split="train")
with open("train.jsonl", "w") as out:
    for row in rows:
        # Rename or reshape here when the dataset's columns are not already
        # `messages` (and `tools`); anything left over is ignored and reported.
        out.write(json.dumps({"messages": row["messages"], "tools": row.get("tools") or []}) + "\n")
```

A Hugging Face export often sets a field to `null` in one row because another row has
that field. The contract reads such a field as absent. It also decodes `tools` when the
dataset stores it as a JSON string. A dataset in another format needs an importer of its
own. For example, [examples/sft_toucan/make_data.py](../../examples/sft_toucan/make_data.py)
turns ms-swift agent trajectories into this format.

For now, SFT refuses a template that writes tool call IDs into its tokens, such as
Kimi K2's `functions.{name}:{index}`. IDs in data from another source would be trained
as they are written, in a format that the model's own parser can't read.

### `tokens`

Each line is one sequence that is already tokenized for the trained model, with its
targets marked:

```json
{"input_ids": [151644, 872, 198, ...], "weights": [0, 0, 0, ..., 1, 1, 1], "metadata": {}}
```

`weights[i]` is 1 when token `i` is a target, which is a token the model learns to
produce. The first token has nothing before it to be predicted from, so `weights[0]` is
always 0. IDs are JSON integers. A weight is 0 or 1, and `1.0` counts as 1, but `true`
is refused.

This format is never rendered, so it needs no renderer and works for any model that
Loops trains. `check` decodes the targets with the model's Hugging Face tokenizer, so
that you can read them. That tokenizer is `[model] tokenizer`, which defaults to
`base_model`.

The `tokens` format gives you full control and full responsibility. Because nothing is
rendered, nothing checks that the targets are the model's own tokens, or that each turn
learns to end. Use it for data that a chat template can't express, or for targets finer
than a whole turn, such as a turn's answer without its reasoning. In the `chat` format,
`weight` always trains a whole turn.

## Which tokens train (`chat`)

In the `chat` format, one rule decides the targets for every model:

**A token trains when the trained model's chat template has the model produce that
token, in a trained turn, after that turn's own generation prompt.**

A turn's generation prompt is the prompt the model would have been given to write that
turn. It is every message before the turn, rendered, followed by the template's opening
for an assistant turn. The renderer reports two facts about each token: which message
owns it, and whether the template has the model generate it. The rule needs nothing
else, so no model needs code of its own.

The rule has four consequences. Each one guards against a way that SFT quietly goes
wrong in other setups.

- **Every turn learns to end.** Qwen and Nemotron close a turn with `<|im_end|>`, inside
  the turn. GLM closes a turn with the next message's role marker, which is
  `<|observation|>` after tool calls and `<|user|>` otherwise. The renderer gives that
  marker to the turn it ends, so the marker trains with that turn. SFT refuses a
  renderer that gives a token the model writes to a message that isn't an assistant
  turn. The last turn has
  no next message, so it trains on the end token that the template renders for the
  message that would follow it. SFT refuses a turn that the template gives no stop
  token. If such a turn trained, the model would never learn to stop, and nothing in the
  loss would show it.
- **Each turn trains in the context it was generated in.** Qwen3.6 and Nemotron drop an
  earlier turn's reasoning from the history once a newer user message arrives. In the
  full conversation, that earlier turn then sits in a context the model never saw when
  it wrote the turn. To catch this, SFT renders each trained turn's generation prompt
  and checks that it is a prefix of the full conversation. Where one isn't, the
  conversation becomes one sequence per trained turn, and each sequence ends at its
  turn. `check` counts these splits.
- **A prefilled token is context.** Qwen and Nemotron prompts end in `<think>`, which the
  model is given rather than writes. Targets start where the generation prompt ends, so
  a prefilled token never trains.
- **A turn is tokenized as the model writes it.** Qwen's tokenizer merges the prefilled
  `<think>\n` at the end of the prompt with the `\n` of an empty reasoning. The full
  render of that turn therefore doesn't start with the prompt's tokens. For such a turn,
  SFT tokenizes the turn's text after the prompt on its own, as the model would write
  it. Every token of that text must be one that the template has the model write.
  Nemotron goes further. After a `<think>\n` prompt, it renders an empty reasoning as
  `<think></think>`, which isn't what the model writes at all. SFT refuses such a row
  unless you set the template's thinking-off option,
  `renderer_chat_template_kwargs = {enable_thinking = false}`. That is the right setting
  for data without reasoning anyway.

A sequence ends at its last target, because nothing after that target is ever read.

SFT refuses a row that the rule can't train, such as a row longer than `max_seq_len`,
a turn with no stop token, or a template that the renderer refuses. It records the
reason, lists the row in `check`'s report and in `data.json`, and never trains the row
some other way.

Contract tests in `tests/contracts/test_sft_sequences.py` check these properties on the
real Qwen3.6, GLM-5.3 and Nemotron 3 Super renderers.

## Rendering time and memory

On long conversations, rendering is the slow part of `check` and of the start of `run`,
because the generation prompt of every trained turn is rendered on its own. A split that
is still rendering after a few seconds hands its remaining rows to worker processes,
and each worker holds its own copy of the renderer. While rows render, the run logs how
many are done, at most once every 30 seconds.

`[data] render_workers` sets the number of worker processes. By default, it is the
number of cores the training process may use, minus one for everything else, and at
most 4. Set more on a machine with cores to spare, fewer when memory is short, or 1 to
render every row in the training process.

In two cases, rows render in the training process instead of in the workers. A custom
renderer that can't be pickled keeps all of its rows in the training process. Rows
that the workers fail to return also render there, with a warning. That happens when
the renderer doesn't load in a fresh process, or when a worker is killed for using too
much memory. Either way, the rows and their order are the same, and so is the digest,
which is a hash of exactly what trains.

A run holds every rendered sequence until it ends, so it keeps each one in about five
bytes a token. Each token's ID is stored as a machine integer, and one more byte
records whether the token trains. The run builds each update's datums, which are the
rows that Loops trains on, only as it sends them. It lets go of the parsed rows once
they are rendered.

## Choosing settings

- Set `max_seq_len` above your longest conversation. `check` reports the 95th
  percentile and the maximum sequence length. It refuses rows longer than `max_seq_len`
  instead of truncating them.
- **Data without reasoning on a thinking model**, such as Qwen3.6 or Nemotron, needs
  `renderer_chat_template_kwargs = {enable_thinking = false}`. Without it, the model is
  trained to write an empty think block. Each turn of a multi-turn conversation also
  becomes its own sequence, which reads the whole history again.
- A `learning_rate` of 1e-4 to 2e-4 with a `lora_rank` of 32 is a reasonable start for
  LoRA. A `cosine` learning rate schedule with a few `warmup_steps` keeps the first
  updates gentle.
- `batch_size` counts sequences. More sequences per update give a smoother `train/nll`,
  which is the mean negative log-likelihood (NLL) of the trained tokens, and fewer
  updates per epoch.
- Validation tells you whether the model is learning the task or memorizing the
  examples. Set `[data] validation` to a file of held-out rows, or `validation_fraction`
  to hold out part of the training rows. The signal to watch is `val/nll` falling while
  `train/nll` falls.

## The training step

- One update trains on `batch_size` sequences. The last update of an epoch takes the
  sequences that remain. Each update is one `forward_backward` call with the
  `cross_entropy` loss, followed by one `optim_step`.
- `loss_aggregation` decides how the gradient is reduced. With `"sum"`, which is the
  default and Loops' own default, the gradient is summed over the update's trained
  positions. With `"token_mean"`, Loops divides that sum by the number of trained
  positions. Both stay correct however the SDK splits the request.
- Epoch `e` visits every training sequence once, in an order shuffled with the seed
  `shuffle_seed + e`. A nonzero `max_steps` stops the run early, in the middle of an
  epoch if necessary.
- The learning rate rises linearly over `warmup_steps`. After that, it holds steady
  when `lr_schedule` is `constant`. When `lr_schedule` is `linear` or `cosine`, it
  decays towards zero by the last update.
- `train/nll` is the mean of `-logprob` per trained position in the update. The run
  computes it from the logprobs that Loops returns, so it is the same whichever
  reduction the gradient uses. `val/nll` is the same measure over the validation
  sequences. The run computes it with `forward`, which takes no gradient. It scores
  validation before the first update, as step -1, then every `[eval] every_steps`
  updates, and after the last update.

## Metrics

Every update reports these metrics to W&B, on the `train/update` axis, and to
`steps.jsonl`:

- `train/nll` is the mean NLL per trained token in the update.
- `train/trained_tokens`, `train/tokens` and `train/sequences` count what the update
  trained on.
- `train/learning_rate` and `train/epoch` show where the schedule is.
- `val/nll` and `val/trained_tokens` report validation, on the updates where the run
  scores it.
- `optim/grad_norm`, `optim/tps`, `optim/step_seconds`, `optim/num_tokens` and
  `optim/num_loss_tokens` are Loops' own report of the optimizer step.
- `time/update_seconds` is the wall-clock time of the update, from the request to the
  response.
- `train/checkpointed` says whether the run kept a checkpoint of the update.

## The run directory

Each run writes its records to `<output_dir>/<run_name>-<time>-<id>/`:

- `config.json` holds the config the run was given.
- `data.json` says what the data became. It lists the rows, sequences, splits, refusals
  with their reasons, and token counts, and it holds the digest of exactly what trains.
- `run.json` records the Loops run that the run created. A resume records its own Loops
  run in `run-from-step-N.json`.
- `steps.jsonl` has one record per update.
- `checkpoints/step-N.json` is a resume manifest. The run writes one whenever it keeps
  trainer state.
- `failure.json` records how the run ended, when it didn't complete.
- `run.log` holds everything the run logged.

## Continue or fork a run

`resume_from` continues a run. `init_from` starts a new run from trained weights.
[Training](training.md#continue-or-fork-a-run) compares the two. This section gives the
rules that are specific to SFT.

```sh
# Continue the latest run of this config, from the manifest it wrote last.
uv run loops run examples/sft_tool_use/sft.toml --resume
# Fork: a new run from the weights another run kept at its step 7.
uv run loops run examples/sft_tool_use/sft.toml \
  --training.init_from bt://loops:<run>/weights/step-7
```

A resume continues in the same run directory, at the next update, with the epoch order
and learning rates that the run had. It refuses any change that would change what an
update means. It refuses a change to the data, including a data file's name, to the
renderer settings or to `max_seq_len`, because the digest in the manifest no longer
matches. It also refuses a change to `batch_size`, `shuffle_seed`, `learning_rate`,
`lr_schedule` or `warmup_steps`. With a `linear` or `cosine` schedule, it refuses a
change to the total number of updates too, because the learning rate decays towards
the last one.

With a constant learning rate, you can resume a run for more updates by passing a
larger `--training.max_steps`. As in RL, `max_steps` counts all of the run's updates,
including the ones before the resume. Steps count from 0, so `checkpoints/step-8.json` is
written after 9 updates, and a resume from it with `max_steps = 12` makes 3 more.

A resume can find no update left, because of `max_steps` or because every epoch is
done. It is checked like any other resume, then it completes with exit code 0 and
starts no trainer. With a decaying learning rate, its message names the `init_from`
value that continues training in a new run.

Suppose you resume from an older manifest, and the resumed run stops before it writes a
manifest of its own. The manifest written last is then still the one from the later
branch, so the next resume of the run continues that later branch. To stay on the
earlier step, resume from that older manifest file again.

`init_from` loads the LoRA adapter, with fresh Adam moments, at step 0 of a new run
directory. The new run can use other data and other settings. The trainer's policy
version isn't reset. It continues from the checkpoint's policy version. A resume of
the new run restores the new run's own state, and it doesn't load the `init_from`
weights again.

Set `init_from_optimizer = true` to also load the source run's optimizer state, which
is its Adam moments and step. Use it for a fork that keeps the source run's optimizer.
Those moments reflect the source run's gradient scale. If the checkpoint was saved
without them, Loops loads the weights alone. `init_from_optimizer` requires
`init_from`, and a resume ignores both of them.

## After the run

The trained LoRA adapter is kept on Loops as sampler weights named `step-N`. The run
saves them every `sampler_checkpoint_every` updates, and always for the last update.
The log names each one, and so does the `checkpoint` field in `steps.jsonl`. The Loops
run ID is `run_id` in `run.json`.

To sample from the adapter, use the SDK to serve it on a standalone sampler, on the
scope you name. A dedicated sampler bills until you deactivate it with
`POST /v1/loops/samplers/<id>/deactivate`.

```python
from baseten.loops import ServiceClient

sampler = ServiceClient().create_sampling_client(
    model_path="bt://loops:<loops run id>/sampler_weights/step-<N>",
    scope="serverless",  # or "dedicated"
)
```

**`loops run` leaves its Loops run up when it exits**, so that on dedicated Loops a
later run's `[compute] reuse_from_run_id` can start warm on it. A dedicated Loops run
bills until you deactivate it, either with
`loops deactivate <run id or run directory>` or from the Loops dashboard. A serverless
Loops run holds a slot in your workspace until then. The run's log warns that the
Loops run is still up and gives its ID. To deactivate the Loops run when `loops run`
exits, set `[compute] keep_run = false`. Its checkpoints stay either way.

[examples/sft_toucan](../../examples/sft_toucan/) goes the whole way. It turns a Hugging
Face dataset into a trained adapter. Then it samples the base model and the adapter on
held-out conversations, and scores both.
