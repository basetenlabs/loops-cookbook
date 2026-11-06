# SFT

Supervised fine-tuning on fixed data: conversations in, a LoRA trained on Loops out.
It shares the trainer, the model and output settings, checkpoints and W&B with the RL
recipe, and nothing else: no runtime, pattern, sandbox or sampler.

```sh
uv run loops check examples/sft_tool_use/sft.toml   # renders every row; launches nothing
uv run loops run   examples/sft_tool_use/sft.toml   # provisions a Loops trainer (billable)
```

Every setting is in the [knob glossary](KNOB_GLOSSARY.md#sft-settings), and `loops
run --help` lists them with their defaults; any of them can be set on the command line
by its dotted path, such as `--training.learning_rate 2e-4`.

## The two data contracts

A dataset is JSONL in one of two formats, named once by `[data] format`. Every row of
the training and validation files is held to that one, and a row that breaks it is
reported by file and line before anything else happens.

### `chat` (the default)

One line is one conversation in OpenAI's chat format, which OpenAI fine-tuning files,
Hugging Face conversational datasets and most trace exports already are, or are a few
lines away from:

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

- `weight` on an assistant message: 1 (the default) trains the turn, 0 keeps it as
  context only. Which turns train is yours to decide; that is the lever.
- `reasoning_content` on an assistant message: the turn's reasoning, rendered where
  the template puts it. Whether your data carries reasoning is yours to decide too.
  `check` reports what trains, so a dataset without it is visible.
- Tool calls are OpenAI's. `arguments` is a JSON object, as a string (OpenAI) or a
  mapping (Hugging Face). A tool message names the call it answers by `tool_call_id`;
  data without ids at all is fine, and then the n-th result answers the n-th call.
  Every call is answered exactly once, and a message with `tool_calls` is followed by
  its results or ends the conversation.
- `name` on a message is passed to the template; `weight` may be `1.0`.
- Anything else in a row -- an `id` column, `parallel_tool_calls`, fields of your own
  -- is kept aside, never rendered or trained, and listed by `check`, so a misspelt
  field shows up rather than going missing.
- `metadata` is carried along and never read.

The conversations can come from any model: the text is re-rendered with the trained
model's own chat template. An importer is anything that writes this format -- a
LangSmith or Langfuse export script, a filter over production logs, another model's
transcripts. A Hugging Face dataset already in this shape is a few lines:

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

Fields a Hub export sets to `null` because another row has them read as absent, and
`tools` stored as a JSON string is decoded. A dataset in another format needs an
importer of its own: [examples/sft_toucan/make_data.py](../../examples/sft_toucan/make_data.py)
turns ms-swift agent trajectories into this contract.

A template that writes tool call ids into its tokens (Kimi K2's
`functions.{name}:{index}`) is refused for now: ids in data from elsewhere would be
trained as written, in a format that model's parser cannot read.

### `tokens`

One line is one sequence already tokenized for the trained model, with its targets
marked:

```json
{"input_ids": [151644, 872, 198, ...], "weights": [0, 0, 0, ..., 1, 1, 1], "metadata": {}}
```

`weights[i]` is 1 when token `i` is a target (so `weights[0]` is always 0). Ids are
JSON integers; a weight is 0 or 1 (`1.0` counts, `true` does not). It needs no
Prime renderer, so it works for any model Loops trains; `check` decodes the targets
with the model's Hugging Face tokenizer (`[model] tokenizer`, default `base_model`). This is full control and full
responsibility: nothing is rendered, so nothing checks that the targets are the
model's own tokens, or that a turn learns to end. Use it for data a chat template
cannot express, or for targets finer than a turn, such as a turn's answer without its
reasoning: `weight` in `chat` trains a whole turn.

## Which tokens train (`chat`)

One rule for every model: **a token is trained when the trained model's chat template
has it produce that token, in a trained turn, after the turn's own generation
prompt.** The renderer reports, per token, which message owns it and whether the
template has the model generate it; the rule needs nothing else, so no model has code
of its own. Four consequences, each a way SFT quietly goes wrong elsewhere:

- **Every turn learns to end.** Qwen and Nemotron close a turn with `<|im_end|>` inside
  it. GLM closes it with the next message's role marker, `<|observation|>` after tool
  calls and `<|user|>` otherwise; the renderer gives that marker to the turn it ends,
  and it trains with that turn. A renderer that gives a token the model writes to a
  message that isn't an assistant turn is refused. The last turn has no next message, so its
  terminator is the one the template renders for the message that would follow it. A
  turn the template gives no stop token is refused: trained, it would never stop, and
  nothing in the loss would show it.
- **Each turn trains in the context it was generated in.** Qwen3.6 and Nemotron drop
  an earlier turn's reasoning from history once a newer user message arrives, so in
  the full conversation that turn sits in a context it never saw. Each trained turn's
  generation prompt is rendered and must be a prefix of the conversation; where one is
  not, the conversation becomes one sequence per trained turn, each ending at its
  turn. `check` counts these splits.
- **A prefilled token is context.** Qwen and Nemotron prompts end in `<think>`, which
  the model is given rather than writes; targets start where the generation prompt
  ends.
- **A turn is tokenized as the model writes it.** Qwen's tokenizer merges the
  prompt's prefilled `<think>\n` with an empty reasoning's `\n`, so its full render is
  not the prompt plus anything; that turn's text after the prompt is tokenized on its
  own, as the model would write it, and every token of it must be one the template
  has the model write. Nemotron goes further and renders an empty
  reasoning as `<think></think>` after a `<think>\n` prompt, which is not what the model
  writes at all: such a row is refused unless the template's thinking-off option is
  set, `renderer_chat_template_kwargs = {enable_thinking = false}`, which is the right
  setting for data without reasoning anyway.

A sequence ends at its last target. A row the rule cannot train -- over
`max_seq_len`, a turn with no stop token, a template the renderer refuses -- is
refused with its reason, listed by `check` and in `data.json`, and never trained some
other way. Contract tests on the real Qwen3.6, GLM-5.3 and Nemotron 3 Super renderers
check these properties, in `tests/contracts/test_sft_sequences.py`; on the same
conversations the targets match upstream tinker-cookbook's renderers, token for token
on GLM-5.3.

Rendering is the slow part of `check` and of the start of `run` on long conversations,
since every trained turn's prompt is rendered on its own. A split still rendering
after a few seconds hands its remaining rows to worker processes, each with a copy of
the renderer, and logs how many rows are done at most every 30 seconds. By default
that is one process per core this one may use, less one for everything else, and at
most 4; `[data] render_workers` sets the number, more on a machine to spare, fewer
when memory is short, 1 to render in this process. A custom renderer that does not
pickle keeps the rows in one process, and rows the workers fail to return (a renderer
that does not load in a fresh process, a worker killed for memory) render in this one,
with a warning. The rows and their order are the same either way, and so is the
digest.

A run holds every rendered sequence until it ends, so each is kept in about five bytes
a token: its ids as machine integers and a byte for whether each trains. Each update's
datums are built as they are sent, and the parsed rows are let go once rendered.

## Choosing settings

- `max_seq_len`: above your longest conversation; `check` reports the p95 and the
  max, and refuses rows over it rather than truncating them.
- **Data without reasoning on a thinking model** (Qwen3.6, Nemotron):
  `renderer_chat_template_kwargs = {enable_thinking = false}`. Without it the model
  is trained to write an empty think block, and each turn of a multi-turn
  conversation becomes its own sequence, re-reading the history every time.
- `learning_rate` 1e-4 to 2e-4 with `lora_rank` 32 is a reasonable start for LoRA;
  `cosine` with a few `warmup_steps` keeps the first updates gentle.
- `batch_size` counts sequences; more sequences per update means a smoother
  `train/nll` and fewer updates per epoch.
- Validation (`[data] validation` or `validation_fraction`) is what tells you whether
  the model is learning the task or the examples; `val/nll` falling while
  `train/nll` falls is the signal.

## The training step

- One update is `batch_size` sequences (the last of an epoch takes the rest): one
  `forward_backward` with `cross_entropy`, then one `optim_step`. With
  `loss_aggregation = "sum"`, the default and Loops' own, the gradient is summed over
  the update's trained positions; `token_mean` asks Loops to divide it by their
  count. Either holds however the SDK splits the request.
- Epoch `e` visits every training sequence once, in an order shuffled with
  `shuffle_seed + e`. A nonzero `max_steps` stops early, mid-epoch if need be.
- The learning rate rises linearly over `warmup_steps`, then holds (`constant`) or
  decays towards zero by the last update (`linear`, `cosine`).
- `train/nll` is the token mean of `-logprob` over the update's trained positions,
  from the logprobs Loops returns, whichever reduction the gradient takes. `val/nll`
  is the same over the validation sequences, from `forward` (no gradient), before the
  first update (step -1), every `eval.every_steps`, and after the last.

## Metrics

Every update reports, on the `train/update` axis in W&B and in `steps.jsonl`:

| Metric | Meaning |
| --- | --- |
| `train/nll` | Mean NLL per trained token in the update |
| `train/trained_tokens`, `train/tokens`, `train/sequences` | What the update trained on |
| `train/learning_rate`, `train/epoch` | Where the schedule is |
| `val/nll`, `val/trained_tokens` | Validation, when scored |
| `optim/grad_norm`, `optim/tps`, `optim/step_seconds`, `optim/num_tokens`, `optim/num_loss_tokens` | Loops' own report of the step |
| `time/update_seconds` | Wall clock of the update, request to response |
| `train/checkpointed` | Whether the update was kept |

## The run directory

`<output_dir>/<run_name>-<time>-<id>/`:

- `config.json`: what was asked for.
- `data.json`: what the data became: rows, sequences, splits, refusals and their
  reasons, token counts, and the digest of exactly what trains.
- `run.json`: the Loops run created; a resume adds `run-from-step-N.json` for its own.
- `steps.jsonl`: one record per update.
- `checkpoints/step-N.json`: a resume manifest whenever trainer state is kept.
- `failure.json`: how the run ended, when it did not complete.
- `run.log`: everything the run logged.

## Continue or fork a run

`resume_from` continues a run. `init_from` starts a new run from trained weights.
[Training](training.md#continue-or-fork-a-run) compares the two. The SFT rules follow.

```sh
# Continue the latest run of this config, from the manifest it wrote last.
uv run loops run examples/sft_tool_use/sft.toml --resume
# Fork: a new run from the weights another run kept at its step 7.
uv run loops run examples/sft_tool_use/sft.toml \
  --training.init_from bt://loops:<run>/weights/step-7
```

A resume continues in the run directory at the next update, with the epoch order and
learning rates it had. It refuses anything that would change what an update means:
data, renderer settings or `max_seq_len` (the digest in the manifest no longer
matches), the batch size or seed, and the learning rate schedule. A constant learning
rate can be resumed for more updates with a larger `--training.max_steps`. It counts
the run's updates, as in RL: a resume from step 8 with `max_steps = 12` makes 4 more.
A resume with no update left, by `max_steps` or by the epochs, is checked as any
resume is, then completes with exit 0 and starts no trainer. With a decaying learning
rate it names the `init_from` that continues in a new run.
A resume
from an older manifest that stops before it writes its own first manifest leaves the
later branch as the one written last. To stay on the earlier step, resume that
manifest file again.

`init_from` loads the LoRA adapter with fresh Adam moments, at step 0 of a new run
directory. The new run can use other data and other settings. The trainer's policy
version is not reset: it continues from the checkpoint's. A resume of that run restores
its own state and does not load the `init_from` weights again. `init_from_optimizer =
true` also loads the source's optimizer state (Adam moments and step), for a fork that
keeps the source run's optimizer. Those moments reflect the source run's gradient
scale. From a checkpoint saved without them, Loops loads the weights alone.
`init_from_optimizer` requires `init_from`, and a resume ignores it with `init_from`.

## After the run

The trained LoRA is kept on Loops as sampler weights named `step-N`, on the update's
cadence and always for the last one (`sampler_checkpoint_every`). The log names it,
as does `steps.jsonl`'s `checkpoint`; the Loops run id is `run_id` in `run.json`. To
sample it, the SDK serves it on a standalone sampler, on the scope you name. A dedicated
one bills until you deactivate it (`POST /v1/loops/samplers/<id>/deactivate`):

```python
from baseten.loops import ServiceClient

sampler = ServiceClient().create_sampling_client(
    model_path="bt://loops:<loops run id>/sampler_weights/step-<N>",
    scope="serverless",  # or "dedicated"
)
```

**`loops run` leaves its Loops run up on exit,** so that on dedicated Loops `[compute]
reuse_from_run_id` can start the next run warm. A dedicated run bills until you run
`loops deactivate <run id or run directory>` or use the Loops dashboard. The run's log
warns, with its ID, that it is still up. To deactivate the Loops run on exit, set `[compute] keep_run =
false`.

[examples/sft_toucan](../../examples/sft_toucan/) goes the whole way: a Hugging Face
dataset in, a trained adapter out, and the base model and the adapter sampled and
scored on held-out conversations.
