# Counting

This example uses a [custom runtime](../../patterns/custom_runtime/in_process_direct/):
the episode runs inside the training process and asks the model directly, with the
cookbook's client. The prompt holds an image. The model sees a picture of 9 to 20
circle outlines and replies with how many there are. Many of the circles overlap or sit
inside another one. Vision models count separate shapes well and overlapping shapes
badly, so the base model starts low and training can raise it.

`count_episode.py` draws each picture on the host from the task's count and seed, and
sends it as an image part:

```python
{"type": "image", "image": picture}  # PNG bytes, beside the text of the question
```

`grade()` takes a quarter point off for each circle miscounted, so a reply one off
scores 0.75 and a reply four or more off scores 0. No sandbox is used.

`tasks.jsonl` holds 512 tasks, written by `make_tasks.py`. A step trains on eight new
pictures, so a reward that rises from step to step is on pictures the model has not
trained on.

## Results

One run of `train.toml` on serverless Qwen3.6-35B-A3B: 50 steps, 24 minutes. The base
model and the step-49 checkpoint each counted the same 64 held-out pictures (9 to 20
circles, none in `tasks.jsonl`), 8 replies a picture at the training temperature.

| | Base | Step 49 |
| --- | --- | --- |
| Exact count | 26% | 66% |
| Off by one or less | 62% | 93% |
| Mean reward | 0.64 | 0.90 |

The base model mostly counts one or two too many. After training, almost every reply is
within two, and most misses are one too few. Every size of picture gains: exact counts
rise from 43% to 83% at 9 to 12 circles, and from 12% to 53% at 17 to 20.

The base model replies with a number, as asked, and so does the trained model. On
Qwen3.8-27B, this example's earlier model, room to reason helped the base model but not
as much as training: with thinking on and 8,192 tokens a reply, it counted 44% exact.

**Run** from the repository root with `BASETEN_API_KEY` in `.env`:

```sh
uv sync --extra vision
uv run loops check examples/counting/train.toml
uv run loops run   examples/counting/train.toml
```

The `vision` extra brings torch and torchvision for the model's image processor. They
run on the CPU.

If Loops has not enabled Qwen3.6-35B-A3B for your workspace, `check` and `run` stop with
`[model] base_model 'Qwen/Qwen3.6-35B-A3B' is not enabled on Loops` and the reason
`needs_approval`. Contact Baseten to approve the model for your workspace.

`check` launches nothing; `run` provisions Loops resources, which stay provisioned
until you deactivate the run. The run ID is logged on exit.
