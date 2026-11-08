# Counting

In this example, a vision model learns to count overlapping circles. Each prompt holds a
picture of 9 to 20 circle outlines, and the model replies with how many there are. Many
of the circles overlap or sit inside another one. Vision models count separate shapes
well and overlapping shapes badly, so the base model starts with a low reward, and
training can raise it.

The example builds on the
[custom rollout runtime, in-process, direct](../../patterns/custom_runtime/in_process_direct/)
pattern. The episode runs inside the training process and asks the model directly, with
the cookbook's model client. No sandbox is used.

## Run it

Run these commands from the repository root, with `BASETEN_API_KEY` set in `.env`:

```sh
uv sync --extra vision
uv run loops check examples/counting/train.toml
uv run loops run   examples/counting/train.toml
```

The `vision` extra brings torch and torchvision for the model's image processor. They
run on the CPU.

`loops check` launches nothing. `loops run` trains on serverless Loops. When the
command exits, it leaves the Loops run up, and it logs the run ID and the
`loops deactivate` command that ends the run.

If Loops hasn't enabled Qwen3.6-35B-A3B for your workspace, `check` and `run` stop with
`[model] base_model 'Qwen/Qwen3.6-35B-A3B' is not enabled on Loops` and the reason
`needs_approval`. Contact Baseten to approve the model for your workspace.

## How it works

`count_episode.py` draws each picture on the host from the task's count and seed. It
sends the picture to the model as an image part, beside the text of the question:

```python
{"type": "image", "image": picture}  # PNG bytes, beside the text of the question
```

`grade()` takes a quarter point off for each circle miscounted. A reply that is one off
scores 0.75, and a reply that is four or more off scores 0.

`make_tasks.py` wrote the 512 tasks in `tasks.jsonl`. Each step trains on eight new
pictures. So when the reward rises from one step to the next, it rises on pictures that
the model hasn't trained on.

## Results

One run of `train.toml` on serverless Qwen3.6-35B-A3B took 50 steps and 24 minutes. The
base model and the step-49 checkpoint each counted the same 64 held-out pictures, with 8
replies for each picture, at the training temperature. The pictures had 9 to 20
circles, and none of them was in `tasks.jsonl`.

| | Base | Step 49 |
| --- | --- | --- |
| Exact count | 26% | 66% |
| Off by one or less | 62% | 93% |
| Mean reward | 0.64 | 0.90 |

The base model mostly counts one or two too many. After training, almost every reply is
within two, and most misses are one too few. Every size of picture gains. Exact counts
rise from 43% to 83% at 9 to 12 circles, and from 12% to 53% at 17 to 20.

Both the base model and the trained model reply with a number, as the prompt asks. A
measurement on Qwen3.8-27B shows that room to reason helps a base model, but not as
much as training. With thinking on and 8,192 tokens for each reply, base Qwen3.8-27B
counted 44% of the pictures exactly.
