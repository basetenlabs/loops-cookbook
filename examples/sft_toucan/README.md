# SFT: real tool-agent trajectories

This example uses supervised fine-tuning (SFT) to train Qwen3.6-35B-A3B on 2,000
conversations from [Agent-Ark/Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M).
The conversations use tools on real Model Context Protocol (MCP) servers. They include
parallel calls, multi-step chains, multi-turn dialogues and requests that no tool should
answer.

After training, the example samples the base model and the adapter on 200 held-out
conversations, and scores both. That comparison is the proof that matters.

## Run it

Run these commands from the repository root, with `BASETEN_API_KEY` set in `.env`:

```sh
uv run --with pyarrow python examples/sft_toucan/make_data.py   # data/train.jsonl, data/validation.jsonl
uv run loops check examples/sft_toucan/sft.toml       # what trains; launches nothing
uv run loops run   examples/sft_toucan/sft.toml       # one epoch on Loops (billable)
uv run python examples/sft_toucan/evaluate.py --adapter bt://loops:<run id>/sampler_weights/step-<N>
```

`<run id>` is the `run_id` in the run directory's `run.json`. `<N>` is the last update,
whose sampler weights are always kept. The run's log prints both.

`evaluate.py` deactivates its two samplers when it finishes. `loops run` leaves its
Loops run up when it exits, so deactivate the run when you're done. Its last log line
says how.

## How it works

`make_data.py` is the importer. Toucan is in ms-swift's agent format, and `make_data.py`
rewrites each conversation into the `chat` contract that SFT reads. Its docstring says
exactly how. It splits the data by tool set, so the validation conversations use tool
combinations that training never saw.

`sft.toml` turns thinking off. Toucan has no reasoning, and with thinking on, Qwen3.6
would learn to write an empty think block. See
[docs/reference/sft.md](../../docs/reference/sft.md#choosing-settings).

`evaluate.py` is a smoke test for a checkpoint, not an evaluation framework. It
measures how well first turns agree with the data. It runs no tools, and it takes one
greedy sample for each prompt. For each validation conversation, it samples the first
assistant turn from the base model and from the adapter. Then it scores whether each
one called a tool when the data does, called the same tools with the same arguments,
and produced a well-formed reply.

Validation negative log-likelihood (NLL) says that the adapter fits the data.
`evaluate.py` says whether the adapter does the task.

## Results

One run of this config on 2026-10-08 made 125 updates, in about 12 minutes of training
once the trainer was up. `val/nll` fell from 0.914 before the first update to 0.795.
Most of the drop came in the first 20 updates, and it was flat over the last 20.

`evaluate.py` scores each reply in these ways:

- `decision` checks that the model calls a tool exactly when the data does.
- `tools` checks that the model calls the same tool names.
- `arguments` checks that the model calls the same tools with exactly the same
  arguments.
- `well_formed` checks that the reply stopped and that every call parsed.

The `irrelevant` conversations are requests that no offered tool can serve. This table
compares the base model with the `step-124` adapter on the 200 validation
conversations:

| Score | Base | Adapter |
| --- | --- | --- |
| `decision` | 88.0% | 99.0% |
| `decision` on the 30 `irrelevant` conversations | 73.3% | 100% |
| `tools` | 81.8% | 88.8% |
| `arguments` | 58.8% | 63.5% |
| `well_formed` | 91.0% | 98.5% |

The data's own replies score 98.5% `well_formed` under Qwen3.6's parser, so that is the
ceiling. The largest change is on requests that no offered tool can serve. There, the
base model reaches for an unrelated tool anyway, and the adapter answers directly.
