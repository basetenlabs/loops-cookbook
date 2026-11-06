# SFT: real tool-agent trajectories

Supervised fine-tuning of Qwen3.6-35B-A3B on 2,000 conversations from
[Agent-Ark/Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M): tool
use over real MCP servers, with parallel calls, multi-step chains, multi-turn
dialogues, and requests no tool should answer. Then the proof that matters: the base
model and the adapter on 200 held-out conversations, sampled and scored.

```sh
uv run --with pyarrow python examples/sft_toucan/make_data.py   # data/train.jsonl, data/validation.jsonl
uv run loops check examples/sft_toucan/sft.toml       # what trains; launches nothing
uv run loops run   examples/sft_toucan/sft.toml       # one epoch on Loops (billable)
uv run python examples/sft_toucan/evaluate.py --adapter bt://loops:<run id>/sampler_weights/step-<N>
```

`<run id>` is `run_id` in the run directory's `run.json`, and `<N>` the last update,
whose sampler weights are always kept; the run's log prints both.

`make_data.py` is the importer: Toucan is in ms-swift's agent format, and it rewrites
each conversation into the `chat` contract (see its docstring for exactly how). The
split is by tool set, so the validation conversations use tool combinations training
never saw.

`sft.toml` turns thinking off: Toucan has no reasoning, and with thinking on Qwen3.6
would learn to write an empty think block (see
[docs/reference/sft.md](../../docs/reference/sft.md#choosing-settings)).

`evaluate.py` is a smoke test for a checkpoint, not an eval framework: it measures
agreement with the data on first turns, runs no tools and takes one greedy sample per
prompt. It samples the first assistant turn of each validation conversation from
the base model and from the adapter, greedily, and scores whether each called a tool
when the data does, called the same tools with the same arguments, and produced a
well-formed reply. Validation NLL says the adapter fits the data; this says it does
the task. Its two samplers are deactivated when it finishes. `loops run` leaves
its Loops run up for reuse; deactivate it when you are done (its last log line says
how).

## Results

One run of this config on 2026-10-08 (125 updates, about 12 minutes of training once
the trainer was up): `val/nll` fell from 0.914 before the first update to 0.795,
most of it in the first 20 updates, flat over the last 20.

`evaluate.py` on the 200 validation conversations, base model against the `step-124`
adapter:

| Score | Base | Adapter |
| --- | --- | --- |
| `decision`: calls a tool exactly when the data does | 88.0% | 99.0% |
| `decision` on the 30 `irrelevant` conversations | 73.3% | 100% |
| `tools`: the same tool names | 81.8% | 88.8% |
| `arguments`: the same tools and arguments, exactly | 58.8% | 63.5% |
| `well_formed`: stopped, and every call parsed | 91.0% | 98.5% |

The data's own replies score 98.5% `well_formed` under Qwen3.6's parser, so that is
the ceiling. The largest change is on requests no offered tool can serve, where the
base model reaches for an unrelated tool anyway and the adapter answers directly.
