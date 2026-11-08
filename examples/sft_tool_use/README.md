# SFT: a tool-using assistant

This example uses supervised fine-tuning (SFT) to train Qwen3.6-35B-A3B on a small,
synthetic tool-use dataset. The dataset teaches a unit-conversion assistant to call a
`convert` tool, read the result and answer in one fixed style. A third of the
conversations continue with a follow-up question, which the assistant answers without
the tool.

## Run it

Run these commands from the repository root, with `BASETEN_API_KEY` set in `.env`:

```sh
uv run loops check examples/sft_tool_use/sft.toml   # what trains, and the first sequence's targets
uv run loops run   examples/sft_tool_use/sft.toml   # 42 updates on Loops (billable)
```

## How it works

Some chat templates split a conversation like these into one sequence for each trained
turn. Qwen3.6's template drops the reasoning of earlier turns from the history. So each
of the 22 conversations with a follow-up trains as three sequences, one for each of its
trained turns. That's why `loops check` reports 64 rows and 108 sequences. Under a
template that keeps the history as it was generated, such as Qwen3.8's, each
conversation trains as one sequence.
[docs/reference/sft.md](../../docs/reference/sft.md) explains when a template splits a
conversation.

The data is committed beside the config, in the `chat` format, in which each JSONL line
is one conversation in OpenAI's chat format. `make_data.py` wrote it, so rerun
`make_data.py` after you change its generator. Here `make_data.py` is the importer, the
program that writes the data in the `chat` format. Anything that writes the same JSONL
can take its place, such as an export of your own traces.
