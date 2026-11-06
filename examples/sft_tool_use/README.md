# SFT: a tool-using assistant

This example fine-tunes Qwen3.6-35B-A3B on a small, synthetic tool-use dataset. The
dataset teaches a unit-conversion assistant to call a `convert` tool, read the result
and answer in one fixed style. A third of the conversations continue with a follow-up
that the assistant answers without the tool.

Some chat templates split a conversation like this into one sequence per trained turn.
Qwen3.6's template drops the reasoning of earlier turns from the history, so the 22
conversations with a follow-up each train as two sequences: `check` reports 64 rows
and 108 sequences. Under a template that keeps the history as generated, such as
Qwen3.8's, each conversation trains as one sequence.
[docs/reference/sft.md](../../docs/reference/sft.md) explains when a template splits a
conversation.

```sh
uv run loops check examples/sft_tool_use/sft.toml   # what trains, and the first sequence's targets
uv run loops run   examples/sft_tool_use/sft.toml   # 42 updates on Loops (billable)
```

The data is in the `chat` format and committed beside the config; `make_data.py`
wrote it (rerun it after changing the generator) and is the importer here, and anything
that writes the same JSONL -- an export of your own traces -- takes its place.
