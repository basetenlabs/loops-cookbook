# A multi-turn agent with tools, in a Blaxel sandbox

[A custom runtime, in-process, direct](../../patterns/custom_runtime/in_process_direct/)
with a multi-turn agent: the model changes files in the attempt's sandbox with two
tools, over as many turns as it needs, and the sandbox's state is what's graded.
This is the custom-environment story an `Episode` is for: your own tasks, your own
tools, your own grader, with the cookbook owning the sandbox, the clocks and the
evidence.

**Use it when** your agent acts on an environment through tool calls and is scored on
what it left there, not on what it said.

**Files**

- `tool_episode.py`: `ToolEpisode(Episode[ToolTask, ToolOptions])`, with `execution_kind = Blaxel`.
  - `setup()` writes the task's files into `/work` through `await setup.sandbox.client()`,
    the Blaxel SDK's client, which the run attaches and closes.
  - `run(sample)` offers the policy `shell` and `write_file`, and runs each call in the
    sandbox until the model answers, runs out of `max_turns`, or the agent deadline cuts it.
  - `grade()` kills every command still running, then runs the task's check: 1 when it
    passes, else 0.
  - `sh()` uses the SDK as Harbor's Blaxel environment does: it starts a command, then
    waits, so one that outlasts `command_timeout_seconds` comes back `killed` (an exec
    that waited itself would raise). Blaxel's record keeps only the last 64 KB of output;
    the cookbook's Blaxel provider refills it from the process's logs within that wait.
    The model reads the last `max_output_chars` of each stream.
- `tasks.jsonl`: eight file tasks, each a `ToolTask`: `files`, an `instruction`, and a
  shell `check` the agent never sees.
- `train_blaxel.toml`: one Blaxel sandbox per episode, and the agent's limits in
  `[rollout_runtime.config.options]`, validated as `ToolOptions`.

**What counts as what.**

| What happened | Then |
| --- | --- |
| The model answers, uses up `max_turns`, or overflows the trajectory budget; or a reply won't parse | Graded; `termination_reason` in the episode's metadata says which. |
| A generation is cut by the token limit or the trajectory budget | The episode ends as `truncated` and is graded; the cut generation trains with the rest. |
| The agent deadline (`agent_timeout_seconds`) cuts `run()` | Graded from what it left; a command it cut is killed before the check. |
| A command fails or is killed at its timeout, a tool is unknown, or its arguments are wrong | The model reads it as the tool's result, and the episode goes on. |
| A call to Blaxel raises | Not caught: the attempt is retried (`max_rollout_retries`), or at grading, graded again (`grading_retries`). |
| The check fails, or its wait ends before it does (status not `completed`) | 0 once `await grading.execution_sandbox.ensure_alive()` returns; a sandbox that went away is graded again. |

The check trusts the sandbox: an agent that detaches processes can influence it.

There is no `train.toml`: the model's commands run in a sandbox, never on the
training machine. The tests run the whole episode offline, through a real run with
the cookbook's Blaxel adapter over a fake Blaxel whose sandboxes are local directories.

**How the tasks are sized.** A group whose rollouts all score alike trains
nothing, as GRPO learns only from the differences within a group, so the
tasks aim for Qwen3.5-4B, thinking off, to solve each some of the time and not always.
They form a ladder: an edit to one setting; finding the one log with `ERROR`; sorting
and deduplicating names with trailing spaces; counting `paid` orders where `grep paid`
also counts `unpaid`; summing a column; renaming files; fixing a script that counts
the wrong lines; deleting one duplicate setting. The easy ones are there so early
groups have some successes, and each harder one has a trap a small model often falls
into. This is unverified until a live run: the first run's per-group rewards are the
measurement, and a task whose groups always score alike should be made harder or easier.

**Run** from the repository root with `BASETEN_API_KEY`, `BL_WORKSPACE` and
`BL_API_KEY` in `.env` (`uv sync --extra blaxel`):

```sh
uv run loops check examples/tool_agent/train_blaxel.toml
uv run loops run   examples/tool_agent/train_blaxel.toml
```

`check` launches nothing; `run` provisions Loops resources, which stay provisioned
until you deactivate the run, and one Blaxel sandbox per episode, which the run
releases. The run ID is logged on exit.
