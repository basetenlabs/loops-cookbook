# A multi-turn agent with tools, in a Blaxel sandbox

This example trains a multi-turn agent. The model changes files in the attempt's
sandbox with two tools, over as many turns as it needs, and the grader scores the state
the sandbox is left in. The example builds on the
[custom rollout runtime, in-process, direct](../../patterns/custom_runtime/in_process_direct/)
pattern.

This is the kind of custom environment that an `Episode` is for. You bring your own
tasks, your own tools and your own grader. The cookbook owns the sandbox, the clocks and
the evidence.

Use it when your agent acts on an environment through tool calls, and is scored on what
it left there, not on what it said.

## Run it

Install the Blaxel extra with `uv sync --extra blaxel`. Then run these commands from the
repository root, with `BASETEN_API_KEY`, `BL_WORKSPACE` and `BL_API_KEY` set in `.env`:

```sh
uv run loops check examples/tool_agent/train_blaxel.toml
uv run loops run   examples/tool_agent/train_blaxel.toml
```

`loops check` launches nothing. `loops run` provisions a dedicated trainer and sampler
for Qwen3.5-4B, which bill until you deactivate the run. It also provisions one Blaxel
sandbox per episode, which the run releases. When the command exits, it leaves the
Loops run up, and it logs the run ID and the `loops deactivate` command that ends the
run.

There is no `train.toml`, because the model's commands always run in a sandbox, never
on the training machine. The tests run the whole episode offline, through a real run.
They use the cookbook's Blaxel adapter over a fake Blaxel, whose sandboxes are local
directories.

## How it works

The example has these files:

- `tool_episode.py` holds `ToolEpisode(Episode[ToolTask, ToolOptions])`, with
  `execution_kind = Blaxel`.
  - `setup()` writes the task's files into `/work` through
    `await setup.sandbox.client()`. That call returns the Blaxel SDK's client, which the
    run attaches and closes.
  - `run(sample)` offers the policy two tools, `shell` and `write_file`. It runs each
    tool call in the sandbox, until the model answers, the episode uses up `max_turns`,
    or the agent's deadline cuts it off.
  - `grade()` kills every command that is still running, then runs the task's check.
    The reward is 1 when the check passes, and 0 otherwise.
  - `sh()` uses the SDK the way Harbor's Blaxel environment does. It starts a command
    and then waits for it, so a command that outlasts `command_timeout_seconds` comes
    back with the status `killed`. An exec that did its own waiting would raise an
    exception instead.
  - Blaxel's process record keeps only the last 64 KB of output. The cookbook's Blaxel
    provider refills the record from the process's logs during that wait. The model
    reads the last `max_output_chars` characters of each stream.
- `tasks.jsonl` holds eight file tasks. Each one is a `ToolTask`, with `files`, an
  `instruction`, and a shell `check` that the agent never sees.
- `train_blaxel.toml` gives each episode one Blaxel sandbox. It sets the agent's limits
  in `[rollout_runtime.config.options]`, which `loops check` validates as
  `ToolOptions`.

## How the tasks are sized

Group relative policy optimization (GRPO) learns only from the differences within a
group. So a group whose rollouts all score alike trains nothing. The tasks therefore aim
to be ones that Qwen3.5-4B, with thinking off, solves some of the time but not always.

The eight tasks form a ladder. They ask the agent to do these things:

- Edit one setting.
- Find the one log that contains `ERROR`.
- Sort and deduplicate names that have trailing spaces.
- Count the `paid` orders, where `grep paid` also counts `unpaid`.
- Sum a column.
- Rename files.
- Fix a script that counts the wrong lines.
- Delete one duplicate setting.

The easy tasks give early groups some successes. Each harder task has a trap that a
small model often falls into. This sizing isn't verified until a live run. The first
run's rewards for each group are the measurement. Make a task harder or easier if its
groups always score alike.

## What counts as what

| What happened | What happens next |
| --- | --- |
| The model answers, uses up `max_turns` or overflows the trajectory budget. A reply that won't parse counts here too. | The episode is graded. `termination_reason` in the episode's metadata says which case happened. |
| The token limit or the trajectory budget cuts a generation. | The episode ends as `truncated` and is graded. The cut generation trains with the rest. |
| The agent's deadline, `agent_timeout_seconds`, cuts `run()` off after the model has replied at least once. | The episode is graded on what it left. Any command that the deadline cut is killed before the check runs. |
| The agent's deadline passes before the model's first reply. | Nothing can train. The cookbook discards the rollout as `unversioned` and counts it as a sampling failure. |
| A command fails, or is killed at its timeout. Or a tool is unknown, or its arguments are wrong. | The model reads the problem as the tool's result, and the episode goes on. |
| A call to Blaxel raises an exception. | The episode doesn't catch it. During sampling, the run retries the attempt, up to `max_rollout_retries`. During grading, the run grades the episode again, up to `grading_retries`. |
| The check fails, or its wait ends before the check does, so its status isn't `completed`. | The episode scores 0 once `await sandbox.ensure_alive()` returns. If the sandbox went away, the run grades the episode again. |

The check trusts the sandbox. An agent that detaches processes can influence the check.
