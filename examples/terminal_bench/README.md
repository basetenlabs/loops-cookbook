# Terminal-Bench 2.1 on Blaxel

This example runs a real agentic workload. A multi-turn bash agent works through
Terminal-Bench 2.1 tasks in Blaxel sandboxes, with GLM-5.3-Flash as the policy, in async
mode. `train.toml` is a small configuration for a first run, not a tuned one.

## Run it

Run these commands from the repository root, with `BASETEN_API_KEY`, `BL_WORKSPACE` and
`BL_API_KEY` set in `.env`:

```sh
uv sync --extra harbor-blaxel
uv run loops check examples/terminal_bench/train.toml
uv run loops run   examples/terminal_bench/train.toml
```

To train with ECHO instead, add `--extra custom-loss` to the `uv sync` command, and pass
`train_echo.toml`.

`loops run` trains on serverless Loops and starts Blaxel sandboxes, and both are
billable. When the command exits, it leaves the Loops run up, and it logs the run ID
and the `loops deactivate` command that ends the run.

## How it works

The example has these files:

- `terminal_agent.py` is the agent. It has a system prompt, one `bash` tool, parallel
  tool dispatch, output truncation and stopping rules. Its settings are declared in
  `TerminalOptions` and supplied from `harbor.yaml`. Commands run in `/`, so a command
  that needs another directory must `cd` to it.
- `harbor.yaml` pins the official Terminal-Bench 2.1 dataset. It selects Blaxel through
  the cookbook's environment class, `loops_cookbook.harbor.blaxel:BlaxelEnvironment`.
  Its `retry` block applies only to a task whose verifier runs in a separate sandbox.
  The run retries a trial whose sandbox or agent fails, up to `max_rollout_retries`.
  It carries an optional agent deadline override, which the section below explains.
- `train.toml` runs in async mode, with eight rollouts per group, four groups per update
  and 50 updates, on serverless Loops.
- `echo_loss.py` and `train_echo.toml` run the same training with
  [ECHO](https://arxiv.org/abs/2605.24517), a custom loss. ECHO uses importance
  sampling on the agent's tokens, plus a next-token loss on what the agent's commands
  print. `observation_weight` weights the next-token loss, and the config sets it to
  the paper's value, 0.05. The module's docstring says where it differs from the paper.

## The agent deadline

Each task sets its own agent deadline, and most of these tasks set it to 900 seconds.
Harbor stops the agent when that deadline passes, and then grades whatever the agent
left in the sandbox. The deadline is part of the task: an agent that solves the task
within it scores, and one that doesn't is cut off.

The deadline covers the agent's wait for the model as well as the commands it runs, so
two different things make a rollout hit it. The agent can take more turns, or spend
more tokens per turn, and run out of task budget. Or the sampler can be slow, and the
agent spends the deadline queueing. These need different fixes, so check which one you
have before changing anything. The sampler's `queue_wait_seconds` and its time to first
token show whether rollouts are waiting; the agent's own turn and token counts show
whether trajectories are growing.

If the sampler is slow, a longer deadline doesn't help, because it doesn't make
sampling faster. Lower `[rollout] max_concurrent_rollouts` so fewer rollouts share the
sampler, or lower `[sampling] max_generation_tokens` so each turn is shorter.

If trajectories are growing, you can give them more time. `harbor.yaml` carries a
commented-out `agent_timeout_multiplier`, and uncommenting it at 2.0 doubles every
task's agent deadline, which gives the 900-second tasks 1800 seconds. The agent's
`max_timeout_sec` caps every deadline at 12000 seconds, which is the longest any task
has today, so the longest trial takes no longer than it does now.

Leave it off unless you want that trade. A longer deadline makes the task easier than
stock Terminal-Bench 2.1, so the scores no longer compare to published ones. It also
removes the task's own penalty for a slow trajectory, which changes what the run
rewards. In async mode, a trajectory that spans more policy updates can be discarded as
`stale_before_training` instead of being cut off, so a slower rollout can cost a group
either way, and it holds one of the `max_concurrent_rollouts` slots and its sandbox for
as long as it runs. We haven't measured how that nets out across a run.

## How the agent handles a cut generation

The token limit, `max_generation_tokens`, can cut a generation short. The agent discards
a generation that the token limit cut, so that generation never trains, and samples the
turn again. It tries each turn three times in total. If all three tries are cut, the
episode ends as `truncated` and is graded. That is this agent's choice. The
[tool_agent](../tool_agent/) example ends the episode and grades it instead, and the
cut generation trains.

If every try at the first turn is cut, the agent has discarded every generation it
made. The cookbook still grades that rollout, and it counts in its group's baseline,
but it trains nothing.

The trajectory budget, `max_trajectory_tokens`, limits the tokens of the whole episode.
The agent doesn't sample a generation again when the trajectory budget cut it, because
every retry would get the same room. It keeps the generation, which trains, and the
episode ends as `context_overflow`.

## Sandbox sizes

The pinned tasks are CPU-only, and they declare 2 to 8 GB of RAM. `harbor.yaml` runs
every task at 8 GB of memory. A Blaxel sandbox's disk is half its memory, so each task
gets 4 GB of disk. `loops check` warns about each task that declares more disk than
that.
