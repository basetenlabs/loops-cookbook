# Terminal-Bench 2.1 on Blaxel

**What it shows:** a real agentic workload. A multi-turn bash agent works through
Terminal-Bench 2.1 tasks in Blaxel sandboxes, with GLM-5.3-Flash as the policy, in
async mode. `train.toml` is a small configuration for a first run, not a tuned one.

**Files**

- `terminal_agent.py`: the agent. System prompt, one `bash` tool, parallel tool
  dispatch, output truncation, and stopping rules. A generation cut by the token
  limit is discarded, so it never trains, and the turn is sampled again, up to three
  times; after that the episode ends as `truncated` and is graded (when that is the
  first turn there is nothing to train, and it is discarded as `unversioned`). That is this
  agent's choice: [tool_agent](../tool_agent/) ends and grades instead, and the cut
  generation trains. A generation the trajectory budget cut is not sampled again, since
  every retry gets the same room: it is kept, trains, and ends the episode as
  `context_overflow`. Settings are declared in
  `TerminalOptions` and supplied from `harbor.yaml`. Commands run in `/`; commands
  needing another directory must `cd`.
- `harbor.yaml`: pins the official TB 2.1 dataset, selects Blaxel through the
  cookbook's lifecycle subclass, and sets Harbor retries.
- `train.toml`: async, eight rollouts per group, four groups per update, 50 updates,
  on serverless Loops.
- `echo_loss.py` and `train_echo.toml`: the same run with
  [ECHO](https://arxiv.org/abs/2605.24517), a custom loss: importance sampling on the
  agent's tokens, plus a next-token loss on what its commands print, weighted by
  `observation_weight` (the paper's 0.05). The module's docstring says where it
  differs from the paper.

**Run** from the repository root with `BASETEN_API_KEY`, `BL_WORKSPACE`, and
`BL_API_KEY` in `.env`:

```sh
uv sync --extra harbor-blaxel
uv run loops check examples/terminal_bench/train.toml
uv run loops run   examples/terminal_bench/train.toml
```

For ECHO, add `--extra custom-loss` to the sync and pass `train_echo.toml`.

`run` provisions Loops resources and Blaxel sandboxes, both billable. The pinned
tasks are CPU-only and declare 2 to 8 GB RAM; `harbor.yaml` runs every one at 8 GB
(4 GB of disk), and `check` warns for each task that declares more disk.
