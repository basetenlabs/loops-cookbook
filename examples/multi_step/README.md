# Multi-step

This example trains on a task with two steps. It builds on the
[Harbor, in-process, direct](../../patterns/harbor/in_process_direct/) pattern, and it
uses the two-step task [tasks/hill_climb_steps](../../tasks/hill_climb_steps/). Each step
has its own hidden target. Harbor verifies each step in the agent's environment as soon
as the step ends.

## Run it

Run these commands from the repository root, with `BASETEN_API_KEY`, `BL_WORKSPACE` and
`BL_API_KEY` set in `.env`:

```sh
uv sync --extra harbor-blaxel
uv run loops check examples/multi_step/train_blaxel.toml
uv run loops run   examples/multi_step/train_blaxel.toml
```

`loops check` launches nothing. `loops run` trains on serverless Loops and starts Blaxel
sandboxes. When the command exits, it leaves the Loops run up, and it logs the run ID
and the `loops deactivate` command that ends the run.

## How it works

- `step_agent.py` is the agent. It is a copy of the pattern's agent with its classes
  renamed, and it does the same thing. Harbor runs it once for each step, with that
  step's instruction, in the same sandbox.
- `job_blaxel.yaml` is the Harbor job, on Blaxel sandboxes.
- `train_blaxel.toml` is the training config.
