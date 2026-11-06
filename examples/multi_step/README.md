# Multi-step

[Harbor, in-process, direct](../../patterns/harbor/in_process_direct/) on a two-step
task, [tasks/hill_climb_steps](../../tasks/hill_climb_steps/). Each step has its own
hidden target and is verified in the agent's environment as soon as it ends. The
agent, `step_agent.py`, is a copy of the pattern's with its classes renamed; what it
does is unchanged.

**Run** from the repository root with `BASETEN_API_KEY` in `.env`, and Blaxel credentials (`BL_WORKSPACE`, `BL_API_KEY`) for the Blaxel config:

```sh
uv sync --extra harbor-blaxel
uv run loops check examples/multi_step/train_blaxel.toml
uv run loops run   examples/multi_step/train_blaxel.toml
```

`check` launches nothing; `run` provisions Loops resources, which stay provisioned
until you deactivate the run. The run ID is logged on exit.
