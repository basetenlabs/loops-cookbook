# Async scheduling

[A custom runtime, in-process, direct](../../patterns/custom_runtime/in_process_direct/)
with `mode = "async"`: rollouts for the next update are sampled while the current one
trains, up to two updates off-policy (`max_steps_off_policy = 2`). The episode,
`answer_episode.py`, is the pattern's own, unchanged: async scheduling is a setting of
the run, not of the episode. `train.toml` is a smoke with no environment
that checkpoints every step; `train_blaxel.toml` is the pattern's Blaxel experiment,
one sandbox per episode, scheduled asynchronously.

**Run** from the repository root with `BASETEN_API_KEY` in `.env`, and Blaxel
credentials (`BL_WORKSPACE`, `BL_API_KEY`) for the Blaxel config (`uv sync --extra blaxel`):

```sh
uv run loops check examples/async_scheduling/train.toml
uv run loops run   examples/async_scheduling/train.toml
uv run loops run   examples/async_scheduling/train_blaxel.toml
```

`check` launches nothing; `run` provisions Loops resources, which stay provisioned
until you deactivate the run. The run ID is logged on exit.
