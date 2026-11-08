# Async scheduling

This example shows asynchronous scheduling. It is the
[custom rollout runtime, in-process, direct](../../patterns/custom_runtime/in_process_direct/)
pattern with `mode = "async"` in `[training]`. In async mode, the run samples the
rollouts for the next update while the current update trains. A rollout can come from a
policy up to two updates older than the one it trains, as `max_steps_off_policy = 2`
sets.

The episode, `answer_episode.py`, is the pattern's own episode, unchanged. Async
scheduling is a setting of the run, not of the episode.

## Run it

Run these commands from the repository root, with `BASETEN_API_KEY` set in `.env`. The
Blaxel config also needs `BL_WORKSPACE` and `BL_API_KEY` in `.env`, and the Blaxel
extra, which `uv sync --extra blaxel` installs.

```sh
uv run loops check examples/async_scheduling/train.toml
uv run loops run   examples/async_scheduling/train.toml
uv run loops run   examples/async_scheduling/train_blaxel.toml
```

`loops check` launches nothing. `loops run` trains on serverless Loops. When the
command exits, it leaves the Loops run up, and it logs the run ID and the
`loops deactivate` command that ends the run.

## The configs

- `train.toml` is a smoke test with no sandbox. It saves a checkpoint at every step.
- `train_blaxel.toml` is the pattern's Blaxel experiment, scheduled asynchronously. It
  gives each episode one Blaxel sandbox.
