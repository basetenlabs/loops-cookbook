# Critic: one policy in two roles

[Harbor, in-process, direct](../../patterns/harbor/in_process_direct/) with two
producers. `main` proposes a label; `critic`, the same Loops model and LoRA at the
same policy version, sees the proposal and gives the final answer. The verifier scores
only the final answer, and both producers are declared trainable in
`[rollout.producers]`, so both generations train the same weights on that one reward.
The table replaces the default, which declares only `main`, so it lists `main` again;
without it, the proposal's call, which names no producer, would be refused.
`loops check` prints the producers a run declares and the one such a call uses.

**Run** from the repository root with `BASETEN_API_KEY` in `.env`, and Blaxel credentials (`BL_WORKSPACE`, `BL_API_KEY`) for the Blaxel config:

```sh
uv sync --extra harbor-blaxel
uv run loops check examples/critic/train_blaxel.toml
uv run loops run   examples/critic/train_blaxel.toml
```

`check` launches nothing; `run` provisions Loops resources, which stay provisioned
until you deactivate the run. The run ID is logged on exit.
