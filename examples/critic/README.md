# Critic: one policy in two roles

In this example, one policy plays two roles, and both roles train. The example builds on
the [Harbor, in-process, direct](../../patterns/harbor/in_process_direct/) pattern,
with two producers. A producer is a named source of model calls. It labels each
generation the model makes, and it is not a separate model.

- The `main` producer proposes a label.
- The `critic` producer sees the proposal and gives the final answer.

Both producers sample the same Loops model and low-rank adaptation (LoRA) adapter, at
the same policy version. The verifier scores only the final answer. Both producers are
declared trainable in `[rollout.producers]`, so both generations train the same weights
on that one reward.

## Run it

Run these commands from the repository root, with `BASETEN_API_KEY`, `BL_WORKSPACE` and
`BL_API_KEY` set in `.env`:

```sh
uv sync --extra harbor-blaxel
uv run loops check examples/critic/train_blaxel.toml
uv run loops run   examples/critic/train_blaxel.toml
```

`loops check` launches nothing. `loops run` trains on serverless Loops and starts Blaxel
sandboxes. When the command exits, it leaves the Loops run up, and it logs the run ID
and the `loops deactivate` command that ends the run.

## How it works

- `critic_agent.py` is the agent. It asks for a proposal with `self.llm()`, then asks
  for the final answer with `self.llm("critic")`. It writes the final answer into the
  sandbox, where the verifier reads it.
- `job_blaxel.yaml` is the Harbor job. It runs the shared
  [hill_climb](../../tasks/hill_climb/) task on Blaxel sandboxes.
- `train_blaxel.toml` is the training config, and it declares the producers.

The `[rollout.producers]` table replaces the default, which declares only `main`. So
the table lists `main` again. The proposal's model call names no producer, so it uses
`main`. Without `main` in the table, the run would refuse that call. `loops check`
prints the producers that a run declares, and the producer that a call with no name
uses.
