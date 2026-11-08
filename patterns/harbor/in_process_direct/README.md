# Harbor, in-process, direct

This is the simplest Harbor pattern. You write a Harbor `LoopsAgent` class, which the
Harbor rollout runtime calls inside the training process. The agent asks the model with
the cookbook's `llm()` client. The client records every token exactly, and each model
call is a Python call with no network hop.

Use this pattern when you have Harbor tasks and you're writing the agent in Python. If
your tasks aren't Harbor tasks, start from the
[custom rollout runtime, in-process, direct](../../custom_runtime/in_process_direct/)
pattern instead, which is the simplest pattern overall.

The pattern trains on the shared [hill_climb](../../../tasks/hill_climb/) task. The task
asks a four-choice question, and its verifier rewards the answer `B`. The policy goes
from mostly wrong to mostly right in a handful of updates. That shows the training loop
works. It doesn't show that the model learns anything that generalizes.

## Run it

Run these commands from the repository root, with `BASETEN_API_KEY` set in `.env`.
`train.toml` runs each sandbox on local Docker, so it needs a running Docker daemon and
the extra that `uv sync --extra harbor` installs. `train_blaxel.toml` runs each sandbox
on Blaxel, so it needs `BL_WORKSPACE` and `BL_API_KEY` in `.env`, and the extra that
`uv sync --extra harbor-blaxel` installs.

```sh
uv run loops check patterns/harbor/in_process_direct/train.toml
uv run loops run   patterns/harbor/in_process_direct/train.toml
uv run loops run   patterns/harbor/in_process_direct/train_blaxel.toml
```

`loops check` launches nothing. `loops run` trains Qwen3.6-35B-A3B on serverless Loops
and starts sandboxes. When the command exits, it leaves the Loops run up, and it logs
the run ID and the `loops deactivate` command that ends the run.

## How it works

The pattern has these files:

- `direct_agent.py` is the agent. It makes one `generate` call, then writes the answer
  to `/app/answer.txt` in the sandbox, where the verifier reads it.
- `job.yaml` and `job_blaxel.yaml` are the Harbor job, one for each environment
  backend, Docker and Blaxel.
- `train.toml` and `train_blaxel.toml` are the training configs for those two jobs.
