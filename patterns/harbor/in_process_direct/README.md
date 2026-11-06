# Harbor, in-process, direct

The simplest pattern: a Harbor `LoopsAgent` class that the runtime calls inside the
training process, and that asks the model with the cookbook's `llm()` client. Every
token is recorded exactly, with no extra hop.

**Use it when** you have Harbor tasks and are writing the agent in Python.

**Task:** [hill_climb](../../../tasks/hill_climb/). A four-choice question whose
verifier rewards `B`. The policy goes from mostly wrong to mostly right in a handful
of updates. This shows that the loop works, not that anything generalizes.

**Files**

- `direct_agent.py`: the agent. One `generate` call, then write the answer to
  `/app/answer.txt` in the sandbox for the verifier.
- `job.yaml`, `job_blaxel.yaml`: the Harbor job, on each backend.
- `train.toml`, `train_blaxel.toml`: the training config for each job.

**Run** from the repository root with `BASETEN_API_KEY` in `.env`, and Docker
running or Blaxel credentials (`uv sync --extra harbor-blaxel`):

```sh
uv run loops check patterns/harbor/in_process_direct/train.toml
uv run loops run   patterns/harbor/in_process_direct/train.toml
uv run loops run   patterns/harbor/in_process_direct/train_blaxel.toml
```

`check` launches nothing. `run` provisions Loops resources and starts sandboxes. The
Loops run stays provisioned until you deactivate it; the run ID is logged on exit.
