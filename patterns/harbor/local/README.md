# Harbor, local

The agent is its own program. `LocalAgent`, the Harbor job's agent entry, starts it
on the training machine for each trial, with the model's address and key in
`OPENAI_BASE_URL` and `OPENAI_API_KEY`, `OPENAI_MODEL` (the model name the program must
send), the task as `AGENT_INSTRUCTION`, and the trial sandbox's usage environment, so
its tools act in the sandbox through the provider's API. It never creates or deletes
the sandbox.

**Use it when** you have Harbor tasks and an agent that runs as its own program: a
binary, a CLI, another language, or code you can't import or edit.

A program that exits nonzero or on a signal raises Harbor's `NonZeroAgentExitCodeError`,
so what happens next is Harbor's, as for any failed agent.

**Files**

- `local_agent.py`: `LocalAgent`, which starts the program. Its command is
  `agents[].kwargs.command` in the job.
- `agent_program.py`: the program. One model call over HTTP, then one tool call that
  writes the answer into the Blaxel sandbox. Standard library only, apart from the
  Blaxel SDK.
- `job_blaxel.yaml`, `train_blaxel.toml`: the Harbor job and training config, on the
  shared [hill_climb](../../../tasks/hill_climb/) task.

Blaxel only: the program needs the sandbox's usage environment, which the cookbook's
Blaxel environment provides and Harbor's Docker environment does not.

**Run** from the repository root with `BASETEN_API_KEY` in `.env`, and Blaxel credentials (`uv sync --extra harbor-blaxel`):

```sh
uv run loops check patterns/harbor/local/train_blaxel.toml
uv run loops run   patterns/harbor/local/train_blaxel.toml
```

`check` launches nothing; `run` provisions Loops resources, which stay provisioned
until you deactivate the run. The run ID is logged on exit.
