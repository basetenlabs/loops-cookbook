# Harbor, local

In this pattern, the agent is its own program. `LocalAgent` is the agent entry in the
Harbor job. For each trial, which is Harbor's word for one attempt at a task,
`LocalAgent` starts the program on the training machine. The program gets these
environment variables:

- `OPENAI_BASE_URL` and `OPENAI_API_KEY` give the model's address and a key.
- `OPENAI_MODEL` is the model name that the program must send.
- `AGENT_INSTRUCTION` holds the task.
- The trial sandbox's usage environment holds the variables that the program's tools
  need to act in the sandbox through the provider's API.

The program never creates or deletes the sandbox.

Use this pattern when you have Harbor tasks and an agent that runs as its own program,
such as a binary, a CLI, code in another language, or code you can't import or edit.

## Run it

This pattern runs on Blaxel. `LocalAgent` needs the trial sandbox's usage environment,
which Harbor's Docker environment doesn't provide. The cookbook's Daytona environment
provides one too, but the example program reaches its sandbox only through the Blaxel
SDK.

Install the extra with `uv sync --extra harbor-blaxel`. Then run these commands from
the repository root, with `BASETEN_API_KEY`, `BL_WORKSPACE` and `BL_API_KEY` set in
`.env`:

```sh
uv run loops check patterns/harbor/local/train_blaxel.toml
uv run loops run   patterns/harbor/local/train_blaxel.toml
```

`loops check` launches nothing. `loops run` provisions a dedicated trainer and sampler
for Qwen3.5-4B, which bill until you deactivate the run, and it starts Blaxel
sandboxes. When the command exits, it leaves the Loops run up, and it logs the run ID
and the `loops deactivate` command that ends the run.

## How it works

The pattern has these files:

- `local_agent.py` holds `LocalAgent`, which starts the program. The job sets the
  program's command in `agents[].kwargs.command`.
- `agent_program.py` is the program. It makes one model call over HTTP, then one tool
  call that writes the answer into the Blaxel sandbox. It uses only the standard
  library, apart from the Blaxel SDK.
- `job_blaxel.yaml` and `train_blaxel.toml` are the Harbor job and the training config.
  They run the shared [hill_climb](../../../tasks/hill_climb/) task.

`LocalAgent` records how the program exited as `agent_exit` in the trial's metadata.
When the program exits with a nonzero status or on a signal, `LocalAgent` raises
Harbor's `NonZeroAgentExitCodeError`. Harbor then handles the trial as it handles any
failed agent. See [harbor.md](../../../docs/reference/harbor.md) for what that means.
