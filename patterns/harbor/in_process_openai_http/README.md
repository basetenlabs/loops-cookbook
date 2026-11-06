# Harbor, in-process, openai_http

A Harbor `LoopsAgent` class that the runtime calls inside the training process, and
that reaches the model through an OpenAI client pointed at the attempt's endpoint.
The run's gateway behind it records the exact tokens of every reply. It is the
[in-process direct pattern](../in_process_direct/) with only the model call changed:
the configs are the same experiment; the agent class declares
`model_access = ModelAccessMode.OPENAI_HTTP`.

**Use it when** you have Harbor tasks and a Python agent built on an OpenAI client,
including one you import but don't own, such as a submodule.

**Files**

- `http_agent.py`: the agent. One `chat.completions.create` call, then write the
  answer to `/app/answer.txt` in the sandbox.
- `job.yaml`, `job_blaxel.yaml`: the Harbor job on each backend, on the shared
  [hill_climb](../../../tasks/hill_climb/) task.
- `train.toml`, `train_blaxel.toml`: the training config for each job.

**Run** from the repository root with `BASETEN_API_KEY` in `.env`, and Docker running or Blaxel credentials (`uv sync --extra harbor-blaxel`):

```sh
uv run loops check patterns/harbor/in_process_openai_http/train.toml
uv run loops run   patterns/harbor/in_process_openai_http/train.toml
uv run loops run   patterns/harbor/in_process_openai_http/train_blaxel.toml
```

`check` launches nothing; `run` provisions Loops resources, which stay provisioned
until you deactivate the run. The run ID is logged on exit.
