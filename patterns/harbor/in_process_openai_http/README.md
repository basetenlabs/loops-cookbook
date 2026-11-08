# Harbor, in-process, openai_http

This pattern is the [Harbor, in-process, direct](../in_process_direct/) pattern with
only the model call changed. You write a Harbor `LoopsAgent` class, which the Harbor
rollout runtime calls inside the training process. The agent reaches the model through
an OpenAI client. The client points at the OpenAI-compatible endpoint that the run
serves for each attempt, and the run's gateway behind that endpoint records the exact
tokens of every reply. The configs run the same experiment as the direct pattern. The
agent class declares `model_access = ModelAccessMode.OPENAI_HTTP`.

Use this pattern when you have Harbor tasks and a Python agent built on an OpenAI
client. That includes an agent that you import but don't own, such as one in a
submodule.

## Run it

Run these commands from the repository root, with `BASETEN_API_KEY` set in `.env`.
`train.toml` runs each sandbox on local Docker, so it needs a running Docker daemon and
the extra that `uv sync --extra harbor` installs. `train_blaxel.toml` runs each sandbox
on Blaxel, so it needs `BL_WORKSPACE` and `BL_API_KEY` in `.env`, and the extra that
`uv sync --extra harbor-blaxel` installs.

```sh
uv run loops check patterns/harbor/in_process_openai_http/train.toml
uv run loops run   patterns/harbor/in_process_openai_http/train.toml
uv run loops run   patterns/harbor/in_process_openai_http/train_blaxel.toml
```

`loops check` launches nothing. `loops run` trains Qwen3.6-35B-A3B on serverless Loops
and starts sandboxes. When the command exits, it leaves the Loops run up, and it logs
the run ID and the `loops deactivate` command that ends the run.

## How it works

The pattern has these files:

- `http_agent.py` is the agent. It makes one `chat.completions.create` call, then
  writes the answer to `/app/answer.txt` in the sandbox.
- `job.yaml` and `job_blaxel.yaml` are the Harbor job, one for each environment
  backend, Docker and Blaxel. Both run the shared
  [hill_climb](../../../tasks/hill_climb/) task.
- `train.toml` and `train_blaxel.toml` are the training configs for those two jobs.
