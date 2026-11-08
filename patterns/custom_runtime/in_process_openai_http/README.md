# Custom rollout runtime, in-process, openai_http

This pattern is the [custom rollout runtime, in-process, direct](../in_process_direct/)
pattern with only the model call changed. Here `run()` calls the model through a
standard OpenAI client. The client points at the OpenAI-compatible endpoint that the
run serves for each attempt, and the run's gateway behind that endpoint records the
exact tokens of every reply. The configs run the same experiment as the direct
pattern. The episode class declares `model_access = ModelAccessMode.OPENAI_HTTP`.

The pattern is an [`Episode`](../../../docs/reference/episode-api.md), so the cookbook
creates one object per attempt and runs it through
`loops_cookbook.episodes:create_runtime`. The cookbook loads the tasks, provisions the
sandbox, enforces the agent's deadline and closes the episode once.

Use this pattern when your tasks aren't Harbor tasks and your episode logic is Python
built on an OpenAI client. [examples/litellm_streaming](../../../examples/litellm_streaming/)
shows the same idea with an agent built on LiteLLM.

## Run it

Run these commands from the repository root, with `BASETEN_API_KEY` set in `.env`. The
Blaxel config also needs `BL_WORKSPACE` and `BL_API_KEY` in `.env`, and the Blaxel
extra, which `uv sync --extra blaxel` installs.

```sh
uv run loops check patterns/custom_runtime/in_process_openai_http/train.toml
uv run loops run   patterns/custom_runtime/in_process_openai_http/train.toml
uv run loops run   patterns/custom_runtime/in_process_openai_http/train_blaxel.toml
```

`loops check` launches nothing. `loops run` trains Qwen3.6-35B-A3B on serverless Loops.
When the command exits, it leaves the Loops run up, and it logs the run ID and the
`loops deactivate` command that ends the run.

## How it works

The pattern has three files:

- `http_episode.py` holds two episodes. `AnswerEpisode` keeps the model's answer, and
  its `grade()` scores that answer. `SandboxAnswerEpisode` writes the answer into the
  attempt's sandbox, and its `grade()` scores what the sandbox holds. Both train on one
  task, a hidden answer to guess, from the dataset
  [tasks/hidden_answer_35b.jsonl](../../../tasks/hidden_answer_35b.jsonl). The configs
  name that dataset in `[rollout_runtime.files] tasks`.
- `train.toml` runs `AnswerEpisode` with no sandbox.
- `train_blaxel.toml` runs `SandboxAnswerEpisode` with one Blaxel sandbox per episode,
  configured in `[rollout_runtime.config.sandbox]`.

`SandboxAnswerEpisode` is an `Episode[AnswerTask, None]` with `execution_kind = Blaxel`,
which declares that it needs a Blaxel sandbox. So `loops check` refuses a config that
has no `[rollout_runtime.config.sandbox]` table, or one that names another provider's
sandbox there. The run provisions the sandbox before `setup()` runs. In `setup()`,
`setup.sandbox.client()` attaches to the sandbox within
`[providers.blaxel] attach_timeout_seconds`. After the episode, the run closes the
client and releases the sandbox.

## When the deadline passes or the answer is missing

The run stops `run()` when the agent's deadline, `agent_timeout_seconds`, passes. If
that happens before the model replies, there is nothing to train on. The cookbook
discards the rollout as `unversioned` and counts it as a sampling failure. If it
happens after the model replies, the cookbook grades the rollout as usual, and an
episode that kept no answer scores 0.

`SandboxAnswerEpisode` also scores 0 when its answer is missing from a sandbox that
still exists, however `run()` ended. `read_file()` tells the two cases apart. When the
file is missing, it asks whether the sandbox is still alive. If the sandbox is gone, or
the read fails in any other way, grading fails and the run retries it.

`grade()` is pure, so it is safe to run again. The episode therefore declares
`grading_retries = 2`, and the run tries a failed grade up to twice more.
