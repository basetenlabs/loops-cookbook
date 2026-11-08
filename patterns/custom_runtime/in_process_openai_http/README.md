# Custom runtime, in-process, openai_http

The [custom runtime, in-process, direct](../in_process_direct/) with only the model call
changed: `run()` calls the model through a stock OpenAI client pointed at the attempt's
endpoint, and the run's gateway records the exact tokens of every reply. The configs
are the same experiment; the class declares `model_access = ModelAccessMode.OPENAI_HTTP`.

The pattern is an [`Episode`](../../../docs/reference/episode-api.md): one
object per attempt, which the cookbook runs through
`loops_cookbook.episodes:create_runtime`. It loads the tasks, provisions the sandbox,
enforces the agent deadline, and closes the episode once.

**Use it when** your tasks are not Harbor tasks and your episode logic is Python
built on an OpenAI client. [examples/litellm_streaming](../../../examples/litellm_streaming/)
is the same idea with a LiteLLM agent.

**Files**

- `http_episode.py`: the episode. `AnswerEpisode` keeps its answer, and `grade()`
  scores it; `SandboxAnswerEpisode` writes the answer into the attempt's sandbox, and
  `grade()` scores what the sandbox holds. Its one task, a hidden answer to guess, is
  the dataset [tasks/hidden_answer_35b.jsonl](../../../tasks/hidden_answer_35b.jsonl) that
  `[rollout_runtime.files] tasks` names.
- `train.toml`: no environment, with `AnswerEpisode`.
- `train_blaxel.toml`: one Blaxel sandbox per episode, from `[rollout_runtime.config.sandbox]`,
  with `SandboxAnswerEpisode`, an `Episode[AnswerTask, None]` with `execution_kind = Blaxel`, so `loops
  check` refuses it without that table or with another provider's. The run provisions
  the sandbox before `setup()`, whose `setup.sandbox.client()` attaches to it within
  `[providers.blaxel] attach_timeout_seconds`, and closes the client and releases the
  sandbox after the episode.

A deadline (`agent_timeout_seconds`) that cuts `run()` before the reply arrives leaves
nothing to train: the rollout is discarded as `unversioned` and counts as a sampling
failure. One cut after the reply is graded, as 0 when no answer was kept. So is an
answer missing from a sandbox still there after the read, however `run()`
ended: `grading.execution_sandbox.read_file()` asks. A sandbox that is gone, or any other failed
read, fails grading, to be retried.
`grade()` is pure, so the episode declares `grading_retries = 2`: a failed grade is
tried twice more.

**Run** from the repository root with `BASETEN_API_KEY` in `.env`, and Blaxel credentials (`BL_WORKSPACE`, `BL_API_KEY`) for the Blaxel config (`uv sync --extra blaxel`):

```sh
uv run loops check patterns/custom_runtime/in_process_openai_http/train.toml
uv run loops run   patterns/custom_runtime/in_process_openai_http/train.toml
uv run loops run   patterns/custom_runtime/in_process_openai_http/train_blaxel.toml
```

`check` launches nothing; `run` provisions Loops resources, which stay provisioned
until you deactivate the run. The run ID is logged on exit.
