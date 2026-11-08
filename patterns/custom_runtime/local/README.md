# Custom runtime, local

An [`Episode`](../../../docs/reference/episode-api.md) that trains an agent
it does not own. For each attempt the cookbook provisions a sandbox from
`[rollout_runtime.config.sandbox]`; then `local_episode.py` starts the agent program on the
training machine as a supervised process, with the model's address and key in
`OPENAI_BASE_URL` and `OPENAI_API_KEY`, `OPENAI_MODEL` (the model name the program must
send), the task as `AGENT_INSTRUCTION`, and the sandbox's usage environment. A
separate verifier program scores what the agent left. The cookbook loads the tasks,
releases the sandbox after the episode closes, and removes the attempt's working
directory.

An agent program that fails after calling the model is still graded: its exit is
recorded as `agent_exit` in the episode's metadata, and the verifier scores whatever
answer it left, a missing one as 0. So is one that runs past `agent_timeout_seconds`:
it is stopped and graded on what it left. One that fails before its first model call
left nothing to train, and is discarded as `unversioned`, counted as a sampling
failure (see [docs/reference/failures.md](../../../docs/reference/failures.md)).

**Use it when** your tasks are not Harbor tasks and your agent runs as its own
program, with your own sandboxes and verifier.

**Files**

- `local_episode.py`: the episode, `LocalAgentEpisode`, written once over its sandbox
  kind and named for each provider: `DirectoryAgentEpisode` and `BlaxelAgentEpisode`.
  `[rollout_runtime.config.options]` names the agent and verifier programs, and
  `grade_against` what the verifier reads: `"files"`, the `result.json` the agent
  wrote in its working directory, or `"sandbox"`, the answer it left in its sandbox.
  When that answer is missing, the episode calls `grading.execution_sandbox.ensure_alive()` after the read:
  a sandbox that is gone fails grading, to be retried. `loops check` refuses a
  config without `[rollout_runtime.config.sandbox]` or with another provider's, options that
  are not its `LocalOptions`, or a dataset line that is not its `AnswerTask`.
- `agent_program.py`: the agent. One model call over HTTP, one tool call in its
  sandbox, one result file. Standard library only, apart from the Blaxel SDK when its
  sandbox is Blaxel's.
- `verifier.py`: a pure verifier, run as a process, so grading is safe to retry. It
  prints `{"reward": <number>, "answered": <bool>}`; anything else is a permanent
  grading failure.
- `sandbox_io.py`: files in the sandbox through its usage environment, shared by the
  agent and the verifier, for either provider.
- `local_sandbox.py`: an example provider in which each sandbox is a directory, so
  `train.toml` runs anywhere. It is a `SandboxProvider` whose `sandbox_kind` is
  `Directory`: the client is the directory's `Path`, alive while it exists.
- `train.toml`, `train_blaxel.toml`: the directory backend, verifying the result
  file, and Blaxel, verifying the sandbox. Both train on
  [tasks/hidden_answer.jsonl](../../../tasks/hidden_answer.jsonl), one hidden answer
  to guess.

**Run** from the repository root with `BASETEN_API_KEY` in `.env`, and Blaxel credentials (`BL_WORKSPACE`, `BL_API_KEY`) for the Blaxel config (`uv sync --extra blaxel`):

```sh
uv run loops check patterns/custom_runtime/local/train.toml
uv run loops run   patterns/custom_runtime/local/train.toml
uv run loops run   patterns/custom_runtime/local/train_blaxel.toml
```

`check` launches nothing; `run` provisions Loops resources, which stay provisioned
until you deactivate the run. The run ID is logged on exit.
