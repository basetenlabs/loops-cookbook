# Custom rollout runtime, local

This pattern trains an agent that your episode doesn't own. The agent is its own
program, and an [`Episode`](../../../docs/reference/episode-api.md) launches it.

For each attempt, the cookbook provisions a sandbox from
`[rollout_runtime.config.sandbox]`. Then `local_episode.py` starts the agent program on
the training machine, as a supervised process that the run can stop. The program gets
these environment variables:

- `OPENAI_BASE_URL` and `OPENAI_API_KEY` give the model's address and a key.
- `OPENAI_MODEL` is the model name that the program must send.
- `AGENT_INSTRUCTION` holds the task.
- The sandbox's usage environment holds the variables that a program needs to reach
  the sandbox.

A separate verifier program scores what the agent left. The cookbook loads the tasks,
releases the sandbox after the episode closes, and removes the attempt's working
directory.

Use this pattern when your tasks aren't Harbor tasks and your agent runs as its own
program, with your own sandboxes and your own verifier.

## Run it

Run these commands from the repository root, with `BASETEN_API_KEY` set in `.env`. The
Blaxel config also needs `BL_WORKSPACE` and `BL_API_KEY` in `.env`, and the Blaxel
extra, which `uv sync --extra blaxel` installs.

```sh
uv run loops check patterns/custom_runtime/local/train.toml
uv run loops run   patterns/custom_runtime/local/train.toml
uv run loops run   patterns/custom_runtime/local/train_blaxel.toml
```

`loops check` launches nothing. Both configs train Qwen3.5-4B on a dedicated
deployment, so `loops run` provisions a trainer and a sampler that bill until you
deactivate the run. When the command exits, it leaves the Loops run up, and it logs the
run ID and the `loops deactivate` command that ends the run.

## How it works

The pattern has these files:

- `local_episode.py` holds the episode, `LocalAgentEpisode`. It is written once for any
  sandbox kind, and named once for each provider, as `DirectoryAgentEpisode` and
  `BlaxelAgentEpisode`. `[rollout_runtime.config.options]` names the agent and
  verifier programs. Its `grade_against` option says what the verifier reads. With
  `"files"`, the verifier reads the `result.json` that the agent wrote in its working
  directory. With `"sandbox"`, it reads the answer that the agent left in its sandbox.
- `agent_program.py` is the agent. It makes one model call over HTTP and one tool call
  in its sandbox, and it writes one result file. It uses only the standard library,
  apart from the Blaxel SDK when its sandbox is a Blaxel sandbox.
- `verifier.py` is a pure verifier that runs as a process, so grading is safe to retry.
  It prints `{"reward": <number>, "answered": <bool>}`.
- `sandbox_io.py` reads and writes files in the sandbox through the sandbox's usage
  environment. The agent and the verifier share it, and it works with either provider.
- `local_sandbox.py` is an example sandbox provider in which each sandbox is a
  directory, so `train.toml` runs anywhere. It is a `SandboxProvider` whose
  `sandbox_kind` is `Directory`. Its client is the directory's `Path`, and the sandbox
  is alive while the directory exists.
- `train.toml` uses the local directory provider and verifies the result file.
  `train_blaxel.toml` uses Blaxel and verifies the sandbox. Both train on
  [tasks/hidden_answer.jsonl](../../../tasks/hidden_answer.jsonl), in which the model
  guesses one hidden answer.

`loops check` refuses a config that has no `[rollout_runtime.config.sandbox]` table, or
one that names another provider's sandbox there. It also refuses options that don't
validate as the episode's `LocalOptions`, and a dataset line that doesn't validate as
its `AnswerTask`.

## When the agent or the verifier fails

An agent program that fails after it calls the model is still graded. The episode
records how the program exited as `agent_exit` in the episode's metadata, such as
`status_1` or `signal_9`. The verifier then scores whatever answer the program left, and
a missing answer scores 0.

The same holds for a program that runs past `agent_timeout_seconds`. The run stops it,
and the verifier grades what it left.

A program that fails before its first model call has left nothing to train on. The
cookbook discards that rollout as `unversioned` and counts it as a sampling failure.
See [docs/reference/failures.md](../../../docs/reference/failures.md).

With `grade_against = "sandbox"`, a missing answer might mean that the sandbox itself
is gone. So when the verifier finds no answer, the episode asks whether the sandbox is
still alive, with `ensure_alive()`, after the read. If the sandbox is gone, grading
fails and the run retries it.

The episode treats a verifier failure in one of two ways:

- If the verifier exits with status 3, or prints anything other than a reward and an
  `answered` flag, it has nothing valid to score. The episode raises
  `PermanentGradingFailure`, and the run doesn't retry it.
- If the verifier exits with any other nonzero status, or on a signal, the episode
  raises an ordinary error, and the run grades the episode again.
