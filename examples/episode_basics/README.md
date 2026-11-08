# Episode basics

This directory holds small, standalone examples of the public `Episode` API. They show
three ways to grade an episode: on the host, in one sandbox that the agent and the
grader share, and in a separate verifier sandbox. For complete training setups, see the
[custom rollout runtime patterns](../../patterns/custom_runtime/in_process_direct/).

## The smallest example

[hill_climb_episode.py](hill_climb_episode.py) is the smallest example, and it scores on
the host only. The model guesses one label from A to D. It receives a reward of 1 for
`B`, and 0 otherwise. `HillClimbEpisode` declares no sandbox kinds and no options.

Only the task's prompt goes to the model. The hidden target stays on the host, where
`grade()` uses it. The episode strips whitespace from the reply, but a lowercase label,
an explanation or punctuation doesn't pass. The defaults of the task model match the
instruction and the reward of the Harbor hill-climb task in
[tasks/hill_climb](../../tasks/hill_climb/).

[train_hill_climb.toml](train_hill_climb.toml) runs the episode on the task in
[hill_climb_tasks.jsonl](hill_climb_tasks.jsonl). The config declares no providers,
sandboxes or options. The usual concurrency settings for rollouts and verification
still apply.

Run it from the repository root, with `BASETEN_API_KEY` set in `.env`:

```sh
uv run loops check examples/episode_basics/train_hill_climb.toml
uv run loops run examples/episode_basics/train_hill_climb.toml
```

`loops check` launches nothing. Every config in this directory trains Qwen3.5-4B on a
dedicated deployment, so `loops run` provisions a trainer and a sampler that bill until
you deactivate the run. When the command exits, it leaves the Loops run up, and it logs
the run ID and the `loops deactivate` command that ends the run.

### Compare loss reductions

The configs train with [CISPO](https://docs.baseten.co/loops/loss-functions#cispo), the
clipped importance-weight loss that Loops provides. To compare ways of reducing the
loss, keep the same episode and override only training settings. For example, this
command checks a fixed denominator of 16 × 16. That is the 16 episodes that the config
sets for each update, times a budget of 16 action tokens.

```sh
uv run loops check examples/episode_basics/train_hill_climb.toml \
  --training.loss_aggregation fixed_budget --training.loss_token_budget 16
```

Replace `check` with `run` to train that configuration. Use
`--training.loss_aggregation task_mean` to weight each task group equally.

This example has one group per update. The differences from `sum` or `token_mean` show
most when there are several groups of different lengths, or episodes whose advantage is
zero. See [loss aggregation](../../docs/reference/training.md#loss-aggregation) for the
denominators, and for how the budget counts tokens across a multi-turn episode.

## Three ways to grade

[episodes.py](episodes.py) is illustrative user code, not part of the framework. It
shows three standalone kinds of episode:

- `HostScoredEpisode` scores on the host.
- `SharedSandboxEpisode` runs and scores in one Blaxel sandbox.
- `SeparateVerifierEpisode` saves its output on the host and releases the execution
  sandbox. Then it grades in a separate Blaxel verifier sandbox.

All three have the same hook signatures. Each one subclasses `Episode[Question, Options]`
directly, and has its own constructor, sampling hook and grading hook. You can read any
one of them on its own, as the code you would write to grade that way. The class names
belong to the examples. They aren't framework types.

Each example also has a runnable training config:

| Episode | Configuration | Resources |
| --- | --- | --- |
| `HostScoredEpisode` | [train_host_scored.toml](train_host_scored.toml) | It runs on the host only. |
| `SharedSandboxEpisode` | [train_shared_sandbox.toml](train_shared_sandbox.toml) | It gets one sandbox per attempt. |
| `SeparateVerifierEpisode` | [train_separate_verifier.toml](train_separate_verifier.toml) | It releases the execution sandbox before a separate verifier runs. |

The configs share [questions.jsonl](questions.jsonl). Run any one of them with
`uv run loops run examples/episode_basics/<configuration>`. Every config needs
`BASETEN_API_KEY`. The sandbox configs also need the Blaxel extra, installed with
`uv sync --extra blaxel`, and `BL_WORKSPACE` and `BL_API_KEY` in `.env`. They use two
sampling slots and two verification slots. They also set a ten-minute time-to-live (TTL)
on each sandbox, as a backstop for cleanup. Change the Blaxel region to suit your
workspace.

[train_host_scored.toml](train_host_scored.toml) also evaluates the letters in
[questions_val.jsonl](questions_val.jsonl) every two updates. See
[evals](../../docs/reference/training.md#evals). Those questions share the training
prompt but have other answers. So the eval set shows how evals run, not how the model
generalizes, and its score falls as the policy learns that the training answer is "B".

The declarations below are abbreviated. [episodes.py](episodes.py) has the complete
hooks.

```python
class HostScoredEpisode(Episode[Question, Options]): ...


class SharedSandboxEpisode(Episode[Question, Options]):
    execution_kind = Blaxel
    ...


class SeparateVerifierEpisode(Episode[Question, Options]):
    execution_kind = Blaxel
    sandbox_lifetime = "sampling"
    grading_kind = Blaxel
    ...
```

## API shape

- `Episode` takes two type arguments, the task model and the options model. Give both,
  and pass `None` for one that you don't use, as
  `HillClimbEpisode(Episode[HillClimbTask, None])` does.
- `execution_kind` and `grading_kind` name the kinds of sandbox that an episode needs.
  The config's sandbox tables choose the actual provider and its spec, and
  `loops check` refuses a table whose provider doesn't match the kind.
- The hooks have fixed signatures: `setup(Setup)`, `run(Sampling)`, `grade(Grading)`
  and `close(Closing)`. Each hook receives an immutable record of its phase.
- The `Grading` and `Closing` records have `timed_out`, which is true when the agent's
  deadline, `agent_timeout_seconds`, cut `run()` off. The `Grading` record's
  `execution_sandbox` is `None` when the episode sets `sandbox_lifetime = "sampling"`,
  because the run released that sandbox after sampling. In the `Closing` record, a
  verifier sandbox that was never created is `None`.
- A record's sandbox fields are typed as any sandbox kind. Call
  `require_sandbox(Blaxel)` or `require_execution_sandbox(Blaxel)` to get the concrete
  kind, or check the field yourself. The records hold borrowed handles, which the run
  closes after the episode, not durable artifacts.

Every sandbox kind gives the sandbox's name, the environment variables a separate
process needs to reach it, the provider's own client and whether the sandbox is still
alive. File helpers such as `read_file()` and `write_file()` belong to each provider's
kind, such as `Blaxel`. Commands go through the provider's own client. So the Blaxel
examples here are written for Blaxel, and they don't run on another provider.

A config beside the examples names its class in `episode`, such as
`episodes:HostScoredEpisode`. A config that names `episodes:SeparateVerifierEpisode`
also needs a grading sandbox table, `[rollout_runtime.config.grading_sandbox]`. The
cookbook's factory, `loops_cookbook.episodes:create_runtime`, handles validation,
admission, deadlines, retries and the cleanup of registered resources. Group grading,
HTTP model access, supervised processes and metrics go through the same runtime. See
[the Episode reference](../../docs/reference/episode-api.md).

## Test the examples

These tests run the examples offline, through the same driver that runs episodes in a
training run:

```sh
uv run pytest tests/examples/test_episode_basics.py tests/examples/test_hill_climb_episode.py -q
```

This live test checks real Blaxel file transfer, verifier retries and the cleanup after
a cancellation, without starting a training job. The policy's replies are fixed for
these lifecycle checks.

```sh
LOOPS_LIVE_BLAXEL=1 uv run --env-file .env pytest tests/live/test_episode_sandboxes.py -v
```
