# LLM judge

[A custom runtime, in-process, direct](../../patterns/custom_runtime/in_process_direct/)
with a group grader: an `Episode` whose groups a judge model ranks. Each rollout
answers an open question once with the cookbook's `llm()` client; once a group is
complete, `grade_group()` shows the judge every answer to the question and asks for a
ranking, best first. The rank becomes the reward, so the policy learns what the judge
prefers.

**Use it when** a rollout can't be scored on its own, but can be compared with its
siblings: an LLM judge ranking answers, pairwise preferences, a diversity bonus.

**Files**

- `judge_episode.py`: the episode, `RankedEpisode(Episode[Question, JudgeOptions])`.
  - Each task line is validated as a `Question`, and `[rollout_runtime.config.options]` as
    `JudgeOptions`, both at `loops check`: a bad or misspelt option is refused there.
  - `run()` asks the policy once and keeps the answer.
  - `grade()` gives each rollout a small reward of its own: 1 when the answer has 1 to
    `max_words` words, else 0.
  - `grade_group()` asks the judge, then composes the two rewards: `rank_weight` of
    the rank, and the rest of `grade()`'s.
- `tasks.jsonl`: eight questions to answer in a sentence or two.
- `train.toml`: no environment. The episode keeps its answer.
- `train_blaxel.toml`: `SandboxedRankedEpisode`, an `Episode[Question, JudgeOptions]` with `execution_kind = Blaxel`. One Blaxel sandbox per episode, provisioned and released by the run:
  `setup()` attaches with `await setup.sandbox.client()`, `run()` writes the answer into
  it, and `grade()` reads it back from there with `grading.execution_sandbox.read_file()`. An answer
  missing from a sandbox still there after the read is no answer; from a sandbox that
  is gone, or after any other failed read, grading fails, to be tried again.

**The judge brings its own client.** `grade_group()` never gets the policy, a sandbox
or a live episode. It gets each member's task, its transcript (the conversation at
the model boundary, each turn marked with whether it trains), its `grade()` reward,
and its metadata, and the validated `JudgeOptions`. It reaches the judge through its
own OpenAI-compatible client, configured in `[rollout_runtime.config.options]`:

| Option | Meaning |
| --- | --- |
| `judge_base_url` | The judge's OpenAI-compatible base URL. |
| `judge_model` | The judge model. |
| `judge_api_key_env` | The name of the environment variable holding the judge's API key. The key itself never goes in the config, which is written to the run's `config.json`. |
| `rank_weight` | How much of the reward is the rank, from 0 to 1; the rest is `grade()`'s. |
| `max_words` | The longest answer `grade()` rewards. |

The judge reads each answer from the member's transcript: the reply to its last
trainable generation, as the agent received it.

**What the run does with the judge.**

- One judge call per complete group, holding one of `max_concurrent_verifiers`
  verification slots, under `[rollout.grading] timeout_seconds`.
- A failed call is retried up to twice, because the episode declares
  `grading_retries = 2`. A reply that doesn't rank each answer once is a
  failure another try may fix. A key variable that is not set is refused at
  `loops check`; one unset after it is a `PermanentGradingFailure`, never retried.
- A group whose judge call fails has its rollouts discarded as
  `grading_failed/<type>`. `max_consecutive_rollout_failures` consecutive failures
  stop the run with "N consecutive group-grading failures", so a judge outage or a
  bad key stops the run instead of discarding every group.
- The judge is never paid for a group about to be dropped. Stale members are dropped
  first, and a group left smaller than `group_size - max_group_losses` is rejected
  before the call.

**Run** from the repository root with `BASETEN_API_KEY` in `.env`, and Blaxel
credentials (`BL_WORKSPACE`, `BL_API_KEY`) for the Blaxel config
(`uv sync --extra blaxel`):

```sh
uv run loops check examples/llm_judge/train.toml
uv run loops run   examples/llm_judge/train.toml
uv run loops run   examples/llm_judge/train_blaxel.toml
```

`check` launches nothing, and does not call the judge. `run` provisions Loops
resources, which stay provisioned until you deactivate the run, and calls the judge
once per group. The run ID is logged on exit.
