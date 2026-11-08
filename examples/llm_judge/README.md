# LLM judge

This example shows group grading, in which one call scores a whole group of rollouts
together. A judge model ranks each group's answers, and the rank becomes the reward, so
the policy learns what the judge prefers.

The example builds on the
[custom rollout runtime, in-process, direct](../../patterns/custom_runtime/in_process_direct/)
pattern. Its `Episode` adds a group grader. Each rollout answers an open question once,
with the cookbook's `llm()` client. Once a group is complete, `grade_group()` shows the
judge every answer to the question and asks for a ranking, best first.

Use it when you can't score a rollout on its own, but you can compare it with the other
rollouts in its group. An LLM judge that ranks answers is one case. Pairwise
preferences and a diversity bonus are others.

## Run it

Run these commands from the repository root, with `BASETEN_API_KEY` set in `.env`. The
Blaxel config also needs `BL_WORKSPACE` and `BL_API_KEY` in `.env`, and the Blaxel
extra, which `uv sync --extra blaxel` installs.

```sh
uv run loops check examples/llm_judge/train.toml
uv run loops run   examples/llm_judge/train.toml
uv run loops run   examples/llm_judge/train_blaxel.toml
```

`loops check` launches nothing, and it doesn't call the judge. `loops run` provisions a
dedicated trainer and sampler for Qwen3.5-4B, which bill until you deactivate the run,
and it calls the judge once per group. When the command exits, it leaves the Loops run
up, and it logs the run ID and the `loops deactivate` command that ends the run.

## How it works

The example has these files:

- `judge_episode.py` holds the episode, `RankedEpisode(Episode[Question, JudgeOptions])`.
  - `loops check` validates each task line as a `Question`, and
    `[rollout_runtime.config.options]` as `JudgeOptions`. So `check` refuses a bad or
    misspelt option.
  - `run()` asks the policy once and keeps the answer.
  - `grade()` gives each rollout a small reward of its own. The reward is 1 when the
    answer has 1 to `max_words` words, and 0 otherwise.
  - `grade_group()` asks the judge for a ranking, then combines the two rewards. The
    rank's share of the final reward is `rank_weight`, and the rest comes from
    `grade()`'s reward. The best answer's rank is worth 1, and the worst answer's rank
    is worth 0.
- `tasks.jsonl` holds eight questions to answer in a sentence or two.
- `train.toml` runs `RankedEpisode` with no sandbox. The episode keeps its answer in
  memory.
- `train_blaxel.toml` runs `SandboxedRankedEpisode`, an `Episode[Question, JudgeOptions]`
  with `execution_kind = Blaxel`. Each episode gets one Blaxel sandbox, which the run
  provisions and releases. `setup()` attaches to the sandbox with
  `await setup.sandbox.client()`. `run()` writes the answer into the sandbox, and
  `grade()` reads it back with `grading.execution_sandbox.read_file()`.

### The judge brings its own client

`grade_group()` never gets the policy, a sandbox or a live episode. For each member of
the group, it gets the task, the transcript, the reward from `grade()` and the
metadata. The transcript is the conversation at the model boundary, with each turn
marked with whether it trains. `grade_group()` also gets the validated `JudgeOptions`.

`grade_group()` reaches the judge through its own OpenAI-compatible client, which you
configure in `[rollout_runtime.config.options]`:

| Option | Meaning |
| --- | --- |
| `judge_base_url` | It is the judge's OpenAI-compatible base URL. |
| `judge_model` | It names the judge model. |
| `judge_api_key_env` | It names the environment variable that holds the judge's API key. The key itself never goes in the config, because the run writes the config to its `config.json`. |
| `rank_weight` | It sets how much of the reward comes from the rank, from 0 to 1. The rest comes from `grade()`. |
| `max_words` | It sets the longest answer that `grade()` rewards. |

The shipped configs use the judge `openai/gpt-oss-120b` at
`https://inference.baseten.co/v1`, with its key in `BASETEN_API_KEY`.

The judge reads each answer from the member's transcript. The answer is the reply to
the member's last trainable generation, as the agent received it.

### What the run does with the judge

- The run makes one judge call for each complete group. The call holds one of the
  `max_concurrent_verifiers` verification slots, and it must finish within
  `[rollout.grading] timeout_seconds`.
- The run retries a failed call up to twice, because the episode declares
  `grading_retries = 2`. A judge reply that doesn't rank each answer exactly once is a
  failure, because another try may fix it.
- If the key's variable isn't set, `loops check` refuses the config. If the variable
  becomes unset after `check`, the call raises `PermanentGradingFailure`, which the run
  never retries.
- When the judge call for a group fails, the run discards the group's rollouts as
  `grading_failed/<type>`. After `max_consecutive_rollout_failures` failures in a row,
  the run stops with "N consecutive group-grading failures". So a judge outage or a bad
  key stops the run, instead of discarding every group.
- The run never pays the judge for a group that it is about to drop. It first drops the
  stale members, which were sampled from a policy too old to train on. Then, before the
  call, it rejects a group left smaller than `group_size - max_group_losses`.

### When the answer is missing from the sandbox

In `train_blaxel.toml`, `grade()` reads the answer back from the sandbox. If the answer
is missing from a sandbox that still exists after the read, the episode has no answer,
and `grade()` gives it 0. If the sandbox is gone, or the read fails in any other way,
grading fails and the run tries it again.
