# 3-SAT

Boolean satisfiability, one model call per problem, with held-out problems evaluated
while it trains. A problem is 21 variables and 89 clauses of three literals; the policy
works it through in its reply, then answers with the variables it sets to true. It is
single-turn, scored on the host in microseconds, and hard but learnable. No sandboxes,
no extras.

```sh
uv run loops check examples/sat/train.toml
uv run loops run   examples/sat/train.toml
```

[sat_episode.py](sat_episode.py) is the whole runtime:

- **A task is a seed:** it generates the problem, clauses of three distinct variables,
  each negated or not, kept only if a hidden assignment satisfies it, so every problem
  has a solution. `variables` and `clause_ratio` set the size; near 4.26 clauses per
  variable random 3-SAT is at its hardest.
- **The answer is the reply's last line,** `True: 3, 8, 14` or `True: none`. Anything
  else scores 0.
- **A reply cut off at `max_generation_tokens` scores 0 and doesn't train.** The episode
  discards it (`conversation.discard(reply)`), so its 0 still counts in the group's
  baseline, but its reasoning isn't pushed down just for running out of room.
- **Reward:** 1 for an assignment that satisfies every clause, any one, not only the
  hidden one; otherwise 0. The share of clauses it satisfies is kept as `satisfied`.

[problems.jsonl](problems.jsonl) holds the training seeds and
[problems_val.jsonl](problems_val.jsonl) held-out ones. `[eval.sets.val]` scores the
held-out problems every 5 updates, before the first and after the last; see
[evals](../../docs/reference/training.md#evals). The run is async: replies vary in
length, so no update waits for the slowest.

## Picking the problem size

A sweep of base Qwen3.6-35B-A3B over problem sizes, 24 problems × 8 samples each, thinking
off:

| variables | 12 | 15 | 18 | 21 | 25 |
|---|---|---|---|---|---|
| pass@1 | 0.49 | 0.36 | 0.09 | 0.11 | 0.03 |
| pass@8 | 1.00 | 0.92 | 0.46 | 0.67 | 0.17 |

21 variables: the model rarely solves a problem on one try but usually does within
eight, so most groups of 8 have both a success and a failure to learn from. Thinking is
off (`enable_thinking = false`): the policy works the problem in its reply instead,
about 12k tokens, and ends it on its own. With thinking on, half its replies ran past
32k tokens or ended mid-thought.

## A run

This config on serverless Loops. Held-out problems solved went from 0.04 before
training to 0.41 after 50 updates. Training reward is shown two ways: over every
rollout (`reward/mean_with_unusable`), and over the rollouts that trained
(`reward/mean`), which leaves out groups whose rollouts all scored the same.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/results-dark.svg">
  <img alt="Held-out eval reward 0.04, 0.06, 0.09, 0.12, 0.13, 0.16, 0.24, 0.29, 0.31, 0.32 and 0.41 after 0 to 50 updates in steps of 5, with training reward over every rollout rising from about 0.06 to about 0.32 and over the rollouts that trained from about 0.18 to about 0.36" src="assets/results.svg">
</picture>
