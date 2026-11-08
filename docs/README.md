# Start here

This page is for engineers who change the cookbook. It says what the cookbook is for,
lists the exact words the project uses, and explains who owns what and the principles
that every design choice follows. Four more pages then explain how the cookbook works.
To learn how to use the cookbook, read the [top-level README](../README.md).

## What problem it solves

The cookbook runs reinforcement learning (RL) on agents, and it trains on Baseten Loops.
For each task, it runs the agent many times against the current policy. Each run of the
agent on a task is an episode, and a grader scores it. The cookbook records each episode
as a rollout, which holds the exact tokens the model produced and the reward. It turns
the rollouts into a training step, and then it repeats.

This loop is easy to write badly and hard to get right, for four reasons:

- **The training signal must be exact.** If you re-tokenize text after the fact and
  train on it, you train on tokens the model never produced. Every step has to use the
  sampler's own token IDs and log probabilities, for exactly the generations the agent
  received.
- **Agents are black boxes.** An agent may be a Python class, another program, or a
  framework that talks to the model over OpenAI-style HTTP. The cookbook can't ask the
  agent what happened, so it has to see for itself.
- **Environments cost money and fail.** A sandbox bills until it is deleted, and
  sandboxes crash, vanish and time out. A training run lasts hours, so a run that leaks
  a sandbox on every crash runs up a bill.
- **Failures are ambiguous.** An agent that timed out or crashed might deserve a zero, a
  partial score, or no training signal at all. The answer depends on the task, not on
  the cookbook.

The cookbook solves the first three problems once, so every user gets the solution. It
refuses to solve the fourth, because only the user can.

## Words we use, exactly

Use these words in code, docs and PR descriptions. The last column lists words to avoid
in their place.

| Word | Means | Not |
| --- | --- | --- |
| **recipe** | The training method, which a config names in its `recipe` key. It is either RL, in the style of group relative policy optimization (GRPO), or supervised fine-tuning (SFT). | "the training process", which is *the run*. Harbor is not a recipe. |
| **rollout runtime** | The part of an RL run that produces episodes, configured in `[rollout_runtime]`. It is either the **Harbor rollout runtime** or a **custom rollout runtime**, which is the user's `Episode` class. Both implement the `RolloutRuntime` contract in `runtime/api.py`. | "runtime" alone, "native", "your own runtime" |
| **agent placement** | Where the agent runs. It is **in-process**, **local**, which means another program on the training machine, or **remote**, which means inside the environment and isn't supported yet. | "subprocess", "sidecar" |
| **model access** | How the agent reaches the model. It is either `direct`, the cookbook's `llm()` client, which only an in-process agent can use, or `openai_http`, an OpenAI-compatible endpoint that the run serves. The code calls that endpoint the gateway. | |
| **env backend** | Where an episode's commands run. It can be none at all, a local directory, Docker, Blaxel or Daytona. | |
| **pattern** | One combination of rollout runtime, agent placement and model access. Each pattern is one directory in `patterns/`. | "shape", "setup" |
| **attempt** | One try at a rollout. A retry is a new attempt. | |

## Who owns what

These docs call the cookbook's core the engine. The diagram and the lists below divide
the work between the engine and the user.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/ownership-dark.svg">
  <img alt="Three columns. The engine owns mechanics and evidence: scheduling, every clock and retry, token capture, sealing the rollout, training, every billed sandbox, and the run record. The user must provide, or check refuses: tasks with a typed model, an agent, a grader, the agent's deadline, how it reaches the model, and credentials in the environment. The user decides meaning: what a reward means, what a failure scores, raise or return, which generations train, batch shape and freshness, and hyperparameters. Below: the boundary is the runtime contract, and across it the engine reads only a reward, an exception, Permanent*Failure, and the generations the agent actually received." src="assets/ownership.svg">
</picture>

**The engine owns the mechanics and the evidence:**

- Scheduling, which covers admission, permits, and sync or async mode
- Every clock and every retry
- Token capture, which covers exact token IDs, freshness and delivery
- Sealing each rollout from what it captured
- Training, which covers the advantage, the loss, publishing and checkpoints
- Every billed sandbox
- The run record

**The user must provide these, or `loops check` refuses the run:**

- Tasks, each validated by a typed task model
- An agent, which is `Episode.run()` or a Harbor agent
- A grader, which is `grade()`, `grade_group()` or a Harbor verifier
- The agent's deadline, `agent_timeout_seconds`
- How the agent reaches the model, which the rollout runtime declares once
- Credentials, which come from the environment and never from the config

**The user decides what results mean:**

- What a reward means
- What a timeout, a crash or a truncated turn scores
- Whether a failure raises, which says infrastructure failed, or returns, which says an
  episode happened
- Which producers train, and whether to discard a delivered turn. A producer is a named
  source of model calls, declared in `[rollout.producers]`.
- The group size, the off-policy window and the hyperparameters

`loops check` enforces the second list. It refuses a run that is missing any item,
before a trainer or a sandbox starts. The engine never decides the third list for the
user. For example, there is no setting for what a timeout scores, because such a setting
would mean the engine decides.

## Principles

Each principle below settled a real argument. When a change seems to need an exception,
the answer so far has always been to change the design, not the principle.

1. **A user story comes first.** A feature that no user story needs doesn't ship. Users
   copy everything here, and each feature is more surface for them to learn.
2. **Mechanics and evidence belong to the engine. Meaning belongs to user code.** When
   an agent's deadline cuts it off, the engine seals the episode and grades it on the
   generations the agent received. `grade()` decides what that scores, and it reads
   `grading.timed_out` to learn about the cut. The cookbook once had settings that
   decided this for the user. Each was a silent contract that the user couldn't see, so
   they were deleted.
3. **No implicit contracts.** Across the rollout runtime contract, the engine reads
   only these explicit signals:
   - a reward
   - an exception, which is retried and then discarded
   - `PermanentRolloutFailure` or `PermanentGradingFailure`, which is not retried
   - the generations the agent received

   The engine never matches message text or exception names. For example, an agent that
   raises its own exception *named* `NonZeroAgentExitCodeError` is not treated as if it
   raised Harbor's (`tests/unit/harbor/harbor_test_agents.py`, `ImpostorExitAgent`).
4. **Evidence comes before convenience.** The engine trains on tokens, not text. It
   trains only on generations the agent received, and it checks freshness everywhere.
   The rollout runtime never hands the engine a `Rollout`. The engine builds the
   rollout from what it captured.
5. **Report, don't guess.** When the engine can't know something, it records the facts
   and lets the user act. Every discard has a `cause` of the form
   `<reason>/<sub_reason>` in `rollouts.jsonl`, and the engine reports leftover
   sandboxes by identity. The engine never acts on a guess about what went wrong. For
   example, it reads how many processes a Harbor sandbox's kernel killed for lack of
   memory, and it reports that count as a metric. The count never decides whether a
   rollout is retried, discarded or scored. A log line names a fix only when the fix is
   obvious, such as the setting a refusal names. It never guesses at one.
6. **Fail at `check`, not at hour three.** `check` checks everything it can before
   launch:
   - the typed task and options models
   - deadlines
   - provider settings
   - the resume contract
   - the model and `max_seq_len`, against what Loops serves (`trainer.py`,
     `check_served_model`)
   - platform facts, such as a Blaxel sandbox's disk being half its memory
     (`harbor/runtime.py`, `_report_sandbox_sizes`)
   - the process's open-file limit, against the number of attempts that can be live at
     once (`rl/run.py`, `check_open_file_limit`)
7. **State it once.** A fact that the user states twice drifts. The Blaxel region lives
   only in `[providers.blaxel]`, and `check` refuses a Harbor job that names a region
   of its own. The rollout runtime declares its model access and its grading retries.
   `check` refuses a config that states the model access again, and the config has no
   setting for grading retries.
8. **Settings exist only for what the cookbook itself does.** Mechanics get defaults,
   such as the cleanup and cancellation graces. Anything that carries meaning must be
   stated, such as the agent's deadline or a sandbox's time-to-live (TTL).
9. **Harbor's semantics are kept, and made to work on Loops.** The cookbook never
   reinterprets a Harbor result. A failed agent is graded exactly when Harbor ran its
   verifier.

## What it is not

- **It is not an agent framework.** Agents are user code, in any language or framework.
- **It is not an environment framework.** Tasks and verifiers are user code, or Harbor's.
- **It is not an isolation layer.** An agent program runs as trusted code, with the
  provider credential it is given.
- **It is not a trainer.** Optimization runs on Loops. The cookbook builds the batches
  and drives the trainer.

## Read next

1. [The training engine](1-engine.md) describes the core, the rules that tests hold it
   to, and the decisions behind them.
2. [Interfaces](2-interfaces.md) explains the two ways a user plugs in an environment
   and an agent: a Harbor dataset and agent, or a custom `Episode` class.
3. [Patterns](3-patterns.md) explains the mechanics behind agent placement, model access
   and env backends.
4. [The SFT recipe](4-sft.md) explains how the cookbook turns fixed conversations into a
   trained low-rank adaptation (LoRA) adapter, the rules the code keeps, and the
   decisions behind them.

The first three pages cover the RL recipe. The SFT recipe shares the trainer, model
and output settings with RL, and nothing else. For the exact rules of each area, read
[reference/](reference/). [reference/sft.md](reference/sft.md) has SFT's.
