# Start here

For engineers who will change the cookbook. This page says what it is for, the
principles every design choice follows, and who owns what. Then four docs explain
how it works. Using the cookbook is in the [top-level README](../README.md).

## What problem it solves

The cookbook runs reinforcement learning on agents, on Baseten Loops: many attempts at
each task, against the live policy, scored, turned into a training step, repeated.
That is easy to write badly and hard to get right, for four reasons:

- **The training signal must be exact.** Train on text re-tokenized after the fact and
  you train on tokens the model never produced. Every step has to use the sampler's own
  token IDs and logprobs, for exactly the generations the agent received.
- **Agents are black boxes.** An agent may be a Python class, another program, or a
  framework talking OpenAI-style HTTP. The cookbook can't ask it what happened; it has
  to see for itself.
- **Environments cost money and fail.** Sandboxes bill until deleted, crash, vanish and
  time out. A training run lasts hours; one that leaks a sandbox per crash is a bill.
- **Failures are ambiguous.** An agent that timed out or crashed might deserve a zero,
  a partial score, or no training signal at all. That depends on the task, not on us.

The cookbook solves the first three once, so every user gets them for free, and refuses
to solve the fourth, because only the user can.

## Who owns what

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/ownership-dark.svg">
  <img alt="Three columns. The engine owns mechanics and evidence: scheduling, every clock and retry, token capture, sealing the rollout, training, every billed sandbox, and the run record. The user must provide, or check refuses: tasks with a typed model, an agent, a grader, the agent's deadline, how it reaches the model, and credentials in the environment. The user decides meaning: what a reward means, what a failure scores, raise or return, which generations train, batch shape and freshness, and hyperparameters. Below: the boundary is the runtime contract, and across it the engine reads only a reward, an exception, Permanent*Failure, and the generations the agent actually received." src="assets/ownership.svg">
</picture>

| The engine owns (mechanics + evidence) | The user must provide (refused at `check`) | The user decides (meaning) |
| --- | --- | --- |
| Scheduling: admission, permits, sync/async | Tasks, each validated by a typed task model | What a reward means |
| Every clock and retry | An agent: `Episode.run()` or a Harbor agent | What a timeout, crash or truncated turn scores |
| Token capture: exact IDs, freshness, delivery | A grader: `grade()` / `grade_group()` or a verifier | Raise (infrastructure) or return (an episode) |
| Sealing the rollout from what it captured | The agent's deadline, `agent_timeout_seconds` | Which producers train; discarding a delivered turn |
| Training: advantage, loss, publish, checkpoint | How it reaches the model, declared once by the runtime | Group size, off-policy window, hyperparameters |
| Every billed sandbox; the run record | Credentials, in the environment, never the config | |

The middle column is enforced: `loops check` refuses a run that is missing any of it,
before a trainer or sandbox starts. The right column is never decided for the user:
there is no setting for what a timeout scores, because a setting would be the engine
deciding.

## Principles

Each of these settled a real argument. When a change seems to need an exception, the
answer so far has always been to change the design, not the principle.

1. **A user story first.** A feature no story needs doesn't ship. Everything here is
   copied by users; each feature is surface they have to learn.
2. **Mechanics and evidence are ours; meaning is user code.** An agent cut by its
   deadline is sealed and graded on what it was delivered, and `grade()` decides what
   that scores (`grading.timed_out`). We once had
   settings for this (`on_agent_timeout`, `train_on_infrastructure`); each was a silent
   contract the user couldn't see, and they were deleted (#50).
3. **No implicit contracts.** Across the runtime contract the engine reads only explicit
   signals: a reward; an exception (retried, then discarded); `PermanentRolloutFailure`
   or `PermanentGradingFailure` (not retried); the generations the agent received. It
   never matches message text or exception names: an agent that raises its own
   exception *named* `NonZeroAgentExitCodeError` is not treated as Harbor's
   (`tests/unit/harbor/harbor_test_agents.py`, `ImpostorExitAgent`).
4. **Evidence over convenience.** Tokens, not text. Only delivered generations. Freshness
   checked everywhere. The runtime never hands the engine a `Rollout`; the engine builds
   it from what it captured.
5. **Report, don't guess.** When the engine can't know, it records the facts and lets the
   user act: every discard has a `cause` of `<reason>/<sub_reason>` in `rollouts.jsonl`;
   leftover sandboxes are reported by identity. No detector feeds a judgement, and no log
   line offers a fix.
6. **Fail at `check`, not at hour three.** Anything checkable is checked before launch:
   typed task and options models, deadlines, provider settings, the resume contract, the
   model and `max_seq_len` against what Loops serves (`trainer.py`, `check_served_model`),
   and platform facts such as a Blaxel sandbox's disk being half its memory
   (`harbor/runtime.py`, `_report_sandbox_sizes`), and the process's open-file limit
   against the attempts that can be live at once (`rl/run.py`, `check_open_file_limit`).
7. **State it once.** A fact the user states twice drifts. The Blaxel region lives only
   in `[providers.blaxel]`; model access and grading retries are declared by the
   runtime; `check` refuses a restatement.
8. **Settings only for what the cookbook itself does.** Mechanics get defaults (cleanup
   and cancellation graces); anything that carries meaning (the agent's deadline, a
   sandbox TTL) must be stated.
9. **Harbor's semantics, made to work on Loops.** The cookbook never reinterprets a
   Harbor result: a failed agent is graded exactly when Harbor ran its verifier.

## What it is not

- **Not an agent framework.** Agents are user code, in any language or framework.
- **Not an environment framework.** Tasks and verifiers are user code or Harbor's.
- **Not an isolation layer.** An agent program runs as trusted code with the provider
  credential it is given.
- **Not a trainer.** Optimization runs on Loops; the cookbook builds the batches and
  drives it.

## Words we use, exactly

| Word | Means | Not |
| --- | --- | --- |
| **recipe** | the training method, named by a config's `recipe` key: RL (GRPO-style) or SFT | "the training process" (say *the run*); Harbor is not a recipe |
| **runtime** | what produces episodes: **Harbor**, or the **custom runtime** (the user's `Episode`) | "native", "your own runtime" |
| **agent placement** | where the agent runs: **in-process**, **local** (another program on the training machine), **remote** (inside the environment; not supported yet) | "subprocess", "sidecar" |
| **model access** | `direct` (a Python client, in-process only) or `openai_http` (the run's gateway) | |
| **env backend** | where commands run: none, a local directory, Docker, Blaxel, Daytona | |
| **pattern** | runtime × placement × model access; one directory each in `patterns/` | "shape", "setup" |
| **attempt** | one try at a rollout; a retry is a new attempt | |

## Read next

1. [The training engine](1-engine.md): the locked-down core, its rules and decisions.
2. [Interfaces](2-interfaces.md): a Harbor dataset and agent, or a custom `Episode`.
3. [Patterns](3-patterns.md): placement, model access and env backends.
4. [The SFT recipe](4-sft.md): fixed conversations to a LoRA, its rules and decisions.

The first three cover the RL recipe. The SFT recipe shares the trainer, model and
output settings and none of the rest. Detail when you need it:
[reference/](reference/) for the exact rules, [reference/sft.md](reference/sft.md)
for SFT's.
