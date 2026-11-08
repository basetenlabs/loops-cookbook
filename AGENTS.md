# Working in the Loops cookbook

This repository is Baseten's cookbook for Loops. It gives Loops users a working
starting point for reinforcement learning (RL) and supervised fine-tuning (SFT) on top
of the `baseten-loops` SDK. Most of what it contains is copied by users, so the code,
the configs and the prose all have to be easy to read and easy to adapt.

Read this file before you change anything. The development commands are in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Learn the concepts first

The cookbook builds on Loops, so you need both sets of concepts before you write about
either.

To understand the cookbook, read these in order:

1. [README.md](README.md) says what the cookbook is for and how a first run works.
2. [patterns/README.md](patterns/README.md) explains the ways an agent can run and
   reach the model.
3. [docs/README.md](docs/README.md) explains how the cookbook is built and lists the
   exact words the project uses for each concept.
4. [docs/reference/](docs/reference/README.md) holds the exact rules for each area.
   Read the page for the area you are changing.

To understand Loops itself, read the Baseten documentation. Start with the
[Loops overview](https://docs.baseten.co/loops/overview) and
[concepts](https://docs.baseten.co/loops/concepts), then read
[designing RL environments](https://docs.baseten.co/loops/rl-environments) and the
[SDK reference](https://docs.baseten.co/reference/sdk/loops/training-client). If your
tools include the Baseten docs search, use it.

When a page in this repository needs a Loops concept, such as a trainer, a sampler or a
checkpoint, link to the Baseten documentation instead of explaining the concept again.

## Write plain English

Write for an engineer who knows Python and the basics of RL but has never used Loops
or the cookbook. They read the page once, top to bottom, and they should understand
each sentence the first time they read it.

Follow these rules in every README, reference page, docstring, comment and PR
description:

1. **Write complete sentences.** Each sentence has a subject and a verb. This applies
   to bullet points and table cells that explain something. A list of names can stay a
   list of names.
2. **Introduce a term before you use it.** Explain a new term in plain words where the
   reader first needs it. After that, use the same word every time, and use the words
   listed in [docs/README.md](docs/README.md#words-we-use-exactly).
3. **Put one idea in each sentence.** Test every joint, meaning each colon, semicolon,
   dash or subordinate clause. Ask whether the part after the joint extends the idea
   before it or branches into a new one. If it branches, split the sentence. Then
   reread each half on its own and add whatever the split shows is missing, such as a
   reason that the joint only implied.
4. **Make every colon deliver what the words before it promise.** Read the colon as
   "namely". If the sentence still holds together, the colon passes, as in "Run three
   commands: `check`, `run` and `resume`." A colon fails when it stands in for a verb,
   as in "`check`: launches nothing", or when it joins two thoughts. Write "`check`
   launches nothing" instead.
5. **Order a page the way a reader learns.** Say what the thing is and when to use it.
   Then show how to run it. Then explain how it works. Put edge cases and exact rules
   last, or link to the reference page that has them.
6. **Show a concrete example** before or instead of an abstract description.
7. **Address the reader and use the active voice.** Write "Run `loops check` first", not
   "a check should be run first".
8. **Spell out an abbreviation the first time a page uses it**, such as group relative
   policy optimization (GRPO) or low-rank adaptation (LoRA).
9. **Keep essential information out of parentheses.** If the reader needs it, give it a
   sentence of its own.
10. **Write about the code as it is now.** Leave out PR numbers, earlier designs and the
    arguments that settled them. That history belongs in commit messages and PR
    descriptions. The contributor docs under `docs/` may explain why a design was
    chosen.
11. **Don't oversell.** Leave out words such as "simply", "just", "easily" and
    "seamlessly".

Once each sentence holds one idea, check its words:

- Give "the same", "this" and "it" something the reader has already seen to point
  back to.
- Keep one subject through a sentence, or link a new subject to the old one.
- Give each action to whoever performs it. A setting doesn't decide anything. The
  cookbook decides, using the setting.

### Before and after

These examples are earlier versions of this repository's docs.

> **Before:** Loops runs the expensive half, a LoRA trainer and a paired sampler that
> hot-swaps to each new policy; the cookbook runs the rest.
>
> **After:** Loops runs the GPUs. It trains a LoRA adapter on your model, and it serves
> the latest version of that adapter from a sampler. The cookbook runs everything else,
> including your agent, its environment and the grading.

> **Before:** A deadline (`agent_timeout_seconds`) that cuts `run()` before the reply
> arrives leaves nothing to train: the rollout is discarded as `unversioned` and counts
> as a sampling failure. One cut after the reply is graded, as 0 when no answer was
> kept.
>
> **After:** The run stops `run()` when the agent's deadline, `agent_timeout_seconds`,
> passes. If that happens before the model replies, there is nothing to train on. The
> cookbook discards the rollout as `unversioned` and counts it as a sampling failure.
> If it happens after the model replies, the cookbook grades the rollout as usual, and
> an episode that kept no answer scores 0.

## Code comments and docstrings

The same rules apply to code. In addition:

- **Explain why, not what.** The code already says what it does. A comment explains a
  reason, a constraint or a non-obvious consequence.
- **Keep comments short.** One or two complete sentences is usually enough.
- **Start a docstring with one sentence** that says what the function or class does.
- **Put a config comment on its own line above the setting** when it needs more than a
  few words. Don't chain fragments after the value.

> **Before:**
>
> ```toml
> agent_timeout_seconds = 90   # cut before the reply: unversioned; after it: graded, 0 if no answer was kept
> ```
>
> **After:**
>
> ```toml
> # The agent's deadline. A rollout cut off before the model replies is discarded.
> # One cut off after the reply is graded, and scores 0 if it kept no answer.
> agent_timeout_seconds = 90
> ```

## Where each kind of text belongs

- **README.md** says what the cookbook is for, how to install it, and how to run a first
  RL job and a first SFT job. It points to everything else.
- **A README in `patterns/` or `examples/`** says what its directory shows, when to use
  it and how to run it.
- **`docs/reference/`** holds the exact rules for one area, such as scheduling or
  failures.
- **`docs/README.md` and the numbered pages beside it** are for contributors who change
  the cookbook itself.
- **`docs/reference/KNOB_GLOSSARY.md` is generated** from the settings models. To change
  it, change the setting's `description` in `src/`, then run
  `uv run python -m loops_cookbook.docs`.

## Checks the docs must pass

The unit tests read the documentation, so run them after you change any Markdown.

- `tests/unit/test_docs.py` checks that every setting a page names in `[table].key`
  form exists in that table, and that the knob glossary matches the settings models.
- `tests/unit/test_public_surface.py` checks that Python code blocks in the docs import
  each name from its public package, not from a submodule.
