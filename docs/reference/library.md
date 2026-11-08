# Using the cookbook as a library

You can load, check and train a training config from your own Python code instead of
the `loops` command. This page explains how `load()` finds files, what `check()` and
`train()` do, and which packages make up the public API.

```python
import asyncio

from loops_cookbook.project import load


async def main() -> None:
    project = load("train.toml", overrides={"training.max_steps": 5})
    for line in project.describe():
        print(line)
    for line in (await project.check()).describe():
        print(line)
    outcome = await project.train()
    print(outcome.status, outcome.steps)


if __name__ == "__main__":
    asyncio.run(main())
```

Keep the `if __name__ == "__main__":` guard. An SFT run renders rows in worker
processes that import your script again, and without the guard each worker would
start the run again.

## Loading a config

`load()` reads a TOML or JSON training config. The config has a top-level `recipe`
key, such as `recipe = "rl"` or `recipe = "sft"`, and `load()` uses that key to pick
the training method, as the `loops` command does. A config without a `recipe` key is
refused.

The returned project keeps the config's path as you named it in `project.path`, where
it came from in `project.source` and the validated settings in `project.config`.
`load()` also makes the config's directory importable, so a module beside the config
can be named by its file name.

To override settings, pass them as dotted keys, such as
`overrides={"training.max_steps": 5}`.

Paths resolve in two different ways:

- A path written in the config is relative to the config's directory. If the config is
  a symlink, the cookbook uses the file it points to, so the config's paths, modules
  and `.env` file are found beside that file.
- A path in an override, and the output directory, are relative to the working
  directory at the time `load()` runs.

`load()` takes the path as given. Unlike the `loops` command, it doesn't fall back to
the copy of an example that is installed with the package.

## Credentials

Set `BASETEN_API_KEY`, and any other credentials your run needs, in the process's
environment. `load()` reads a `.env` file only when you pass `env_file`, which is
relative to the config. It sets only the variables that the process hasn't already
set. The `loops` command looks for a `.env` file beside the config and in the working
directory, with the same rule.

## What a check prints

`loops check` starts by printing which files the config used, and a run's `run.log`
starts with the same lines:

```text
Config: /home/me/project/train.toml -> /home/me/shared/train.toml
  paths, modules: beside /home/me/shared
  .env: /home/me/shared/.env, /home/me/project/.env
  runs: /home/me/project/runs
```

`project.describe()` returns those lines as a list of strings.

## Checking and training

Both operations are async.

`check()` does what `loops check` does and returns a report. The report's `describe()`
returns, as a list of strings, the lines that `loops check` prints after the
configuration lines above. A check can download model or data files, and an RL check
writes a run directory under `output_dir/preflight`. A check never starts a trainer.

`train()` does what `loops run` does and returns the run's `Outcome`. The outcome has
the run's `status`, the number of optimizer `steps` this call completed and a `reason`
when the run ended without completing. Its `ok` is true when the status is `completed`.

The library doesn't configure logging, so your application keeps control of it. For
custom wiring, call each recipe's `recipe.check` and `recipe.train` directly.

## The public packages

Import each name from the one package that defines it.

- `loops_cookbook.runtime` defines the `RolloutRuntime` contract that a rollout runtime
  implements, and the attempt and evidence types the rollout runtime receives. A
  training config names its rollout runtime's factory in
  `[rollout_runtime].factory`.
- `loops_cookbook.model_access` provides attempt leases, token capture,
  `ModelAccessMode` and the OpenAI-compatible gateway. The gateway serves agents that
  call the model through the Chat Completions API, which is what a rollout runtime with
  `openai_http` model access uses.
- `loops_cookbook.rl` is the RL recipe. Its core, which includes `Run`, `Pipeline`,
  `Trainer` and GRPO, doesn't depend on any environment. `rl.cli` and `rl.recipe` wire
  that core to a rollout runtime and to providers.
- `loops_cookbook.llm` provides `LoopsLLM`, which is the handle that a rollout runtime's
  agents call the model through. It also provides the model client and the Prime
  renderer adapter.
- `loops_cookbook.resources` provides the resource registry and journal behind
  `attempt.services.resources`, which work the same way for every provider.
  `loops_cookbook.providers` holds the adapters, configured in `[providers.<name>]`,
  that the registry acts through.
- `loops_cookbook.harbor` runs Harbor as one rollout runtime among others. It provides
  `create_runtime` and `LoopsAgent`.
- `loops_cookbook.sft` is the SFT recipe. Its core is its data contracts, the rule for
  which tokens train, and `Run`. `sft.cli` and `sft.recipe` wire that core together,
  and `sft.recipe.train` is the entry point that `loops run` calls.

A custom rollout runtime is an `Episode` class from `loops_cookbook.episodes`. The
cookbook runs the class as a `RolloutRuntime` and reuses everything else.
[patterns/custom_runtime](../../patterns/custom_runtime/) has one `Episode` for each
supported way of running an agent, and the smallest is
[in_process_direct](../../patterns/custom_runtime/in_process_direct/).
