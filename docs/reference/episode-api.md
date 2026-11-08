# The Episode API

An `Episode` is the Python class you write for a custom rollout runtime, when your
tasks aren't Harbor tasks. Its `run()` method is your agent, and its `grade()` method
scores what the agent did. The cookbook builds one `Episode` object for each attempt.
It loads the tasks, holds capacity for the attempt, provisions the attempt's sandbox,
enforces the agent's deadline and calls `close()` once at the end. Underneath, the
cookbook runs your class as an ordinary rollout runtime, so every rule in
[failures.md](failures.md) applies to it.

The training config names your class and your tasks in its rollout runtime tables:

```toml
[rollout_runtime]
factory = "loops_cookbook.episodes:create_runtime"   # model access: the class's model_access

[rollout_runtime.config]
episode = "my_env:AnswerEpisode"   # module:Class, importable beside this file
agent_timeout_seconds = 600        # required

[rollout_runtime.files]
tasks = "tasks.jsonl"
```

```python
from loops_cookbook.episodes import Episode, Grading, Sampling


class AnswerEpisode(Episode[None, None]):
    def __init__(self, task, options):
        super().__init__(task, options)
        self.answer = None                       # grade() reads it, however far run() got

    async def run(self, sample: Sampling):
        reply = await sample.model.llm().conversation().generate(
            messages=[{"role": "user", "content": self.task.data["prompt"]}]
        )
        self.answer = (reply.content or "").strip()

    async def grade(self, grading: Grading):
        return float(self.answer == self.task.data["answer"])
```

The `factory` line selects the custom rollout runtime, which runs an `Episode`
class. The config doesn't say how the agent reaches the model, because the class declares that in its `model_access`
variable. `episode` names your class as `module:Class`, and the module must be
importable from the config's directory. `agent_timeout_seconds` is required. The
`tasks` file holds one task on each line.

This episode sends each task's prompt to the model once and keeps the reply. `grade()`
scores 1 when the reply equals the task's answer and 0 otherwise. `__init__` sets
`self.answer` to `None`, so `grade()` can read it even when the deadline stopped `run()`
before it set an answer.

## What you write

The cookbook calls your hooks in a fixed order for each attempt.
[The lifecycle](#the-lifecycle) shows the whole sequence. Two kinds of capacity bound
the hooks. An attempt holds a sampling slot while its agent runs, and the run has
`max_concurrent_rollouts` of them. An attempt holds a verification slot while it is
graded, and the run has `max_concurrent_verifiers` of them. When `run()` ends, the
cookbook seals the rollout. Sealing fixes the generations the agent received as the
evidence the rollout trains on.

| Hook | When it runs | What to know |
| --- | --- | --- |
| `__init__(task, options)` | It runs once per attempt. If the class declares a sandbox, it runs after the sandbox is provisioned. | Initialize everything that `grade()` reads. |
| `async setup(setup: Setup)` | It is optional, and it runs before the agent starts. | It runs under the setup clock, `setup_timeout_seconds`. It holds no sampling slot, and the model isn't available. |
| `async run(sample: Sampling)` | It runs while the attempt holds its sampling slot. | This is the agent. Return episode metadata as a mapping, or return `None`. Any other return value fails the attempt, and the attempt isn't retried. Raise an exception only when the episode didn't happen. |
| `async grade(grading: Grading)` | It runs once the rollout is sealed, in a verification slot. | Return a reward, or a `GradeResult` that also carries metadata. The policy isn't available, so a large language model (LLM) judge brings its own client. When the class defines `grade_group()`, `grade()` is optional and may return `None`. |
| `grade_group(members, options)` | It is an optional async classmethod, and it runs once for each complete group. | Return one final reward for each member, in member order. It sees only recorded evidence. `options` is the episode's validated options. See [group grading](#group-grading). |
| `async close(closing: Closing)` | It is optional, and it runs once when the attempt ends, however the attempt ended. | It runs under the cleanup deadline, `[rollout.cleanup] timeout_seconds`. It also runs after a `setup()` or `run()` that raised or was cut off. It doesn't run when `__init__` raised or never ran, because then there is no episode. When an attempt is kept live because its `setup()`, `run()` or `grade()` ignored cancellation, `close()` runs once that hook ends. |

The class also declares a few facts about itself in class variables:

| Class variable | Default | What it declares |
| --- | --- | --- |
| `model_access` | `ModelAccessMode.DIRECT` | It says how the agent reaches the model. Set it to `OPENAI_HTTP` for an agent that speaks Chat Completions. |
| `grading_retries` | `0` | It says how many more times a failed `grade()` or `grade_group()` is tried. Set it above 0 only when grading is safe to run again, as an LLM judge is. |
| `execution_kind` | `None` | It names the kind of sandbox the agent runs in, such as `Blaxel` or `Daytona`. See [declaring a sandbox](#declaring-a-sandbox). |
| `grading_kind` | `None` | It names the kind of separate sandbox that `grade()` uses. See [a separate grading sandbox](#a-separate-grading-sandbox). |
| `sandbox_lifetime` | `"attempt"` | It says how long the execution sandbox lives. With `"sampling"`, the cookbook closes the sandbox's attached client and releases the sandbox after `run()` stops, before the rollout is admitted to verification. Then `grade()` and `close()` can't use the execution sandbox. See [grading saved artifacts](#grading-saved-artifacts-without-retaining-a-sandbox). |

The class also takes two explicit type arguments, as in
`Episode[TaskModel, OptionsModel]`. They are the pydantic models for the task data and
for the options. Use `None` for one you don't use. See
[typed tasks and options](#typed-tasks-and-options).

`loops check` refuses a class in each of these cases:

- It doesn't define `run()`.
- It defines neither `grade()` nor `grade_group()`.
- Its `setup()`, `run()`, `grade()` or `close()` isn't an `async def`.
- Its `grade_group()` isn't an async classmethod.

A retry is a fresh attempt. The cookbook builds a new `Episode` with a new sandbox, and
the agent runs again.

## What you are given

Your episode receives its task and its options when it is built. Each hook receives a
phase record, which carries what that hook may use.

### The task and the options

`self.task` is the attempt's task. It is one line of the `tasks` file that
`[rollout_runtime.files]` names. The file is in JSON Lines (JSONL) format, with one
JSON object for each task.

- `task.id` is the line's `id`. A line without an `id` is named by its position among
  the tasks, counting from 0.
- `task.data` is the line's data. Each attempt gets its own copy.
- `task.value` is the line as the episode's task model validated it, or `None` when the
  episode has no task model.

The run records a digest of each task's canonical JSON.

`self.options` holds the episode's validated options, as a private copy for each
attempt. It is `None` when the episode has no options model.

### The model

`sample.model` is the policy, and it is available only in `run()`. It gives the agent
the model in one of two ways:

- `sample.model.llm(producer=None)` returns the cookbook's `llm()` client, for direct
  model access.
- `await sample.model.http_connection(producer=None)` returns an OpenAI-compatible
  endpoint and a credential for it.

The optional `producer` argument names a producer from `[rollout.producers]`.
`sample.model.conversation(producer=..., conversation_id=...)` opens a direct session.
It records against the same attempt and producer as a session from
`llm().conversation()`, and it stops working at the same time.

Once `run()` has ended, both `llm()` and `http_connection()` refuse with "model access
for this attempt is sealed". A client taken from `llm()` refuses to sample once the
attempt is sealed, and the HTTP credential is revoked then. So a model that you keep on
`self` and use later fails loudly instead of sampling off the record.

If `run()` starts a task that keeps generating after `run()` returns, you have to stop
it yourself. The cookbook doesn't police generations made between the end of `run()`
and the seal, but any generation after the seal fails.

### Phase records

Each hook receives its own phase record. The records are immutable, keyword-only
dataclasses. Every record has `attempt_id`, `metrics` and `processes`. A hook always
receives its record, and the type annotation on its parameter doesn't change what it
receives.

| Record | Hook | Fields besides the three shared ones |
| --- | --- | --- |
| `Setup` | `setup()` | `sandbox` is the execution sandbox, or `None`. The record has no model. |
| `Sampling` | `run()` | It has `model`, `sandbox` and `trajectory_tokens`. `sandbox` is the execution sandbox, or `None`. |
| `Grading` | `grade()` | It has `rollout`, `timed_out`, `execution_sandbox` and `grading_sandbox`. |
| `Closing` | `close()` | It has `timed_out`, `execution_sandbox` and `grading_sandbox`. |

`Grading` and `Closing` expose only the sandboxes that are still accessible. An
execution sandbox released after sampling is `None`, and so is a grading sandbox that
was never provisioned successfully. The record doesn't stop you from keeping a
software development kit (SDK) client beyond its lifetime, so don't reuse a client or environment that you saved once
its sandbox has been released.

`sample.trajectory_tokens` is the run's trajectory budget. It is
`[sampling] max_trajectory_tokens` when that is set, then `[model] max_seq_len`, and
`None` when neither is set. An episode that shapes its reward near the context limit
reads the budget here instead of restating it in its options.
[Model access](model-access.md) explains how a generation is fitted to the budget.

The records have helpers that return a sandbox as a specific kind:

- `setup.require_sandbox(Kind)` and `sample.require_sandbox(Kind)` return the execution
  sandbox.
- The `Grading` and `Closing` records have `require_execution_sandbox(Kind)` and
  `require_grading_sandbox(Kind)`.

Each helper raises `RuntimeError` when the sandbox is missing and `TypeError` when the
sandbox is of an incompatible kind. The optional sandbox fields stay set after a
`setup()` that failed partway, so `close()` can still clean up.

### Using a sandbox

Every sandbox is a `SandboxKind`, which has `name`, `environment`, `client()`,
`alive()` and `ensure_alive()`. What the client can do, such as file and command
operations, depends on the provider. Narrow the sandbox to your provider's kind before
you use its SDK:

```python
from loops_cookbook.episodes import Sampling
from loops_cookbook.providers.blaxel import Blaxel


async def prepare(sample: Sampling):
    sandbox = sample.sandbox
    assert isinstance(sandbox, Blaxel)
    box = await sandbox.client()  # blaxel.core.sandbox.SandboxInstance
```

The first `client()` call attaches to the sandbox under the provider's attach
deadline, and later calls return the same client. The cookbook closes the client after
`close()` returns, or when it releases the sandbox after sampling.

`ensure_alive()` raises `SandboxGone` when the sandbox has been lost. When `grade()`
can't find an answer, check that the sandbox is still alive before you score the
answer as zero, because a missing file in a lost sandbox proves nothing.

Blaxel and Daytona sandboxes also provide `await sandbox.write_file(path, content)`,
which writes UTF-8 text through the managed client. Their `read_file(path)` returns
the file's text. It returns `None` only after it has proved that a 404 means the file
is missing on a live sandbox. Any other failure propagates. The `Episode`
rollout runtime doesn't retry provider SDK calls for you.

### Host processes

`await phase.processes.run(argv, env=..., cwd=...)` starts a supervised process on the
training machine. The process receives the environment you pass, plus a small allowlist
of non-secret variables that [agent processes](resources.md#agent-processes) describes.
A nonzero exit is returned, not raised. Cancelling the call ends the whole process
group.

By default the process runs in a directory that belongs to this attempt. The directory
survives through grading and `close()`, and the cookbook removes it during cleanup.

Your hooks run on the event loop that every episode shares, so they must yield for
other episodes to make progress. Run short blocking host code with
`asyncio.to_thread()`. Give long blocking calls an executor of your own.
`asyncio.to_thread()` uses the loop's default executor, which the Loops SDK shares, and
many long calls there stall sampling and training.

## The lifecycle

```
admit()                          backpressure and pre-start places apply
  provision the sandbox          off both clocks; no sampling slot held
  Episode(task, options)
  await episode.setup(Setup(...))          setup clock; no sampling slot; no model
hold(sampling):
  await episode.run(Sampling(...))       rollout clock; cut at agent_timeout_seconds, still graded
release
grade()                          staleness check first, then a verification slot
finish / abort → close()         once, under the cleanup deadline; then the sandbox is released
```

The cookbook first admits the attempt. Backpressure applies at admission, so under
backpressure no new sandbox starts. The pre-start places let a few admitted attempts
start their sandbox before a sampling slot frees. There are
`[rollout] prestart_allowance` × `max_concurrent_rollouts` of them, rounded up, and
[scheduling.md](scheduling.md) explains them. Creating the sandbox counts toward
neither clock and never holds a sampling slot.

The cookbook then builds the `Episode` and runs `setup()` under the setup clock, still
without a sampling slot or a model. Next, `run()` runs while the attempt holds a
sampling slot, under the rollout clock. When `agent_timeout_seconds` passes, `run()` is
cut off, and the rollout is still graded. After the sampling slot is released, the
cookbook checks that the rollout isn't stale, which means too far off policy to train,
and then waits for a verification slot to run `grade()`. When the attempt finishes or
is aborted, `close()` runs once under the cleanup deadline. Then the cookbook releases
the sandbox.

## The agent deadline

`agent_timeout_seconds` is the agent's own time limit, and it runs while the attempt
holds its sampling slot. When it passes, the cookbook cancels `run()`, ends its
processes, and seals the episode with the generations the agent received.
`grading.timed_out` is then true, and `grade()` scores what the agent left. If the
agent received no trainable generation before the cut, there is nothing to train on,
and the rollout is discarded as `unversioned`.

Once the deadline has cut `run()` off, whatever `run()` raises while it stops is not
the outcome. If `run()` catches the cancellation and returns, the cookbook keeps the
metadata it returns.

The metric `runtime/episodes/agent_timed_out` is observed once for each `run()` that
returned or was cut off. It is 1 when the deadline cut `run()` and 0 otherwise, so its
mean is the share of episodes that the deadline cut.

The rollout clock, `rollout_timeout_seconds`, is the backstop behind the agent
deadline. The two measure different time. The agent deadline is wall-clock time from
the start of `run()`, including any resource-service calls made inside it. The rollout
clock counts the time the attempt holds sampling capacity, minus resource-service time.
A `run()` that doesn't stop after the agent deadline is cancelled again when
`rollout_timeout_seconds` runs out, and that rollout is discarded as
`agent_timeout/rollout_clock`.

The `Episode` rollout runtime derives the rollout clock so that a cut `run()` that
stops within the cancellation grace is always graded. A cut `run()` returns only once
its processes have exited. A process that ignores `SIGTERM` gets up to
`process_kill_grace_seconds` before it is killed, and the killed process group is then
waited for up to 5 seconds more. So the rollout clock is
`agent_timeout_seconds + [rollout.cleanup] process_kill_grace_seconds + 5 + cancellation_grace_seconds`.
Leave `rollout_timeout_seconds` unset to use this sum. `loops check` refuses a value set
below it, and it prints the clock the run keeps.

## Declaring a sandbox

An episode can provision a sandbox for its agent, called the execution sandbox, and a
separate grading sandbox. It can have both, either or neither. You declare each
sandbox in two places:

- In the class, `execution_kind` or `grading_kind` names the kind of sandbox the episode
  is written for, such as `Blaxel` or `Daytona`. [daytona.md](daytona.md#for-an-episode)
  covers Daytona.
- In the training config, the `[rollout_runtime.config.sandbox]` or
  `[rollout_runtime.config.grading_sandbox]` table chooses the provider and the
  sandbox's spec.

`loops check` refuses a table that is missing or unused, a provider whose kind doesn't
match the class's kind, and an invalid declaration.

Each table holds `provider`, which names an entry in `[providers]`. Beside it, written
flat, are the keys of the spec of the sandbox kind that the provider provides. For
Blaxel, the kind is `Blaxel` and its spec is `BlaxelSandbox`. All three of its keys are
required:

- `image` is the image the sandbox runs.
- `memory_mb` is the sandbox's memory, as in Harbor's `memory_mb`.
- `ttl` is how long Blaxel keeps the sandbox, such as `"1h"`, as in Harbor's Blaxel
  environment. It is a backstop behind the run's own release of the
  sandbox, so it must outlast one episode. Write it as Blaxel takes it, a whole number
  and one unit: `s`, `m`, `h`, `d` or `w`.

`loops check` refuses a key that the spec doesn't declare, so a typo is refused rather
than ignored. It also refuses a value of another type, because a spec validates
strictly. For example, `memory_mb = true` is refused, not read as 1.

A provider's kind declares its spec in its `spec` class variable, which is a
`SandboxSpec`. The [model rules](#the-model-rules) hold for the spec. The validated
spec's dump, defaults included, is exactly what each sandbox is asked for. So the run
records its digest as `sandbox` in `runtime_evidence_digests.json`, as it does for the
options.

The `[rollout_runtime.config.grading_sandbox]` table follows the same provider and
spec rules as `sandbox`, but it binds `grading_kind`. The cookbook provisions the
grading sandbox only when a per-attempt `grade()` is admitted.

## Grading saved artifacts without retaining a sandbox

`grade()` runs on the training machine, with its own verification capacity. Unless the
class declares a separate grading sandbox, grading provisions no sandbox. By default,
the execution sandbox stays available to `grade()` and `close()`. An external judge or
an evaluator that reads saved artifacts on the host doesn't need it, so the episode can
release the sandbox as soon as `run()` stops:

```python
class ArtifactEpisode(Episode[TaskModel, OptionsModel]):
    execution_kind = Blaxel
    sandbox_lifetime = "sampling"

    async def run(self, sample: Sampling):
        try:
            await self.agent(sample)
        finally:
            # Also collect partial output when the agent deadline cuts run().
            # Use your provider's client to download; persist what your judge reads.
            self.artifacts = await self.collect_artifacts(sample)

    async def grade(self, grading: Grading):
        return await self.external_judge(self.artifacts)
```

Collect everything the evaluator needs before `run()` returns. That includes the case
where the agent deadline cuts `run()` off, because that rollout is still graded. The
collection in `finally` must finish within the rollout clock and the cancellation
grace, and the cookbook doesn't guarantee time for arbitrary work after a timeout.
Initialize the grading state in `__init__` for attempts that produce no output. The
attempt's local process directory survives through grading and `close()`, but the
cookbook deletes it afterward. Keep evidence that must outlive the attempt in your own
durable artifact store.

With the `"sampling"` lifetime, the rollout runtime does three things before the
rollout enters the verification queue. It releases the sampling capacity, closes the
attached client and asks the resource registry to release the execution sandbox. Grading retries neither keep nor recreate the execution sandbox.
The `Grading` and `Closing` records then have `execution_sandbox = None`. A sandbox
client or liveness handle saved earlier fails with a clear error after the release.
Don't reuse any client or environment that your episode code saved.

If the release is unconfirmed or deferred, the sampling attempt fails with
`CleanupIncomplete`. The cookbook doesn't pretend that the sandbox is gone and start
the judge. Normal attempt cleanup retries the unresolved resources, and the resource
journal keeps the evidence needed to recover them. A release that is cancelled
continues under the registry's cancellation protection. No sandbox is released while an
agent hook that ignored cancellation is still running. Keep the default `"attempt"`
lifetime when your evaluator needs the sandbox.

The run records the effective lifetime in its runtime evidence, so a resume
compares it. A checkpoint whose runtime evidence has no `sandbox_lifetime`
entry logs the changed-settings warning on its first resume, even with the default
lifetime. The resume isn't refused. A checkpoint with no runtime evidence at all has
nothing to compare, so the resume doesn't report its runtime evidence as changed.

## A separate grading sandbox

An episode can provision an execution sandbox, a grading sandbox, both or neither.
Declare each sandbox on its own:

```python
class CheckedEpisode(Episode[TaskModel, OptionsModel]):
    execution_kind = Blaxel
    grading_kind = Blaxel
    sandbox_lifetime = "sampling"  # optional: grade does not need the execution sandbox
    grading_retries = 1            # only when rerunning the verifier is safe

    def __init__(self, task, options):
        super().__init__(task, options)
        self.artifacts = {}

    async def run(self, sample: Sampling):
        try:
            await self.agent(sample)
        finally:
            # Partial-output collection must respect the timeout bounds above.
            self.artifacts = await self.collect_outputs(sample)

    async def grade(self, grading: Grading):
        sandbox = grading.grading_sandbox
        assert isinstance(sandbox, Blaxel)
        verifier = await sandbox.client()
        await self.upload_outputs(verifier, self.artifacts)
        return await self.check_outputs(verifier)
```

```toml
[rollout_runtime.config.sandbox]
provider = "blaxel"
image = "agent-image"
memory_mb = 4096
ttl = "1h"

[rollout_runtime.config.grading_sandbox]
provider = "blaxel"
image = "verifier-image"
memory_mb = 2048
ttl = "30m"
```

The two tables can name different providers, and each kind and spec is validated on
its own. Choose the declarations for where you grade:

- To grade only in a grading sandbox, declare `grading_kind`, and leave out
  `execution_kind` and the `sandbox` table.
- To grade on the host or remotely, leave out `grading_kind` and the `grading_sandbox`
  table.
- To grade in the execution sandbox, declare only `execution_kind`, keep the default
  `"attempt"` lifetime, and use `grading.execution_sandbox`.

The cookbook creates the grading sandbox after the rollout acquires a verification
slot, immediately before the first `grade()` call. Sampling and waiting for
verification don't create one, and a discarded rollout that never reaches grading
creates none. Provisioning the sandbox and attaching its client count toward the
grading try's wall-time deadline, together with the verifier itself, so size
`[rollout.grading] timeout_seconds` to cover them. The provider's own deadlines still
apply. A provisioning failure fails that grading try, and the episode's grading retries
can try again. It never becomes a zero reward.

Once the cookbook has created a grading sandbox and attached its client, it reuses both
across `grade()` retries for the same attempt. If your verifier changes the sandbox's
state, restore that state before a retry. The cookbook doesn't create a fresh sandbox
for each try and doesn't transfer artifacts for you. `grade()` uploads or reads whatever
it needs from the outputs that `run()` saved. A fresh sampling attempt gets fresh
sandboxes.

`sandbox_lifetime` affects only the execution sandbox. The grading sandbox stays
available through `close()`. Normal attempt cleanup closes both clients and releases
both registered resources, including one whose provisioning failed. An attempt whose
grading hook ignores cancellation keeps its resources, under the rule for
[a hook that ignores cancellation](failures.md#what-the-cookbook-decides-itself).

`loops check` validates both tables without provisioning anything. It refuses a
provider whose kind doesn't match and a table that is missing or unused. It records the
effective grading spec, defaults included, as `grading_sandbox` in the
runtime evidence. Without a grading table, it records no such key. A grading sandbox
requires a per-attempt `grade()`. An episode that defines only `grade_group()` receives
recorded member evidence, not an attempt or its sandboxes, so it can't have a grading
sandbox.

## Typed tasks and options

An episode names its task model and its options model once, as its two type arguments:

```python
import os

from pydantic import BaseModel, ConfigDict

from loops_cookbook.episodes import Episode, Grading, Sampling


class MathTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    answer: str


class JudgeOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    model: str = "gpt-4.1-mini"
    api_key_env: str = "JUDGE_API_KEY"   # the name of the variable, not the key


class JudgedEpisode(Episode[MathTask, JudgeOptions]):
    grading_retries = 2

    async def grade(self, grading: Grading):
        task, options = self.task.value, self.options   # a MathTask, a JudgeOptions
        key = os.environ[options.api_key_env]
        ...
```

```toml
[rollout_runtime.config.options]
url = "https://judge.example.com/v1"
```

`Episode[None, JudgeOptions]` takes options and reads its tasks as raw data, and
`class MyEpisode(Episode[None, None])` declares neither. You can use a generic base
class of your own, as long as the class you name as `episode` binds every type
parameter. A parameter that no class binds takes its default, and `loops check`
refuses a class that leaves a parameter without a default unbound.

### Tasks

When the rollout runtime is built, every line of the `tasks` file in
`[rollout_runtime.files]` is validated with the task model, except the line's `id`.
`loops check` refuses a line that the model rejects, and names the file and the line.

`task.data` is then the validated model's JSON dump with the line's `id` kept, not the
raw line. The dump comes from the model's schema, so a `model_dump` that the model
overrides isn't used. Each rollout's `task_digest` is taken from this dump. Defaults
that the model fills in are part of what ran, so the record covers them. Changing a
default in a task model therefore changes the recorded digests. A resume still keeps
its place in the task order. Only a changed task list or `shuffle_seed` restarts it.

A line's `id` names its task and doesn't belong to the model, so `check` refuses a task
model with an `id` field. Read the ID as `task.id`.

### Options

When the rollout runtime is built, `[rollout_runtime.config.options]` is
validated with the options model. `check` refuses options that the model rejects. It
also refuses an options table for an episode that has no options model.

The validated options' JSON dump, defaults included, is what ran. So the run records
its digest, never its values, as `options` in `runtime_evidence_digests.json` and in
each checkpoint's contract.

### The model rules

One allow list holds for task models and options models alike. It also holds for the
spec of the provided sandbox kind, and for every model that their fields reach. The
list keeps two promises:

- **Every key is read.** A model that ignores unknown keys, which is pydantic's default,
  would drop a misspelled `anwser` and fill in its default without anyone noticing.
- **The dump is what was validated.** A task's digest is the digest of its dump, and the
  options' digest is the digest of theirs.

A model is allowed when all of these hold:

- Its `model_config` sets `extra = "forbid"`.
- It has no computed field.
- Its core schema sets no serialization of its own, in any model or type it reaches.
  That rules out a `field_serializer`, a `model_serializer`, a serializer in
  `Annotated`, and a config's `json_encoders`.
- Each field is read and dumped by its own name, so it has no `alias`,
  `validation_alias` or `serialization_alias`.
- No field is ever excluded with `exclude` or `exclude_if`.
- Each field's type is one of the allowed types below.

These are the allowed types:

- a JSON scalar: `str`, `int`, `float`, `bool` or `None`
- a `Literal` of strings, ints, bools or `None`
- an `Enum` whose values are all JSON scalars
- a `list[...]` or a `dict[str, ...]` of allowed types
- a union of allowed types, such as `X | None`
- a type alias of an allowed type, including a recursive one such as the cookbook's
  `JsonValue`
- pydantic's `JsonValue`
- another allowed model

`Annotated` may add constraints, validators and `Field()` to an allowed type, but never
a serializer.

`check` refuses anything else and names the model and the field. Here are some types
it refuses, and why:

- An untyped `dict`, `list` or `Any` is refused, because the keys inside it would go
  unchecked.
- A `SecretStr` or `Secret` is refused, because it dumps as asterisks, so two different
  values would share a digest.
- A dataclass, `TypedDict` or `NamedTuple` is refused. Make it a `BaseModel` instead.
- A `tuple`, a `set` and a `datetime` are refused.

An `Enum` dumps as its value, so it is allowed when that value is a JSON scalar. When a
task or the options are loaded, the cookbook also refuses a validator that returned a
value its field's type doesn't allow, because the dump would then not be what the types
say.

Keep secrets out of options. A secret comes from the environment or `.env`. An option
may name the variable that holds the secret, and the episode reads that variable. A
default that an options model fills in, from the environment or not, changes the
options digest, but its value never reaches disk.

### Resuming

The `runtime_config_digest` covers `[rollout_runtime.config]` as written. The
contract's `runtime_evidence_digests` cover the validated options and sandbox spec.
When you resume with changed options, the run logs "Resuming with changed settings
(runtime_config_digest, runtime_evidence_digests)". When only a default moved, it names
only `runtime_evidence_digests`. Either way, the run continues from the checkpoint. The
resumed run is then no longer comparable with the one that produced the checkpoint,
because a different judge, for example, gives different rewards. A checkpoint with no
runtime evidence has nothing to compare, so it is not reported as changed.

## Group grading

Some rewards exist only across a group. Examples are an LLM judge that ranks eight
answers, pairwise comparisons, a diversity bonus and a self-play game. An `Episode`
that defines the classmethod `grade_group()` scores its whole group at once, after
every member is sealed and graded:

```python
class RankedEpisode(Episode[Question, JudgeOptions]):
    grading_retries = 2                          # asking the judge again is safe

    async def run(self, sample: Sampling):
        ...

    @classmethod
    async def grade_group(cls, members, options):   # options: a JudgeOptions
        places = await my_judge(options, members)   # its own client, never the policy
        worst = max(1, len(members) - 1)             # a group can be a single member
        # places[i] is member i's place, 0 the best: one reward per member, in member order
        return [1 - place / worst for place in places]
```

[examples/llm_judge](../../examples/llm_judge/) is the complete example.

**The capability.** A class that defines `grade_group()` makes its rollout runtime
declare `pending_group_grade`, and `loops check` reports `group grading true`. The
class's `grade()` may then be left out, or it may return `None`, because the group
grader gives the reward. Without a group grader, `grade()` must give a reward, and a
`None` fails as a grader bug. A rollout runtime that you write yourself declares
`pending_group_grade` in its capabilities and implements
`GroupGrader.grade_group(members)`. That method receives each member's sealed rollout
and transcript as `MemberEvidence`.

**What it sees.** Each `GroupMember` is plain data, never a sandbox or a live episode,
so no attempt's cleanup waits for its group. A member has these fields:

- `task` is the group's task, as a private copy. `task.value` is the task as the task
  model validated it.
- `transcript` is the conversation at the model boundary, as a `SealedTranscript`. Each
  turn is one generation. A turn holds the request as the agent sent it, the tools, and
  the reply as the agent received it. Each turn is marked with whether it was delivered
  and whether it trains. A turn whose request or reply isn't JSON has `unrecordable`,
  which says why, in place of what the run couldn't record. A request that no
  generation answered has no `sample_id` and no reply. Transcripts are recorded
  whenever a group grader exists, whatever `[output] trace` says, but only
  `trace = "transcripts"` or `"full"` writes them to disk.
- `reward` is what `grade()` gave, or `None`.
- `metadata` is what `run()` returned, together with `grade()`'s metadata.

`options` is `[rollout_runtime.config.options]` as the episode's options model
validated it at `check`. It is the same value that `self.options` holds, as a private
copy for each call, and it is `None` when the episode has no options model.
`grade_group()` never gets the policy. A judge brings its own client, configured in the
options. Options hold no secrets, so a judge's options name the environment variable
that holds its key.

**The order.** A complete group goes through one more stage before it can be selected
for training:

```
members sealed -> grade() per rollout (optional) -> group complete
  -> drop stale members -> size >= group_size - max_group_losses? -> grade_group()
     (before every try of grade_group(), retries included)
  -> no-variance check -> ready
```

The stale members are dropped and the group's size is checked before every try of
`grade_group()`, so the judge is never paid to score a group that is about to be
dropped. That holds on a retry too. A member that goes stale during the backoff is
dropped before the next try, and a group that it leaves too small isn't scored again.

- Stale members are discarded as `stale_before_training`.
- A group left too small is rejected as `group_too_small`.
- A group whose final rewards are all equal is rejected as `no_variance`.

Every group rejected as unusable, here or at selection, is counted, recorded and added
to the streak of unusable groups in the same way. So a group shrunk by infrastructure
failures still counts toward `max_consecutive_unusable_groups`. Selection still
rechecks staleness, as it does for every group.

**Rewards.** `grade_group()` returns one reward for each member, in order, and these
rewards are final. They replace whatever `grade()` gave, so if you want both, combine
them inside `grade_group()`. The stage makes new rollouts with the rewards, and the
evidence that the grader read never changes. There are no placeholder rewards. A member
discarded before its group was scored is recorded with `reward: null`.

**Failures, capacity and deadlines.** Each call holds one of the run's
`max_concurrent_verifiers` verification slots, the same slots that `grade()` uses, and
runs under `[rollout.grading] timeout_seconds`. A failed call is retried up to the
class's `grading_retries`, with the same backoff that `grade()` retries use. Each retry
is counted as `retried/groups/grading_failed/<type>`. When the class has no `grade()`,
a rollout's grading takes no verification slot at all. `check` then reports that the
rollout runtime acquires verification itself, though it never needs to.

A result that isn't one finite number for each member is a grader bug, such as the
wrong count, a NaN or a bool. Like `PermanentGradingFailure`, it is never retried. When
the grader fails a group, the group's members are discarded as
`grading_failed/<type>` with `reward: null`. The failure counts toward the
[`group_grading` stop](failures.md#the-systemic-stop). A failed group is a grading
failure, not an unusable group.

A `grade_group()` call that is still running `cancellation_grace_seconds` after its
deadline cancelled it always stops the run, and the run doesn't end cleanly. Like a
`grade()` that ignores cancellation, the call keeps its verification slot until it
ends. A kept `grade()` is different. It stops the run only when more kept hooks are
running than 5% of `max_concurrent_rollouts`, rounded up, or when kept hooks hold every
slot of a phase or every start admission place.
[failures.md](failures.md#what-the-cookbook-decides-itself) explains kept hooks.

**The async caveat.** In async mode, a member can still go stale between the judge and
selection. Selection drops it, and its siblings keep the rewards that the judge gave
them while it was present. This can't happen in sync mode, which is the default.

## Where to see it

- `steps.jsonl` counts discarded rollouts as `discard/rollouts/<reason>/<type>` and
  retried attempts as `retried/attempts/<reason>/<type>`. An example is
  `discard/rollouts/agent_timeout/rollout_clock`.
- Each record in `rollouts.jsonl` carries `reason`, `cause` and `retries`. `cause` is
  `<reason>`, or `<reason>/<type>` for a failure. It is `cleanup_failed/<type>` when
  the attempt's `abort()` or `finish()` failed and nothing before it did.
- The episode metadata of the rollouts that trained is summarized as `rollout/*`
  metrics. So a label such as `agent_exit`, from the
  [local pattern](../../patterns/custom_runtime/local/), is counted on
  each step, up to `[output] rollout_metrics_limit` metrics. Each numeric
  `rollout/<key>` is also reported as `rollout/<key>_with_unusable`, which also counts
  the members of the groups dropped as unusable since the previous update.
- For a group grader, `discard/groups/grading_failed` counts the groups it failed, and
  `retried/groups/grading_failed/<type>` counts its retries.
  `pipeline_calls/grade_group` counts its calls in flight, and
  `queue/group_grading_groups` counts the groups waiting for a call.
  `time/pipeline/group_grading/*` measures how long each call took, from its
  verification slot to its end.
