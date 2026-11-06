# The Episode API

An `Episode` is what you write for a custom environment: one object per attempt. The
cookbook loads the tasks, holds the attempt's capacity, provisions its sandbox, enforces
the agent deadline, and calls `close()` once. It is an ordinary runtime
underneath, so everything in [failures.md](failures.md) holds for it.

```toml
[runtime]
factory = "loops_cookbook.episodes:create_runtime"   # model access: the class's model_access

[runtime.config]
episode = "my_env:AnswerEpisode"   # module:Class, importable beside this file
agent_timeout_seconds = 600        # required

[runtime.files]
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

## What you write

| | When it runs | Notes |
| --- | --- | --- |
| `__init__(task, options)` | once per attempt, after execution resources are provisioned, if declared | Initialise everything `grade()` reads. |
| `async setup(setup: Setup)` | before the agent starts; optional | On the setup clock, holding no sampling slot, with no model in scope. |
| `async run(sample: Sampling)` | inside the attempt's sampling hold | The agent. Return episode metadata (a mapping) or `None`; anything else fails the attempt, not retried. Raise only when the episode didn't happen. |
| `async grade(grading: Grading)` | once the rollout is sealed, in a verification slot | Return a reward, or a `GradeResult` with metadata. The policy is not available: an LLM judge brings its own client. With `grade_group()` it is optional, and may return `None`. |
| `grade_group(members, options)` | an async classmethod, once per complete group; optional | Return one final reward per member, in order. It sees recorded evidence only, and `options` is the episode's validated options. See [group grading](#group-grading). |
| `async close(closing: Closing)` | once, when the attempt ends, however it ended; optional | Under the cleanup deadline. It also runs after a `setup()` or `run()` that raised or was cut. It does not run when `__init__` raised or never ran, since there is no episode. For an attempt kept live because its `setup()`, `run()` or `grade()` ignored cancellation, it runs once that hook ends. |
| `model_access` | class variable, default `ModelAccessMode.DIRECT` | `OPENAI_HTTP` for an agent that speaks Chat Completions. |
| `sandbox_lifetime` | class variable, default `"attempt"` | `"sampling"` closes the attached client and releases the sandbox after `run()` stops, before verification admission. `grade()` and `close()` cannot use the execution sandbox afterward. No grading sandbox is created unless `grading_sandbox` is separately declared. |
| `grading_retries` | class variable, default `0` | How many more times a failed `grade()` or `grade_group()` is tried. Nonzero only if grading is safe to run again, as an LLM judge is. |
| `Episode[TaskModel, OptionsModel]` | two explicit type arguments | Pydantic models for task data and options; use `None` when unused. |
| `execution_kind`, `grading_kind` | class variables, each default `None` | Independently declare sandbox requirements, such as `Blaxel` or `Daytona` ([daytona.md](daytona.md#for-an-episode)). The TOML tables choose providers and specs. `check` refuses missing or unused tables, provider/kind mismatches, and invalid declarations. |
| `[runtime.config.sandbox]` | beside `provider`, the keys of the provided kind's spec, flat | For `Blaxel` (`BlaxelSandbox`), each required: `image`; `memory_mb`, as Harbor's; `ttl` as in Harbor's Blaxel environment, such as `"1h"`: a backstop behind the run's own release, so it must outlast one episode; a whole number and one unit (`s`, `m`, `h`, `d`, `w`) as Blaxel takes it. `loops check` refuses a key the spec does not declare, so a typo is refused, not ignored, and a value of another type, since a spec validates strictly (`memory_mb = true` is refused, not read as 1). A provider's kind declares its spec as `spec`, a `SandboxSpec`, which the [model rules](#typed-tasks-and-options) hold for. The validated spec's dump, defaults included, is what each sandbox is asked for, so the run records its digest as `sandbox` in `runtime_evidence_digests.json`, as it does the options'. |

`[runtime.config.grading_sandbox]` uses the same provider/spec rules as `sandbox`,
but binds `grading_kind` and is provisioned only when per-attempt `grade()` is admitted.

A retry is a fresh attempt: a new `Episode`, a new sandbox, and the agent runs again.
`loops check` refuses a class that does not define `run()`, one that defines
neither `grade()` nor `grade_group()`, and one whose `setup()`, `run()`, `grade()` or
`close()` is not an `async def`. It refuses a `grade_group()` that is not an async
classmethod.

## What you are given

- **`task`**: `task.id` and `task.data`, one line of `[runtime.files] tasks`. The file
  is JSONL, one object per task. A line's `id` names its task; a line without one is
  named by its position, counting from 0. Each task is digested by its canonical JSON,
  and each attempt gets its own copy of the data. `task.value` is the line as the
  episode's task model validated it, or `None` without one.
- **`sample.model`**, only in `run()`: `sample.model.llm(producer=None)` for direct access, or
  `await sample.model.http_connection(producer=None)` for an OpenAI-compatible endpoint and
  credential. Once `run()` has ended, both refuse with "model access for this attempt
  is sealed". A client taken from `llm()` refuses once the attempt is sealed, and an
  HTTP credential is revoked then, so a model kept on `self` and used later fails
  loudly instead of sampling off the record. A task `run()` spawned that keeps
  generating after `run()` returns and before the seal is yours to stop: the cookbook
  does not police it, but any generation after the seal fails.
- **`self.options`**: validated options, a private copy per attempt, or `None`.
- **Phase records** are immutable, keyword-only dataclasses. Every phase includes
  `attempt_id`, `metrics`, and `processes`. The fixed hooks receive these records
  directly; annotations do not configure injection.

  | Phase | Additional fields |
  | --- | --- |
  | `Setup` | `sandbox`: execution sandbox or `None`; no model |
  | `Sampling` | `model`, `sandbox`: execution sandbox or `None`, `trajectory_tokens` |
  | `Grading` | `rollout`, `timed_out`, `execution_sandbox`, `grading_sandbox` |
  | `Closing` | `timed_out`, `execution_sandbox`, `grading_sandbox` |

  Grading and closing expose only resources still accessible. An execution sandbox
  released after sampling is `None`; a verifier never successfully initialized is
  `None`. A phase snapshot does not prevent retaining an SDK client beyond its
  lifetime; do not reuse saved clients or environments after release.
- **`sample.trajectory_tokens`**: the run's trajectory budget, `[sampling]
  max_trajectory_tokens`, else `[model] max_seq_len`, or `None` when neither is set.
  An episode that shapes its reward near the context limit reads it here instead of
  restating it in its options. How a generation is fitted to it is in
  [model access](model-access.md).
- **Convenience methods**: `sample.model.conversation(producer=..., conversation_id=...)`
  opens a direct session with the same binding and lifetime as `llm().conversation()`.
  `setup.require_sandbox(Kind)` and `sample.require_sandbox(Kind)` return a typed
  sandbox. Grading and closing provide `require_execution_sandbox(Kind)` and
  `require_grading_sandbox(Kind)`. Missing resources raise `RuntimeError`; an
  incompatible kind raises `TypeError`. Optional sandbox fields remain available
  for cleanup after partial setup. Blaxel and Daytona also provide
  `await sandbox.write_file(path, content)` for UTF-8 text through the managed client.
- **Sandbox access**: `SandboxKind` supplies `name`, `environment`, `client()`,
  `alive()` and `ensure_alive()`. Client/file/command operations depend on the
  provider. Narrow explicitly before using its SDK:

```python
from loops_cookbook.episodes import Sampling
from loops_cookbook.providers.blaxel import Blaxel


async def prepare(sample: Sampling):
    sandbox = sample.sandbox
    assert isinstance(sandbox, Blaxel)
    box = await sandbox.client()  # blaxel.core.sandbox.SandboxInstance
```

  The first client call attaches under the provider's attach deadline; later calls
  reuse that client. The cookbook closes it after `close()` or at sampling release.
  `ensure_alive()` raises `SandboxGone` if the sandbox is lost. A missing answer is
  a zero reward only after checking liveness. Blaxel's `read_file(path)` returns text,
  or `None` after proving a 404 means a missing file on a live sandbox; other failures
  propagate. Provider SDK calls are not implicitly retried by the Episode runtime.
- **`await phase.processes.run(argv, env=..., cwd=...)`** starts a supervised host
  process. It receives the supplied environment plus the existing small non-secret
  allowlist. Nonzero exit is returned; cancellation ends the process group. Its default
  directory belongs to this attempt, survives through grading and closing, and is
  removed during cleanup. Hooks run on the shared event loop and must yield for other
  episodes to progress. Use `asyncio.to_thread()` for short blocking host code. Give
  long blocking calls an executor of your own: `asyncio.to_thread()` uses the loop's
  default executor, which the Loops SDK shares, and many long calls there stall
  sampling and training.

## Typed tasks and options

An episode names its task model and its options model as its type arguments, once:

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
[runtime.config.options]
url = "https://judge.example.com/v1"
```

`Episode[None, JudgeOptions]` takes options and reads its tasks as raw data, and a
`class MyEpisode(Episode[None, None])` declares neither. A generic base of your own is
fine as long as the class you name as `episode` binds every type parameter; a
parameter no class binds is its default, and `loops check` refuses one that leaves
a parameter with no default open.

- **Tasks.** Every line of `[runtime.files] tasks` but its `id` is validated with the
  task model when the runtime is built, and `loops check` refuses a line it rejects,
  naming the file and line. `task.data` is then the validated model's JSON dump (its
  schema's, so a `model_dump` the model overrides is not it) with the line's `id`
  kept, not the raw line, and the task's digest is taken from it: defaults the model
  fills in are part of what ran, so the run record covers them. Changing a task
  model's default therefore changes the task digests, and a resume then starts the
  task order from the beginning, as it does for any change to the task set. A line's
  `id` names its task and is not the model's, so `check` refuses a task model with an
  `id` field; read it as `task.id`.
- **Options.** `[runtime.config.options]` is validated with the options model when the
  runtime is built. `check` refuses options it rejects, and refuses an options table
  from an episode with no options model. The validated options' JSON dump, defaults
  included, is what ran, so the run records its digest, never its values, as
  `options` in `runtime_evidence_digests.json` and in each checkpoint's contract.
- **The model rules.** One allow list holds for task and options models alike, for
  the provided sandbox kind's spec, and for every model their fields reach. It keeps two promises: every key is read (a
  model that ignores unknown keys, pydantic's default, would drop a misspelt `anwser`
  and fill in its default, unnoticed), and the dump is what was validated (a task's
  digest is its dump's, and the options' digest is theirs). A model is allowed when:
  - its `model_config` sets `extra = "forbid"`;
  - it has no computed field, and its core schema sets no serialization of its own
    (a `field_serializer`, a `model_serializer`, a serializer in `Annotated`, or a
    config's `json_encoders`), in any model or type it reaches;
  - each field is read and dumped by its own name (no `alias`, `validation_alias`
    or `serialization_alias`) and is never excluded (`exclude`, `exclude_if`);
  - each field's type is a JSON scalar (`str`, `int`, `float`, `bool`, `None`), a
    `Literal` of strings, ints, bools or `None`, an `Enum` whose values are all JSON scalars, a `list[...]` or
    `dict[str, ...]` of allowed types, a union (`X | None`) of allowed types, a type
    alias of an allowed type (a recursive one included, such as the cookbook's
    `JsonValue`), pydantic's `JsonValue`, or another allowed model; `Annotated` may
    add constraints, validators and `Field()`, never a serializer.

  `check` refuses anything else, naming the model and the field: an untyped `dict`,
  `list` or `Any` (the keys within would go unchecked), a `SecretStr` or `Secret`
  (it dumps as asterisks, so two different values would share a digest), a dataclass,
  `TypedDict` or `NamedTuple` (make it a `BaseModel`), a `tuple` or `set`, a
  `datetime`. An `Enum` dumps as its value, so it is allowed when that value is a
  JSON scalar. When a task or the options are loaded, a validator that returned a
  value its field's type is not (so the dump would not be what the types say) is
  refused too. A secret comes from the environment or `.env`; an option may name the
  variable that holds it, and the episode reads it. A default an options model fills
  in, from the environment or not, changes the options digest and never reaches disk.
- **Resuming.** `[runtime.config]` as written is covered by the
  `runtime_config_digest`, and the validated options and sandbox spec by the
  contract's `runtime_evidence_digests`. Resuming with changed options logs "Resuming with
  changed settings (runtime_config_digest, runtime_evidence_digests)", or only
  `runtime_evidence_digests` when just a default moved, and continues from the
  checkpoint, and that run is no longer comparable with the one that produced it: a
  different judge gives different rewards. A checkpoint recorded before runs kept
  runtime evidence has none to compare, and is not reported as changed.

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

A sandbox can be up before a sampling slot frees, within the pre-start places, but its
create never holds a slot.

## Grading saved artifacts without retaining a sandbox

`grade()` already runs on the host, with its own verification capacity. Without
a separate `grading_sandbox` declaration, no grading sandbox is provisioned. By default the sampling sandbox remains available to
`grade()` and `close()`. For an external judge or a host-side artifact evaluator:

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

Collect everything needed by the evaluator before `run()` returns, including on
an agent timeout that is still graded. Collection in `finally` must stop within the
rollout backstop and cancellation grace; arbitrary post-timeout work is not
guaranteed. Initialize the grading state in `__init__`
for attempts with no output. The attempt's local process directory survives through
grading and `close()`, but is then deleted; evidence that must survive the attempt
belongs in your durable artifact store.

The runtime releases sampling capacity, closes the attached client, and asks the
resource registry to release the sampling sandbox **before the rollout enters the
verification queue**. Grading retries neither retain nor recreate the execution sandbox. Without a separate
`grading_sandbox` declaration, grading creates no sandbox. Later grading and closing records contain `execution_sandbox = None`. Saved sandbox
client and liveness handles fail clearly after release;
any client or environment already saved by episode code must not be reused.

An unconfirmed or deferred release fails the sampling attempt with
`CleanupIncomplete`, rather than pretending the sandbox is gone and starting the
judge. Normal attempt cleanup retries unresolved resources, and the resource
journal retains recovery evidence. A cancelled release continues under the
registry's existing cancellation protection. No sandbox is released while an
agent hook that ignored cancellation remains live. Episodes whose evaluator needs
the sandbox should keep the default `"attempt"` lifetime. The effective lifetime is
recorded in runtime evidence for resume comparisons. A checkpoint that already
records runtime evidence but predates this field produces the existing changed-
settings warning on its first resume, even with the default lifetime; it is not
refused. Older checkpoints without any runtime evidence retain the existing
legacy comparison behavior.

## A separate grading sandbox

The Episode API can provision an execution sandbox, a grading sandbox, both, or
neither. Declare each sandbox requirement independently:

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
[runtime.config.sandbox]
provider = "blaxel"
image = "agent-image"
memory_mb = 4096
ttl = "1h"

[runtime.config.grading_sandbox]
provider = "blaxel"
image = "verifier-image"
memory_mb = 2048
ttl = "30m"
```

The tables can name different providers, with independently validated kinds/specs.
A grading-only environment declares `grading_kind` and omits `execution_kind` and
`sandbox`. Host or remote grading omits `grading_kind` and `grading_sandbox`.
To grade in the execution sandbox, declare only `execution_kind`, keep the default
attempt lifetime, and use `grading.execution_sandbox`.

The grading sandbox is created **after a verification slot is acquired**, immediately
before the first `grade()` call. Sampling and waiting for verification do not create
one. A discarded rollout that never reaches grading creates none. Provisioning and
attachment are part of the grading try's wall-time deadline, alongside the
verifier: size `[rollout.grading] timeout_seconds` accordingly. Provider deadlines
still apply. Provisioning failure fails that grading try and is eligible for the
Episode's existing grading retries; it does not become a zero reward.

A successfully created grading sandbox and attached client are reused across
`grade()` retries for the same attempt; the Episode owns restoring any state its
verifier mutates before a retry. There is no implicit fresh sandbox per try or
automatic artifact transfer. `grade()` explicitly uploads/reads whatever it needs
from the outputs `run()` saved. A fresh sampling attempt gets fresh environments.
`sandbox_lifetime` affects only the execution sandbox; the separate grading sandbox
remains available through `close()`, and normal attempt cleanup closes both clients
and releases their registered resources, including failed provisioning. An attempt
whose grading hook ignores cancellation retains its resources under the existing
nonquiescent-hook safety rule.

`loops check` validates both tables without provisioning, rejects provider/kind
mismatches and missing or unused tables, and records the effective grading spec
(defaults included) as `grading_sandbox` in runtime evidence. With no grading table,
no new evidence key is recorded. Configuring this sandbox requires a per-attempt
`grade()` override: a `grade_group()`-only Episode receives recorded member evidence,
not an attempt or its environments.

## The agent deadline

`agent_timeout_seconds` is the agent's own time, inside the sampling hold. When it
expires, `run()` is cancelled, its processes are ended, and the episode is sealed with
the generations it was delivered. `grading.timed_out` is then true, and `grade()` scores what
the agent left (a cut with no delivered trainable generation is `unversioned`). Once the deadline has cut `run()`, what `run()` raises while it stops is
not the outcome; if it catches the cancellation and returns, its metadata is kept.
`runtime/episodes/agent_timed_out` is observed once per `run()` that returned or was
cut, 1 when it was cut, so
its mean is the share of episodes the deadline cut.

The rollout clock stays the backstop behind it. The two count different time: the
agent deadline is wall-clock time from the start of `run()`, resource-service calls
made inside it included, while the rollout clock counts the time the attempt holds
sampling capacity, less resource-service time. A `run()` that does not stop is
cancelled again when `rollout_timeout_seconds` runs out, and that rollout is discarded
as `agent_timeout/rollout_clock`. A cut `run()` returns only once its processes have
exited, which takes up to `process_kill_grace_seconds` for one that ignores `SIGTERM`.
A process group killed after that grace is then waited for, up to 5 s more. So that a
`run()` which then stops within the cancellation grace is always graded, the `Episode`
runtime derives its rollout clock as `agent_timeout_seconds + [rollout.cleanup]
process_kill_grace_seconds + 5 + cancellation_grace_seconds`. Leave
`rollout_timeout_seconds` unset to use it; `loops check` refuses one set below it, and
prints the clock the run keeps.

## Group grading

Some rewards exist only across a group: an LLM judge ranking eight answers, pairwise
comparisons, a diversity bonus, a self-play game. An `Episode` that defines the
classmethod `grade_group()` scores its whole group at once, once every member is
sealed and graded:

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

[examples/llm_judge](../../examples/llm_judge/) is the whole example.

**The capability.** A class with `grade_group()` makes its runtime declare
`pending_group_grade`, and `loops check` reports `group grading true`. Its `grade()`
may then be left out, or return `None`: the group grader gives the reward. Without a
group grader, `grade()` must give a reward, and `None` fails as a grader bug. A
hand-written runtime declares `pending_group_grade` in its capabilities and implements
`GroupGrader.grade_group(members)`, which receives each member's sealed rollout and
transcript as `MemberEvidence`.

**What it sees.** Each `GroupMember` is plain data, never a sandbox or a live episode,
so no attempt's cleanup waits for its group:

- `task`: the group's task, a private copy; `task.value` as the task model validated it.
- `transcript`: the conversation at the model boundary, as a `SealedTranscript`. Each
  turn is one generation: the request as the agent sent it, the tools, and the reply
  as the agent received it, marked with whether it was delivered and trains. A turn
  whose request or reply is not JSON has `unrecordable` saying why, in place of what
  it could not record; a request no generation answered has no `sample_id` or reply.
  Transcripts are recorded whenever a group grader exists, whatever `[output] trace`
  says; only `trace = "transcripts"` or `"full"` writes them to disk.
- `reward`: what `grade()` gave, or `None`.
- `metadata`: what `run()` returned, with `grade()`'s metadata.

`options` is `[runtime.config.options]` as the episode's options model validated it at
`check`, the same value `self.options` holds, a private copy per call (`None` without an
options model). `grade_group()` never gets the policy: a judge brings its own client,
configured there. Options hold no secrets, so a judge's options name the environment
variable that holds its key.

**The order.** A complete group goes through one stage before it can be selected:

```
members sealed -> grade() per rollout (optional) -> group complete
  -> drop stale members -> size >= group_size - max_group_losses? -> grade_group()
     (before every try of grade_group(), retries included)
  -> no-variance check -> ready
```

So the judge is never paid to score a group about to be dropped, including on a retry:
a member that goes stale during the backoff is dropped before the next try, and a group
it leaves too small is not scored again. Stale members are
discarded as `stale_before_training`, a group left too small is rejected as
`group_too_small`, and one whose final rewards are all equal as `no_variance`. Every
group rejected as unusable, here or at selection, is counted, recorded and streaked the
same way: a group shrunk by infrastructure failures still counts toward
`max_consecutive_unusable_groups`. Selection still rechecks staleness, as it does for
every group.

**Rewards.** One reward per member, in order, and final: they replace whatever `grade()`
gave, so compose the two inside `grade_group()` if you want both. The stage makes new
rollouts with them; the evidence the grader read is never changed. There are no
placeholder rewards: a member discarded before its group was scored is recorded with
`reward: null`.

**Failures, capacity and deadlines.** Each call holds one of the run's
`max_concurrent_verifiers` verification slots, the same slots `grade()` uses, under
`[rollout.grading] timeout_seconds`. It is retried up to the class's `grading_retries`,
with the backoff `grade()` retries use, and
each retry is counted as `retried/groups/grading_failed/<type>`. Without `grade()`, a
rollout's grading takes no verification slot at all: `check` reports the runtime as
acquiring verification itself, which it never needs to. A
result that is not one finite number per member (the wrong count, a NaN, a bool) is a
grader bug, and like `PermanentGradingFailure` is never retried. When the grader fails a
group, its members are discarded as `grading_failed/<type>`, with `reward: null`, and
the failure counts toward the [`group_grading` stop](failures.md#the-systemic-stop); a failed
group is a grading failure, not an unusable group. A `grade_group()` still running
`cancellation_grace_seconds` after its deadline cancelled it stops the run, keeps its
verification slot, and leaves the run not settled, like a `grade()` that does the same.

**The async caveat.** In async mode a member can still go stale between the judge and
selection. Selection drops it, and its siblings keep the rewards the judge gave them
with it present. This can't happen in sync mode, the default.

## Migrating existing episodes

This changes the author API directly; there is no compatibility adapter.
Scheduling, capacity, deadlines, registered cleanup and group grading retain their
existing contracts. Migrate a class and its direct hook callers together:

| Previous API | Current API |
| --- | --- |
| `Episode[TaskModel, OptionsModel, ExecutionKind, GradingKind]` | `Episode[TaskModel, OptionsModel]`, with `execution_kind` and `grading_kind` class variables |
| `EpisodeContext` import | Import the phase records your hooks use: `Setup`, `Sampling`, `Grading`, `Closing` |
| `__init__(task, ctx)` and `ctx.options` | `__init__(task, options)` and `self.options` |
| `setup()` / `run(model)` / `grade(rollout)` / `close()` | `setup(setup)` / `run(sample)` / `grade(grading)` / `close(closing)` |
| `ctx.sandbox` during setup or sampling | `setup.sandbox` / `sample.sandbox` |
| `ctx.sandbox` during grading or cleanup | `grading.execution_sandbox` / `closing.execution_sandbox`, possibly `None` after release |
| `ctx.grading_sandbox` | `grading.grading_sandbox` / `closing.grading_sandbox` |
| `ctx.run_process(...)` | `phase.processes.run(...)` |
| `ctx.timed_out` | `grading.timed_out` / `closing.timed_out` |
| `ctx.metrics` / `ctx.attempt_id` | `phase.metrics` / `phase.attempt_id` |

Use explicit `None` for unused task/options arguments. Constructors receive stable
input data only; move resource work into `setup()`. Narrow a phase's optional
sandbox to your declared provider kind before calling its SDK.

## Where to see it

- `discard/rollouts/<reason>/<type>` and `retried/attempts/<reason>/<type>` in
  `steps.jsonl`, for example `discard/rollouts/agent_timeout/rollout_clock`.
- Each `rollouts.jsonl` record carries `reason`, `cause` (`<reason>`, or
  `<reason>/<type>` for a failure, or `cleanup_failed/<type>` when its `abort()` or `finish()` failed and nothing
  before it did) and `retries`.
- Episode metadata of the rollouts that trained is summarized as `rollout/*`, so a
  label such as the local pattern's `agent_exit` is counted per step, up to
  `[output] rollout_metrics_limit` metrics. Each numeric `rollout/<key>` is also
  reported as `rollout/<key>_with_unusable`, which counts the members of the groups
  dropped as unusable since the previous update too.
- For a group grader: `discard/groups/grading_failed` counts the groups it failed,
  `retried/groups/grading_failed/<type>` its retries, `pipeline_calls/grade_group` and
  `queue/group_grading_groups` its calls in flight and the groups waiting for one, and
  `time/pipeline/group_grading/*` how long each call took, from its verification slot
  to its end.
