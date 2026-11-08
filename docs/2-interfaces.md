# 2 · Interfaces

This page is for engineers who change the cookbook. It describes the two ways a user
plugs an environment and an agent into the engine, how each one maps onto the single
contract, and why they look the way they do. Read
[1 · The training engine](1-engine.md) first.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/interfaces-dark.svg">
  <img alt="Two interfaces, one contract. In the middle, the six hooks of runtime/api.py (prepare, sample, grade, finish and abort, close) and the optional grade_group. On the left, what the custom rollout runtime does for each: load and validate tasks.jsonl; admit, create the sandbox, construct the Episode, setup, hold, run; Episode.grade; the optional grade_group classmethod; close the episode, the sandbox client and the attempt directory; nothing at close. On the right, the Harbor rollout runtime: load the job with one TaskRef per task; submit a trial with Harbor's retries off, start the environment, run the agent, park at verification; release the parked verifier or regrade from artifacts; no group grading; wait for the environment stops; cancel retained trials at close." src="assets/interfaces.svg">
</picture>

## The idea

There is one seam between the engine and user code. It is `RolloutRuntime`, in
`runtime/api.py`, a contract of six hooks that the [driver](1-engine.md#the-parts) calls
around its own clocks, retries and sealing. Users almost never implement it directly.
Instead, they pick one of two adapters that implement it for them:

- **The custom rollout runtime**, in `episodes/`, is for new environments. The user
  writes one `Episode` class, and the cookbook creates one object of that class per
  attempt.
- **The Harbor rollout runtime**, in `harbor/`, is for existing Harbor datasets. It
  is an optional extra. The user brings a Harbor job, which names tasks that have
  verifiers, and a `LoopsAgent`.

Both adapters are ordinary rollout runtimes underneath. So everything in
[1 · The training engine](1-engine.md) holds for both, and the engine can't tell them
apart.

## The contract

| Hook | When the engine calls it | What it must do |
| --- | --- | --- |
| `prepare()` | Once, at preflight. | It returns the tasks as `TaskRef(id, digest)`, with one digest per id. |
| `sample(attempt)` | Once per attempt. | It runs the episode and returns `EpisodeResult` metadata, never evidence. |
| `grade(attempt, rollout)` | Once per graded try. | It returns `GradeResult(reward, metadata)`. The reward must be a finite number that isn't a bool. |
| `grade_group(members)` | Once per complete group. This hook isn't one of the six. It is an optional `GroupGrader` that a rollout runtime may also implement. | It returns one final reward per member, in order. |
| `finish(attempt, rollout, disposition)` and `abort(attempt, reason)` | One of them runs once per attempt, in the background. | They only clean up. `finish()` never decides an outcome. |
| `close()` | Once, at shutdown. | It releases run-level state. |

A rollout runtime also declares its `RuntimeCapabilities`. These state:

- which model access modes it supports
- which capacity phases it acquires itself, `SAMPLING` or `VERIFICATION`
- its `grading_retries`
- whether it grades groups
- its rollout clock, if it can derive one

Preflight checks the config against these capabilities (`runtime/context.py`).
`tests/unit/test_contract.py` freezes the contract's signatures, fields and exports.

A rollout runtime reaches every cookbook service through `AttemptContext.services`:

- `model_access` gives the lease-backed client or the HTTP connection.
- `capacity` gives `admit()` and `hold(phase)`.
- `resources` provisions sandboxes through the registry.
- `processes` runs supervised agent programs.
- `metrics` records the rollout runtime's own metrics.

## The custom rollout runtime: `Episode`

```python
from loops_cookbook.episodes import Episode, Grading, Sampling


class AnswerEpisode(Episode[AnswerTask, None]):  # task model, options model
    grading_retries = 1  # grade() is pure, so safe to run again

    async def run(self, sample: Sampling) -> dict[str, str]:
        reply = await sample.model.llm().conversation().generate(
            messages=[{"role": "user", "content": self.task.value.instruction}]
        )
        self.answer = (reply.content or "").strip()
        return {}

    async def grade(self, grading: Grading) -> float:
        return float(getattr(self, "answer", None) == self.task.value.answer)
```

Working examples are in `patterns/custom_runtime/`, such as
[`patterns/custom_runtime/in_process_direct/`](../patterns/custom_runtime/in_process_direct/). The
adapter in `episodes/runtime.py` maps the class onto the contract like this:

- **`prepare()`** loads the `tasks` file named in `[rollout_runtime.files]`, and
  validates every line with the task model. A task's id is the line's `id`, or the
  line's position when it has none. Its digest is the line's canonical JSON.
- **`sample()`** runs these steps in order:
  1. It calls `capacity.admit()`. The pre-start places apply here.
  2. If the class declares a sandbox type, and `[rollout_runtime.config]` has a
     `sandbox` table, it provisions the sandbox.
  3. It builds the episode with `Episode(task.copy(), options)`.
  4. It calls `setup(Setup(...))` on the setup clock.
  5. It takes `hold(SAMPLING)`.
  6. It calls `run(Sampling(...))` under the agent's deadline, `agent_timeout_seconds`.

  When the deadline cuts `run()` off, the adapter sets `grading.timed_out`, and the
  episode is graded on what was delivered. Its metadata is empty unless `run()` caught
  the cancellation and returned some. If the cut came before the first reply arrived,
  there is nothing to train, and the rollout is discarded as `unversioned`.
- **`grade()`** calls `Episode.grade(Grading(...))` on the same object. An `Episode`
  without `grade()` must define `grade_group()`.
- **`grade_group()`** is a classmethod. It sees plain-data `GroupMember` records, each
  with a task, a transcript, a reward and metadata, and it sees the validated options.
  It never sees a live episode or the policy.
- **`finish()`** and **`abort()`** both run `close(Closing(...))`, close the sandbox
  client and remove the attempt directory. The driver then releases the sandbox.

The class declares two facts about its rollout runtime as class attributes.
`model_access` defaults to `direct`, and `grading_retries` defaults to 0. Each built-in
rollout runtime supports exactly one mode, so `[rollout_runtime] model_access` is only
for a hand-written rollout runtime that declares several.

For an untyped `Episode`, use `Episode[None, None]`. It reads raw JSON from
`self.task.data`, as the example in [reference/episode-api.md](reference/episode-api.md)
does. Prefer a typed task model, though, because `check` validates the tasks against
it.

Everything else goes in `[rollout_runtime.config]`. That table holds
`episode = "module:Class"`, `agent_timeout_seconds`, and the optional `sandbox`,
`grading_sandbox` and `options` tables.

## The Harbor rollout runtime: job + `LoopsAgent`

The user writes two things. The first is a Harbor job file, named by the `job` key in
`[rollout_runtime.files]`, in which every task has an enabled verifier. The second
is an agent that subclasses `LoopsAgent` and states its `model_access`. With `direct`
access, the agent reaches the model through `self.llm()`. With `openai_http` access, it
uses `self.services.model_access`. The adapter calls `bind_loops_services` before
Harbor's `setup`.

The adapter in `harbor/runtime.py` and `harbor/integration.py` maps the job onto the
contract like this. A trial is Harbor's word for one run of an agent on one task.

- **`prepare()`** has Harbor resolve the dataset, and returns one `TaskRef` per task
  with Harbor's content digest. It refuses an unsupported job here, so the refusal
  happens at `check`.
- **`sample()`** submits a trial to a Harbor queue that the cookbook hooks into, with
  Harbor's own retries forced to 0. Then it follows the trial through Harbor's hooks.
  It takes admission when the environment starts, holds sampling capacity around the
  agent phase, and parks the trial when it reaches verification.
- **`grade()`** depends on the kind of verifier. A *shared* verifier is released to run
  in the same environment. A *separate* verifier, which the task marks with
  `environment_mode = "separate"`, runs as a regrade trial in a second environment,
  from the task's declared artifacts. The reward is Harbor's, read as it is. When the
  verifier returns several rewards, `reward_key` in `[rollout_runtime.config]`
  picks one.
- **Group grading isn't supported.** Regrades follow the job's own `retry:` block, so
  the rollout runtime declares `grading_retries = 0`.
- **On Blaxel and Daytona**, the cookbook's `BlaxelEnvironment` and `DaytonaEnvironment`
  subclasses provision every sandbox through the resource registry
  (`harbor/blaxel.py`, `harbor/daytona.py`). Other Harbor environments are Harbor's
  own, and the registry doesn't track them.

## The rules the code keeps

1. **An `Episode` class is checked when the rollout runtime is built.** It must
   subclass `Episode`, define `run()`, and define `grade()` or `grade_group()`. Its
   hooks must be `async`, and it must bind its type parameters. `loops check` refuses
   anything else. The test is
   `tests/unit/episodes/test_factory.py::test_a_misconfigured_episode_runtime_is_refused_when_it_is_built`.
2. **User data is typed and validated at check.** The task model validates every task
   line, and the options model validates the options table. Both models must use types
   from an allow list, which excludes secrets and untyped dicts. The tests are
   `test_tasks.py::test_a_line_the_task_model_refuses_is_refused_by_file_and_line` and
   `test_task_and_options_models.py::test_a_field_type_outside_the_allow_list_is_refused_as_task_or_options_by_model_and_field`.
3. **A sandbox is matched by type, not by name.** The episode's sandbox kind must be the
   provider adapter's kind or a subclass of it. The test is
   `tests/unit/episodes/test_run.py::test_check_matches_a_sandbox_to_its_adapters_kind_not_to_the_providers_name`.
4. **The rollout runtime declares its own facts, once.** Model access and grading
   retries come from the class or the agent, and the config may not restate them. A
   `LoopsAgent` that doesn't declare `model_access` is refused. The tests are
   `tests/unit/runtime/test_context.py::test_capabilities_are_validated_at_preflight`
   and
   `tests/unit/harbor/test_prepare.py::test_prepare_rejects_an_agent_it_cannot_bind_before_expanding_the_job`.
5. **A group grader never runs on a group that can't train.** Stale members are dropped
   and the group's size is checked first. The group grader's rewards replace
   `grade()`'s. The test is
   `tests/unit/episodes/test_group_grading.py::test_a_member_gone_stale_before_the_judge_is_dropped_and_its_group_never_judged`.
6. **Harbor never retries a sampling trial.** Inside one lease, a retried trial's samples
   can't be separated from the failed try's. The cookbook retries with a fresh attempt
   instead. The test is
   `tests/unit/harbor/test_trials.py::test_sampling_trials_never_retry_inside_harbor`.
7. **Harbor's results are read, not reinterpreted.** A failed agent is graded exactly
   when Harbor ran its verifier. The test is
   `test_trials.py::test_a_failed_agent_is_graded_as_harbor_grades_it`.

## Decisions, and why

| Decision | Why | Rejected |
| --- | --- | --- |
| **Two adapters over one contract**, not two engines. | The engine's guarantees for evidence, clocks and sandboxes hold once, for both adapters. | A Harbor-only engine. |
| **One `Episode` object per attempt.** A retry builds a new one. | State can't leak between attempts, and `grade()` reads what `run()` left on `self`. | A long-lived environment object reused across attempts. |
| **`run()` returns, `grade()` scores, and raising means "didn't happen".** | This is the only signal the engine needs to tell an episode from an infrastructure failure, and the user chooses it. | Config settings that decide outcomes. |
| **Typed task and options models**, validated at check. | A key read from an untyped dict fails on each attempt, and the failure is misfiled as infrastructure. | Raw JSON lines and option dicts. |
| **`grade_group()` sees recorded evidence only.** | A group grader must be reproducible from the record. It brings its own client, and never uses the policy. | Passing live episodes or sandboxes to the group grader. |
| **Harbor is an optional extra.** | The core must run without Harbor, and a CI job proves it does. | Harbor as a core dependency. |
| **The raw `RolloutRuntime` stays public.** [patterns/README.md](../patterns/README.md) names it as the lower-level contract that a user can implement instead, and [reference/library.md](reference/library.md) lists it in the library's public API. | It is the seam that both adapters are tested against. Whether any user story needs it directly is an open question. | None. |

## Changing it safely

- A new hook, field or export changes the frozen contract. Update
  `tests/unit/test_contract.py` deliberately, and update both adapters.
- A new `Episode` capability goes in `EpisodeRuntime.capabilities`, and preflight checks
  it. Never discover a capability at run time.
- Anything that reaches Harbor's private members goes through `harbor/integration.py`,
  `harbor/blaxel.py` or `harbor/daytona.py`. `tests/unit/test_architecture.py` enforces
  this.
- For details, read [the failure contract](reference/failures.md),
  [the Episode API](reference/episode-api.md) and [Harbor](reference/harbor.md).

## Known gaps

Harbor environments other than Blaxel and Daytona are accepted, but the registry doesn't
track them. An agent can't run inside the sandbox yet. See
[3 · Patterns](3-patterns.md).
