# 2 · Interfaces

For engineers changing the cookbook. The two ways a user plugs an environment and an
agent into the engine, how each maps onto the one contract, and why they look the way
they do. Read [1 · The training engine](1-engine.md) first.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/interfaces-dark.svg">
  <img alt="Two interfaces, one contract. In the middle, the six hooks of runtime/api.py (prepare, sample, grade, finish and abort, close) and the optional grade_group. On the left, what the Episode runtime does for each: load and validate tasks.jsonl; admit, create the sandbox, construct the Episode, setup, hold, run; Episode.grade; the optional grade_group classmethod; close the episode, the sandbox client and the attempt directory; nothing at close. On the right, the Harbor runtime: load the job with one TaskRef per task; submit a trial with Harbor's retries off, start the environment, run the agent, park at verification; release the parked verifier or regrade from artifacts; no group grading; wait for the environment stops; cancel retained trials at close." src="assets/interfaces.svg">
</picture>

## The idea

There is one seam between the engine and user code: `RolloutRuntime`
(`runtime/api.py`), six hooks the [driver](1-engine.md#the-parts) calls around its own
clocks, retries and sealing. Users almost never implement it directly. They pick one of
two adapters that implement it for them:

- **The custom runtime** (`episodes/`). The user writes one `Episode` class, one object
  per attempt. For new environments.
- **The Harbor runtime** (`harbor/`, an optional extra). The user brings a Harbor job
  (tasks with verifiers) and a `LoopsAgent`. For existing Harbor datasets.

Both are ordinary runtimes underneath, so everything in doc 1 holds for both, and the
engine can't tell them apart.

## The contract

| Hook | Called | Must |
| --- | --- | --- |
| `prepare()` | once, at preflight | return the tasks as `TaskRef(id, digest)`, one digest per id |
| `sample(attempt)` | per attempt | run the episode; return `EpisodeResult` metadata, never evidence |
| `grade(attempt, rollout)` | per graded try | return `GradeResult(reward, metadata)`; a finite, non-bool reward |
| `grade_group(members)` | per complete group; not one of the six — an optional `GroupGrader` a runtime may also implement | one final reward per member, in order |
| `finish(attempt, rollout, disposition)` / `abort(attempt, reason)` | once per attempt, in the background | clean up only; `finish()` never decides an outcome |
| `close()` | once, at shutdown | release run-level state |

A runtime also declares `RuntimeCapabilities`: its model-access mode(s), which capacity
phases it acquires itself (`SAMPLING`, `VERIFICATION`), its `grading_retries`, whether
it grades groups, and, if it can derive one, its rollout clock. Preflight checks the
config against them (`runtime/context.py`). The signatures, fields and exports are
frozen by `tests/unit/test_contract.py`.

A runtime reaches everything through `AttemptContext.services`: `model_access` (the
lease-backed client or HTTP connection), `capacity` (`admit()`, `hold(phase)`),
`resources` (sandboxes via the registry), `processes` (supervised agent programs) and
`metrics`.

## The custom runtime: `Episode`

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

Real ones: [`patterns/custom_runtime/`](../patterns/custom_runtime/). How the adapter
maps it onto the contract (`episodes/runtime.py`):

- **`prepare()`** loads `[rollout_runtime.files] tasks` and validates every line with the task
  model; the id is the line's `id` or its position, the digest the canonical JSON.
- **`sample()`**: `capacity.admit()` (pre-start places apply here) → provision the
  sandbox if the class declares a kind → `Episode(task.copy(), options)` → `setup(Setup(...))` on the
  setup clock → `hold(SAMPLING)` → `run(Sampling(...))` under `agent_timeout_seconds`. When the
  deadline cuts `run()`, `grading.timed_out` is set and the episode is graded on what was
  delivered; its metadata is empty unless `run()` caught the cancellation and returned
  some. A cut before the first reply arrived leaves nothing to train (`unversioned`).
- **`grade()`** calls `Episode.grade(Grading(...))` on the same object. An Episode without
  `grade()` must define `grade_group()`.
- **`grade_group()`** is a classmethod: it sees plain-data `GroupMember`s (task,
  transcript, reward, metadata) and the validated options, never a live episode or the
  policy.
- **`finish()` / `abort()`** run `close(Closing(...))`, close the sandbox client, and remove the
  attempt directory; the driver then releases the sandbox.

The class declares its runtime facts as class attributes: `model_access` (default
`direct`) and `grading_retries` (default 0). Because each built-in runtime supports
exactly one mode, `[rollout_runtime] model_access` is only for a hand-written runtime that
declares several. Use `Episode[None, None]` for an untyped Episode, which reads raw
JSON from `self.task.data` (the example in [reference/episode-api.md](reference/episode-api.md)
is one), but a typed task model is what gets validated at `check`, so prefer it. Everything else is `[rollout_runtime.config]`:
`episode = "module:Class"`, `agent_timeout_seconds`, and optional `sandbox`, `grading_sandbox` and
`options` tables.

## The Harbor runtime: job + `LoopsAgent`

The user writes a Harbor job file (`[rollout_runtime.files] job`) whose tasks each have an
enabled verifier, and an agent that subclasses `LoopsAgent` and states its
`model_access`. Its model is `self.llm()` (direct) or `self.services.model_access`
(HTTP); `bind_loops_services` is called before Harbor's `setup`. How the adapter maps it
(`harbor/runtime.py`, `harbor/integration.py`):

- **`prepare()`** has Harbor resolve the dataset and returns one `TaskRef` per task with
  Harbor's content digest; unsupported jobs are refused here, at `check`.
- **`sample()`** submits a trial to a hooked Harbor queue with Harbor's own retries
  forced to 0, then follows the trial through Harbor's hooks: admission at environment
  start, the sampling hold around the agent phase, and a park at verification.
- **`grade()`**: a *shared* verifier is released to run in the same environment; a
  *separate* one (`environment_mode = "separate"` in the task) is run as a regrade trial
  in a second environment, from the task's declared artifacts. The reward is Harbor's,
  read as-is, or `[rollout_runtime.config]`'s `reward_key` when the verifier returns several.
- **No group grading.** Regrades follow the job's own `retry:` block, so the runtime
  declares `grading_retries = 0`.
- On Blaxel and Daytona, the cookbook's `BlaxelEnvironment` and `DaytonaEnvironment`
  subclasses provision every sandbox through the resource registry (`harbor/blaxel.py`,
  `harbor/daytona.py`); other Harbor environments are Harbor's own and untracked.

## The rules the code keeps

1. **An Episode class is checked when the runtime is built.** It must subclass
   `Episode`, define `run()` and `grade()` or `grade_group()`, make its hooks `async`,
   and bind its type parameters; `loops check` refuses anything else. —
   `tests/unit/episodes/test_factory.py::test_a_misconfigured_episode_runtime_is_refused_when_it_is_built`.
2. **User data is typed and validated at check.** Every task line is validated by the
   task model, the options table by the options model; both must use types from an
   allow list (no secrets, no untyped dicts). —
   `test_tasks.py::test_a_line_the_task_model_refuses_is_refused_by_file_and_line`,
   `test_task_and_options_models.py::test_a_field_type_outside_the_allow_list_is_refused_as_task_or_options_by_model_and_field`.
3. **A sandbox is matched by type, not by name.** The episode's sandbox kind must be the
   provider adapter's kind or a subclass of it. —
   `tests/unit/episodes/test_run.py::test_check_matches_a_sandbox_to_its_adapters_kind_not_to_the_providers_name`.
4. **Runtime facts are declared once, by the runtime.** Model access and grading
   retries come from the class or agent; the config may not restate them. A `LoopsAgent`
   that doesn't declare `model_access` is refused. —
   `tests/unit/runtime/test_context.py::test_capabilities_are_validated_at_preflight`,
   `tests/unit/harbor/test_prepare.py::test_prepare_rejects_an_agent_it_cannot_bind_before_expanding_the_job`.
5. **A group judge never runs on a group that can't train.** Stale members are dropped
   and the size checked first; its rewards replace `grade()`'s. —
   `tests/unit/episodes/test_group_grading.py::test_a_member_gone_stale_before_the_judge_is_dropped_and_its_group_never_judged`.
6. **Harbor never retries a sampling trial.** A retried trial's samples can't be
   separated from the failed try's inside one lease; the cookbook retries with a fresh
   attempt instead. — `tests/unit/harbor/test_trials.py::test_sampling_trials_never_retry_inside_harbor`.
7. **Harbor's results are read, not reinterpreted.** A failed agent is graded exactly
   when Harbor ran its verifier. — `test_trials.py::test_a_failed_agent_is_graded_as_harbor_grades_it`.

## Decisions, and why

| Decision | Why | Rejected |
| --- | --- | --- |
| **Two adapters over one contract**, not two engines. | The engine's guarantees (evidence, clocks, sandboxes) hold once, for both. | A Harbor-only engine (develop). |
| **One `Episode` object per attempt.** A retry builds a new one. | State can't leak between attempts; `grade()` reads what `run()` left on `self`. | A long-lived environment object reused across attempts. |
| **`run()` returns, `grade()` scores, raising means "didn't happen".** | The only signal the engine needs to tell an episode from infrastructure, and the user chooses it. | Outcome knobs on the config. |
| **Typed task and options models**, validated at check. | A key read from an untyped dict fails per attempt, misfiled as infrastructure. | Raw JSON lines and option dicts. |
| **`grade_group()` sees recorded evidence only.** | A judge must be reproducible from the record; it brings its own client, never the policy. | Passing live episodes or sandboxes to the judge. |
| **Harbor is an optional extra.** | The core must run without it (a CI job proves it). | Harbor as a core dependency. |
| **The raw `RolloutRuntime` stays public.** The README names it; the patterns guide calls it "the layer underneath". | It is the seam both adapters are tested against. Whether any user story needs it directly is an open question. | — |

## Changing it safely

- A new hook, field or export changes the frozen contract: update
  `tests/unit/test_contract.py` deliberately, and both adapters.
- A new Episode capability goes in `EpisodeRuntime.capabilities` and is checked at
  preflight, never discovered at run time.
- Anything reaching Harbor's private members goes through `harbor/integration.py`,
  `harbor/blaxel.py` or `harbor/daytona.py`; `tests/unit/test_architecture.py` enforces it.
- Detail: [the failure contract](reference/failures.md) and
  [the Episode API](reference/episode-api.md), [Harbor](reference/harbor.md).

## Known gaps

Harbor environments other than Blaxel and Daytona are accepted and not tracked by the
registry.
There is no remote agent placement yet; see [3 · Patterns](3-patterns.md).
