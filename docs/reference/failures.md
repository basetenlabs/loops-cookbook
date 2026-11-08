# Failures: what your code's outcomes mean

This page is for anyone writing a rollout runtime. It covers what the cookbook does with
each way your code can end, how many times it tries again, which clocks bound your
code, and when the run stops. Everything here holds for every rollout runtime,
including Harbor.

Most custom environments don't need a whole rollout runtime. An
[`Episode` class](episode-api.md) is one object per attempt, and the cookbook runs it
as a rollout runtime for you. The hooks on this page belong to the lower-level
`RolloutRuntime` contract. For an `Episode`, the cookbook's `sample()` provisions the
sandbox and calls your `__init__`, `setup()` and `run()`. Its `grade()` calls your
`grade()`, and its `finish()` and `abort()` call your `close()`.

One rule is behind all of it. Your code decides what an episode *means*, and the
cookbook does not decide that an agent's failure should go unscored. An agent that
called the policy and then crashed, gave up, or ran out of its own time still had an
episode, and your grader scores it. The cookbook decides only what it owns, which is
its own clocks, the evidence it captured, and whether the run can go on.

That includes an agent that died before its first model call. Such an agent left
nothing to train, so the cookbook discards the rollout as `unversioned` and counts it
as a sampling failure.

## The failure contract

When `sample()` ends, the cookbook seals the rollout. Sealing fixes the generations the
agent received as the evidence the rollout trains on. The cookbook then grades the
sealed rollout with `grade()`.

### Sampling: `sample()`

| `sample()` | What it means | What happens next |
| --- | --- | --- |
| It returns an `EpisodeResult`. | An episode happened, however badly the agent did. | The rollout is graded. |
| An `Episode`'s agent deadline expires. | The agent ran out of its own time. | The cookbook cancels `run()`, seals what was delivered and sets `grading.timed_out` to true. `grade()` then scores the rollout if it delivered at least one trainable generation. Otherwise the rollout is discarded as `unversioned`. |
| It raises. | The infrastructure failed, so the episode didn't happen. | The cookbook retries it up to `max_rollout_retries` times, then discards it as `rollout_failed/<type>`. |
| It raises `PermanentRolloutFailure`. | The episode can never succeed. | The cookbook discards it at once as `rollout_failed/<type>`, without a retry. |

`<type>` is the exception's type in snake case, without an `Error`, `Exception` or
`Failure` suffix. For example, `SandboxLost` becomes `sandbox_lost`. A `NamedFailure`
raised with a `name` is named by that name instead. A `PermanentRolloutFailure` or
`PermanentGradingFailure` raised `from` another exception is named by that other
exception.

So an agent that fails after calling the policy is graded. When your agent program
exits with a nonzero code, is killed, or stops at a deadline of your rollout runtime's
own, return from `sample()` and let `grade()` score what the agent left. If you want
that fact in the record, put it in the episode's metadata. The
[local pattern](../../patterns/custom_runtime/local/) records it as `agent_exit`, and
its verifier scores a missing answer 0. An agent that exits with an error after writing
a correct answer scores 1, because what the verifier says stands.

Raise only for something that stopped the episode from happening at all, such as a
sandbox that could not start or an environment that went away.

### Grading: `grade()`

| `grade()` | What it means | What happens next |
| --- | --- | --- |
| It returns a `GradeResult`. | The result holds the rollout's reward. | The rollout trains, if its group is usable. |
| It raises. | The grader failed this time. | The cookbook grades the rollout again, up to the rollout runtime's `grading_retries`, then discards it as `grading_failed/<type>`. |
| It raises `PermanentGradingFailure`. | Grading this rollout again cannot help. | The cookbook discards the rollout at once, without a retry. |

A reward that is not a finite number is a grader bug. So is grade metadata that repeats
a key of the episode's metadata. Either one fails as `PermanentGradingFailure` does,
because trying again would give the same result. A reward of `None` is allowed only
from a rollout runtime with a [group grader](episode-api.md#group-grading), which gives
the reward instead.

### What the cookbook decides itself

- **The rollout clock.** `rollout_timeout_seconds` bounds the time a rollout spends
  holding sampling capacity, less resource-service time such as a sandbox create. When
  the clock runs out, the cookbook cancels `sample()`. There is no episode to grade, so
  the rollout is discarded as `agent_timeout/rollout_clock`. This is the only agent
  outcome the cookbook decides. The rollout is not retried, and it counts as a sampling
  failure toward the [systemic stop](#the-systemic-stop). The clock fires only when
  `sample()` outlived every deadline of its own, so a run of expiries has a systemic
  cause.

  The clock is a backstop, so a rollout runtime must bound its own agent below
  `rollout_timeout_seconds`. An agent that runs out of that deadline returns and is
  graded, while a rollout that the clock stops is discarded and counted. A rollout
  runtime that derives the clock from its own deadline, as the `Episode` rollout
  runtime does, supplies it as `rollout_timeout_seconds` in its capabilities. You can
  then leave the setting unset. For any other rollout runtime, the setting is required.
- **Model calls and capture.** Sometimes the sampler or the transport loses a
  generation, or capture breaks. That decides the attempt's outcome, whatever
  `sample()` then raised or swallowed. The rollout is discarded as
  `rollout_failed/<type>` and never retried, because the Loops software development
  kit (SDK) has already retried every model-call failure that it treats as transient.
- **A completion the renderer cannot read.** Parsing a sampled completion can raise, or
  return no native message without reporting a tool call it could not parse. Either way
  the renderer failed, not the model. The attempt fails whatever `sample()` then raised
  or swallowed, and nothing of it trains or is graded. The cookbook retries it as a
  fresh attempt up to `max_rollout_retries` times, because a new attempt samples anew.
  After that, it discards the rollout as `rollout_failed/renderer`. Every failed attempt
  counts as a sampling failure.
- **Staleness.** An attempt whose evidence is too far off policy is discarded as
  `stale_before_grading` or `internal_delta`. It is not retried, and it doesn't count
  as a failure.
- **No trainable evidence.** An episode can seal with no delivered generation from a
  trainable producer, because it never called the policy or because every call was cut
  before delivery. The cookbook discards it as `unversioned`, doesn't retry it, and
  counts it as a sampling failure. An episode whose agent discarded every generation
  it was delivered is still graded. It counts in its group's baseline, and nothing of
  it trains.
- **Contract violations** stop the run at once. An example is a `sample()` that returns
  something other than an `EpisodeResult`.
- **A hook that ignores cancellation** is one that is still running more than
  `[rollout.cleanup] cancellation_grace_seconds` after it was cancelled. The cookbook
  keeps it running, because it runs in the training process and cannot be killed. The
  next section describes what happens then.

### Hooks that ignore cancellation

When the cookbook keeps a `sample()` or `grade()` running, it discards the rollout as
`unquiesced/sample` or `unquiesced/grade`. A kept `finish()` or `abort()` is a cleanup
failure, and it counts toward cleanup.

A kept hook keeps the capacity it holds and its episode's resources until it ends,
because it may still be using them. So every limit stays true. The hook gets nothing
new meanwhile. Any capacity it asks for raises `CapacityRevoked` in its own task. That
includes `hold()`, `admit()`, and a slot it was already waiting for. A process it
starts raises `ProcessesClosed`. The run goes on.

Once the hook ends, the driver takes back whatever capacity the hook left held, and it
gives the hook's attempt the cleanup it was owed. A kept `sample()` gets its `abort()`,
with the reason it was stopped. A kept `grade()` gets its `finish()` as `failed`. Each
of these counts like any other cleanup. A kept `finish()` or `abort()` has its
episode's resources released. If a release fails then, the cookbook reports it and
doesn't count it again.

The run stops in two cases. The first is when more kept hooks are still running than
5% of `max_concurrent_rollouts`, rounded up. The second is when kept hooks hold every
sampling slot, every verification slot or every start admission place, because no
other rollout could then get one. A group grader that ignores cancellation always stops
the run.

## Counters, per stage

| | Sampling (`sample()`) | Grading (`grade()`, `grade_group()`) |
| --- | --- | --- |
| Retries | `[rollout] max_rollout_retries` sets them. The default is 0. | The rollout runtime's `grading_retries` sets them, which for an `Episode` is the class's own. The default is 0. Harbor's is 0. A separate verifier's job `retry:` regrades happen inside one `grade()` call, with Harbor's backoff, all under one grading deadline. |
| What a retry is | A retry is a fresh attempt. The cookbook aborts and cleans up the failed attempt, then runs `sample()` again under a new attempt ID. Nothing from the failed attempt trains. | A retry grades the same rollout, or the same group, again. It follows a backoff that starts at 0.5 s and doubles up to 8 s. |
| When a retry is allowed | Any exception that isn't on the never-retried list earns a retry. | Retries happen only as many times as the rollout runtime declares. That is 0 when grading isn't safe to run again. |
| Never retried | These are never retried: `PermanentRolloutFailure`, the rollout clock, model-call and capture failures other than a completion the renderer cannot read, `ModelAccessError`, episode metadata that is not JSON, an agent process group that survives `SIGKILL`, and staleness. An undeclared producer or a future policy version stops the run instead. | These are never retried: `PermanentGradingFailure`, capture failures, model-access errors and grader bugs. |
| Deadline | `rollout_timeout_seconds` applies inside sampling holds, and the rollout is discarded when it expires. `setup_timeout_seconds` applies outside them. For an `Episode`, `agent_timeout_seconds` applies inside the hold, and the rollout is graded when it expires. | `[rollout.grading] timeout_seconds` applies to each try. |
| Rechecked before each try | Nothing is rechecked. | Staleness is rechecked before every try. |
| When retries run out | The rollout is discarded as `rollout_failed/<type>`. | The rollout, or every member of the group, is discarded as `grading_failed/<type>`. |
| What counts toward the stop | Every attempt that failed as infrastructure counts, whether it was retried or not, including a `PermanentRolloutFailure`. So does every `unversioned` rollout and every rollout-clock expiry. | Only the final failure counts, in the `grading` or `group_grading` count. |

## Deadlines

- **Rollout clock** (`rollout_timeout_seconds`). It runs only while the attempt holds
  sampling capacity. When a rollout runtime doesn't take the sampling hold itself, the
  cookbook holds it for the whole of `sample()`. When the clock expires, the cookbook
  cancels `sample()` and discards the rollout, as described above.
- **Setup clock** (`setup_timeout_seconds`). It bounds the time an attempt spends
  before sealing outside both sampling and verification capacity, such as preparing an
  environment before the agent starts. It doesn't count time spent waiting for
  admission or capacity, or resource-service time such as a sandbox create. Its expiry
  is an infrastructure failure, so the cookbook retries the attempt like any other.
  When the retries run out, it discards the rollout as `setup_timeout`, with the cause
  `setup_timeout/hook_deadline`.
- **Grading deadline** (`[rollout.grading] timeout_seconds`). It applies to each
  `grade()` try. A try that runs out fails as `grading_failed/hook_deadline`, and the
  cookbook retries it if grading retries are allowed. A try that returns at or after
  its deadline fails the same way, and its result never trains. That happens when
  `grade()` blocks the event loop, so the deadline's timer can't stop it.
- **Cleanup deadline** (`[rollout.cleanup] timeout_seconds`). It applies to each
  `finish()`, `abort()` and `close()`, less resource-service time such as a sandbox
  release. The attempt's processes are ended before this deadline starts.

When a deadline cancels a hook, the hook's processes get
`[rollout.cleanup] process_kill_grace_seconds` to exit after `SIGTERM`. The hook then
gets `cancellation_grace_seconds` to return. A second cancellation, as on shutdown,
shortens neither wait. A hook that is still running after both waits is handled as
described in [Hooks that ignore cancellation](#hooks-that-ignore-cancellation). A
rollout runtime's `close()` that doesn't stop at shutdown is reported, and it leaves
the run unclean.

Some client libraries leak a `CancelledError` that the cookbook didn't cause. When a
hook raises one, it is that hook's own failure. From `sample()` it is
`rollout_failed/cancelled`, and from `grade()` it is `grading_failed/cancelled`, and
each is retried and counted like any other failure. From `finish()` or `abort()` it is
a cleanup failure, and it counts toward cleanup.

## The systemic stop

`[rollout] max_consecutive_rollout_failures` counts consecutive infrastructure
failures. The default is 64, and 0 turns the stop off. The cookbook keeps a separate
count for each stage, which is sampling, grading and cleanup. A success in a stage
resets that stage's count. When one count reaches the limit, the run stops with
"N consecutive <stage> infrastructure failures: <the last one>", for example "the
rollout clock expired". A rollout-clock expiry counts toward sampling. A failed
`finish()` or `abort()` counts toward cleanup, and so does an episode whose resources
could not be proven released.

A group grader has a count of its own, `group_grading`, with the same limit. Every
group it fails counts, and every group it scores resets the count. When the count
reaches the limit, the run stops with "N consecutive group-grading failures". So a
judge outage or a bad key stops the run, instead of discarding every group.

The run also stops at once, whatever the limit, when Loops says the run has ended. A
spot preemption of the trainer ends the run, for example. The cookbook never takes a
sampling failure as proof of that. Instead, a sampling failure makes the run ask Loops,
at most once every 45 seconds. The run stops only if the Loops run is `INACTIVE`, or
its trainer deployment is `PREEMPTED`, `FAILED` or `STOPPED`. If the run is live, or
Loops can't give a status, the failure is left to the limit.

Whatever noticed first, a run that stops or fails while its Loops run has ended is
reported as `loops_run_ended`, not as `aborted` or `failed`. If a failure did not come
from that check, the run asks Loops once, for up to 10 seconds, before it decides.

### Exit codes

The command exits with one of these codes:

| Code | When |
| --- | --- |
| 0 | The run completed, or a resume found that the run had already made every update and started no trainer. |
| 1 | The run stopped or failed in any way other than those below. |
| 2 | The configuration has an error. |
| 75 | The Loops run ended under the run. This is `EX_TEMPFAIL`. |
| 130 | You pressed Ctrl-C. |
| 143 | The process received SIGTERM. |

After exit code 75, the run can resume from its latest checkpoint manifest with
`loops run <config> --resume`. `max_steps` counts the run's updates across resumes, so
the same command, run again, stops where the first run would have. A wrapper that
retries on 75 therefore retries only what a resume can fix. You decide whether to
retry, and on what capacity.

### What the limit doesn't catch

This limit catches a broken platform, not a bad agent. Agents that fail and are graded
0 are not failures here. `[training] max_consecutive_unusable_groups` catches a wave of
them instead, because a group in which every rollout scores alike is rejected as
`no_variance`.
