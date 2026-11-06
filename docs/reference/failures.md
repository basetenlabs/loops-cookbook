# Failures: what your code's outcomes mean

This page is for anyone writing a runtime. It covers what the cookbook does with each way
your code can end, how many times it tries again, which clocks bound it, and when the
run stops. Everything here holds for every runtime, Harbor included.

Most custom environments don't need a whole runtime: an [`Episode` class](episode-api.md)
is one object per attempt, and the cookbook runs it as a runtime for you.

The rule behind all of it: what an episode *means* is up to your code. The cookbook
does not decide that an agent's failure should go unscored. An agent that called the
policy and then crashed, gave up, or ran out of its own time still had an episode,
and your grader scores it. The cookbook decides only what it owns: its own clocks,
the evidence it captured, and whether the run can go on. That includes an agent that
died before its first model call: it left nothing to train, so it is discarded as
`unversioned` and counted as a sampling failure.

## The failure contract

### Sampling: `sample()`

| `sample()` | Means | Then |
| --- | --- | --- |
| returns an `EpisodeResult` | An episode happened, however badly the agent did. | Graded. |
| an `Episode`'s agent deadline expires | The agent ran out of its own time. | Graded: `run()` is cancelled, what was delivered is sealed, `grading.timed_out` is true, and `grade()` scores it (if it delivered at least one trainable generation; otherwise `unversioned`). |
| raises | Infrastructure: the episode didn't happen. | Retried up to `max_rollout_retries`, then discarded as `rollout_failed/<type>`. |
| raises `PermanentRolloutFailure` | This can never succeed. | Discarded now as `rollout_failed/<type>`, not retried. |

`<type>` is the exception's type in snake case, without an `Error`, `Exception` or
`Failure` suffix (`SandboxLost` becomes `sandbox_lost`). A `NamedFailure` raised with
a `name` is named by that name instead, and a `PermanentRolloutFailure` or
`PermanentGradingFailure` raised `from` another exception is named by that exception.

So an agent that fails after calling the policy is graded. When your agent program
exits nonzero, is killed, or stops at a deadline of your runtime's own, return from
`sample()` and let `grade()` score what it left. If you want that fact in the record, put it in the episode's
metadata. The [local pattern](../../patterns/custom_runtime/local/) records it as
`agent_exit`, and its verifier scores a missing answer 0. An agent that exits with an
error after writing a correct answer scores 1: what the verifier says stands.

Only raise for what stopped the episode from happening at all, such as a sandbox that
could not start or an environment that went away.

### Grading: `grade()`

| `grade()` | Means | Then |
| --- | --- | --- |
| returns a `GradeResult` | The rollout's reward. | Trained on, if its group is usable. |
| raises | The grader failed this time. | Graded again up to the runtime's `grading_retries`, then discarded as `grading_failed/<type>`. |
| raises `PermanentGradingFailure` | Grading this rollout again cannot help. | Discarded now, not retried. |

A reward that is not a finite number, or grade metadata that repeats a key of the
episode's metadata, is a grader bug. It fails as `PermanentGradingFailure` does, since
trying again would give the same result. No reward at all (`None`) is allowed only from
a runtime with a [group grader](episode-api.md#group-grading), which gives the reward instead.

### What the cookbook decides itself

- **The rollout clock.** `rollout_timeout_seconds` bounds the time a rollout spends
  holding sampling capacity, less resource-service time such as a sandbox create. When
  it runs out, the cookbook cancels `sample()`. There is no episode to grade, so the
  rollout is discarded as `agent_timeout/rollout_clock`. This is the only agent
  outcome the cookbook decides. It is not retried, and it counts as a sampling failure
  toward the systemic stop: the clock fires only when `sample()` outlived every
  deadline of its own, so a run of expiries has a systemic cause. The clock is a
  backstop, so a runtime must bound its own agent below `rollout_timeout_seconds`:
  an agent that runs out of that deadline returns and is graded, where one the clock
  stops is discarded and counted. A runtime that derives the clock from its own
  deadline, as the `Episode` runtime does, supplies it as its capabilities'
  `rollout_timeout_seconds`, and the setting may be left unset; for any other runtime
  it is required.
- **Model calls and capture.** A generation the sampler or the transport lost, or
  capture that broke, decides the attempt's outcome whatever `sample()` then raised or
  swallowed. The rollout is discarded as `rollout_failed/<type>` and never retried,
  because the Loops SDK has already retried every model-call failure it treats as
  transient.
- **A completion the renderer cannot read.** When parsing a sampled completion raises,
  or returns no native message without reporting a tool call it could not parse, the
  renderer failed, not the model. The attempt fails whatever `sample()` then raised or
  swallowed, and nothing of it trains or is graded. It is retried as a fresh attempt
  up to `max_rollout_retries`, since a new attempt samples anew, then discarded as
  `rollout_failed/renderer`, and every failed attempt counts as a sampling failure.
- **Staleness.** An attempt whose evidence is too far off policy is discarded as
  `stale_before_grading` or `internal_delta`. It is not retried and not counted as a
  failure.
- **No trainable evidence.** An episode that seals with no delivered generation from a
  trainable producer (it never called the policy, or every call was discarded or cut)
  is discarded as `unversioned`, is not retried, and counts as a sampling failure.
- **Contract violations** stop the run at once. For example, `sample()` returning
  something other than an `EpisodeResult`.
- **A hook that ignores cancellation**, still running more than
  `[rollout.cleanup] cancellation_grace_seconds` after it was cancelled, is kept
  running. A kept `sample()` or `grade()` has its rollout discarded as
  `unquiesced/sample` or `unquiesced/grade`. A kept `finish()` or `abort()` is a
  cleanup failure, counted toward cleanup. A kept hook keeps the capacity it holds and
  its episode's resources until it ends, since it may still be using them, so every
  limit stays true. It gets nothing new meanwhile: any capacity it asks for
  (`hold()`, `admit()`, or a slot it was already waiting for) raises `CapacityRevoked`
  in its own task, a process it starts raises `ProcessesClosed`, and the run goes on.
  Once it ends, the driver takes back whatever capacity it left held and gives its
  attempt the cleanup it was owed: a kept `sample()` gets its `abort()`, with the
  reason it was stopped for, and a kept `grade()` its `finish()` as `failed`, each
  counted like any other cleanup. A kept `finish()` or `abort()` has its episode's
  resources released; a release that fails then is reported, not counted again.
  The run stops once more kept hooks are still running than 5% of
  `max_concurrent_rollouts`, rounded up, or once they hold every sampling slot, every
  verification slot or every start admission place, which no other rollout could then
  get. A group grader that ignores cancellation still stops the run.

## Counters, per stage

| | Sampling (`sample()`) | Grading (`grade()`, `grade_group()`) |
| --- | --- | --- |
| retries | `[rollout] max_rollout_retries`, default 0 | the runtime's `grading_retries` (an `Episode` class's), default 0; Harbor's is 0, and a separate verifier's job `retry:` regrades happen inside one `grade()` call, with Harbor's backoff, all under one grading deadline |
| a retry is | a fresh attempt: the failed one is aborted and cleaned up, then `sample()` runs again under a new attempt ID. Nothing from the failed attempt trains. | the same rollout, or the same group, graded again, after a backoff that starts at 0.5 s and doubles up to 8 s |
| allowed when | any exception not on the never-retried list | only as many times as the runtime declares; 0 when grading is not safe to run again |
| never retried | `PermanentRolloutFailure`, the rollout clock, model-call and capture failures (a completion the renderer cannot read excepted), `ModelAccessError`, episode metadata that is not JSON, an agent process group that survives `SIGKILL`, staleness; an undeclared producer or a future policy version stops the run instead | `PermanentGradingFailure`, capture failures, model-access errors, grader bugs |
| deadline | `rollout_timeout_seconds` inside sampling holds (discarded); `setup_timeout_seconds` outside them; for an `Episode`, `agent_timeout_seconds` inside the hold (graded) | `[rollout.grading] timeout_seconds`, per try |
| rechecked | — | staleness, before every try |
| exhausted | `rollout_failed/<type>` | `grading_failed/<type>`, for the rollout or for every member of the group |
| counts toward the stop | every attempt that failed as infrastructure, retried or not (a `PermanentRolloutFailure` included), every `unversioned` rollout, and every rollout-clock expiry | the final failure only, in the `grading` or `group_grading` count |

## Deadlines

- **Rollout clock** (`rollout_timeout_seconds`). It runs only while the attempt holds
  sampling capacity. When a runtime does not take the sampling hold itself, the
  cookbook holds it for the whole of `sample()`. Expiry cancels `sample()` and discards
  the rollout, as above.
- **Setup clock** (`setup_timeout_seconds`). It bounds an attempt's time before
  sealing that is spent outside both sampling and verification capacity, such as
  preparing an environment before the agent starts, less time waiting for admission
  or capacity and resource-service time such as a sandbox create. Expiry is an infrastructure
  failure, discarded as `setup_timeout` (cause `setup_timeout/hook_deadline`) and
  retried like any other.
- **Grading deadline** (`[rollout.grading] timeout_seconds`). It applies to each
  `grade()` try. A try that runs out fails, as `grading_failed/hook_deadline`, and is
  retried if grading retries are allowed.
- **Cleanup deadline** (`[rollout.cleanup] timeout_seconds`). It applies to each
  `finish()`, `abort()` and `close()`, less resource-service time such as a sandbox
  release; the attempt's processes are ended before it starts.

When a deadline cancels a hook, the hook's processes get
`[rollout.cleanup] process_kill_grace_seconds` to exit after `SIGTERM`, and the hook
then gets `cancellation_grace_seconds` to return. A second cancellation, as on
shutdown, shortens neither wait. A hook that is still running after
that is handled as above; a runtime `close()` that does not stop at shutdown is
reported and leaves the run unclean.
A `CancelledError` a hook raises when the cookbook did not cancel it, as some client
libraries leak, is that hook's own failure: `rollout_failed/cancelled` from `sample()`,
`grading_failed/cancelled` from `grade()`, retried and counted like any other.
From `finish()` or `abort()` it is a cleanup failure, counted toward cleanup.

## The systemic stop

`[rollout] max_consecutive_rollout_failures` (default 64, 0 disables) is a count of
consecutive infrastructure failures, kept separately for each stage: sampling, grading
and cleanup. A success in a stage resets that stage's count. When one count reaches
the limit, the run stops with "N consecutive <stage> infrastructure failures: <the
last one>", for example "the rollout clock expired". A rollout-clock expiry counts
toward sampling. A failed `finish()` or `abort()` counts toward cleanup, and so does an
episode whose resources could not be proven released.

A group grader has a count of its own, `group_grading`, with the same limit. Every
group it fails counts, and every group it scores resets the count. When it reaches the
limit, the run stops with "N consecutive group-grading failures", so a judge outage or
a bad key stops the run rather than discarding every group.

The run also stops at once, whatever the limit, when Loops says the run has ended, as
a spot preemption of its trainer does. A sampling failure is never taken as proof of
that. Instead, it makes the run ask Loops, at most once every 45 seconds. The run
stops only if the Loops run is `INACTIVE` or its trainer deployment is `PREEMPTED`,
`FAILED` or `STOPPED`. A live run, or a status Loops cannot give, leaves the failure to
the limit.

Whatever noticed first, a run that stops or fails while its Loops run has ended is
reported as `loops_run_ended`, not `aborted` or `failed`. A failure that did not come
from that check makes the run ask Loops once, for up to 10 seconds, before it decides.
The command then exits 75 (`EX_TEMPFAIL`): the run can resume from its latest
checkpoint manifest, with `loops run <config> --resume`. `max_steps` counts the run's
updates across resumes, so the same command, run again, stops where the first would
have. When the run already made every update, it exits 0 and starts no trainer.
Otherwise it exits 0 when the run completes, 1 when it stops or
fails any other way, 2 for a configuration error, 130 on Ctrl-C and 143 on SIGTERM. So a wrapper that
retries on 75 retries only what a resume can fix. Whether to retry, and on what
capacity, is yours to decide.

This limit catches a broken platform, not a bad agent. Agents that fail and are graded
0 are not failures here. A wave of them is caught by
`[training] max_consecutive_unusable_groups` instead, because groups where every
rollout scores alike are rejected as `no_variance`.
