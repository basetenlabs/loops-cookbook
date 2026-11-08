# Runtime: Harbor

Bring a Harbor dataset, define a `LoopsAgent`, run. Harbor owns the tasks, the
environments and the verifier (on Blaxel or Daytona, the run provisions and owns every sandbox
Harbor starts, below); you own the agent; the cookbook owns everything between the
agent's model calls and the optimizer.

Last run end to end with Harbor 0.23.0 on baseten-loops 0.28.0; the cookbook now requires
baseten-loops 0.32.0 or newer. Harbor is an optional extra, bounded to `harbor>=0.23,<0.24`
while the cookbook overrides Harbor's private members:

```sh
uv sync --extra harbor      # Harbor on Docker
uv sync --extra harbor-blaxel  # Harbor on Blaxel sandboxes
```

## 1. Bring a Harbor dataset

Any Harbor job works: a dataset from a Harbor registry or git repo, or local task
directories. Write a normal `JobConfig`:

```yaml
# job.yaml
datasets:
  - repo: https://github.com/harbor-framework/terminal-bench-2-1@<commit>
    path: tasks
# or:
# tasks:
#   - path: ./my_task
agents:
  - import_path: my_agent:MyAgent
    kwargs: { max_turns: 16 }
environment:
  type: docker            # or blaxel, below
retry:
  max_retries: 2
```

### What the job must satisfy

- **Agent execution plus an enabled verifier.** `install_only`, regrade-only, and
  `verifier.disable` jobs are rejected by `check`.
- **One numeric reward, or `reward_key`.** The verifier's reward map must have a
  single entry, or you name the one to train on in `[rollout_runtime.config] reward_key`. Booleans,
  non-finite values, and a missing reward are failures, not zero.
- **A reward gate drops rewards, never zeroes them.** `[rollout_runtime.config] reward_gate` names an
  async `(trial, verifier_result) -> str | None`, run once the reward is final and before the
  sandboxes still running stop; a separate verifier's agent sandbox is already released. A
  reason drops the trial from training as `grading_failed/<reason>`, without counting toward
  the consecutive-failure stop; its reward stays in the trial directory. A reason is the metric's
  name: lowercase letters, digits and underscores. A gate that raises fails grading as
  `grading_failed/reward_gate`; one that returns anything but a reason or None, as
  `grading_failed/reward_gate_result`. The gate runs inside `grade()`, even for a multi-step
  trial that stopped early, its sandbox held until it answers, so it is bounded as `grade()`
  is: past the grading deadline it fails as `grading_failed/hook_deadline`, and one that ignores
  its cancellation is kept as any `grade()` that does (`unquiesced/grade`).
- **Separate verifiers grade from artifacts.** For single-step tasks whose verifier
  runs in its own environment, sampling and grading are decoupled: the agent's
  sandbox is released as soon as the agent finishes (with `environment.delete: false`,
  when its attempt ends), and Harbor regrades from the
  recorded trial. Anything the verifier needs must therefore be declared as a task
  artifact. Shared-verifier tasks are graded live in the same sandbox. Multi-step
  tasks keep Harbor's step ordering.
- **A failed agent is graded, as Harbor grades it.** When the agent times out
  (`AgentTimeoutError`) or exits with an error (`NonZeroAgentExitCodeError`, or any of
  its subclasses, such as the API errors Harbor's installed agents raise), Harbor
  still runs the verifier, and the trial trains on that reward: an agent that runs out
  of time earns its low reward. A separate-verifier trial is regraded from the saved
  trial exactly when Harbor carried it on to its verification; one Harbor ended
  before then is a rollout failure. A generation the deadline cut mid-flight is
  dropped from the evidence; the rest of the episode trains. A shared verifier sees the
  sandbox, and a separate one the saved trial's artifacts, never the trial's exception: an agent can leave evidence there for a custom
  verifier to read, but a timeout or error Harbor enforced is invisible to it.
  No runtime discards a failed agent's episode: what the verifier scores stands, and
  only the cookbook's own rollout clock discards (see
  [failures.md](failures.md)). The run
  reports how often each happened as `runtime/harbor/trial_exceptions/<type>` against
  `runtime/harbor/sampling_trials`.
- **A sandbox that runs out is a sizing question.** Its disk or memory is the task's
  or the job's size; the cookbook does not reinterpret the trial, and Harbor's result
  stands. On Blaxel and Daytona, each trial reports its sandbox's OOM-kill count, read
  when the agent phase ends: on Blaxel, a micro-VM, its kernel's (`/proc/vmstat`); on
  Daytona, a container on a shared host kernel, its cgroup's (`memory.events`), as
  `runtime/harbor/sandbox_oom_kills/{count,mean,max}` (and
  `runtime/harbor/oom_killed_trials` for trials with any). It is a runtime metric
  only: the agent's metadata is left as the agent wrote it. A count that cannot be
  read is not reported.
- **Other errors fail the rollout.** Harbor never retries a sampling trial itself.
  An exception raised before sampling ends that leaves no reward is a rollout failure:
  permanent if its type is in the job's `retry.exclude_exceptions`, otherwise eligible
  for `[rollout] max_rollout_retries`, and it counts toward
  `max_consecutive_rollout_failures`. A shared verifier's failure surfaces in
  `grade()` instead, as `grading_failed/<type>` with no cookbook retry (Harbor's
  `grading_retries` is 0). The job's `retry` block applies only to
  separate-verifier regrades.
- **Harbor's trial cap is derived**, as `max_concurrent_rollouts` plus the pre-start
  places plus `max_concurrent_verifiers` (see
  [scheduling.md](scheduling.md)). Setting `n_concurrent_trials` in the
  job file has no effect; the cookbook is the only thing that knows these numbers.
- **The grading deadline covers Harbor's.** `check` refuses a job whose
  `[rollout.grading] timeout_seconds` does not exceed Harbor's own bound for one
  `grade()`: for a separate verifier with R retries, `(R+1) x (build + verifier)` plus
  Harbor's backoff.
- **Resume checks contents.** A checkpoint records each task's ID and Harbor content
  digest, and the job file's digest. Resuming after editing a task, or changing the
  task set, restarts the task order from the beginning with a warning; a changed job
  file is reported as a changed setting.
- **Remote datasets download at `check`.** Private ones use your own git credentials
  or `harbor auth login`; the cookbook adds nothing.

### Blaxel

```sh
uv sync --extra harbor-blaxel
```

```yaml
environment:
  import_path: loops_cookbook.harbor.blaxel:BlaxelEnvironment
  override_memory_mb: 24576   # optional: one size for every task
  kwargs:
    ttl: 8h                   # required
```

Each task's `memory_mb` sizes its sandbox unless the job sets `override_memory_mb`.
Blaxel's disk is a tmpfs of half the memory, and it does not apply a task's
`storage_mb` or `cpus`: a sandbox gets one CPU core per 2048 MB of memory.
`loops check` logs the sandbox sizes, grouping tasks by memory, and warns about each
task whose declared `storage_mb` does not fit, and about any that declares `cpus` (or
when the job sets `override_cpus`). Then add `[providers.blaxel]` to the training TOML with the region; its deadlines
default (see [resources.md](resources.md#provider-resources)), and the defaults'
start bound, 630 s, fits a 600 s build timeout times the shipped jobs'
`environment_build_timeout_multiplier` of 1.1. Put
`BL_WORKSPACE` and `BL_API_KEY` in `.env`. The region is stated in `[providers.blaxel]`
only: every sandbox is created in it, and `loops check` refuses a job that sets
`environment.kwargs.region`. Images are built there too: every build carries a `blaxel.toml` naming it, a task's own `blaxel.toml` gets it added in the build's copy when it names none, and `check` refuses one that names another region.

The `import_path` is required: `type: blaxel` on its own, or Harbor's own Blaxel
class, is refused, because Harbor would create sandboxes the run never owns. With
the cookbook's class, every sandbox a trial starts (the agent's, a separate
verifier's, a regrade's) is provisioned by the run under Harbor's own name and
released when Harbor stops it, so a crashed run's sandboxes are recovered by
resuming the run, which releases what its previous process left before admitting an
attempt, or with `loops cleanup`. A job's `environment.delete: false` does not keep
a sandbox: whatever Harbor leaves running is released when its attempt ends. A stop
whose release does not prove the sandbox gone raises from `stop()`, where Harbor's own
would log and carry on, so the attempt's cleanup reports it and tries again; Daytona's
environment follows the same rule. Harbor still runs commands and verification; the run's Blaxel provider
builds images, and an image already built is reused and never deleted, unless a task starts from an image already in Blaxel (below). Images are named by what they build, not by task, so tasks with identical environments share one image and one build. `check` also refuses a task whose
effective build timeout (the task's `build_timeout_sec`, 600s by default, times the
job's `environment_build_timeout_multiplier`, or `timeout_multiplier` when unset) does
not cover the provider's
reconcile (prerequisite resolution), image-build, create, and attach deadlines
together, since Harbor bounds each environment start by it. The wait for a
provisioning slot (`[rollout] max_concurrent_provisioning`) is not covered: it depends
on how many starts overlap, and a start that runs out of time is a failed rollout,
retried under `max_rollout_retries`. The shipped Blaxel jobs set the multiplier to 1.1.
Only Harbor's direct mode is supported, not compose.

#### Starting from an image already in Blaxel

A task that should start from an image you built yourself, rather than one the run
builds, says so once, in its own `task.toml`:

```toml
[environment]
docker_image = "blaxel://sandbox/my-image:v1"
```

Everything after `blaxel://` goes to Blaxel's sandbox create unchanged, so any image
reference Blaxel accepts works, and nothing is built, so `force_build` does not apply.
As for any Harbor task with a `docker_image`, a nonempty `environment/` with no
Dockerfile is uploaded into the working directory once the sandbox starts; with a
Dockerfile, the Dockerfile is ignored and nothing is uploaded. A `docker_image` without
the scheme keeps Harbor's meaning, a base image the run builds on. The scheme is not a
Docker reference, so neither can be mistaken for the other.

An imported image must carry Blaxel's sandbox API, which every Blaxel sandbox runs as
its entrypoint. An image the run builds gets it from Blaxel's builder; an imported one
skips that build, so its Dockerfile adds it, at a pinned version:

```dockerfile
COPY --from=ghcr.io/blaxel-ai/sandbox:<version> /sandbox-api /usr/local/bin/sandbox-api
ENTRYPOINT ["/usr/local/bin/sandbox-api"]
```

Blaxel requires it: every command and file operation in the sandbox goes through it.

Get the image into the workspace with `bl push` (see Blaxel's
[custom images](https://docs.blaxel.ai/Sandboxes/Templates)), or have Blaxel import one
from a registry with `POST /images` (`name`, `resourceType: "sandbox"`, `image`, `tag`,
and a `dockerConfig` holding the registry's pull credential). Blaxel reserves the tag
`latest`, and tags are immutable. `bl get image sandbox/<name> --latest` shows an
image's tags; `sandbox/<name>:<tag>` is the reference to write.

A separate verifier's environment, or a step verifier's, may name one the same way, in
its own `[verifier.environment]`. `loops check`, and `loops run` before it starts
anything, refuse a task naming a `blaxel://` image in any of its environments:

- on a job not backed by Blaxel (Daytona, local Docker), which would pull it as a
  Docker image and fail on every trial;
- written `sandbox/<name>` with no tag;
- written `sandbox/<name>:<tag>` when the workspace has no image `<name>`, or the image
  has no tag `<tag>`.

Each image is looked up once, whatever tags name it, with
`GET /images/sandbox/<name>/tags`, and only when a task names one. Blaxel looks images up by name only, so a reference in any other
form it accepts goes unchecked, as does one Blaxel does not answer for; the check logs
each such image. When every environment of a task starts from a `blaxel://` image,
nothing is built, so the task's build timeout need not cover the image build deadline,
and its own `blaxel.toml` is not checked for region.

### Daytona

```sh
uv sync --extra harbor-daytona
```

```yaml
environment:
  import_path: loops_cookbook.harbor.daytona:DaytonaEnvironment
  override_cpus: 1            # optional: one size for every task
  override_memory_mb: 4096
  override_storage_mb: 3072
```

Add `[providers.daytona]` to the training TOML, with its sandbox sizes and `ttl_minutes`, and put
`DAYTONA_API_KEY` in `.env`. Daytona applies a task's `cpus`, `memory_mb`, and
`storage_mb` independently; a sandbox the run loses stops after idling, is deleted as
it stops, and is destroyed at a wall-clock TTL. As on Blaxel, the `import_path` is
required, every sandbox is provisioned and released by the run, and `check` refuses a
build timeout that does not cover the provider's reconcile, create, and attach
deadlines. See [daytona.md](daytona.md) for what the cookbook adds to Harbor's Daytona
environment, sizing, and choosing between the two.

## 2. Define a `LoopsAgent`

Your agent is a Harbor `BaseAgent` that subclasses `LoopsAgent`. The cookbook binds the
attempt's services before `setup`, and `self.llm()` then returns its model client. Declare its settings with a Harbor `AgentOptions` model so
`agents[].kwargs` is validated.

```python
# my_agent.py
from harbor.agents.options import AgentOptions
from pydantic import Field

from loops_cookbook.harbor import LoopsAgent
from loops_cookbook.model_access import ModelAccessMode


class Options(AgentOptions):
    max_turns: int = Field(default=16, ge=1)
    max_tokens: int = Field(default=2048, ge=1)


class MyAgent(LoopsAgent):
    model_access = ModelAccessMode.DIRECT  # required: how the agent reaches the model
    options_model = Options
    options: Options

    @staticmethod
    def name() -> str:
        return "my-agent"

    def version(self) -> str:
        return "1"

    async def setup(self, environment) -> None:
        pass

    async def run(self, instruction, environment, context) -> None:
        history = [{"role": "user", "content": instruction}]
        conversation = self.llm().conversation()
        for _ in range(self.options.max_turns):
            reply = await conversation.generate(
                messages=history, tools=TOOLS, max_tokens=self.options.max_tokens
            )
            if reply.truncated:
                break
            turn = reply.history_message  # the turn as written, parsed or not
            history.append(turn)
            if not reply.tool_calls:
                break
            for call, written in zip(reply.tool_calls, turn["tool_calls"]):
                call_id = written.get("id") or ""  # answer the call by the ID the turn gives it
                if call.status != "ok":  # a call that did not parse: tell the model
                    history.append({"role": "tool", "tool_call_id": call_id, "name": call.name or "",
                                    "content": f"could not parse: {call.raw}"})
                    continue
                result = await environment.exec(call.arguments["command"])
                history.append({"role": "tool", "tool_call_id": call_id, "name": call.name,
                                "content": result.stdout})
```

`generate` renders the history with the model's chat template, samples, records the
exact tokens, and returns a reply with `content`, `reasoning_content`, `tool_calls`
(parsed, each with a `status`; `arguments` is a dict when it parsed, and `id` may be
`None`), `stop_reason`, `usage`, `parse_error`, `native_message`, `history_message`
(what to append: the native message, or the turn as written when a call did not
parse, appended as it is: it continues only from its sampled tokens, see
[model-access.md](model-access.md)), and `truncated`. Pass the full history and the current tools every call; append-only
continuations reuse the sampled prefix. Everything about tokens is in
[model-access.md](model-access.md).

Check `truncated` before you use a reply. A generation cut short — by `max_tokens`,
inside reasoning, or inside a tool call — can be indistinguishable from a finished
turn: an empty `content` with no `tool_calls` is what both look like. Appending one
as the model's reply, or grading whatever it left behind, scores an answer the model
never gave. `truncated` folds those signals into one question; what to do with a cut
turn is yours, but it is never an answer.

If you decline a turn you have already sampled — a cut generation, a losing
candidate in best-of-n, a speculative probe — tell the conversation:

```python
reply = await conversation.generate(messages=history, tools=TOOLS, max_tokens=512)
if reply.truncated:
    conversation.discard(reply)          # then re-ask, or give up on the rollout
```

Sampling it recorded it, and everything recorded is trained. A rejected generation
left in place is trained with the rollout's advantage, and because chains are
covered by exact token prefix, it also splits the trajectory in two — so the datum
it lands in ends on the completion you rejected. `discard` removes it and restores
the sampled prefix the next turn extends, which is what lets a retry render one
clean chain. Only the most recent generation can be discarded, and the sampling
record is untouched: this decides what trains, not what happened.

A rollout whose agent discards every generation is still graded, and its reward still
counts in its group's baseline; it just has nothing to train. So a cut-off reply can
score 0 against the replies that finished without its long reasoning being pushed
down for running out of room (examples/sat does this). A group whose only nonzero
advantages fall on such rollouts is rejected as `nothing_to_train`.

`loops_cookbook.llm` publishes the shapes for the history you build: `SystemMessage`,
`UserMessage` and `ToolMessage`, plus `ToolSchema` for a tool declaration and
`Content` for a message body. The message shapes are `TypedDict`s and `Content` is a
type alias, so you keep writing plain dicts
and a type checker tells you when a tool result is missing its `tool_call_id`.
`AssistantMessage` is deliberately open — that one comes back from the renderer as
`native_message`, and what it carries differs by model family (GLM-5.3 has
`reasoning_content`), so append what you were given rather than building your own.
`Message` and `Tool` remain the broad types the renderer itself accepts.

Keep your own base class if you have one: implement `bind_loops_services(self,
services)` as a plain method that stores the attempt's services, and declare
`model_access = ModelAccessMode.DIRECT` on the class. The runtime calls the
setter before `setup`; `services.model_access.llm(producer)` then returns a client
locked to that producer. An agent that uses several producers asks once per
producer: `self.llm("critic")`. The shortcuts `self.conversation(producer="critic")`
and `await self.http_connection(producer="critic")` use the same attempt binding.
Every producer an agent names is declared in
`[rollout.producers]`, with whether it trains; `llm()` with no name is `main`, and a
name the table does not declare raises `ModelAccessError`.

Complete examples: [patterns/harbor/in_process_direct](../../patterns/harbor/in_process_direct/)
(one call, write an answer, runs on local Docker) and
[terminal_bench](../../examples/terminal_bench/) (multi-turn bash agent with
parallel tool dispatch and output truncation).

Images work the same way for vision models: put `{"type": "image", "image": <PNG or
JPEG bytes>}` parts in a user or tool message's content list alongside text parts. See
[renderers.md](renderers.md#images) for accepted shapes and which models render them.

Supported today: agent code running on the host, calling the model through
`self.llm()`, with the task executing in the sandbox; and an agent program run as a
process on the host, calling the model over HTTP. An in-process agent can also call
the model over HTTP through an OpenAI client
([patterns/harbor/in_process_openai_http](../../patterns/harbor/in_process_openai_http/)). For the second, declare
`model_access = ModelAccessMode.OPENAI_HTTP`, run the program with
`self.services.processes.run(...)`, and give it
`self.services.model_access.http_connection()` and the trial environment's
`usage_environment()`, so its tools act on the trial's
sandbox through the provider's API. The cookbook's Blaxel and Daytona environments
provide that usage environment. [patterns/harbor/local](../../patterns/harbor/local/) is the
complete example. Agents that run entirely inside the sandbox are not supported yet.

Over HTTP, every response the program received trains. A program that declines a
turn — a losing best-of-n candidate, a malformed tool call it re-asks for — posts
`{"id": "<completion id>"}` to `<base_url>/loops/discard` with the same key, which
does what `conversation.discard(reply)` does above;
[model-access.md](model-access.md#over-http) has the rules and errors.

## 3. Run

Copy [patterns/harbor/train_template.toml](../../patterns/harbor/train_template.toml) as `train.toml` next to
`job.yaml` and `my_agent.py`, set `base_model` and `[rollout_runtime.files] job`, put
`BASETEN_API_KEY` in a `.env` in the same directory, then:

```sh
uv run loops check train.toml     # resolves the dataset, imports the agent, builds the renderer
uv run loops run   train.toml     # provisions Loops resources (billable) and trains
uv run loops run   train.toml --training.max_steps 1 --training.group_size 2 --training.max_group_losses 0 --training.groups_per_step 1   # one-update smoke
```

The config's directory is importable, which is how `import_path: my_agent:MyAgent`
resolves. `[rollout_runtime.files]` and `resume_from` are relative to `train.toml`; `output_dir`
is relative to where you run, and so are the paths inside the job file, such as
`tasks[].path`, which Harbor resolves itself. `loops run --help` documents every setting, and
[`KNOB_GLOSSARY.md`](KNOB_GLOSSARY.md) groups them by table with the few that
only make sense together.

From Python:

```python
from loops_cookbook.project import load

project = load("train.toml", env_file=".env")
outcome = await project.train()
```

`load()` makes the config's directory importable, as `loops` does. Unlike `loops`, it
reads a `.env` only when you name one with `env_file`, which is relative to the config.
See [using the cookbook as a library](library.md) for the rest.

Output goes to `output_dir/<run_name>-<time>-<id>/`; layout and resume are described
in [training.md](training.md#outputs). By default the Loops run stays up when the
process exits: the run ID is logged with a warning, and on dedicated Loops
`reuse_from_run_id` reuses its warm pods next time. A dedicated run that stays up may still be billing. A serverless
run costs nothing while it waits, but holds one of the workspace's serverless slots.
With `[compute] keep_run = false`, the Loops run is deactivated when the process exits,
however the run ended.
