# Rollout runtime: Harbor

The Harbor rollout runtime trains your agent on the tasks of a
[Harbor](https://github.com/laude-institute/harbor) job. Use it when your tasks are
Harbor tasks, such as a Harbor dataset like Terminal-Bench. You bring a Harbor dataset,
define your agent as a `LoopsAgent` class, and run.

Harbor owns the tasks, the sandboxes they run in and the verifier that scores each
attempt. Harbor calls a sandbox an *environment*, as in a job's `environment:` key and
in class names such as `BlaxelEnvironment`. This page says *sandbox* except where it
names one of those. You own the agent. The cookbook owns everything between the
agent's model calls and the optimizer. When the sandboxes run on Blaxel or Daytona,
the run itself creates and owns every sandbox that Harbor starts, as
[Blaxel](#blaxel) and [Daytona](#daytona) below describe.

Harbor is an optional extra. Its version is bounded to `harbor>=0.23,<0.24`, because
the cookbook overrides some of Harbor's private members. The Harbor rollout runtime
was last run end to end with Harbor 0.23.0 on baseten-loops 0.28.0. The cookbook now
requires baseten-loops 0.34. Install the extra for the place your sandboxes will run:

```sh
uv sync --extra harbor      # Harbor on Docker
uv sync --extra harbor-blaxel  # Harbor on Blaxel sandboxes
```

## 1. Bring a Harbor dataset

Any Harbor job works. Its tasks can come from a dataset in a Harbor registry or a git
repository, or from local task directories. Write a normal Harbor `JobConfig`:

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

Harbor runs each attempt as a *trial*. A trial starts the task's sandbox, runs your
agent, and then runs the task's verifier, which scores what the agent did. A task's
verifier either shares the agent's sandbox or runs in a separate sandbox of its own.

The job's `environment:` key chooses where the sandboxes run. With `type: docker`
they run on local Docker. To run them on Blaxel or Daytona, use the cookbook's
environment class for that provider, as the next two sections show. The rules every
job must follow are in [What the job must satisfy](#what-the-job-must-satisfy).

### Blaxel

```sh
uv sync --extra harbor-blaxel
```

To run the job's sandboxes on Blaxel, name the cookbook's Blaxel class in the job's
`environment:` block:

```yaml
environment:
  import_path: loops_cookbook.harbor.blaxel:BlaxelEnvironment
  override_memory_mb: 24576   # optional: one size for every task
  kwargs:
    ttl: 8h                   # required
```

The `import_path` is required. `loops check` refuses `type: blaxel` on its own, and it
refuses Harbor's own Blaxel class, because with either one Harbor would create
sandboxes that the run never owns. The job must also set `kwargs.ttl`, the
time-to-live (TTL) for which Blaxel keeps a sandbox.

Then add a `[providers.blaxel]` table to the training TOML, with the region, and put
`BL_WORKSPACE` and `BL_API_KEY` in `.env`. The table's deadlines have defaults, which
[resources.md](resources.md#provider-resources) describes.

You state the region only in `[providers.blaxel]`. The run creates every sandbox in
that region, and `loops check` refuses a job that sets `environment.kwargs.region`.
The run builds images in that region too. Every build carries a `blaxel.toml` file
that names the region. When a task has its own `blaxel.toml` that names no region, the
run adds the region to the build's copy of that file. `loops check` refuses a task
whose `blaxel.toml` names another region.

Blaxel sizes a sandbox by its memory alone. A task's `memory_mb` sets that memory,
unless the job sets `override_memory_mb`, which gives every task the same size. The
disk is a tmpfs of half the memory, and the sandbox gets one CPU core per 2048 MB of
memory. Blaxel doesn't apply a task's `storage_mb` or `cpus`. `loops check` logs the
sandbox sizes, grouping tasks by memory. It warns about each task whose declared
`storage_mb` doesn't fit in that disk. It also warns about each task that declares
`cpus`, and about every task when the job sets `override_cpus`.

With the cookbook's class, the run creates every sandbox that a trial starts, under
Harbor's own name. That includes the agent's sandbox, a separate verifier's sandbox and
a regrade's sandbox. The run releases each one when Harbor stops it. If the run
crashes, you can recover its sandboxes in two ways. You can resume the run, which
releases what its previous process left before it admits an attempt. Or you can run
`loops cleanup`.

A job's `environment.delete: false` doesn't keep a sandbox. Whatever Harbor leaves
running is released when its attempt ends.

Harbor's own environment logs a failed stop and carries on. The cookbook's environment
raises from `stop()` instead whenever a release doesn't prove the sandbox gone, so the
attempt's cleanup reports the sandbox and tries again. The cookbook's Daytona
environment follows the same rule.

Harbor still runs commands and verification. The run's Blaxel provider builds the
images, unless a task starts from an image that is already in Blaxel, as
[Starting from an image already in Blaxel](#starting-from-an-image-already-in-blaxel)
describes. The provider reuses an image that is already built, and it never deletes an
image.

The run names an image by what it builds, not by the task, so tasks with identical
environments share one image and one build. It can do this only when Harbor's content
hash covers everything in the task's `environment/` directory. A symlink, an empty
directory or a file that Harbor's hash ignores means the hash doesn't describe the
build. Such a task keeps Harbor's own image name, which includes the task.

Harbor bounds each sandbox start by the task's effective build timeout. That timeout
is the task's `build_timeout_sec`, 600 s by default, times the job's
`environment_build_timeout_multiplier`, or times `timeout_multiplier` when that is
unset. `loops check` refuses a task whose effective build timeout doesn't cover the
provider's deadlines for these steps together:

- reconcile, which resolves the sandbox's prerequisites,
- image build,
- create,
- attach.

With the default deadlines, those steps add up to 630 s. That fits a 600 s build
timeout times 1.1, which is the `environment_build_timeout_multiplier` the shipped
Blaxel jobs set.

The check doesn't count the wait for a provisioning slot, which
`[rollout].max_concurrent_provisioning` limits, because that wait depends on how many
starts overlap. A start that runs out of time is a failed rollout, which the run can
retry under `[rollout].max_rollout_retries`.

The cookbook supports only Harbor's direct mode, in which a task's sandbox comes from a
Dockerfile or a `docker_image`. It doesn't support Harbor's compose mode.

#### Starting from an image already in Blaxel

A task can start from an image that you built yourself, instead of one that the run
builds. The task says so once, in its own `task.toml`:

```toml
[environment]
docker_image = "blaxel://sandbox/my-image:v1"
```

The run passes everything after `blaxel://` to Blaxel's sandbox create unchanged, so
any image reference that Blaxel accepts works. Nothing is built, so `force_build`
doesn't apply.

The task's `environment/` directory follows Harbor's rule for any task with a
`docker_image`. If the directory isn't empty and has no Dockerfile, its contents are
uploaded into the working directory once the sandbox starts. If it has a Dockerfile,
the Dockerfile is ignored and nothing is uploaded.

A `docker_image` without the `blaxel://` scheme keeps Harbor's meaning, which is a base
image that the run builds on. The scheme isn't a Docker reference, so neither form can
be mistaken for the other.

An imported image must carry Blaxel's sandbox API. Every Blaxel sandbox runs it as its
entrypoint, and every command and file operation in the sandbox goes through it. An
image that the run builds gets it from Blaxel's builder. An imported image skips that
build, so its Dockerfile must add the sandbox API, at a pinned version:

```dockerfile
COPY --from=ghcr.io/blaxel-ai/sandbox:<version> /sandbox-api /usr/local/bin/sandbox-api
ENTRYPOINT ["/usr/local/bin/sandbox-api"]
```

There are two ways to get the image into the workspace. You can push it with `bl push`,
as Blaxel's [custom images](https://docs.blaxel.ai/Sandboxes/Templates) guide
describes. Or you can have Blaxel import it from a registry with `POST /images`. That
request takes `name`, `resourceType: "sandbox"`, `image`, `tag`, and a `dockerConfig`
that holds the registry's pull credential.

Blaxel reserves the tag `latest`, and its tags are immutable. Run
`bl get image sandbox/<name> --latest` to see an image's tags. The reference to write
is `sandbox/<name>:<tag>`.

A separate verifier's sandbox, or a step verifier's, can name an imported image the
same way, in its own `[verifier.environment]`. `loops check` refuses a task that names
a `blaxel://` image in any of its sandboxes in these cases:

- The job isn't backed by Blaxel. A Daytona or local Docker job would pull the
  reference as a Docker image, and every trial would fail.
- The reference is `sandbox/<name>` with no tag.
- The reference is `sandbox/<name>:<tag>`, and the workspace has no image `<name>` or
  the image has no tag `<tag>`.

`loops run` makes the same check before it starts anything.

The check looks up each image once with `GET /images/sandbox/<name>/tags`, however
many tags name it, and only when a task names one. Blaxel looks images up by name only,
so the check can't verify a reference in any other form that Blaxel accepts. It also
can't verify an image that Blaxel doesn't answer for. The check logs each image that it
couldn't verify.

When every sandbox of a task starts from a `blaxel://` image, nothing is built. The
task's build timeout then doesn't need to cover the image build deadline, and
`loops check` doesn't check the task's own `blaxel.toml` for its region.

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

Add a `[providers.daytona]` table to the training TOML. It must set the default sandbox
size, `cpu`, `memory_gib` and `disk_gib`, and `ttl_minutes`. Its other keys have
defaults. Put `DAYTONA_API_KEY` in `.env`.

Daytona applies a task's `cpus`, `memory_mb` and `storage_mb` independently. A sandbox
that the run loses stops after it has been idle, is deleted as it stops, and is
destroyed when it reaches a wall-clock TTL.

The rules that apply on Blaxel apply here too. The `import_path` is required, the run
creates and releases every sandbox, and `loops check` refuses a build timeout that
doesn't cover the provider's reconcile, create and attach deadlines together.
[daytona.md](daytona.md) explains what the cookbook adds to Harbor's Daytona
environment, how sizing works, and how to choose between Blaxel and Daytona.

## 2. Define a `LoopsAgent`

Your agent is a Harbor agent, a `BaseAgent`, that subclasses the cookbook's
`LoopsAgent`. Before Harbor calls `setup`, the cookbook binds the attempt's services
to the agent. After that, `self.llm()` returns the attempt's model client. Declare the
agent's settings with a Harbor `AgentOptions` model, so that the job's
`agents[].kwargs` is validated against it.

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

`generate` renders the history with the model's chat template, samples a reply,
records the exact tokens and returns the reply. The reply has these fields:

- `content` and `reasoning_content` hold the reply's text and its reasoning.
- `tool_calls` holds the parsed tool calls. Each call has a `status`. Its `arguments`
  is a dict when the call parsed, and its `id` may be `None`.
- `stop_reason` says why the generation stopped, and `usage` counts its tokens.
- `parse_error` says why a tool call didn't parse.
- `native_message` is the assistant turn in the model's own format.
- `history_message` is the turn to append to your history. It is the native message,
  or the turn as the model wrote it when a call didn't parse. Append it as it is,
  because the next turn continues only from its sampled tokens.
  [model-access.md](model-access.md) explains why.
- `truncated` says whether the generation was cut short, as described below.

Pass the full history and the current tools on every call. When a call's history only
appends messages to the previous call's, the call reuses the tokens already sampled as
the start of its prompt. [model-access.md](model-access.md) covers everything about
tokens.

Check `truncated` before you use a reply. A generation can be cut short by
`max_tokens`, inside its reasoning or inside a tool call. A cut generation can look
exactly like a finished turn, because both can have an empty `content` and no
`tool_calls`. If you append a cut turn as the model's reply, or grade whatever it left
behind, you score an answer the model never gave. `truncated` combines those signals
into one flag. You decide what to do with a cut turn, but it is never an answer.

Sometimes you decline a turn that you have already sampled, such as a cut generation,
a losing candidate in best-of-n, or a speculative probe. Tell the conversation when you
do:

```python
reply = await conversation.generate(messages=history, tools=TOOLS, max_tokens=512)
if reply.truncated:
    conversation.discard(reply)          # then re-ask, or give up on the rollout
```

The conversation recorded the generation when it sampled it, and the run trains
everything that was recorded. If you leave a rejected generation in place, it trains
with the rollout's advantage. It also splits the trajectory in two. The run builds
training data from chains of generations, in which each generation's prompt starts with
the exact tokens of the one before it. So the rejected generation ends a chain, and the
training datum it lands in ends on the completion you rejected.

`discard` removes the generation and restores the sampled prefix that the next turn
extends, so a retry renders one clean chain. Only the most recent generation can be
discarded. The sampling record isn't changed, because `discard` decides what trains,
not what happened.

The run still grades a rollout whose agent discards every generation, and its reward
still counts in its group's baseline. The rollout only has nothing to train. So a reply
cut off by `max_tokens` can score 0 against the replies that finished, without pushing
down its long reasoning for running out of room. [examples/sat](../../examples/sat/)
does this. When a group's only nonzero advantages fall on such rollouts, the run
rejects the group as `nothing_to_train`.

`loops_cookbook.llm` exports types for the history you build. `SystemMessage`,
`UserMessage` and `ToolMessage` are the message shapes. `ToolSchema` is a tool
declaration, and `Content` is a message body. The message shapes are `TypedDict`s and
`Content` is a type alias, so you keep writing plain dicts, and a type checker tells you
when a tool result is missing its `tool_call_id`. `AssistantMessage` is deliberately
open. The assistant message comes back from the renderer as `native_message`, and what
it carries differs by model family. For example, a GLM-5.3 message has
`reasoning_content`. Append the message you were given rather than building your own.
`Message` and `Tool` are still the broad types that the renderer itself accepts.

You can keep your own base class instead of `LoopsAgent`. Implement
`bind_loops_services(self, services)` as a plain, synchronous method that stores the
attempt's services, and declare `model_access` on the class, such as
`model_access = ModelAccessMode.DIRECT`. The Harbor rollout runtime calls
`bind_loops_services` before `setup`. After that,
`services.model_access.llm(producer)` returns a client locked to that producer.

A producer is a named source of model calls. Every producer an agent names must be
declared in `[rollout.producers]`, together with whether its generations train. `llm()` with no name uses `main`, and a name the table
doesn't declare raises `ModelAccessError`. An agent that uses several producers asks
for a client once per producer, such as `self.llm("critic")`. The shortcuts
`self.conversation(producer="critic")` and
`await self.http_connection(producer="critic")` use the same attempt binding.

Two complete examples show a `LoopsAgent`.
[patterns/harbor/in_process_direct](../../patterns/harbor/in_process_direct/) makes one
model call, writes the answer, and runs on local Docker.
[terminal_bench](../../examples/terminal_bench/) is a multi-turn bash agent with
parallel tool dispatch and output truncation.

Images work the same way for vision models. Put
`{"type": "image", "image": <PNG or JPEG bytes>}` parts in the content list of a user
or tool message, alongside text parts. [renderers.md](renderers.md#images) lists the
accepted shapes and the models that render them.

### Agent placement

A Harbor agent can run in two places today:

- **In-process.** The agent is Python code that runs inside the training process and
  calls the model through `self.llm()`, while the task's commands run in the sandbox. An
  in-process agent can also call the model over HTTP through an OpenAI client, as
  [patterns/harbor/in_process_openai_http](../../patterns/harbor/in_process_openai_http/)
  shows.
- **Local.** The agent is a separate program that runs as a process on the training
  machine and calls the model over HTTP.

Remote agents, which run entirely inside the sandbox, aren't supported yet.

To run a local agent, declare
`model_access = ModelAccessMode.OPENAI_HTTP` and start the program with
`self.services.processes.run(...)`. Give the program two things. The first is
`self.services.model_access.http_connection()`, the attempt's connection to the model.
The second is the trial environment's `usage_environment()`, which lets the program's
tools act on the trial's sandbox through the provider's API. The cookbook's Blaxel and
Daytona environments provide that usage environment.
[patterns/harbor/local](../../patterns/harbor/local/) is the
complete example.

Over HTTP, every response the program received trains. A program can decline a turn,
such as a losing best-of-n candidate or a malformed tool call that it asks for again.
To decline it, the program posts `{"id": "<completion id>"}` to
`<base_url>/loops/discard` with the same key. That does what
`conversation.discard(reply)` does above.
[model-access.md](model-access.md#over-http) has the rules and the errors.

## 3. Run

Copy [patterns/harbor/train_template.toml](../../patterns/harbor/train_template.toml)
as `train.toml`, next to `job.yaml` and `my_agent.py`. In it, set
`[model].base_model`, and point `job` in `[rollout_runtime.files]` at your job
file. Put `BASETEN_API_KEY` in a `.env` file in the same directory. Then run:

```sh
uv run loops check train.toml     # resolves the dataset, imports the agent, builds the renderer
uv run loops run   train.toml     # provisions Loops resources (billable) and trains
uv run loops run   train.toml --training.max_steps 1 --training.group_size 2 --training.max_group_losses 0 --training.groups_per_step 1   # one-update smoke
```

The `loops` command makes the config's directory importable, which is how
`import_path: my_agent:MyAgent` resolves. Paths resolve from two places:

- The paths in `[rollout_runtime.files]` and `[training].resume_from` are relative
  to `train.toml`.
- `[output].output_dir` is relative to the directory you run from. So are the paths
  inside the job file, such as `tasks[].path`, because Harbor resolves them itself.

`loops run --help` documents every setting. [`KNOB_GLOSSARY.md`](KNOB_GLOSSARY.md)
lists the settings by table, and points to the few that only make sense together.

You can also run the config from Python:

```python
from loops_cookbook.project import load

project = load("train.toml", env_file=".env")
outcome = await project.train()
```

`load()` makes the config's directory importable, as `loops` does. Unlike `loops`, it
reads a `.env` file only when you name one with `env_file`, and that path is relative
to the config. [Using the cookbook as a library](library.md) covers the
rest.

The run writes its output to `output_dir/<run_name>-<time>-<id>/`.
[training.md](training.md#outputs) describes the layout and how to resume.

By default, the Loops run stays up when the process exits, and the cookbook logs the
run ID with a warning. On dedicated Loops, a run that stays up may still be billing,
and a later run's `[compute].reuse_from_run_id` can reuse its warm pods. A serverless
run costs nothing while it waits, but it holds one of the workspace's serverless slots.
With `[compute].keep_run = false`, the cookbook deactivates the Loops run when the
process exits, however the run ended.

## What the job must satisfy

A Harbor job must meet these rules to train:

- **The job must run the agent and an enabled verifier.** `loops check` rejects
  `install_only` jobs, regrade-only jobs and jobs that set `verifier.disable`.
- **The verifier must produce one numeric reward.** Its reward map must have a single
  entry, or you name the entry to train on with `[rollout_runtime.config].reward_key`.
  A boolean reward, a reward that isn't finite and a missing reward are failures, not
  zeros.
- **The cookbook sets Harbor's trial cap.** The cap is `max_concurrent_rollouts`, plus
  the pre-start places, plus `max_concurrent_verifiers`. The pre-start places let a few
  trials start their sandbox before a sampling slot is free.
  [scheduling.md](scheduling.md) explains all three. Setting `n_concurrent_trials` in
  the job file has no effect, because only the cookbook knows these numbers.
- **The grading deadline must exceed Harbor's own.** `loops check` refuses a job whose
  `[rollout.grading].timeout_seconds` doesn't exceed Harbor's own bound for one
  `grade()` call. For a separate verifier with R retries, that bound is
  `(R+1) x (build + verifier)`, plus Harbor's backoff between retries.
- **Remote datasets download at `loops check`.** A private dataset uses your own git
  credentials or `harbor auth login`. The cookbook adds nothing.

## Grading and failures

### Shared and separate verifiers

A task whose verifier shares the agent's sandbox is graded live, in that sandbox.

A single-step task whose verifier runs in its own sandbox is graded from artifacts,
because the cookbook separates its sampling from its grading. The run releases the
agent's sandbox as soon as the agent finishes. When the job sets
`environment.delete: false`, it releases the sandbox when the attempt ends instead.
Harbor then regrades the trial from what it recorded. Anything the verifier needs must
therefore be declared as a task artifact.

A multi-step task keeps Harbor's step order, and the cookbook doesn't separate its
grading from its sampling.

### When the agent fails

A failed agent is graded, as Harbor grades it. The agent can fail by timing out, with
`AgentTimeoutError`, or by exiting with an error, with `NonZeroAgentExitCodeError` or
any of its subclasses. Those subclasses include the API errors that Harbor's installed
agents raise. Either way, Harbor still runs the verifier, and the trial trains on its
reward. So an agent that runs out of time earns its low reward.

A separate-verifier trial is regraded from the saved trial exactly when Harbor carried
the trial on to its verification. A trial that Harbor ended before then is a rollout
failure.

When the deadline cuts a generation off mid-flight, that generation is dropped from
the evidence, and the rest of the episode trains.

The verifier never sees the trial's exception. A shared verifier sees the sandbox, and
a separate verifier sees the saved trial's artifacts. An agent can leave evidence in
either place for a custom verifier to read, but the verifier can't see a timeout or an
error that Harbor enforced.

The Harbor rollout runtime never discards a failed agent's episode, and what the
verifier scores stands. Only the cookbook's own rollout clock discards an episode, as
[failures.md](failures.md) explains. The run reports how often each exception
happened as `runtime/harbor/trial_exceptions/<type>`, which you can compare with the
number of trials in `runtime/harbor/sampling_trials`.

### When a sandbox runs out of memory or disk

A sandbox that runs out of memory or disk is a sizing question. Its memory and disk
are the sizes that the task or the job set. The cookbook doesn't reinterpret the
trial, and Harbor's result stands.

On Blaxel and Daytona, each trial reports its sandbox's out-of-memory (OOM) kill
count, which the run reads when the agent phase ends. Where the count comes from
depends on the provider:

- A Blaxel sandbox is a micro-VM, so the count is its own kernel's, from
  `/proc/vmstat`.
- A Daytona sandbox is a container on a shared host kernel, so the count is its
  cgroup's, from `memory.events`.

The run reports the count as `runtime/harbor/sandbox_oom_kills/{count,mean,max}`, and
`runtime/harbor/oom_killed_trials` counts the trials that had any OOM kill. The count
is only a metric. The run leaves the agent's metadata as the
agent wrote it. When the count can't be read, it isn't reported.

### Other errors

Any other error fails the rollout. Harbor never retries a sampling trial itself. An
exception that is raised before sampling ends, and that leaves no reward, is a rollout
failure. It is permanent if its type is in the job's `retry.exclude_exceptions`.
Otherwise the run may retry it under `[rollout].max_rollout_retries`. Either way, it
counts toward `[rollout].max_consecutive_rollout_failures`.

A shared verifier's failure surfaces in `grade()` instead, as
`grading_failed/<type>`. The cookbook doesn't retry it, because the Harbor rollout
runtime's `grading_retries` is 0. The job's `retry` block applies only to
separate-verifier regrades.

### Dropping a reward with a reward gate

A reward gate drops a reward from training. It never turns the reward into a zero. Name
the gate in `[rollout_runtime.config].reward_gate` as `module:attribute`. The gate is an
async function `(trial, verifier_result) -> str | None`:

- `None` keeps the reward.
- A reason drops the trial from training. The run reports the drop as
  `grading_failed/<reason>`, so a reason may hold only lowercase letters, digits and
  underscores.

The run calls the gate once the reward is final, before it stops the sandboxes that are
still running. For a task with a separate verifier, the agent's sandbox is already
released by then.

A dropped trial doesn't count toward the consecutive-failure stop. Its reward stays in
the trial directory, so you can still read it.

A gate that raises fails grading as `grading_failed/reward_gate`. A gate that returns
anything other than a reason or `None` fails grading as
`grading_failed/reward_gate_result`.

The gate runs inside `grade()`, even for a multi-step trial that stopped early, and the
run holds the trial's sandbox until the gate answers. The gate is therefore bounded as
`grade()` is. A gate still running past the grading deadline fails as
`grading_failed/hook_deadline`, and so does a gate that blocks the event loop and
answers after the deadline. A gate that ignores its cancellation is handled like any
`grade()` that does, and is reported as `unquiesced/grade`.

## Resuming

A checkpoint records each task's ID and Harbor content digest, and the digest of the
job file. If you resume after editing a task or changing the set of tasks, the run
restarts the task order from the beginning, with a warning. A changed job file is
reported as a changed setting.
