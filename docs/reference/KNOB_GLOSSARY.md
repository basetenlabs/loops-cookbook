# Knob glossary

Every setting the cookbook reads. Settings live in `train.toml` under the section
shown, and the CLI exposes each key of a table that is always present by its dotted
path, such as `--rollout.max_concurrent_rollouts`.

A rollout runtime's own settings are opaque to the training config and live in
`[rollout_runtime.config]`; the built-in rollout runtimes' and providers' tables are listed after
the training config's. For Harbor, `[rollout_runtime.config]` is only `reward_key`, and
agent behaviour and environment configuration belong to the Harbor job. A custom
provider defines its own `[providers.<name>]`. For the settings that only make sense
together, see [`settings-that-interact.md`](settings-that-interact.md).

**Defaults or required.** A setting that changes what trains or what a run costs is
required, such as `learning_rate`, `group_size` or a sandbox's size. A setting has a
default only when it is purely operational: a wrong value fails loudly and nothing
about training depends on it, such as the `[rollout.cleanup]` graces. A setting the
platform has its own default for, such as `adam_beta2`, is optional with no value of
ours: unset, the request carries the Loops SDK's default, which a `baseten-loops`
release may change. An opt-in feature, such as `router_replay`, is off unless set. A
new setting follows the same rule.

<!-- generated from the settings models; `python -m loops_cookbook.docs` rewrites it -->

## `[rollout_runtime]`

Where episodes come from: the rollout runtime factory and what it may resolve.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `factory` | string | *required* | Rollout runtime factory as module:callable. Harbor: loops_cookbook.harbor:create_runtime; a custom rollout runtime: its module beside this file. |
| `model_access` | string | `None` | How the agent reaches the model. direct: the cookbook's llm() client, for in-process agents; openai_http: an OpenAI-compatible Chat Completions endpoint per attempt. Unset: the one the rollout runtime declares (an Episode class's model_access, a Harbor job's agents'); set only for a rollout runtime that supports several. |
| `openai` | table | `None` | The Chat Completions gateway; only with model_access = openai_http. |
| `config` | table | empty | Opaque rollout runtime settings, handed to the factory unchanged. |
| `files` | table | empty | Files the rollout runtime may resolve, as alias = path. Relative to this config file, or to the shell when passed as a command-line flag. |
| `has_images` | boolean | `False` | Whether the agent's prompts carry images. Loops cannot record an image prompt, so router replay needs this false. |

## `[rollout_runtime.openai]`

The Chat Completions gateway, with model_access = openai_http.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `bind` | string | `"127.0.0.1:0"` | Where the gateway listens, as host:port; loopback and an ephemeral port by default. |
| `advertised_base_url` | string | `None` | The origin agents use, such as http://127.0.0.1:43210; default is the bound loopback address. |
| `model_name` | string | `None` | The one model the gateway advertises; default is [model].base_model. |

## `[model]`

The policy being trained and the compute it is configured for.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `base_model` | string | *required* | Loops base model, e.g. Qwen/Qwen3.6-35B-A3B. |
| `tokenizer` | string | `None` | Where the policy's tokenizer loads from: a Hugging Face repo id, or a local directory starting with ./, / or ~ (relative to this file); default is base_model. For a model Loops serves under a name with no Hugging Face repo. A private repo reads HF_TOKEN. |
| `lora_rank` | integer, ≥ 1 | *required* | LoRA rank for the trained adapter. |
| `max_seq_len` | integer, ≥ 1 | `None` | Per-sequence token budget for the trainer. RL defaults to the model's. SFT requires it, so that check refuses a row longer than this before anything is provisioned. |
| `renderer` | string | `None` | Renderer factory as module:callable; default is the cookbook's renderer for the models it renders (docs/reference/renderers.md), otherwise Prime auto-detection from the tokenizer. |
| `renderer_chat_template_kwargs` | table | empty | Model-specific renderer options (JSON on the command line). |
| `reasoning_efforts` | table | empty | Accepted reasoning_effort values, each mapped to the template options it overrides; a request selects one, and an undeclared value is rejected. |

## `[sampling]`

Defaults for what the policy produces. Agents inherit these and state only what they need to differ on.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `temperature` | number, ≥ 0 | `1.0` | Default sampling temperature for a generation that names none; a request may set its own. |
| `max_generation_tokens` | integer, ≥ 1 | *required* | Default length limit for a generation that names none; a request may set its own, so this is not a cap. Either is lowered to the room the trajectory budget leaves. |
| `max_trajectory_tokens` | integer, ≥ 1 | `None` | Trajectory budget, in tokens, for prompt plus generation: each call's max_tokens is lowered to the room its prompt leaves, and only a prompt that leaves none is refused. Over HTTP it always applies; a direct call may pass its own. Defaults to max_seq_len, which is the capacity the trainer was configured for. |

## `[training]`

The optimizer, the batch, and what data is acceptable to train on.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `seed` | integer | `None` | The Loops trainer's random seed. Unset: Loops chooses. |
| `max_steps` | integer, ≥ 0 | *required* | Stop once the run has made this many optimizer updates, the ones before a resume included. 0 runs without a limit: RL until stopped, SFT every epoch. |
| `learning_rate` | number, > 0 | *required* | Adam learning rate; under SFT's lr_schedule, its peak. |
| `grad_clip_norm` | number, ≥ 0 | `None` | Gradient norm clipping; 0 disables it. Unset: the Loops SDK's. |
| `adam_epsilon` | number, > 0 | `None` | Adam epsilon. Unset: the Loops SDK's. |
| `adam_beta1` | number, ≥ 0 | `None` | Adam beta1. Unset: the Loops SDK's. |
| `adam_beta2` | number, ≥ 0 | `None` | Adam beta2. Unset: the Loops SDK's. |
| `weight_decay` | number, ≥ 0 | `None` | Adam weight decay. Unset: the Loops SDK's. |
| `resume_from` | string | `None` | A checkpoints/step-N.json manifest from a prior run (for SFT, of the same data), or that run's directory, which resumes from the manifest it wrote last. Relative to this config file, or to the shell when passed as a command-line flag. |
| `init_from` | string | `None` | Loops trainer weights a fresh run starts from, such as bt://loops:<run>/weights/step-N: the LoRA adapter, with fresh Adam moments; the run's steps count from 0, the trainer's policy version continues from the checkpoint's. Same base model and LoRA rank. A resume ignores it: the resumed state already holds those weights. |
| `init_from_optimizer` | boolean | `False` | Also load init_from's optimizer state (Adam moments and step): a fork that continues the source run's optimizer under this run's config. The moments reflect the source run's gradient scale; from a checkpoint saved without them Loops loads the weights only. Requires init_from; a resume ignores it. |
| `mode` | `sync` \| `async` | `"sync"` | sync: one batch at a time. async: overlap collection and training. |
| `group_size` | integer, ≥ 2 | *required* | Rollouts sampled per task group. |
| `max_group_losses` | integer, ≥ 0 | *required* | Rollouts a group may lose and still train. A tolerance rather than a floor, so it keeps its meaning when group_size changes. |
| `groups_per_step` | integer, ≥ 1 | *required* | Groups per optimizer update. |
| `loss_fn` | `importance_sampling` \| `ppo` \| `cispo` \| `dro` \| `dppo` \| `custom` | *required* | The loss forward_backward computes: one of Loops', or custom, your own from [training.custom_loss], run through forward_backward_custom. |
| `loss_fn_config` | table | empty | The built-in loss's settings, handed to Loops as they are; a key left unset keeps Loops' default. importance_sampling: none; ppo: clip_low_threshold (0.8), clip_high_threshold (1.2); cispo: clip_low_threshold (0.8), clip_high_threshold (1.2); dro: beta (0.05); dppo: dppo_mask_high (0.2), dppo_mask_low (0.2), adv_tau (1), kl_tau (0.001). |
| `custom_loss` | table | `None` | Your own loss; required with loss_fn = "custom". |
| `loss_aggregation` | `sum` \| `token_mean` \| `fixed_budget` \| `task_mean` | `"sum"` | Built-in loss reduction over a complete optimizer update. sum is Loops' default: no division; token_mean divides by the update's action tokens with nonzero advantage; fixed_budget divides by group_size * groups_per_step * loss_token_budget; task_mean averages each task group's action-token mean. Custom losses reduce in their own function and keep sum. |
| `loss_token_budget` | integer, ≥ 1 | `None` | Action-token budget per whole episode for fixed_budget aggregation, across all turns; required only with loss_aggregation = fixed_budget. A normalization constant, not a generation limit. |
| `router_replay` | boolean | `False` | Replay eligible batches' expert routes (R3); off by default. |
| `max_consecutive_unusable_groups` | integer, ≥ 0 | `16` | Abort after this many consecutive groups rejected for training; 0 disables. Independent of infrastructure failures: it catches a wave of agent failures graded as zeros. |
| `max_steps_off_policy` | integer, ≥ 0 | `None` | Async: oldest policy version, in updates, a rollout may train on. Required with mode = async; sync trains only on the current policy and refuses it. |
| `admission_steps_ahead` | number, > 0 | `None` | Async: batches beyond the one training that admission starts groups for, at most max_steps_off_policy, which stays the staleness cap; fractions admit part of a batch. Lower, groups wait less for the trainer and train closer to the current policy, while one that runs long still trains up to the cap, its finished members' slots lent to new groups meanwhile. Unset: max_steps_off_policy. |
| `max_policy_delta_in_trajectory` | integer, ≥ 0 | `None` | Async: policy-version spread allowed inside one rollout. Required with mode = async; sync samples each rollout from one policy and refuses it. |

## `[training.custom_loss]`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `factory` | string | *required* | Loss factory as module:callable, such as a module beside this file; it takes config and returns the loss function forward_backward_custom runs. Needs the custom-loss extra. |
| `config` | table | empty | The loss's own settings, handed to the factory unchanged. |

## `[rollout]`

How much runs at once, how long each episode may take, and when to give up.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `max_concurrent_rollouts` | integer, ≥ 1 | *required* | Concurrent rollouts holding sampling capacity. |
| `max_concurrent_verifiers` | integer, ≥ 1 | *required* | Concurrent rollouts holding verification capacity. |
| `prestart_allowance` | number, ≥ 0 | `0.1` | Admitted rollouts that may start their environment before a sampling slot is free, as a fraction of max_concurrent_rollouts: enough ready environments that a freed slot is taken at once, without paying for many idle. |
| `max_concurrent_provisioning` | integer, ≥ 1 | `8` | Run-wide limit on provider operations in flight: creates, reconciliation, releases, and absence checks. Also where each provider's create window starts, and its cap: refusals narrow it. |
| `max_consecutive_rollout_failures` | integer, ≥ 0 | `64` | Abort after this many consecutive failures in one stage: sampling, grading and cleanup infrastructure, or a group grader; 0 disables. |
| `shuffle_seed` | integer | `0` | Seed for the task order. |
| `shutdown_drain_seconds` | number, > 0 | `120` | Time allowed for in-flight attempts to settle on shutdown, and again for the run's resource creates and releases to finish; what is left is cut off, given a few seconds to stop, and reported for `loops cleanup`. An operation that ignores being cut off is reported, and holds the process until it stops. |
| `producers` | table | `{'main': {'trainable': True}}` | Every producer a rollout runtime may use, by name, and whether its generations receive gradient; a call that names no producer uses main, and an undeclared name is refused. Setting the table replaces the default, so list main if calls rely on it. |
| `rollout_timeout_seconds` | number, > 0 | `None` | Backstop on a rollout's time holding sampling capacity, less resource-service time such as a sandbox create; a rollout runtime bounds its own agent below it. Unset: the rollout runtime's derived clock (Episodes derive one from their agent deadline); required for a rollout runtime that derives none. |
| `setup_timeout_seconds` | number, > 0 | *required* | Bound on an attempt's time before sealing outside sampling and verification capacity. Required. |
| `max_rollout_retries` | integer, ≥ 0 | `0` | Extra attempts a rollout may use after an episode failure before it has a valid rollout. A last resort: rollout runtimes retry their own steps, and model-call failures are never retried here because the Loops SDK already retried them. |

## `[rollout.producers.<name>]`

One producer a rollout runtime may name, keyed by that name.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `trainable` | boolean | *required* | Whether this producer's generations receive gradient. |

## `[rollout.grading]`

Each grade() try's deadline; how many tries a failure earns is the rollout runtime's.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `timeout_seconds` | number, > 0 | *required* | Deadline for each grade() try, and each group grader call. Required. |

## `[rollout.cleanup]`

Deadlines for the rollout runtime's cleanup hooks.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `timeout_seconds` | number, > 0 | `600` | Deadline for each finish(), abort(), and close(). Generous: it covers a Harbor trial's teardown, and one set too long only delays the report of a hook that hangs. |
| `cancellation_grace_seconds` | number, > 0 | `5` | How long a cancelled rollout runtime hook may take to stop before the run fails. |
| `process_kill_grace_seconds` | number, > 0 | `10` | How long an agent process may take to exit after SIGTERM before it is killed; spent before a cancelled hook's own grace starts. |

## `[compute]`

Where the run executes. Nothing here changes what is learned.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `scope` | `serverless` \| `dedicated` | *required* | Where the run's trainer, samplers and evals run. serverless: Loops' shared capacity, billed for tokens; a model or configuration it cannot serve fails the run's creation. dedicated: deployments of the run's own, billed for GPU time until the run is deactivated. |
| `spot_job` | boolean | `False` | Use reclaimable Loops spot capacity; interrupted runs stop. Dedicated only. |
| `reuse_from_run_id` | string | `None` | Reuse a prior run's warm pods with fresh LoRAs; does not restore a checkpoint. Dedicated only. |
| `keep_run` | boolean | `True` | Leave the Loops run up when the process exits. A dedicated run bills until it is deactivated with loops deactivate, and a later dedicated run's reuse_from_run_id can start warm on it; a serverless run holds a workspace slot. false deactivates the run on exit; its checkpoints stay. |
| `trainer_replicas` | integer, ≥ 1 | `1` | Parallel copies of the trainer. More raise update throughput at proportionally more GPUs; above 1 always provisions fresh capacity. Dedicated only. |
| `sampler_replica_affinity` | boolean | `True` | Pin a rollout's requests to one sampler replica so its growing prompt keeps hitting a warm prefix cache. A routing hint: where Loops cannot pin a request, it is served without affinity. |
| `sampler_replicas` | integer, ≥ 1 | `None` | Fixed replica count for the paired sampler. Unset: the Loops backend default. Dedicated only. |

## `[output]`

What the run leaves behind.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `run_name` | string | *required* | Prefix for the run directory and the Loops run name. Required, so that two configs in one output_dir do not share run directories: `loops run --resume` takes the latest one of this name. |
| `output_dir` | string | `"runs"` | Where run directories are created. |
| `wandb_project` | string | `None` | Mirror metrics to this W&B project (best effort). |
| `console` | `auto` \| `plain` \| `rich` | `"auto"` | Terminal output style. auto: coloured logs, and a live progress region where the recipe has one, when the log stream is a terminal and the console extra is installed, plain otherwise. Redirected output is always plain, so a captured log never contains control characters. |
| `trainer_checkpoint_every` | integer, ≥ 0 | *required* | Save resumable trainer state every N updates; 0 disables. The last update is always saved: the one at max_steps, and on the way out of a run that ends any other way (bounded, best effort). |
| `sampler_checkpoint_every` | integer, ≥ 0 | *required* | Stage servable adapter weights every N updates; 0 disables. The update at max_steps is always saved. |
| `checkpoint_ttl_seconds` | integer, ≥ 0 | *required* | Remote checkpoint retention, in seconds; 0 keeps them. |
| `progress_every_seconds` | number, ≥ 0 | `60` | Seconds between progress reports while a batch collects; 0 is off. Optimizer steps can be tens of minutes apart and nothing else reports in between, so without it a long collection phase and a stall look identical. Every slots, queue and admission series in W&B comes from these reports. |
| `rollout_metrics_limit` | integer, ≥ 1 | `32` | Most rollout/* metrics a step reports from what the agents return about their episodes, counting each label value as one. Past it, the keys the fewest rollouts reported are dropped (rollout/dropped_keys counts them), so a stray per-rollout id cannot bury the rest; raise it when an agent reports more. |
| `trace` | `off` \| `samples` \| `transcripts` \| `full` | `"off"` | Per-generation evidence to keep beside the run record. off: none. samples: one row per generation without token arrays, about 1% of full. transcripts: those rows and each attempt's transcript (every generation's request as the agent sent it and the reply it received, marked with whether it is trainable), to read what the policy saw and said. full: transcripts, token arrays and the exact training rows, for replay and audit. Every attempt's transcript is written, a failed one's as far as it got, and recording never changes what trains: a request or reply that is not JSON (bytes or a set in a direct agent's history) is written as unrecordable. |

## `[resources]`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `state_dir` | string | `"state"` | Directory of the resource journal, relative to the run directory. It must be on a local filesystem. |

## `[eval]`

Held-out evals during training, each on a sampler pinned to one step's weights. Nothing they sample trains.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `every_steps` | integer, ≥ 0 | `0` | Evaluate every N updates; 0 evaluates only before the first update and after the last. After the last needs [training] max_steps. |
| `sampler_start_timeout_seconds` | number, > 0 | `900` | How long an eval's sampler may take to be ready before that set's eval is skipped. On dedicated Loops each eval starts a sampler deployment. |
| `sets` | table | empty | Held-out task sets, by name. None: no evals. |

## `[eval.sets.<name>]`

One held-out task set: training's rollout runtime with these files. A key at the path of a training setting overrides that setting for this set.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `samples_per_task` | integer, ≥ 1 | `1` | Rollouts per task. |
| `max_tasks` | integer, ≥ 1 | `None` | Evaluate the first N tasks; unset: all. |
| `max_concurrent_rollouts` | integer, ≥ 1 | `16` | This set's rollouts at once, on top of training's: they share training's sandbox providers, so those see training's limit plus every set's. |

## `[eval.sets.<name>.rollout_runtime]`

Overrides [rollout_runtime] for this set.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `files` | table | *required* | Replaces these [rollout_runtime.files] entries for this set; the rollout runtime, its config and every other file are training's. |

## `[eval.sets.<name>.sampling]`

Overrides [sampling] for this set.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `temperature` | number, ≥ 0 | `None` | Unset: [sampling].temperature. |

## `[rollout_runtime.config]` for Episodes

With factory = "loops_cookbook.episodes:create_runtime". Its tasks file is the `tasks` alias under `[rollout_runtime.files]`.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `episode` | string | *required* | The Episode subclass, as module:Class. |
| `agent_timeout_seconds` | number, > 0 | *required* | How long run() may take before it is cut and graded. |
| `sandbox` | table | `None` | One sandbox per attempt, provisioned before setup(). |
| `grading_sandbox` | table | `None` | Optional separate sandbox, provisioned on first grading admission. |
| `options` | table | `None` | The episode's own non-secret settings, validated with its options model. |

## `[rollout_runtime.config.sandbox]` for Episodes

An Episode's sandbox. The keys beside `provider` are the provider's sandbox spec, below for Blaxel and Daytona.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `provider` | string | *required* | The [providers] entry that provides it. |

## `[rollout_runtime.config.sandbox]` on Blaxel

Beside provider = "blaxel".

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `image` | string | *required* | The image it runs. |
| `memory_mb` | integer, > 0 | *required* | Its memory, as Harbor's memory_mb. |
| `ttl` | string | *required* | How long Blaxel keeps it, such as 1h, required as Harbor's Blaxel ttl is: a backstop, as the run releases it. A whole number and one unit: s, m, h, d or w. |

## `[rollout_runtime.config.sandbox]` on Daytona

Beside provider = "daytona". Its lifecycle is `[providers.daytona]`'s.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `image` | string | `None` | The registry image it runs, such as python:3.12-slim. Unset, Daytona's default sandbox image. |
| `snapshot` | string | `None` | A Daytona snapshot it starts from, by name, instead of an image. The snapshot fixes both its image and its size. |
| `cpu` | integer, > 0 | `None` | Its vCPUs. Unset, [providers.daytona] cpu. |
| `memory_gib` | integer, > 0 | `None` | Its memory in GiB. Unset, [providers.daytona] memory_gib. |
| `disk_gib` | integer, > 0 | `None` | Its disk in GiB, the image not counted. Unset, [providers.daytona] disk_gib. |

## `[rollout_runtime.config.grading_sandbox]` for Episodes

An Episode's optional separate verifier sandbox, provisioned only on grading admission. The keys beside `provider` are the provider's sandbox spec.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `provider` | string | *required* | The [providers] entry that provides it. |

## `[rollout_runtime.config.grading_sandbox]` on Blaxel

Beside provider = "blaxel"; independent of the execution sandbox spec.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `image` | string | *required* | The image it runs. |
| `memory_mb` | integer, > 0 | *required* | Its memory, as Harbor's memory_mb. |
| `ttl` | string | *required* | How long Blaxel keeps it, such as 1h, required as Harbor's Blaxel ttl is: a backstop, as the run releases it. A whole number and one unit: s, m, h, d or w. |

## `[providers.blaxel]`

The built-in Blaxel provider. Its credentials, BL_WORKSPACE and BL_API_KEY, come from the environment or a .env beside the config, never from this table.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `region` | string | *required* | The region every sandbox is created and every image is built in. Required. |
| `create_timeout_seconds` | number, > 0 | `120` | Bound on creating one sandbox, until Blaxel reports it deployed. |
| `reconcile_timeout_seconds` | number, > 0 | `60` | Bound on looking an ambiguous create up by name, and on deciding prerequisites. |
| `image_build_timeout_seconds` | number, > 0 | `420` | Bound on building one sandbox image, builder sandbox included. |
| `image_build_memory_mb` | integer, ≥ 1, ≤ 32768 | `None` | Memory of the environment each image is built in, in MB. Unset: Blaxel's default, which a large image can run out of mid-build. It sizes the build alone; a sandbox's memory is its task's. |
| `image_build_volume_mb` | integer, ≥ 0, ≤ 131072 | `None` | Disk for each build's intermediate layers, in MB; 0 builds in memory. Unset: Blaxel's default, which a large image can run out of mid-build. |
| `release_timeout_seconds` | number, > 0 | `120` | Bound on deleting one sandbox. |
| `confirm_absent_timeout_seconds` | number, > 0 | `60` | Bound on confirming a deleted sandbox is gone. |
| `list_timeout_seconds` | number, > 0 | `60` | Bound on each page of the workspace listing a label sweep reads; the whole listing is not bounded, since it grows with the workspace. |
| `attach_timeout_seconds` | number, > 0 | `30` | Bound on attaching a client to a created sandbox. |
| `labels` | table | empty | Labels every sandbox, image builders included, carries besides the run's ownership labels, such as an owner to find a team's sandboxes by. They win over the same keys from a Harbor job's or [rollout_runtime.config.sandbox]'s labels. |

## `[providers.daytona]`

The built-in Daytona provider; see [daytona.md](daytona.md). Its credential, DAYTONA_API_KEY (and DAYTONA_API_URL for a self-hosted Daytona), comes from the environment or a .env beside the config, never from this table.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `target` | string | `None` | The Daytona target (region), such as us or eu, of every sandbox and every Daytona call, Harbor's included. Unset, the organization's default target. A DAYTONA_TARGET in the environment that differs is refused. |
| `create_timeout_seconds` | number, > 0 | `600` | Bound on creating one sandbox until Daytona reports it started, a first build of its declarative image included. |
| `reconcile_timeout_seconds` | number, > 0 | `60` | Bound on looking an ambiguous create up by name. |
| `release_timeout_seconds` | number, > 0 | `120` | Bound on deleting one sandbox. |
| `confirm_absent_timeout_seconds` | number, > 0 | `60` | Bound on confirming a deleted sandbox is gone. |
| `list_timeout_seconds` | number, > 0 | `60` | Bound on each page of the labelled listing a label sweep reads. |
| `attach_timeout_seconds` | number, > 0 | `30` | Bound on attaching a client to a created sandbox. |
| `cpu` | integer, > 0 | *required* | vCPUs of a sandbox whose request names none. |
| `memory_gib` | integer, > 0 | *required* | Memory in GiB of a sandbox whose request names none. |
| `disk_gib` | integer, > 0 | *required* | Disk in GiB of a sandbox whose request names none: its writable layer only, as the image is not counted. |
| `thread_variables_from_vcpus` | boolean | `False` | Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to a Harbor sandbox's vCPUs for every command, as nproc in a sandbox counts its host's cores. Harbor passes them with each command, so they override the image's ENV; the task's [environment.env] wins. |
| `auto_stop_minutes` | integer, ≥ 0 | `120` | Minutes without a Daytona API call (a command counts; a process inside does not) after which Daytona stops a sandbox whose request sets none; with auto_delete_minutes 0 it is then deleted. It must outlast an attempt's longest quiet spell, such as a long model turn or a wait for grading. 0 disables it, which needs ttl_minutes. |
| `auto_delete_minutes` | integer, ≥ 0 | `0` | Minutes a stopped sandbox is kept before Daytona deletes it; 0 deletes it as it stops. |
| `ttl_minutes` | integer, ≥ 0 | *required* | Wall-clock minutes after which Daytona destroys a sandbox whatever its state: a backstop for one the run lost. It must outlast the longest attempt, as a sandbox still working at it is destroyed; 0 sets none. |
| `labels` | table | empty | Labels every sandbox carries besides the run's ownership labels, such as an owner to find a team's sandboxes by. They win over the same keys from a Harbor job's or [rollout_runtime.config.sandbox]'s labels. |
| `max_sandbox_cpu` | integer, > 0 | `None` | The most vCPUs Daytona gives one sandbox; a larger task is refused before the run. |
| `max_sandbox_memory_gib` | integer, > 0 | `None` | The most memory one sandbox may have, in GiB. |
| `max_sandbox_disk_gib` | integer, > 0 | `None` | The most disk one sandbox may have, in GiB. |

# SFT settings

An SFT config, `recipe = "sft"`, has its own TOML; `[model]`, `[compute]` and
`[output]` are the RL recipe's, spelled the same, except that SFT requires
`max_seq_len`.

## SFT `[model]`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `base_model` | string | *required* | Loops base model, e.g. Qwen/Qwen3.6-35B-A3B. |
| `tokenizer` | string | `None` | Where the policy's tokenizer loads from: a Hugging Face repo id, or a local directory starting with ./, / or ~ (relative to this file); default is base_model. For a model Loops serves under a name with no Hugging Face repo. A private repo reads HF_TOKEN. |
| `lora_rank` | integer, ≥ 1 | *required* | LoRA rank for the trained adapter. |
| `max_seq_len` | integer, ≥ 1 | *required* | Per-sequence token budget for the trainer. RL defaults to the model's. SFT requires it, so that check refuses a row longer than this before anything is provisioned. |
| `renderer` | string | `None` | Renderer factory as module:callable; default is the cookbook's renderer for the models it renders (docs/reference/renderers.md), otherwise Prime auto-detection from the tokenizer. |
| `renderer_chat_template_kwargs` | table | empty | Model-specific renderer options (JSON on the command line). |

## SFT `[data]`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `format` | `chat` \| `tokens` | `"chat"` | The contract every row meets. chat: OpenAI chat fine-tuning JSONL, rendered with the model's template, which decides the targets. tokens: input_ids and 0/1 weights per row, trained as given, with no checks that the targets are the model's own tokens. |
| `train` | string | *required* | Training rows as JSONL in [data] format, one per line. Relative to this config file, or to the shell when passed as a command-line flag. |
| `validation` | string | `None` | Held-out rows in the same format, scored for validation NLL. Resolved like train. Set this or validation_fraction, or neither for no validation. |
| `validation_fraction` | number, > 0 | `None` | Hold out this fraction of the training rows, chosen by shuffle_seed, for validation instead of a separate file. |
| `shuffle_seed` | integer | `0` | Seed for the held-out rows and for each epoch's order; epoch e shuffles with shuffle_seed + e, so a resumed run sees the same order. |
| `render_workers` | integer, ≥ 1 | `None` | Worker processes that render a split still rendering after a few seconds; 1 renders in this process. Unset: the cores this process may use less one, at most 4. Each holds a copy of the renderer and its tokenizer. Changes how fast and how much memory check and the start of run take, never what trains. |

## SFT `[training]`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `seed` | integer | `None` | The Loops trainer's random seed. Unset: Loops chooses. |
| `max_steps` | integer, ≥ 0 | *required* | Stop once the run has made this many optimizer updates, the ones before a resume included. 0 runs without a limit: RL until stopped, SFT every epoch. |
| `learning_rate` | number, > 0 | *required* | Adam learning rate; under SFT's lr_schedule, its peak. |
| `grad_clip_norm` | number, ≥ 0 | `None` | Gradient norm clipping; 0 disables it. Unset: the Loops SDK's. |
| `adam_epsilon` | number, > 0 | `None` | Adam epsilon. Unset: the Loops SDK's. |
| `adam_beta1` | number, ≥ 0 | `None` | Adam beta1. Unset: the Loops SDK's. |
| `adam_beta2` | number, ≥ 0 | `None` | Adam beta2. Unset: the Loops SDK's. |
| `weight_decay` | number, ≥ 0 | `None` | Adam weight decay. Unset: the Loops SDK's. |
| `resume_from` | string | `None` | A checkpoints/step-N.json manifest from a prior run (for SFT, of the same data), or that run's directory, which resumes from the manifest it wrote last. Relative to this config file, or to the shell when passed as a command-line flag. |
| `init_from` | string | `None` | Loops trainer weights a fresh run starts from, such as bt://loops:<run>/weights/step-N: the LoRA adapter, with fresh Adam moments; the run's steps count from 0, the trainer's policy version continues from the checkpoint's. Same base model and LoRA rank. A resume ignores it: the resumed state already holds those weights. |
| `init_from_optimizer` | boolean | `False` | Also load init_from's optimizer state (Adam moments and step): a fork that continues the source run's optimizer under this run's config. The moments reflect the source run's gradient scale; from a checkpoint saved without them Loops loads the weights only. Requires init_from; a resume ignores it. |
| `lr_schedule` | `constant` \| `linear` \| `cosine` | *required* | Learning rate after warmup: constant, or decayed towards zero by the last update, linearly or along a half cosine. |
| `warmup_steps` | integer, ≥ 0 | *required* | Updates over which the learning rate rises linearly to its peak; 0 starts at the peak. |
| `batch_size` | integer, ≥ 1 | *required* | Training sequences per optimizer update. A conversation is one sequence, or one per trained turn when its template renders history differently from how each turn was generated. |
| `epochs` | integer, ≥ 1 | *required* | Passes over the training sequences. |
| `loss_aggregation` | `sum` \| `token_mean` | `"sum"` | Reduction over each update's trained tokens. sum is Loops' default: no division; token_mean divides by the update's trained-token count. |

## SFT `[eval]`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `every_steps` | integer, ≥ 0 | *required* | Score validation NLL every N updates; 0 scores only before the first update and after the last. |

## SFT `[compute]`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `scope` | `serverless` \| `dedicated` | *required* | Where the run's trainer, samplers and evals run. serverless: Loops' shared capacity, billed for tokens; a model or configuration it cannot serve fails the run's creation. dedicated: deployments of the run's own, billed for GPU time until the run is deactivated. |
| `spot_job` | boolean | `False` | Use reclaimable Loops spot capacity; interrupted runs stop. Dedicated only. |
| `reuse_from_run_id` | string | `None` | Reuse a prior run's warm pods with fresh LoRAs; does not restore a checkpoint. Dedicated only. |
| `keep_run` | boolean | `True` | Leave the Loops run up when the process exits. A dedicated run bills until it is deactivated with loops deactivate, and a later dedicated run's reuse_from_run_id can start warm on it; a serverless run holds a workspace slot. false deactivates the run on exit; its checkpoints stay. |
| `trainer_replicas` | integer, ≥ 1 | `1` | Parallel copies of the trainer. More raise update throughput at proportionally more GPUs; above 1 always provisions fresh capacity. Dedicated only. |

## SFT `[output]`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `run_name` | string | *required* | Prefix for the run directory and the Loops run name. Required, so that two configs in one output_dir do not share run directories: `loops run --resume` takes the latest one of this name. |
| `output_dir` | string | `"runs"` | Where run directories are created. |
| `wandb_project` | string | `None` | Mirror metrics to this W&B project (best effort). |
| `console` | `auto` \| `plain` \| `rich` | `"auto"` | Terminal output style. auto: coloured logs, and a live progress region where the recipe has one, when the log stream is a terminal and the console extra is installed, plain otherwise. Redirected output is always plain, so a captured log never contains control characters. |
| `trainer_checkpoint_every` | integer, ≥ 0 | *required* | Save resumable trainer state every N updates; 0 disables. The last update is always saved: the one at max_steps, and on the way out of a run that ends any other way (bounded, best effort). |
| `sampler_checkpoint_every` | integer, ≥ 0 | *required* | Stage servable adapter weights every N updates; 0 disables. The update at max_steps is always saved. |
| `checkpoint_ttl_seconds` | integer, ≥ 0 | *required* | Remote checkpoint retention, in seconds; 0 keeps them. |

<!-- /generated -->
