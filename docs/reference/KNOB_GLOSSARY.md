# Knob glossary

Every setting the cookbook reads. Settings live in `train.toml` under the section
shown, and the CLI exposes each key of a table that is always present by its dotted
path, such as `--rollout.max_concurrent_rollouts`.

A rollout runtime's own settings live in `[rollout_runtime.config]`, and the training
config treats them as opaque. The tables of the built-in rollout runtimes and providers
are listed after the training config's. For the Harbor rollout runtime,
`[rollout_runtime.config]` holds only `reward_key`. Agent behaviour and environment
configuration belong to the Harbor job. A custom provider defines its own
`[providers.<name>]`. For the settings that only make sense together, see
[`settings-that-interact.md`](settings-that-interact.md).

**Defaults or required.** A setting that changes what trains or what a run costs is
required, such as `learning_rate`, `group_size` or a sandbox's size. A setting has a
default only when it is purely operational, which means a wrong value fails loudly
and nothing about training depends on it. The `[rollout.cleanup]` graces are an
example. A setting that the platform has its own default for, such as `adam_beta2`, is
optional and has no value of ours. When it is unset, the request carries the Loops
SDK's default, which a `baseten-loops` release may change. An opt-in feature, such as
`router_replay`, is off unless it is set. A new setting follows the same rule.

<!-- generated from the settings models; `python -m loops_cookbook.docs` rewrites it -->

## `[rollout_runtime]`

These settings say where episodes come from. They name the rollout runtime factory and what it may resolve.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `factory` | string | *required* | The rollout runtime factory, as module:callable. For the Harbor rollout runtime, use loops_cookbook.harbor:create_runtime. For a custom rollout runtime, name its module beside this config file. |
| `model_access` | string | `None` | How the agent reaches the model. direct is the cookbook's llm() client, which only an in-process agent can use. openai_http is an OpenAI-compatible Chat Completions endpoint for each attempt. When unset, the run uses the model access the rollout runtime declares, which is an Episode class's model_access or the model access of a Harbor job's agents. Set it only for a rollout runtime that supports more than one. |
| `openai` | table | `None` | Settings for the Chat Completions gateway. They apply only when the model access is openai_http. |
| `config` | table | empty | The rollout runtime's own settings. The cookbook hands them to the factory unchanged. |
| `files` | table | empty | Files the rollout runtime may resolve, as alias = path. A path in this file is relative to this config file. A path passed as a command-line flag is relative to the shell's working directory. |
| `has_images` | boolean | `False` | Whether the agent's prompts carry images. Loops can't record an image prompt, so router replay needs this to be false. |

## `[rollout_runtime.openai]`

These settings configure the Chat Completions gateway, which the run serves when model_access = openai_http.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `bind` | string | `"127.0.0.1:0"` | The host:port the gateway listens on. By default it listens on loopback, on a port the system picks. |
| `advertised_base_url` | string | `None` | The origin agents use to reach the gateway, such as http://127.0.0.1:43210. By default it is the gateway's bound loopback address. |
| `model_name` | string | `None` | The one model name the gateway advertises. By default it is [model].base_model. |

## `[model]`

These settings name the policy being trained and the compute it is configured for.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `base_model` | string | *required* | The Loops base model, such as Qwen/Qwen3.6-35B-A3B. |
| `tokenizer` | string | `None` | Where the policy's tokenizer loads from. Give a Hugging Face repo ID, or a local directory that starts with ./, / or ~. A local directory is relative to this file. The default is base_model. Set it for a model that Loops serves under a name with no Hugging Face repo. A private repo reads HF_TOKEN. |
| `lora_rank` | integer, ≥ 1 | *required* | The LoRA rank of the trained adapter. |
| `max_seq_len` | integer, ≥ 1 | `None` | The trainer's token budget for each sequence. RL defaults to the model's own limit. SFT requires it, so that check can refuse a row longer than this before anything is provisioned. |
| `renderer` | string | `None` | The renderer factory, written as module:callable. By default the cookbook uses its own renderer for the models it renders, which docs/reference/renderers.md lists. For other models, Prime detects the renderer from the tokenizer. |
| `renderer_chat_template_kwargs` | table | empty | Model-specific renderer options. On the command line, pass them as JSON. |
| `reasoning_efforts` | table | empty | The reasoning_effort values a request may send. Each value maps to the renderer_chat_template_kwargs entries it overrides. The run rejects a request that sends a value not declared here. |

## `[sampling]`

These settings are the defaults for what the policy produces. Agents inherit them and state only what they need to change.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `temperature` | number, ≥ 0 | `1.0` | The sampling temperature for a generation whose request names none. A request may set its own. |
| `max_generation_tokens` | integer, ≥ 1 | *required* | The length limit, in tokens, for a generation whose request names none. A request may set its own, so this value is not a cap. The run lowers either limit to the room the trajectory budget leaves. |
| `max_trajectory_tokens` | integer, ≥ 1 | `None` | The trajectory budget, in tokens, for the prompt plus the generation. The run lowers each call's max_tokens to the room its prompt leaves, and it refuses only a prompt that leaves no room. The budget always applies over HTTP. A direct call may pass its own. The default is max_seq_len, which is the capacity the trainer was configured for. |

## `[training]`

These settings configure the optimizer, the batch, and what data is acceptable to train on.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `seed` | integer | `None` | The Loops trainer's random seed. When it is unset, Loops chooses one. |
| `max_steps` | integer, ≥ 0 | *required* | The run stops once it has made this many optimizer updates, counting the updates made before a resume. 0 sets no limit. RL then runs until it is stopped, and SFT runs every epoch. |
| `learning_rate` | number, > 0 | *required* | The Adam learning rate. Under SFT's lr_schedule, this is the peak learning rate. |
| `grad_clip_norm` | number, ≥ 0 | `None` | The gradient norm to clip at. 0 turns clipping off. When it is unset, the Loops SDK's default applies. |
| `adam_epsilon` | number, > 0 | `None` | Adam's epsilon. When it is unset, the Loops SDK's default applies. |
| `adam_beta1` | number, ≥ 0 | `None` | Adam's beta1. When it is unset, the Loops SDK's default applies. |
| `adam_beta2` | number, ≥ 0 | `None` | Adam's beta2. When it is unset, the Loops SDK's default applies. |
| `weight_decay` | number, ≥ 0 | `None` | Adam's weight decay. When it is unset, the Loops SDK's default applies. |
| `resume_from` | string | `None` | The run to resume. Give a checkpoints/step-N.json manifest from a prior run, or that run's directory, which resumes from the manifest the run wrote last. For SFT, the prior run must have trained the same data. The path is relative to this config file, or to the shell's directory when it is passed as a command-line flag. |
| `init_from` | string | `None` | The Loops trainer weights that a fresh run starts from, such as bt://loops:<run>/weights/step-N. The run loads the LoRA adapter with fresh Adam moments. The run's steps count from 0, and the trainer's policy version continues from the checkpoint's. The checkpoint must have the same base model and LoRA rank. A resume ignores this setting, because the resumed state already holds those weights. |
| `init_from_optimizer` | boolean | `False` | Also load the optimizer state of init_from, meaning the Adam moments and step. The run then continues the source run's optimizer under this run's config. The moments reflect the source run's gradient scale. From a checkpoint saved without them, Loops loads only the weights. This setting requires init_from, and a resume ignores it. |
| `mode` | `sync` \| `async` | `"sync"` | sync collects a batch and then trains on it, one batch at a time. async collects the next batch while the current one trains. |
| `group_size` | integer, ≥ 2 | *required* | The number of rollouts sampled for each task's group. |
| `max_group_losses` | integer, ≥ 0 | *required* | How many rollouts a group may lose and still train. It is a tolerance rather than a minimum, so it keeps its meaning when group_size changes. |
| `groups_per_step` | integer, ≥ 1 | *required* | The number of groups in each optimizer update. |
| `loss_fn` | `importance_sampling` \| `ppo` \| `cispo` \| `dro` \| `dppo` \| `custom` | *required* | The loss that forward_backward computes. Name one of Loops' built-in losses, or custom for your own loss from [training.custom_loss], which runs through forward_backward_custom. |
| `loss_fn_config` | table | empty | The built-in loss's settings, which the cookbook hands to Loops as they are. A key left unset keeps Loops' default. Each loss's settings follow, with defaults in parentheses. importance_sampling takes no settings. ppo takes clip_low_threshold (0.8), clip_high_threshold (1.2). cispo takes clip_low_threshold (0.8), clip_high_threshold (1.2). dro takes beta (0.05). dppo takes dppo_mask_high (0.2), dppo_mask_low (0.2), adv_tau (1), kl_tau (0.001). |
| `custom_loss` | table | `None` | Your own loss. It is required when loss_fn = "custom". |
| `loss_aggregation` | `sum` \| `token_mean` \| `fixed_budget` \| `task_mean` | `"sum"` | How the built-in loss is reduced over a complete optimizer update. sum is Loops' default, and it divides by nothing. token_mean divides by the update's action tokens that have a nonzero advantage. fixed_budget divides by group_size * groups_per_step * loss_token_budget. task_mean averages the task groups' means over action tokens. A custom loss reduces inside its own function and keeps sum. |
| `loss_token_budget` | integer, ≥ 1 | `None` | The action-token budget for one whole episode, across all its turns, that fixed_budget aggregation divides by. It is required with loss_aggregation = fixed_budget and refused with any other. It is a normalization constant, not a generation limit. |
| `router_replay` | boolean | `False` | Whether to replay the expert routes of eligible batches, which is router replay (R3). It is off by default. |
| `max_consecutive_unusable_groups` | integer, ≥ 0 | `16` | The run aborts after this many groups in a row are rejected for training. 0 turns the check off. The count ignores infrastructure failures. It catches a wave of agent failures that were graded as zeros. |
| `max_steps_off_policy` | integer, ≥ 0 | `None` | In async mode, how many updates old a rollout's policy may be when the rollout trains. It is required with mode = async. Sync trains only on the current policy, so sync refuses it. |
| `admission_steps_ahead` | number, > 0 | `None` | In async mode, how many batches beyond the one in training that admission starts groups for. It may be at most max_steps_off_policy, which stays the staleness cap. A fraction admits part of a batch. With a lower value, groups wait less for the trainer and train closer to the current policy. A group that runs long still trains, up to the cap, and its finished members' slots go to new groups in the meantime. When unset, it equals max_steps_off_policy. |
| `max_policy_delta_in_trajectory` | integer, ≥ 0 | `None` | In async mode, how many updates apart the policy versions inside one rollout may be. It is required with mode = async. Sync samples each rollout from one policy, so sync refuses it. |

## `[training.custom_loss]`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `factory` | string | *required* | The loss factory, as module:callable, such as a module beside this config file. The factory takes this table's config and returns the loss function that forward_backward_custom runs. A custom loss needs the custom-loss extra. |
| `config` | table | empty | The loss's own settings. The cookbook hands them to the factory unchanged. |

## `[rollout]`

These settings say how much runs at once, how long each episode may take, and when to give up.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `max_concurrent_rollouts` | integer, ≥ 1 | *required* | How many rollouts may hold sampling capacity at once. |
| `max_concurrent_verifiers` | integer, ≥ 1 | *required* | How many rollouts may hold verification capacity at once. |
| `prestart_allowance` | number, ≥ 0 | `0.1` | How many admitted rollouts may start their environment before a sampling slot is free, as a fraction of max_concurrent_rollouts. The aim is enough ready environments that a freed slot is taken at once, without paying for many idle ones. |
| `max_concurrent_provisioning` | integer, ≥ 1 | `8` | The run-wide limit on provider operations in flight, which are creates, reconciliation, releases and absence checks. Each provider's create window also starts at this value and never grows past it. Refusals from a provider narrow its window. |
| `max_consecutive_rollout_failures` | integer, ≥ 0 | `64` | The run aborts after this many failures in a row in one stage. The stages are sampling infrastructure, grading infrastructure, cleanup infrastructure and a group grader. 0 turns the check off. |
| `shuffle_seed` | integer | `0` | The seed for the task order. |
| `shutdown_drain_seconds` | number, > 0 | `120` | How long in-flight attempts get to settle on shutdown. The run's resource creates and releases then get the same time again to finish. Whatever is left is cut off, given a few seconds to stop, and reported for `loops cleanup`. An operation that ignores the cut-off is reported, and it holds the process until it stops. |
| `producers` | table | `{'main': {'trainable': True}}` | Every producer a rollout runtime may use, by name, and whether its generations receive gradient. A call that names no producer uses main, and the run refuses an undeclared name. Setting this table replaces the default, so list main if calls rely on it. |
| `rollout_timeout_seconds` | number, > 0 | `None` | A backstop on how long a rollout may hold sampling capacity, not counting resource-service time such as a sandbox create. A rollout runtime bounds its own agent below this value. When unset, the run uses the clock the rollout runtime derives, and an Episode derives one from its agent deadline. It is required for a rollout runtime that derives no clock. |
| `setup_timeout_seconds` | number, > 0 | *required* | The bound on the time an attempt spends before sealing while it holds neither sampling nor verification capacity, such as preparing an environment. It is required. |
| `max_rollout_retries` | integer, ≥ 0 | `0` | How many extra attempts a rollout may use after an episode failure, before it has a valid rollout. Use it as a last resort. Rollout runtimes retry their own steps, and the run never retries model-call failures here because the Loops SDK already retried them. |

## `[rollout.producers.<name>]`

Each table configures one producer that a rollout runtime may name, keyed by that name.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `trainable` | boolean | *required* | Whether this producer's generations receive gradient. |

## `[rollout.grading]`

This setting is the deadline for each try of grade(). The rollout runtime decides how many tries a failure earns.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `timeout_seconds` | number, > 0 | *required* | The deadline for each grade() try and each group grader call. It is required. |

## `[rollout.cleanup]`

These settings are the deadlines for the rollout runtime's cleanup hooks.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `timeout_seconds` | number, > 0 | `600` | The deadline for each finish(), abort() and close() call. It is generous because it covers a Harbor trial's teardown. A value set too long only delays the report of a hook that hangs. |
| `cancellation_grace_seconds` | number, > 0 | `5` | How long a cancelled rollout runtime hook may take to stop before the run fails. |
| `process_kill_grace_seconds` | number, > 0 | `10` | How long an agent process may take to exit after SIGTERM before it is killed. This time is spent before a cancelled hook's own grace starts. |

## `[compute]`

These settings say where the run executes. Nothing here changes what is learned.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `scope` | `serverless` \| `dedicated` | *required* | Where the run's trainer, samplers and evals run. serverless uses Loops' shared capacity and bills for tokens. If that capacity cannot serve the model or the configuration, creating the run fails. dedicated gives the run deployments of its own, which bill for GPU time until the run is deactivated. |
| `spot_job` | boolean | `False` | Run on reclaimable Loops spot capacity. A run whose capacity is reclaimed stops. Dedicated only. |
| `reuse_from_run_id` | string | `None` | Reuse a prior run's warm pods, with fresh LoRA adapters. It does not restore a checkpoint. Dedicated only. |
| `keep_run` | boolean | `True` | Leave the Loops run up when the process exits. A dedicated run bills until you deactivate it with loops deactivate, and a later dedicated run can start warm on it through reuse_from_run_id. A serverless run holds a workspace slot. false deactivates the run on exit, and its checkpoints stay. |
| `trainer_replicas` | integer, ≥ 1 | `1` | The number of parallel copies of the trainer. More copies raise update throughput and use proportionally more GPUs. Any value above 1 always provisions fresh capacity. Dedicated only. |
| `sampler_replica_affinity` | boolean | `True` | Whether to pin a rollout's requests to one sampler replica, so that its growing prompt keeps hitting a warm prefix cache. It is a routing hint. Where Loops can't pin a request, it serves the request without affinity. |
| `sampler_replicas` | integer, ≥ 1 | `None` | A fixed replica count for the paired sampler. When unset, the Loops backend default applies. Dedicated only. |

## `[output]`

These settings say what the run leaves behind.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `run_name` | string | *required* | The prefix for the run directory and the Loops run name. It is required so that two configs in one output_dir do not share run directories. `loops run --resume` takes the latest run directory with this name. |
| `output_dir` | string | `"runs"` | The directory where run directories are created. |
| `wandb_project` | string | `None` | Mirror metrics to this W&B project. The mirror is best effort. |
| `console` | `auto` \| `plain` \| `rich` | `"auto"` | The terminal output style. auto gives coloured logs, plus a live progress region where the recipe has one, when the log stream is a terminal and the console extra is installed. Otherwise auto gives plain output. Redirected output is always plain, so a captured log never contains control characters. |
| `trainer_checkpoint_every` | integer, ≥ 0 | *required* | Save resumable trainer state every N updates. 0 turns this off. The last update is always saved. That is the update at max_steps, or the last update of a run that ends any other way, saved on the way out within a bounded time and on a best-effort basis. |
| `sampler_checkpoint_every` | integer, ≥ 0 | *required* | Stage servable adapter weights every N updates. 0 turns this off. The update at max_steps is always saved. |
| `checkpoint_ttl_seconds` | integer, ≥ 0 | *required* | How long Loops keeps each remote checkpoint, in seconds. 0 keeps checkpoints indefinitely. |
| `progress_every_seconds` | number, ≥ 0 | `60` | Seconds between progress reports while a batch collects. 0 turns the reports off. Optimizer steps can be tens of minutes apart, and nothing else reports in between. Without these reports, a long collection phase and a stall look the same. Every slots, queue and admission series in W&B comes from them. |
| `rollout_metrics_limit` | integer, ≥ 1 | `32` | The most rollout/* metrics a step reports from what the agents return about their episodes. Each label value counts as one metric. Past the limit, the run drops the keys that the fewest rollouts reported, and rollout/dropped_keys counts them. This stops a stray per-rollout ID from burying the rest. Raise the limit when an agent reports more. |
| `trace` | `off` \| `samples` \| `transcripts` \| `full` | `"off"` | The per-generation evidence to keep beside the run record. off keeps none. samples keeps one row per generation without token arrays, at about 1% of the size of full. transcripts keeps those rows and each attempt's transcript, so you can read what the policy saw and said. A transcript holds every generation's request as the agent sent it and the reply it received, each marked with whether it is trainable. full keeps the transcripts, the token arrays and the exact training rows, for replay and audit. The run writes every attempt's transcript, and a failed attempt's as far as it got. Recording never changes what trains. A request or reply that isn't JSON, such as bytes or a set in a direct agent's history, is written as unrecordable. |

## `[resources]`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `state_dir` | string | `"state"` | The directory of the resource journal, relative to the run directory. It must be on a local filesystem. |

## `[eval]`

These settings run held-out evals during training. Each eval runs on a sampler pinned to one step's weights. Nothing an eval samples trains.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `every_steps` | integer, ≥ 0 | `0` | Evaluate every N updates. 0 evaluates only before the first update and after the last. Evaluating after the last update needs [training] max_steps. |
| `sampler_start_timeout_seconds` | number, > 0 | `900` | How long an eval's sampler may take to become ready before the run skips that set's eval. On dedicated Loops, each eval starts a sampler deployment. |
| `sets` | table | empty | Held-out task sets, by name. With no sets, the run runs no evals. |

## `[eval.sets.<name>]`

Each table is one held-out task set, which runs training's rollout runtime with these files. A key at the path of a training setting overrides that setting for this set.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `samples_per_task` | integer, ≥ 1 | `1` | The number of rollouts for each task. |
| `max_tasks` | integer, ≥ 1 | `None` | Evaluate only the first N tasks. When unset, the set evaluates every task. |
| `max_concurrent_rollouts` | integer, ≥ 1 | `16` | How many of this set's rollouts may run at once, on top of training's. The set shares training's sandbox providers, so those providers see training's limit plus every set's. |

## `[eval.sets.<name>.rollout_runtime]`

This table overrides [rollout_runtime] for this set.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `files` | table | *required* | The [rollout_runtime.files] entries this set replaces. The rollout runtime, its config and every other file stay the same as training's. |

## `[eval.sets.<name>.sampling]`

This table overrides [sampling] for this set.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `temperature` | number, ≥ 0 | `None` | The sampling temperature for this set. When unset, it is [sampling].temperature. |

## `[rollout_runtime.config]` for Episodes

These settings apply with factory = "loops_cookbook.episodes:create_runtime". The tasks file is the `tasks` alias under `[rollout_runtime.files]`.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `episode` | string | *required* | The Episode subclass to run, written as module:Class. |
| `agent_timeout_seconds` | number, > 0 | *required* | How long run() may take, in seconds. The cookbook then cuts run() and grades the episode. |
| `sandbox` | table | `None` | The sandbox each attempt gets. The cookbook provisions it before setup(). |
| `grading_sandbox` | table | `None` | An optional separate sandbox for grading. The cookbook provisions it when an attempt is first admitted to grading. It requires the episode to define grade(). |
| `options` | table | `None` | The episode's own settings, validated with its options model. They are written to config.json with the rest of the config, so they must not hold secrets. |

## `[rollout_runtime.config.sandbox]` for Episodes

This table configures an Episode's sandbox. The keys beside `provider` are the provider's sandbox spec, which the next tables list for Blaxel and Daytona.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `provider` | string | *required* | The [providers] entry that provides this sandbox. |

## `[rollout_runtime.config.sandbox]` on Blaxel

These keys sit beside provider = "blaxel".

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `image` | string | *required* | The image the sandbox runs. |
| `memory_mb` | integer, > 0 | *required* | The sandbox's memory, in MB, like Harbor's memory_mb. |
| `ttl` | string | *required* | How long Blaxel keeps the sandbox, such as 1h. It is a backstop, because the run releases the sandbox itself. It is required, as Harbor's Blaxel ttl is. Write a whole number followed by one unit, which is s, m, h, d or w. |

## `[rollout_runtime.config.sandbox]` on Daytona

These keys sit beside provider = "daytona". `[providers.daytona]` sets the sandbox's lifecycle.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `image` | string | `None` | The registry image the sandbox runs, such as python:3.12-slim. When it is unset, the sandbox runs Daytona's default sandbox image. |
| `snapshot` | string | `None` | The name of a Daytona snapshot that the sandbox starts from instead of an image. The snapshot fixes both the sandbox's image and its size. |
| `cpu` | integer, > 0 | `None` | The sandbox's vCPUs. When it is unset, [providers.daytona] cpu applies. |
| `memory_gib` | integer, > 0 | `None` | The sandbox's memory, in GiB. When it is unset, [providers.daytona] memory_gib applies. |
| `disk_gib` | integer, > 0 | `None` | The sandbox's disk, in GiB, not counting the image. When it is unset, [providers.daytona] disk_gib applies. |

## `[rollout_runtime.config.grading_sandbox]` for Episodes

This table configures an Episode's optional separate verifier sandbox, which is provisioned only when grading admits the rollout. The keys beside `provider` are the provider's sandbox spec.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `provider` | string | *required* | The [providers] entry that provides this sandbox. |

## `[rollout_runtime.config.grading_sandbox]` on Blaxel

These keys sit beside provider = "blaxel". They are independent of the execution sandbox's spec.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `image` | string | *required* | The image the sandbox runs. |
| `memory_mb` | integer, > 0 | *required* | The sandbox's memory, in MB, like Harbor's memory_mb. |
| `ttl` | string | *required* | How long Blaxel keeps the sandbox, such as 1h. It is a backstop, because the run releases the sandbox itself. It is required, as Harbor's Blaxel ttl is. Write a whole number followed by one unit, which is s, m, h, d or w. |

## `[providers.blaxel]`

This table configures the built-in Blaxel provider. Its credentials, BL_WORKSPACE and BL_API_KEY, come from the environment or a .env beside the config, never from this table.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `region` | string | *required* | The region where the run creates every sandbox and builds every image. It is required. |
| `create_timeout_seconds` | number, > 0 | `120` | The longest the run waits for Blaxel to create one sandbox and report it deployed. |
| `reconcile_timeout_seconds` | number, > 0 | `60` | The longest the run spends looking up an ambiguous create by name. It also bounds working out what a sandbox needs created first, such as its image. |
| `image_build_timeout_seconds` | number, > 0 | `420` | The longest the run waits for one sandbox image to build. The time includes the builder sandbox. |
| `image_build_memory_mb` | integer, ≥ 1, ≤ 32768 | `None` | The memory, in MB, of the environment that builds each image. When it is unset, Blaxel uses its default, which a large image can run out of mid-build. It sizes only the build. A sandbox's memory comes from its task. |
| `image_build_volume_mb` | integer, ≥ 0, ≤ 131072 | `None` | The disk, in MB, for each build's intermediate layers. 0 builds in memory. When it is unset, Blaxel uses its default, which a large image can run out of mid-build. |
| `release_timeout_seconds` | number, > 0 | `120` | The longest the run waits for one sandbox to be deleted. |
| `confirm_absent_timeout_seconds` | number, > 0 | `60` | The longest the run spends confirming that a deleted sandbox is gone. |
| `list_timeout_seconds` | number, > 0 | `60` | The longest the run waits for each page of the workspace listing that a label sweep reads. The whole listing has no bound, because it grows with the workspace. |
| `attach_timeout_seconds` | number, > 0 | `30` | The longest the run spends attaching a client to a created sandbox. |
| `labels` | table | empty | Labels that every sandbox carries in addition to the run's ownership labels. Image builders carry them too. For example, an owner label lets a team find its sandboxes. These labels override the same keys from a Harbor job's labels or from [rollout_runtime.config.sandbox] labels. |

## `[providers.daytona]`

This table configures the built-in Daytona provider, which [daytona.md](daytona.md) describes. Its credential, DAYTONA_API_KEY, comes from the environment or a .env beside the config, never from this table. A self-hosted Daytona also needs DAYTONA_API_URL from the same place.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `target` | string | `None` | The Daytona target, meaning the region, such as us or eu. Every sandbox and every Daytona call uses it, including Harbor's. When it is unset, the organization's default target applies. The cookbook refuses a DAYTONA_TARGET in the environment that differs from it. |
| `create_timeout_seconds` | number, > 0 | `600` | The longest the run waits for Daytona to create one sandbox and report it started. The time includes a first build of the sandbox's declarative image. |
| `reconcile_timeout_seconds` | number, > 0 | `60` | The longest the run spends looking up an ambiguous create by name. |
| `release_timeout_seconds` | number, > 0 | `120` | The longest the run waits for one sandbox to be deleted. |
| `confirm_absent_timeout_seconds` | number, > 0 | `60` | The longest the run spends confirming that a deleted sandbox is gone. |
| `list_timeout_seconds` | number, > 0 | `60` | The longest the run waits for each page of the labelled listing that a label sweep reads. |
| `attach_timeout_seconds` | number, > 0 | `30` | The longest the run spends attaching a client to a created sandbox. |
| `cpu` | integer, > 0 | *required* | The vCPUs of a sandbox whose request names none. |
| `memory_gib` | integer, > 0 | *required* | The memory, in GiB, of a sandbox whose request names none. |
| `disk_gib` | integer, > 0 | *required* | The disk, in GiB, of a sandbox whose request names none. It covers only the sandbox's writable layer, because the image is not counted. |
| `thread_variables_from_vcpus` | boolean | `False` | Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to a Harbor sandbox's vCPUs for every command. This matters because nproc in a sandbox counts its host's cores. Harbor passes these variables with each command, so they override the image's ENV. The task's [environment.env] still takes precedence. |
| `auto_stop_minutes` | integer, ≥ 0 | `120` | The minutes without a Daytona API call after which Daytona stops a sandbox whose request sets none. A command counts as a call, but a process inside the sandbox does not. With auto_delete_minutes 0, Daytona then deletes the sandbox. It must outlast an attempt's longest quiet spell, such as a long model turn or a wait for grading. 0 disables it, and then ttl_minutes must be set. |
| `auto_delete_minutes` | integer, ≥ 0 | `0` | The minutes Daytona keeps a stopped sandbox before deleting it. 0 deletes the sandbox as it stops. |
| `ttl_minutes` | integer, ≥ 0 | *required* | The wall-clock minutes after which Daytona destroys a sandbox, whatever its state. It is a backstop for a sandbox the run lost. It must outlast the longest attempt, because Daytona destroys a sandbox that is still working when the time passes. 0 sets no limit. |
| `labels` | table | empty | Labels that every sandbox carries in addition to the run's ownership labels. For example, an owner label lets a team find its sandboxes. These labels override the same keys from a Harbor job's labels or from [rollout_runtime.config.sandbox] labels. |
| `max_sandbox_cpu` | integer, > 0 | `None` | The most vCPUs Daytona gives one sandbox. The cookbook refuses a larger task before the run starts. |
| `max_sandbox_memory_gib` | integer, > 0 | `None` | The most memory, in GiB, that one sandbox may have. |
| `max_sandbox_disk_gib` | integer, > 0 | `None` | The most disk, in GiB, that one sandbox may have. |

# SFT settings

An SFT config, `recipe = "sft"`, has its own TOML. Its `[model]`, `[compute]` and
`[output]` tables are the RL recipe's, spelled the same, except that SFT requires
`max_seq_len`.

## SFT `[model]`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `base_model` | string | *required* | The Loops base model, such as Qwen/Qwen3.6-35B-A3B. |
| `tokenizer` | string | `None` | Where the policy's tokenizer loads from. Give a Hugging Face repo ID, or a local directory that starts with ./, / or ~. A local directory is relative to this file. The default is base_model. Set it for a model that Loops serves under a name with no Hugging Face repo. A private repo reads HF_TOKEN. |
| `lora_rank` | integer, ≥ 1 | *required* | The LoRA rank of the trained adapter. |
| `max_seq_len` | integer, ≥ 1 | *required* | The trainer's token budget for each sequence. RL defaults to the model's own limit. SFT requires it, so that check can refuse a row longer than this before anything is provisioned. |
| `renderer` | string | `None` | The renderer factory, written as module:callable. By default the cookbook uses its own renderer for the models it renders, which docs/reference/renderers.md lists. For other models, Prime detects the renderer from the tokenizer. |
| `renderer_chat_template_kwargs` | table | empty | Model-specific renderer options. On the command line, pass them as JSON. |

## SFT `[data]`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `format` | `chat` \| `tokens` | `"chat"` | The contract that every row meets. chat rows are OpenAI chat fine-tuning JSONL. The cookbook renders them with the model's template, which decides the targets. tokens rows give input_ids and 0/1 weights, and they train as given. Nothing checks that their targets are the model's own tokens. |
| `train` | string | *required* | The training rows, as JSONL with one row per line in the format that [data] format names. The path is relative to this config file, or to the shell's directory when it is passed as a command-line flag. |
| `validation` | string | `None` | Held-out rows in the same format, which the run scores for validation negative log-likelihood (NLL). The path resolves like train. Set this or validation_fraction. Set neither for no validation. |
| `validation_fraction` | number, > 0 | `None` | Hold out this fraction of the training rows for validation, instead of reading a separate file. shuffle_seed chooses the rows. |
| `shuffle_seed` | integer | `0` | The seed that chooses the held-out rows and each epoch's order. Epoch e shuffles with shuffle_seed + e, so a resumed run sees the same order. |
| `render_workers` | integer, ≥ 1 | `None` | The number of worker processes that take over a split which is still rendering after a few seconds. 1 renders in this process. When it is unset, the cookbook uses one fewer than the cores this process may use, and at most 4. Each worker holds a copy of the renderer and its tokenizer. This setting changes how fast check and the start of a run are, and how much memory they use. It never changes what trains. |

## SFT `[training]`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `seed` | integer | `None` | The Loops trainer's random seed. When it is unset, Loops chooses one. |
| `max_steps` | integer, ≥ 0 | *required* | The run stops once it has made this many optimizer updates, counting the updates made before a resume. 0 sets no limit. RL then runs until it is stopped, and SFT runs every epoch. |
| `learning_rate` | number, > 0 | *required* | The Adam learning rate. Under SFT's lr_schedule, this is the peak learning rate. |
| `grad_clip_norm` | number, ≥ 0 | `None` | The gradient norm to clip at. 0 turns clipping off. When it is unset, the Loops SDK's default applies. |
| `adam_epsilon` | number, > 0 | `None` | Adam's epsilon. When it is unset, the Loops SDK's default applies. |
| `adam_beta1` | number, ≥ 0 | `None` | Adam's beta1. When it is unset, the Loops SDK's default applies. |
| `adam_beta2` | number, ≥ 0 | `None` | Adam's beta2. When it is unset, the Loops SDK's default applies. |
| `weight_decay` | number, ≥ 0 | `None` | Adam's weight decay. When it is unset, the Loops SDK's default applies. |
| `resume_from` | string | `None` | The run to resume. Give a checkpoints/step-N.json manifest from a prior run, or that run's directory, which resumes from the manifest the run wrote last. For SFT, the prior run must have trained the same data. The path is relative to this config file, or to the shell's directory when it is passed as a command-line flag. |
| `init_from` | string | `None` | The Loops trainer weights that a fresh run starts from, such as bt://loops:<run>/weights/step-N. The run loads the LoRA adapter with fresh Adam moments. The run's steps count from 0, and the trainer's policy version continues from the checkpoint's. The checkpoint must have the same base model and LoRA rank. A resume ignores this setting, because the resumed state already holds those weights. |
| `init_from_optimizer` | boolean | `False` | Also load the optimizer state of init_from, meaning the Adam moments and step. The run then continues the source run's optimizer under this run's config. The moments reflect the source run's gradient scale. From a checkpoint saved without them, Loops loads only the weights. This setting requires init_from, and a resume ignores it. |
| `lr_schedule` | `constant` \| `linear` \| `cosine` | *required* | The learning rate after warmup. constant keeps it at its peak. linear and cosine decay it towards zero by the last update, along a straight line or a half cosine. |
| `warmup_steps` | integer, ≥ 0 | *required* | The number of updates over which the learning rate rises linearly to its peak. 0 starts at the peak. |
| `batch_size` | integer, ≥ 1 | *required* | The number of training sequences in each optimizer update. A conversation is usually one sequence. It becomes one sequence per trained turn when its template renders history differently from how each turn was generated. |
| `epochs` | integer, ≥ 1 | *required* | The number of passes over the training sequences. |
| `loss_aggregation` | `sum` \| `token_mean` | `"sum"` | How the loss combines each update's trained tokens. sum is Loops' default, and it does not divide. token_mean divides by the number of trained tokens in the update. |

## SFT `[eval]`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `every_steps` | integer, ≥ 0 | *required* | Score validation NLL every N updates. Validation is always scored after the last update, and before the first update of a run that is not a resume. 0 scores only at those two points. |

## SFT `[compute]`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `scope` | `serverless` \| `dedicated` | *required* | Where the run's trainer, samplers and evals run. serverless uses Loops' shared capacity and bills for tokens. If that capacity cannot serve the model or the configuration, creating the run fails. dedicated gives the run deployments of its own, which bill for GPU time until the run is deactivated. |
| `spot_job` | boolean | `False` | Run on reclaimable Loops spot capacity. A run whose capacity is reclaimed stops. Dedicated only. |
| `reuse_from_run_id` | string | `None` | Reuse a prior run's warm pods, with fresh LoRA adapters. It does not restore a checkpoint. Dedicated only. |
| `keep_run` | boolean | `True` | Leave the Loops run up when the process exits. A dedicated run bills until you deactivate it with loops deactivate, and a later dedicated run can start warm on it through reuse_from_run_id. A serverless run holds a workspace slot. false deactivates the run on exit, and its checkpoints stay. |
| `trainer_replicas` | integer, ≥ 1 | `1` | The number of parallel copies of the trainer. More copies raise update throughput and use proportionally more GPUs. Any value above 1 always provisions fresh capacity. Dedicated only. |

## SFT `[output]`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `run_name` | string | *required* | The prefix for the run directory and the Loops run name. It is required so that two configs in one output_dir do not share run directories. `loops run --resume` takes the latest run directory with this name. |
| `output_dir` | string | `"runs"` | The directory where run directories are created. |
| `wandb_project` | string | `None` | Mirror metrics to this W&B project. The mirror is best effort. |
| `console` | `auto` \| `plain` \| `rich` | `"auto"` | The terminal output style. auto gives coloured logs, plus a live progress region where the recipe has one, when the log stream is a terminal and the console extra is installed. Otherwise auto gives plain output. Redirected output is always plain, so a captured log never contains control characters. |
| `trainer_checkpoint_every` | integer, ≥ 0 | *required* | Save resumable trainer state every N updates. 0 turns this off. The last update is always saved. That is the update at max_steps, or the last update of a run that ends any other way, saved on the way out within a bounded time and on a best-effort basis. |
| `sampler_checkpoint_every` | integer, ≥ 0 | *required* | Stage servable adapter weights every N updates. 0 turns this off. The update at max_steps is always saved. |
| `checkpoint_ttl_seconds` | integer, ≥ 0 | *required* | How long Loops keeps each remote checkpoint, in seconds. 0 keeps checkpoints indefinitely. |

<!-- /generated -->
