# Episode basics

Small, standalone examples of the public Episode API: host-only scoring, a shared
sandbox, and a separate verifier. For complete training configurations, see the
[custom runtime patterns](../../patterns/custom_runtime/in_process_direct/).
For the smallest host-only example, read [hill_climb_episode.py](hill_climb_episode.py):
the model guesses one label from A–D and receives reward 1 for `B`, otherwise 0.
`HillClimbEpisode` declares no sandbox kinds or options. Only the task's prompt
goes to the model; the hidden target is used by `grade()` on the host. Whitespace
is stripped, but lowercase labels, explanations and punctuation do not pass.
The task model defaults match the existing Harbor hill-climb instruction and reward.
[train_hill_climb.toml](train_hill_climb.toml) runs it with the task in
[hill_climb_tasks.jsonl](hill_climb_tasks.jsonl). It declares no providers, sandboxes
or options; normal rollout/verification concurrency settings still apply.

From the repository root, with `BASETEN_API_KEY` in `.env`:

```sh
uv run loops check examples/episode_basics/train_hill_climb.toml
uv run loops run examples/episode_basics/train_hill_climb.toml
```

The configs train with CISPO. To compare reductions, keep the same Episode and
override only training settings. For example, check a fixed denominator of
16 configured episodes × 16 action tokens:

```sh
uv run loops check examples/episode_basics/train_hill_climb.toml \
  --training.loss_aggregation fixed_budget --training.loss_token_budget 16
```

Replace `check` with `run` to train that configuration. Use
`--training.loss_aggregation task_mean` for equal weighting across task groups.
This example has one group per update; differences from `sum` or `token_mean` are most
visible with multiple groups of different lengths or zero-advantage episodes.
See [loss aggregation](../../docs/reference/training.md#loss-aggregation) for the
denominators and multi-turn budget semantics.

[episodes.py](episodes.py) is illustrative user code, not part of the framework.
It shows three standalone shapes: `HostScoredEpisode` scores on the host,
`SharedSandboxEpisode` executes and scores in one Blaxel sandbox, and
`SeparateVerifierEpisode` saves its output on the host and releases the execution
sandbox before using a separate Blaxel verifier. Their hook signatures stay the same.

Each example directly subclasses `Episode[Question, Options]` and contains its own
constructor, sampling hook and grading hook. Read any one independently as the code
an author would write for that mode. These are example names, not framework types.

The declarations below are abbreviated; the linked file contains the complete hooks.

Each standalone example also has a runnable training configuration:

| Episode | Configuration | Resources |
| --- | --- | --- |
| `HostScoredEpisode` | [train_host_scored.toml](train_host_scored.toml) | Host only |
| `SharedSandboxEpisode` | [train_shared_sandbox.toml](train_shared_sandbox.toml) | One sandbox per attempt |
| `SeparateVerifierEpisode` | [train_separate_verifier.toml](train_separate_verifier.toml) | Execution sandbox released before a separate verifier |

They share [questions.jsonl](questions.jsonl).
[train_host_scored.toml](train_host_scored.toml) also evaluates the letters in
[questions_val.jsonl](questions_val.jsonl) every two updates (see Evals in
[training.md](../../docs/reference/training.md#evals)). They share training's prompt with
other answers, so the set shows how evals run, not generalization: its score falls as the
policy learns training's "B". Run any one with
`uv run loops run examples/episode_basics/<configuration>`.
All need `BASETEN_API_KEY`; the sandbox configurations also need `BL_WORKSPACE` and
`BL_API_KEY` in `.env`. The sandbox configurations use two sampling and two verification slots and a
ten-minute sandbox TTL as a cleanup backstop. Change the Blaxel region for your
workspace as needed.

```python
class HostScoredEpisode(Episode[Question, Options]): ...


class SharedSandboxEpisode(Episode[Question, Options]):
    execution_kind = Blaxel
    ...


class SeparateVerifierEpisode(Episode[Question, Options]):
    execution_kind = Blaxel
    sandbox_lifetime = "sampling"
    grading_kind = Blaxel
    ...
```

## API shape

- Two generics describe stable task/options data. Python 3.12's generic syntax is
  sufficient; both arguments are explicit, including `None` when unused.
- `execution_kind` and `grading_kind` name requirements. The existing TOML tables
  choose actual providers/specs and are validated by the existing factory.
- Fixed `setup(Setup)`, `run(Sampling)`, `grade(Grading)`, and `close(Closing)` hooks
  receive immutable phase records. Hook annotations do not configure injection.
- Grading receives `timed_out` explicitly. An execution sandbox released after
  sampling is `None`; a verifier never created is `None` during close. Optionality
  and concrete provider narrowing are the deliberate ergonomic costs of dropping
  resource generics. These records hold borrowed handles, not durable artifacts.

`SandboxKind` currently provides client/liveness access, not portable command/file
operations. The Blaxel example is intentionally provider-specific; these examples
do not add Modal support or pretend SDK calls are portable.

A config beside the example names `episodes:HostScoredEpisode`, or
`episodes:SeparateVerifierEpisode` with a grading sandbox table. The public
factory owns validation, admission, deadlines, retries and registered cleanup.
Group grading, HTTP model access, supervised processes and metrics use the same
public runtime; see [the Episode reference](../../docs/reference/episode-api.md).

Run the driver-backed examples offline:

```sh
uv run pytest tests/examples/test_episode_basics.py tests/examples/test_hill_climb_episode.py -q
```

Check real Blaxel file transfer, verifier retry and cancellation cleanup without
starting a training job (the policy reply is fixed for these lifecycle checks):

```sh
LOOPS_LIVE_BLAXEL=1 uv run --env-file .env pytest tests/live/test_episode_sandboxes.py -v
```
