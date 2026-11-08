# Tests

The tests are laid out like this:

```
tests/
  unit/        offline, fast, fakes only            uv run pytest tests/unit
    rl/        scheduler, trainer, datums, config
    llm/       client, Prime adapter, images
    harbor/    runtime, agent base, Blaxel
  contracts/   real tokenizers and image processors  see below
  patterns/    the shipped patterns train, offline, through the shared pipeline
  examples/    the examples built on them
  live/        real providers, opt-in and billable     see below
  support/     fakes, image builders, the run audit (not tests)
  fixtures/    offline stand-ins for the model and a Harbor environment (not tests)
  test_shipped_configs.py  every shipped training config passes `loops check`, offline
```

`uv run pytest` runs everything. The contract tests skip unless the model files are
available.

## Contract tests

The contract tests render with the real Hugging Face tokenizers and processors of the
supported models. So a renderer bug or a change in `transformers` fails here, with a
diff.

You can give the tests the model files in two ways:

- **Offline.** Point the variable for each model family at a Hugging Face hub cache
  that already holds the pinned snapshots. The variables are
  `LOOPS_QWEN_TOKENIZER_CACHE`, `LOOPS_GLM_TOKENIZER_CACHE`,
  `LOOPS_NEMOTRON_TOKENIZER_CACHE` and `LOOPS_MUSE_TOKENIZER_CACHE`.
  `contracts/conftest.py` names the pinned snapshots. Also set `HF_HUB_OFFLINE=1`, so
  that processors looked up by name stay local.
- **Download.** Set `LOOPS_DOWNLOAD_CONTRACT_MODELS=1` to fetch the tokenizer and
  processor configs. CI does this, with `uv sync --extra vision` for the image
  processors. The Muse Spark repository is gated, so its tests also need an `HF_TOKEN`
  that has access, and they skip without one.

## Live tests

The live tests run against real Blaxel sandboxes, and the end-to-end tests also use a
real Loops trainer. They cost money, so they skip unless you ask for them:

- `LOOPS_LIVE_BLAXEL=1 uv run --env-file .env pytest tests/live -m live_blaxel` runs
  the Blaxel tests. They need `BL_WORKSPACE` and `BL_API_KEY`.
  - The output tests use one small sandbox and take about three minutes. They check
    that a command's whole output reaches Harbor past Blaxel's 64 KB process record.
    They also check that Blaxel still marks a cut record in the way the cookbook looks
    for.
  - The episode tests run the file transfer, verifier retry and cancellation cleanup
    of the [episode_basics](../examples/episode_basics/) sandbox examples.
  - The image build test builds a new image through the cookbook on every run. It
    checks that a sandbox started from the image runs it and carries the labels from
    `[providers.blaxel] labels`.
- `LOOPS_LIVE_LOOPS=1 uv run --env-file .env pytest tests/live -m live_loops` runs the
  Loops tests. They need `BASETEN_API_KEY` as well as the Blaxel credentials. Each
  end-to-end test deactivates its Loops run, and requires `loops cleanup` to prove that
  every resource was released.
  - The output test trains one step of `live/output_job` on Qwen3.5-4B, which takes
    about twenty minutes.
  - The image test trains one step of `live/vision_job` on GLM-5.3-Flash. In that job,
    a tool returns images over several turns. Loops can't record an image prompt, so
    the job trains without router replay.
  - The trainer normalization contract uses one trainer, with no sampler or sandbox.
    It needs the `custom-loss` extra, and it fails without it. It takes
    zero-learning-rate steps, and compares the gradient norms of the built-in losses,
    for every `loss_aggregation` mode, with those of explicitly reduced custom losses.
    CI never runs it on a PR or a release. The **trainer normalization contract**
    workflow runs it when a maintainer starts it by hand.

## Fixtures

Each shared fixture lives in the `conftest.py` nearest the tests that use it:

- At the root, `conftest.py` provides `config`, `loops_capabilities` and `loops_check`.
  `loops_capabilities` is what every test, except a live one, is told that Loops serves.
  `loops_check` is an offline `loops check`. The root also provides the Blaxel fakes,
  `blaxel`, `blaxel_services` and `blaxel_sandbox`. It loads the episode rig's `make`
  and `script` from `support/episode_rig.py`.
- Under `unit/`, `conftest.py` provides the in-memory fakes `fake_client`, `fake_prime`
  and `prime_renderer`.
- Under `contracts/`, `conftest.py` provides the model fixtures `glm_tokenizer`,
  `flash_processor`, `flash_renderer`, `qwen_tokenizer` and `cached_vlm_tokenizer`.

The fakes themselves are plain classes in `support/`, in files such as `fakes.py`,
`harness.py`, `fake_blaxel.py` and `offline_run.py`. `ScriptedSampler` and
`ScriptedRenderer` replace the stubs that each test file would otherwise write.
