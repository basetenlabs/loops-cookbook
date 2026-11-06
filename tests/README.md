# Tests

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

`uv run pytest` runs everything; the contract tests skip unless model files are available.

## Contract tests

They render against the real Hugging Face tokenizers and processors for the supported
models, so a renderer bug or a `transformers` change fails here with a diff.

- Offline: point `LOOPS_QWEN_TOKENIZER_CACHE` / `LOOPS_GLM_TOKENIZER_CACHE` /
  `LOOPS_NEMOTRON_TOKENIZER_CACHE` / `LOOPS_MUSE_TOKENIZER_CACHE` at a hub cache
  that already holds the pinned snapshots (see `contracts/conftest.py`), and set
  `HF_HUB_OFFLINE=1` so processors looked up by name stay local.
- Download: `LOOPS_DOWNLOAD_CONTRACT_MODELS=1` fetches tokenizer and processor configs
  (this is what CI does, with `uv sync --extra vision` for the image processors).

## Live tests

They run against real Blaxel sandboxes and, for the end-to-end test, a real Loops
trainer, so they never run in CI and skip unless asked for:

- `LOOPS_LIVE_BLAXEL=1 uv run --env-file .env pytest tests/live -m live_blaxel`: one
  small sandbox, about three minutes. Checks that a command's whole output reaches
  Harbor past Blaxel's 64 KB process record, and that Blaxel still marks a cut record
  the way the cookbook looks for. Also runs the
  [episode_basics](../examples/episode_basics/) sandbox examples' file transfer, verifier
  retry and cancellation cleanup. Needs `BL_WORKSPACE` and `BL_API_KEY`.
- `LOOPS_LIVE_LOOPS=1 uv run --env-file .env pytest tests/live -m live_loops`: one
  training step of `live/output_job` on Qwen3.5-4B, about twenty minutes, and one of
  `live/vision_job` on GLM-5.3-Flash: images a tool returns, over several turns, with
  router replay checking that the sampler and trainer agree on the image
  placeholders. Needs `BASETEN_API_KEY` as well; each test deactivates its Loops run
  and requires `loops cleanup` to prove every resource released. It also runs the
  trainer normalization contract, which needs the `custom-loss` extra and fails without
  it: one trainer, zero-learning-rate steps comparing CISPO gradient norms against an
  explicitly reduced custom loss.

## Fixtures

Shared fixtures live in the nearest `conftest.py`. At the root: `config`,
`loops_capabilities` (what every test but a live one is told Loops serves),
`loops_check` (an offline `loops check`), the Blaxel fakes (`blaxel`,
`blaxel_services`, `blaxel_sandbox`) and the episode rig's `make` and `script`. Under
`unit/`: in-memory fakes (`fake_client`, `fake_prime`, `prime_renderer`). Under
`contracts/`: the model fixtures (`glm_tokenizer`, `flash_processor`, `flash_renderer`,
`qwen_tokenizer`, `cached_vlm_tokenizer`). The fakes themselves are plain classes in
`support/` (`fakes.py`, `harness.py`, `fake_blaxel.py`, `offline_run.py`, ...);
`ScriptedSampler` and `ScriptedRenderer` replace per-file stubs.
