# LiteLLM, streaming

This example trains an agent built on [LiteLLM](https://github.com/BerriAI/litellm),
which streams every reply. It builds on the
[custom rollout runtime, in-process, openai_http](../../patterns/custom_runtime/in_process_openai_http/)
pattern. Start here when your agent is built on another framework that calls the model
through an OpenAI-compatible API.

## Run it

LiteLLM isn't a cookbook dependency, so run the example with LiteLLM added. Run these
commands from the repository root, with `BASETEN_API_KEY` set in `.env`:

```sh
uv run --with 'litellm!=1.107.0' loops check examples/litellm_streaming/train.toml
uv run --with 'litellm!=1.107.0' loops run   examples/litellm_streaming/train.toml
```

The version filter skips LiteLLM 1.107.0, which the cookbook excludes because of a
security advisory.

`loops check` launches nothing. `loops run` trains on serverless Loops. When the
command exits, it leaves the Loops run up, and it logs the run ID and the
`loops deactivate` command that ends the run.

## How it works

`litellm_episode.py` is the pattern's `AnswerEpisode` with only the client changed.
`run()` calls `litellm.acompletion(..., stream=True)` and passes the attempt's endpoint
as an OpenAI-compatible `api_base`, exactly as it would call any other provider. Then
it assembles the chunks into the reply.

The run's gateway samples each reply whole and records its exact tokens. Then it
streams the reply back as server-sent events. So the chunks arrive together once the
reply is done, not token by token. Streaming works unchanged, but it doesn't make the
first token arrive sooner.
