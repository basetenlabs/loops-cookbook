# LiteLLM, streaming

[A custom runtime, in-process, openai_http](../../patterns/custom_runtime/in_process_openai_http/)
whose agent is built on [LiteLLM](https://github.com/BerriAI/litellm) and streams every
reply. `litellm_episode.py` is that pattern's `AnswerEpisode` with only the client
changed: `run()` calls `litellm.acompletion(..., stream=True)` with the attempt's
endpoint as an OpenAI-compatible `api_base`, exactly as it would call any other
provider, and assembles the chunks.

The run's gateway samples each reply whole and records its exact tokens, then streams
it back as server-sent events: the chunks arrive together once the reply is done, not
token by token. Streaming works unchanged; it just doesn't make the first token
arrive sooner.

LiteLLM is not a cookbook dependency, so run the example with it added:

```sh
uv run --with 'litellm!=1.107.0' loops check examples/litellm_streaming/train.toml
uv run --with 'litellm!=1.107.0' loops run   examples/litellm_streaming/train.toml
```

`check` launches nothing; `run` provisions Loops resources, which stay provisioned
until you deactivate the run. The run ID is logged on exit.
