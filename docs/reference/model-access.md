# Model access

Model access is how an agent's model calls reach the Loops sampler. The
rollout runtime declares one of two kinds:

- `direct` means an in-process agent calls the cookbook's Python client, `llm()`.
- `openai_http` means the agent sends requests to an OpenAI-compatible Chat Completions
  endpoint that the run serves. This page calls that endpoint the gateway.

Both kinds turn each model call into training evidence along the same path. This page
explains that path for a direct call first, and then what changes over HTTP.

## How a model call becomes training evidence

In a custom rollout runtime, an episode's `run()` opens a session with
`sample.model.conversation(producer=..., conversation_id=...)`. The longer form,
`sample.model.llm(producer).conversation()`, also works. Either way, the agent gets a
`LoopsLLM` client and opens a `Conversation` on it. A producer is a named source of
model calls. Each producer is declared in `[rollout.producers]`, and a call that names
none uses `main`. Reuse one session across the turns of a conversation, and pass the
full history on each `generate()` call.

`generate()` accepts these sampling options as keywords: `temperature`, `max_tokens`,
`top_p`, `top_k`, `seed`, `stop` and `structured_outputs`. It rejects any other keyword.
An option you leave out keeps the run's default. `max_tokens=None` lets the generation
use all the room left in the trajectory budget. The trajectory budget is the most
tokens that a prompt and its generation may use together. It is
`[sampling].max_trajectory_tokens`, or `[model].max_seq_len` when that isn't set.

You can pass a `SamplingParams` object instead of keyword options. The client takes the
object as given, so it replaces the run's defaults, although the trajectory budget
still applies. You can't mix a `SamplingParams` object with keyword options.

Each `generate(messages=..., tools=..., ...)` call does four things:

1. **Render.** The renderer turns the message list and the tool schemas into the exact
   prompt token IDs that the model expects. It uses the model's own chat template,
   through [Prime Intellect's renderers](renderers.md). When the new message list is
   the previous history plus new messages that aren't from the assistant, and the tools
   haven't changed, the renderer's *bridge* extends the previously sampled tokens
   instead of rendering the history again. This guarantees that the prompt starts with
   the exact tokens that were sampled on the last turn. Any other change renders the
   prompt again from scratch.
2. **Sample.** The client sends the token IDs to the Loops sampler with the sampling
   parameters. First it lowers `max_tokens` to the room that the prompt leaves in the
   trajectory budget, as a model API clamps `max_tokens` to the model's context. A
   generation therefore never runs past the budget, and one that reaches the budget
   stops with `stop_reason` `"length"`. The client refuses only a prompt that leaves no
   room at all, with `ContextWindowExceeded`. The sampler's response carries the
   completion token IDs, a logprob for each token, and the policy version that produced
   them. A completion longer than the `max_tokens` it was sampled with can't be used as
   evidence, so it fails the attempt as a `SamplingFailure`.

   Each request names the rollout ID that Loops traces it under. Every attempt at a
   rollout sends the same ID, so Loops counts a retried rollout once. The ID is a
   digest of the rollout and of a random value that the run picks when it starts.
   Two runs therefore never share an ID, even when they have the same name. The run
   saves that value in its checkpoint manifests, so a resumed run sends the same ID
   when it replays a rollout. The run
   files each attempt's samples and transcript under the attempt's own ID. With
   `[compute].sampler_replica_affinity` on, which is the default, each request also
   carries an affinity key. The key is the same for every attempt of a rollout, so
   Loops sends all of the rollout's requests to one sampler replica, whose prefix cache
   already holds the growing prompt. Affinity is a routing hint. When Loops can't pin
   a request, it serves the request without affinity.
3. **Record.** The client stores a `Sample`. A `Sample` holds the prompt IDs, the
   completion IDs, the logprobs, the policy version, the stop reason, the `max_tokens`
   it was sampled with, and the conversation and producer it came from. When the budget
   lowered `max_tokens`, `clamped_from` holds the value that was asked for. A `Sample`
   is immutable, and it is the only thing that training ever reads.

   The `Sample` keeps its prompt compactly. `prompt_ids` is a span of one buffer that
   all of the conversation's prompts share, at four bytes a token. `prompt_chunks` is
   the prompt's chunk structure over that span, and it holds image chunks by reference.
   `sample.prompt_input` builds the Loops `ModelInput` from the two each time it is
   read. A reader that keeps a `ModelInput` keeps a Python int for each token.

   With `[output].trace` set to anything but `off`, the run also appends each
   `Sample` to `samples/<attempt ID>.jsonl` in the run directory. Only `trace = "full"`
   writes the token arrays.
4. **Parse.** The renderer parses the completion into `content`, `reasoning_content` and
   `tool_calls`. It also builds `native_message`, which is the assistant turn in the
   model's own format, for the agent to append to its history.

   A tool call that the renderer couldn't parse is still the model's output. The reply
   carries every call with its `status`, `native_message` is `None`, and `parse_error`
   says why. The generation trains like any other, so the verifier's verdict on what
   the agent does next is the training signal. To answer the calls, append
   `history_message`, which is the turn as the model wrote it. The next turn continues
   from that turn's sampled tokens and never renders it again.

   A completion that the renderer can't read at all is the renderer's failure, not the
   model's. That happens when the parse raises an error, or when it returns no
   `native_message` without reporting a tool call that it couldn't parse. Then
   `generate` raises `RendererFailure`, and the attempt fails whether or not the agent
   catches the error. The run retries the rollout as a fresh attempt, up to
   `[rollout].max_rollout_retries` times, and nothing from the failed attempt trains or
   is graded.

The client never tokenizes text again to build training data. If an agent edits its
history, the next call renders the prompt again, and the edited part becomes masked
prompt, which doesn't train. The earlier sampled tokens stay attached to the earlier
`Sample`.

`generate` also takes `reasoning_effort`. Each value declared in
`[model.reasoning_efforts]` maps to the template options that it overrides, and the
run builds one renderer for each value at startup. A generation uses the renderer for
the value it names. The bridge continues only on the renderer that sampled the previous
turn.

## Structured output

The sampler can constrain a generation to a JSON Schema, a regular expression, a grammar
or a fixed set of choices. With direct access, pass the Loops SDK's
`structured_outputs` like any other sampling option. It goes to vLLM unchanged:

```python
reply = await conversation.generate(
    messages=messages,
    structured_outputs={"json": {"type": "object", "properties": {...}}},
)
```

The keys are vLLM's: `json`, `regex`, `choice`, `grammar` and `json_object`.

Over HTTP, send OpenAI's `response_format` instead. A `response_format` of
`json_schema` becomes `{"json": schema}`, and `json_object` becomes
`{"json_object": true}`. The gateway accepts `strict` with either value, because the
sampler always enforces the schema. It refuses a schema `description`, because nothing
renders the description to the model. It also refuses a JSON `response_format` when the
request offers tools. The constraint holds from the first token, so no tool call could
start. Drop the tools, or send `tool_choice: "none"`, which renders the prompt without
them.

Say what format you want in the messages too. The constraint only removes tokens that
would break it.

Use a constraint with thinking turned off. The constraint holds from the first sampled
token, so with thinking on, it replaces the reasoning, and the reply parses as
reasoning with empty content. The logprobs that Loops records for a constrained
generation come from the constrained distribution, but the trainer scores the
unconstrained one. So the more the constraint overrides what the model would have
written, the further off-policy the sample is.

## Over HTTP

An `Episode` class or a Harbor agent declares its model access in its `model_access`
attribute. Set `[rollout_runtime].model_access` only for a rollout runtime
that supports more than one kind.

When the model access is `openai_http`, the rollout runtime calls
`services.model_access.http_connection(producer)` instead of `llm()`. In an `Episode`,
that call is `sample.model.http_connection()`. It hands the agent three things: an
OpenAI-compatible base URL, a key and a model name. The key is bound to that attempt
and producer, and nothing in a request can change either one.

The run has one gateway. It listens on `[rollout_runtime.openai].bind`, which is a
loopback address and an ephemeral port by default. It advertises `advertised_base_url`
and one model, `model_name`, which defaults to `[model].base_model`. It serves
`POST /v1/chat/completions`, streamed or not, and `GET /v1/models`. It also serves
`POST /v1/loops/discard`, which is described below.

### Requests and conversations

The gateway renders, samples, records and parses each request exactly as a direct call
does, with the same run renderer.

Send the `X-Loops-Conversation-Id` header to keep one conversation's bridge across
requests. The gateway recognises the assistant message it returned when the agent
echoes that message back in the next request. The prompt then extends the sampled
tokens, as a direct `Conversation` does. Without the header, every request is its own
conversation.

Reasoning comes back in `reasoning_content`, never in `content`.
`usage.completion_tokens_details.reasoning_tokens` counts the reasoning's sampled
tokens. Those are the tokens before the reasoning closed, or all of them when the
generation was cut off first. Only renderers with a single-token `</think>`, which are
Prime's thinking renderers, report this count.

### Errors

Before anything is sampled, the gateway refuses any field that it can't honour, with
`unsupported_parameter` and the field's name. It never accepts a field and then ignores
it. For example, it refuses `logprobs`, constrained tool choice and remote image URLs.
Images must be base64 PNG or JPEG data URIs, and the gateway accepts them only in user
messages. A JSON `response_format` is honoured, as [Structured output](#structured-output)
describes.

A request whose prompt plus `max_tokens` would exceed the trajectory budget is sampled
with `max_tokens` lowered to the room left, as Baseten Model APIs clamp it to the
model's context. If the generation reaches that limit, the completion ends with
`finish_reason` `"length"`. If the generation was cut inside a tool call, the response
is `422 tool_call_truncated` instead.

The errors below don't poison the attempt, so the attempt goes on:

- `400 context_length_exceeded` means the prompt alone fills the budget.
- `400 rendering_failed` means the renderer can't render the history.
- `409 conversation_busy` means the request names a conversation that is still
  generating.

`409 rollout_stale` means the generation broke the freshness limits, which
[Freshness checks](#freshness-checks) describes. This error does poison the attempt,
and the attempt is discarded.

A completion that the renderer can't read is a `500 server_error`. It fails the attempt
exactly as it does with direct access.

### When a tool call doesn't parse

A reply with a tool call that the renderer couldn't read whole is different. It was
sampled, and it is the model's output, so the gateway returns it with status `422`, a
`type` of `model_output_invalid`, and a `code`. The code is `tool_call_truncated` when
the generation was cut short, whether by `max_tokens`, inside the reasoning or inside
the call. Otherwise the code is `tool_call_parse_error`, which covers a call whose
structure broke and a call without a name. This holds whether or not the request was
streamed.

A call that the renderer read whole is a `200` like any other, whatever the parser made
of it. With the cookbook's renderers, a call to a tool that the request didn't declare
reaches the agent as the model wrote it, and so does an argument that isn't its type's
JSON. The agent's tool layer judges them. A model on Prime's own parser, which may read
part of a block it marks `invalid_json` or `unknown_tool`, gets the `422` for those, as
before.

`error.reply` holds what the model wrote:

- `id` is the completion's ID, as on a `200` response.
- `content` and `reasoning_content` are the reply's text and reasoning.
- `tool_calls` lists each call as an assistant tool call, with `id`, `type` and
  `function`. `function` holds the `name` and the `arguments` as the model wrote them.
  The name is `""` when the model wrote none. The arguments are the call's raw text when
  they couldn't be read. Each call also carries its `raw` text and its parse `status`.
- `usage` holds the generation's token counts, as on a `200` response.

The gateway counts this response as delivered once it has written all of it, so it
trains unless the agent discards it by its `id`. What to do next is up to the agent. A
stock OpenAI client raises the response as an `UnprocessableEntityError` and doesn't
retry it.

To answer the calls, do what an agent does for any tool call that failed. Echo the reply
as an assistant message, with `content` and each call's `id`, `type` and `function`.
Then send one tool result per call, on the conversation named by
`X-Loops-Conversation-Id`. The gateway recognises the echo and continues from the
sampled tokens. The next prompt is the previous prompt plus the exact completion, so
the model sees what it wrote, and the episode trains as one chain. Direct access works
the same way when you append `reply.history_message`.

A turn that didn't parse is never rendered from text, because no message renders back
to what the model wrote. So a request has to extend that turn's sampled tokens. A
request can't do that when the tools, the renderer or the history before the turn
changed. It also can't when the renderer has no bridge or declines to bridge, which a
renderer's rule for keeping earlier reasoning can cause. The gateway refuses such a
request with `400 rendering_failed`, and direct access raises `RenderingError`. Either
way, the attempt goes on.

A reply that was discarded by its `id` didn't happen, so the gateway refuses an echo of
it in the same way. Ask again instead. Without the conversation header, the echo is the
agent's own history, and the gateway refuses arguments that aren't JSON.

### A request that offers no tools

A request offers no tools when it sends `tool_choice: "none"` or no `tools` at all. If
the model writes a well-formed tool call anyway, the gateway leaves the call out of the
reply. The reply carries the text alone, and it never finishes with `tool_calls`.

The gateway never edits what the model sampled, though. On a conversation named with
`X-Loops-Conversation-Id`, the gateway substitutes its own native message, call
included, for the agent's echo of that reply. The next prompt then continues the exact
sampled tokens, and the chain stays whole. Without the header, the gateway renders the
history from what the agent sent, which has no call.

### Tool call IDs

Chat Completions requires an ID on every tool call. When the model's format carries no
ID, the gateway gives the call one, `call_<sample_id>_<n>`. The agent's tool result
echoes that ID back, and the gateway uses it only to match the result to its call. The
native message that the next turn continues has no ID, as with direct access.

A template that prints IDs, such as Kimi's, prints the model's own IDs, and the gateway
passes them through. Only a call that the model wrote with an empty ID would show an ID
from the gateway.

### Delivery, retries and discards

A response counts as delivered only once the gateway has written all of it, and only
delivered generations train. A client that disconnects after its generation was sampled
never received that generation, so the gateway drops it, and the rest of the episode
trains. A client that times out and retries disconnects in this way, so a stock OpenAI
client needs nothing special.

An `Idempotency-Key` header makes a retry replay the same response instead of sampling
again. The attempt's keys are revoked when the attempt is sealed or aborted, or when its
rollout clock runs out.

Every delivered response trains unless the agent says otherwise. An agent may decline a
turn it received, such as a losing candidate in best-of-n, or a malformed tool call that
it asks for again. To decline it, the agent sends `POST /v1/loops/discard` with
`{"id": "<the completion's id>"}`, authenticated with the same key. The gateway then
removes that generation from the evidence, so it doesn't train. This works exactly as a
direct `conversation.discard(reply)` does, which [harbor.md](harbor.md) shows, and the
same rules apply. An agent can discard only the most recent turn of a conversation,
only once, and only a turn that the key's attempt and producer received.

The discard endpoint returns these errors, and none of them poisons the attempt:

- `404 generation_not_found` means that this key's attempt and producer never generated
  that ID. This includes an ID from another attempt.
- `409 generation_not_discardable` means that the generation wasn't delivered in full,
  was already built on, or was already discarded.
- `409 conversation_busy` means that the generation's conversation is still generating.

Stock OpenAI clients never call this endpoint, and without it every delivered turn
trains.

## Freshness checks

Two freshness checks run as samples arrive. The run discards a rollout whose trainable
samples span more than `[training].max_policy_delta_in_trajectory` policy versions. It
also discards a rollout whose oldest trainable sample is more than
`[training].max_steps_off_policy` versions behind the current policy. The run checks
both again later, because the policy keeps moving while the rollout waits for grading.
