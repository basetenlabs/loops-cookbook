# Model access

## Sampling: how an agent's model call becomes training evidence

Episodes can open a session with `sample.model.conversation(producer=...,
conversation_id=...)`; `sample.model.llm(producer).conversation()` remains available.
Reuse the session across turns and supply history on each `generate()` call.
`generate()` explicitly accepts `temperature`, `max_tokens`, `top_p`, `top_k`,
`seed`, `stop`, and `structured_outputs`; unknown keywords are rejected.
Omitted options preserve existing defaults; `max_tokens=None`
uses the remaining trajectory budget. A `SamplingParams` object retains its
replacement semantics and cannot be mixed with keyword sampling options.

The agent gets a `LoopsLLM` and opens a `Conversation`. Each `generate(messages=...,
tools=..., ...)` call does four things:

1. **Render.** The renderer turns the message list and tool schemas into the exact
   prompt token IDs the model expects, using the model's own chat template through
   [Prime Intellect's renderers](renderers.md). If the new messages are the previous
   history plus appended non-assistant messages, and the tools are unchanged, the
   renderer's *bridge* extends the previously sampled tokens instead of re-rendering,
   so the prompt is guaranteed to start with the exact tokens that were sampled last
   turn. Any other change re-renders from scratch.
2. **Sample.** The token IDs go to the Loops sampler with the sampling parameters.
   `max_tokens` is lowered to the room the prompt leaves in the trajectory budget
   (`[sampling] max_trajectory_tokens`, else `[model] max_seq_len`), as a model API
   clamps it to the model's context, so a generation never runs past the budget; one
   that reaches it stops with `stop_reason` `"length"`. Only a prompt that leaves no
   room at all is refused, with `ContextWindowExceeded`. The response carries the
   completion token IDs, per-token logprobs, and the policy version that produced
   them; a completion longer than the `max_tokens` it was sampled with is not
   evidence, and fails the attempt as a `SamplingFailure`.
3. **Record.** A `Sample` is stored: prompt IDs, completion IDs, logprobs, policy
   version, stop reason, the `max_tokens` it was sampled with (and, in
   `clamped_from`, what was asked for when the budget lowered it), and which
   conversation and producer it came from. This is
   immutable and is the only thing training ever reads. The prompt is kept compactly:
   `prompt_ids` is a span of one buffer the conversation's prompts share, at four
   bytes a token, and `prompt_chunks` is the prompt's chunk structure over it, with
   image chunks held by reference. `sample.prompt_input` builds the Loops
   `ModelInput` from the two on each read; a reader that keeps one keeps a Python
   int per token. With `[output] trace` set to
   `samples` or `full`, it is also appended to `samples/*.jsonl` in the run directory.
4. **Parse.** The renderer parses the completion into `content`,
   `reasoning_content`, and `tool_calls`, and builds `native_message`, the assistant
   turn in the model's own format, for the agent to append to its history. A tool
   call the renderer could not parse is the model's output: the reply carries every
   call with its `status`, `native_message` is `None` and `parse_error` says why, and
   the generation trains like any other, so the verifier's verdict on what the agent
   does next is the signal. To answer the calls, append `history_message`, the turn
   as written: the next turn continues from its sampled tokens, never re-rendering it. A completion the renderer cannot read (the parse
   raises, or returns no `native_message` without reporting a tool call it could
   not parse) is the renderer's failure, not the model's: `generate` raises
   `RendererFailure`, and the attempt fails whether or not the agent catches it. It
   is retried as a fresh attempt up to `max_rollout_retries`, and nothing of it
   trains or is graded.

Text is never re-tokenized to build training data. If an agent edits its history,
the next call re-renders and the edited region becomes masked prompt; the earlier
sampled tokens stay attached to the earlier `Sample`.

`generate` also takes `reasoning_effort`. Each value declared in
`[model.reasoning_efforts]` maps to the template options it overrides, and the run
builds one renderer per value at startup; a generation uses the one it names, and
bridging only continues on the renderer that sampled the previous turn.

### Structured output

The sampler can constrain a generation to a JSON Schema, a regex, a grammar or a
fixed set of choices. Directly, pass the Loops SDK's `structured_outputs`, which goes
to vLLM unchanged, as any other sampling option:

```python
reply = await conversation.generate(
    messages=messages,
    structured_outputs={"json": {"type": "object", "properties": {...}}},
)
```

The keys are vLLM's: `json`, `regex`, `choice`, `grammar`, `json_object`. Over HTTP,
a `response_format` of `json_schema` becomes `{"json": schema}` and `json_object`
becomes `{"json_object": true}`; `strict` is accepted either way because the sampler
always enforces the schema, and a schema `description` is refused because nothing
renders it to the model. A JSON `response_format` is refused alongside offered tools,
since the constraint holds from the first token and no tool call could start; drop
the tools or send `tool_choice: "none"`, which renders without them. Say what
format you want in the messages too: the constraint only removes tokens that would
break it.

Use a constraint with thinking off. It holds from the first sampled token, so with
thinking on it replaces the reasoning, and the reply parses as reasoning with empty
content. The logprobs Loops records for a constrained generation are the constrained
distribution's; the trainer scores the unconstrained one, so the more the constraint
overrides what the model would have written, the further off-policy the sample is.

### Over HTTP

When the runtime's model access is `openai_http` (an `Episode` class or a Harbor agent
declares it as `model_access`; `[runtime] model_access` is only for a runtime that
supports several), a runtime calls
`services.model_access.http_connection(producer)` instead of `llm()`, and hands the
agent an OpenAI-compatible base URL, a key, and a model name. The key is bound to
that attempt and producer; nothing in a request can change either. The run has one
gateway, which listens on `[runtime.openai].bind` (loopback and an ephemeral port by
default), advertises `advertised_base_url` and one model, `model_name` (default
`[model] base_model`), and it serves
`POST /v1/chat/completions` and `GET /v1/models`, streamed or not, and
`POST /v1/loops/discard` (below).

Each request renders, samples, records, and parses exactly as above, through the
same run renderer. Send `X-Loops-Conversation-Id` to keep one conversation's
bridge across requests: the gateway recognises the assistant message it returned
when the agent echoes it back, so the prompt extends the sampled tokens as a
direct `Conversation` does. Without the header, every request is its own
conversation. Reasoning comes back in `reasoning_content`, never in `content`, and
`usage.completion_tokens_details.reasoning_tokens` counts its sampled tokens: those
before the reasoning closed, or all of them when the generation was cut first. Only
renderers with a single-token `</think>` (Prime's thinking renderers) report it. A
completion the renderer cannot read is a `500 server_error`, and fails the attempt
exactly as it does directly.

A field the gateway cannot honour is refused before anything samples, with
`unsupported_parameter` and the field's name, and nothing is accepted and ignored:
`logprobs`, constrained tool choice, and remote image URLs
(images are base64 PNG or JPEG data URIs, and only in user messages) are all refused.
A JSON `response_format` is honoured: see [Structured output](#structured-output).
A request whose prompt plus `max_tokens` would exceed the trajectory budget
(`[sampling] max_trajectory_tokens`, else `[model] max_seq_len`) is sampled with
`max_tokens` lowered to the room left, as Baseten Model APIs clamp it to the model's
context: if the generation reaches it, the completion ends with `finish_reason`
`"length"`, or `422 tool_call_truncated` when it was cut inside a tool call. A
prompt that alone fills the budget is `400 context_length_exceeded`, a history the
renderer cannot render is `400 rendering_failed`, and a request on a conversation
still generating is `409 conversation_busy`. None of these poison the attempt. A generation
that breaks the freshness limits is `409 rollout_stale`, and that one does: the attempt
is discarded.

A reply with a tool call the renderer could not parse is different: it was sampled,
it is the model's output, and it is returned as `422` with `type`
`model_output_invalid` and `code` `tool_call_truncated` when the generation was cut
short (by `max_tokens`, inside reasoning, or inside the call) or
`tool_call_parse_error` otherwise, streamed request or not. `error.reply` holds what
the model wrote: `id` (the completion's, as on a `200`), `content`,
`reasoning_content`, and `tool_calls`, each call as an assistant tool call (`id`,
`type`, and `function` with the `name` and the `arguments` as the model wrote them:
`""` for a name it did not write, the call's raw text for arguments that could not be
read) plus its `raw` text and parse `status`, and `usage`, the generation's token
counts as on a `200`. It counts as a delivery once written
whole, so it trains unless discarded by that `id`; what to do next is the agent's,
and a stock OpenAI client raises it as an `UnprocessableEntityError` without retrying.

To answer the calls, as an agent answers a tool call that failed, echo the reply as
an assistant message (`content` and each call's `id`, `type` and `function`) and send
a tool result per call, on the conversation named by `X-Loops-Conversation-Id`. The
gateway recognises the echo and continues from the sampled tokens: the next prompt is
the previous prompt plus the exact completion, so the model sees what it wrote and
the episode trains as one chain. Direct mode is the same: append
`reply.history_message`. A turn that did not parse is never rendered from text, since
no message renders back to what the model wrote; a request that cannot extend its
tokens (the tools, the renderer or the history before it changed, or the renderer has
no bridge or declines one, as a thinking-retention rule can) is `400 rendering_failed` (`RenderingError` directly), and the attempt goes
on. A reply discarded by its `id` did not happen, so echoing it back is refused the
same way: re-ask instead. Without the header, the echo is the agent's own history, and
arguments that are not JSON are refused.

When a request offers no tools (`tool_choice: "none"`, or no `tools` at all), a
well-formed tool call the model writes anyway is left out of the reply, which carries
the text alone and never finishes with `tool_calls`. What the model sampled is never edited, though: on a
conversation named with `X-Loops-Conversation-Id`, the gateway substitutes its
native message, call included, for the agent's echo of that reply, so the next
prompt continues the exact sampled tokens and the chain stays whole. Without the
header, the history is rendered from what the agent sent, which has no call.

A tool call whose model format carries no id gets one from the gateway,
`call_<sample_id>_<n>`, because Chat Completions requires it. The agent's tool result
echoes it back, and it only matches that result to its call: the native message the
next turn continues has no id, as in direct mode. A template that prints ids, such as
Kimi's, prints the model's own, which the gateway passes through; only a call the
model wrote with an empty id would show the gateway's.

A response counts as delivered only once all of it has been written, and only
delivered generations train. A client that disconnects after its generation sampled,
as one that times out and retries does, never received it, so that generation is
dropped and the rest of the episode trains; a stock OpenAI client needs nothing
special. An `Idempotency-Key` makes a retry replay the same response instead of
sampling again. Sealing, abort, and the rollout clock revoke the attempt's keys.

Every delivered response trains unless the agent says otherwise. An agent that
declines a turn it received — a losing candidate in best-of-n, a malformed tool call
it re-asks for — sends `POST /v1/loops/discard` with `{"id": "<the completion's
id>"}` under the same key, and that generation leaves the evidence exactly as a
direct `conversation.discard(reply)` does (see
[harbor.md](harbor.md)): only the most recent turn of its
conversation, only once, and only one the key's attempt and producer received. An
ID this key's attempt and producer never generated, another attempt's included, is
`404 generation_not_found`; one not delivered whole, already built on, or already
discarded is `409 generation_not_discardable`, and one on a conversation still
generating is `409 conversation_busy`; the attempt is not poisoned either way. Stock OpenAI
clients never call it, and without it every delivered turn trains.

Two freshness checks run as samples arrive. A rollout whose trainable samples span
more than `max_policy_delta_in_trajectory` policy versions is discarded. A rollout whose
oldest trainable sample is more than `max_steps_off_policy` versions behind the
current policy is discarded. Both are rechecked later, because the policy keeps
moving while the rollout waits for grading.
