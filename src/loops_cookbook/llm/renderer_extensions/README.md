# Renderer extensions

This directory holds the cookbook's own renderers. Each one subclasses a renderer from
Prime Intellect's `renderers` package, which this file calls Prime, and changes only
what Prime gets wrong for one model. `COOKBOOK_RENDERERS` in `llm/factory.py` selects
them by the tokenizer's name. This file is for maintainers. It says how each extension
is built and how to keep it correct. To find out which renderer a model needs and how
to select it, see
[docs/reference/renderers.md](../../../../docs/reference/renderers.md).

## GLM-5.3 (`glm53.py`)

The GLM-5.3 extension renders GLM-5.3 and GLM-5.3-Flash so that the tokens match the
model's Hugging Face chat template. Prime has its own GLM-5.3 and GLM-5.3-Flash
renderers. On `renderers` 0.2.0, they differ from the template and the image processor
in two ways, and each one changes the tokens that train:

- They strip the whitespace around reasoning. The template renders
  `<think>\nplan\n</think>`, and Prime renders `<think>plan</think>`.
- Prime's Flash renderer writes one `<|image|>` per image and leaves the inference
  engine to expand it. The trainer needs the expanded form, one `<|image|>` per
  vision token, which the extension renders as the Hugging Face processor does.

Before 0.2.0, Prime also declared tools that have `defer_loading` set, and it rendered
a tool message holding several results as one empty `<tool_response>`. Version 0.2.0
matches the template on both. The extension's overrides for them now give the same
tokens as Prime.

The extension subclasses Prime's GLM-5.1 renderer, because Prime had no GLM-5.3
renderer when the extension was written. From GLM-5.1 it gets the completion parser,
the stop tokens and the bridge. The bridge extends the sampled tokens with new messages
and doesn't render the history again. The extension's overrides add the GLM-5.3 rules:
the reasoning-effort prefix, the thinking format, the tool declarations and the tool
results.

GLM-5.3 orders the results in a tool message by the IDs of the calls that the
previous assistant turn made. Before the bridge runs, `prepare_extension` reads those
IDs and orders only the new tool results. If an ID is missing, duplicated or matches
no call, the results keep the order they came in, as in the Hugging Face template.
This step reads the previous assistant turn but doesn't render it again. The cookbook
calls `prepare_extension` with the history it already keeps, so an agent doesn't call
it.

`tests/contracts/test_glm53_renderer.py` and `tests/contracts/test_glm53_images.py`
pin the reference template revision. They compare the extension's tokens with the
Hugging Face template and processor, and they test the bridge.

## Muse Spark Open (`muse_spark.py`)

Prime has no renderer for Muse Spark Open's format. The extension subclasses Prime's
`DefaultRenderer`, so it renders prompts with the model's own chat template. It adds
the two things a template can't do, a parser and a bridge.

An assistant turn is a run of messages, each written as
`<|start|>assistant to=<recipient><|message|>...`. The parser reads each message by its
recipient. A message to `self` is reasoning. A message to `commentary` or `user` is
text. A message to any other recipient is one tool call, written as
`<atem:function_calls>`. The template's examples name calls `tool.function`, so for a
tool without a namespace the parser reads a doubled name such as `bash.bash` as `bash`.

The parser types arguments as Prime does. It keeps a string parameter as written and
reads any other parameter as JSON. A value that isn't JSON makes the call invalid,
instead of passing on as a string. The parser reports a call message as malformed,
instead of reading part of it, unless the message is exactly one block holding one
invocation whose parameters are complete and distinctly named.

The bridge keeps the sampled turn as it was and writes the new messages as the template
does. A turn cut off at the token limit gets the `<|eot|>` that ends a turn, as Prime's
renderers close one. The bridge declines unless every new message is a tool result or
a user, system or developer message. When it declines, the whole conversation is
rendered again.

The template writes each tool result under its tool's name, which it finds through the
result's `tool_call_id`. Before the bridge runs, `prepare_extension` gives each result
the name of the previous assistant turn's call with that ID. If that turn's calls
didn't parse, they have no IDs. The bridge then names the result after the sampled
call in the same position.

The cookbook selects this renderer for Muse Spark Open NVFP4 by the tokenizer's name.
The template sets `current_date` to today by default, so set it in
`renderer_chat_template_kwargs`. Otherwise the system prompt changes during a run.

`tests/contracts/test_muse_spark_extension.py` pins the template. It compares prompts
and the bridge with the Hugging Face template, and tests parsing. The repo is gated, so
the tests skip unless `LOOPS_MUSE_TOKENIZER_CACHE` points at a hub cache that holds the
pinned snapshot, or `LOOPS_DOWNLOAD_CONTRACT_MODELS=1` downloads it with an `HF_TOKEN`
that has access.

## Qwen3.5 tool-call parsing (`qwen35.py`)

The Qwen3.5 extension is opt-in. It renders exactly as Prime's Qwen3.5 renderer does,
and it changes only how tool calls are parsed. The template writes `<parameter=NAME>`,
and the vLLM Qwen3 parser also accepts whitespace after `=`. Prime puts that whitespace
in the parameter name, so the extension removes it. Use the extension with the model's
own tokenizer.

The extension still refuses malformed or ambiguous calls. For example, it reports
tool-call markup inside a parameter value as `malformed_structure`. In well-formed
markup, the extension keeps tool and parameter names exactly as the model wrote them,
so the tool layer reports an unknown tool or argument. The raw text and the token spans
stay as they were sampled.

`tests/contracts/test_qwen35_extension.py` runs the extension on the Qwen3.5-9B
tokenizer at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

## Nemotron 3 Ultra and 3.5 Super VL (`nemotron.py`)

For text, Nemotron 3.5 Super VL's chat template is Ultra's: the same reasoning format
and the same options, `enable_thinking`, `truncate_history_thinking` and
`medium_effort`. Super VL adds images. Prime has no renderer that matches it:

- Prime maps only the early-access Super checkpoint, not the published ones, to its
  Nemotron 3.5 renderer.
- That renderer was written for 3.5 Lightning, whose template has no
  `medium_effort`, so its config has none.
- It writes an image that follows text with no separator. The template puts a
  newline between them.
- Its bridge declines whenever earlier image data is passed or a new message holds an
  image, so a conversation with images renders in full every turn.

`super_vl` subclasses Prime's Nemotron 3.5 renderer for its image processing. It takes
Ultra's config, and it renders each message's text and images as the template's
`render_content` does. The image processor expands each `<image>` into `<img>`, then
one `<image>` per vision token, then `</img>`, as Prime's renderer does.

Both renderers bridge where Prime's decline. Under `medium_effort`, Prime's Ultra
bridge declines, because a new user message would move the hint off the rendered
prompt. When no user message is added, the hint stays where it was, and the new
turns' tokens are those of the same renderer without the hint, so the extension
bridges through that renderer. Super VL bridges with images too. The template numbers
images within one message, so the new turns' tokens don't depend on earlier images.
Its bridge renders the new messages' images and reports the earlier images'
placeholders ahead of them.

`ultra` is Prime's Ultra renderer with only that bridge change, so the cookbook also
selects it for Ultra FP8, which Prime maps to the same template.

`tests/contracts/test_nemotron_extension.py` pins Ultra BF16 and Super VL BF16. It
also pins Super VL NVFP4's processor, because that checkpoint ships its own processor
config. The tests compare prompts with the template under every combination of the
three options. They compare images with each checkpoint's processor, and bridged turns
with the whole conversation rendered by the template.

## When you change the Prime pin

The extensions use private Prime methods and constants, and a Prime release can
change them without notice. Before you change the `renderers` pin, run the contract
tests above on the new version. The tests skip unless they can get the pinned
tokenizers, so set `LOOPS_DOWNLOAD_CONTRACT_MODELS=1` to download them, as CI does:

```sh
LOOPS_DOWNLOAD_CONTRACT_MODELS=1 uv run --all-extras pytest tests/contracts
```

If Prime fixes one of the differences above, remove the override that corrects it.
If Prime fixes all of a model's differences, remove the model's entries from
`COOKBOOK_RENDERERS` in `llm/factory.py`, so that auto-detection selects Prime's
renderer again. For GLM-5.3, also remove its entry from `_REFUSED_PRIME_RENDERERS`.
