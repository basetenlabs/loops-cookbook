# Renderers

A renderer turns an agent's messages and tool schemas into the exact token IDs a
model expects, and parses the model's completion back into content, reasoning, and
tool calls. Training is only as correct as this step: a prompt rendered differently
from how the model was trained teaches the model the wrong thing.

The cookbook uses [Prime Intellect's renderers](https://github.com/PrimeIntellect-ai/renderers)
as the provider. Prime implements each model family's chat template natively, in
Python, rather than through the tokenizer's Jinja template, which is what makes
incremental rendering (the *bridge*) and reliable parsing possible.

## Is my model supported?

A model being available in Loops does not mean its chat template is supported here.
If you do not set `[model] renderer`, the cookbook selects its own renderer for the
models in the table below, and Prime selects a renderer from the tokenizer's name for
any other. Where Prime's renderer does not match a model's template, the cookbook
overrides it with its own and selects that one instead.

Qwen3.5 and Qwen3.6 (Qwen3.6-35B-A3B, for example) get Prime's renderer, which matches
their templates: `tests/contracts/test_qwen_renderer.py` checks its prompts, and the
image tests check its images. GLM-5.3 and GLM-5.3-Flash get the cookbook's.

The cookbook ships a renderer for these models, each tested against the model's own
chat template, and selects it from the tokenizer's name:

| Model | Renderer | Why |
| --- | --- | --- |
| Muse Spark Open | `loops_cookbook.llm.renderer_extensions.muse_spark:renderer` | Prime has no renderer for its format. Prompts come from the model's chat template; the cookbook's renderer adds the parser and the bridge. Pin `current_date` in `renderer_chat_template_kwargs`: the template defaults it to today. |
| GLM-5.3, GLM-5.3-Flash | `loops_cookbook.llm.renderer_extensions.glm53:renderer` | Prime's GLM-5.3 renderers strip the whitespace the template keeps around reasoning, and Prime's Flash renderer writes one `<|image|>` per image, where the trainer needs one per vision token. The cookbook's renderer matches the template and the image processor. |
| Nemotron 3.5 Super VL (BF16, NVFP4) | `loops_cookbook.llm.renderer_extensions.nemotron:super_vl` | Prime does not recognise the model. Its Nemotron 3.5 renderer, written for 3.5 Lightning, has no `medium_effort` and renders an image after text differently from the template. The cookbook's renderer renders text and images as the template and image processor do. |
| Nemotron 3 Ultra | `loops_cookbook.llm.renderer_extensions.nemotron:ultra` | It renders as Prime does, and it also extends a conversation turn by turn under `medium_effort`, where Prime renders every turn again. |

So a Nemotron 3.5 Super VL config needs only the model:

```toml
[model]
base_model = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
renderer_chat_template_kwargs = { medium_effort = true }
```

Setting `[model] renderer` overrides the selection. One renderer is opt-in:

| Model | `[model] renderer` | Why |
| --- | --- | --- |
| Qwen3.5 | `loops_cookbook.llm.renderer_extensions.qwen35:renderer` | It renders as Prime does, but it also parses tool calls that have whitespace after `<parameter=`, as vLLM does. |

The cookbook's selection goes by the tokenizer's name. With a tokenizer from a local
directory or another repo (see below), name the renderer from the table in
`[model] renderer`. If Prime picks its own GLM-5.3 renderer for a model the table
does not list, `loops check` fails and names the value to set. The
[extensions README](../../src/loops_cookbook/llm/renderer_extensions/README.md) lists
how Prime's renderers differ from the templates.

`loops check` builds the renderer and prints which one it got and where it came
from, for example `Renderer: Qwen3Renderer (auto-detected by Prime)`. Read that line
when you try a new model. The usual failure is that Prime selects the renderer of a
related model family. `run` logs the same choice as
`Renderer for <model>: <class> (<source>); tokenizer <where it loaded from>; images ...`.

If you are not sure the auto-detected renderer matches the model's template, render
a few histories with your agent's real tool schemas and compare the token IDs to
`tokenizer.apply_chat_template(...)` for the same input. `tests/contracts/test_qwen_renderer.py`,
`tests/contracts/test_glm53_renderer.py` and `tests/contracts/test_nemotron_extension.py` show how.

If auto-detection would fall back to the generic tokenizer chat template, renderer
creation raises an error. When that template is intentional, select Prime's
`DefaultRenderer` explicitly:

```toml
[model]
base_model = "org/model"
renderer = "loops_cookbook.llm.factory:default_renderer"
```

## A model whose tokenizer is published under another name

The renderer needs the policy's Hugging Face tokenizer, and by default loads it from
the repo named `base_model`. Some models Loops serves have no Hugging Face repo under
their Loops name: an early-access model, for example, whose weights and tokenizer
live in a private repo. Name where the tokenizer is with `tokenizer`, and keep
`base_model` as the name Loops serves, since that is what provisions the trainer and
sampler and what `check` asks Loops about:

```toml
[model]
base_model = "org/model"
tokenizer = "org/model-tokenizer"
renderer = "loops_cookbook.llm.factory:default_renderer"
```

`tokenizer` is a Hugging Face repo id or a local directory holding the tokenizer
files (`tokenizer.json`, `tokenizer_config.json`, and the chat template). A local
directory starts with `./`, `../`, `/` or `~`, and anything else is a repo id:
`tokenizers/nemotron-35` names a repo, `./tokenizers/nemotron-35` a directory. A
repo id that a directory of the same name in the working directory would shadow is
refused rather than loaded from either. A
relative directory is relative to the config file that names it (a symlink followed
to its file), or to the shell
when it comes from `--model.tokenizer`, like every other path a config names, and
`check` refuses one that is not a directory. Every load
of the policy's tokenizer reads it: the renderer for RL and SFT, both
`check` and `run`, and the SFT `loops check` preview of a tokens dataset. An image
processor a renderer loads on demand comes from the same place, since Prime and the
cookbook's renderers load it from the tokenizer's own `name_or_path`. Prime refuses to
load the Nemotron 3.5 Super VL processor, which runs the repo's own code, from a local
directory, so with a local tokenizer that model renders text, and an image part fails. `loops check`
prints `tokenizer from <tokenizer>` on its renderer line when the setting is used.

A private or gated repo authenticates with `HF_TOKEN`, which Hugging Face's libraries
read from the environment; put it in the `.env` beside the config (see
`.env.example`).

Prime's auto-detection and the cookbook's selection both go by the tokenizer's
`name_or_path`, which is now the tokenizer's repo or directory rather than the model's
published name, so neither will usually recognise the model. Set `renderer` explicitly, to a model-specific
renderer or to `default_renderer` when the model's own chat template is intended.

There is no revision setting. To pin a tokenizer from a repo that changes, download
the revision you want beside the config and point `tokenizer` at the directory:

```sh
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 --revision <commit> \
  --include "tokenizer*" "chat_template.jinja" "*config.json" --local-dir tokenizers/nemotron-35
```

```toml
[model]
tokenizer = "./tokenizers/nemotron-35"
```

A resume compares `tokenizer` and `renderer` with the run it continues, and warns when
either changed: the run trains on, but its two halves may not have rendered alike.

## Model-specific options

`renderer_chat_template_kwargs` is passed through to the renderer. What it accepts is
model-specific, for example `{ enable_thinking = false }` for Qwen or
`{ reasoning_effort = "high", clear_thinking = false }` for GLM-5.3, or
`{ medium_effort = true }` for Nemotron 3 Ultra and 3.5 Super VL. There is no
universal flag for reasoning retention; each model's renderer decides how prior
`reasoning_content` is rendered.

## Adding a renderer for an unsupported model

Write a factory next to your training config and point at it:

```toml
[model]
base_model = "org/New-Model"
renderer = "my_renderers:new_model"
renderer_chat_template_kwargs = { some_option = true }
```

```python
# my_renderers.py
from renderers.newmodel import NewModelRenderer, NewModelRendererConfig  # a Prime renderer

from loops_cookbook.llm import PrimeRenderer


def new_model(tokenizer, chat_template_kwargs):
    renderer = NewModelRenderer(tokenizer, NewModelRendererConfig(**chat_template_kwargs))
    return PrimeRenderer(renderer)
```

The factory receives the model's Hugging Face tokenizer (from `tokenizer`, default
`base_model`; see above) and the
`renderer_chat_template_kwargs` dict, and returns anything satisfying the `Renderer`
protocol. Wrapping a Prime renderer in `PrimeRenderer` is the usual route: you get
render, parse, stop tokens, and the bridge for free, and the cookbook's message and
tool validation on top. For a model Prime does not cover, subclass the nearest Prime
renderer (the GLM-5.3 extension in
`loops_cookbook/llm/renderer_extensions/glm53.py` is a complete example) or
implement the protocol directly.

## The `Renderer` protocol

```python
class Renderer(Protocol):
    @property
    def stop_sequences(self) -> tuple[int, ...] | tuple[str, ...]: ...
    def render(self, messages, tools) -> RenderedPrompt | Sequence[int]: ...
    def parse(self, prompt_ids, completion_ids, tools) -> ParsedReply: ...
    # optional:
    def bridge(self, prompt_ids, completion_ids, new_messages, tools, *,
               previous_messages, previous_prompt_input) -> RenderedPrompt | Sequence[int] | None: ...
```

- `render` must be deterministic and must produce exactly what the model saw in
  post-training for that history. The generation prompt is included. Text-only
  renderers may return plain token IDs; renderers that emit image placeholders return
  a `RenderedPrompt(ids, prompt_input)` whose `prompt_input` is the Loops `ModelInput`
  with `ImageChunk`s at the placeholder positions and `prompt_input.length == len(ids)`.
  `PrimeRenderer` does this for you.
- `parse` returns `ParsedReply(content, reasoning_content, tool_calls,
  reasoning_complete, native_message)`. `native_message` is the assistant turn in the
  format `render` accepts, so the agent can append it verbatim. Return `None` for it
  exactly when a tool call's `status` is not `OK`: no message renders back to what the
  model wrote, so the agent appends `history_message`, the turn as written, which only
  ever continues from the sampled tokens. Either mismatch -- `None` with every call
  `OK`, or a message with a call that is not -- is a `RendererFailure`, which fails the
  attempt.
- `stop_sequences` go with every sampling request. A caller's own stops of the same
  kind (token ids, or strings) are joined with them; stops of the other kind go as the
  caller gave them, since one request carries one kind. Either way the sampler still
  ends the turn at the model's own end-of-sequence tokens.
- `bridge`, if present, is asked to extend `prompt_ids + completion_ids` with the new
  messages. Return `None` to decline. The result must start with exactly those
  tokens; the client checks and raises if it does not. `previous_prompt_input` is the
  previous prompt's chunk structure, a `ModelInput` built for this call from the
  captured sample; an image-capable renderer must decline when it is missing, since
  earlier images cannot be carried forward without it.
  `prompt_ids` is the captured prompt itself, a `TokenSpan` over the attempt's token
  buffer (four bytes a token, read as a tuple); convert it with `list()` only where a
  library needs one.
- `RenderedPrompt.ids` is a `TokenSpan` whatever the renderer returned: it compares,
  hashes and slices as the tuple of its tokens.

The renderer instance is shared across all conversations in a run and must not keep
per-conversation state. A renderer with a true `thread_safe` attribute is called from
several threads at once: an RL run calls its `render` and `bridge` off the event loop,
on a pool of a few threads (`RenderPool`), so one long render does not hold up every
other rollout, and its `parse` stays on the loop. Any other renderer is called on the
loop, one call at a time. `PrimeRenderer` is `thread_safe` for Prime's text renderers,
which keep nothing between calls, and not for its multimodal ones, which cache images
and load their processor on first use. A thread-safe renderer's tokenizer must not have
truncation or padding turned on by anyone sharing it: Hugging Face sets those on the
shared backend for the length of a call, so a concurrent encode would see them. That
holds for the transformers versions `pyproject.toml` allows, and
`tests/unit/llm/test_tokenizer_sharing.py` is what qualifies another. A custom factory's
result without `render`, `parse` or `stop_sequences` is refused when it is built, and
one that handles images must also have a true `supports_images` property, or it is
reported and probed as text only.

## Images

Install the vision extra, which brings torch and torchvision for the model's Hugging
Face image processor. Nothing runs on a GPU on the host, so CPU wheels are enough:

```sh
uv sync --extra vision
```

It works out of the box on every supported host. On Linux, PyPI's torch wheel
bundles CUDA libraries (several gigabytes) even though nothing here uses a GPU. That
is only a download-size cost; if you want the small CPU-only build, add PyTorch's CPU
index to your own project (this repository does it for its own development, but a
package cannot pass index settings on to the projects that depend on it):

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
torchvision = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
```

Supported hosts for the vision extra: Apple Silicon macOS 14+, Linux x86_64 with
glibc 2.28 or newer, Windows x64.

Image parts then flow through for renderers where one image renders to exactly the
token run the trainer expects: Qwen3.5, Qwen3.6, Qwen3.8, Qwen3-VL, and GLM-5.3-Flash
through the cookbook's GLM-5.3 extension. The `contracts` CI job checks the GLM path
against the HF processor's own expansion and the Qwen paths against Prime's placeholder
runs and the real image processors' token counts; both paths have run end to end
through Loops (sampling, grading, datums, audit).

One image has to render to the same tokens the sampler feeds the model, and three
independent checks hold it there. The sampler validates vLLM's expanded prompt against
its own before returning a completion. The runner compares the sampler's echoed
`prompt_token_ids` against the tokens it submitted, on every generation. For a text
prompt this proves little: when the sampler echoes nothing, baseten-loops 0.28 fills the
field with the submitted ids, so the comparison is with itself. It bites on image
prompts, where the SDK has no ids to fill in: an image prompt the sampler does not
echo fails, and one it does is compared token for token. Router replay is the
strongest of the three: the trainer checks every token it scores against the
sampler's record of the generation it came from. It reaches only text prompts, which
are all Loops records, and only MoE models, because a dense model has no routes to
replay -- there set `router_replay = false`, or each generation is recorded for
nothing.

Gemma 4 follows Prime's Qwen convention and should work but has not been run. Kimi K2.5
renders a single placeholder per image and relies on the engine to expand it; the
cookbook rejects Kimi image parts with a clear error. Kimi is on the
[roadmap](../../ROADMAP.md).

GLM-5.3-Flash: the extension writes each image as `<|begin_of_image|>` + N×`<|image|>`
+ `<|end_of_image|>` inline in the message text, exactly as the HF processor expands
the chat template, with N from the model's image processor. A user string that
literally contains `<|image|>` is tokenized as that special token, which is also what
the HF template does.

GLM-5.3 also accepts its template's bundled tool shape, one tool message whose content
is a list of `{"tool_call_id": ..., "output": ...}` entries, with images inside an
`output` list. For Prime's Qwen renderers the cookbook rejects bundled entries
carrying images with a message pointing at the flat form below; Prime's Qwen
renderers refuse text-only bundled entries themselves.

Put images in user or tool message content lists in any shape Prime accepts:

```python
{"type": "image", "image": png_bytes}                       # bytes, PNG or JPEG
{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
{"type": "image", "image": pil_image}                       # re-encoded as JPEG
{"type": "image", "path": "/tmp/screen.png"}                # or url: "https://..."
```

Prime renders each image to the model's placeholder tokens and reports where they
are; the cookbook turns that into the text-and-image chunk structure the Loops
sampler and trainer expect. The sampler expands the same placeholders and echoes the
prompt, and the client refuses to train if that echo differs from what was rendered.
The Loops trainer recomputes each image's token count with the model's processor
and refuses inputs whose `expected_tokens` disagree, so a bad count fails at the
first update rather than silently.

PNG and JPEG payloads, as bytes or base64 data URIs, are sent verbatim. Everything
else (PIL objects, paths, URLs, other encodings) is loaded and re-encoded as JPEG,
which keeps the dimensions and so the token count. Pass bytes or data URIs where you
can: the client normalizes image parts to bytes on every call over your full history.
Bytes and data URIs are decoded once per conversation, however many turns resend them;
a PIL object, path or URL can change between calls, so it is re-read each turn, and a
URL whose content changes would also defeat the incremental bridge.

`loops check` prints `Images: supported (N tokens for a 32x32 probe)` when the
renderer handles images, and `Images: not supported by this renderer (text only)`
otherwise. The first render downloads the model's processor config.

Run records (`samples/*.jsonl`, `batches/step-N.json`, `transcripts/*.json`) never
contain image bytes, only a SHA-256 digest and size per image.

Images are rendered in user and tool messages, directly in the content list. GLM-5.3
also renders images in system and assistant messages and inside bundled tool `output`
entries (above); the Qwen renderers reject both of those. Text-only renderers reject image parts
rather than dropping them. Video and audio parts are rejected everywhere.
