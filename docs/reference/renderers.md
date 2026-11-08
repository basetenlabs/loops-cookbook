# Renderers

A renderer turns an agent's messages and tool schemas into the exact token IDs that a
model expects. It also parses the model's completion back into content, reasoning and
tool calls. Training is only as correct as this step. A prompt rendered differently
from the way the model was trained teaches the model the wrong thing.

The cookbook uses [Prime Intellect's renderers](https://github.com/PrimeIntellect-ai/renderers),
which this page calls Prime. Prime implements each model family's chat template natively
in Python, rather than through the tokenizer's Jinja template. That makes two things
possible: reliable parsing, and incremental rendering, which the cookbook calls the
*bridge*. The bridge extends the tokens the model already sampled with the new
messages, instead of rendering the whole history again.

## Is my model supported?

A model that is available in Loops doesn't necessarily have a chat template that the
cookbook supports. If you don't set `[model] renderer`, the cookbook selects a renderer
by the tokenizer's name. For the models in the first table below, it selects its own
renderer. For any other model, Prime selects a renderer from the tokenizer's name.

Qwen3.5 and Qwen3.6, such as Qwen3.6-35B-A3B, get Prime's renderer, which matches their
templates. `tests/contracts/test_qwen_renderer.py` checks its prompts, and the image
tests check its images.

The cookbook ships its own renderer for the models in this table. Each one is tested
against the model's own chat template.

| Model | Renderer | Why |
| --- | --- | --- |
| Muse Spark Open | `loops_cookbook.llm.renderer_extensions.muse_spark:renderer` | Prime has no renderer for this model's format. Prompts come from the model's chat template, and the cookbook's renderer adds the parser and the bridge. The template sets `current_date` to today by default, so set it in `renderer_chat_template_kwargs`. |
| GLM-5.3, GLM-5.3-Flash | `loops_cookbook.llm.renderer_extensions.glm53:renderer` | Prime's GLM-5.3 renderers strip the whitespace that the template keeps around reasoning. Prime's Flash renderer writes one `<\|image\|>` per image, and the trainer needs one per vision token. The cookbook's renderer matches the template and the image processor. |
| Nemotron 3.5 Super VL (BF16, NVFP4) | `loops_cookbook.llm.renderer_extensions.nemotron:super_vl` | Prime doesn't recognise the model. Prime's Nemotron 3.5 renderer was written for 3.5 Lightning. It has no `medium_effort` option, and it renders an image that follows text differently from the template. The cookbook's renderer renders text and images as the template and the image processor do. |
| Nemotron 3 Ultra | `loops_cookbook.llm.renderer_extensions.nemotron:ultra` | It renders as Prime does. Under `medium_effort`, it also extends a conversation turn by turn, where Prime renders every turn again. |

A Nemotron 3.5 Super VL config therefore needs only the model:

```toml
[model]
base_model = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
renderer_chat_template_kwargs = { medium_effort = true }
```

Setting `[model] renderer` overrides the selection. One renderer is opt-in, and the
cookbook never selects it for you:

| Model | `[model] renderer` | Why |
| --- | --- | --- |
| Qwen3.5 | `loops_cookbook.llm.renderer_extensions.qwen35:renderer` | It renders as Prime does, but it also parses tool calls that have whitespace after `<parameter=`, as vLLM does. |

The cookbook selects its renderers by the tokenizer's name. If you load the tokenizer
from a local directory or from another repository, as the next section describes, set
`[model] renderer` to the renderer from the table.

If Prime picks its own GLM-5.3 renderer for a model that the table doesn't list,
`loops check` fails and names the value to set. The
[extensions README](../../src/loops_cookbook/llm/renderer_extensions/README.md) lists
how Prime's renderers differ from the templates.

`loops check` builds the renderer and prints which one it got and where it came from,
for example `Renderer: Qwen3Renderer (auto-detected by Prime)`. Read that line when you try a new
model, because the usual failure is that Prime selects the renderer of a related model
family. `loops run` logs the same choice as
`Renderer for <model>: <class> (<source>); tokenizer <where it loaded from>; images ...`.

If you aren't sure that the auto-detected renderer matches the model's template, render
a few histories with your agent's real tool schemas. Then compare the token IDs with
`tokenizer.apply_chat_template(...)` for the same input.
`tests/contracts/test_qwen_renderer.py`, `tests/contracts/test_glm53_renderer.py` and
`tests/contracts/test_nemotron_extension.py` show how.

For a model that Prime doesn't recognise, auto-detection would fall back to Prime's
`DefaultRenderer`, which uses the tokenizer's generic chat template. The cookbook raises
an error instead. When you do want that template, select `DefaultRenderer` explicitly:

```toml
[model]
base_model = "org/model"
renderer = "loops_cookbook.llm.factory:default_renderer"
```

## A model whose tokenizer is published under another name

The renderer needs the policy's Hugging Face tokenizer. By default, it loads the
tokenizer from the repository named by `base_model`. Some models that Loops serves have
no Hugging Face repository under their Loops name. For example, an early-access model
may keep its weights and tokenizer in a private repository. Set `tokenizer` to where
the tokenizer is, and keep `base_model` as the name that Loops serves. `base_model` is
what provisions the trainer and the sampler, and it is what `check` asks Loops about:

```toml
[model]
base_model = "org/model"
tokenizer = "org/model-tokenizer"
renderer = "loops_cookbook.llm.factory:default_renderer"
```

`tokenizer` is either a Hugging Face repository ID or a local directory that holds the
tokenizer files. Those files are `tokenizer.json`, `tokenizer_config.json` and the chat
template. A value that starts with `./`, `../`, `/` or `~` is a local directory, and
anything else is a repository ID. So `tokenizers/nemotron-35` names a repository, and
`./tokenizers/nemotron-35` names a directory. When a directory in the working directory
has the same name as a repository ID, the cookbook refuses the value instead of loading
from either one.

A relative directory is relative to the config file that names it. If the config file
is a symlink, the path is relative to the file that the symlink points to. A directory
given with `--model.tokenizer` is relative to the shell's working directory instead.
Every other path in a config follows the same rules. `check` refuses a value that isn't
a directory.

Every load of the policy's tokenizer reads this setting. That includes the renderer for
RL and for SFT, in both `check` and `run`, and the SFT `loops check` preview of a
`tokens` dataset. A renderer that loads an image processor on demand loads it from the
same place, because Prime and the cookbook's renderers load the processor from the
tokenizer's own `name_or_path`. When the setting is used, `loops check` prints
`tokenizer from <tokenizer>` on its renderer line.

The Nemotron 3.5 Super VL processor runs code from the model's repository, and Prime
refuses to load it from a local directory. With a local tokenizer, that model still
renders text, but a message with an image fails.

A private or gated repository authenticates with `HF_TOKEN`, which Hugging Face's
libraries read from the environment. Put it in the `.env` file beside the config.
`.env.example` is a template for that file.

Prime's auto-detection and the cookbook's selection both go by the tokenizer's
`name_or_path`. With this setting, that is the tokenizer's repository or directory
instead of the model's published name, so neither will usually recognise the model. Set `renderer`
explicitly. Set it to a model-specific renderer, or to `default_renderer` when you want
the model's own chat template.

There is no setting for a revision. To pin a tokenizer from a repository that changes,
download the revision you want beside the config, and point `tokenizer` at the
directory:

```sh
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 --revision <commit> \
  --include "tokenizer*" "chat_template.jinja" "*config.json" --local-dir tokenizers/nemotron-35
```

```toml
[model]
tokenizer = "./tokenizers/nemotron-35"
```

An RL resume compares `tokenizer` and `renderer` with the run it continues, and warns
when either one changed. The run keeps training, but the steps before and after the
resume may not have rendered alike. An SFT resume refuses a change that alters what
trains, as [sft.md](sft.md#continue-or-fork-a-run) explains.

## Model-specific options

`renderer_chat_template_kwargs` is passed to the renderer unchanged. What it accepts
depends on the model. For example, Qwen accepts `{ enable_thinking = false }`, and
GLM-5.3 accepts `{ reasoning_effort = "high", clear_thinking = false }`. Nemotron 3
Ultra and Nemotron 3.5 Super VL accept `{ medium_effort = true }`. There is no
universal setting for keeping reasoning in the history. Each model's renderer decides
how it renders earlier `reasoning_content`.

## Adding a renderer for an unsupported model

Write a factory next to your training config, and point `[model] renderer` at it:

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

The factory receives two arguments. The first is the model's Hugging Face tokenizer,
which comes from `tokenizer` and defaults to `base_model`, as described above. The
second is the `renderer_chat_template_kwargs` dict. The factory returns any object that
satisfies the `Renderer` protocol, which the next section describes.

The usual route is to wrap a Prime renderer in `PrimeRenderer`. You then get rendering,
parsing, stop tokens and the bridge without writing them, and the cookbook's message
and tool validation on top. For a model that Prime doesn't cover, subclass the nearest
Prime renderer, or implement the protocol directly. The GLM-5.3 extension in
`loops_cookbook/llm/renderer_extensions/glm53.py` is a complete example of a subclass.

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

- `render` must be deterministic. It must produce exactly what the model saw in
  post-training for that history, including the generation prompt. A text-only renderer
  may return plain token IDs. A renderer that emits image placeholders returns a
  `RenderedPrompt(ids, prompt_input)`. Its `prompt_input` is the Loops `ModelInput`, with
  `ImageChunk`s at the placeholder positions, and `prompt_input.length` must equal
  `len(ids)`. `PrimeRenderer` does this for you.
- `parse` returns `ParsedReply(content, reasoning_content, tool_calls,
  reasoning_complete, native_message)`. `native_message` is the assistant turn in the
  format that `render` accepts, so the agent can append it to its history as it is.
  Return `None` for `native_message` exactly when some tool call's `status` isn't `OK`.
  No message would render back to what the model wrote, so the agent appends the
  reply's `history_message` instead. `history_message` is the turn as the model wrote
  it, and the conversation only ever continues it from the sampled tokens. The client
  treats either mismatch as a `RendererFailure`, which fails the attempt. One mismatch
  is `None` when every call is `OK`. The other is a message when some call isn't `OK`.
- The client sends `stop_sequences` with every sampling request. A caller's own stops
  of the same kind, token IDs or strings, are joined with them. When the caller's stops
  are of the other kind, the request carries the caller's stops as given, because one
  request carries only one kind. Either way, the sampler still ends the turn at the
  model's own end-of-sequence tokens.
- `bridge` is optional. When a renderer has one, the client asks it to extend
  `prompt_ids + completion_ids` with the new messages. Return `None` to decline. The
  result must start with exactly those tokens, and the client checks this and raises an
  error if it doesn't. `previous_prompt_input` is the previous prompt's chunk structure,
  a `ModelInput` that the client builds for this call from the captured sample. A
  renderer that handles images must decline when `previous_prompt_input` is missing,
  because it can't carry earlier images forward without it. `prompt_ids` is the
  captured prompt itself, a `TokenSpan` over the attempt's token buffer. A `TokenSpan`
  stores four bytes a token and reads as a tuple. Convert it with `list()` only where a
  library needs a list.
- `RenderedPrompt.ids` is always a `TokenSpan`, whatever the renderer returned. It
  compares, hashes and slices as the tuple of its tokens.

A run shares one renderer instance across all of its conversations, so a renderer
must not keep any per-conversation state.

A renderer whose `thread_safe` attribute is true may be called from several threads at
once. An RL run calls such a renderer's `render` and `bridge` off the event loop, on a
pool of a few threads called the `RenderPool`, so that one long render doesn't hold up
every other rollout. Its `parse` still runs on the event loop. The run calls any other
renderer on the event loop, one call at a time. `PrimeRenderer` is `thread_safe` for
Prime's text renderers, which keep nothing between calls. It isn't `thread_safe` for
Prime's multimodal renderers, which cache images and load their processor on first use.

Nothing that shares a thread-safe renderer's tokenizer may turn on truncation or
padding. Hugging Face sets those options on the shared backend for the length of a call,
so a concurrent encode would see them. This holds for the transformers versions that
`pyproject.toml` allows. Another version qualifies only when
`tests/unit/llm/test_tokenizer_sharing.py` passes on it.

When the cookbook builds a custom factory's result, it refuses one that lacks `render`,
`parse` or `stop_sequences`. A renderer that handles images must also have a
`supports_images` property that is true. Otherwise the cookbook reports it as text only
and probes it as text only.

## Images

Image parts work with renderers where one image renders to exactly the run of tokens
that the trainer expects. Those renderers are Qwen3.5, Qwen3.6, Qwen3.8 and Qwen3-VL,
and GLM-5.3-Flash through the cookbook's GLM-5.3 extension. The `contracts` CI job
checks the GLM path against the Hugging Face (HF) processor's own expansion. It checks
the Qwen paths against Prime's placeholder runs and the real image processors' token
counts. Both paths have also run end to end through Loops, covering sampling, grading,
the training datums and the audit.

Gemma 4 follows Prime's Qwen convention, so it should work, but it hasn't been run.
Kimi K2.5 renders a single placeholder per image and relies on the engine to expand it,
so the cookbook rejects Kimi image parts with a clear error.

### Install the vision extra

Install the vision extra, which brings torch and torchvision for the model's Hugging
Face image processor. Nothing runs on a GPU on the host, so CPU wheels are enough:

```sh
uv sync --extra vision
```

The vision extra works without further setup on every supported host. The supported
hosts are Apple Silicon macOS 14 or later, Linux x86_64 with glibc 2.28 or newer, and
Windows x64.

On Linux, PyPI's torch wheel bundles CUDA libraries of several gigabytes, even though
nothing here uses a GPU. That costs only download size. If you want the small CPU-only
build, add PyTorch's CPU index to your own project. This repository does that for its
own development, but a package can't pass index settings on to the projects that depend
on it.

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
torchvision = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
```

`loops check` prints `Images: supported (N tokens for a 32x32 probe)` when the renderer
handles images, and `Images: not supported by this renderer (text only)` otherwise. The
first render downloads the model's processor config.

### Put images in messages

Put images in the content list of a user or tool message, in any shape that Prime
accepts:

```python
{"type": "image", "image": png_bytes}                       # bytes, PNG or JPEG
{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
{"type": "image", "image": pil_image}                       # re-encoded as JPEG
{"type": "image", "path": "/tmp/screen.png"}                # or url: "https://..."
```

PNG and JPEG payloads, given as bytes or as base64 data URIs, are sent as they are.
Everything else, such as PIL objects, paths, URLs and other encodings, is loaded and
re-encoded as JPEG. Re-encoding keeps the image's dimensions, and so its token count.

Pass bytes or data URIs where you can. On every call, the client converts the image
parts of your full history to bytes. It decodes bytes and data URIs once per
conversation, however many turns resend them. A PIL object, path or URL can change
between calls, so the client reads it again on every turn. A URL whose content changes
would also defeat the bridge.

Images render in user and tool messages, directly in the content list. GLM-5.3 also
renders images in system and assistant messages, and inside bundled tool `output`
entries, which are described below. The Qwen renderers reject images in both of those
places. Text-only renderers reject image parts instead of dropping them. Video and audio
parts are rejected everywhere.

GLM-5.3 also accepts its template's bundled tool shape. That is one tool message whose
content is a list of `{"tool_call_id": ..., "output": ...}` entries, with any images
inside an `output` list. For Prime's Qwen renderers, the cookbook rejects bundled entries
that carry images, with a message that points to the flat form, in which images go
directly in the tool message's content list. Prime's Qwen renderers refuse text-only
bundled entries themselves.

For GLM-5.3-Flash, the extension writes each image inline in the message text, as
`<|begin_of_image|>`, then N copies of `<|image|>`, then `<|end_of_image|>`. This is
exactly how the HF processor expands the chat template, and N comes from the model's
image processor. A user string that literally contains `<|image|>` is tokenized as that
special token, which is also what the HF template does.

Run records never contain image bytes. The records are `samples/*.jsonl`,
`batches/step-N.json` and `transcripts/*.json`, and they hold only a SHA-256 digest and
the size of each image.

### How an image reaches the trainer

Prime renders each image to the model's placeholder tokens and reports where they are.
The cookbook turns that into the structure of text and image chunks that the Loops
sampler and trainer expect. The sampler expands the same placeholders and echoes the
prompt back, and the client refuses to train if that echo differs from what it
rendered. The Loops trainer recomputes each image's token count with the model's
processor, and it refuses inputs whose `expected_tokens` disagree. A bad count
therefore fails at the first update instead of training silently.

One image has to render to the same tokens that the sampler feeds the model. Three
independent checks hold it there:

- The sampler validates vLLM's expanded prompt against its own before it returns a
  completion.
- The cookbook compares the sampler's echoed `prompt_token_ids` with the tokens it
  submitted, on every generation. For a text prompt, this proves little. When the
  sampler echoes nothing, the Loops SDK fills the field with the submitted IDs, so the
  cookbook compares the IDs with themselves. The check matters for image prompts,
  where the SDK has no IDs to fill in. An image prompt that the sampler doesn't echo
  fails, and one that it does echo is compared token for token.
- Router replay, also called R3, lets the trainer of a mixture-of-experts (MoE) model
  replay the expert routes that the sampler used. It is the strongest of the three
  checks. The trainer checks every token it scores against the sampler's record of the
  generation that the token came from. It covers only text prompts, because Loops
  records only those, and only MoE models, because a dense model has no routes to
  replay. Router replay is off unless you set `[training] router_replay = true`. Leave
  it off for a dense model, or Loops records each generation for nothing.
