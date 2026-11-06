"""Renderer selection: an explicit ``module:callable``, the cookbook's for the models it
renders, otherwise Prime auto-detection."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping
from pathlib import Path

import renderers
from renderers.glm5 import GLM53Renderer

from loops_cookbook.evidence import JsonValue
from loops_cookbook.llm.prime import PrimeRenderer
from loops_cookbook.llm.rendering import Renderer, RenderingError
from loops_cookbook.settings import is_local_path

logger = logging.getLogger(__name__)

_EXTENSIONS = "loops_cookbook.llm.renderer_extensions"
_GLM53 = f"{_EXTENSIONS}.glm53:renderer"
_ULTRA = f"{_EXTENSIONS}.nemotron:ultra"
_SUPER_VL = f"{_EXTENSIONS}.nemotron:super_vl"
_MUSE_SPARK = f"{_EXTENSIONS}.muse_spark:renderer"

# The cookbook's renderer for these models, by tokenizer name as Prime matches them,
# selected without a [model] renderer setting. Each is tested against the model's own
# template (renderer_extensions/README.md).
COOKBOOK_RENDERERS: dict[str, str] = {
    "zai-org/GLM-5.3": _GLM53,
    "zai-org/GLM-5.3-BF16": _GLM53,
    "zai-org/GLM-5.3-Flash": _GLM53,
    "zai-org/GLM-5.3-Flash-BF16": _GLM53,
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": _ULTRA,
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8": _ULTRA,
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16": _SUPER_VL,
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4": _SUPER_VL,
    "meta-models/Muse-Spark-Open-816B-A42B-NVFP4": _MUSE_SPARK,
}

_BY_FOLDED_NAME = {name.casefold(): path for name, path in COOKBOOK_RENDERERS.items()}


def cookbook_renderer(tokenizer_name: str) -> str | None:
    """The cookbook's renderer for a tokenizer name, ignoring case as the Hub does."""
    return _BY_FOLDED_NAME.get(tokenizer_name.casefold())


# Prime renderers that do not match their model's template. Auto-detection that picks
# one for a model missing above is refused, naming the cookbook renderer to set.
_REFUSED_PRIME_RENDERERS: dict[type, str] = {GLM53Renderer: _GLM53}


def renderer_source(tokenizer_name: str, renderer: str | None) -> str:
    """Where the renderer for a tokenizer comes from, as ``check`` and the run log say."""
    if renderer:
        return renderer
    if (selected := cookbook_renderer(tokenizer_name)) is not None:
        return f"the cookbook's for {tokenizer_name}: {selected}"
    return "auto-detected by Prime"


def default_renderer(tokenizer, chat_template_kwargs: Mapping[str, JsonValue]) -> Renderer:
    return PrimeRenderer(
        renderers.DefaultRenderer(
            tokenizer,
            renderers.DefaultRendererConfig(**dict(chat_template_kwargs)),
        )
    )


def load_tokenizer(source: str) -> object:
    """The Hugging Face tokenizer at ``source``, a repo id or a local directory.

    A private or gated repo authenticates with ``HF_TOKEN`` from the environment. A
    repo id is refused when a directory of that name is beside the shell, since
    Hugging Face would load the directory, and which one loads would depend on where
    the command runs; a directory is named by a path (``is_local_path``).
    """
    if not is_local_path(source) and Path(source).is_dir():
        raise ValueError(
            f"tokenizer {source} names a Hugging Face repo, but a directory of that name "
            f"is here too; write ./{source} for the directory"
        )
    # Deferred: transformers takes seconds to import, and not every caller needs it.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(source)


def create_renderer(
    model_name: str,
    *,
    tokenizer=None,
    tokenizer_source: str | None = None,
    load: Callable[[str], object] = load_tokenizer,
    chat_template_kwargs: Mapping[str, JsonValue] | None = None,
    renderer: str | None = None,
) -> Renderer:
    """Build the renderer for ``model_name``.

    The tokenizer is ``tokenizer`` when given, otherwise ``load(tokenizer_source)``,
    and ``tokenizer_source`` defaults to ``model_name`` (the ``[model] tokenizer``
    setting, for a model whose tokenizer is published under another name).

    ``renderer`` is a ``module:callable`` import path (the ``[model] renderer``
    setting) to a factory ``(tokenizer, chat_template_kwargs) -> Renderer``. Without
    it, a model in ``COOKBOOK_RENDERERS`` gets the cookbook's renderer, and Prime
    auto-detects any other from the tokenizer. Either way the selected
    renderer and where its tokenizer came from are logged; check that line when
    trying a new model.
    """
    if tokenizer is None:
        origin = tokenizer_source or model_name
        tokenizer = load(origin)
    else:
        origin = "supplied by the caller"
    options = dict(chat_template_kwargs or {})
    tokenizer_name = getattr(tokenizer, "name_or_path", None) or ""
    selected = renderer or cookbook_renderer(tokenizer_name)
    if selected is None:
        inner = renderers.create_renderer(
            tokenizer, renderers.AutoRendererConfig(), chat_template_kwargs=options
        )
        if isinstance(inner, renderers.DefaultRenderer):
            raise ValueError(
                "Prime-RL's renderer auto-detection fell back to DefaultRenderer. "
                f"Configure {model_name} renderer explicitly if the model requires a custom "
                "format, or explicitly select DefaultRenderer if the generic format is intended. "
                "See docs/reference/renderers.md"
            )
        replacement = next(
            (path for cls, path in _REFUSED_PRIME_RENDERERS.items() if isinstance(inner, cls)), None
        )
        if replacement is not None:
            raise ValueError(
                f"Prime-RL's renderer auto-detection picked {type(inner).__name__} for "
                f"{model_name}, which does not match the model's reference template. "
                f'Set [model] renderer = "{replacement}". See docs/reference/renderers.md'
            )
        built = PrimeRenderer(inner)
    else:
        built = _import_factory(selected)(tokenizer, options)
        for required in ("render", "parse", "stop_sequences"):
            if not hasattr(built, required):
                raise RenderingError(f"{selected} returned an object without {required!r}")
    logger.info(
        "Renderer for %s: %s (%s); tokenizer %s; images %s",
        model_name,
        describe(built),
        renderer_source(tokenizer_name, renderer),
        origin,
        "supported" if getattr(built, "supports_images", False) else "not supported",
    )
    return built


def describe(renderer: object) -> str:
    """Human-readable renderer identity, unwrapping the Prime adapter."""
    inner = getattr(renderer, "_renderer", None)
    return type(inner).__name__ if inner is not None else type(renderer).__name__


def _import_factory(path: str) -> Callable[[object, Mapping[str, JsonValue]], Renderer]:
    """Import the ``module:callable`` named by ``[model] renderer``."""
    module_name, _, attribute = path.partition(":")
    if not module_name or not attribute:
        raise RenderingError(f"renderer must be 'module:callable', got {path!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise RenderingError(
            f"could not import renderer module {module_name!r}: {error}"
        ) from error
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise RenderingError(f"{path!r} is not a callable renderer factory")
    return factory
