"""Real tokenizers and image processors for the supported models.

Every fixture here skips unless the model files are available:
``LOOPS_QWEN_TOKENIZER_CACHE`` / ``LOOPS_GLM_TOKENIZER_CACHE`` /
``LOOPS_NEMOTRON_TOKENIZER_CACHE`` point at a Hugging Face hub cache for offline runs, or ``LOOPS_DOWNLOAD_CONTRACT_MODELS=1`` fetches them (CI).
"""

import os
from pathlib import Path

import pytest

REVISION = "aca966e4e02791568aa6a4ced368624b3d897f42"  # GLM-5.3 tokenizer pin


def pinned_snapshot(cache: str, repo: str, revision: str) -> Path:
    """``repo`` at ``revision`` in the Hugging Face hub cache ``cache``; skips if absent."""
    snapshot = Path(cache) / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    if not snapshot.is_dir():
        pytest.skip(f"missing pinned {repo} snapshot")
    return snapshot


def pinned_tokenizer(repo: str, revision: str, variable: str):
    """``repo`` at ``revision``: from the hub cache ``variable`` names, or downloaded when
    ``LOOPS_DOWNLOAD_CONTRACT_MODELS`` is set; otherwise the test skips."""
    from transformers import AutoTokenizer

    cache = os.environ.get(variable)
    if cache:
        snapshot = pinned_snapshot(cache, repo, revision)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    elif os.environ.get("LOOPS_DOWNLOAD_CONTRACT_MODELS"):
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    else:
        pytest.skip(f"set {variable} or LOOPS_DOWNLOAD_CONTRACT_MODELS=1")
    tokenizer.name_or_path = repo  # lets an extension find the processor by name
    return tokenizer


GLM_MODELS = {
    "GLM-5.3": REVISION,
    "GLM-5.3-Flash": "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
}


def _glm_tokenizer(model):
    return pinned_tokenizer(f"zai-org/{model}", GLM_MODELS[model], "LOOPS_GLM_TOKENIZER_CACHE")


@pytest.fixture(scope="module", params=sorted(GLM_MODELS))
def glm_tokenizer(request):
    """Both GLM-5.3 tokenizers: the text renderer must agree with each template."""
    return _glm_tokenizer(request.param)


@pytest.fixture(scope="module")
def glm53_tokenizer():
    """GLM-5.3 only, for contracts that compare several model families."""
    return _glm_tokenizer("GLM-5.3")


@pytest.fixture(scope="module")
def flash_tokenizer():
    """GLM-5.3-Flash only: the one GLM with a vision tower."""
    return _glm_tokenizer("GLM-5.3-Flash")


@pytest.fixture(scope="module")
def flash_processor(flash_tokenizer):
    """The real Glm5Next processor (needs the ``vision`` extra for torch)."""
    pytest.importorskip("torch", reason="image processors need the vision extra")
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(
            flash_tokenizer.name_or_path, revision=GLM_MODELS["GLM-5.3-Flash"]
        )
    except OSError:
        pytest.skip("GLM-5.3-Flash processor not available offline")


QWEN_TEXT_MODEL = "Qwen/Qwen3.6-35B-A3B"
QWEN_TEXT_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"


@pytest.fixture(scope="module")
def qwen_tokenizer():
    """Qwen3.6 tokenizer for the text-rendering contracts (cache or download)."""
    return pinned_tokenizer(QWEN_TEXT_MODEL, QWEN_TEXT_REVISION, "LOOPS_QWEN_TOKENIZER_CACHE")


VLM_CONTRACT_MODELS = {
    "Qwen/Qwen3.6-35B-A3B": "995ad96eacd98c81ed38be0c5b274b04031597b0",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "Qwen/Qwen3-VL-4B-Instruct": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
}


@pytest.fixture(params=sorted(VLM_CONTRACT_MODELS))
def cached_vlm_tokenizer(request):
    """Real tokenizer for a supported VLM: from the local cache, or downloaded in CI.

    Set ``LOOPS_QWEN_TOKENIZER_CACHE`` to a Hugging Face hub cache to run offline (with
    ``HF_HUB_OFFLINE=1`` so Prime's processor lookup by name stays local), or
    ``LOOPS_DOWNLOAD_CONTRACT_MODELS=1`` to fetch the pinned tokenizer and processor configs.
    """
    pytest.importorskip("torch", reason="image processors need the vision extra")
    return pinned_tokenizer(
        request.param, VLM_CONTRACT_MODELS[request.param], "LOOPS_QWEN_TOKENIZER_CACHE"
    )


@pytest.fixture
def flash_renderer(flash_tokenizer, flash_processor):
    """``flash_renderer(**chat_template_kwargs)``: the cookbook's GLM-5.3-Flash renderer
    with the real image processor attached."""
    from loops_cookbook.llm.factory import create_renderer

    def build(**chat_template_kwargs):
        renderer = create_renderer(
            "zai-org/GLM-5.3-Flash",
            renderer="loops_cookbook.llm.renderer_extensions.glm53:renderer",
            tokenizer=flash_tokenizer,
            chat_template_kwargs=chat_template_kwargs,
        )
        renderer._renderer._processor = flash_processor
        return renderer

    return build


NEMOTRON_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
NEMOTRON_REVISION = "2dc98e2afe4face0e4ce40972a915c45368bd34a"


@pytest.fixture(scope="module")
def nemotron_tokenizer():
    """Nemotron 3 Super tokenizer (cache or download); ``LOOPS_NEMOTRON_TOKENIZER_CACHE``
    points at a Hugging Face hub cache for offline runs."""
    return pinned_tokenizer(NEMOTRON_MODEL, NEMOTRON_REVISION, "LOOPS_NEMOTRON_TOKENIZER_CACHE")


NEMOTRON_ULTRA = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
NEMOTRON_ULTRA_REVISION = "77df655d5e9f8362164ed14dd8b48f8bce657498"
NEMOTRON_SUPER_VL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
# Super VL checkpoints: one template and tokenizer, each its own image processor config.
NEMOTRON_SUPER_VL_REVISIONS = {
    NEMOTRON_SUPER_VL: "fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe",
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4": "da5b2650a835292c678994a1fa3fb653250fb508",
}


@pytest.fixture(scope="module")
def super_vl_tokenizer():
    return pinned_tokenizer(
        NEMOTRON_SUPER_VL,
        NEMOTRON_SUPER_VL_REVISIONS[NEMOTRON_SUPER_VL],
        "LOOPS_NEMOTRON_TOKENIZER_CACHE",
    )


@pytest.fixture(scope="module", params=sorted(NEMOTRON_SUPER_VL_REVISIONS))
def super_vl_processor(request, super_vl_tokenizer):
    """Each Super VL checkpoint's own processor (remote code, at its pinned revision)."""
    pytest.importorskip("torch", reason="image processors need the vision extra")
    from transformers import AutoProcessor

    revision = NEMOTRON_SUPER_VL_REVISIONS[request.param]
    cache = os.environ.get("LOOPS_NEMOTRON_TOKENIZER_CACHE")
    try:
        if cache:
            snapshot = pinned_snapshot(cache, request.param, revision)
            return AutoProcessor.from_pretrained(
                snapshot, local_files_only=True, trust_remote_code=True
            )
        return AutoProcessor.from_pretrained(
            request.param, revision=revision, trust_remote_code=True
        )
    except OSError:
        pytest.skip(f"{request.param} processor not available offline")


# model -> (pinned revision, cache variable, template options, reply prefix: what closes
# the reasoning its generation prompt opens)
TOOL_CALL_MODELS = {
    "Qwen/Qwen3.5-9B": (
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "LOOPS_QWEN_TOKENIZER_CACHE",
        {"enable_thinking": False},
        "",
    ),
    NEMOTRON_ULTRA: (
        NEMOTRON_ULTRA_REVISION,
        "LOOPS_NEMOTRON_TOKENIZER_CACHE",
        {},
        "</think>\n",
    ),
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": (
        "2dc98e2afe4face0e4ce40972a915c45368bd34a",
        "LOOPS_NEMOTRON_TOKENIZER_CACHE",
        {},
        "</think>\n",
    ),
}


@pytest.fixture(scope="module", params=sorted(TOOL_CALL_MODELS))
def tool_call_model(request):
    """Qwen and Nemotron, which write tool calls as XML: tokenizer, renderer, reply prefix."""
    from loops_cookbook.llm.factory import create_renderer

    name = request.param
    revision, variable, options, prefix = TOOL_CALL_MODELS[name]
    tokenizer = pinned_tokenizer(name, revision, variable)
    renderer = create_renderer(name, tokenizer=tokenizer, chat_template_kwargs=options)
    return tokenizer, renderer, prefix
