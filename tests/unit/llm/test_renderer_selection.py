"""Renderer selection: an explicit module:callable, otherwise Prime auto-detection."""

import logging

import pytest
import renderers

from loops_cookbook.llm.factory import COOKBOOK_RENDERERS, create_renderer, describe
from loops_cookbook.llm.rendering import RenderingError


class FakeRenderer:
    def __init__(self, tokenizer, options):
        self.tokenizer, self.options = tokenizer, options

    stop_sequences = ()

    def render(self, messages, tools=()):
        return (1,)

    def parse(self, prompt_ids, completion_ids, tools=()):
        raise NotImplementedError


def make_fake(tokenizer, chat_template_kwargs):
    return FakeRenderer(tokenizer, chat_template_kwargs)


def test_import_path_factory_receives_tokenizer_and_options(caplog):
    with caplog.at_level(logging.INFO, logger="loops_cookbook.llm.factory"):
        built = create_renderer(
            "org/any-model",
            tokenizer="tok",
            chat_template_kwargs={"effort": "low"},
            renderer=f"{__name__}:make_fake",
        )
    assert isinstance(built, FakeRenderer)
    assert built.tokenizer == "tok" and built.options == {"effort": "low"}
    assert "FakeRenderer" in caplog.text and "make_fake" in caplog.text


def test_the_tokenizer_loads_from_its_own_source_when_one_is_set(caplog):
    loaded = []

    def load(source):
        loaded.append(source)
        return "tok"

    with caplog.at_level(logging.INFO, logger="loops_cookbook.llm.factory"):
        built = create_renderer(
            "org/served-name",
            tokenizer_source="org/tokenizer-repo",
            load=load,
            renderer=f"{__name__}:make_fake",
        )
    assert loaded == ["org/tokenizer-repo"]
    assert built.tokenizer == "tok"
    assert "Renderer for org/served-name" in caplog.text
    assert "tokenizer org/tokenizer-repo" in caplog.text


def test_the_tokenizer_loads_from_the_model_name_by_default():
    loaded = []

    def load(source):
        loaded.append(source)
        return "tok"

    create_renderer("org/served-name", load=load, renderer=f"{__name__}:make_fake")
    assert loaded == ["org/served-name"]


def test_a_supplied_tokenizer_is_not_loaded(caplog):
    def load(source):
        raise AssertionError("a supplied tokenizer is used as given")

    with caplog.at_level(logging.INFO, logger="loops_cookbook.llm.factory"):
        built = create_renderer("m", tokenizer="tok", load=load, renderer=f"{__name__}:make_fake")
    assert built.tokenizer == "tok"
    assert "tokenizer supplied by the caller" in caplog.text


def test_a_tokenizer_loads_from_a_local_directory(tmp_path):
    from support.local_tokenizer import save_tokenizer

    built = create_renderer(
        "org/no-such-repo",
        tokenizer_source=str(save_tokenizer(tmp_path)),
        renderer=f"{__name__}:make_fake",
    )
    assert built.tokenizer.decode([1]) == "hi"


def test_a_repo_id_never_loads_a_directory_of_the_same_name(tmp_path, monkeypatch):
    from support.local_tokenizer import save_tokenizer

    from loops_cookbook.llm.factory import load_tokenizer

    save_tokenizer(tmp_path / "tokenizers" / "model")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=r"write \./tokenizers/model"):
        load_tokenizer("tokenizers/model")


@pytest.mark.parametrize(
    "path,match",
    [
        ("nocolon", "module:callable"),
        ("missing_module_xyz:f", "could not import"),
        (f"{__name__}:FakeRenderer.nope", "not a callable"),
    ],
)
def test_bad_import_paths_are_rendering_errors(path, match):
    with pytest.raises(RenderingError, match=match):
        create_renderer("m", tokenizer="tok", renderer=path)


def not_a_renderer(tokenizer, options):
    return object()


def test_factory_result_must_look_like_a_renderer():
    with pytest.raises(RenderingError, match="without"):
        create_renderer("m", tokenizer="tok", renderer=f"{__name__}:not_a_renderer")


def test_describe_unwraps_the_prime_adapter():
    from loops_cookbook.llm.prime import PrimeRenderer

    class Inner:
        pass

    assert describe(PrimeRenderer(Inner())) == "Inner"
    assert describe(FakeRenderer("t", {})) == "FakeRenderer"


def test_renderer_setting_is_plain_config():
    from support.builders import make_config

    assert make_config(renderer="pkg:factory").model.renderer == "pkg:factory"
    assert make_config().model.renderer is None


class UnknownTokenizer:
    name_or_path = "org/unknown-model"
    eos_token_id = None


def test_auto_detected_default_renderer_raises():
    with pytest.raises(
        ValueError,
        match="renderer auto-detection fell back to DefaultRenderer",
    ):
        create_renderer("org/unknown-model", tokenizer=UnknownTokenizer())


@pytest.mark.parametrize("prime_renderer", ["GLM53Renderer", "GLM53FlashRenderer"])
def test_prime_glm53_picked_for_a_model_the_cookbook_does_not_list_raises(
    monkeypatch, prime_renderer
):
    from renderers import glm5

    picked = object.__new__(getattr(glm5, prime_renderer))
    monkeypatch.setattr(renderers, "create_renderer", lambda *a, **k: picked)
    with pytest.raises(ValueError, match="renderer_extensions.glm53:renderer"):
        create_renderer("org/glm-5.3-finetune", tokenizer=UnknownTokenizer())


def test_the_renderer_setting_wins_over_the_cookbook_renderer_for_a_listed_model():
    class GLM53Tokenizer(UnknownTokenizer):
        name_or_path = "zai-org/GLM-5.3"

    renderer = create_renderer(
        "zai-org/GLM-5.3",
        tokenizer=GLM53Tokenizer(),
        renderer="loops_cookbook.llm.factory:default_renderer",
    )
    assert describe(renderer) == renderers.DefaultRenderer.__name__


def test_default_renderer_can_be_selected_explicitly(caplog):
    with caplog.at_level(logging.INFO, logger="loops_cookbook.llm.factory"):
        renderer = create_renderer(
            "org/unknown-model",
            tokenizer=UnknownTokenizer(),
            renderer="loops_cookbook.llm.factory:default_renderer",
        )

    assert describe(renderer) == renderers.DefaultRenderer.__name__
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


@pytest.mark.parametrize(
    "tokenizer_name,renderer,source",
    [
        ("org/unknown-model", None, "auto-detected by Prime"),
        (
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
            None,
            (
                "the cookbook's for nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16: "
                "loops_cookbook.llm.renderer_extensions.nemotron:super_vl"
            ),
        ),
        ("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16", "pkg:factory", "pkg:factory"),
        (
            "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
            None,
            (
                "the cookbook's for meta-models/Muse-Spark-Open-816B-A42B-NVFP4: "
                "loops_cookbook.llm.renderer_extensions.muse_spark:renderer"
            ),
        ),
    ],
    ids=["prime", "cookbook", "setting", "cookbook muse"],
)
def test_check_names_where_the_renderer_comes_from(tokenizer_name, renderer, source):
    from loops_cookbook.llm.factory import renderer_source

    assert renderer_source(tokenizer_name, renderer) == source


@pytest.mark.parametrize("model", sorted(COOKBOOK_RENDERERS))
def test_each_listed_model_names_a_renderer_factory_that_imports(model):
    from loops_cookbook.llm.factory import _import_factory

    assert callable(_import_factory(COOKBOOK_RENDERERS[model]))


def test_a_listed_model_is_matched_whatever_its_case():
    from loops_cookbook.llm.factory import cookbook_renderer

    assert cookbook_renderer("ZAI-ORG/glm-5.3-flash") == COOKBOOK_RENDERERS["zai-org/GLM-5.3-Flash"]
    assert cookbook_renderer("zai-org/GLM-5.3-Flash-FP8") is None
