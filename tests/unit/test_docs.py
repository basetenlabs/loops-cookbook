"""The knob reference is generated, so the copy on disk has to match."""

import re
import xml.dom.minidom
from pathlib import Path

import pytest

from loops_cookbook.docs import GLOSSARY_PATH, render_knob_glossary, sections
from loops_cookbook.rl.config import Config as RlConfig
from loops_cookbook.rl.config import TrainingSettings
from loops_cookbook.sft.config import Config as SftConfig


def _blocks():
    rendered = render_knob_glossary()
    return dict(block.split("\n", 1) for block in rendered.split("\n## ")[1:])


def test_the_glossary_on_disk_matches_the_settings():
    """Run `python -m loops_cookbook.docs` when this fails."""
    assert GLOSSARY_PATH.read_text() == render_knob_glossary(), (
        "docs/reference/KNOB_GLOSSARY.md is stale; regenerate it with `python -m loops_cookbook.docs`"
    )


@pytest.mark.parametrize(
    ("config", "prefix"), [(RlConfig, ""), (SftConfig, "SFT ")], ids=["rl", "sft"]
)
def test_every_setting_is_documented_exactly_once(config, prefix):
    blocks = _blocks()
    for section, model in sections(config):
        body = blocks[f"{prefix}`[{section}]`"].split("\n# ")[0]
        for name, info in model.model_fields.items():
            if isinstance(info.annotation, type) and hasattr(info.annotation, "model_fields"):
                continue
            assert body.count(f"| `{name}` |") == 1, f"{section}.{name}"


def test_a_setting_without_a_description_cannot_be_shipped(monkeypatch):
    """The generator is the gate: a new knob is documented or the build fails."""
    schema = TrainingSettings.model_json_schema()
    schema["properties"] = {
        name: ({**prop, "description": None} if name == "group_size" else prop)
        for name, prop in schema["properties"].items()
    }
    monkeypatch.setattr(TrainingSettings, "model_json_schema", classmethod(lambda cls, **_: schema))
    with pytest.raises(ValueError, match="without a description"):
        render_knob_glossary()


def test_table_cells_are_never_broken_by_a_type_containing_a_pipe():
    """`mode` and `trace` are unions; an unescaped pipe would close the cell."""
    for line in render_knob_glossary().splitlines():
        if line.startswith("| `"):
            assert re.sub(r"\\\|", "", line).count("|") == 5, line


def test_every_table_the_training_config_validates_has_its_keys_listed():
    """Read from the schema, not from the generator's own walk: a table that is
    optional or keyed by name is still one a user writes."""
    from loops_cookbook.rl.config import Config

    listed = {
        heading: {line.split("`")[1] for line in body.splitlines() if line.startswith("| `")}
        for heading, body in _blocks().items()
    }
    for name, definition in Config.model_json_schema()["$defs"].items():
        # A table that is always present is its own section rather than a row.
        properties = definition.get("properties", {})
        keys = {key for key, prop in properties.items() if "$ref" not in prop}
        assert any(keys <= rows for rows in listed.values()), name


def test_the_built_in_runtime_and_provider_tables_are_listed_once_each():
    from loops_cookbook.docs import BUILT_IN_SECTIONS

    blocks = _blocks()
    for heading, model, _ in BUILT_IN_SECTIONS:
        body = blocks[heading]
        for name in model.model_fields:
            assert body.count(f"| `{name}` |") == 1, f"{heading} {name}"
    headings = {heading for heading, _, _ in BUILT_IN_SECTIONS}
    assert {"`[rollout_runtime.config]` for Episodes", "`[providers.blaxel]`"} <= headings


def _names_a_setting(table: str, key: str) -> bool:
    """Whether ``[table].key`` is a setting ``Config`` validates there. A table that is
    a mapping (``[providers.*]``, ``[rollout_runtime.config]``) holds keys the user or the
    runtime owns, so any key in it is taken as named."""
    from types import NoneType, UnionType
    from typing import Union, get_args, get_origin

    from pydantic import BaseModel

    from loops_cookbook.rl.config import Config

    model: type[BaseModel] = Config
    for part in [*table.split("."), key]:
        if part not in model.model_fields:
            return False
        annotation = model.model_fields[part].annotation
        if get_origin(annotation) in (Union, UnionType):
            [annotation] = [arg for arg in get_args(annotation) if arg is not NoneType]
        if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
            return True  # a leaf, or a mapping whose keys are not the config's
        model = annotation
    return True


def test_every_setting_a_message_or_doc_names_is_in_the_table_it_names():
    """``[compute].max_concurrent_rollouts`` in an error sends the user to a table that
    rejects the key; every ``[table].key`` the package or its docs name must exist."""
    from loops_cookbook.rl.config import Config

    root = Path(__file__).resolve().parents[2]
    tables = "|".join(Config.model_fields)
    reference = re.compile(rf"\[((?:{tables})(?:\.\w+)*)\]\.(\w+)")
    files = [*root.glob("src/**/*.py"), *root.glob("**/*.md")]
    named = {
        (str(path.relative_to(root)), *match.groups())
        for path in files
        # Hidden directories hold the environment and scratch, not what ships.
        if not any(part.startswith(".") for part in path.relative_to(root).parts)
        for match in reference.finditer(path.read_text())
    }
    assert len(named) > 5
    assert {entry for entry in named if not _names_a_setting(*entry[1:])} == set()


_ASSETS = sorted((Path(__file__).resolve().parents[2] / "docs" / "assets").glob("*.svg"))


@pytest.mark.parametrize("path", _ASSETS, ids=[path.name for path in _ASSETS])
def test_every_diagram_is_well_formed_svg(path):
    """A bare `<` in hand-edited label text renders as a broken image on GitHub."""
    xml.dom.minidom.parse(str(path))
