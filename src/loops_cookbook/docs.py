"""Render the knob reference from the settings themselves.

One declaration feeds three readers: ``--help`` (through ``cli``), the hint for a
misspelt command-line flag, and this table. A hand-written copy of the same list is
how the one in the Harbor runtime's guide came to be missing two whole sections, so this
one is generated and a test fails when the file on disk disagrees.

Only the reference belongs here. What a setting *means together with another one*
is prose, lives in a guide, and would be a framework if it were metadata.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel

from loops_cookbook.episodes.runtime import EpisodeSettings, SandboxSettings
from loops_cookbook.providers.blaxel import BlaxelSandbox
from loops_cookbook.providers.blaxel.settings import BlaxelSettings
from loops_cookbook.providers.daytona import DaytonaSandbox
from loops_cookbook.providers.daytona.settings import DaytonaSettings
from loops_cookbook.rl.config import Config
from loops_cookbook.sft.config import Config as SftConfig

GENERATED_BEGIN = (
    "<!-- generated from the settings models; `python -m loops_cookbook.docs` rewrites it -->"
)
GENERATED_END = "<!-- /generated -->"

SECTION_BLURBS = {
    "rollout_runtime": "Where episodes come from: the rollout runtime factory and what it may resolve.",
    "compute": "Where the run executes. Nothing here changes what is learned.",
    "model": "The policy being trained and the compute it is configured for.",
    "sampling": "Defaults for what the policy produces. Agents inherit these and state "
    "only what they need to differ on.",
    "training": "The optimizer, the batch, and what data is acceptable to train on.",
    "rollout": "How much runs at once, how long each episode may take, and when to give up.",
    "rollout.grading": "Each grade() try's deadline; how many tries a failure earns is the "
    "rollout runtime's.",
    "rollout.cleanup": "Deadlines for the rollout runtime's cleanup hooks.",
    "output": "What the run leaves behind.",
    "rollout_runtime.openai": "The Chat Completions gateway, with model_access = openai_http.",
    "rollout.producers.<name>": "One producer a rollout runtime may name, keyed by that name.",
    "eval": "Held-out evals during training, each on a sampler pinned to one step's "
    "weights. Nothing they sample trains.",
    "eval.sets.<name>": "One held-out task set: training's rollout runtime with these files. A key "
    "at the path of a training setting overrides that setting for this set.",
    "eval.sets.<name>.rollout_runtime": "Overrides [rollout_runtime] for this set.",
    "eval.sets.<name>.sampling": "Overrides [sampling] for this set.",
}

# The built-in runtimes' and providers' own tables, as (heading, model, blurb): opaque
# to the training config, read by the factory that owns them, and declared as settings
# models all the same. Harbor's `reward_key` is checked by hand, so it is prose above.
BUILT_IN_SECTIONS: tuple[tuple[str, type[BaseModel], str], ...] = (
    (
        "`[rollout_runtime.config]` for Episodes",
        EpisodeSettings,
        (
            'With factory = "loops_cookbook.episodes:create_runtime". Its tasks file is the '
            "`tasks` alias under `[rollout_runtime.files]`."
        ),
    ),
    (
        "`[rollout_runtime.config.sandbox]` for Episodes",
        SandboxSettings,
        (
            "An Episode's sandbox. The keys beside `provider` are the provider's sandbox "
            "spec, below for Blaxel and Daytona."
        ),
    ),
    (
        "`[rollout_runtime.config.sandbox]` on Blaxel",
        BlaxelSandbox,
        'Beside provider = "blaxel".',
    ),
    (
        "`[rollout_runtime.config.sandbox]` on Daytona",
        DaytonaSandbox,
        'Beside provider = "daytona". Its lifecycle is `[providers.daytona]`\'s.',
    ),
    (
        "`[rollout_runtime.config.grading_sandbox]` for Episodes",
        SandboxSettings,
        (
            "An Episode's optional separate verifier sandbox, provisioned only on grading "
            "admission. The keys beside `provider` are the provider's sandbox spec."
        ),
    ),
    (
        "`[rollout_runtime.config.grading_sandbox]` on Blaxel",
        BlaxelSandbox,
        'Beside provider = "blaxel"; independent of the execution sandbox spec.',
    ),
    (
        "`[providers.blaxel]`",
        BlaxelSettings,
        (
            "The built-in Blaxel provider. Its credentials, BL_WORKSPACE and BL_API_KEY, "
            "come from the environment or a .env beside the config, never from this table."
        ),
    ),
    (
        "`[providers.daytona]`",
        DaytonaSettings,
        (
            "The built-in Daytona provider; see [daytona.md](daytona.md). Its credential, "
            "DAYTONA_API_KEY (and DAYTONA_API_URL for a self-hosted Daytona), comes from the "
            "environment or a .env beside the config, never from this table."
        ),
    ),
)


NAMES = {
    "string": "string",
    "boolean": "boolean",
    "integer": "integer",
    "number": "number",
    "object": "table",
    "array": "list",
}


def _describe(prop: dict[str, Any]) -> str:
    """Type and bounds, from pydantic's own schema rather than the annotation.

    The schema has already resolved a ``Literal`` to ``enum``, an optional to
    ``anyOf``, and every ``Field`` constraint to a bound -- which the annotation
    alone does not carry. A pipe would close the table cell, so an enum joins with
    an escaped one.
    """
    if enum := prop.get("enum"):
        return " \\| ".join(f"`{value}`" for value in enum)
    variants = prop.get("anyOf") or [prop]
    body = next((v for v in variants if v.get("type") != "null"), variants[0])
    # A nested settings model is a reference into the schema's definitions.
    kind = "object" if "$ref" in body else body.get("type", "")
    name = NAMES.get(kind, kind or "value")
    bounds = [
        text
        for key, text in (
            ("minimum", f"\u2265 {body.get('minimum')}"),
            ("exclusiveMinimum", f"> {body.get('exclusiveMinimum')}"),
            ("maximum", f"\u2264 {body.get('maximum')}"),
        )
        if body.get(key) is not None
    ]
    return f"{name}, {', '.join(bounds)}" if bounds else name


def _default(prop: dict[str, Any], required: bool) -> str:
    if required:
        return "*required*"
    if "default" not in prop:
        return "empty"
    value = prop.get("default")
    if value is None:
        return "`None`"
    if isinstance(value, str):
        return f'`"{value}"`'
    return f"`{value}`"


def _model(annotation: Any) -> type[BaseModel] | None:
    return (
        annotation if isinstance(annotation, type) and issubclass(annotation, BaseModel) else None
    )


def _table(annotation: Any) -> tuple[str, type[BaseModel]] | None:
    """The settings model a field's TOML table is validated with, and the key it sits
    under: the field itself, or ``.<name>`` for a table of named tables.

    A table that may be absent is still a table, and a table of named tables is one
    table per name with the same keys.
    """
    if model := _model(annotation):
        return "", model
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, UnionType):
        present = [arg for arg in args if arg is not type(None)]
        if len(present) == 1 and (model := _model(present[0])):
            return "", model
    if origin is dict and len(args) == 2 and (model := _model(args[1])):
        return ".<name>", model
    return None


def sections(
    model: type[BaseModel] = Config, prefix: str = ""
) -> Iterator[tuple[str, type[BaseModel]]]:
    """Every TOML table the settings declare, as ``(dotted name, model)``."""
    for name, info in model.model_fields.items():
        if table := _table(info.annotation):
            suffix, nested = table
            path = f"{prefix}{name}{suffix}"
            yield path, nested
            yield from sections(nested, f"{path}.")


def _leaves(model: type[BaseModel]) -> list[str]:
    """The keys a table's own rows list: a nested table always given in full has its
    own section instead; one that may be absent, or holds named tables, keeps its row
    too, for its default."""
    return [name for name, info in model.model_fields.items() if not _model(info.annotation)]


def _block(heading: str, model: type[BaseModel], blurb: str | None) -> str:
    schema = model.model_json_schema()
    properties, required = schema["properties"], set(schema.get("required", ()))
    names = _leaves(model)
    undocumented = sorted(n for n in names if not properties[n].get("description"))
    if undocumented:
        raise ValueError(
            f"settings without a description cannot be documented, under {heading}: "
            + ", ".join(undocumented)
        )
    lines = [f"## {heading}", ""]
    if blurb:
        lines += [blurb, ""]
    lines += ["| Setting | Type | Default | Description |", "| --- | --- | --- | --- |"]
    for name in names:
        prop = properties[name]
        lines.append(
            f"| `{name}` | {_describe(prop)} | {_default(prop, name in required)} "
            f"| {prop['description']}{' Dedicated only.' if prop.get('dedicated_only') else ''} |"
        )
    return "\n".join(lines)


def render_knob_tables() -> str:
    """The reference tables, one per TOML section, in declaration order."""
    return "\n\n".join(
        _block(f"`[{section}]`", model, SECTION_BLURBS.get(section))
        for section, model in sections()
    )


def render_built_in_tables() -> str:
    """The built-in runtimes' and providers' own tables."""
    return "\n\n".join(_block(*entry) for entry in BUILT_IN_SECTIONS)


def render_sft_tables() -> str:
    """The SFT recipe's tables, under headings of their own."""
    return "\n\n".join(
        _block(f"SFT `[{section}]`", model, None) for section, model in sections(SftConfig)
    )


def render_knob_glossary() -> str:
    """The whole file, generated markers included."""
    return f"""\
# Knob glossary

Every setting the cookbook reads. Settings live in `train.toml` under the section
shown, and the CLI exposes each key of a table that is always present by its dotted
path, such as `--rollout.max_concurrent_rollouts`.

A rollout runtime's own settings are opaque to the training config and live in
`[rollout_runtime.config]`; the built-in rollout runtimes' and providers' tables are listed after
the training config's. For Harbor, `[rollout_runtime.config]` is only `reward_key`, and
agent behaviour and environment configuration belong to the Harbor job. A custom
provider defines its own `[providers.<name>]`. For the settings that only make sense
together, see [`settings-that-interact.md`](settings-that-interact.md).

**Defaults or required.** A setting that changes what trains or what a run costs is
required, such as `learning_rate`, `group_size` or a sandbox's size. A setting has a
default only when it is purely operational: a wrong value fails loudly and nothing
about training depends on it, such as the `[rollout.cleanup]` graces. A setting the
platform has its own default for, such as `adam_beta2`, is optional with no value of
ours: unset, the request carries the Loops SDK's default, which a `baseten-loops`
release may change. An opt-in feature, such as `router_replay`, is off unless set. A
new setting follows the same rule.

{GENERATED_BEGIN}

{render_knob_tables()}

{render_built_in_tables()}

# SFT settings

An SFT config, `recipe = "sft"`, has its own TOML; `[model]`, `[compute]` and
`[output]` are the RL recipe's, spelled the same, except that SFT requires
`max_seq_len`.

{render_sft_tables()}

{GENERATED_END}
"""


GLOSSARY_PATH = Path(__file__).resolve().parents[2] / "docs" / "reference" / "KNOB_GLOSSARY.md"


if __name__ == "__main__":
    GLOSSARY_PATH.write_text(render_knob_glossary())
    print(f"wrote {GLOSSARY_PATH}")
