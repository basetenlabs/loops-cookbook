"""Render the knob reference from the settings models.

One declaration feeds three readers: ``--help`` through ``cli``, the hint for a
misspelt command-line flag, and this table. A hand-written copy of the list drifts
from the settings, so this one is generated, and a test fails when the file on disk
disagrees.

Only the reference belongs here. What a setting means together with another setting
is prose, and it belongs in a guide. As metadata it would turn into a framework.
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
    "rollout_runtime": "These settings say where episodes come from. They name the rollout "
    "runtime factory and what it may resolve.",
    "compute": "These settings say where the run executes. Nothing here changes what is learned.",
    "model": "These settings name the policy being trained and the compute it is configured for.",
    "sampling": "These settings are the defaults for what the policy produces. Agents "
    "inherit them and state only what they need to change.",
    "training": "These settings configure the optimizer, the batch, and what data is "
    "acceptable to train on.",
    "rollout": "These settings say how much runs at once, how long each episode may take, "
    "and when to give up.",
    "rollout.grading": "This setting is the deadline for each try of grade(). The rollout "
    "runtime decides how many tries a failure earns.",
    "rollout.cleanup": "These settings are the deadlines for the rollout runtime's cleanup hooks.",
    "output": "These settings say what the run leaves behind.",
    "rollout_runtime.openai": "These settings configure the Chat Completions gateway, which "
    "the run serves when model_access = openai_http.",
    "rollout.producers.<name>": "Each table configures one producer that a rollout runtime "
    "may name, keyed by that name.",
    "eval": "These settings run held-out evals during training. Each eval runs on a "
    "sampler pinned to one step's weights. Nothing an eval samples trains.",
    "eval.sets.<name>": "Each table is one held-out task set, which runs training's "
    "rollout runtime with these files. A key at the path of a training setting overrides "
    "that setting for this set.",
    "eval.sets.<name>.rollout_runtime": "This table overrides [rollout_runtime] for this set.",
    "eval.sets.<name>.sampling": "This table overrides [sampling] for this set.",
}

# The built-in rollout runtimes' and providers' own tables, as (heading, model, blurb).
# The training config treats them as opaque, and the factory that owns each one reads
# it, but each is still declared as a settings model. Harbor's `reward_key` is checked
# by hand, so the glossary's introduction describes it in prose.
BUILT_IN_SECTIONS: tuple[tuple[str, type[BaseModel], str], ...] = (
    (
        "`[rollout_runtime.config]` for Episodes",
        EpisodeSettings,
        (
            'These settings apply with factory = "loops_cookbook.episodes:create_runtime". '
            "The tasks file is the `tasks` alias under `[rollout_runtime.files]`."
        ),
    ),
    (
        "`[rollout_runtime.config.sandbox]` for Episodes",
        SandboxSettings,
        (
            "This table configures an Episode's sandbox. The keys beside `provider` are "
            "the provider's sandbox spec, which the next tables list for Blaxel and Daytona."
        ),
    ),
    (
        "`[rollout_runtime.config.sandbox]` on Blaxel",
        BlaxelSandbox,
        'These keys sit beside provider = "blaxel".',
    ),
    (
        "`[rollout_runtime.config.sandbox]` on Daytona",
        DaytonaSandbox,
        (
            'These keys sit beside provider = "daytona". `[providers.daytona]` sets the '
            "sandbox's lifecycle."
        ),
    ),
    (
        "`[rollout_runtime.config.grading_sandbox]` for Episodes",
        SandboxSettings,
        (
            "This table configures an Episode's optional separate verifier sandbox, which "
            "is provisioned only when grading admits the rollout. The keys beside "
            "`provider` are the provider's sandbox spec."
        ),
    ),
    (
        "`[rollout_runtime.config.grading_sandbox]` on Blaxel",
        BlaxelSandbox,
        (
            'These keys sit beside provider = "blaxel". They are independent of the '
            "execution sandbox's spec."
        ),
    ),
    (
        "`[providers.blaxel]`",
        BlaxelSettings,
        (
            "This table configures the built-in Blaxel provider. Its credentials, "
            "BL_WORKSPACE and BL_API_KEY, come from the environment or a .env beside the "
            "config, never from this table."
        ),
    ),
    (
        "`[providers.daytona]`",
        DaytonaSettings,
        (
            "This table configures the built-in Daytona provider, which "
            "[daytona.md](daytona.md) describes. Its credential, DAYTONA_API_KEY, comes "
            "from the environment or a .env beside the config, never from this table. A "
            "self-hosted Daytona also needs DAYTONA_API_URL from the same place."
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
    """Describe a setting's type and bounds from pydantic's schema, not the annotation.

    The schema has already resolved a ``Literal`` to ``enum``, an optional to
    ``anyOf``, and every ``Field`` constraint to a bound. The annotation alone does not
    carry the bounds. A pipe would close the table cell, so an enum's values are
    joined with an escaped pipe.
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
    """Return the settings model that validates a field's TOML table, and the key suffix.

    The suffix is empty for the field's own table, or ``.<name>`` for a table of named
    tables. A table that may be absent is still a table. A table of named tables is one
    table per name, each with the same keys.
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
    """Yield every TOML table the settings declare, as ``(dotted name, model)``."""
    for name, info in model.model_fields.items():
        if table := _table(info.annotation):
            suffix, nested = table
            path = f"{prefix}{name}{suffix}"
            yield path, nested
            yield from sections(nested, f"{path}.")


def _leaves(model: type[BaseModel]) -> list[str]:
    """Return the keys that get a row in a table's own block.

    A nested table that is always present gets its own section instead. A table that
    may be absent, or that holds named tables, also keeps its row, to show its default.
    """
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
    """Render the reference tables, one per TOML section, in declaration order."""
    return "\n\n".join(
        _block(f"`[{section}]`", model, SECTION_BLURBS.get(section))
        for section, model in sections()
    )


def render_built_in_tables() -> str:
    """Render the built-in rollout runtimes' and providers' own tables."""
    return "\n\n".join(_block(*entry) for entry in BUILT_IN_SECTIONS)


def render_sft_tables() -> str:
    """Render the SFT recipe's tables, under headings of their own."""
    return "\n\n".join(
        _block(f"SFT `[{section}]`", model, None) for section, model in sections(SftConfig)
    )


def render_knob_glossary() -> str:
    """Render the whole glossary file, including the generated markers."""
    return f"""\
# Knob glossary

Every setting the cookbook reads. Settings live in `train.toml` under the section
shown, and the CLI exposes each key of a table that is always present by its dotted
path, such as `--rollout.max_concurrent_rollouts`.

A rollout runtime's own settings live in `[rollout_runtime.config]`, and the training
config treats them as opaque. The tables of the built-in rollout runtimes and providers
are listed after the training config's. For the Harbor rollout runtime,
`[rollout_runtime.config]` holds only `reward_key`. Agent behaviour and environment
configuration belong to the Harbor job. A custom provider defines its own
`[providers.<name>]`. For the settings that only make sense together, see
[`settings-that-interact.md`](settings-that-interact.md).

**Defaults or required.** A setting that changes what trains or what a run costs is
required, such as `learning_rate`, `group_size` or a sandbox's size. A setting has a
default only when it is purely operational, which means a wrong value fails loudly
and nothing about training depends on it. The `[rollout.cleanup]` graces are an
example. A setting that the platform has its own default for, such as `adam_beta2`, is
optional and has no value of ours. When it is unset, the request carries the Loops
SDK's default, which a `baseten-loops` release may change. An opt-in feature, such as
`router_replay`, is off unless it is set. A new setting follows the same rule.

{GENERATED_BEGIN}

{render_knob_tables()}

{render_built_in_tables()}

# SFT settings

An SFT config, `recipe = "sft"`, has its own TOML. Its `[model]`, `[compute]` and
`[output]` tables are the RL recipe's, spelled the same, except that SFT requires
`max_seq_len`.

{render_sft_tables()}

{GENERATED_END}
"""


GLOSSARY_PATH = Path(__file__).resolve().parents[2] / "docs" / "reference" / "KNOB_GLOSSARY.md"


if __name__ == "__main__":
    GLOSSARY_PATH.write_text(render_knob_glossary())
    print(f"wrote {GLOSSARY_PATH}")
