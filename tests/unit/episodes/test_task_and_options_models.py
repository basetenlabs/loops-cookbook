"""The task and options models an episode declares: the models and field types the
allow list accepts, and those it refuses because a dump would hide what validated.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

import pytest
from pydantic import (
    Secret,
    SecretStr,
)

from loops_cookbook.episodes.tasks import task_digest
from loops_cookbook.runtime.context import EvidenceRecord, RuntimeConfigurationError

from .declared import (
    DECLARED,
    DOCUMENT,
    ROLES,
    ComputedId,
    Documents,
    DumpOverridden,
    EverySerialized,
    Everything,
    FieldSerialized,
    Handle,
    Hidden,
    HiddenIf,
    HiddenWithin,
    JsonEncoded,
    Masked,
    ModelSerialized,
    PlainSerialized,
    ReadAs,
    SerializedInUnion,
    SerializedWithin,
    ServedAsId,
    Shouted,
    WrapSerializedWithin,
    build,
    build_declaring,
    with_field,
)


@pytest.mark.parametrize(
    ("config", "problem"),
    [
        (
            {"episode": f"{DECLARED}:Answer", "options": {"url": "u"}},
            "declares no options model",
        ),
        ({"episode": f"{DECLARED}:Answer", "options": {}}, "declares no options model"),
        ({"episode": f"{DECLARED}:Judged"}, r"(?s)not a valid Judge.*url"),  # required, absent
        ({"episode": f"{DECLARED}:Judged", "options": {"url": "u", "rubirc": "x"}}, "rubirc"),
        ({"episode": f"{DECLARED}:Judged", "options": {"url": 3}}, "not a valid Judge"),
        ({"episode": f"{DECLARED}:NotAModel"}, "task model <class 'dict'> is not a pydantic"),
        ({"episode": f"{DECLARED}:LenientTask"}, "task model Lenient reaches Lenient, which"),
        ({"episode": f"{DECLARED}:LenientOptions"}, "options model Lenient reaches Lenient,"),
        ({"episode": f"{DECLARED}:NestedLenient"}, "LenientWithin reaches Inner, which does"),
        ({"episode": f"{DECLARED}:TaskWithId"}, "task model WithId declares a field 'id'"),
        (
            {"episode": f"{DECLARED}:TaskWithAliasedId"},
            "AliasedId, whose field 'name' is read or dumped as 'id'",
        ),
        (
            {"episode": f"{DECLARED}:SecretOptions"},
            "Keyed, whose field 'api_key' is typed with SecretStr",
        ),
        (
            {"episode": f"{DECLARED}:NestedSecret"},
            "reaches Credentials, whose field 'token' is typed with SecretBytes",
        ),
        (
            {"episode": f"{DECLARED}:WrappedSecret"},
            r"Wrapped, whose field \'key\' is typed with pydantic.types.Secret\[str\]",
        ),
        ({"episode": f"{DECLARED}:LeftOpen"}, "leaves its task model open: ~TaskT"),
    ],
)
def test_what_an_episode_declares_is_checked_when_the_runtime_is_built(tmp_path, config, problem):
    with pytest.raises(RuntimeConfigurationError, match=problem):
        build(tmp_path, **config)


def test_an_options_model_with_every_default_needs_no_options_table(tmp_path):
    build(tmp_path, episode=f"{DECLARED}:JudgedByDefault")


@pytest.mark.parametrize(
    "episode",
    [
        "OnQuestions",
        "RefinedQuestions",
        "ThroughDefault",
        "ThroughDefaultAgain",
        "ModernQuestions",
        "ModernRefined",
    ],
)
async def test_a_generic_base_bound_by_a_subclass_declares_the_models_it_is_bound_to(
    tmp_path, episode
):
    runtime = build(
        tmp_path,
        tasks=({"prompt": "p", "answer": "a"},),
        episode=f"{DECLARED}:{episode}",
        options={"url": "u"},
    )
    (task,) = await runtime.prepare()
    assert runtime._task_digest(task.id) == task_digest({"prompt": "p", "answer": "a", "points": 1})


@pytest.mark.parametrize(
    ("episode", "problem"),
    [
        ("DataclassOptions", "DataOptions, whose field 'd' is typed with KeyedData,"),
        ("TypedDictOptions", "DictOptions, whose field 'd' is typed with KeyedDict,"),
        ("NamedTupleTask", "PairTask, whose field 'pairs' is typed with Pair,"),
        ("PydanticDataclassTask", "LooseTask, whose field 'd' is typed with Loose,"),
        ("NestedStructureTask", r"task model LooseWithin reaches LooseTask, whose field 'd'"),
    ],
)
def test_a_structure_that_is_not_a_pydantic_model_is_refused_by_model_and_field(
    tmp_path, episode, problem
):
    # Each hides what check reads from it: a secret, or a key it would drop unread.
    with pytest.raises(RuntimeConfigurationError, match=problem):
        build(tmp_path, episode=f"{DECLARED}:{episode}")


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize(
    ("model", "problem"),
    [
        (Hidden, r"reaches Hidden, whose field 'hidden' is excluded from its dump"),
        (HiddenIf, r"reaches HiddenIf, whose field 'hidden' is excluded from its dump"),
        (HiddenWithin, r"HiddenWithin reaches Hidden, whose field 'hidden' is excluded"),
        (ComputedId, r"ComputedId, which computes 'id'"),
        (FieldSerialized, r"FieldSerialized, whose field 'q' serializes its own way"),
        (EverySerialized, r"EverySerialized, whose field 'q' serializes its own way"),
        (ModelSerialized, r"ModelSerialized, which serializes itself its own way"),
        (JsonEncoded, r"JsonEncoded, whose field 'q' serializes its own way"),
        (SerializedWithin, r"SerializedWithin reaches FieldSerialized, whose field 'q' serial"),
        (SerializedInUnion, r"SerializedInUnion reaches FieldSerialized, whose field 'q' ser"),
        (PlainSerialized, r"PlainSerialized, whose field 'q' is annotated with PlainSerializer"),
        (WrapSerializedWithin, r"WrapSerializedWithin, whose field 'tags' is annotated with Wrap"),
        (ServedAsId, r"ServedAsId, whose field 'q' is read or dumped as 'id', not by its name"),
        (ReadAs, r"ReadAs, whose field 'q' is read or dumped as 'question', not by its name"),
    ],
)
def test_a_model_whose_dump_is_not_what_it_validated_is_refused_as_task_or_options(
    tmp_path, monkeypatch, role, model, problem
):
    # A task is digested by its dump, and the run records the options' digest: an
    # excluded, computed, renamed or re-serialized field would let two different
    # tasks, or two different options, share a digest, or the data disagree with task.id.
    with pytest.raises(RuntimeConfigurationError, match=rf"{role} .*{problem}"):
        build_declaring(tmp_path, monkeypatch, role, model, {"q": "a"})


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize(
    ("annotation", "shown"),
    [
        (dict, "dict"),  # untyped: the keys within go unchecked
        (Any, "Any"),
        (dict[str, Any], "Any"),
        (list, "list"),
        (dict[int, str], r"dict\[int, str\]"),
        (SecretStr, "SecretStr"),  # dumps as asterisks, so two secrets share a digest
        (SecretStr | None, "SecretStr"),
        (list[Secret[str]], r"Secret\[str\]"),
        (tuple[int, ...], r"tuple\[int, \.\.\.\]"),
        (set[str], r"set\[str\]"),  # dumped in no fixed order
        (datetime, "datetime"),
        (Handle, "Handle"),
        (Literal[b"x"], r"Literal\[b'x'\]"),
        (Annotated[str, "a note"], "is annotated with str"),
    ],
)
def test_a_field_type_outside_the_allow_list_is_refused_as_task_or_options_by_model_and_field(
    tmp_path, monkeypatch, role, annotation, shown
):
    with pytest.raises(
        RuntimeConfigurationError,
        match=rf"{role} Fielded reaches Fielded, whose field 'f' .*{shown}",
    ):
        build_declaring(tmp_path, monkeypatch, role, with_field(annotation), {})


@pytest.mark.parametrize("role", ROLES)
def test_every_kind_of_field_the_allow_list_names_is_accepted_as_task_or_options(
    tmp_path, monkeypatch, role
):
    value = {
        "text": "t",
        "colour": 1,
        "points": {"a": [{"x": 1}]},
        "shouted": "hi",
        "tree": {"children": [{}]},
    }
    build_declaring(tmp_path, monkeypatch, role, Everything, value)


@pytest.mark.parametrize(
    ("role", "where"),
    [("task model", r"tasks\.jsonl:1"), ("options model", r"\[rollout_runtime\.config\.options\]")],
)
def test_a_validated_value_that_does_not_dump_as_its_field_type_is_refused(
    tmp_path, monkeypatch, role, where
):
    # It would dump as asterisks, so two different values would share a digest.
    with pytest.raises(
        RuntimeConfigurationError,
        match=rf"{where}: the validated Masked does not dump as its field types",
    ):
        build_declaring(tmp_path, monkeypatch, role, Masked, {"q": "a"})


@pytest.mark.parametrize("role", ROLES)
async def test_a_model_dump_override_is_not_what_task_or_options_are_digested_by(
    tmp_path, monkeypatch, role
):
    evidence = EvidenceRecord()
    runtime = build_declaring(
        tmp_path, monkeypatch, role, DumpOverridden, {"q": "a"}, evidence=evidence
    )
    if role == "task model":
        (task,) = await runtime.prepare()
        digest = runtime._task_digest(task.id)
    else:
        digest = evidence.digests["options"]
    assert digest == task_digest({"q": "a"})


@pytest.mark.parametrize("role", ROLES)
def test_a_recursive_json_type_is_accepted_as_task_or_options(tmp_path, monkeypatch, role):
    value = {"cookbooks": DOCUMENT, "pydantics": DOCUMENT, "owns": DOCUMENT}
    build_declaring(tmp_path, monkeypatch, role, Documents, value)


@pytest.mark.parametrize("role", ROLES)
def test_a_type_alias_is_checked_as_what_it_names(tmp_path, monkeypatch, role):
    with pytest.raises(
        RuntimeConfigurationError,
        match=rf"{role} Fielded reaches Fielded, whose field 'f' is annotated with PlainSerializer",
    ):
        build_declaring(tmp_path, monkeypatch, role, with_field(Shouted), {})
