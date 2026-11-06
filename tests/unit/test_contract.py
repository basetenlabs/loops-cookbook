"""The normative runtime contract: exports, fields, signatures, and no ``Any``."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "loops_cookbook"


def _source(module: str) -> str:
    parts = module.split(".")[1:]
    path = SRC.joinpath(*parts)
    return (path / "__init__.py" if path.is_dir() else path.with_suffix(".py")).read_text()


PACKAGE_EXPORTS = {
    "loops_cookbook.runtime": {
        "AttemptAbortReason",
        "AttemptContext",
        "AttemptProcesses",
        "AttemptIdentity",
        "AttemptServices",
        "EpisodeResources",
        "AttemptSpec",
        "CapacityBroker",
        "EpisodeDisposition",
        "EpisodeResult",
        "GradeResult",
        "GroupGrader",
        "InternalPolicyDeltaError",
        "JsonValue",
        "MemberEvidence",
        "NamedFailure",
        "PermanentGradingFailure",
        "PermanentRolloutFailure",
        "Phase",
        "ProcessCommand",
        "ProcessResult",
        "Rollout",
        "RolloutRuntime",
        "RunResources",
        "RuntimeCapabilities",
        "RuntimeContext",
        "RuntimeEvidence",
        "RuntimeFiles",
        "RuntimeMetrics",
        "Sample",
        "StaleRolloutError",
        "TaskRef",
    },
    # What runtimes and agents use.
    "loops_cookbook.model_access": {
        "CaptureIntegrityFailure",
        "ModelAccess",
        "ModelAccessError",
        "ModelAccessMode",
        "ModelConnection",
        "ModelTransportFailure",
        "SamplingFailure",
    },
    # An agent subclasses LoopsAgent; the training TOML names create_runtime.
    "loops_cookbook.harbor": {"LoopsAgent", "create_runtime"},
    # The programmatic training surface.
    "loops_cookbook.rl": {
        "Config",
        "CredentialsError",
        "Outcome",
        "Pipeline",
        "Run",
        "StepReport",
        "Trainer",
    },
    # The client surface, and the transcript types a GroupMember's transcript is read with.
    "loops_cookbook.llm": {
        "CompletionTurnReply",
        "ParsedTurnReply",
        "SealedTranscript",
        "SealedTurn",
        "TranscriptTurn",
        "TurnDisposition",
        "AssistantMessage",
        "Content",
        "ContextWindowExceeded",
        "Conversation",
        "LLMResponse",
        "LoopsLLM",
        "Message",
        "ParsedReply",
        "PrimeRenderer",
        "RenderedPrompt",
        "Renderer",
        "RenderingError",
        "SystemMessage",
        "Tool",
        "ToolMessage",
        "ToolSchema",
        "UserMessage",
        "create_renderer",
        "describe",
    },
}


@pytest.mark.parametrize("package", sorted(PACKAGE_EXPORTS))
def test_each_package_exports_exactly_its_contract(package):
    module = importlib.import_module(package)
    assert set(module.__all__) == PACKAGE_EXPORTS[package]
    assert all(hasattr(module, name) for name in module.__all__)


FIELDS = {
    "TaskRef": ("id", "digest"),
    "AttemptIdentity": ("run_id", "group_id", "rollout_id", "attempt_id"),
    "AttemptSpec": ("identity", "task"),
    "AttemptServices": ("model_access", "capacity", "resources", "processes", "metrics"),
    "ProcessCommand": ("argv", "cwd", "env"),
    "ProcessResult": ("exit_code", "signal", "stdout_tail", "stderr_tail", "elapsed_seconds"),
    "AttemptContext": ("spec", "services"),
    "EpisodeResult": ("metadata",),
    "GradeResult": ("reward", "metadata"),
    "RuntimeCapabilities": (
        "model_access_modes",
        "internally_acquired_phases",
        "grading_retries",
        "pending_group_grade",
        "rollout_timeout_seconds",
        "model_access_source",
        "grading_retries_source",
    ),
    "MemberEvidence": ("rollout", "transcript"),
    "RuntimeContext": (
        "work_dir",
        "project_root",
        "config_dir",
        "files",
        "limits",
        "run_resources",
        "metrics",
        "evidence",
        "sandbox_providers",
        "provider_regions",
        "provider_capacity",
        "provider_settings",
        "provider_images",
    ),
    "Sample": (
        "sample_id",
        "producer",
        "conversation_id",
        "trainable",
        "prompt_ids",
        "completion_ids",
        "logprobs",
        "policy_version",
        "prompt_chunks",
        "stop_reason",
        "sequence_id",
        "max_tokens",
        "clamped_from",
    ),
    "Rollout": ("identity", "task", "samples", "reward", "metadata", "discarded_versions"),
}


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_public_dataclasses_have_exactly_the_normative_fields(name):
    runtime = importlib.import_module("loops_cookbook.runtime")
    cls = getattr(runtime, name)
    assert tuple(field.name for field in dataclasses.fields(cls)) == FIELDS[name]
    params = cls.__dataclass_params__
    assert params.frozen


NORMATIVE = {
    "prepare": ["self"],
    "sample": ["self", "attempt"],
    "grade": ["self", "attempt", "rollout"],
    "finish": ["self", "attempt", "rollout", "disposition"],
    "abort": ["self", "attempt", "reason"],
    "close": ["self"],
}
# What a runtime may add beside the hooks: ``GroupGrader``, for one that declares
# ``pending_group_grade``.
OPTIONAL = {"grade_group": ["self", "members"]}
PATTERNS = Path(__file__).resolve().parents[2] / "patterns"


def _runtime_class(where: str) -> type:
    """A runtime class by ``module:Class``, or by ``path/to/file.py:Class`` for an example."""
    module_name, _, name = where.partition(":")
    if module_name.endswith(".py"):
        spec = importlib.util.spec_from_file_location(
            Path(module_name).stem, PATTERNS / module_name
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    return getattr(module, name)


@pytest.mark.parametrize(
    "where",
    [
        "loops_cookbook.runtime:RolloutRuntime",
        "loops_cookbook.harbor.runtime:HarborRuntime",
        "loops_cookbook.episodes:EpisodeRuntime",
    ],
    ids=["protocol", "harbor", "episodes"],
)
def test_every_runtime_has_exactly_the_normative_signatures_and_nothing_public_besides(where):
    runtime = _runtime_class(where)
    for name, parameters in NORMATIVE.items():
        method = getattr(runtime, name)
        assert inspect.iscoroutinefunction(method), name
        assert list(inspect.signature(method).parameters) == parameters, name
    assert isinstance(inspect.getattr_static(runtime, "capabilities"), property)
    public = {name for name in vars(runtime) if not name.startswith("_")}
    extra = public - {*NORMATIVE, "capabilities"}
    assert extra <= OPTIONAL.keys(), extra
    for name in extra:
        method = getattr(runtime, name)
        assert inspect.iscoroutinefunction(method), name
        assert list(inspect.signature(method).parameters) == OPTIONAL[name], name


@pytest.mark.parametrize(
    "module",
    ["loops_cookbook.runtime.api", "loops_cookbook.model_access.api", "loops_cookbook.evidence"],
)
def test_the_public_contracts_contain_no_any(module):
    tree = ast.parse(_source(module))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "Any" not in names | attributes


def test_the_renderer_options_setting_is_json():
    from loops_cookbook.rl.config import ModelSettings

    annotation = str(ModelSettings.model_fields["renderer_chat_template_kwargs"].annotation)
    assert "Any" not in annotation and "JsonValue" in annotation


def _resolve(target: str) -> object:
    module, _, path = target.partition(":")
    value = importlib.import_module(module)
    for part in path.split("."):
        value = (
            inspect.getattr_static(value, part) if isinstance(value, type) else getattr(value, part)
        )
    return value


def annotations_of(value: object) -> list[str]:
    """Every annotation a caller sees on ``value``: a class's fields (its bases' too), a
    property's return, a callable's parameters and return."""
    if isinstance(value, property):
        value = value.fget
    if isinstance(value, type):
        return [
            str(annotation)
            for cls in value.__mro__
            for annotation in vars(cls).get("__annotations__", {}).values()
        ]
    if callable(value):
        signature = inspect.signature(value)
        return [str(p.annotation) for p in signature.parameters.values()] + [
            str(signature.return_annotation)
        ]
    return [str(value)]


NO_ANY = [
    "loops_cookbook.rl.recipe:train",
    "loops_cookbook.harbor.runtime:reward_of",
    "loops_cookbook.llm.client:Conversation.generate",
    "loops_cookbook.llm.client:LLMResponse.native_message",
    *(f"loops_cookbook.llm:{name}" for name in PACKAGE_EXPORTS["loops_cookbook.llm"]),
]


@pytest.mark.parametrize("target", sorted(NO_ANY))
def test_the_exported_entry_points_expose_no_any(target):
    assert not [text for text in annotations_of(_resolve(target)) if "Any" in text]


def test_model_access_offers_only_the_selected_mode_and_a_producer_locked_client():
    from loops_cookbook.llm.client import LoopsLLM
    from loops_cookbook.model_access import ModelAccess

    assert list(inspect.signature(ModelAccess.llm).parameters) == ["self", "producer"]
    assert list(inspect.signature(LoopsLLM.conversation).parameters) == ["self", "conversation_id"]
    assert list(inspect.signature(ModelAccess.http_connection).parameters) == ["self", "producer"]
