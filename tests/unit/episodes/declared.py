"""Episodes, models and sandbox specs the factory tests declare, and ``build()``.

The tests name an episode by path, as ``[runtime.config] episode`` does:
``f"{DECLARED}:Name"``.
"""

from __future__ import annotations

import dataclasses
import enum
import sys
import types
from types import SimpleNamespace as NS
from typing import Annotated, Any, Literal, NamedTuple, TypedDict

import pydantic
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    Secret,
    SecretBytes,
    SecretStr,
    StringConstraints,
    WrapSerializer,
    computed_field,
    field_serializer,
    model_serializer,
)
from support.builders import runtime_context, task_file
from support.resources import Fake
from typing_extensions import TypeVar

from loops_cookbook.episodes import Episode, Grading, Sampling, create_runtime
from loops_cookbook.providers.blaxel import Blaxel
from loops_cookbook.resources import SandboxKind, SandboxSpec
from loops_cookbook.runtime import JsonValue
from loops_cookbook.runtime.context import EvidenceRecord

DECLARED = __name__


class Answer(Episode[None, None]):
    async def run(self, sample: Sampling):
        return None

    async def grade(self, grading: Grading):
        return 0.0


class NoGrade(Episode[None, None]):
    async def run(self, sample: Sampling):
        return None


class NoRun(Episode[None, None]):
    async def grade(self, grading: Grading):
        return 0.0


class SyncSetup(Answer):
    def setup(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        pass


class SyncRun(Answer):
    def run(self, model):  # pyright: ignore[reportIncompatibleMethodOverride]
        return None


class SyncGrade(Answer):
    def grade(self, rollout):  # pyright: ignore[reportIncompatibleMethodOverride]
        return 0.0


class SyncClose(Answer):
    def close(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        pass


class StringAccess(Answer):
    model_access = "direct"  # pyright: ignore[reportAssignmentType]


class InstanceGrader(Episode[None, None]):
    async def run(self, sample: Sampling):
        return None

    async def grade_group(self, members, options):  # pyright: ignore[reportIncompatibleMethodOverride]
        return []


class NotAnEpisode:
    async def run(self, sample: Sampling):
        return None


def context(
    tasks_path,
    *,
    rollout=60.0,
    grace=5.0,
    kill_grace=4.0,
    providers=None,
    sandbox_providers=None,
    evidence=None,
):
    """A runtime context; every configured provider provides ``Fake`` sandboxes unless
    ``sandbox_providers`` says otherwise."""
    return runtime_context(
        tasks_path,
        limits=NS(
            rollout_timeout_seconds=rollout,
            cancellation_grace_seconds=grace,
            process_kill_grace_seconds=kill_grace,
            provider_deadlines=providers or {},
            trajectory_tokens=None,
        ),
        work_dir=tasks_path.parent / "work",
        evidence=evidence or EvidenceRecord(),
        sandbox_providers=dict.fromkeys(providers or {}, Fake)
        if sandbox_providers is None
        else sandbox_providers,
    )


ABSENT = object()  # a setting the [runtime.config] table leaves out


def build(tmp_path, *, tasks=({"id": "t"},), limits=None, evidence=None, **config):
    settings = {"episode": f"{__name__}:Answer", "agent_timeout_seconds": 10, **config}
    settings = {key: value for key, value in settings.items() if value is not ABSENT}
    return create_runtime(
        settings,
        context=context(task_file(tmp_path, list(tasks)), evidence=evidence, **(limits or {})),
    )


class OnFake(Answer, Episode[None, None]):
    execution_kind = Fake


class OnBlaxel(Answer, Episode[None, None]):
    execution_kind = Blaxel


class Unnamed(SandboxKind[str]):  # a kind that names no provider
    pass


class OnUnnamed(Answer, Episode[None, None]):
    execution_kind = Unnamed


class OnNotAKind(Answer, Episode[None, None]):  # pyright: ignore[reportInvalidTypeArguments]
    execution_kind = str


KindT = TypeVar("KindT", bound=SandboxKind[Any])


class OnAnyKind(Episode[None, None]):  # a generic base of the user's own
    execution_kind = KindT

    async def run(self, sample: Sampling):
        return None

    async def grade(self, grading: Grading):
        return 0.0


class OnFakeKind(OnAnyKind):
    execution_kind = Fake


class KindLeftOpen(OnAnyKind):  # pyright: ignore[reportMissingTypeArgument]
    pass


FAKE_SANDBOX = {"provider": "fake", "image": "x"}


class LenientSpec(SandboxSpec):  # a key it does not declare would reach nothing
    model_config = ConfigDict(extra="allow")


class KeyedSpec(SandboxSpec):  # config.json would hold the secret
    token: SecretStr = SecretStr("t")


class RenamedSpec(SandboxSpec):
    image: str = Field(default="x", alias="img")


class SerializedSpec(SandboxSpec):  # what is digested would not be what validated
    image: str = "x"

    @field_serializer("image")
    def _same(self, image: str) -> str:
        return "same"


class LooseSpec(SandboxSpec):
    labels: dict = Field(default_factory=dict)  # pyright: ignore[reportMissingTypeArgument]


class NotASpec(BaseModel):
    model_config = ConfigDict(extra="forbid")


def fake_kind(spec):
    """A kind of ``Fake`` sandbox, as a provider would ship it, typed by ``spec``."""
    return types.new_class("FakeKind", (Fake,), exec_body=lambda body: body.update(spec=spec))


class NotACount(Answer):
    grading_retries = "2"  # pyright: ignore[reportAssignmentType]


FORBID = ConfigDict(extra="forbid")


class Question(BaseModel):
    model_config = FORBID

    prompt: str
    answer: str
    points: int = 1  # a default the raw line does not carry


class Declares(Answer, Episode[Question, None]):
    pass


class Judge(BaseModel):
    model_config = FORBID

    url: str
    rubric: str = "exact"


class Judged(Episode[None, Judge]):
    async def run(self, sample: Sampling):
        return None

    async def grade(self, grading: Grading):
        return 0.0


class DefaultsOnly(BaseModel):
    model_config = FORBID

    rubric: str = "exact"


class JudgedByDefault(Judged, Episode[None, DefaultsOnly]):
    pass


class Lenient(BaseModel):  # pydantic's default: an undeclared key is dropped
    answer: str = ""


class Inner(BaseModel):
    note: str = ""


class LenientWithin(BaseModel):
    model_config = FORBID

    inner: list[Inner] = []


class WithId(BaseModel):
    model_config = FORBID

    id: str = "same"
    prompt: str


class AliasedId(BaseModel):
    model_config = FORBID

    name: str = Field(alias="id")


class Keyed(BaseModel):
    model_config = FORBID

    api_key: SecretStr


class Credentials(BaseModel):
    model_config = FORBID

    token: SecretBytes | None = None


class KeyedWithin(BaseModel):
    model_config = FORBID

    judges: dict[str, list[Credentials]] = {}


class Wrapped(BaseModel):
    model_config = FORBID

    key: Secret[str]


class LenientTask(Answer, Episode[Lenient, None]):
    pass


class LenientOptions(Answer, Episode[None, Lenient]):
    pass


class NestedLenient(Answer, Episode[None, LenientWithin]):
    pass


class TaskWithId(Answer, Episode[WithId, None]):
    pass


class TaskWithAliasedId(Answer, Episode[AliasedId, None]):
    pass


class SecretOptions(Answer, Episode[None, Keyed]):
    pass


class NestedSecret(Answer, Episode[None, KeyedWithin]):
    pass


class WrappedSecret(Answer, Episode[None, Wrapped]):
    pass


class NotAModel(Answer, Episode[dict, None]):
    pass


@dataclasses.dataclass
class KeyedData:  # a stdlib dataclass: pydantic validates it, and check did not look inside
    key: SecretStr


class KeyedDict(TypedDict):
    key: SecretStr


class Pair(NamedTuple):
    a: int
    b: int


@pydantic.dataclasses.dataclass(config=ConfigDict(extra="ignore"))
class Loose:
    a: int


class DataOptions(BaseModel):
    model_config = FORBID

    d: KeyedData


class DictOptions(BaseModel):
    model_config = FORBID

    d: dict[str, KeyedDict] = {}


class PairTask(BaseModel):
    model_config = FORBID

    pairs: list[Pair] = []


class LooseTask(BaseModel):
    model_config = FORBID

    d: Loose


class LooseWithin(BaseModel):  # a structure two models deep
    model_config = FORBID

    tasks: list[LooseTask] = []


class DataclassOptions(Answer, Episode[None, DataOptions]):
    pass


class TypedDictOptions(Answer, Episode[None, DictOptions]):
    pass


class NamedTupleTask(Answer, Episode[PairTask, None]):
    pass


class PydanticDataclassTask(Answer, Episode[LooseTask, None]):
    pass


class NestedStructureTask(Answer, Episode[LooseWithin, None]):
    pass


class Hidden(BaseModel):
    model_config = FORBID

    q: str
    hidden: str = Field(default="", exclude=True)


class HiddenIf(BaseModel):
    model_config = FORBID

    q: str
    hidden: str = Field(default="", exclude_if=lambda value: value == "y")


class ComputedId(BaseModel):
    model_config = FORBID

    q: str

    @computed_field
    @property
    def id(self) -> str:
        return "zz"


class FieldSerialized(BaseModel):
    model_config = FORBID

    q: str

    @field_serializer("q")
    def _q(self, value: str) -> str:
        return "same"


class EverySerialized(BaseModel):
    model_config = FORBID

    q: str

    @field_serializer("*")
    def _every(self, value: str) -> str:
        return "same"


class ModelSerialized(BaseModel):
    model_config = FORBID

    q: str

    @model_serializer
    def _all(self) -> dict[str, str]:
        return {"q": "same"}


class PlainSerialized(BaseModel):
    model_config = FORBID

    q: Annotated[str, PlainSerializer(lambda value: "same")]


class WrapSerializedWithin(BaseModel):
    model_config = FORBID

    q: str
    tags: list[Annotated[str, WrapSerializer(lambda value, handler: "same")]] = []


class JsonEncoded(BaseModel):  # the config's encoder dumps every str as one value
    model_config = ConfigDict(extra="forbid", json_encoders={str: lambda value: "same"})

    q: str


class SerializedWithin(BaseModel):
    model_config = FORBID

    inner: FieldSerialized


class SerializedInUnion(BaseModel):
    model_config = FORBID

    inner: int | FieldSerialized = 0


class HiddenWithin(BaseModel):
    model_config = FORBID

    inner: Hidden


class ServedAsId(BaseModel):  # dumps q as "id", so the data would name another task
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    q: str = Field(serialization_alias="id")


class ReadAs(BaseModel):
    model_config = FORBID

    q: str = Field(validation_alias="question")


TaskT = TypeVar("TaskT", bound=BaseModel)


class OnAnyTask(Episode[TaskT, Judge]):  # a generic base of the user's own
    async def run(self, sample: Sampling):
        return None

    async def grade(self, grading: Grading):
        return 0.0


class OnQuestions(OnAnyTask[Question]):
    pass


class RefinedQuestions(OnQuestions):  # subclassed again without type arguments
    pass


DefaultedT = TypeVar("DefaultedT", bound=BaseModel, default=Question)


class OnDefaultTask(Episode[DefaultedT, Judge]):  # a generic base whose parameter has a default
    async def run(self, sample: Sampling):
        return None

    async def grade(self, grading: Grading):
        return 0.0


class ThroughDefault(OnDefaultTask):  # pyright: ignore[reportMissingTypeArgument]
    pass


class ThroughDefaultAgain(ThroughDefault):  # an intermediate class between
    pass


class LeftOpen(OnAnyTask):  # pyright: ignore[reportMissingTypeArgument]
    pass


ROLES = ["task model", "options model"]


def declaring(monkeypatch, role, model):
    """An episode, importable by path, declaring ``model`` as its task or options model."""
    task, options = (model, None) if role == "task model" else (None, model)
    episode = types.new_class("Declaring", (Answer, Episode[task, options]))
    monkeypatch.setattr(sys.modules[__name__], "Declaring", episode, raising=False)
    return f"{__name__}:Declaring"


def build_declaring(tmp_path, monkeypatch, role, model, value, **settings):
    """Build with ``model`` as the role's model, and ``value`` as its one task or its options."""
    episode = declaring(monkeypatch, role, model)
    if role == "task model":
        return build(tmp_path, tasks=(value,), episode=episode, **settings)
    return build(tmp_path, episode=episode, options=value, **settings)


def with_field(annotation, **field):
    return pydantic.create_model(
        "Fielded", __config__=FORBID, f=(annotation, Field(default=None, **field))
    )


class Colour(enum.Enum):
    RED = "red"
    ONE = 1


class Handle(enum.Enum):  # a member whose value is no JSON scalar
    PAIR = (1, 2)


class Point(BaseModel):
    model_config = FORBID

    x: float
    label: Annotated[str, StringConstraints(min_length=1)] = "p"


class Node(BaseModel):  # a model that reaches itself
    model_config = FORBID

    children: list[Node] = []


class Everything(BaseModel):
    """Every kind of field the allow list names."""

    model_config = FORBID

    text: str = Field(min_length=1)
    count: int = 0
    ratio: float = 0.5
    flag: bool = False
    nothing: None = None
    mode: Literal["a", "b", 1, True] | None = "a"
    colour: Colour = Colour.RED
    tags: list[str] = []
    weights: dict[str, float] = {}
    maybe: int | None = None
    either: int | str = 0
    point: Point | None = None
    points: dict[str, list[Point]] = {}
    shouted: Annotated[str, AfterValidator(str.upper)] = ""
    tree: Node | None = None


class Masked(BaseModel):  # its types are allowed, and its validator returns another type
    model_config = FORBID

    q: Annotated[str, AfterValidator(SecretStr)] = ""


class DumpOverridden(BaseModel):  # the cookbook dumps what it validated, not what this says
    model_config = FORBID

    q: str

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {"q": "same"}


type OwnJson = None | bool | int | float | str | list[OwnJson] | dict[str, OwnJson]


class Documents(BaseModel):
    model_config = FORBID

    cookbooks: JsonValue = None
    pydantics: pydantic.JsonValue = None
    owns: dict[str, OwnJson] = {}


DOCUMENT = {"a": [1, 2.5, "s", None, True, {"b": [], "c": {}}]}


type Shouted = Annotated[str, PlainSerializer(lambda value: "same")]


class SamplingLifetime(Answer):
    sandbox_lifetime = "sampling"


class InvalidLifetime(Answer):
    sandbox_lifetime = "typo"


class SeparateSandboxes(Answer, Episode[None, None]):
    execution_kind = Fake
    grading_kind = Fake


class GradeSandboxOnly(Answer, Episode[None, None]):
    execution_kind = None
    grading_kind = Fake


class GroupOnlySandbox(Episode[None, None]):
    execution_kind = None
    grading_kind = Fake

    async def run(self, sample: Sampling):
        return None

    @classmethod
    async def grade_group(cls, members, options):
        return [0.0] * len(members)


VerifierT = TypeVar("VerifierT", bound=SandboxKind[Any])


class GenericVerifier(Answer, Episode[None, None]):
    execution_kind = None
    grading_kind = VerifierT


class BoundVerifier(GenericVerifier):
    grading_kind = Fake


class UnboundVerifier(GenericVerifier):
    pass


class ModernTask[TaskT: BaseModel](Episode[TaskT, Judge]):
    async def run(self, sample: Sampling):
        return None

    async def grade(self, grading: Grading):
        return 0.0


class ModernQuestions(ModernTask[Question]):
    pass


class ModernRefined(ModernQuestions):
    pass


class NoPhaseSetup(Answer):
    async def setup(self):
        pass


class ExtraRunArgument(Answer):
    async def run(self, sample, injected_sandbox):
        pass


class OldConstructor(Answer):
    def __init__(self, task):
        pass
