"""Executable architecture: dependency direction, cycles, exports, and the normative API.

Imports are read from source, including inline and ``TYPE_CHECKING`` imports, so a
dependency cannot be hidden from these rules by where it is written.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "loops_cookbook"
CORE = (
    "evidence",
    "llm",
    "model_access",
    "runtime",
    "rl",
    "sft",
    "trainer",
    "settings",
    "cli",
    "telemetry",
    "metrics",
    "run_record",
)


def _modules() -> dict[str, Path]:
    found = {}
    for path in SRC.rglob("*.py"):
        parts = path.relative_to(SRC.parent).with_suffix("").parts
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        found[name] = path
    return found


def _imports(path: Path) -> set[str]:
    return _imports_of(ast.parse(path.read_text()))


def _imports_of(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _package(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else parts[0]


MODULES = _modules()


RECIPES = ("rl", "sft")
# A recipe's composition roots: the modules that wire its core to the providers and
# runtimes a run uses. Everything else in a recipe's package is core.
COMPOSITION_ROOTS = ("cli", "recipe")


# The `loops` command and `project.load` wire every recipe together, so they are roots too.
ROOTS = (
    *(f"loops_cookbook.{recipe}.{root}" for recipe in RECIPES for root in COMPOSITION_ROOTS),
    "loops_cookbook.command",
    "loops_cookbook.project",
)


def _composition_root(module: str) -> bool:
    """A root, or anything inside one that grew into a package."""
    return any(module == root or module.startswith(root + ".") for root in ROOTS)


@pytest.mark.parametrize(("recipe", "other"), [("sft", "rl"), ("rl", "sft")])
def test_recipes_share_only_through_common_modules(recipe, other):
    forbidden = f"loops_cookbook.{other}"
    offending = sorted(
        (module, name)
        for module in MODULES
        if _package(module) == recipe
        for name in _imports(MODULES[module])
        if name == forbidden or name.startswith(forbidden + ".")
    )
    assert not offending, f"{recipe} imports {other}: {offending}"


@pytest.mark.parametrize(
    "module", sorted(m for m in MODULES if not _composition_root(m)), ids=lambda m: m
)
def test_only_a_composition_root_imports_one(module):
    """Otherwise a core module, or a package `__init__` re-exporting a root, would pull
    providers and Harbor into core one import away from where the rule looks."""
    tree = ast.parse(MODULES[module].read_text())
    # `from loops_cookbook.rl import recipe` imports the root as surely as naming it.
    named = {
        f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    }
    offending = sorted(name for name in _imports_of(tree) | named if _composition_root(name))
    assert not offending, f"{module} imports a composition root: {offending}"


@pytest.mark.parametrize(
    "module",
    sorted(m for m in MODULES if _package(m) in CORE and not _composition_root(m)),
    ids=lambda m: m,
)
def test_core_packages_import_no_harbor_provider_or_supervision_code(module):
    forbidden = ("harbor", "blaxel", "loops_cookbook.harbor", "loops_cookbook.providers")
    offending = sorted(
        name
        for name in _imports(MODULES[module])
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
    )
    assert not offending, f"{module} imports {offending}"


@pytest.mark.parametrize(
    "module", sorted(m for m in MODULES if _package(m) == "providers"), ids=lambda m: m
)
def test_providers_import_no_harbor_code(module):
    """Provider rules live under ``providers``; Harbor's hooks stay in ``harbor``."""
    offending = sorted(
        name
        for name in _imports(MODULES[module])
        if name.split(".")[0] == "harbor" or name.startswith("loops_cookbook.harbor")
    )
    assert not offending, f"{module} imports {offending}"


def test_cookbook_packages_have_no_dependency_cycle():
    graph: dict[str, set[str]] = {}
    for module, path in MODULES.items():
        source = _package(module)
        for name in _imports(path):
            if name.startswith("loops_cookbook.") or name == "loops_cookbook":
                target = _package(name)
                if target != source and target != "loops_cookbook":
                    graph.setdefault(source, set()).add(target)
    visiting, done = set(), set()

    def visit(node, trail):
        if node in visiting:
            raise AssertionError(" -> ".join([*trail, node]))
        if node in done:
            return
        visiting.add(node)
        for child in sorted(graph.get(node, ())):
            visit(child, [*trail, node])
        visiting.discard(node)
        done.add(node)

    for node in sorted(graph):
        visit(node, [])


def test_the_evidence_types_depend_on_no_cookbook_package():
    imported = _imports(MODULES["loops_cookbook.evidence"])
    assert not any(name.startswith("loops_cookbook") for name in imported)


# The audited seams: the integration, and each provider's Harbor environment.
SEAMS = {
    "loops_cookbook.harbor.integration",
    "loops_cookbook.harbor.blaxel",
    "loops_cookbook.harbor.daytona",
}


def _harbor_names(tree: ast.Module) -> set[str]:
    """Names a module binds to Harbor objects, by ``from harbor... import`` or ``import``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("harbor"):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(
                (alias.asname or alias.name).split(".")[0]
                for alias in node.names
                if alias.name.startswith("harbor")
            )
    return names


def _root(node: ast.expr) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _last(node: ast.expr) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else None


def _private(name: str) -> bool:
    return name.startswith("_") and not name.endswith("__")


def _harbor_derived(sources: dict[str, str]) -> set[str]:
    """Cookbook classes deriving from Harbor, directly or through another cookbook class."""
    bases: dict[str, set[str]] = {}
    derived: set[str] = set()
    for source in sources.values():
        tree = ast.parse(source)
        harbor = _harbor_names(tree)
        for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            bases[cls.name] = {name for base in cls.bases if (name := _last(base))}
            if any(_root(base) in harbor for base in cls.bases):
                derived.add(cls.name)
    grew = True
    while grew:
        grew = False
        for name, parents in bases.items():
            if name not in derived and parents & derived:
                derived.add(name)
                grew = True
    return derived


HARBOR_DERIVED = _harbor_derived({name: path.read_text() for name, path in MODULES.items()})


def _violations(module: str, source: str, derived: set[str]) -> list[str]:
    """Private Harbor members reached outside the audited seam, in one module's source."""
    if module in SEAMS:
        return []
    tree = ast.parse(source)
    harbor = _harbor_names(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("harbor"):
            found.extend(f"imports {alias.name}" for alias in node.names if _private(alias.name))
    for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        if not any(_root(base) in harbor or _last(base) in derived for base in cls.bases):
            continue
        for node in ast.walk(cls):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _private(node.name):
                found.append(f"{cls.name} overrides {node.name}")
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "super"
                and _private(node.attr)
            ):
                found.append(f"{cls.name} calls super().{node.attr}")
    if harbor:
        # A module that holds Harbor objects touches no private member of anything
        # but itself: reads and assignments alike.
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and _private(node.attr)
                and not (isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"))
            ):
                found.append(f"line {node.lineno} reaches {ast.unparse(node)}")
    return found


@pytest.mark.parametrize("module", sorted(MODULES), ids=lambda m: m)
def test_only_the_audited_seam_reaches_private_harbor_members(module):
    assert not _violations(module, MODULES[module].read_text(), HARBOR_DERIVED)


@pytest.mark.parametrize(
    "source",
    [
        # overriding a private member through the seam's own Harbor subclass
        (
            "from loops_cookbook.harbor import integration\n"
            "class Q(integration.HookedTrialQueue):\n    def _setup_hooks(self, trial): ...\n"
        ),
        # calling up to a private member
        (
            "from harbor.trial.queue import TrialQueue\n"
            "class Q(TrialQueue):\n    def setup(self, t):\n        super()._setup_hooks(t)\n"
        ),
        # assigning a private attribute of a Harbor object
        (
            "from harbor.trial.trial import Trial\n"
            "def patch(trial: Trial):\n    trial._stop_agent_environment = None\n"
        ),
        # reading one
        "import harbor\ndef read(trial):\n    return trial._trial_lock\n",
        # importing a private Harbor name
        "from harbor.trial.trial import _load_task\n",
    ],
)
def test_the_seam_rule_catches_each_way_of_reaching_a_private_member(source):
    assert _violations("loops_cookbook.harbor.example", source, HARBOR_DERIVED)


def _sdk_violations(module: str, source: str, sdk: str = "blaxel") -> list[str]:
    """A provider's SDK, or a private member of one of its objects, reached outside the
    provider.

    A module that holds the SDK's objects, through the SDK or Harbor's environment for
    that provider, may reach private members only of itself.
    """
    if module == f"loops_cookbook.providers.{sdk}" or module.startswith(
        f"loops_cookbook.providers.{sdk}."
    ):
        return []
    tree = ast.parse(source)
    imported = _imports_of(tree)
    found = [f"imports {name}" for name in imported if name.split(".")[0] == sdk]
    if any(
        name.split(".")[0] == sdk or name.startswith(f"harbor.environments.{sdk}")
        for name in imported
    ):
        found += [
            f"line {node.lineno} reaches {ast.unparse(node)}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and _private(node.attr)
            and not (isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"))
        ]
    return found


@pytest.mark.parametrize("sdk", ["blaxel", "daytona"])
@pytest.mark.parametrize("module", sorted(MODULES), ids=lambda m: m)
def test_only_a_provider_reaches_its_sdk_or_its_objects_private_members(module, sdk):
    assert not _sdk_violations(module, MODULES[module].read_text(), sdk)


@pytest.mark.parametrize(
    "source",
    [
        "from blaxel.core import ImageInstance\n",
        "import blaxel.core\n",
        # a Blaxel object Harbor's environment returned
        (
            "from harbor.environments.blaxel import BlaxelEnvironment\n"
            "class E(BlaxelEnvironment):\n"
            "    def spec(self):\n        return self._build_image_from_dockerfile(None)._context\n"
        ),
    ],
)
def test_the_blaxel_rule_catches_sdk_imports_and_private_members(source):
    assert _sdk_violations("loops_cookbook.harbor.example", source)


@pytest.mark.parametrize(
    "source",
    [
        "from daytona import AsyncDaytona\n",
        "import daytona.common.image\n",
        (
            "from harbor.environments.daytona import DaytonaEnvironment\n"
            "class E(DaytonaEnvironment):\n"
            "    def spec(self, image):\n        return image._context_list\n"
        ),
    ],
)
def test_the_daytona_rule_catches_sdk_imports_and_private_members(source):
    assert _sdk_violations("loops_cookbook.harbor.example", source, "daytona")


# Provider modules Harbor code may use: usage and request-building, never an adapter.
HARBOR_MAY_USE = {
    "loops_cookbook.providers.blaxel.api",
    "loops_cookbook.providers.blaxel.build_file",
    "loops_cookbook.providers.blaxel.images",
    "loops_cookbook.providers.blaxel.incidents",
    "loops_cookbook.providers.blaxel.usage",
    "loops_cookbook.providers.daytona.api",
    "loops_cookbook.providers.daytona.settings",
}


@pytest.mark.parametrize(
    "module", sorted(m for m in MODULES if _package(m) == "harbor"), ids=lambda m: m
)
def test_harbor_code_never_holds_a_provider_adapter(module):
    """An adapter could create or delete behind the registry; usage code cannot."""
    offending = sorted(
        name
        for name in _imports(MODULES[module])
        if name.startswith("loops_cookbook.providers") and name not in HARBOR_MAY_USE
    )
    assert not offending, f"{module} imports {offending}"
