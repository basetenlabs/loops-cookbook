"""Code a user copies imports each cookbook name from its package, never a submodule.

Examples, patterns and the docs' code blocks are what customers copy, so the import
they show is the one a name is known by. A submodule path works too, which is how
two spellings of the same import spread; this keeps them to the public surface.
"""

import ast
import importlib
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# The public import roots: the packages (each with ``__all__``), the two
# composition modules the README drives an RL run from, and the shared CLI helpers.
PACKAGES = (
    "loops_cookbook.episodes",
    "loops_cookbook.harbor",
    "loops_cookbook.llm",
    "loops_cookbook.model_access",
    "loops_cookbook.providers.blaxel",
    "loops_cookbook.resources",
    "loops_cookbook.rl",
    "loops_cookbook.runtime",
    "loops_cookbook.sft",
)
MODULES = (
    "loops_cookbook.cli",
    "loops_cookbook.project",
    "loops_cookbook.rl.cli",
    "loops_cookbook.rl.recipe",
    "loops_cookbook.sft.cli",
    "loops_cookbook.sft.recipe",
)


def _tracked(*patterns: str) -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", *patterns], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    return [ROOT / name for name in listed]


def _sources() -> list[tuple[str, str]]:
    sources = [
        (str(path.relative_to(ROOT)), path.read_text())
        for path in _tracked("examples/**.py", "patterns/**.py")
    ]
    for path in _tracked("README.md", "docs/**.md", "examples/**.md", "patterns/**.md"):
        for number, block in enumerate(
            re.findall(r"```python\n(.*?)```", path.read_text(), re.DOTALL)
        ):
            sources.append((f"{path.relative_to(ROOT)} block {number}", block))
    return sources


def _imports(source: str) -> list[tuple[str, tuple[str, ...]]]:
    """Every cookbook module imported, with the names taken from it."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("loops_cookbook"):
            found.append((node.module or "", tuple(alias.name for alias in node.names)))
        elif isinstance(node, ast.Import):
            found += [(a.name, ()) for a in node.names if a.name.startswith("loops_cookbook")]
    return found


SOURCES = _sources()


def test_the_copied_code_is_found():
    """A glob that matched nothing would pass every check below."""
    names = {where.split(" ")[0] for where, _ in SOURCES}
    assert "examples/terminal_bench/terminal_agent.py" in names
    assert "docs/reference/harbor.md" in names


@pytest.mark.parametrize(("where", "source"), SOURCES, ids=[where for where, _ in SOURCES])
def test_copied_code_imports_only_the_public_surface(where, source):
    for module, names in _imports(source):
        assert module in PACKAGES + MODULES, f"{where}: {module} is not a public import root"
        if module in PACKAGES:
            exported = set(importlib.import_module(module).__all__)
            assert set(names) <= exported, f"{where}: {sorted(set(names) - exported)}"


def test_every_name_is_exported_by_exactly_one_package():
    """A name's one import path is its defining package; a re-export is a second.

    Two packages may each export a name they define themselves -- each recipe has a
    ``Config`` and a ``Run``, each rollout runtime a ``create_runtime`` -- since those are
    different objects, not two spellings of one.
    """
    exporters: dict[str, list[str]] = {}
    for package in PACKAGES:
        for name in importlib.import_module(package).__all__:
            exporters.setdefault(name, []).append(package)
    for name, found in exporters.items():
        if len(found) > 1:
            for package in found:
                defined = getattr(importlib.import_module(package), name).__module__
                assert defined.startswith(package), f"{package} re-exports {name} from {defined}"
