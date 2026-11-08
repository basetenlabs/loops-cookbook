"""The core and a non-Harbor rollout runtime work with Harbor not installed.

Each test runs a fresh interpreter in which ``import harbor`` fails, as it does when
the ``harbor`` extra is absent; CI also runs these with Harbor really uninstalled.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATHS = [
    ROOT / "tests" / "fixtures" / "offline",
    ROOT / "patterns" / "custom_runtime" / "in_process_direct",
]

WITHOUT_HARBOR = """
import importlib.abc, sys

class NoHarbor(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "harbor" or name.startswith("harbor."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)

sys.meta_path.insert(0, NoHarbor())
"""


def without_harbor(body: str, cwd: Path) -> subprocess.Popen[str]:
    """Start ``body`` in a fresh interpreter where ``import harbor`` fails."""
    # A fresh interpreter has none of pytest's HTTP fakes. Keep the offline check
    # independent of developer credentials, including ones a local .env would load.
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(str(path) for path in PATHS),
        "BASETEN_API_KEY": "",
        "PYTHON_DOTENV_DISABLED": "1",
    }
    return subprocess.Popen(
        [sys.executable, "-c", WITHOUT_HARBOR + textwrap.dedent(body)],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


CORE_IMPORTS = """
import importlib, pkgutil
import loops_cookbook
from loops_cookbook.runtime.context import RuntimeConfigurationError, import_factory

# The Harbor rollout runtime and the providers' SDK code need their own extras. Each
# provider package itself imports without them.
extras = (
    "loops_cookbook.harbor",
    "loops_cookbook.providers.blaxel.",
    "loops_cookbook.providers.daytona.",
)
for module in pkgutil.walk_packages(loops_cookbook.__path__, "loops_cookbook."):
    if not module.name.startswith(extras):
        importlib.import_module(module.name)
try:
    import_factory("loops_cookbook.harbor:create_runtime")
except RuntimeConfigurationError as error:
    print("refused:", error)
"""

HELLO = ROOT / "patterns" / "custom_runtime" / "in_process_direct" / "train.toml"
TRAINS = f"""
import asyncio
from pathlib import Path
from baseline_model import BaselineTrainer, create_renderer
from loops_cookbook import command
from loops_cookbook.rl import cli
import loops_cookbook.rl.recipe as recipe_module
from loops_cookbook.rl.recipe import assemble

recipe_module.configured_renderer = lambda config: create_renderer()
assert command.main(["check", {str(HELLO)!r}]) == 0
config = cli.load_training_config(
    Path({str(HELLO)!r}),
    {{
        "rollout_runtime.files": {{"tasks": "tasks.jsonl"}},
        "compute.sampler_replica_affinity": True,
        "output.output_dir": "runs",
        "training.max_steps": 1,
    }},
)
trainer = BaselineTrainer()
run = assemble(config, trainer=trainer, renderer_factory=lambda _config: create_renderer())
outcome = asyncio.run(run.execute())
assert outcome.ok and outcome.steps == 1, outcome
print("trained", len(trainer.steps[0]["rewards"]), "rollouts")
"""

BLAXEL_PROVIDER = """
import importlib
for name in ("adapter", "api", "incidents", "settings", "usage"):
    importlib.import_module(f"loops_cookbook.providers.blaxel.{name}")
from loops_cookbook.providers.blaxel import create_adapter
print("provider ready")
"""

PROGRAMS = {
    "core": CORE_IMPORTS,
    "trains": TRAINS,
    "blaxel": BLAXEL_PROVIDER,
    "harbor": "import harbor",
    "harbor.trial.trial": "import harbor.trial.trial",
}


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """Every program's ``(returncode, stdout, stderr)``, run side by side: each pays for a
    fresh interpreter and import, so together they take about as long as one."""
    cwd = tmp_path_factory.mktemp("without_harbor")
    (cwd / "tasks.jsonl").write_text('{"name": "b", "instruction": "Say B", "answer": "B"}\n')
    started = {name: without_harbor(body, cwd) for name, body in PROGRAMS.items()}
    results = {}
    try:
        for name, process in started.items():
            out, err = process.communicate(timeout=120)
            results[name] = (process.returncode, out, err)
    finally:
        for process in started.values():
            if process.poll() is None:
                process.kill()
                process.wait()
    return results


def test_every_core_module_imports_and_harbor_asks_for_its_extra(ran):
    code, out, err = ran["core"]
    assert code == 0, err
    assert "baseten-loops-cookbook[harbor]" in out


def test_a_non_harbor_runtime_checks_and_trains(ran):
    code, out, err = ran["trains"]
    assert code == 0, err[-2000:]
    assert "trained 16 rollouts" in out


def test_the_native_blaxel_provider_needs_only_its_sdk(ran):
    """The ``blaxel`` extra is the SDK alone: the provider imports with Harbor absent."""
    pytest.importorskip("blaxel")
    code, out, err = ran["blaxel"]
    assert code == 0, err
    assert "provider ready" in out


@pytest.mark.parametrize("module", ["harbor", "harbor.trial.trial"])
def test_the_blocker_really_hides_harbor(ran, module):
    code, _, err = ran[module]
    assert code != 0 and "ModuleNotFoundError" in err
