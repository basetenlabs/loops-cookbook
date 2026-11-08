"""The Harbor rollout runtime tests: the agents and environments Harbor imports by path."""

import sys
from pathlib import Path

import pytest

# ``harbor_test_agents``, ``harbor_test_environments`` and ``harbor_test_daytona`` are
# named by the tests' Harbor jobs, which Harbor imports by ``import_path``.
HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


@pytest.fixture(autouse=True)
def fresh_runs():
    """Every test starts with no recorded agent runs."""
    import harbor_test_agents

    harbor_test_agents.RUNS.clear()
