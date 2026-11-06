"""The custom-runtime patterns' directories, importable as the CLI makes a config's.

The root conftest does the same for the Harbor patterns. ``custom_runtime/local``
has an ``agent_program.py``, as ``harbor/local`` does; neither is imported here, only
run as a program with its own ``sys.path``.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for directory in (
    "patterns/custom_runtime/in_process_direct",
    "patterns/custom_runtime/in_process_openai_http",
    "patterns/custom_runtime/local",
    # The in-process twins lock the LiteLLM example's episode beside the patterns'.
    "examples/litellm_streaming",
):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))


@pytest.fixture
def sandboxes(tmp_path):
    """Where the custom local pattern's directory sandboxes live."""
    return tmp_path / "sandboxes"
