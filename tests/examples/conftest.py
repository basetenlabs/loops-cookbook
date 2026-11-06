"""The custom-runtime examples' directories, importable as the CLI makes a config's.

The root conftest does the same for the Harbor examples.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for directory in (
    "examples/async_scheduling",
    "examples/counting",
    "examples/litellm_streaming",
    "examples/llm_judge",
    "examples/tool_agent",
):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))
