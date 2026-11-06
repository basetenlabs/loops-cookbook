"""Live tests against real providers: opt-in and billable.

``LOOPS_LIVE_BLAXEL=1`` runs the Blaxel tests; they need ``BL_WORKSPACE`` and
``BL_API_KEY`` (``uv run --env-file .env pytest tests/live``) and create small
sandboxes, released at the end. ``LOOPS_LIVE_LOOPS=1`` also runs the end-to-end test,
which trains one step on Loops and needs ``BASETEN_API_KEY`` as well.
The trainer normalization contract uses only a trainer and has a manual Actions
workflow; ordinary PR CI and releases skip it.
"""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker, variable in (
            ("live_blaxel", "LOOPS_LIVE_BLAXEL"),
            ("live_loops", "LOOPS_LIVE_LOOPS"),
        ):
            if item.get_closest_marker(marker) and os.environ.get(variable) != "1":
                item.add_marker(pytest.mark.skip(reason=f"set {variable}=1 to run live"))
