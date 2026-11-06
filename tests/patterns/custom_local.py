"""What the custom local pattern's tests share: its directory, tasks and sandbox."""

from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2] / "patterns" / "custom_runtime" / "local"
TASKS = [
    {"name": "b", "instruction": "Say B", "answer": "B"},
    {"name": "c", "instruction": "Say C", "answer": "C"},
]
SANDBOX = {"provider": "local_sandbox"}
DEADLINES = {
    f"{operation}_timeout_seconds": 10
    for operation in ("create", "reconcile", "release", "confirm_absent", "list", "attach")
}
