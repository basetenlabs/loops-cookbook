"""The Harbor runtime: your Harbor job and agent become the rollout runtime.

Select it with ``[rollout_runtime].factory = "loops_cookbook.harbor:create_runtime"`` and
name the Harbor job as the ``job`` alias under ``[rollout_runtime.files]``.
"""

try:
    import harbor  # noqa: F401 - Harbor is an optional extra
except ModuleNotFoundError as error:
    if error.name != "harbor":
        raise
    raise ModuleNotFoundError(
        "the Harbor runtime needs Harbor; install the extra: baseten-loops-cookbook[harbor]",
        name="harbor",
    ) from error

from loops_cookbook.harbor.agent import LoopsAgent
from loops_cookbook.harbor.runtime import create_runtime

__all__ = ["LoopsAgent", "create_runtime"]
