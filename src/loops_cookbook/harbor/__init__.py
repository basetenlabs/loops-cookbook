"""The Harbor rollout runtime, which runs your Harbor job and agent as an RL run's rollouts.

Select it with ``[rollout_runtime].factory = "loops_cookbook.harbor:create_runtime"``.
Then name the Harbor job as the ``job`` alias under ``[rollout_runtime.files]``.
"""

try:
    import harbor  # noqa: F401 - Harbor is an optional extra
except ModuleNotFoundError as error:
    if error.name != "harbor":
        raise
    raise ModuleNotFoundError(
        "the Harbor rollout runtime needs Harbor. Install the extra "
        "baseten-loops-cookbook[harbor].",
        name="harbor",
    ) from error

from loops_cookbook.harbor.agent import LoopsAgent
from loops_cookbook.harbor.runtime import create_runtime

__all__ = ["LoopsAgent", "create_runtime"]
