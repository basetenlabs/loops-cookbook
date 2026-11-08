"""The Loops cookbook, which runs RL and SFT recipes on Baseten Loops.

Subpackages:
  loops_cookbook.runtime       The RolloutRuntime contract and the types it receives.
  loops_cookbook.episodes      The Episode class, which the cookbook runs as a custom rollout runtime.
  loops_cookbook.harbor        The Harbor rollout runtime, which runs Harbor jobs and agents.
  loops_cookbook.model_access  Attempt leases, token capture, and the OpenAI-compatible gateway.
  loops_cookbook.llm           The token-exact model client and the Prime renderer adapter.
  loops_cookbook.rl            The RL runner, GRPO, scheduling, and the Loops trainer transport.
  loops_cookbook.sft           Supervised fine-tuning on fixed data.
  loops_cookbook.resources     The provider-neutral resource registry and journal.
  loops_cookbook.providers     Resource provider adapters, such as Blaxel and Daytona.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("baseten-loops-cookbook")
except (
    PackageNotFoundError
):  # The package was imported from a source tree that was never installed.
    __version__ = "0+unknown"
