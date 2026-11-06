"""Loops cookbook: RL recipes on Baseten Loops.

Subpackages:
  loops_cookbook.runtime       The RolloutRuntime contract and the types it receives.
  loops_cookbook.episodes      The Episode class, run by the cookbook as a runtime.
  loops_cookbook.harbor        Harbor jobs and agents as a runtime.
  loops_cookbook.model_access  Attempt leases, capture, and the OpenAI-compatible gateway.
  loops_cookbook.llm           Token-exact model client and Prime renderer adapter.
  loops_cookbook.rl            Training runner, GRPO, scheduling, Loops trainer transport.
  loops_cookbook.resources     Provider-neutral resource registry and journal.
  loops_cookbook.providers     Resource provider adapters, such as Blaxel.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("baseten-loops-cookbook")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0+unknown"
