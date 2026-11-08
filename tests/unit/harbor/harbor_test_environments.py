"""The cookbook's Blaxel environment class on a local directory, for Harbor rollout runtime tests.

Harbor constructs it through ``import_path``, so the rollout runtime treats the job as
Blaxel-backed; every command runs locally, and the sandbox kernel's /proc/vmstat is
whatever the variant says it is.
"""

from baseline_environment import LocalEnvironment
from harbor.environments.base import ExecResult

from loops_cookbook.harbor.blaxel import BlaxelEnvironment


class OomKilledSandbox(LocalEnvironment, BlaxelEnvironment):
    """A sandbox whose kernel has OOM-killed ``KILLS`` processes."""

    KILLS = 2

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None) -> ExecResult:
        if "/proc/vmstat" in command:
            vmstat = f"nr_free_pages 1\noom_kill {self.KILLS}\n"
            return ExecResult(stdout=vmstat, stderr="", return_code=0)
        return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)


class CleanSandbox(OomKilledSandbox):
    """A sandbox whose kernel has OOM-killed nothing."""

    KILLS = 0


class UnreadableSandbox(LocalEnvironment, BlaxelEnvironment):
    """A sandbox whose OOM count cannot be read: the read itself fails."""

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None) -> ExecResult:
        if "/proc/vmstat" in command:
            raise RuntimeError("the read failed")
        return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)
