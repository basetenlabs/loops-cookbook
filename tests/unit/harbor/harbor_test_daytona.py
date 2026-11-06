"""The cookbook's Daytona environment class on a local directory, for Harbor runtime tests.

Harbor constructs it through ``import_path``, so the runtime treats the job as
Daytona-backed and the environment reads the attempt as the real one does; every
command runs locally.
"""

from baseline_environment import LocalEnvironment

from loops_cookbook.harbor.daytona import DaytonaEnvironment


class LocalDaytona(LocalEnvironment, DaytonaEnvironment):
    """A Daytona-backed environment whose sandbox is a local directory."""
