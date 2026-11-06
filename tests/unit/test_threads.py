import os
import sys
import threading
import time

import pytest

from loops_cookbook import threads
from loops_cookbook.cli import run_main


def test_run_main_returns_only_once_the_cookbooks_threads_have_finished():
    """A write still in a cookbook thread when the run ends is finished before
    run_main returns, as the runner finishes the default executor's work, so a
    process that exits then never leaves half a file."""
    finished = threading.Event()

    def write() -> None:
        time.sleep(0.3)
        finished.set()

    async def main() -> None:
        threads.in_thread(write)

    run_main(main())
    assert finished.is_set()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_a_forked_child_gets_threads_of_its_own():
    """A process forked after the cookbook's threads started inherits none of
    them; its own work must still run rather than wait on threads that are gone."""
    assert threads.executor().submit(lambda: 1).result(timeout=5) == 1
    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
        try:
            ok = threads.executor().submit(lambda: 42).result(timeout=5) == 42
        except TimeoutError:
            ok = False
        sys.stdout.flush()
        os._exit(0 if ok else 1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
