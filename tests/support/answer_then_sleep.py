"""A local agent program that answers, then outlives its agent deadline.

It runs ``local_agent_program.py`` with the same arguments, so the answer is in the
sandbox, then sleeps until the deadline kills it.
"""

import pathlib
import runpy
import sys
import time

here = pathlib.Path(__file__).parent
sys.argv = [str(here / "local_agent_program.py"), *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
time.sleep(60)
