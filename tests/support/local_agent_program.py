"""``agent_program.py`` for a local sandbox: the same model call, the answer written under
``SANDBOX_ROOT`` instead of through Blaxel."""

import os
import pathlib
import runpy
import sys

tool = runpy.run_path(sys.argv[1])  # the pattern's own program, for its model call
answer = tool["ask"](os.environ["AGENT_INSTRUCTION"])
target = pathlib.Path(os.environ["SANDBOX_ROOT"] + tool["ANSWER"])
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(answer)
print(f"wrote {len(answer)} characters to {tool['ANSWER']}")
