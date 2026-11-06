"""Files in the episode's sandbox, reached through its provider's usage environment.

``agent_program.py`` and ``verifier.py`` share this so either provider works unchanged:

- ``local_sandbox`` gives ``SANDBOX_DIR``, the sandbox's directory;
- ``[providers.blaxel]`` gives ``BLAXEL_SANDBOX_NAME`` and Blaxel's ``BL_*``
  credentials, and files live under ``/tmp`` in the sandbox. That path needs the
  Blaxel SDK in the program's interpreter.

Neither creates or deletes the sandbox: the run's registry owns it.
"""

import asyncio
import os
import pathlib

BLAXEL_DIR = "/tmp"


def write(name: str, text: str) -> None:
    if "BLAXEL_SANDBOX_NAME" in os.environ:
        asyncio.run(_blaxel(lambda sandbox: sandbox.fs.write(f"{BLAXEL_DIR}/{name}", text)))
    else:
        pathlib.Path(os.environ["SANDBOX_DIR"], name).write_text(text)


def read(name: str) -> str:
    """The file's text; ``FileNotFoundError`` when the read found no such file.

    That is not proof the file is missing: the whole sandbox may be gone, and
    neither Blaxel's 404 nor a missing path tells the two apart. So
    ``local_episode``, not the programs, asks whether the sandbox is still alive
    after the read.
    """
    if "BLAXEL_SANDBOX_NAME" in os.environ:
        return asyncio.run(_blaxel_read(f"{BLAXEL_DIR}/{name}"))
    return pathlib.Path(os.environ["SANDBOX_DIR"], name).read_text()


async def _blaxel_read(path: str) -> str:
    from blaxel.core.sandbox.types import ResponseError

    async def read(sandbox) -> str:
        # Every 404 from the read is a missing file, whether the file or the sandbox
        # is gone; the episode tells the two apart. A sandbox Blaxel no longer has
        # at all fails SandboxInstance.get, not the read, and raises as it is.
        try:
            return await sandbox.fs.read(path)
        except ResponseError as error:
            if error.response.status_code == 404:
                raise FileNotFoundError(path) from error
            raise

    return await _blaxel(read)


async def _blaxel(use):
    from blaxel.core import SandboxInstance

    return await use(await SandboxInstance.get(os.environ["BLAXEL_SANDBOX_NAME"]))
