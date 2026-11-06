"""The Blaxel SDK's view of one sandbox, as ``SandboxInstance.get`` returns it."""

from types import SimpleNamespace as NS

import httpx


def sdk_sandbox(name, labels, status):
    """A sandbox read with its own two connection pools, one for files, one for processes."""
    fs, process = httpx.AsyncClient(), httpx.AsyncClient()
    return NS(
        status=status,
        metadata=NS(to_dict=lambda: {"name": name, "labels": dict(labels)}),
        fs=NS(_client=fs, get_client=lambda: fs),
        process=NS(_client=process, get_client=lambda: process, get_api_client=NS, wait=None),
    )
