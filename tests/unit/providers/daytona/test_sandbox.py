"""The SDK upload overload must receive text bytes, never a local filename."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("daytona")

from daytona._async.filesystem import AsyncFileSystem

from loops_cookbook.providers.daytona import Daytona


@pytest.mark.parametrize("content", ["answer.txt", "雪\nπ", ""])
async def test_write_file_uses_the_sdk_bytes_overload_and_propagates_failures(content):
    fs = AsyncMock(spec=AsyncFileSystem)
    session = SimpleNamespace(client=AsyncMock(return_value=SimpleNamespace(fs=fs)))
    sandbox = Daytona("box", {}, session)
    await sandbox.write_file("/tmp/answer.txt", content)
    fs.upload_file.assert_awaited_once_with(content.encode("utf-8"), "/tmp/answer.txt")
    fs.upload_file.side_effect = ConnectionError("sandbox unavailable")
    with pytest.raises(ConnectionError, match="sandbox unavailable"):
        await sandbox.write_file("/tmp/answer.txt", content)
