"""The shared answer check the sandbox episodes use: ``Blaxel.read_file`` and
``SandboxKind.ensure_alive``, through the real SDK and the cookbook's Blaxel adapter."""

import pytest

from loops_cookbook.resources import SandboxGone

PATH = "/tmp/answer.txt"


@pytest.mark.parametrize("content", ["B", "雪\nπ", ""])
async def test_a_file_written_to_the_sandbox_is_read_as_its_text(blaxel_sandbox, content):
    on = await blaxel_sandbox()
    await on.write_file(PATH, content)
    assert await on.read_file(PATH) == content


async def test_a_file_missing_from_a_live_sandbox_is_none(blaxel_sandbox):
    on = await blaxel_sandbox()
    assert await on.read_file(PATH) is None
    await on.ensure_alive()


@pytest.mark.parametrize("how", ["deleted", "TERMINATED", "DELETING", "FAILED", "DEPLOYING"])
async def test_a_404_from_a_sandbox_not_alive_raises_sandbox_gone(blaxel, blaxel_sandbox, how):
    on = await blaxel_sandbox()
    await on.client()  # attached while alive, as an episode's setup() attaches
    blaxel.lose(on.name, how)  # its file reads answer 404 all the same
    with pytest.raises(SandboxGone, match=f"sandbox {on.name} is gone"):
        await on.read_file(PATH)
    with pytest.raises(SandboxGone, match=on.name):
        await on.ensure_alive()


async def test_any_other_failed_read_raises_as_the_sdk_raised_it(blaxel_sandbox):
    from blaxel.core.sandbox.types import ResponseError

    on = await blaxel_sandbox()
    (await on.client()).failing = 503
    with pytest.raises(ResponseError):
        await on.read_file(PATH)
