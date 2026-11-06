"""Keep development tooling outside the public package interface."""

import importlib.util

import pytest

from loops_cookbook.command import main


@pytest.mark.parametrize("command", ["audit", "smoke"])
def test_development_verbs_are_not_public_cli_commands(command):
    with pytest.raises(SystemExit) as error:
        main([command, "unused"])
    assert error.value.code == 2


def test_audit_is_not_in_the_public_package():
    assert importlib.util.find_spec("loops_cookbook.acceptance") is None
