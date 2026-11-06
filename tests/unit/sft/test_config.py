"""SFT config validation: what a config must and must not say."""

import re

import pytest
from pydantic import ValidationError
from support.configs import SFT_MINIMAL, merged
from support.rl_sft import chat_row

from loops_cookbook.sft.config import Config


def test_one_validation_source_at_most(sft_config):
    with pytest.raises(ValidationError, match="not both"):
        sft_config(validation=[chat_row(("q", "a"))], data={"validation_fraction": 0.1})


def test_an_optimizer_to_start_from_needs_the_weights_it_comes_with(sft_config):
    with pytest.raises(ValidationError, match="set init_from"):
        sft_config(training={"init_from_optimizer": True})


def test_sft_requires_a_sequence_budget(sft_config):
    with pytest.raises(ValidationError, match=r"model\.max_seq_len"):
        sft_config(model={"max_seq_len": None})


def _leaves(table, prefix=""):
    for key, value in table.items():
        if isinstance(value, dict):
            yield from _leaves(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}"


@pytest.mark.parametrize("path", sorted(set(_leaves(SFT_MINIMAL)) - {"recipe"}))
def test_every_required_setting_is_refused_when_missing(path):
    *sections, leaf = path.split(".")
    values = merged(SFT_MINIMAL)
    table = values
    for section in sections:
        table = table[section]
    del table[leaf]
    with pytest.raises(ValidationError, match=rf"{re.escape(path)}\n  Field required"):
        Config.model_validate(values)
