"""A sample keeps its completion and behavior logprobs packed, and reads as the tuples."""

import copy
import gc
import json
import math
import pickle
import struct
import tracemalloc
from dataclasses import fields, replace
from hashlib import sha256

import pytest
from support.builders import make_rollout, make_sample
from support.fakes import ScriptedSampler
from support.leases import direct_llm
from support.unit_rest import ChainRenderer, captured_chain

from loops_cookbook.evidence import Logprobs, Sample, TokenSpan, TrajectoryError
from loops_cookbook.rl.grpo import build_rows
from loops_cookbook.rl.metrics import trajectory_metrics
from loops_cookbook.rl.run import sample_record

# Values a sampler can return whose bits a careless conversion would change.
ODD = (-0.0, -5e-324, -2.2250738585072014e-308, -1e-300, -0.1, -1 / 3, -88.72283935546875)


def _bits(values):
    return [struct.pack("<d", value) for value in values]


# --- Logprobs reads as the tuple of its values ------------------------------------------


def test_logprobs_compare_hash_and_print_as_their_tuple():
    values = (-0.5, -1.25, -0.0)
    packed = Logprobs(values)
    assert packed == values and values == packed and packed == Logprobs(list(values))
    assert packed != (-0.5, -1.25) and packed != (-0.5, -1.25, -1.0)
    assert not (packed != values)  # noqa: SIM202 -- __ne__ itself
    # Like a tuple: never equal to a list, equal to a tuple of equal numbers.
    assert packed != list(values) and list(values) != packed
    assert Logprobs((-1.0, 0.0)) == (-1, 0) and Logprobs((1.0,)) == (True,)
    assert Logprobs((-1.0,)) != ("-1.0",)
    assert hash(packed) == hash(values) and {packed: 1}[values] == 1
    assert repr(packed) == repr(values) and str(packed) == str(values)
    assert list(packed) == list(values) and len(packed) == 3 and bool(packed)
    assert not Logprobs(()) and Logprobs(()) == ()


def test_logprobs_index_slice_and_concatenate_as_a_tuple_does():
    values = (-0.5, -1.25, -2.0, -3.5)
    packed = Logprobs(values)
    for index in range(-4, 4):
        assert packed[index] == values[index] and type(packed[index]) is float
    for bad in (4, -5):
        with pytest.raises(IndexError):
            packed[bad]
    for part in (slice(1, 3), slice(None, None, 2), slice(3, 1), slice(-2, None)):
        assert packed[part] == values[part] and type(packed[part]) is tuple
    assert packed + (-9.0,) == values + (-9.0,) and (-9.0,) + packed == (-9.0,) + values
    assert packed + packed == values + values
    with pytest.raises(TypeError):
        packed + [-9.0]  # a tuple refuses a list too


def test_logprobs_keep_every_bit_and_copy_and_pickle_as_the_same_values():
    packed = Logprobs(ODD)
    assert _bits(packed) == _bits(ODD) and math.copysign(1, packed[0]) == -1
    assert copy.copy(packed) is packed and copy.deepcopy(packed) is packed
    restored = pickle.loads(pickle.dumps(packed))
    assert type(restored) is Logprobs and _bits(restored) == _bits(ODD)
    assert packed.tolist() == list(ODD) and _bits(packed.tolist()) == _bits(ODD)


@pytest.mark.parametrize("bad", [(-1,), (True,), (-0.5, -1), ("-0.5",)])
def test_only_exact_floats_become_logprobs(bad):
    with pytest.raises(ValueError, match="floats"):
        Logprobs(bad)


# --- a sample packs what it can and validates exactly as before -------------------------


def test_a_sample_packs_its_completion_and_logprobs():
    sample = make_sample("s", (1, 2), [5, 6, 7], [-0.5, -1.25, -0.0])
    assert type(sample.completion_ids) is TokenSpan and sample.completion_ids == (5, 6, 7)
    assert type(sample.logprobs) is Logprobs and sample.logprobs == (-0.5, -1.25, -0.0)
    again = replace(sample, sample_id="t")
    assert again.completion_ids is sample.completion_ids and again.logprobs is sample.logprobs


def test_what_does_not_pack_stays_the_tuple_it_was_and_is_judged_as_before():
    # A token too large for a buffer was accepted as a tuple, and still is.
    large = make_sample("s", (1,), (2**40,), (-0.5,))
    assert type(large.completion_ids) is tuple and large.completion_ids == (2**40,)
    # Logprobs that are not all floats -- an int a test writes -- stay a tuple.
    ints = make_sample("s", (1,), (2, 3), (0, -1.5))
    assert type(ints.logprobs) is tuple and ints.logprobs == (0, -1.5)
    for completion in [(-1,), (True,), (1.0,)]:
        with pytest.raises(TrajectoryError, match="nonnegative integers"):
            make_sample("s", (1,), completion, (-0.5,))
    for logprobs in [(math.nan,), (-math.inf,), (-0.5, math.inf)]:
        with pytest.raises(TrajectoryError, match="finite"):
            make_sample("s", (1,), (2,) * len(logprobs), logprobs)
    with pytest.raises(TrajectoryError, match="align"):
        make_sample("s", (1,), (2, 3), (-0.5,))
    with pytest.raises(TrajectoryError, match="prompt and completion"):
        make_sample("s", (1,), (), ())


# --- captured evidence: the sampler's values, and everything read from them -------------


async def test_a_captured_completion_is_the_samplers_values_exactly():
    tokens = [7, 131_071, 0, 4_000_000_000, 12, 13, 14]
    llm = direct_llm(ScriptedSampler(tokens=tokens, logprobs=ODD), ChainRenderer())
    response = await llm.conversation().generate(
        messages=[{"role": "user", "content": "x"}], max_tokens=7
    )
    sample = response.sample
    assert type(sample.completion_ids) is TokenSpan and type(sample.logprobs) is Logprobs
    assert list(sample.completion_ids) == tokens
    assert all(type(token) is int for token in sample.completion_ids)
    assert _bits(sample.logprobs) == _bits(ODD)


def _as_tuples(sample):
    """The sample exactly as it was captured before packing: tuples of Python numbers."""
    twin = object.__new__(Sample)
    for field in fields(Sample):
        object.__setattr__(twin, field.name, getattr(sample, field.name))
    object.__setattr__(twin, "completion_ids", tuple(sample.completion_ids))
    object.__setattr__(twin, "logprobs", tuple(sample.logprobs))
    return twin


async def test_everything_read_from_packed_evidence_is_what_tuples_gave():
    llm, samples, _ = await captured_chain()
    kept = list(llm.samples)
    assert all(type(sample.completion_ids) is TokenSpan for sample in samples)
    assert all(type(sample.logprobs) is Logprobs for sample in samples)
    rollout = make_rollout("task", "group", "r", kept, 1.0)
    reference = make_rollout("task", "group", "r", [_as_tuples(s) for s in kept], 1.0)
    assert all(type(s.completion_ids) is tuple for s in reference.samples)
    assert rollout == reference

    def chains(of):
        return [[sample.sample_id for sample in chain] for chain in of.chains]

    assert chains(rollout) == chains(reference) and len(chains(rollout)) == 2

    def datums(of):
        return json.dumps(
            [row.datum.model_dump(mode="json") for row in build_rows(of, 0.5)], allow_nan=False
        )

    assert datums(rollout) == datums(reference)

    def records(of):
        return "".join(
            json.dumps(sample_record(sample, tokens=tokens), allow_nan=False) + "\n"
            for sample in of.samples
            for tokens in (True, False)
        )

    assert (
        sha256(records(rollout).encode()).digest() == sha256(records(reference).encode()).digest()
    )
    first = json.loads(records(rollout).splitlines()[0])
    assert first["completion_ids"] == [41, 51] and first["logprobs"] == [-0.1, -0.2]
    assert trajectory_metrics([rollout]) == trajectory_metrics([reference])


# --- what a graded rollout waiting to train holds ---------------------------------------


def test_a_completion_and_its_logprobs_hold_twelve_bytes_a_token():
    """Graded rollouts wait for training holding their completions: up to a thousand of
    them at 50k completion tokens each. As tuples of Python numbers that was 68 bytes a
    token (a pointer and an int, a pointer and a float); packed it is 4 + 8."""
    tokens = [1_000 + index % 150_000 for index in range(50_000)]
    logprobs = [-index / 50_000 - 1e-9 for index in range(50_000)]
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        sample = make_sample("s", (1,), tokens, logprobs)
        gc.collect()
        held = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    assert sample.completion_ids == tuple(tokens) and sample.logprobs == tuple(logprobs)
    assert held < 13 * len(tokens), f"held {held / len(tokens):.1f} bytes a token"
