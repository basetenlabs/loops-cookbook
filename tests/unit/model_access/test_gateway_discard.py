"""The Chat Completions gateway's discard endpoint: a completion the agent drops does not
train.

Every test runs the real gateway on loopback, a real attempt lease, and a scripted
sampler; nothing reaches the network.
"""

import asyncio

import pytest
from support.fakes import ScriptedSampler
from support.leases import lease

from loops_cookbook.model_access.api import (
    ModelAccessError,
)
from loops_cookbook.model_access.openai.access import AccessEnded
from loops_cookbook.model_access.producers import ProducerSettings

from .rig import MODEL, _discard, attempt, client, connect, hello, raw


@pytest.mark.parametrize("discards", [True, False])
async def test_best_of_n_trains_only_the_kept_candidate_when_the_agent_discards_the_rest(
    gateway, discards
):
    """Without the discard call every delivered candidate trains, as a stock client's do."""
    discarded = []
    access, held, _ = attempt(gateway, discarded=discarded)
    openai_client, connection = await client(access)
    candidates = await asyncio.gather(
        *(
            openai_client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": "hi"}]
            )
            for _ in range(3)
        )
    )
    kept, *losers = candidates
    if discards:
        for loser in losers:
            status, body = await _discard(connection, loser.id)
            assert status == 200 and body["id"] == loser.id
    await access.quiesce(1.0)
    trained = [f"chatcmpl-{sample.sample_id}" for sample in (await held.seal()).samples]
    if discards:
        assert trained == [kept.id]
        assert sorted(f"chatcmpl-{s.sample_id}" for s in discarded) == sorted(c.id for c in losers)
    else:
        assert sorted(trained) == sorted(c.id for c in candidates) and discarded == []


async def test_a_discard_under_another_attempts_or_producers_credential_is_refused(gateway):
    both = ProducerSettings({"main": True, "critic": True})
    owner, held, _ = attempt(gateway, name="a1", producers=both)
    other, _, _ = attempt(gateway, name="a2", producers=both)
    as_main = await connect(owner, "main")
    as_critic = await connect(owner, "critic")
    as_other_attempt = await connect(other)
    _, body = await raw(as_main, body=hello())
    for foreign in (as_other_attempt, as_critic):
        status, refusal = await _discard(foreign, body["id"])
        assert status == 404 and refusal["error"]["code"] == "generation_not_found"
    await owner.quiesce(1.0)
    assert [f"chatcmpl-{s.sample_id}" for s in (await held.seal()).samples] == [body["id"]]


@pytest.mark.parametrize("completion_id", ["chatcmpl-unknown", "unknown"])
async def test_a_discard_of_an_unknown_id_is_refused_as_a_direct_discard_is(gateway, completion_id):
    direct = lease(ScriptedSampler(stop_reason="stop"))
    with pytest.raises(ModelAccessError):
        direct.discard("main", completion_id.removeprefix("chatcmpl-"))
    access, held, _ = attempt(gateway)
    connection = await connect(access)
    _, body = await raw(connection, body=hello())
    status, refusal = await _discard(connection, completion_id)
    assert status == 404 and refusal["error"]["code"] == "generation_not_found"
    assert held.failure is None
    await access.quiesce(1.0)
    assert [f"chatcmpl-{s.sample_id}" for s in (await held.seal()).samples] == [body["id"]]


async def test_only_the_latest_turn_of_a_conversation_is_discarded_and_only_once(gateway):
    access, held, _ = attempt(gateway)
    connection = await connect(access)
    headers = {"X-Loops-Conversation-Id": "session"}
    _, first = await raw(connection, body=hello(), headers=headers)
    history = [
        {"role": "user", "content": "hi"},
        first["choices"][0]["message"],
        {"role": "user", "content": "more"},
    ]
    _, second = await raw(connection, body=hello(messages=history), headers=headers)
    status, refusal = await _discard(connection, first["id"])  # already built on
    assert status == 409 and refusal["error"]["code"] == "generation_not_discardable"
    assert (await _discard(connection, second["id"]))[0] == 200
    status, refusal = await _discard(connection, second["id"])
    assert status == 409 and refusal["error"]["code"] == "generation_not_discardable"
    await access.quiesce(1.0)
    assert [f"chatcmpl-{s.sample_id}" for s in (await held.seal()).samples] == [first["id"]]


async def test_a_discard_after_sealing_is_refused_with_the_credential(gateway):
    access, held, _ = attempt(gateway)
    connection = await connect(access)
    _, body = await raw(connection, body=hello())
    await access.quiesce(1.0)
    with pytest.raises(AccessEnded):  # a handler that authenticated before revocation
        access.discard("main", body["id"])
    await held.seal()
    status, refusal = await _discard(connection, body["id"])
    assert status == 401 and refusal["error"]["code"] == "invalid_api_key"


async def test_a_discard_names_the_completion_id_not_the_bare_sample_id(gateway):
    access, held, _ = attempt(gateway)
    connection = await connect(access)
    _, body = await raw(connection, body=hello())
    status, refusal = await _discard(connection, body["id"].removeprefix("chatcmpl-"))
    assert status == 404 and refusal["error"]["code"] == "generation_not_found"
    await access.quiesce(1.0)
    assert len((await held.seal()).samples) == 1


@pytest.mark.parametrize("body", [{}, {"id": 3}, {"id": ""}, ["chatcmpl-x"]])
async def test_a_discard_without_a_completion_id_is_invalid(gateway, body):
    access, _, _ = attempt(gateway)
    connection = await connect(access)
    status, refusal = await raw(connection, "/loops/discard", body=body)
    assert status == 400 and refusal["error"]["param"] == "id"
