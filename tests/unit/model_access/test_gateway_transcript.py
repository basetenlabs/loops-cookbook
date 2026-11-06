"""The Chat Completions gateway's transcript: each request as the agent sent it and the
body it got back.

Every test runs the real gateway on loopback, a real attempt lease, and a scripted
sampler; nothing reaches the network.
"""

import base64

import pytest
from support.fakes import EchoSampler, FakePrime

from loops_cookbook.evidence import StaleRolloutError
from loops_cookbook.llm.client import SamplingPolicy
from loops_cookbook.llm.prime import PrimeRenderer
from loops_cookbook.llm.rendering import ParsedReply, RendererFailure
from loops_cookbook.llm.transcript import (
    CompletionTurnReply,
    ParsedTurnReply,
    TurnDisposition,
    thaw,
)
from loops_cookbook.model_access.producers import ProducerSettings

from .rig import (
    PNG,
    Renderer,
    _discard,
    _hang_up_mid_stream,
    _huge_reply,
    _sealed,
    attempt,
    connect,
    hello,
    raw,
)


def _key_order(value):
    """``value``'s object keys, in order, all the way down."""
    if isinstance(value, dict):
        return [(key, _key_order(item)) for key, item in value.items()]
    if isinstance(value, list):
        return [_key_order(item) for item in value]
    return None


async def test_the_transcript_keeps_each_request_as_the_agent_sent_it_and_the_body_it_got(
    gateway,
):
    """The gateway renders the native message for an echo; the judge sees the echo."""
    renderer = Renderer()
    renderer.reply = lambda n: ParsedReply(
        f"answer {n}", "thinking", (), True, {"role": "assistant", "content": f"native {n}"}
    )
    access, held, _ = attempt(gateway, renderer=renderer, transcripts=[])
    connection = await connect(access)
    headers = {"X-Loops-Conversation-Id": "session"}
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    start = [{"role": "user", "content": "start"}]
    _, first = await raw(connection, body=hello(messages=start, tools=tools), headers=headers)
    echo = {"role": "assistant", "content": first["choices"][0]["message"]["content"]}
    history = [*start, echo, {"role": "user", "content": "more"}]
    _, second = await raw(connection, body=hello(messages=history, tools=tools), headers=headers)
    await access.quiesce(1.0)

    # The model saw the native message: the second turn bridged from the first's sample.
    assert len(renderer.renders) == 1 and renderer.bridges == [[history[-1]]]
    transcript, (one, two) = await _sealed(held)
    assert thaw(one.messages) == start and one.extends is None
    assert two.extends == one.sample_id and thaw(two.messages) == [echo, history[-1]]
    assert thaw(transcript.request(two.sample_id)) == history
    assert isinstance(one.reply, CompletionTurnReply) and thaw(one.reply.body) == first
    assert isinstance(two.reply, CompletionTurnReply) and thaw(two.reply.body) == second
    assert [thaw(transcript.tools[digest]) for digest in two.tools] == tools
    assert [sealed.disposition for sealed in transcript.turns] == [TurnDisposition.DELIVERED] * 2


async def test_the_transcript_keeps_the_keys_in_the_order_the_agent_sent_them(gateway):
    access, held, _ = attempt(gateway, transcripts=[])
    connection = await connect(access)
    messages = [{"role": "user", "content": "hi"}]  # the schema declares content first
    tools = [{"function": {"parameters": {"type": "object"}, "name": "lookup"}, "type": "function"}]
    status, _ = await raw(connection, body=hello(messages=messages, tools=tools))
    assert status == 200
    await access.quiesce(1.0)
    transcript, (turn,) = await _sealed(held)
    assert _key_order(thaw(transcript.request(turn.sample_id))) == _key_order(messages)
    assert _key_order([thaw(transcript.tools[digest]) for digest in turn.tools]) == _key_order(
        tools
    )


async def test_an_http_history_that_is_not_a_prefix_is_recorded_whole(gateway):
    access, held, _ = attempt(gateway, transcripts=[])
    connection = await connect(access)
    headers = {"X-Loops-Conversation-Id": "session"}
    await raw(connection, body=hello(), headers=headers)
    rewritten = [{"role": "user", "content": "hi, again"}]
    await raw(connection, body=hello(messages=rewritten), headers=headers)
    await access.quiesce(1.0)
    _, (_, second) = await _sealed(held)
    assert second.extends is None and thaw(second.messages) == rewritten


async def test_without_the_header_a_request_extends_its_producers_latest_prefix(gateway):
    """A stock client names no conversation; the delta is found by exact prefix."""
    access, held, _ = attempt(gateway, transcripts=[])
    connection = await connect(access)
    start = [{"role": "user", "content": "start"}]
    _, first = await raw(connection, body=hello(messages=start))
    echo = {"role": "assistant", "content": first["choices"][0]["message"]["content"]}
    await raw(connection, body=hello(messages=[{"role": "user", "content": "unrelated"}]))
    grown = [*start, echo, {"role": "user", "content": "more"}]
    await raw(connection, body=hello(messages=grown))
    edited = [{**start[0], "content": "start!"}, echo]
    await raw(connection, body=hello(messages=edited))
    await access.quiesce(1.0)
    _, (one, _unrelated, three, four) = await _sealed(held)
    assert three.extends == one.sample_id and thaw(three.messages) == grown[1:]
    assert four.extends is None and thaw(four.messages) == edited


async def test_without_the_header_a_request_never_extends_another_producers_turn(gateway):
    producers = ProducerSettings({"main": True, "critic": False})
    access, held, _ = attempt(gateway, producers=producers, transcripts=[])
    main = await connect(access)
    critic = await connect(access, "critic")
    await raw(main, body=hello())
    grown = [{"role": "user", "content": "hi"}, {"role": "user", "content": "more"}]
    await raw(critic, body=hello(messages=grown))
    await access.quiesce(1.0)
    _, turns = await _sealed(held)
    assert [(turn.producer, turn.extends) for turn in turns] == [("main", None), ("critic", None)]


async def test_a_named_conversation_never_extends_another_conversations_turn(gateway):
    access, held, _ = attempt(gateway, transcripts=[])
    connection = await connect(access)
    await raw(connection, body=hello(), headers={"X-Loops-Conversation-Id": "a"})
    grown = [{"role": "user", "content": "hi"}, {"role": "user", "content": "more"}]
    await raw(connection, body=hello(messages=grown), headers={"X-Loops-Conversation-Id": "b"})
    await access.quiesce(1.0)
    _, turns = await _sealed(held)
    assert [turn.extends for turn in turns] == [None, None]


async def test_an_http_image_is_the_sampled_bytes_stored_once_in_the_agents_part(gateway):
    access, held, _ = attempt(
        gateway, sampler=EchoSampler(), renderer=PrimeRenderer(FakePrime()), transcripts=[]
    )
    connection = await connect(access)
    uri = "data:image/png;base64," + base64.b64encode(PNG).decode()
    part = {"type": "image_url", "image_url": {"url": uri, "detail": "auto"}}
    for _ in range(2):
        status, _ = await raw(
            connection, body=hello(messages=[{"role": "user", "content": [part]}])
        )
        assert status == 200
    await access.quiesce(1.0)
    transcript, turns = await _sealed(held)
    (digest,) = transcript.images
    assert transcript.images[digest].data == PNG
    first, again = turns
    assert thaw(first.messages) == [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": digest, "detail": "auto"}}],
        }
    ]
    # The identical request, header-less, extends the first by nothing.
    assert again.extends == first.sample_id and again.messages == ()


async def test_a_turn_whose_delivery_failed_mid_write_is_marked_undelivered(gateway):
    """The turn was written before the gateway knew; only the seal can say it failed."""
    renderer = Renderer(reply=_huge_reply)
    access, held, _ = attempt(gateway, renderer=renderer, transcripts=[])
    connection = await connect(access)
    await _hang_up_mid_stream(connection)
    await access.quiesce(1.0)
    sealed = await held.seal()
    assert sealed.samples == () and sealed.transcript is not None
    (turn,) = sealed.transcript.turns
    assert turn.disposition is TurnDisposition.UNDELIVERED and not turn.trainable
    assert isinstance(turn.turn.reply, CompletionTurnReply)


@pytest.mark.parametrize("recording", [True, False])
@pytest.mark.parametrize("unread", ["raises", "no_native_message"])
async def test_a_completion_the_renderer_cannot_read_is_a_500_that_fails_the_attempt(
    gateway, unread, recording
):
    """Recorded, the turn says what the generation raised; the outcome is recording-off's."""
    renderer = Renderer()

    def broken(_n):
        raise KeyError("renderer bug")

    renderer.reply = (
        broken if unread == "raises" else (lambda _n: ParsedReply("partial", None, (), True, None))
    )
    access, held, sampler = attempt(
        gateway, renderer=renderer, transcripts=[] if recording else None
    )
    connection = await connect(access)
    status, body = await raw(connection, body=hello())
    assert status == 500 and body["error"]["type"] == "server_error"
    status, _ = await raw(connection, body=hello())  # what a stock client's retry sends
    assert status == 500 and len(sampler.prompts) == 1  # the attempt samples nothing more
    if recording:
        assert held.transcript is not None
        turn, retry = held.transcript.recorded().turns
        assert isinstance(turn.reply, ParsedTurnReply) and turn.reply.parsed is None
        assert (turn.reply.parse_error or "").startswith("RendererFailure: the renderer")
        # The retry was sent and refused before sampling: a request nothing answered.
        assert retry.sample_id is None and retry.reply is None
    await access.quiesce(1.0)
    with pytest.raises(RendererFailure):
        await held.seal()


async def test_a_generation_found_stale_once_sampled_is_named_by_its_turn(gateway):
    """Its sample row is written before the check; the turn joins it, as undelivered."""
    recorded, transcripts = [], []
    access, held, _ = attempt(
        gateway,
        transcripts=transcripts,
        recorded=recorded,
        policy_version=lambda: 1,  # the sampler serves version 0
        max_steps_off_policy=0,
    )
    connection = await connect(access)
    status, _ = await raw(connection, body=hello())
    assert status >= 400
    await access.quiesce(1.0)
    with pytest.raises(StaleRolloutError):
        await held.seal()
    (sample,) = recorded
    (transcript,) = transcripts
    (turn,) = transcript.turns
    assert turn.turn.sample_id == sample.sample_id
    assert turn.disposition is TurnDisposition.UNDELIVERED and not transcript.sealed
    assert isinstance(turn.turn.reply, ParsedTurnReply)
    assert (turn.turn.reply.parse_error or "").startswith("StaleRolloutError")


async def test_a_request_refused_before_sampling_leaves_no_turn(gateway):
    """As over direct access: an agent retrying a request too long for the trajectory
    budget stores nothing, rather than its whole history once a retry."""
    access, held, _ = attempt(
        gateway,
        transcripts=[],
        sampling_policy=SamplingPolicy(max_generation_tokens=4, max_trajectory_tokens=2),
    )
    connection = await connect(access)
    for _ in range(3):
        status, body = await raw(connection, body=hello())
        assert status == 400 and body["error"]["code"] == "context_length_exceeded"
    assert held.transcript is not None and held.transcript.recorded().turns == ()


async def test_a_transcript_that_cannot_be_written_never_fails_the_seal(gateway, caplog):
    """How the attempt is recorded does not decide what it trains or reports dropped."""

    class FullDisk(list):
        def append(self, _transcript):
            raise OSError("no space left on device")

    renderer = Renderer(reply=_huge_reply)
    dropped = []
    access, held, _ = attempt(gateway, renderer=renderer, transcripts=FullDisk(), discarded=dropped)
    connection = await connect(access)
    await _hang_up_mid_stream(connection)
    await access.quiesce(1.0)
    sealed = await held.seal()
    assert sealed.samples == () and len(dropped) == 1
    assert "transcript was not written" in caplog.text


async def test_a_completion_the_agent_discarded_is_marked_discarded(gateway):
    access, held, _ = attempt(gateway, transcripts=[])
    connection = await connect(access)
    _, loser = await raw(connection, body=hello())
    _, kept = await raw(connection, body=hello())
    await _discard(connection, loser["id"])
    await access.quiesce(1.0)
    transcript, _ = await _sealed(held)
    marks = {
        f"chatcmpl-{sealed.turn.sample_id}": (sealed.disposition, sealed.trainable)
        for sealed in transcript.turns
    }
    assert marks == {
        loser["id"]: (TurnDisposition.DISCARDED, False),
        kept["id"]: (TurnDisposition.DELIVERED, True),
    }


async def test_a_keyed_retry_is_one_generation_and_one_turn(gateway):
    access, held, sampler = attempt(gateway, transcripts=[])
    connection = await connect(access)
    for _ in range(2):
        await raw(connection, body=hello(), headers={"Idempotency-Key": "k1"})
    await access.quiesce(1.0)
    _, turns = await _sealed(held)
    assert len(sampler.prompts) == 1 and len(turns) == 1


async def test_without_recording_the_gateway_records_nothing(gateway):
    access, held, _ = attempt(gateway)
    connection = await connect(access)
    status, _ = await raw(connection, body=hello())
    await access.quiesce(1.0)
    sealed = await held.seal()
    assert status == 200 and held.transcript is None
    assert len(sealed.samples) == 1 and sealed.transcript is None
