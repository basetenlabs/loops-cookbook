"""The transcript direct model access records, and what sealing marks on it.

Every test captures through a real attempt lease over a scripted sampler, and reads
the transcript the lease's seal returns.
"""

import asyncio
import base64
import copy
import gc
import hashlib
import json
import math
import pickle
import tracemalloc
from enum import StrEnum

import pytest
from baseten.loops.models import ImageChunk
from PIL import Image
from renderers.base import ParsedToolCall, ToolCallParseStatus
from support.builders import make_sample
from support.fakes import EchoSampler, FakePrime, ScriptedRenderer, ScriptedSampler
from support.images import png
from support.leases import lease

from loops_cookbook.evidence import StaleRolloutError
from loops_cookbook.llm.client import Renderers
from loops_cookbook.llm.images import image_parts
from loops_cookbook.llm.prime import PrimeRenderer
from loops_cookbook.llm.rendering import ParsedReply, RendererFailure, RenderingError
from loops_cookbook.llm.transcript import (
    FrozenMap,
    ParsedTurnReply,
    TranscriptRecorder,
    TurnDisposition,
    seal_transcript,
    thaw,
)
from loops_cookbook.model_access.api import CaptureIntegrityFailure
from loops_cookbook.model_access.lease import DirectModelAccess
from loops_cookbook.model_access.producers import ProducerSettings

PNG = png((2, 2))
LOOKUP = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
SEARCH = {"type": "function", "function": {"name": "search"}}


def access(renderer=None, *, sampler=None, producers=None, recording=True, **options):
    """Direct model access over a fresh lease that records transcripts, or not."""
    held = lease(
        sampler or ScriptedSampler(stop_reason="stop"),
        producers=producers,
        transcripts=[] if recording else None,
        **options,
    )
    return DirectModelAccess(held, Renderers(renderer or ScriptedRenderer())), held


def user(text):
    return {"role": "user", "content": text}


async def sealed_transcript(held):
    sealed = await held.seal()
    assert sealed.transcript is not None
    return sealed.transcript


async def test_each_turn_records_only_what_its_request_adds_to_the_last():
    model, held = access()
    chat = model.llm().conversation()
    first = [user("start")]
    one = await chat.generate(messages=first, tools=[LOOKUP])
    second = [*first, one.native_message, user("more")]
    two = await chat.generate(messages=second, tools=[LOOKUP])
    third = [*second, two.native_message, user("last")]
    await chat.generate(messages=third, tools=[LOOKUP])

    transcript = await sealed_transcript(held)
    turns = [sealed.turn for sealed in transcript.turns]
    assert [turn.extends for turn in turns] == [None, turns[0].sample_id, turns[1].sample_id]
    assert thaw(turns[0].messages) == [user("start")]
    assert thaw(turns[1].messages) == [{"role": "assistant", "content": "answer 1"}, user("more")]
    assert thaw(turns[2].messages) == [{"role": "assistant", "content": "answer 2"}, user("last")]
    # The deltas rebuild each request exactly as the agent sent it.
    assert [thaw(transcript.request(turn.sample_id)) for turn in turns] == [first, second, third]
    reply = turns[2].reply
    assert isinstance(reply, ParsedTurnReply) and reply.parsed is not None
    assert thaw(reply.parsed) == {
        "content": "answer 3",
        "reasoning_content": None,
        "reasoning_complete": True,
        "tool_calls": [],
        "native_message": {"role": "assistant", "content": "answer 3"},
    }


async def test_a_history_that_is_not_a_prefix_of_the_request_is_recorded_whole():
    """An agent that rewrites its history (a summary, a trimmed context) is not a delta."""
    model, held = access()
    chat = model.llm().conversation()
    one = await chat.generate(messages=[user("start")])
    rewritten = [user("a summary of the start"), one.native_message, user("more")]
    await chat.generate(messages=rewritten)

    _, second = (sealed.turn for sealed in (await sealed_transcript(held)).turns)
    assert second.extends is None and thaw(second.messages) == rewritten


async def test_a_rebuilt_request_keeps_the_keys_in_the_order_the_agent_sent_them():
    model, held = access()
    chat = model.llm().conversation()
    first = [{"role": "user", "content": "hi"}]
    one = await chat.generate(messages=first)
    second = [{"content": "hi", "role": "user"}, one.native_message, user("more")]
    await chat.generate(messages=second)

    transcript = await sealed_transcript(held)
    rebuilt = [thaw(transcript.request(sealed.turn.sample_id)) for sealed in transcript.turns]
    assert [[list(message) for message in request] for request in rebuilt] == [
        [list(message) for message in request] for request in (first, second)
    ]


def _held_after(turns: int) -> int:
    """Bytes a recorder still holds after ``turns`` turns of one growing conversation."""
    recorder, history = TranscriptRecorder(), []
    tracemalloc.start()
    try:
        for n in range(turns):
            history.append(user(f"observation {n}: " + "x" * 4000))
            pending = recorder.begin(
                history, [], rendered=history, named=True, producer="main", conversation_id="main"
            )
            pending.parsed(make_sample(f"s{n}", (1,), (2,), (-0.1,)), None, ValueError("cut"))
        turns_recorded = recorder.recorded().turns
        assert len(turns_recorded) == turns and not any(t.unrecordable for t in turns_recorded)
        gc.collect()
        return tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()


def test_what_recording_holds_grows_with_the_conversation_not_its_square():
    """Each turn extends the last, so the history it shares is held once, not per turn."""
    shorter, longer = _held_after(40), _held_after(80)
    assert longer < 2.5 * shorter, (shorter, longer)  # the square would be about 4x


class Tone(StrEnum):
    CALM = "calm"


@pytest.mark.parametrize(
    ("sent", "resent", "extends"),
    [
        (1, True, False),
        (True, 1, False),
        (1, 1.0, False),
        (1.0, 1, False),
        (0.0, -0.0, False),
        (math.nan, math.nan, True),  # the same JSON, though NaN != NaN
        (Tone.CALM, "calm", True),  # a str enum is recorded as its value
        ("calm", Tone.CALM, True),
    ],
)
async def test_a_request_extends_a_turn_only_when_its_history_is_recorded_identically(
    sent, resent, extends
):
    model, held = access()
    chat = model.llm().conversation()
    await chat.generate(messages=[{"role": "user", "content": "x", "flag": sent}])
    await chat.generate(messages=[{"role": "user", "content": "x", "flag": resent}, user("y")])
    first, second = (sealed.turn for sealed in (await sealed_transcript(held)).turns)
    assert (second.extends == first.sample_id) is extends


async def test_an_image_changed_in_place_between_turns_is_not_the_same_history():
    """The agent's history holds one PIL object; the sampler got different bytes."""
    model, held = access(PrimeRenderer(FakePrime()), sampler=EchoSampler())
    chat = model.llm().conversation()
    picture = Image.new("RGB", (4, 4), "red")
    history = [{"role": "user", "content": [{"type": "image", "image": picture}]}]
    one = await chat.generate(messages=history)
    picture.putpixel((0, 0), (0, 0, 255))
    await chat.generate(messages=[*history, one.native_message, user("again")])
    first, second = (sealed.turn for sealed in (await sealed_transcript(held)).turns)
    assert second.extends is None
    assert first.messages[0] != second.messages[0]


@pytest.mark.parametrize(
    "spelled",
    [
        lambda data: {"type": "image", "image": data},
        lambda data: {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()},
        },
    ],
    ids=["bytes", "data-uri"],
)
async def test_a_resent_history_image_is_decoded_once_and_never_by_the_recording(
    spelled, monkeypatch
):
    """A long screenshot agent resends every image each turn; the client decodes each
    once, recording takes the bytes the sampler got as they are, and stores each once."""
    from loops_cookbook.llm import client, images, transcript

    decodes = {"client": 0, "transcript": 0}
    decode = images.image_bytes

    for name, module in (("client", client), ("transcript", transcript)):

        def counted(part, name=name):
            decodes[name] += 1
            return decode(part)

        monkeypatch.setattr(module, "image_bytes", counted)
    model, held = access(PrimeRenderer(FakePrime()), sampler=EchoSampler())
    chat = model.llm().conversation()
    history, turns = [], 6
    for n in range(turns):
        history.append({"role": "user", "content": [spelled(png((n + 2, 2)))]})
        reply = await chat.generate(messages=list(history))
        history.append(reply.native_message)

    assert decodes == {"client": turns, "transcript": 0}
    transcript_ = await sealed_transcript(held)
    ids = [sealed.turn.sample_id for sealed in transcript_.turns]
    assert [sealed.turn.extends for sealed in transcript_.turns] == [None, *ids[:-1]]
    assert len(transcript_.images) == turns
    assert len(transcript_.request(ids[-1])) == 2 * turns - 1


def test_a_rendered_history_with_an_image_the_request_lacks_is_not_recorded():
    """Pairing digests by position is only sound when both sides hold the same images."""
    recorder = TranscriptRecorder()
    rendered = [{"role": "user", "content": [{"type": "image", "image": PNG}]}]
    pending = recorder.begin(
        [user("hi")], [], rendered=rendered, named=True, producer="main", conversation_id="main"
    )
    pending.parsed(make_sample("s1", (1,), (2,), (-0.1,)), None, None)
    (turn,) = recorder.recorded().turns
    assert turn.messages == () and turn.reply is not None
    assert (turn.unrecordable or "").startswith("request: TrajectoryError")


async def test_a_conversation_extends_only_its_latest_turn_never_an_older_one():
    model, held = access()
    chat = model.llm().conversation()
    await chat.generate(messages=[user("a")])
    await chat.generate(messages=[user("b")])
    await chat.generate(messages=[user("a"), user("again")])
    *_, third = (sealed.turn for sealed in (await sealed_transcript(held)).turns)
    assert third.extends is None


async def test_one_conversation_id_under_two_producers_is_two_conversations():
    model, held = access(producers=ProducerSettings({"main": True, "critic": False}))
    await model.llm().conversation(conversation_id="shared").generate(messages=[user("same")])
    await (
        model.llm("critic")
        .conversation(conversation_id="shared")
        .generate(messages=[user("same"), user("more")])
    )
    assert [sealed.turn.extends for sealed in (await sealed_transcript(held)).turns] == [None, None]


async def test_turns_of_separate_conversations_never_extend_each_other():
    model, held = access()
    await model.llm().conversation().generate(messages=[user("same")])
    await model.llm().conversation().generate(messages=[user("same"), user("more")])
    assert [sealed.turn.extends for sealed in (await sealed_transcript(held)).turns] == [None, None]


def _unread(unread):
    def raises(_n):
        raise KeyError("renderer bug")

    return raises if unread == "raises" else (lambda _n: ParsedReply("cut", None, (), False, None))


@pytest.mark.parametrize("recording", [True, False])
@pytest.mark.parametrize("unread", ["raises", "no_native_message"])
async def test_a_completion_the_renderer_cannot_read_is_recorded_and_fails_the_seal(
    unread, recording
):
    """Recorded, the turn says what the agent received; the outcome is recording-off's."""
    renderer = ScriptedRenderer()
    renderer.reply = _unread(unread)
    model, held = access(renderer, recording=recording)
    with pytest.raises(RendererFailure) as failed:
        await model.llm().conversation().generate(messages=[user("hi")])
    if recording:
        assert held.transcript is not None
        (turn,) = held.transcript.recorded().turns
        assert turn.reply == ParsedTurnReply(None, f"RendererFailure: {failed.value}")
    with pytest.raises(RendererFailure):
        await held.seal()


async def test_a_tool_call_the_renderer_could_not_parse_reaches_the_agent_and_trains():
    """The model's own output: each call with its status, recorded, and trained."""
    good = ParsedToolCall(raw='{"name": "lookup"}', name="lookup", arguments={})
    bad = ParsedToolCall(raw='{"name": "look', status=ToolCallParseStatus.INVALID_JSON)
    renderer = ScriptedRenderer()
    renderer.reply = lambda _n: ParsedReply("", None, (good, bad), True, None)
    model, held = access(renderer)
    response = await model.llm().conversation().generate(messages=[user("hi")], tools=[LOOKUP])
    assert [call.status for call in response.tool_calls] == [
        ToolCallParseStatus.OK,
        ToolCallParseStatus.INVALID_JSON,
    ]
    assert response.native_message is None and isinstance(response.parse_error, RenderingError)

    sealed = await held.seal()
    assert len(sealed.samples) == 1 and sealed.transcript is not None
    (turn,) = sealed.transcript.turns
    assert turn.disposition is TurnDisposition.DELIVERED and turn.trainable
    reply = turn.turn.reply
    assert isinstance(reply, ParsedTurnReply) and reply.parsed is not None
    assert [call["status"] for call in reply.parsed["tool_calls"]] == ["ok", "invalid_json"]
    assert reply.parse_error == f"RenderingError: {response.parse_error}"


def nan_arguments(_n):
    call = ParsedToolCall(raw="<call>", name="lookup", arguments={"x": math.nan, "y": math.inf})
    return ParsedReply("", None, (call,), True, {"role": "assistant", "content": ""})


async def test_a_non_finite_tool_argument_is_recorded_as_the_policy_wrote_it():
    """The policy's own output never changes an outcome: it is kept, and it trains."""
    from loops_cookbook.rl.run import transcript_record

    renderer = ScriptedRenderer()
    renderer.reply = nan_arguments
    model, held = access(renderer)
    await model.llm().conversation().generate(messages=[user("hi")])
    sealed = await held.seal()
    assert sealed.transcript is not None and len(sealed.samples) == 1
    (turn,) = sealed.transcript.turns
    assert turn.trainable
    reply = turn.turn.reply
    assert isinstance(reply, ParsedTurnReply) and reply.parsed is not None
    arguments = reply.parsed["tool_calls"][0]["arguments"]
    assert math.isnan(arguments["x"]) and arguments["y"] == math.inf
    # On disk, strict JSON: the strings JavaScript uses for them.
    written = json.loads(json.dumps(transcript_record(sealed.transcript), allow_nan=False))
    assert written["turns"][0]["reply"]["parsed"]["tool_calls"][0]["arguments"] == {
        "x": "NaN",
        "y": "Infinity",
    }


async def test_an_unrecordable_reply_is_marked_and_the_attempt_still_trains():
    """How a turn is recorded never decides whether its attempt trains."""
    renderer = ScriptedRenderer()
    renderer.reply = lambda n: (
        ParsedReply("", None, (), True, {"role": "assistant", "content": "", "note": {1}})
        if n == 1
        else ScriptedRenderer().reply(n)
    )
    model, held = access(renderer)
    chat = model.llm().conversation()
    chat.discard(await chat.generate(messages=[user("hi")]))
    await chat.generate(messages=[user("hi")])
    sealed = await held.seal()
    assert sealed.transcript is not None and sealed.transcript.sealed
    unread, kept = sealed.transcript.turns
    assert unread.turn.reply is None and unread.disposition is TurnDisposition.DISCARDED
    assert (unread.turn.unrecordable or "").startswith("reply: TrajectoryError")
    assert thaw(unread.turn.messages) == [user("hi")]  # the request is still recorded
    assert kept.turn.unrecordable is None and kept.trainable
    assert [sample.sample_id for sample in sealed.samples] == [kept.turn.sample_id]


async def test_an_unrecordable_request_is_marked_and_the_next_one_is_recorded_whole():
    from loops_cookbook.rl.run import transcript_record

    model, held = access()
    odd = {"role": "user", "content": "hi", "note": {1, 2}}
    assert (await model.llm().conversation().generate(messages=[odd])).content == "answer 1"
    await model.llm().conversation().generate(messages=[user("hi"), user("go")])
    transcript = await sealed_transcript(held)
    first, second = transcript.turns
    assert first.turn.messages == () and first.trainable
    assert (first.turn.unrecordable or "").startswith("request: TrajectoryError")
    # Nothing extends a request that was never recorded, so the next is whole.
    assert second.turn.extends is None and second.turn.unrecordable is None
    assert thaw(second.turn.messages) == [user("hi"), user("go")]
    written = json.loads(json.dumps(transcript_record(transcript)))
    assert [turn["unrecordable"] is not None for turn in written["turns"]] == [True, False]


@pytest.mark.parametrize("answered", [True, False])
def test_a_tool_that_cannot_be_spelled_marks_its_request_and_keeps_the_rest(answered):
    """A 5,001-digit integer freezes but has no JSON spelling under Python's limit."""
    recorder = TranscriptRecorder()
    kept = recorder.begin(
        [user("hi")],
        [LOOKUP],
        rendered=[user("hi")],
        named=True,
        producer="main",
        conversation_id="main",
    )
    sample = make_sample("s1", (1,), (2,), (-0.1,))
    kept.parsed(sample, None, None)
    odd = {"type": "function", "function": {"name": "big", "x": 10**5000}}
    pending = recorder.begin(
        [user("go")],
        [odd],
        rendered=[user("go")],
        named=True,
        producer="main",
        conversation_id="other",
    )
    later = make_sample("s2", (1,), (2,), (-0.1,))
    if answered:
        pending.parsed(later, None, None)
    first, marked = recorder.recorded().turns
    assert first.unrecordable is None and first.tools
    assert marked.tools == () and marked.messages == ()
    assert (marked.unrecordable or "").startswith("request: ValueError")
    assert marked.sample_id == ("s2" if answered else None)


@pytest.mark.parametrize("where", ["answered_request", "unanswered_request", "reply"])
def test_a_value_that_cannot_be_spelled_marks_only_its_part_and_the_rest_is_written(where):
    """A 5,001-digit integer freezes but has no JSON spelling: the request or reply
    holding it is marked unrecordable, and the transcript still writes."""
    from loops_cookbook.llm.transcript import seal_transcript
    from loops_cookbook.rl.run import transcript_record

    odd = {"role": "user", "content": "go", "x": 10**5000}
    recorder = TranscriptRecorder()
    kept = recorder.begin(
        [user("hi")], [], rendered=[user("hi")], named=True, producer="main", conversation_id="a"
    )
    kept.parsed(make_sample("s1", (1,), (2,), (-0.1,)), None, None)
    message = odd if where != "reply" else user("go")
    pending = recorder.begin(
        [message], [], rendered=[user("go")], named=True, producer="main", conversation_id="b"
    )
    if where != "unanswered_request":
        native = {"role": "assistant", "content": "", "x": 10**5000} if where == "reply" else None
        reply = ParsedReply("", None, (), True, native) if native is not None else None
        pending.parsed(make_sample("s2", (1,), (2,), (-0.1,)), reply, None)
    first, marked = recorder.recorded().turns
    assert first.unrecordable is None
    part = "reply" if where == "reply" else "request"
    assert (marked.unrecordable or "").startswith(f"{part}: ValueError")
    sealed = seal_transcript(
        recorder.recorded(), evidence=(), discarded=(), undelivered={"s1", "s2"}, sealed=False
    )
    json.dumps(transcript_record(sealed), allow_nan=False)


def test_a_request_no_generation_answered_is_kept_whole_and_marked_unanswered():
    from loops_cookbook.rl.run import transcript_record, transcript_requests

    recorder = TranscriptRecorder()
    answered = recorder.begin(
        [user("hi")],
        [LOOKUP],
        rendered=[user("hi")],
        named=True,
        producer="main",
        conversation_id="main",
    )
    sample = make_sample("s1", (1,), (2,), (-0.1,))
    answered.parsed(sample, None, None)
    history = [user("hi"), {"role": "assistant", "content": "a"}, user("again")]
    recorder.begin(
        history, [SEARCH], rendered=history, named=True, producer="main", conversation_id="main"
    )
    sealed = seal_transcript(recorder.recorded(), evidence=(sample,), discarded=(), undelivered=())
    first, waiting = sealed.turns
    assert first.disposition is TurnDisposition.DELIVERED
    assert waiting.disposition is TurnDisposition.UNANSWERED and not waiting.trainable
    assert waiting.turn.sample_id is None and waiting.turn.reply is None
    assert waiting.turn.extends is None and len(waiting.turn.messages) == 3
    (digest,) = waiting.turn.tools
    assert thaw(sealed.tools[digest]) == SEARCH  # its definition is kept with the rest
    written = json.loads(json.dumps(transcript_record(sealed)))
    assert written["turns"][1]["sample_id"] is None
    assert list(transcript_requests(written)) == ["s1"]


def test_evidence_without_a_turn_is_refused_as_is_a_turn_without_a_fact():
    sample = make_sample("s1", (1,), (2,), (-0.1,))
    empty = TranscriptRecorder().recorded()
    with pytest.raises(ValueError, match="no recorded turn"):
        seal_transcript(empty, evidence=(sample,), discarded=(), undelivered=())

    recorder = TranscriptRecorder()
    pending = recorder.begin(
        [user("hi")],
        [],
        rendered=[user("hi")],
        named=True,
        producer="main",
        conversation_id="main",
    )
    pending.parsed(sample, None, None)
    with pytest.raises(ValueError, match="no generation"):
        seal_transcript(recorder.recorded(), evidence=(), discarded=(), undelivered=())


async def test_images_and_tool_definitions_are_stored_once_by_digest():
    data_uri = "data:image/png;base64," + base64.b64encode(PNG).decode()
    model, held = access(PrimeRenderer(FakePrime()), sampler=EchoSampler())
    chat = model.llm().conversation()
    first = [{"role": "user", "content": [{"type": "image", "image": PNG}]}]
    one = await chat.generate(messages=first, tools=[LOOKUP, SEARCH])
    # The same image again, spelled as a data URI: the same bytes reach the sampler.
    again = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_uri}}]}
    await chat.generate(messages=[*first, one.native_message, again], tools=[LOOKUP, SEARCH])

    transcript = await sealed_transcript(held)
    (digest,) = transcript.images
    assert digest.startswith("sha256:") and transcript.images[digest].data == PNG
    assert transcript.images[digest].format == "png"
    first_turn, second_turn = (sealed.turn for sealed in transcript.turns)
    assert thaw(first_turn.messages) == [
        {"role": "user", "content": [{"type": "image", "image": digest}]}
    ]
    assert thaw(second_turn.messages[-1]) == {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": digest}}],
    }
    assert len(transcript.tools) == 2
    assert first_turn.tools == second_turn.tools
    assert [thaw(transcript.tools[d]) for d in first_turn.tools] == [LOOKUP, SEARCH]


class ReorderingPrime(FakePrime):
    """Renders messages in reverse and reports its image parts in that order, as GLM-5.3
    reports the tool results it sorts into call order."""

    def render(self, messages, **kwargs):
        ordered = list(reversed(messages))
        rendered = self._render(ordered)
        if rendered.multi_modal_data is not None:
            rendered.multi_modal_data.mm_items["image"] = image_parts(ordered)
        return rendered


@pytest.mark.parametrize("prime", [FakePrime, ReorderingPrime])
async def test_each_image_part_is_named_by_its_own_bytes(prime):
    red, blue = png((3, 3), "red"), png((5, 5), "blue")
    sampler = EchoSampler()
    model, held = access(PrimeRenderer(prime()), sampler=sampler)
    history = [
        {"role": "user", "content": [{"type": "image", "image": red}]},
        {"role": "user", "content": [{"type": "image", "image": blue}]},
    ]
    await model.llm().conversation().generate(messages=history)
    sampled = [c.data for c in sampler.prompts[0].chunks if isinstance(c, ImageChunk)]
    assert sampled == ([red, blue] if prime is FakePrime else [blue, red])

    transcript = await sealed_transcript(held)
    digest = {raw: "sha256:" + hashlib.sha256(raw).hexdigest() for raw in (red, blue)}
    assert [
        message["content"][0]["image"] for message in thaw(transcript.turns[0].turn.messages)
    ] == [
        digest[red],
        digest[blue],
    ]
    assert {name: image.data for name, image in transcript.images.items()} == {
        digest[red]: red,
        digest[blue]: blue,
    }


async def test_an_image_is_recorded_as_the_bytes_the_sampler_received():
    """A PIL image is re-encoded for the sampler; the transcript names what it got."""
    sampler = EchoSampler()
    model, held = access(PrimeRenderer(FakePrime()), sampler=sampler)
    picture = Image.new("RGB", (4, 4), "red")
    await (
        model.llm()
        .conversation()
        .generate(messages=[{"role": "user", "content": [{"type": "image", "image": picture}]}])
    )
    (chunk,) = [c for c in sampler.prompts[0].chunks if isinstance(c, ImageChunk)]
    transcript = await sealed_transcript(held)
    digest = "sha256:" + hashlib.sha256(chunk.data).hexdigest()
    assert dict(transcript.images) == {digest: transcript.images[digest]}
    assert transcript.images[digest].data == chunk.data


async def test_a_history_edited_while_its_generation_runs_is_recorded_as_it_was_sent():
    sampler = ScriptedSampler(stop_reason="stop")
    sampler.release = asyncio.Event()
    model, held = access(sampler=sampler)
    history = [{"role": "user", "content": [{"type": "text", "text": "as sent"}]}]
    call = asyncio.create_task(model.llm().conversation().generate(messages=history))
    await sampler.entered.wait()
    history[0]["content"][0]["text"] = "edited meanwhile"
    history.append(user("appended meanwhile"))
    sampler.release.set()
    await call
    (sealed,) = (await sealed_transcript(held)).turns
    assert thaw(sealed.turn.messages) == [
        {"role": "user", "content": [{"type": "text", "text": "as sent"}]}
    ]


async def test_seal_marks_the_turns_that_are_not_trainable():
    model, held = access(producers=ProducerSettings({"main": True, "critic": False}))
    chat = model.llm().conversation()
    rejected = await chat.generate(messages=[user("try")])
    chat.discard(rejected)
    kept = await chat.generate(messages=[user("try")])
    critique = await model.llm("critic").conversation().generate(messages=[user("judge")])

    marks = {
        sealed.turn.sample_id: (sealed.disposition, sealed.trainable)
        for sealed in (await sealed_transcript(held)).turns
    }
    assert marks == {
        rejected.response_id: (TurnDisposition.DISCARDED, False),
        kept.response_id: (TurnDisposition.DELIVERED, True),
        critique.response_id: (TurnDisposition.DELIVERED, False),
    }


async def test_a_conversation_reused_after_seal_fails_and_changes_no_transcript():
    model, held = access()
    chat = model.llm().conversation()
    one = await chat.generate(messages=[user("hi")])
    transcript = await sealed_transcript(held)
    before = copy.deepcopy(transcript)

    with pytest.raises(CaptureIntegrityFailure, match="sealed"):
        await chat.generate(messages=[user("hi"), one.native_message, user("more")])
    assert transcript == before
    assert [turn.turn.sample_id for turn in transcript.turns] == [one.response_id]


async def test_a_transcript_is_deeply_immutable_and_survives_copy_and_pickle():
    """It is handed to user code, which must not be able to change the evidence."""
    model, held = access()
    content = [{"type": "text", "text": "hi"}]
    await (
        model.llm()
        .conversation()
        .generate(messages=[{"role": "user", "content": content}], tools=[LOOKUP])
    )
    content[0]["text"] = "changed after the call"
    transcript = await sealed_transcript(held)
    (message,) = transcript.turns[0].turn.messages
    assert isinstance(message, FrozenMap)
    parts = message["content"]
    assert isinstance(parts, tuple) and parts[0]["text"] == "hi"
    with pytest.raises(TypeError):
        parts[0]["text"] = "edited"  # pyright: ignore[reportIndexIssue]
    with pytest.raises(AttributeError):
        parts[0]._items = {}  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(AttributeError):
        del parts[0]._items  # pyright: ignore[reportAttributeAccessIssue]
    assert copy.deepcopy(transcript) == transcript
    assert pickle.loads(pickle.dumps(transcript)) == transcript
    # Frozen all the way down, so a turn, a reply and the whole transcript hash.
    assert hash(pickle.loads(pickle.dumps(transcript))) == hash(transcript)
    assert len({transcript.turns[0].turn, copy.deepcopy(transcript.turns[0].turn)}) == 1


async def test_without_recording_nothing_is_recorded_or_converted():
    model, held = access(recording=False)
    assert held.transcript is None
    await (
        model.llm()
        .conversation()
        .generate(messages=[{"role": "user", "content": "hi", "note": {1, 2}}])
    )
    sealed = await held.seal()
    assert len(sealed.samples) == 1 and sealed.transcript is None


def test_a_transcript_sink_requires_recording():
    with pytest.raises(ValueError, match="records_transcripts"):
        lease(ScriptedSampler(), transcripts=[], records_transcripts=False)


async def test_the_written_transcript_names_images_and_never_embeds_them():
    from loops_cookbook.rl.run import transcript_record

    model, held = access(PrimeRenderer(FakePrime()), sampler=EchoSampler())
    message = {"role": "user", "content": [{"type": "image", "image": PNG}]}
    await model.llm().conversation().generate(messages=[message], tools=[LOOKUP])

    record = transcript_record(await sealed_transcript(held))
    written = json.dumps(record)
    assert base64.b64encode(PNG).decode() not in written
    ((digest, image),) = record["images"].items()
    assert image == {"format": "png", "bytes": len(PNG)}
    (turn,) = record["turns"]
    assert turn["messages"] == [{"role": "user", "content": [{"type": "image", "image": digest}]}]
    assert [record["tools"][d] for d in turn["tools"]] == [LOOKUP]
    assert turn["trainable"] is True


async def test_a_written_transcript_rebuilds_every_request_through_any_turn_it_extends():
    """A trainable turn can extend a discarded one; a reader must follow it anyway."""
    from loops_cookbook.rl.run import transcript_record, transcript_requests

    model, held = access()
    chat = model.llm().conversation()
    first = [user("try")]
    chat.discard(await chat.generate(messages=first))
    kept = await chat.generate(messages=first)
    second = [*first, kept.native_message, user("more")]
    await chat.generate(messages=second)
    transcript = await sealed_transcript(held)
    rejected, retried, last = (sealed.turn for sealed in transcript.turns)
    assert retried.extends == rejected.sample_id and retried.messages == ()

    written = json.loads(json.dumps(transcript_record(transcript)))
    assert transcript_requests(written) == {
        rejected.sample_id: first,
        retried.sample_id: first,
        last.sample_id: second,
    }


@pytest.mark.parametrize("ending", ["revoked", "stale_at_seal"])
async def test_an_attempt_that_ends_unsealed_keeps_its_transcript_as_far_as_it_got(ending):
    kept, version = [], [0]
    held = lease(
        ScriptedSampler(stop_reason="stop"),
        transcripts=kept,
        policy_version=lambda: version[0],
        max_steps_off_policy=0,
    )
    chat = DirectModelAccess(held, Renderers(ScriptedRenderer())).llm().conversation()
    await chat.generate(messages=[user("hi")])
    if ending == "revoked":
        held.revoke()
        await held.keep_unsealed()
        await held.keep_unsealed()  # once only
    else:
        version[0] = 1  # the policy moves on: the next capture is stale
        with pytest.raises(StaleRolloutError):
            await chat.generate(messages=[user("hi"), {"role": "assistant", "content": "a"}])
        with pytest.raises(StaleRolloutError):
            await held.seal()
        await held.keep_unsealed()  # the seal kept it already
    (transcript,) = kept
    assert not transcript.sealed and transcript.turns
    assert not any(turn.trainable for turn in transcript.turns)
    assert transcript.turns[0].disposition is TurnDisposition.DELIVERED


async def test_a_seal_cancelled_while_writing_leaves_a_transcript_that_never_trains():
    """The cancellation aborts the attempt, so the last word on disk is unsealed."""
    kept, writing, release = [], asyncio.Event(), asyncio.Event()

    async def keep(transcript):
        writing.set()
        await release.wait()
        kept.append(transcript)

    held = lease(ScriptedSampler(stop_reason="stop"), keep_transcript=keep)
    chat = DirectModelAccess(held, Renderers(ScriptedRenderer())).llm().conversation()
    await chat.generate(messages=[user("hi")])
    sealing = asyncio.create_task(held.seal())
    await writing.wait()
    sealing.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await sealing
    first, last = kept
    assert first.sealed and first.turns[0].trainable
    assert not last.sealed and not any(turn.trainable for turn in last.turns)
    assert last.turns[0].disposition is TurnDisposition.DELIVERED


async def test_a_history_that_contains_itself_is_sampled_and_marked_unrecordable():
    """The one request recording cannot even copy is still sampled, and marked."""
    loop: list = []
    loop.append(loop)
    model, held = access()
    chat = model.llm().conversation()
    response = await chat.generate(messages=[{"role": "user", "content": "hi", "x": loop}])
    assert response.content == "answer 1"
    (sealed,) = (await sealed_transcript(held)).turns
    assert sealed.trainable and sealed.turn.messages == () and sealed.turn.reply is not None
    assert (sealed.turn.unrecordable or "").startswith("request: RecursionError")


async def test_a_transcript_the_run_cannot_write_is_logged_and_the_attempt_seals(caplog):
    """How an attempt is recorded never decides whether it trains."""

    class FullDisk(list):
        def append(self, _transcript):
            raise OSError("no space left on device")

    held = lease(ScriptedSampler(stop_reason="stop"), transcripts=FullDisk())
    model = DirectModelAccess(held, Renderers(ScriptedRenderer()))
    await model.llm().conversation().generate(messages=[user("hi")])
    sealed = await held.seal()
    assert len(sealed.samples) == 1
    assert "transcript was not written" in caplog.text and "no space left" in caplog.text
