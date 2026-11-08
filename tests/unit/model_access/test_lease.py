"""The direct attempt lease: admission, identity-only attribution, capture, sealing."""

import asyncio
from types import SimpleNamespace

import pytest
from baseten.loops.models import SamplingParams
from support.cancellation import (
    ALL_LANDINGS,
    Gate,
    Landing,
    Response,
    Trigger,
    cases,
    settle_loop,
)

from loops_cookbook.evidence import (
    AttemptIdentity,
    InternalPolicyDeltaError,
    StaleRolloutError,
)
from loops_cookbook.llm.client import Renderers, SamplingPolicy
from loops_cookbook.llm.rendering import RenderedPrompt
from loops_cookbook.model_access.api import (
    CaptureIntegrityFailure,
    FuturePolicyVersion,
    ModelAccessError,
    ModelAccessMode,
    ModelTransportFailure,
    SamplingFailure,
    UndeclaredProducer,
)
from loops_cookbook.model_access.lease import (
    AttemptLease,
    DirectModelAccess,
    Freshness,
    LeaseState,
    LeaseWiring,
    affinity_key,
)
from loops_cookbook.model_access.producers import ProducerSettings
from loops_cookbook.trainer import Capacity

PARAMS = SamplingParams(max_tokens=2)


class Sampler:
    def __init__(self, version=2, tokens=(97, 98)):
        self.version, self.tokens = version, tokens
        self.requests = []

    async def sample_async(self, prompt, **kwargs):
        self.requests.append(kwargs)
        await asyncio.sleep(0)
        return SimpleNamespace(
            prompt_token_ids=prompt.to_ints(),
            policy_version=self.version,
            sequences=[
                SimpleNamespace(
                    tokens=list(self.tokens),
                    logprobs=[-0.1] * len(self.tokens),
                    sequence_id=None,
                    stop_reason="length",
                )
            ],
        )


class Tampered(Sampler):
    """Answers as ``Sampler`` does, then lets ``edit`` change the response."""

    def __init__(self, edit):
        super().__init__()
        self.edit = edit

    async def sample_async(self, prompt, **kwargs):
        response = await super().sample_async(prompt, **kwargs)
        self.edit(response)
        return response


def identity(attempt="a1", rollout="r1"):
    return AttemptIdentity("run", "g", rollout, attempt)


def wiring(client: object, **overrides: object):
    captured, discarded = [], []

    async def on_sample(identity, sample):
        captured.append((identity, sample))

    values = {
        "sampler": lambda: client,
        "capacity": Capacity.DEDICATED,
        "policy_version": lambda: 2,
        "producers": ProducerSettings({"main": True, "critic": False}),
        "freshness": Freshness(max_steps_off_policy=3, max_policy_delta_in_trajectory=1),
        "sampling_policy": SamplingPolicy(max_generation_tokens=2),
        "on_sample": on_sample,
        # What left the evidence, and how: ("discarded" | "undelivered", identity, sample).
        "on_discard": lambda who, sample: discarded.append(("discarded", who, sample)),
        "on_undelivered": lambda who, sample: discarded.append(("undelivered", who, sample)),
        **overrides,
    }
    built = LeaseWiring(**values)
    return built, captured, discarded


def lease(sampler=None, *, affinity=None, **overrides):
    built, captured, discarded = wiring(sampler or Sampler(), **overrides)
    return AttemptLease(identity(), built, affinity_key=affinity), captured, discarded


def prompt(*ids):
    return RenderedPrompt.text(ids or (1,))


async def test_generation_is_captured_under_the_attempt_before_it_returns():
    attempt, captured, _ = lease()
    sample = await attempt.generate("main", prompt(1, 2), PARAMS, conversation_id="c")
    assert captured == [(attempt.identity, sample)]
    assert (sample.producer, sample.conversation_id, sample.trainable) == ("main", "c", True)
    assert (await attempt.seal()).samples == (sample,)
    assert attempt.state is LeaseState.SEALED


async def test_trainable_comes_only_from_the_configured_producer_set():
    attempt, _, _ = lease()
    critic = await attempt.generate("critic", prompt(), PARAMS, conversation_id="c")
    main = await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    assert (critic.trainable, main.trainable) == (False, True)


def test_with_no_table_the_one_producer_is_a_trainable_main():
    settings = ProducerSettings()
    assert settings.resolve(None) == "main"
    assert settings.trainable("main") is True
    with pytest.raises(ModelAccessError, match="'critic' is not declared"):
        settings.resolve("critic")


@pytest.mark.parametrize("declared", [{}, {"judge": False}, {"": True}])
def test_producer_settings_need_a_trainable_named_producer(declared):
    with pytest.raises(ValueError, match="trainable|nonempty"):
        ProducerSettings(declared)


async def test_an_undeclared_producer_never_samples_even_through_the_lease():
    sampler = Sampler()
    attempt, captured, _ = lease(sampler)
    with pytest.raises(ModelAccessError, match=r"'mian' is not declared.*\['critic', 'main'\]"):
        await attempt.generate("mian", prompt(), PARAMS, conversation_id="c")
    assert sampler.requests == [] and captured == []
    # A configuration error, not the attempt's: the attempt cannot seal past it.
    with pytest.raises(UndeclaredProducer):
        await attempt.seal()


@pytest.mark.parametrize("typo", ["critc", "Main", ""])
async def test_direct_access_refuses_an_undeclared_producer_by_name(typo):
    attempt, _, _ = lease()
    access = DirectModelAccess(attempt, Renderers(object()))
    with pytest.raises(ModelAccessError, match=rf"{typo!r} is not declared.*\['critic', 'main'\]"):
        access.llm(typo)


async def test_direct_access_locks_each_client_to_its_producer():
    attempt, _, _ = lease()
    access = DirectModelAccess(attempt, Renderers(object()))
    assert access.mode is ModelAccessMode.DIRECT
    assert access.llm().producer == "main"
    assert access.llm("critic").producer == "critic"
    assert access.llm() is access.llm("main")
    conversation = access.llm("critic").conversation(conversation_id="x")
    assert conversation.producer == "critic"
    with pytest.raises(TypeError):
        access.llm().conversation(producer="critic")  # pyright: ignore[reportCallIssue]


async def test_simultaneous_identical_requests_attribute_by_identity_only():
    """Two attempts send byte-identical prompts and get identical completions."""
    sampler = Sampler()
    built, captured, _ = wiring(sampler)
    first = AttemptLease(identity("a1", "r1"), built, affinity_key=None)
    second = AttemptLease(identity("a2", "r2"), built, affinity_key=None)
    one, two = await asyncio.gather(
        first.generate("main", prompt(5, 6), PARAMS, conversation_id="same"),
        second.generate("main", prompt(5, 6), PARAMS, conversation_id="same"),
    )
    assert one.completion_ids == two.completion_ids and one.sample_id != two.sample_id
    assert (await first.seal()).samples == (one,)
    assert (await second.seal()).samples == (two,)
    assert {(i.attempt_id, s.sample_id) for i, s in captured} == {
        ("a1", one.sample_id),
        ("a2", two.sample_id),
    }


@pytest.mark.parametrize("recording", [False, True])
async def test_routes_are_captured_exactly_when_generations_are_recorded(recording):
    sampler = Sampler()
    attempt, _, _ = lease(sampler, records_generations=recording)
    await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    assert sampler.requests[0]["include_routed_experts"] is recording


def image_prompt():
    from baseten.loops.models import EncodedTextChunk, ImageChunk, ModelInput

    chunks = [
        EncodedTextChunk(tokens=[1]),
        ImageChunk(data=b"png", format="png", expected_tokens=2),
    ]
    return RenderedPrompt((1, 7, 7), ModelInput(chunks=chunks))


@pytest.mark.parametrize("records", [False, True])
async def test_a_recording_sampler_is_never_sent_an_image_prompt(records):
    """Loops records text prompts only: under replay, an image prompt fails here, says
    which settings to change, and never reaches the sampler."""

    class Echoing(Sampler):
        """Echoes the expanded prompt, as a sampler does for an image."""

        async def sample_async(self, prompt, **kwargs):
            self.requests.append(kwargs)
            return SimpleNamespace(
                prompt_token_ids=[1, 7, 7],
                policy_version=self.version,
                sequences=[
                    SimpleNamespace(
                        tokens=[97], logprobs=[-0.1], sequence_id=None, stop_reason="stop"
                    )
                ],
            )

    sampler, said = Echoing(), []
    attempt, _, _ = lease(
        sampler, records_generations=records, on_unrecordable_prompt=lambda: said.append(1)
    )
    if records:
        with pytest.raises(SamplingFailure, match="has_images = true"):
            await attempt.generate("main", image_prompt(), PARAMS, conversation_id="c")
        assert sampler.requests == [] and said == [1]
    else:
        await attempt.generate("main", image_prompt(), PARAMS, conversation_id="c")
        assert len(sampler.requests) == 1 and said == []


async def test_a_recording_sampler_still_samples_text_and_keeps_its_record():
    class Recording(Sampler):
        async def sample_async(self, prompt, **kwargs):
            response = await super().sample_async(prompt, **kwargs)
            response.sequences[0].sequence_id = "seq-1"
            return response

    attempt, captured, _ = lease(Recording(), records_generations=True)
    await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    assert [sample.sequence_id for _, sample in captured] == ["seq-1"]


@pytest.mark.parametrize("affinity", [None, "run-unique:rollout-1"])
async def test_affinity_is_opt_in_on_actual_sdk_request(monkeypatch, affinity):
    import httpx
    from baseten.loops.sampling_client import LocalDeployment, SamplingClient

    requests = []

    def serve(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"x-trainers-policy-version": "2"},
            json={"choices": [{"token_ids": [97], "logprobs": {"token_logprobs": [-0.1]}}]},
        )

    sampler = SamplingClient(base_model="m", deployment=LocalDeployment(base_url="http://test"))
    monkeypatch.setattr(
        sampler, "_async_client", httpx.AsyncClient(transport=httpx.MockTransport(serve))
    )
    attempt, _, _ = lease(sampler, affinity=affinity)
    await attempt.generate("main", prompt(), SamplingParams(max_tokens=1), conversation_id="c")
    assert requests[0].headers.get("x-baseten-session-id") == affinity


def sdk_sampler(monkeypatch, requests):
    """A real SDK sampling client whose requests land in ``requests``: the SDK's own
    cached client over a mock transport, so one sampler serves every attempt."""
    import httpx
    from baseten.loops._async_client import _LazyAsyncClient
    from baseten.loops.sampling_client import LocalDeployment, SamplingClient

    def serve(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"x-trainers-policy-version": "2"},
            json={"choices": [{"token_ids": [97], "logprobs": {"token_logprobs": [-0.1]}}]},
        )

    sampler = SamplingClient(base_model="m", deployment=LocalDeployment(base_url="http://test"))
    monkeypatch.setattr(
        sampler,
        "_async_client",
        _LazyAsyncClient(transport_factory=lambda: httpx.MockTransport(serve)),
    )
    return sampler


async def test_each_request_names_its_attempt_for_loops_tracing(monkeypatch):
    """The attempt's ID, which always fits Loops' rule and names the attempt's own
    samples and transcript in the run record."""
    requests = []
    built, _, _ = wiring(sdk_sampler(monkeypatch, requests))
    # A task ID can be anything, so a rollout ID can be past what Loops traces.
    for attempt_id, rollout_id in (("a1", "g0000001-task-r0"), ("a2", "g0000002-" + "t" * 90)):
        attempt = AttemptLease(identity(attempt_id, rollout_id), built, affinity_key=None)
        for _ in range(2):
            await attempt.generate(
                "main", prompt(), SamplingParams(max_tokens=1), conversation_id="c"
            )
    assert [request.headers["x-rollout-id"] for request in requests] == ["a1", "a1", "a2", "a2"]


async def test_a_rollouts_affinity_key_holds_across_its_attempts_and_carries_nothing_of_its_task(
    monkeypatch,
):
    """A task ID is the user's, and may not fit in a header: a key carrying it failed
    every generation of a task named outside ASCII."""
    requests = []
    built, _, _ = wiring(sdk_sampler(monkeypatch, requests))
    for attempt_id, rollout_id in (
        ("a1", "g0000001-café-1-r0"),
        ("a2", "g0000001-café-1-r0"),  # a retry of that rollout
        ("a3", "g0000001-café-1-r1"),
    ):
        who = identity(attempt_id, rollout_id)
        attempt = AttemptLease(who, built, affinity_key=affinity_key("run-ns", who))
        await attempt.generate("main", prompt(), SamplingParams(max_tokens=1), conversation_id="c")
    keys = [request.headers["x-baseten-session-id"] for request in requests]
    assert keys[0] == keys[1] != keys[2]
    assert all(key.startswith("run-ns:") and "caf" not in key for key in keys)


def publishing_wiring(capacity, first, published):
    """Wiring whose ``publish()`` swaps the handle and its version, as the trainer does."""
    current = [first, 1]
    built, captured, _ = wiring(
        None,
        capacity=capacity,
        sampler=lambda: current[0],
        sampler_version=lambda: current[1],
    )

    def publish():
        current[:] = [published, 2]

    return built, captured, publish


async def test_a_dedicated_attempt_keeps_the_handle_it_first_sampled_with():
    """A publish must not raise an in-flight episode's min-policy-version floor: the
    paired sampler hot-swaps behind the handle the attempt already holds."""
    first, published = Sampler(), Sampler()
    built, _, publish = publishing_wiring(Capacity.DEDICATED, first, published)
    attempt = AttemptLease(identity(), built, affinity_key=None)
    await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    publish()
    await attempt.generate("main", prompt(2), PARAMS, conversation_id="c")
    assert len(first.requests) == 2 and not published.requests


async def test_a_serverless_attempt_moves_to_each_new_publication():
    """A serverless handle names one adapter, so a publish mid-attempt moves the next
    generation to the new handle, at the version it was published at."""
    first, published = Sampler(version=None), Sampler(version=None)
    built, captured, publish = publishing_wiring(Capacity.SERVERLESS, first, published)
    attempt = AttemptLease(identity(), built, affinity_key=None)
    await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    publish()
    await attempt.generate("main", prompt(2), PARAMS, conversation_id="c")
    assert (len(first.requests), len(published.requests)) == (1, 1)
    assert [sample.policy_version for _, sample in captured] == [1, 2]


@pytest.mark.parametrize("capacity", list(Capacity))
async def test_a_publish_during_a_generation_keeps_the_version_of_the_handle_that_served_it(
    capacity,
):
    """A pinned handle's samples take the version read with it; a publish while the
    request is in flight does not relabel them."""
    publish = None

    class Publishing(Sampler):
        async def sample_async(self, prompt, **kwargs):
            publish()
            return await super().sample_async(prompt, **kwargs)

    built, captured, publish = publishing_wiring(
        capacity, Publishing(version=None), Sampler(version=None)
    )
    attempt = AttemptLease(identity(), built, affinity_key=None)
    await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    assert [sample.policy_version for _, sample in captured] == [1]


async def test_async_capture_is_awaited_without_blocking_other_generations():
    capturing, release = asyncio.Event(), asyncio.Event()

    async def slow(identity, sample):
        capturing.set()
        await release.wait()

    attempt, _, _ = lease(on_sample=slow)
    other, _, _ = lease()
    task = asyncio.create_task(attempt.generate("main", prompt(), PARAMS, conversation_id="c"))
    await asyncio.wait_for(capturing.wait(), 1)
    await asyncio.wait_for(other.generate("main", prompt(), PARAMS, conversation_id="c"), 1)
    assert not task.done()
    release.set()
    await task


async def test_a_generation_that_raises_after_admission_poisons_the_lease_with_its_cause():
    class Broken(Sampler):
        async def sample_async(self, prompt, **kwargs):
            raise ConnectionError("sampler unreachable")

    attempt, _, _ = lease(Broken())
    with pytest.raises(ModelTransportFailure):
        await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    assert isinstance(attempt.failure, ModelTransportFailure)
    with pytest.raises(ModelTransportFailure, match="earlier generation"):
        await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    with pytest.raises(ModelTransportFailure):
        await attempt.seal()


def _sdk_errors():
    import httpx
    from baseten.loops.sampling_client import SamplerTerminatedError, SampleTimeoutError

    request = httpx.Request("POST", "http://sampler/sample")
    return [
        (httpx.ConnectError("refused", request=request), ModelTransportFailure),
        (
            httpx.HTTPStatusError(
                "503", request=request, response=httpx.Response(503, request=request)
            ),
            ModelTransportFailure,
        ),
        (SampleTimeoutError(30.0), SamplingFailure),
        # No special handling: a dead sampler fails every attempt, and the
        # consecutive-failure budget stops the run.
        (SamplerTerminatedError("dep", "UNHEALTHY"), SamplingFailure),
    ]


@pytest.mark.parametrize(("raised", "expected"), _sdk_errors())
async def test_sdk_errors_map_to_the_specified_failure_and_poison_the_lease(raised, expected):
    class Failing(Sampler):
        async def sample_async(self, prompt, **kwargs):
            raise raised

    attempt, _, _ = lease(Failing())
    with pytest.raises(expected) as caught:
        await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    assert caught.value.__cause__ is raised
    assert type(attempt.failure) is expected
    with pytest.raises(expected):
        await attempt.seal()


async def test_a_version_the_sampler_reports_is_kept_over_the_published_one():
    """A paired sampler serves whatever live adapter meets the floor, which can be newer
    than the publication the handle came from."""
    attempt, captured, _ = lease(Sampler(version=2), sampler_version=lambda: 1)
    await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    assert [sample.policy_version for _, sample in captured] == [2]


async def test_a_completion_longer_than_its_max_tokens_is_a_sampling_failure():
    """The trajectory budget holds only if the sampler stops at max_tokens; a longer
    completion is not evidence, whatever made it."""
    attempt, captured, _ = lease(Sampler(tokens=(97, 98, 99)))
    with pytest.raises(SamplingFailure, match=r"3 completion tokens, over the max_tokens \(2\)"):
        await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    assert captured == [] and type(attempt.failure) is SamplingFailure
    with pytest.raises(SamplingFailure):
        await attempt.seal()


async def test_a_sample_records_the_max_tokens_it_was_sampled_with_and_any_clamp():
    attempt, _, _ = lease()
    plain = await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    clamped = await attempt.generate("main", prompt(), PARAMS, conversation_id="c", clamped_from=10)
    unlimited = await attempt.generate("main", prompt(), SamplingParams(), conversation_id="c")
    assert (plain.max_tokens, plain.clamped_from) == (2, None)
    assert (clamped.max_tokens, clamped.clamped_from) == (2, 10)
    assert (unlimited.max_tokens, unlimited.clamped_from) == (None, None)


async def test_a_cancelled_generation_is_not_evidence_and_not_a_failure():
    """Whoever awaited it gave up on it, such as a deadline; the rest of the episode trains."""

    class HangsOnSecond(Sampler):
        calls = 0

        async def sample_async(self, prompt, **kwargs):
            self.calls += 1
            if self.calls == 2:
                await asyncio.Event().wait()
            return await super().sample_async(prompt, **kwargs)

    attempt, _, discarded = lease(HangsOnSecond())
    kept = await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    task = asyncio.create_task(attempt.generate("main", prompt(), PARAMS, conversation_id="c"))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert attempt.failure is None
    later = await attempt.generate("main", prompt(), PARAMS, conversation_id="c")  # admitted
    assert (await attempt.seal()).samples == (kept, later)
    assert discarded == []  # it never sampled, so there was nothing to drop


def _no_version(response):
    response.policy_version = None


def _no_logprobs(response):
    response.sequences[0].logprobs = None


def _nan_logprob(response):
    response.sequences[0].logprobs[0] = float("nan")


def _other_prompt(response):
    response.prompt_token_ids = [9]


def _two_sequences(response):
    response.sequences *= 2


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (_no_version, "no"),
        (_no_logprobs, "no"),
        (_nan_logprob, "finite"),
        (_other_prompt, "differs"),
        (_two_sequences, "exactly one"),
    ],
    ids=["no_version", "no_logprobs", "nan_logprob", "other_prompt", "two_sequences"],
)
async def test_unusable_sampler_evidence_poisons_the_lease_as_a_sampling_failure(edit, match):
    attempt, _, _ = lease(Tampered(edit))
    with pytest.raises(SamplingFailure, match=match):
        await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    with pytest.raises(SamplingFailure):
        await attempt.seal()


def echoing(echo):
    """A sampler that reports ``echo(submitted ids)`` as the prompt it saw."""

    def edit(response):
        response.prompt_token_ids = echo(response.prompt_token_ids)

    return Tampered(edit)


LONG = tuple(range(10, 1_010))


@pytest.mark.parametrize(
    "echo",
    [
        lambda ids: ids[:-1],
        lambda ids: [*ids, 7],
        lambda ids: [7, *ids[1:]],
        lambda ids: [*ids[:-1], 7],
        lambda ids: [*ids[:-64], 7, *ids[-63:]],
    ],
    ids=["shorter", "longer", "first", "last", "edge"],
)
async def test_a_text_echo_that_differs_in_length_or_at_either_end_fails(echo):
    attempt, _, _ = lease(echoing(echo))
    with pytest.raises(SamplingFailure, match="differs"):
        await attempt.generate("main", prompt(*LONG), PARAMS, conversation_id="c")


class Unwalkable(list):
    def __iter__(self):
        raise AssertionError("the echo was walked in full")


async def test_a_text_echo_is_checked_without_walking_it():
    attempt, _, _ = lease(echoing(Unwalkable))
    sample = await attempt.generate("main", prompt(*LONG), PARAMS, conversation_id="c")
    assert sample.prompt_ids == LONG


@pytest.mark.parametrize(
    "echo", [[5, 6, 8], [5, 6], [5, 6, 7, 8], [5, -6, 7], [5, 6, 2**40], [5, "6", 7], []]
)
async def test_an_echo_that_is_not_the_prompt_tuple_is_refused(echo):
    attempt, _, _ = lease(echoing(lambda _: echo))
    with pytest.raises(SamplingFailure, match="differs"):
        await attempt.generate("main", prompt(5, 6, 7), PARAMS, conversation_id="c")


@pytest.mark.parametrize("echo", [[5, 6, 7], (5, 6, 7), [5.0, 6, 7]])
async def test_an_echo_equal_to_the_prompt_tuple_is_accepted(echo):
    """Compared as ``tuple(echo) == prompt.ids`` always was, without building the tuple."""
    attempt, _, _ = lease(echoing(lambda _: echo))
    sample = await attempt.generate("main", prompt(5, 6, 7), PARAMS, conversation_id="c")
    assert sample.prompt_ids == (5, 6, 7)


async def test_a_future_policy_version_is_fatal_to_the_run():
    attempt, _, _ = lease(Sampler(version=3))
    with pytest.raises(FuturePolicyVersion):
        await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    assert attempt.fatal is not None
    with pytest.raises(FuturePolicyVersion):
        await attempt.seal()


@pytest.mark.parametrize(
    "freshness,versions,error",
    [
        (
            Freshness(max_steps_off_policy=0, max_policy_delta_in_trajectory=5),
            (1,),
            StaleRolloutError,
        ),
        (
            Freshness(max_steps_off_policy=5, max_policy_delta_in_trajectory=0),
            (1, 2),
            InternalPolicyDeltaError,
        ),
    ],
)
async def test_freshness_is_enforced_during_the_episode(freshness, versions, error):
    sampler = Sampler()
    attempt, _, _ = lease(sampler, freshness=freshness)
    with pytest.raises(error):
        for version in versions:
            sampler.version = version
            await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    assert isinstance(attempt.stale, error)
    with pytest.raises(error):
        await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    with pytest.raises(error):
        await attempt.seal()


async def test_an_untrainable_producer_cannot_make_a_rollout_stale():
    sampler = Sampler(version=0)
    attempt, _, _ = lease(
        sampler, freshness=Freshness(max_steps_off_policy=0, max_policy_delta_in_trajectory=0)
    )
    await attempt.generate("critic", prompt(), PARAMS, conversation_id="c")
    assert attempt.stale is None


async def test_a_request_still_in_flight_when_the_attempt_seals_stays_out_of_the_evidence():
    release = asyncio.Event()

    class Slow(Sampler):
        calls = 0

        async def sample_async(self, prompt, **kwargs):
            self.calls += 1
            if self.calls == 2:
                await release.wait()
            return await super().sample_async(prompt, **kwargs)

    attempt, _, _ = lease(Slow())
    kept = await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    task = asyncio.create_task(attempt.generate("main", prompt(), PARAMS, conversation_id="c"))
    await asyncio.sleep(0)
    assert (await attempt.seal()).samples == (kept,)
    release.set()
    with pytest.raises(CaptureIntegrityFailure, match="sealed"):  # its caller is told
        await task


async def test_revocation_rejects_later_requests_and_is_idempotent():
    attempt, _, _ = lease()
    attempt.revoke()
    attempt.revoke()
    assert attempt.state is LeaseState.REVOKED
    with pytest.raises(CaptureIntegrityFailure, match="revoked"):
        await attempt.generate("main", prompt(), PARAMS, conversation_id="c")


async def test_a_discarded_generation_is_excluded_from_sealed_evidence():
    attempt, _, discarded = lease()
    kept = await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    rejected = await attempt.generate("main", prompt(2), PARAMS, conversation_id="c")
    attempt.discard("main", rejected.sample_id)
    assert discarded == [("discarded", attempt.identity, rejected)]
    assert (await attempt.seal()).samples == (kept,)


async def test_a_client_cannot_discard_another_producers_generation():
    attempt, _, _ = lease()
    sample = await attempt.generate("main", prompt(), PARAMS, conversation_id="c")
    with pytest.raises(ModelAccessError):
        attempt.discard("critic", sample.sample_id)
    with pytest.raises(ModelAccessError):
        attempt.discard("main", "unknown")


# --------------------------------------------------------- transport delivery


async def _for_delivery(attempt, conversation="c"):
    return await attempt.generate_for_delivery(
        "main", prompt(1, 2), PARAMS, conversation_id=conversation
    )


async def test_a_transport_sample_stays_unsettled_until_it_is_delivered():
    attempt, captured, _ = lease()
    sample = await _for_delivery(attempt)
    assert captured == [(attempt.identity, sample)]  # captured before delivery
    attempt.delivered(sample.sample_id)
    assert (await attempt.seal()).samples == (sample,)


async def test_only_a_delivered_transport_sample_is_evidence():
    """Undelivered at sealing, a sampled response is dropped, never a failure."""
    attempt, _, discarded = lease()
    delivered = await _for_delivery(attempt, conversation="a")
    attempt.delivered(delivered.sample_id)
    dropped = await _for_delivery(attempt, conversation="b")
    assert attempt.failure is None
    later = await _for_delivery(attempt, conversation="c")  # later requests are still admitted
    assert (await attempt.seal()).samples == (delivered,)
    assert discarded == [
        ("undelivered", attempt.identity, dropped),
        ("undelivered", attempt.identity, later),
    ]


async def test_settling_a_generation_from_another_attempt_is_refused():
    attempt, _, _ = lease()
    other, _, _ = lease()
    sample = await _for_delivery(other)
    with pytest.raises(ModelAccessError, match="not sampled by this attempt"):
        attempt.delivered(sample.sample_id)


async def test_a_direct_generation_is_delivered_when_it_returns():
    attempt, _, _ = lease()
    sample = await attempt.generate("main", prompt(1, 2), PARAMS, conversation_id="c")
    attempt.delivered(sample.sample_id)  # idempotent
    assert (await attempt.seal()).samples == (sample,)


@pytest.mark.parametrize("close", ["seal", "revoke"])
async def test_close_listeners_run_once_when_admission_stops(close):
    attempt, _, _ = lease()
    calls = []
    attempt.on_close(lambda: calls.append(close))
    if close == "seal":
        await attempt.seal()
    else:
        attempt.revoke()
    assert calls == [close]
    attempt.revoke()
    assert calls == [close]
    attempt.on_close(lambda: calls.append("late"))  # a closed lease calls back at once
    assert calls == [close, "late"]


# ------------------------------------------------------ a seal and its transcript


def _keeping(write: Gate, kept: list):
    """A lease that records transcripts, each kept through ``write``."""

    async def keep(identity, transcript):
        await write.passage()
        kept.append(transcript)

    return lease(records_transcripts=True, on_transcript=keep)


def _cancelling_the_next_task(*, and_its_creator: bool):
    """Cancel the next task created before it starts, as a shutdown that cancels every
    task can; with ``and_its_creator``, the task that creates it too."""
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()

    def factory(loop, coro, **options):
        loop.set_task_factory(previous)
        task = asyncio.Task(coro, loop=loop, **options)
        task.cancel()
        if and_its_creator:
            creator = asyncio.current_task()
            assert creator is not None
            creator.cancel()
        return task

    loop.set_task_factory(factory)


# A seal outlasts its write and never cancels it, so the write has no response to
# choose: a cancellation starts the stop, then the write ends, is cancelled again, or
# fails. Or the write is cancelled before it starts, by the shutdown that cancels the
# seal: the seal's own cancellation is then the only one it has.
_SEAL_CASES = [
    *cases(
        triggers=(Trigger.CANCEL,),
        points={"start": (Landing.NOTHING,), "write": ALL_LANDINGS},
        responses=(Response.STOPS,),
    ),
    *cases(
        triggers=(Trigger.CANCEL,),
        points={"start": (Landing.FAILURE,), "write": (Landing.NOTHING,)},
        responses=(Response.STOPS,),
    ),
]


@pytest.mark.parametrize("case", _SEAL_CASES, ids=[case.id for case in _SEAL_CASES])
async def test_a_seal_cancelled_while_keeping_its_transcript_settles_then_is_cancelled(case):
    write, kept = Gate(OSError("no space left on device")), []
    attempt, _, dropped = _keeping(write, kept)
    undelivered = await _for_delivery(attempt)

    async def seal():
        try:
            return await attempt.seal()
        except BaseException as error:  # noqa: BLE001 - the outcome under test, as raised
            return error

    sealing = asyncio.create_task(seal())
    lost = case.at("start") is Landing.FAILURE
    async with asyncio.timeout(5):
        if lost:
            write.open(Landing.NOTHING)  # for the write that follows the lost one
            _cancelling_the_next_task(and_its_creator=True)
        else:
            await write.entered.wait()
            sealing.cancel()
            await settle_loop()
            if case.at("write") is Landing.CANCEL:
                sealing.cancel()
                await settle_loop()
            write.open(case.at("write"))
        outcome = await sealing

    assert isinstance(outcome, asyncio.CancelledError), outcome  # the shutdown goes on
    assert outcome.__cause__ is None  # a lost write is logged, never the attempt's failure
    # The cancellation aborts the attempt: what it leaves says so, never that it sealed.
    if case.at("write") is Landing.FAILURE:
        assert kept == []  # the disk is still full for the second write
    else:
        assert len(kept) == (1 if lost else 2)
        assert [transcript.sealed for transcript in kept] == ([False] if lost else [True, False])
    # Reported whatever became of the write: what was dropped does not depend on it.
    assert dropped == [("undelivered", attempt.identity, undelivered)]


@pytest.mark.parametrize("cancelled", ["before_it_starts", "while_it_runs"])
async def test_a_transcript_write_cancelled_on_its_own_is_logged_and_the_seal_returns(
    cancelled, caplog
):
    """Nothing cancels a seal's write but the write, or what cancels it alone: its
    CancelledError is a lost write, not a cancellation of whoever seals."""

    async def keep(identity, transcript):
        raise asyncio.CancelledError

    attempt, _, dropped = lease(records_transcripts=True, on_transcript=keep)
    undelivered = await _for_delivery(attempt)
    if cancelled == "before_it_starts":
        _cancelling_the_next_task(and_its_creator=False)
    sealed = await attempt.seal()
    assert sealed.transcript is not None
    assert "transcript was not written" in caplog.text
    assert dropped == [("undelivered", attempt.identity, undelivered)]


async def test_asking_direct_access_for_a_connection_states_the_runtimes_mode():
    from support.harness import Renderer

    from loops_cookbook.llm.client import Renderers

    held, _, _ = lease()
    with pytest.raises(ModelAccessError) as refused:
        await DirectModelAccess(held, Renderers(Renderer())).http_connection()
    message = str(refused.value)
    assert "declares direct" in message and "[rollout_runtime]" not in message
