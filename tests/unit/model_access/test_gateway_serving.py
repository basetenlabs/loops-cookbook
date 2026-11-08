"""The Chat Completions gateway serves: the official client, sampling fields, credentials,
and what it refuses before sampling.

Every test runs the real gateway on loopback, a real attempt lease, and a scripted
sampler; nothing reaches the network.
"""

import base64
import json

import aiohttp
import openai
import pytest
from aiohttp import web
from renderers.base import ToolCallParseStatus
from support.fakes import ScriptedSampler

from loops_cookbook.llm.client import Renderers, SamplingPolicy
from loops_cookbook.llm.rendering import ParsedReply
from loops_cookbook.model_access.api import (
    ModelAccessError,
)
from loops_cookbook.model_access.openai.gateway import Gateway, GatewaySettings

from .rig import (
    MODEL,
    OPENED_CLIENTS,
    PNG,
    TOOLS,
    Renderer,
    _broken_call,
    attempt,
    client,
    connect,
    hello,
    raw,
)


async def test_the_official_client_completes_and_the_sample_is_the_attempts(gateway):
    access, held, _sampler = attempt(gateway)
    openai_client, _ = await client(access)
    reply = await openai_client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "hi"}]
    )
    assert reply.choices[0].message.content == "answer 1"
    assert reply.choices[0].finish_reason == "stop"
    assert reply.usage is not None and reply.usage.completion_tokens == 2
    (sample,) = (await held.seal()).samples
    assert reply.id == f"chatcmpl-{sample.sample_id}"
    assert sample.producer == "main"


async def test_a_streamed_response_arrives_whole_and_settles_after_its_termination(gateway):
    renderer = Renderer()
    renderer.reply = lambda n: ParsedReply(
        "answer",
        "because",
        (),
        True,
        {"role": "assistant", "content": "answer", "reasoning_content": "because"},
    )
    access, held, _ = attempt(gateway, renderer=renderer)
    openai_client, _ = await client(access)
    stream = await openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    content, reasoning, finish, usage = "", "", None, None
    async for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        for choice in chunk.choices:
            content += choice.delta.content or ""
            reasoning += (choice.delta.model_extra or {}).get("reasoning_content") or ""
            finish = choice.finish_reason or finish
    assert (content, reasoning, finish) == ("answer", "because", "stop")
    assert usage is not None and usage.total_tokens > 0
    assert len((await held.seal()).samples) == 1


async def test_omitted_sampling_fields_take_the_runs_defaults_and_stop_replaces_the_renderers(
    gateway,
):
    policy = SamplingPolicy(temperature=0.4, max_generation_tokens=12)
    access, _, sampler = attempt(gateway, sampling_policy=policy)
    openai_client, _ = await client(access)
    await openai_client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "a"}]
    )
    await openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "b"}],
        temperature=0.9,
        max_tokens=3,
        top_p=0.5,
        seed=11,
        stop=["END"],
    )
    first, second = sampler.params
    assert (first.temperature, first.max_tokens, first.stop) == (0.4, 12, [99])
    assert (second.temperature, second.max_tokens, second.top_p, second.seed, second.stop) == (
        0.9,
        3,
        0.5,
        11,
        ["END"],
    )


async def test_a_response_format_constrains_the_sample(gateway):
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    access, _, sampler = attempt(gateway)
    openai_client, _ = await client(access)
    await openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "a"}],
        response_format={"type": "json_schema", "json_schema": {"name": "a", "schema": schema}},
    )
    await openai_client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "b"}]
    )
    assert [params.structured_outputs for params in sampler.params] == [{"json": schema}, None]


async def test_the_model_list_names_exactly_the_advertised_model(gateway):
    access, _, _ = attempt(gateway)
    openai_client, connection = await client(access)
    models = await openai_client.models.list()
    assert [model.id for model in models.data] == [MODEL]
    assert (await openai_client.models.retrieve(MODEL)).id == MODEL
    status, body = await raw(connection, "/models/other", method="GET")
    assert status == 404 and body["error"]["code"] == "model_not_found"


async def test_unknown_paths_are_a_canonical_404(gateway):
    access, _, _ = attempt(gateway)
    connection = await connect(access)
    status, body = await raw(connection, "/embeddings", body={"input": "x"})
    assert status == 404
    assert (
        body["error"]["type"] == "invalid_request_error" and body["error"]["code"] == "unknown_url"
    )


@pytest.mark.parametrize("path", ["/chat/completions", "/models", "/loops/discard", "/anything"])
async def test_every_endpoint_requires_the_attempt_credential(gateway, path):
    access, _, sampler = attempt(gateway)
    connection = await connect(access)
    async with aiohttp.ClientSession() as session:
        async with session.post(connection.base_url + path, json=hello()) as response:
            assert response.status == 401
        async with session.post(
            connection.base_url + path, json=hello(), headers={"Authorization": "Bearer nope"}
        ) as response:
            assert response.status == 401
            assert (await response.json())["error"]["code"] == "invalid_api_key"
    assert sampler.prompts == []


@pytest.mark.parametrize("field", ["logprobs", "top_logprobs"])
async def test_logprobs_fail_with_the_canonical_error_before_sampling(gateway, field):
    access, held, sampler = attempt(gateway)
    connection = await connect(access)
    status, body = await raw(connection, body=hello(**{field: True if field == "logprobs" else 2}))
    assert status == 400
    assert body["error"] == {
        "message": body["error"]["message"],
        "type": "invalid_request_error",
        "param": field,
        "code": "unsupported_parameter",
    }
    assert sampler.prompts == [] and held.failure is None


async def test_a_malformed_body_fails_schema_validation_before_sampling(gateway):
    access, held, sampler = attempt(gateway)
    connection = await connect(access)
    status, body = await raw(connection, body={"model": MODEL, "messages": []})
    assert status == 400 and body["error"]["code"] == "invalid_value"
    assert sampler.prompts == [] and held.failure is None


async def test_another_model_is_not_found(gateway):
    access, _, sampler = attempt(gateway)
    connection = await connect(access)
    status, body = await raw(connection, body=hello(model="gpt-4o"))
    assert status == 404 and body["error"]["code"] == "model_not_found"
    assert sampler.prompts == []


async def test_request_level_template_options_are_rejected_unless_empty(gateway):
    access, _, _sampler = attempt(gateway)
    connection = await connect(access)
    assert (await raw(connection, body=hello(chat_template_kwargs={})))[0] == 200
    status, body = await raw(connection, body=hello(chat_template_kwargs={"enable_thinking": True}))
    assert status == 400 and body["error"]["param"] == "chat_template_kwargs"


@pytest.mark.parametrize("asked", [None, 10], ids=["run_default", "request"])
async def test_a_request_over_the_trajectory_budget_is_sampled_with_the_room_left(gateway, asked):
    """As Model APIs clamp max_tokens to the context: prompt (2) of a budget of 5
    leaves 3, and a generation that reaches it ends on "length"."""
    access, held, sampler = attempt(
        gateway,
        sampler=ScriptedSampler(stop_reason="length"),
        sampling_policy=SamplingPolicy(max_generation_tokens=4, max_trajectory_tokens=5),
    )
    connection = await connect(access)
    status, body = await raw(connection, body=hello(**({"max_tokens": asked} if asked else {})))
    assert status == 200 and body["choices"][0]["finish_reason"] == "length"
    assert sampler.params[0].max_tokens == 3
    (sample,) = (await held.seal()).samples
    assert (sample.max_tokens, sample.clamped_from) == (3, asked or 4)


async def test_a_clamped_generation_cut_inside_a_tool_call_is_tool_call_truncated(gateway):
    renderer = Renderer(reply=_broken_call(ToolCallParseStatus.INVALID_JSON))
    access, held, sampler = attempt(
        gateway,
        sampler=ScriptedSampler(stop_reason="length"),
        renderer=renderer,
        sampling_policy=SamplingPolicy(max_generation_tokens=4, max_trajectory_tokens=6),
    )
    connection = await connect(access)
    status, body = await raw(connection, body=hello(tools=TOOLS))  # a 3-token prompt
    assert status == 422 and body["error"]["code"] == "tool_call_truncated"
    assert sampler.params[0].max_tokens == 3
    (sample,) = (await held.seal()).samples
    assert sample.clamped_from == 4


async def test_a_prompt_that_fills_the_trajectory_budget_is_context_length_exceeded(gateway):
    access, held, sampler = attempt(
        gateway, sampling_policy=SamplingPolicy(max_generation_tokens=4, max_trajectory_tokens=2)
    )
    connection = await connect(access)
    status, body = await raw(connection, body=hello())
    assert status == 400 and body["error"]["code"] == "context_length_exceeded"
    assert sampler.prompts == [] and held.failure is None
    assert (await held.seal()).samples == ()


async def test_a_remote_image_url_is_refused_and_a_data_uri_reaches_the_renderer(gateway):
    renderer = Renderer()
    access, _, sampler = attempt(gateway, renderer=renderer)
    connection = await connect(access)
    remote = {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}
    status, body = await raw(
        connection, body=hello(messages=[{"role": "user", "content": [remote]}])
    )
    assert status == 400 and body["error"]["param"] == "messages[0].content[0].image_url.url"
    assert sampler.prompts == []
    uri = "data:image/png;base64," + base64.b64encode(PNG).decode()
    part = {"type": "image_url", "image_url": {"url": uri}}
    status, _ = await raw(connection, body=hello(messages=[{"role": "user", "content": [part]}]))
    assert status == 200
    ((messages, _),) = renderer.renders
    assert messages[0]["content"] == [{"type": "image", "image": PNG}]


async def test_reasoning_effort_selects_a_declared_renderer_over_http(gateway):
    base, high = Renderer("base"), Renderer("high")
    access, _, _ = attempt(gateway, renderers=Renderers(base, {"high": high}))
    openai_client, connection = await client(access)
    await openai_client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "a"}], reasoning_effort="high"
    )
    assert len(high.renders) == 1 and not base.renders
    status, body = await raw(connection, body=hello(reasoning_effort="low"))
    assert status == 400 and body["error"]["param"] == "reasoning_effort"
    assert body["error"]["code"] == "unsupported_parameter"


async def test_the_gateway_listens_on_its_configured_bind(monkeypatch):
    import socket

    from aiohttp import web

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    hosts = []
    real = web.TCPSite

    def recording(runner, host, bound, **options):
        hosts.append((host, bound))
        return real(runner, host, bound, **options)

    monkeypatch.setattr(web, "TCPSite", recording)
    served = Gateway(GatewaySettings(f"127.0.0.1:{port}", None, MODEL, shutdown_seconds=1.0))
    try:
        assert await served.base_url() == f"http://127.0.0.1:{port}/v1"
        assert hosts == [("127.0.0.1", port)]
    finally:
        await served.close()


async def test_a_server_fault_is_logged_and_its_detail_kept_from_the_agent(gateway, caplog):
    renderer = Renderer()

    def broken(n):
        raise RuntimeError("internal detail: /srv/keys/agent.pem")

    renderer.reply = broken
    access, _, _ = attempt(gateway, renderer=renderer)
    connection = await connect(access)
    status, body = await raw(connection, body=hello())
    assert status == 500 and "internal detail" not in json.dumps(body)
    assert body["error"]["type"] == "server_error"
    assert "internal detail" in caplog.text


async def test_an_advertised_url_that_forwards_to_the_listener_keeps_attribution():
    """The seam a remote sandbox uses: agents reach the gateway through another origin."""
    served = Gateway(
        GatewaySettings("127.0.0.1:0", "http://127.0.0.1:1", MODEL, shutdown_seconds=1.0)
    )
    access, held, _ = attempt(served)
    advertised = (await access.http_connection()).base_url
    assert advertised == "http://127.0.0.1:1/v1"
    listener = served._runner.addresses[0][1]

    async def forward(request):
        async with (
            aiohttp.ClientSession() as session,
            session.request(
                request.method,
                f"http://127.0.0.1:{listener}{request.rel_url}",
                data=await request.read(),
                headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
            ) as upstream,
        ):
            return web.Response(
                status=upstream.status,
                body=await upstream.read(),
                content_type=upstream.content_type,
            )

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", forward)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    proxy = runner.addresses[0][1]
    try:
        connection = await access.http_connection()
        openai_client = openai.AsyncOpenAI(
            base_url=f"http://127.0.0.1:{proxy}/v1",
            api_key=connection.api_key.get_secret_value(),
            max_retries=0,
        )
        OPENED_CLIENTS.append(openai_client)
        reply = await openai_client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "hi"}]
        )
        (sample,) = (await held.seal()).samples
        assert reply.id == f"chatcmpl-{sample.sample_id}"
    finally:
        await runner.cleanup()
        await served.close()


async def test_a_rendering_error_is_a_400_a_stock_client_does_not_retry(gateway):
    """Refused before sampling, so nothing is lost and a retry would fail the same way."""
    renderer = Renderer()
    renderer.bridge = lambda prompt, completion, messages, tools, **kw: (
        renderer.bridges.append(messages) or (0, *completion, 3)
    )
    access, held, sampler = attempt(gateway, renderer=renderer)
    connection = await access.http_connection()
    stock = openai.AsyncOpenAI(  # the SDK's default retries: 408, 409, 429 and 5xx
        base_url=connection.base_url, api_key=connection.api_key.get_secret_value()
    )
    OPENED_CLIENTS.append(stock)
    headers = {"X-Loops-Conversation-Id": "session"}
    first = await stock.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "hi"}], extra_headers=headers
    )
    history = [
        {"role": "user", "content": "hi"},
        first.choices[0].message.model_dump(exclude_none=True),
        {"role": "user", "content": "more"},
    ]
    with pytest.raises(openai.BadRequestError) as refused:
        await stock.chat.completions.create(model=MODEL, messages=history, extra_headers=headers)
    assert refused.value.status_code == 400 and refused.value.code == "rendering_failed"
    assert refused.value.type == "invalid_request_error"
    assert len(renderer.bridges) == 1  # asked once: the client did not retry
    assert len(sampler.prompts) == 1  # the broken bridge never sampled
    assert held.failure is None and len((await held.seal()).samples) == 1


async def test_a_schema_error_names_its_param_as_the_policy_does(gateway):
    access, _, _ = attempt(gateway)
    connection = await connect(access)
    status, body = await raw(
        connection, body={"model": MODEL, "messages": [{"role": "user", "content": 7}]}
    )
    assert status == 400 and body["error"]["param"].startswith("messages[0]")


async def test_asking_http_access_for_a_client_states_the_runtimes_mode(gateway):
    access, _, _ = attempt(gateway)
    with pytest.raises(ModelAccessError) as refused:
        access.llm()
    message = str(refused.value)
    assert "declares openai_http" in message and "[rollout_runtime]" not in message
