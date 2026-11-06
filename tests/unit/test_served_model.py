"""What Loops serves, asked before anything launches: the model, whether it is enabled,
and how long a sequence it holds. Loops out of reach is reported, never refused."""

from __future__ import annotations

from typing import cast

import pytest
from baseten.loops import ServerCapabilities, ServiceClient
from support.fakes import CAPABILITIES_URL

from loops_cookbook.settings import ModelSettings
from loops_cookbook.trainer import CredentialsError, check_served_model

NEMOTRON = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"


def served(name, *, limit=262144, enabled_limit=None, enabled=True, details=None):
    return {
        "model_name": name,
        "max_seq_len": limit,
        "max_enabled_seq_len": limit if enabled_limit is None else enabled_limit,
        "max_context_length": limit,
        "supports_vision_language": False,
        "enabled": enabled,
        "enablement_details": details,
    }


SERVED = [
    served(NEMOTRON),
    served("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"),
    served("Qwen/Qwen3.5-9B", limit=131072),
    served("Qwen/Qwen3.5-4B", limit=131072),
]


@pytest.fixture
def loops(loops_capabilities, monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "key")
    monkeypatch.delenv("LOOPS_BASE_URL", raising=False)
    loops_capabilities.models = SERVED
    return loops_capabilities


async def test_a_served_model_reports_its_sequence_budget_and_the_enabled_limit(loops):
    line = await check_served_model(
        ModelSettings(lora_rank=32, base_model=NEMOTRON, max_seq_len=262144)
    )

    assert line == f"Loops model: {NEMOTRON}; max_seq_len 262144 (enabled up to 262144)"
    [request] = loops.requests
    assert (request.method, str(request.url)) == ("GET", CAPABILITIES_URL)
    assert request.headers["Authorization"] == "Api-Key key"


async def test_an_unset_sequence_budget_reports_the_models_limit(loops):
    line = await check_served_model(ModelSettings(lora_rank=32, base_model="Qwen/Qwen3.5-9B"))

    assert line == "Loops model: Qwen/Qwen3.5-9B; max_seq_len model default (enabled up to 131072)"


@pytest.mark.parametrize(
    ("name", "suggested"),
    [
        (
            "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
            f"{NEMOTRON}, nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
        ),
        ("Qwen/Qwen3.5-35B", "Qwen/Qwen3.5-9B, Qwen/Qwen3.5-4B"),
        ("nvidia/Llama-4", f"{NEMOTRON}, nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"),
    ],
    ids=["by spelling", "the same family", "the same organisation"],
)
async def test_a_model_loops_does_not_serve_is_refused_with_the_closest_it_does(
    loops, name, suggested
):
    with pytest.raises(ValueError, match="not a model Loops serves") as refused:
        await check_served_model(ModelSettings(lora_rank=32, base_model=name))

    assert repr(name) in str(refused.value)
    assert str(refused.value).endswith(f"closest: {suggested}")


async def test_a_model_with_no_near_name_is_refused_with_everything_loops_serves(loops):
    with pytest.raises(ValueError) as refused:
        await check_served_model(ModelSettings(lora_rank=32, base_model="acme/Unheard-Of"))

    assert str(refused.value).endswith(
        "it serves Qwen/Qwen3.5-4B, Qwen/Qwen3.5-9B, "
        f"{NEMOTRON}, nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
    )


async def test_a_listed_model_loops_has_not_enabled_is_refused_with_why(loops):
    loops.models = [
        served(
            NEMOTRON,
            enabled=False,
            details={
                "reason": "needs_approval",
                "reason_detail": "Needs hardware the workspace is not approved for.",
                "remediation": "Contact Baseten.",
            },
        )
    ]

    with pytest.raises(ValueError) as refused:
        await check_served_model(ModelSettings(lora_rank=32, base_model=NEMOTRON))

    assert str(refused.value).endswith(
        "is not enabled on Loops: Needs hardware the workspace is not approved for. "
        "Contact Baseten."
    )


async def test_a_listed_model_loops_has_not_enabled_is_refused_without_a_reason_too(loops):
    loops.models = [served(NEMOTRON, enabled=False)]

    with pytest.raises(ValueError) as refused:
        await check_served_model(ModelSettings(lora_rank=32, base_model=NEMOTRON))

    assert str(refused.value).endswith(f"base_model {NEMOTRON!r} is not enabled on Loops")


async def test_a_sequence_budget_above_the_enabled_limit_is_refused_with_both(loops):
    with pytest.raises(ValueError, match="max_seq_len 300000 is above the 262144 Loops enables"):
        await check_served_model(
            ModelSettings(lora_rank=32, base_model=NEMOTRON, max_seq_len=300000)
        )


async def test_a_sequence_budget_loops_supports_on_unapproved_hardware_says_whom_to_ask(loops):
    loops.models = [served(NEMOTRON, limit=262144, enabled_limit=131072)]

    with pytest.raises(ValueError) as refused:
        await check_served_model(
            ModelSettings(lora_rank=32, base_model=NEMOTRON, max_seq_len=200000)
        )

    assert str(refused.value) == (
        f"[model] max_seq_len 200000 is above the 131072 this workspace is enabled for on "
        f"{NEMOTRON}. Loops supports up to 262144 on hardware the workspace is not approved "
        "for: lower it, or contact Baseten to approve that hardware"
    )


async def test_a_supported_ceiling_below_the_enabled_one_still_holds(loops):
    """Both ceilings hold: an enabled ceiling above what Loops supports lets nothing more."""
    loops.models = [served(NEMOTRON, limit=131072, enabled_limit=262144)]

    with pytest.raises(ValueError, match="max_seq_len 200000 is above the 131072"):
        await check_served_model(
            ModelSettings(lora_rank=32, base_model=NEMOTRON, max_seq_len=200000)
        )
    line = await check_served_model(ModelSettings(lora_rank=32, base_model=NEMOTRON))
    assert line.endswith("(enabled up to 131072)")


async def test_an_older_server_without_the_enabled_ceiling_is_held_to_the_supported_one(loops):
    model = served(NEMOTRON, limit=131072)
    del model["max_enabled_seq_len"]
    loops.models = [model]

    line = await check_served_model(ModelSettings(lora_rank=32, base_model=NEMOTRON))

    assert line == f"Loops model: {NEMOTRON}; max_seq_len model default (enabled up to 131072)"


@pytest.mark.parametrize("status", [401, 403])
async def test_a_key_loops_rejects_is_refused(loops, status):
    loops.status = status

    with pytest.raises(CredentialsError, match=f"HTTP {status}"):
        await check_served_model(ModelSettings(lora_rank=32, base_model=NEMOTRON))


async def test_without_a_key_loops_is_not_asked(loops, monkeypatch):
    monkeypatch.delenv("BASETEN_API_KEY")

    line = await check_served_model(ModelSettings(lora_rank=32, base_model=NEMOTRON))

    assert line == f"Loops model: {NEMOTRON} not checked (BASETEN_API_KEY is not set)"
    assert not loops.requests


async def test_loops_out_of_reach_is_reported_not_refused(loops):
    loops.models = None  # ServiceClient() opens a session first, and that raises ConnectError

    line = await check_served_model(ModelSettings(lora_rank=32, base_model="acme/Unheard-Of"))

    assert line == (
        "Loops model: acme/Unheard-Of not checked "
        "(ConnectError from https://api.baseten.co/v1/loops/sessions)"
    )


async def test_a_given_service_client_is_asked_and_no_other_is_made(loops, monkeypatch):
    made = []
    monkeypatch.setattr("loops_cookbook.trainer.ServiceClient", made.append)

    class Service:
        async def get_server_capabilities_async(self):
            return ServerCapabilities.model_validate({"supported_models": SERVED})

    line = await check_served_model(
        ModelSettings(lora_rank=32, base_model=NEMOTRON), cast(ServiceClient, Service())
    )

    assert line == f"Loops model: {NEMOTRON}; max_seq_len model default (enabled up to 262144)"
    assert not made


async def test_an_unexpected_status_or_response_is_reported_not_refused(loops, monkeypatch):
    monkeypatch.setenv("LOOPS_BASE_URL", "https://loops.example/")
    loops.status = 503

    line = await check_served_model(ModelSettings(lora_rank=32, base_model=NEMOTRON))
    assert line.endswith("not checked (HTTP 503 from https://loops.example/v1/loops/capabilities)")

    loops.status, loops.models = 200, [{"name": NEMOTRON}]
    line = await check_served_model(ModelSettings(lora_rank=32, base_model=NEMOTRON))
    assert line.endswith("not checked (unrecognised response to GET /v1/loops/capabilities)")


async def test_opening_the_session_does_not_spend_the_capabilities_budget(loops, monkeypatch):
    """The timeout is the capabilities request's: a slow session open leaves it whole."""
    import time

    monkeypatch.setattr("loops_cookbook.trainer._CAPABILITIES_TIMEOUT", 0.2)

    class SlowSession:
        def __init__(self):
            time.sleep(0.3)  # the constructor's blocking session request

        async def get_server_capabilities_async(self):
            return ServerCapabilities.model_validate({"supported_models": SERVED})

    monkeypatch.setattr("loops_cookbook.trainer.ServiceClient", SlowSession)
    line = await check_served_model(ModelSettings(lora_rank=32, base_model=NEMOTRON))
    assert line == f"Loops model: {NEMOTRON}; max_seq_len model default (enabled up to 262144)"


async def test_a_client_whose_own_settings_are_wrong_fails_the_check(loops, monkeypatch):
    """A ServiceClient refusing its settings is the config's error, never 'not checked'."""

    def refuse():
        raise ValueError("LOOPS_TEAM 'reserch' is not a team this key can use")

    monkeypatch.setattr("loops_cookbook.trainer.ServiceClient", refuse)
    with pytest.raises(ValueError, match="reserch"):
        await check_served_model(ModelSettings(lora_rank=32, base_model=NEMOTRON))
