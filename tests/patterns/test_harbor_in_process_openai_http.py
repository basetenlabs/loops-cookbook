"""The Harbor in-process openai_http pattern: the direct pattern, reaching the model over HTTP."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from harbor.models.job.config import JobConfig
from pydantic import SecretStr

from loops_cookbook.harbor.runtime import _validate_agent_binding
from loops_cookbook.model_access.api import ModelAccessMode, ModelConnection
from loops_cookbook.runtime import AttemptServices

ROOT = Path(__file__).resolve().parents[2]
PATTERN = ROOT / "patterns" / "harbor" / "in_process_openai_http"
DIRECT = ROOT / "patterns" / "harbor" / "in_process_direct"


@pytest.mark.parametrize("suffix", ["", "_blaxel"], ids=["docker", "blaxel"])
def test_each_job_is_the_direct_patterns_with_the_http_agent(suffix):
    job = JobConfig.model_validate(yaml.safe_load((PATTERN / f"job{suffix}.yaml").read_text()))
    same = JobConfig.model_validate(yaml.safe_load((DIRECT / f"job{suffix}.yaml").read_text()))
    assert _validate_agent_binding(job.agents[0]) is ModelAccessMode.OPENAI_HTTP
    assert job.environment == same.environment and job.tasks == same.tasks


async def test_the_agent_calls_the_attempts_endpoint_with_an_openai_client(tmp_path, monkeypatch):
    import http_agent

    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content=" B\n")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class Client:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.chat = SimpleNamespace(completions=Completions())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(http_agent, "AsyncOpenAI", Client)
    connection = ModelConnection("http://127.0.0.1:9/v1", SecretStr("attempt-key"), "policy")

    async def http_connection(producer=None):
        return connection

    access = SimpleNamespace(mode="openai_http", http_connection=http_connection)
    services = AttemptServices(
        model_access=access, capacity=None, resources=None, processes=None, metrics=None
    )  # pyright: ignore[reportArgumentType]

    class Environment:
        async def exec(self, command):
            assert command == "mkdir -p /app && printf %s B > /app/answer.txt"
            return SimpleNamespace(return_code=0)

    agent = http_agent.HttpAgent(logs_dir=tmp_path)
    agent.bind_loops_services(services)
    await agent.run("Pick a label", Environment(), None)
    assert calls == [
        {"base_url": "http://127.0.0.1:9/v1", "api_key": "attempt-key"},
        {"model": "policy", "messages": [{"role": "user", "content": "Pick a label"}]},
    ]
