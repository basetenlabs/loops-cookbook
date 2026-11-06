"""What both recipes' trainers ask of Loops, and a valid SFT config, for the RL and SFT tests."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
from baseten.loops import ServerCapabilities
from baseten.loops.http_errors import raise_for_status_with_details
from baseten.loops.models import LoadWeightsResponse, TensorData

from support.builders import write_jsonl
from support.configs import SFT_MINIMAL, merged
from support.fakes import CAPABILITIES_URL, FakeClient, Future


class LoopsClient(FakeClient):
    """``FakeClient`` with the loads a start makes, per-datum logprobs, and a policy-version
    read a test can break.

    Each load records ``(method, path)`` in ``loads`` and leaves the trainer at
    ``restored``, the step Loops reports (``None``: the response names no step).
    ``read_version``, when set, answers every read of ``policy_version`` from the version
    the trainer holds: it may raise, block, record, or report another version.
    """

    def __init__(self, policy_version=0, *, restored=7):
        self.read_version = None
        super().__init__()
        self.policy_version = policy_version
        self.restored = restored
        self.loads: list[tuple[str, str]] = []

    @property
    def policy_version(self):
        if self.read_version is None:
            return self._version
        return self.read_version(self._version)

    @policy_version.setter
    def policy_version(self, value):
        self._version = value

    async def forward_backward_async(self, data, loss_fn, loss_fn_config=None):
        future = await super().forward_backward_async(data, loss_fn, loss_fn_config)
        result = await future.result_async()
        outputs = [
            {"logprobs": TensorData(data=[-1.0] * n, dtype="float32", shape=[n])}
            for n in (datum.model_input.length for datum in data)
        ]
        return Future(
            SimpleNamespace(loss=result.loss, metrics=result.metrics, loss_fn_outputs=outputs)
        )

    async def optim_step_async(self, params):
        self.adam_params = params
        self.optim_steps += 1
        self._version += 1
        # The trainer reports its step count with the step, as Loops does.
        return Future(
            SimpleNamespace(metrics={"grad_norm": 0.25, "step_seconds": 2.0, "step": self._version})
        )

    async def load_state_async(self, path):
        return self._load("load_state", path)

    async def load_state_with_optimizer_async(self, path):
        return self._load("load_state_with_optimizer", path)

    def _load(self, method, path):
        self.loads.append((method, path))
        if self.restored is None:
            return Future(LoadWeightsResponse())
        self._version = self.restored
        return Future(LoadWeightsResponse(step=self.restored))


class LoopsService:
    """``ServiceClient`` as a trainer's start sees it.

    It provisions ``client``; ``opened`` and ``provisioned`` keep the arguments the
    service and each training client were created with, and ``deactivated`` each run it
    ended (raising ``deactivate_error`` instead, when set). Its control plane answers
    the run record, whose ``deployment_id`` is ``None`` for a serverless run.
    """

    _base_url = "https://loops.test"

    def __init__(self, client):
        self.client = client
        self.opened: list[dict] = []
        self.provisioned: list[dict] = []
        self.deactivated: list[str] = []
        self.deactivate_error: Exception | None = None
        self.deployment_id: str | None = "7wljz0q"

    def _headers(self):
        return {}

    async def deactivate_run_async(self, run_id):
        if self.deactivate_error is not None:
            raise self.deactivate_error
        self.deactivated.append(run_id)

    def answer(self, request):
        run_id = request.url.path.removeprefix("/v1/loops/runs/")
        assert run_id != request.url.path, request.url
        run = {"id": run_id, "deployment_id": self.deployment_id, "status": {"name": "ACTIVE"}}
        return httpx.Response(200, json={"run": run}, request=request)

    def __call__(self, **kwargs):
        self.opened.append(kwargs)
        return self

    def create_lora_training_client(self, **kwargs):
        self.provisioned.append(kwargs)
        return self.client

    async def get_server_capabilities_async(self):
        """Asks the conftest's ``loops_capabilities`` fake, as the real client does."""
        async with httpx.AsyncClient() as client:
            response = await client.get(CAPABILITIES_URL)
        raise_for_status_with_details(response)
        return ServerCapabilities.model_validate(response.json())


def serve_loops(monkeypatch):
    """Make ``monkeypatch`` give a trainer's ``start`` a ``LoopsService`` with a ``LoopsClient``."""
    service = LoopsService(LoopsClient())
    monkeypatch.setattr("loops_cookbook.trainer.ServiceClient", service)
    network = httpx.AsyncHTTPTransport.handle_async_request

    async def handle_async(transport, request):
        if request.url.host == "loops.test":
            return service.answer(request)
        return await network(transport, request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async)
    return service


def chat_row(*turns, tools=None):
    """A user/assistant conversation from ``(user, assistant)`` pairs."""
    messages = []
    for user, assistant in turns:
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return {"messages": messages, **({"tools": tools} if tools else {})}


def sft_config(directory, rows=None, validation=None, **sections):
    """A valid SFT ``Config`` over a small chat dataset written to ``directory``; each
    keyword is a section's overrides."""
    from loops_cookbook.sft.config import Config

    train = write_jsonl(
        directory / "train.jsonl", rows or [chat_row((f"q{i}", f"a{i}")) for i in range(10)]
    )
    data = {"train": str(train)}
    if validation is not None:
        data["validation"] = str(write_jsonl(directory / "validation.jsonl", validation))
    values = merged(
        SFT_MINIMAL,
        {
            "model": {"max_seq_len": 256},
            "data": data,
            "training": {"batch_size": 4, "epochs": 2},
            "output": {"output_dir": str(directory / "runs")},
        },
    )
    for section, overrides in sections.items():
        values[section] = {**values.get(section, {}), **overrides}
    return Config.model_validate(values)
