"""The in-process twins, locked identically: the same episode behind three model clients.

``direct_episode``, ``http_episode`` and the LiteLLM example's ``litellm_episode`` are
one episode with only the model call changed. Each case here runs against all three,
with the model's reply faked at the client each one uses: ``llm()``, a stock OpenAI
client, and LiteLLM's streaming ``acompletion``. LiteLLM is not a cookbook
dependency, so its cases run where it is installed.
"""

import asyncio
import contextlib
import importlib.util
from types import SimpleNamespace as NS

import pytest
from pydantic import SecretStr

from loops_cookbook.episodes import EpisodeRuntime, Task
from loops_cookbook.model_access.api import ModelConnection
from loops_cookbook.resources import SandboxGone
from loops_cookbook.runtime import GradeResult

NO_LITELLM = importlib.util.find_spec("litellm") is None


class Reply:
    """The model's one reply, ``text`` (``None`` for a reply with no content), or never.

    ``requests`` keeps what each twin's client was asked, as that client takes it.
    """

    def __init__(self, text=None, *, hang=False):
        self.text, self.hang, self.asked = text, hang, asyncio.Event()
        self.requests = []

    async def arrive(self, request):
        self.requests.append(request)
        self.asked.set()
        if self.hang:
            await asyncio.Event().wait()
        return self.text


def direct(monkeypatch, reply):
    """The model serves both direct hook calls and the runtime's model-access port."""
    import direct_episode

    class Conversation:
        async def generate(self, **kwargs):
            return NS(content=await reply.arrive(kwargs))

    client = NS(conversation=lambda **kwargs: Conversation())
    return direct_episode, NS(conversation=client.conversation, llm=lambda producer=None: client)


async def _connection(producer=None):
    return ModelConnection("http://127.0.0.1:9/v1", SecretStr("attempt-key"), "policy")


def http(monkeypatch, reply):
    """``http_episode``, with the stock OpenAI client it builds replying."""
    import http_episode

    class Client:
        def __init__(self, **connection):
            self.connection = connection
            self.chat = NS(completions=NS(create=self._create))

        async def _create(self, **request):
            content = await reply.arrive((self.connection, request))
            return NS(choices=[NS(message=NS(content=content))])

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(http_episode, "AsyncOpenAI", Client)
    return http_episode, NS(http_connection=_connection)


def litellm(monkeypatch, reply):
    """``litellm_episode``, with LiteLLM's ``acompletion`` streaming the reply by character."""
    import litellm_episode

    async def acompletion(**request):
        text = await reply.arrive(request)
        pieces = [None] if text is None else list(text)

        async def stream():
            for piece in pieces:
                yield NS(choices=[NS(delta=NS(content=piece))])

        return stream()

    monkeypatch.setattr(litellm_episode.litellm, "acompletion", acompletion)
    return litellm_episode, NS(http_connection=_connection)


TWINS = [
    pytest.param(direct, id="direct"),
    pytest.param(http, id="http"),
    pytest.param(
        litellm, id="litellm", marks=pytest.mark.skipif(NO_LITELLM, reason="needs litellm")
    ),
]
MESSAGES = [{"role": "user", "content": "Say B"}]
# What each twin asks its client for the task "Say B", over the attempt's connection.
ASKED = {
    direct: {"messages": MESSAGES},
    http: (
        {"base_url": "http://127.0.0.1:9/v1", "api_key": "attempt-key"},
        {"model": "policy", "messages": MESSAGES},
    ),
    litellm: {
        "model": "openai/policy",
        "api_base": "http://127.0.0.1:9/v1",
        "api_key": "attempt-key",
        "messages": MESSAGES,
        "stream": True,
    },
}


def episode(module, answer="B"):
    data = {"name": "b", "instruction": "Say B", "answer": answer}
    task = Task("0", data, module.AnswerTask.model_validate(data))
    return module.AnswerEpisode(task, None)


def attempt_with(model):
    """One attempt at task ``0``, whose model access is ``model``."""

    @contextlib.asynccontextmanager
    async def hold(phase):
        yield

    async def admit():
        return None

    return NS(
        spec=NS(task=NS(id="0"), identity=NS(attempt_id="a1")),
        services=NS(
            capacity=NS(admit=admit, hold=hold),
            resources=None,
            model_access=model,
            processes=None,  # the episode runs no process
            metrics=NS(observe=lambda name, value: None),
        ),
    )


@pytest.mark.parametrize("twin", TWINS)
@pytest.mark.parametrize(
    ("text", "reward"),
    [("B", 1.0), (" B\n", 1.0), ("C", 0.0), ("", 0.0), (None, 0.0)],
    ids=["right", "right_padded", "wrong", "empty", "no_content"],
)
async def test_the_reply_is_scored_against_the_tasks_answer_and_kept_out_of_the_rollout(
    monkeypatch, tmp_path, twin, text, reward
):
    reply = Reply(text)
    module, model = twin(monkeypatch, reply)
    runtime = EpisodeRuntime(
        module.AnswerEpisode,
        [episode(module).task],
        agent_timeout_seconds=5.0,
        rollout_timeout_seconds=15.0,
        sandbox=None,
        options=None,
        work_dir=tmp_path,
    )
    attempt = attempt_with(model)
    rollout = await runtime.sample(attempt)
    assert reply.requests == [ASKED[twin]]
    assert rollout.metadata == {}  # the answer lives on the episode, not in the rollout
    assert await runtime.grade(attempt, rollout) == GradeResult(reward=reward)
    await runtime.finish(attempt, rollout, None)


@pytest.mark.parametrize("twin", TWINS)
async def test_grading_again_as_a_retry_would_scores_the_same(monkeypatch, twin):
    module, model = twin(monkeypatch, Reply("B"))
    answering = episode(module)
    await answering.run(NS(model=model, sandbox=None))
    assert [await answering.grade(NS()) for _ in range(3)] == [1.0] * 3


@pytest.mark.parametrize("twin", TWINS)
async def test_a_run_cut_before_the_reply_scores_zero_and_closes_cleanly(monkeypatch, twin):
    reply = Reply("B", hang=True)
    module, model = twin(monkeypatch, reply)
    cut = episode(module)
    running = asyncio.create_task(cut.run(NS(model=model, sandbox=None)))
    await reply.asked.wait()
    running.cancel()  # the agent deadline, as the run applies it
    with pytest.raises(asyncio.CancelledError):
        await running
    assert await cut.grade(NS()) == 0.0
    await cut.close(NS())


SANDBOX_TWINS = TWINS[:2]  # the LiteLLM example has no sandbox variant


async def sandbox_episode(module, blaxel_sandbox):
    """A sandbox episode, set up on a fresh sandbox."""
    data = {"name": "b", "instruction": "Say B", "answer": "B"}
    task = Task("0", data, module.AnswerTask.model_validate(data))
    on = await blaxel_sandbox()
    episode = module.SandboxAnswerEpisode(task, None)
    await episode.setup(NS(sandbox=on))
    return episode, on


@pytest.mark.parametrize("twin", SANDBOX_TWINS)
async def test_on_a_sandbox_the_stripped_answer_is_written_and_graded(
    monkeypatch, twin, blaxel, blaxel_sandbox
):
    module, model = twin(monkeypatch, Reply(" B\n"))
    answering, on = await sandbox_episode(module, blaxel_sandbox)
    await answering.run(NS(model=model, sandbox=on))
    assert blaxel.files[on.name] == {module.ANSWER: "B"}
    assert [await answering.grade(NS(execution_sandbox=on)) for _ in range(2)] == [1.0] * 2
    blaxel.files[on.name][module.ANSWER] = "C"  # what is graded is what the sandbox holds
    assert await answering.grade(NS(execution_sandbox=on)) == 0.0
    assert blaxel.attached() == [await on.client()]  # one client, for every hook
    await answering.close(NS())


async def cut_on_a_sandbox(monkeypatch, twin, blaxel_sandbox):
    """A sandbox episode whose run() the agent deadline cut before it wrote anything."""
    reply = Reply("B", hang=True)
    module, model = twin(monkeypatch, reply)
    cut, on = await sandbox_episode(module, blaxel_sandbox)
    running = asyncio.create_task(cut.run(NS(model=model, sandbox=on)))
    await reply.asked.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    return module, cut, on


@pytest.mark.parametrize("twin", SANDBOX_TWINS)
async def test_on_a_sandbox_a_cut_run_that_wrote_nothing_scores_zero_once_the_sandbox_is_alive(
    monkeypatch, twin, blaxel, blaxel_sandbox
):
    _, cut, on = await cut_on_a_sandbox(monkeypatch, twin, blaxel_sandbox)
    assert await cut.grade(NS(execution_sandbox=on)) == 0.0
    assert blaxel.files[on.name] == {}
    await cut.close(NS())


@pytest.mark.parametrize("twin", SANDBOX_TWINS)
async def test_on_a_sandbox_that_is_not_alive_a_cut_runs_404_fails_grading_to_be_retried(
    monkeypatch, twin, blaxel, blaxel_sandbox
):
    _, cut, on = await cut_on_a_sandbox(monkeypatch, twin, blaxel_sandbox)
    blaxel.lose(on.name, "deleted")  # its file reads answer 404 all the same
    with pytest.raises(SandboxGone, match=on.name):
        await cut.grade(NS(execution_sandbox=on))
    await cut.close(NS())


@pytest.mark.parametrize("twin", SANDBOX_TWINS)
async def test_on_a_sandbox_a_run_cut_mid_write_scores_zero(monkeypatch, twin, blaxel_sandbox):
    module, model = twin(monkeypatch, Reply("B"))
    cut, on = await sandbox_episode(module, blaxel_sandbox)
    client = await on.client()
    client.stuck = True
    with pytest.raises(TimeoutError):  # the agent deadline, as the run applies it
        await asyncio.wait_for(cut.run(NS(model=model, sandbox=on)), 0.05)
    assert client.files == {}  # it answered, and wrote nothing
    assert await cut.grade(NS(execution_sandbox=on)) == 0.0
    await cut.close(NS())


@pytest.mark.parametrize("twin", SANDBOX_TWINS)
async def test_on_a_sandbox_any_other_failed_read_fails_grading_to_be_retried(
    monkeypatch, twin, blaxel_sandbox
):
    from blaxel.core.sandbox.types import ResponseError

    module, model = twin(monkeypatch, Reply("B"))
    answering, on = await sandbox_episode(module, blaxel_sandbox)
    await answering.run(NS(model=model, sandbox=on))
    (await on.client()).failing = 503  # a sandbox it cannot read is not a missing answer
    with pytest.raises(ResponseError):
        await answering.grade(NS(execution_sandbox=on))
    await answering.close(NS())


@pytest.mark.parametrize("twin", SANDBOX_TWINS)
async def test_close_is_the_layers_and_safe_before_any_attach(monkeypatch, twin, blaxel):
    module, _ = twin(monkeypatch, Reply("B"))
    await module.SandboxAnswerEpisode(episode(module).task, None).close(NS())
    await episode(module).close(NS())
    assert blaxel.attached() == []
