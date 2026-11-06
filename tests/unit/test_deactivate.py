"""``loops deactivate``: end a Loops run by its ID or by the run directory that recorded it."""

from __future__ import annotations

import json

import httpx
import pytest

from loops_cookbook import command
from loops_cookbook.cli import TERMINATED_EXIT_CODE, Terminated


@pytest.fixture
def loops(monkeypatch, tmp_path):
    """Loops as the SDK reaches it for ``loops deactivate``: the session the client opens,
    then the deactivation, answered ``status``. ``requests`` keeps the deactivations."""

    class Loops:
        def __init__(self):
            self.status = 200
            self.body: dict = {}
            self.sessions = 0
            self.requests: list[httpx.Request] = []

        def answer(self, request):
            if request.url.path == "/v1/loops/sessions":
                self.sessions += 1
                return httpx.Response(200, json={"session": {"id": "session-1"}})
            self.requests.append(request)
            return httpx.Response(self.status, json=self.body)

    fake = Loops()
    transport = httpx.MockTransport(fake.answer)
    monkeypatch.setattr(
        "baseten.loops.service_client.make_sync_client",
        lambda **kwargs: httpx.Client(transport=transport, **kwargs),
    )
    monkeypatch.setattr(
        "baseten.loops._runs.make_async_client",
        lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs),
    )
    monkeypatch.setenv("BASETEN_API_KEY", "key")
    monkeypatch.delenv("LOOPS_BASE_URL", raising=False)
    # No .env from the repository: the working directory is the test's own.
    monkeypatch.chdir(tmp_path)
    return fake


def test_a_run_id_is_deactivated(loops, capsys):
    assert command.main(["deactivate", "abc123"]) == 0

    [request] = loops.requests
    assert (request.method, str(request.url)) == (
        "POST",
        "https://api.baseten.co/v1/loops/runs/abc123/deactivate",
    )
    assert request.headers["Authorization"] == "Api-Key key"
    assert "Deactivated Loops run abc123" in capsys.readouterr().out


def test_a_run_directory_names_its_run(loops, tmp_path):
    run_dir = tmp_path / "runs" / "counting-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"run_id": "xyz789", "url": None}))

    assert command.main(["deactivate", str(run_dir)]) == 0

    [request] = loops.requests
    assert str(request.url).endswith("/v1/loops/runs/xyz789/deactivate")


def test_a_resumed_sft_directory_ends_every_run_it_recorded(loops, tmp_path, capsys):
    """An SFT resume records its own Loops run in ``run-from-step-N.json`` beside the
    first one's ``run.json``; either may still be up."""
    run_dir = tmp_path / "runs" / "sft-1"
    run_dir.mkdir(parents=True)
    for name, run_id in [
        ("run.json", "first1"),
        ("run-from-step-12.json", "third3"),
        ("run-from-step-4.json", "second2"),
        ("run-from-step-20.json", "third3"),
    ]:
        (run_dir / name).write_text(json.dumps({"run_id": run_id}))

    assert command.main(["deactivate", str(run_dir)]) == 0

    assert [request.url.path.split("/")[4] for request in loops.requests] == [
        "first1",
        "second2",
        "third3",
    ]
    # Which runs the directory named is said before any of them is ended.
    assert "recorded Loops run(s) first1, second2, third3" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argument",
    ["runs/counting-1/run.json", "runs/missing", "../x", "../../x", "abc?x", "abc#x", "a/b"],
)
def test_an_argument_that_is_neither_a_run_id_nor_a_run_directory_is_refused_before_loops(
    loops, tmp_path, capsys, argument
):
    run_dir = tmp_path / "runs" / "counting-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"run_id": "xyz789"}))
    with pytest.raises(SystemExit) as error:
        command.main(["deactivate", argument])

    assert error.value.code == 2
    assert "is not a Loops run ID or a run directory" in capsys.readouterr().err
    assert (loops.sessions, loops.requests) == (0, [])


def test_a_directory_with_no_run_record_is_refused_before_loops(loops, tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit) as error:
        command.main(["deactivate", str(tmp_path / "empty")])

    assert error.value.code == 2
    assert "run.json" in capsys.readouterr().err
    assert loops.requests == []


@pytest.mark.parametrize(("status", "message"), [(403, "rejected the credentials"), (404, "404")])
def test_a_refused_deactivation_says_why(loops, capsys, status, message):
    loops.status = status
    loops.body = {"error": "run abc123 belongs to another workspace"}
    with pytest.raises(SystemExit) as error:
        command.main(["deactivate", "abc123"])

    assert error.value.code == 2
    err = capsys.readouterr().err
    assert message in err
    assert "belongs to another workspace" in err


def test_a_sigterm_while_loops_answers_exits_as_terminated(loops, monkeypatch):
    async def terminated(run_id):
        raise Terminated

    monkeypatch.setattr(command, "deactivate_run", terminated)
    assert command.main(["deactivate", "abc123"]) == TERMINATED_EXIT_CODE


def test_no_api_key_is_refused_before_loops(loops, monkeypatch, capsys):
    monkeypatch.delenv("BASETEN_API_KEY")
    with pytest.raises(SystemExit) as error:
        command.main(["deactivate", "abc123"])

    assert error.value.code == 2
    assert "BASETEN_API_KEY" in capsys.readouterr().err
    assert (loops.sessions, loops.requests) == (0, [])
