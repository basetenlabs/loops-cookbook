"""The agent program: one model call, one tool call in its sandbox, one result file.

It is a separate program the cookbook does not import, standard library only, so
it runs under any Python. It reads everything from its environment:

- ``OPENAI_BASE_URL``, ``OPENAI_API_KEY``, ``OPENAI_MODEL``: the attempt's model
  connection, through the run's gateway;
- ``AGENT_INSTRUCTION``: the task;
- its sandbox's usage environment, which ``sandbox_io`` reads: the sandbox is where
  it works, and it never creates or deletes it.

It writes ``result.json`` in its working directory, which is the attempt's own.
"""

import json
import os
import pathlib
import urllib.request

import sandbox_io


def ask(instruction: str) -> str:
    request = urllib.request.Request(
        os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions",
        data=json.dumps(
            {
                "model": os.environ["OPENAI_MODEL"],
                "messages": [{"role": "user", "content": instruction}],
            }
        ).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.load(response)
    return (body["choices"][0]["message"]["content"] or "").strip()


def main() -> None:
    answer = ask(os.environ["AGENT_INSTRUCTION"])
    # The tool call: the agent's work lands in its sandbox.
    sandbox_io.write("answer.txt", answer)
    # Written aside and renamed into place, so a kill mid-write leaves no result
    # rather than half of one, which the verifier could not score.
    partial = pathlib.Path("result.json.partial")
    partial.write_text(json.dumps({"answer": answer}))
    os.replace(partial, "result.json")


if __name__ == "__main__":
    main()
