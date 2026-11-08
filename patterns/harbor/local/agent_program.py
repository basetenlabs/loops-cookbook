"""The program ``LocalAgent`` runs, which makes one model call and one tool call in the
sandbox.

It uses only the standard library, apart from the Blaxel SDK when it reaches a Blaxel
sandbox, so it runs in any interpreter that has that SDK. It reads these variables:

- ``OPENAI_BASE_URL``, ``OPENAI_API_KEY`` and ``OPENAI_MODEL`` give the model
  connection.
- ``AGENT_INSTRUCTION`` gives the task.
- ``BLAXEL_SANDBOX_NAME`` and Blaxel's ``BL_*`` credentials are the sandbox's usage
  environment.
"""

import asyncio
import json
import os
import urllib.request

# The task's verifier looks for the answer here.
ANSWER = "/app/answer.txt"


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


async def write_in_blaxel(path: str, content: str) -> None:
    from blaxel.core import SandboxInstance

    sandbox = await SandboxInstance.get(os.environ["BLAXEL_SANDBOX_NAME"])
    await sandbox.fs.write(path, content)


def main() -> None:
    answer = ask(os.environ["AGENT_INSTRUCTION"])
    # This write is the tool call. It puts the answer in the trial's sandbox, through
    # its provider.
    asyncio.run(write_in_blaxel(ANSWER, answer))
    print(f"wrote {len(answer)} characters to {ANSWER}")


if __name__ == "__main__":
    main()
