"""The example's verifier, a pure function of the episode's result that runs as a process.

It scores the answer the agent left against ``EXPECTED_ANSWER``. With
``VERIFY_AGAINST=files``, it reads ``result.json`` in the working directory the agent
and the verifier share. With ``VERIFY_AGAINST=sandbox``, it reads ``answer.txt``
through ``sandbox_io``. It prints ``{"reward": <float>, "answered": <bool>}``, where
``answered`` is false when the agent left no answer.

An agent that left no answer, because it failed or never got that far, scores 0. How
the agent program exited doesn't matter. A result the verifier cannot parse is not a
measurement. The verifier then exits with ``INVALID_MEASUREMENT``, which the rollout
runtime reports as a permanent grading failure. Any other failure is worth retrying.
"""

import json
import os
import pathlib
import sys

import sandbox_io

INVALID_MEASUREMENT = 3


class Unparseable(ValueError):
    """The agent left a result file that is not a result."""


def answer() -> str | None:
    """The agent's answer, or ``None`` when it left none.

    Only parsing ``result.json`` can raise ``Unparseable``. A failed sandbox read raises
    as it is, so the rollout runtime retries it.
    """
    try:
        if os.environ["VERIFY_AGAINST"] == "sandbox":
            return sandbox_io.read("answer.txt")
        data = pathlib.Path("result.json").read_bytes()
    except FileNotFoundError:
        return None
    try:
        result = json.loads(data.decode())
    except ValueError as error:  # UnicodeDecodeError is a ValueError.
        raise Unparseable(f"result.json is not JSON text: {error}") from None
    if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
        raise Unparseable("result.json has no answer")
    return result["answer"]


def main() -> int:
    try:
        given = answer()
    except Unparseable as error:  # This result is not a measurement.
        print(f"nothing to score: {error}", file=sys.stderr)
        return INVALID_MEASUREMENT
    correct = given is not None and given.strip() == os.environ["EXPECTED_ANSWER"]
    print(json.dumps({"reward": float(correct), "answered": given is not None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
