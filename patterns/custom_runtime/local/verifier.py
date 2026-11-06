"""The example's verifier: a pure function of the episode's result, run as a process.

It scores the answer the agent left either in its result file
(``VERIFY_AGAINST=files``, reading ``result.json`` in the working directory the
agent and the verifier share) or in its sandbox (``VERIFY_AGAINST=sandbox``,
reading ``answer.txt`` through ``sandbox_io``), against ``EXPECTED_ANSWER``. It
prints ``{"reward": <float>, "answered": <bool>}``: ``answered`` is false when the
agent left no answer.

An agent that left no answer, because it failed or never got that far, scores 0;
how the agent program exited does not enter into it. A result it cannot parse is
not a measurement: it exits with ``INVALID_MEASUREMENT``, which the runtime reports
as a permanent grading failure. Any other failure is worth retrying.
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

    Only parsing ``result.json`` can make it ``Unparseable``: a failed sandbox read
    raises as it is, so the runtime retries it.
    """
    try:
        if os.environ["VERIFY_AGAINST"] == "sandbox":
            return sandbox_io.read("answer.txt")
        data = pathlib.Path("result.json").read_bytes()
    except FileNotFoundError:
        return None
    try:
        result = json.loads(data.decode())
    except ValueError as error:  # UnicodeDecodeError is one
        raise Unparseable(f"result.json is not JSON text: {error}") from None
    if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
        raise Unparseable("result.json has no answer")
    return result["answer"]


def main() -> int:
    try:
        given = answer()
    except Unparseable as error:  # not a measurement
        print(f"nothing to score: {error}", file=sys.stderr)
        return INVALID_MEASUREMENT
    correct = given is not None and given.strip() == os.environ["EXPECTED_ANSWER"]
    print(json.dumps({"reward": float(correct), "answered": given is not None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
