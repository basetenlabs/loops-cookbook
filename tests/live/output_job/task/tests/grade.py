"""Half the reward for the hidden label, half for a command output that arrived whole."""

from pathlib import Path


def score(answer: str, output: str) -> float:
    return 0.5 * (answer.strip() == "B") + 0.5 * (output.strip() == "intact")


if __name__ == "__main__":
    answer = Path("/app/answer.txt").read_text()
    output = Path("/app/output.txt").read_text()
    reward = Path("/logs/verifier/reward.txt")
    reward.parent.mkdir(parents=True, exist_ok=True)
    reward.write_text(str(score(answer, output)))
