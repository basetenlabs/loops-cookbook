"""Fixed reward; the target is never selected using baseline observations."""

from pathlib import Path


def score(answer: str) -> float:
    return float(answer.strip() == "B")


if __name__ == "__main__":
    answer = Path("/app/answer.txt").read_text()
    output = Path("/logs/verifier/reward.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(str(score(answer)))
