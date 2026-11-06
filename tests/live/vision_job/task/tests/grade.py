"""Per-card credit: the cards read C, A, T in order."""

from pathlib import Path

WORD = "CAT"


def score(answer: str) -> float:
    word = "".join(answer.split()).upper()
    if len(word) != len(WORD):
        return 0.0
    return sum(a == b for a, b in zip(word, WORD)) / len(WORD)


if __name__ == "__main__":
    answer = Path("/app/answer.txt").read_text()
    output = Path("/logs/verifier/reward.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(str(score(answer)))
