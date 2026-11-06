"""Write data/train.jsonl and data/validation.jsonl from Agent-Ark/Toucan-1.5M.

Toucan is tool-agent trajectories over 495 real MCP servers (arXiv 2510.01179,
Apache-2.0). Its SFT config is in ms-swift's agent format; this is the importer that
turns it into the `chat` contract:

- an assistant message and the `tool_call` messages after it become one assistant
  message with `tool_calls`, and the `tool_response` messages answer them in order
  (no ids: the contract pairs the n-th result with the n-th call);
- every assistant turn trains;
- a fixed mix of Toucan's subsets: single-turn, rephrased, multi-turn, and
  `irrelevant`, where the right move is to answer without a tool;
- the split is by the set of tools offered, so no validation conversation shares its
  exact tool set with a training one;
- a conversation longer than MAX_TOKENS with the config's renderer is left out.

Run from the repository root; it downloads about 425 MB of parquet once:

    uv run --with pyarrow python examples/sft_toucan/make_data.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from loops_cookbook.sft import ChatExample, build_renderer
from loops_cookbook.sft.cli import load_config

HERE = Path(__file__).resolve().parent
DATA, RAW = HERE / "data", HERE / "data" / "raw"
SHARDS = [
    f"https://huggingface.co/api/datasets/Agent-Ark/Toucan-1.5M/parquet/SFT/train/{i}.parquet"
    for i in range(3)
]
MAX_TOKENS = 3584  # headroom under sft.toml's max_seq_len
SIZES = {"train": 2000, "validation": 200}
VALIDATION_SHARE = 0.10
MIX = {
    "single-turn-original": 0.35,
    "single-turn-diversify": 0.15,
    "irrelevant": 0.15,
    "multi-turn": 0.35,
}


class Skip(Exception):
    """A source row this importer leaves out, and why."""


def download() -> list[Path]:
    RAW.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, url in enumerate(SHARDS):
        path = RAW / f"toucan-sft-{index}.parquet"
        if not path.exists():
            print(f"downloading {url}", file=sys.stderr)
            urllib.request.urlretrieve(url, path)
        paths.append(path)
    return paths


def tool_call(content: str) -> dict:
    """A swift `tool_call` message's content: a Python-repr dict."""
    try:
        call = ast.literal_eval(content)
    except (ValueError, SyntaxError):
        raise Skip("tool call content is not a mapping") from None
    if not isinstance(call, dict) or not str(call.get("name") or "").strip():
        raise Skip("tool call without a name")
    arguments = call.get("arguments") or "{}"
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        raise Skip("tool call arguments are not JSON") from None
    if not isinstance(parsed, dict):
        raise Skip("tool call arguments are not an object")
    return {"type": "function", "function": {"name": call["name"], "arguments": parsed}}


def convert(row: dict) -> dict:
    source = json.loads(row["messages"])
    messages: list[dict] = []
    index = 0
    while index < len(source):
        role, content = source[index]["role"], source[index].get("content") or ""
        if role in ("system", "user"):
            messages.append({"role": role, "content": content})
            index += 1
        elif role in ("assistant", "tool_call"):
            message = {"role": "assistant", "content": content if role == "assistant" else ""}
            index += role == "assistant"
            calls = []
            while index < len(source) and source[index]["role"] == "tool_call":
                calls.append(tool_call(source[index]["content"]))
                index += 1
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
        elif role == "tool_response":
            messages.append({"role": "tool", "content": content})
            index += 1
        else:
            raise Skip(f"unknown role {role}")
    example = {
        "messages": messages,
        "tools": json.loads(row["tools"]) if row.get("tools") else [],
        "metadata": {
            "source": "Agent-Ark/Toucan-1.5M",
            "uuid": row["uuid"],
            "subset": row["subset_name"],
        },
    }
    if messages[-1]["role"] != "assistant":
        raise Skip("does not end on an assistant turn")
    try:
        ChatExample.model_validate(example)
    except ValueError as error:
        raise Skip(str(error).splitlines()[1].strip()) from None
    return example


def held_out(example: dict) -> bool:
    names = sorted(tool["function"]["name"] for tool in example["tools"] if "function" in tool)
    digest = hashlib.sha256("\n".join(names).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < VALIDATION_SHARE


def main() -> None:
    from pyarrow import parquet

    renderer = build_renderer(load_config(HERE / "sft.toml"))
    by_subset: dict[str, list[dict]] = defaultdict(list)
    for path in download():
        for row in parquet.read_table(path).to_pylist():
            by_subset[row["subset_name"]].append(row)
    rng = random.Random(0)
    for rows in by_subset.values():
        rng.shuffle(rows)

    out: dict[str, list[dict]] = {"train": [], "validation": []}
    skipped: Counter[str] = Counter()
    for subset, share in MIX.items():
        wanted = {split: round(size * share) for split, size in SIZES.items()}
        taken = Counter()
        for row in by_subset[subset]:
            if all(taken[split] >= count for split, count in wanted.items()):
                break
            try:
                example = convert(row)
            except Skip as reason:
                skipped[str(reason)] += 1
                continue
            split = "validation" if held_out(example) else "train"
            if taken[split] >= wanted[split]:
                continue
            parsed = ChatExample.model_validate(example)
            length = len(renderer.render_conversation(parsed.chat_messages(), example["tools"]).ids)
            if length > MAX_TOKENS:
                skipped[f"over {MAX_TOKENS} tokens"] += 1
                continue
            out[split].append(example)
            taken[split] += 1

    for split, examples in out.items():
        rng.shuffle(examples)
        with (DATA / f"{split}.jsonl").open("w") as handle:
            handle.writelines(
                json.dumps(example, ensure_ascii=False) + "\n" for example in examples
            )
        subsets = Counter(example["metadata"]["subset"] for example in examples)
        print(f"{split}: {len(examples)} conversations, {dict(subsets)}")
    print("left out (of the rows read before each subset filled):")
    for reason, count in skipped.most_common():
        print(f"  {count:6d}  {reason}")


if __name__ == "__main__":
    main()
