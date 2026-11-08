"""Tabulate a renderer check run from its recorded transcripts and samples.

    uv run python examples/renderer_check/report.py runs/renderer-check/<run dir>

For every tool call the transcripts hold the raw text the model wrote, the arguments
the agent received and the call's status. The report counts, per tool and argument:

- the status of every call, and whether the turn's native message was kept
  (a turn with a non-ok call whose message was kept continued the episode);
- how each argument arrived: its JSON kind, and for strings whether it is byte
  identical to the raw text between the template's markers, including any leading or
  trailing whitespace, versus stripped or otherwise changed; a pair written in the
  raw text that the renderer did not deliver is reported as dropped;
- the rewards per task, so a renderer that mangles arguments shows up as failed tasks.

From the samples it checks the bridge: for consecutive turns of one conversation the
later prompt must start with the earlier prompt followed by the earlier completion.
"""

from __future__ import annotations

import html
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

# Each template's pairs are read from the call's raw text the way its renderer reads
# them. GLM writes <arg_key>k</arg_key><arg_value>v</arg_value>.
_GLM = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
# Qwen and Nemotron write <parameter=k>v</parameter> with one newline after the opener
# and one before the closer, which the renderer removes along with any whitespace after
# the ``=``; the value is what lies between those.
_XML = re.compile(r"<parameter=\s*([^>]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
# Muse Spark writes <atem:parameter name="k">v</atem:parameter> with the name HTML
# escaped; a value ends only at a closer followed by the next parameter or the
# invocation's end.
_MUSE = re.compile(
    r'<atem:parameter name="([^"]*)">(.*?)</atem:parameter>'
    r'(?=\s*(?:<atem:parameter name="|</atem:invoke>|\Z))',
    re.DOTALL,
)

# Qwen3.5 and Nemotron templates write these spellings; Qwen3.8 writes them at times.
_PYTHON_SCALARS = {"True": True, "False": False, "None": None}


def written(raw: str) -> dict[str, str]:
    if pairs := _GLM.findall(raw):
        return {key.strip(): value for key, value in pairs}  # the renderer strips a key
    if pairs := _XML.findall(raw):
        return dict(pairs)
    return {html.unescape(key): value for key, value in _MUSE.findall(raw)}


def kind(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    return "object"


def arrival(value: object, text: str | None) -> str:
    if text is None:
        return "raw text not found"
    if isinstance(value, str):
        if value == text:
            return "verbatim" + (" (edge whitespace)" if text != text.strip() else "")
        if value == text.strip():
            return "STRIPPED"
        return "CHANGED"
    try:
        wrote = json.loads(text)
    except ValueError:
        if text.strip() not in _PYTHON_SCALARS:
            return f"typed {kind(value)} from non-JSON text"
        wrote = _PYTHON_SCALARS[text.strip()]
    same = json.dumps(wrote, sort_keys=True) == json.dumps(value, sort_keys=True)
    return f"typed {kind(value)}" if same else f"typed {kind(value)} MISMATCH"


def main(run_dir: Path) -> None:
    statuses: Counter[tuple[str, str, bool]] = Counter()
    arrivals: Counter[tuple[str, str, str]] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    tool_names: Counter[str] = Counter()
    turns = calls = 0
    for path in sorted((run_dir / "transcripts").glob("*.json")):
        record = json.loads(path.read_text())
        for turn in record["turns"]:
            parsed = (turn.get("reply") or {}).get("parsed") or {}
            turn_calls = parsed.get("tool_calls") or []
            if not turn_calls:
                continue
            turns += 1
            kept = parsed.get("native_message") is not None
            for call in turn_calls:
                calls += 1
                name = call.get("name") or "<no name>"
                tool_names[name] += 1
                statuses[(name, str(call.get("status")), kept)] += 1
                arguments = call.get("arguments")
                if not isinstance(arguments, dict):
                    arrivals[(name, "<arguments>", f"not a mapping: {kind(arguments)}")] += 1
                    continue
                texts = written(call.get("raw") or "")
                for key in texts.keys() - arguments.keys():
                    arrivals[(name, key, "DROPPED")] += 1
                    if len(examples["DROPPED"]) < 5:
                        examples["DROPPED"].append(
                            f"{name}.{key}: raw={texts[key]!r} not delivered"
                        )
                for key, value in arguments.items():
                    how = arrival(value, texts.get(key))
                    arrivals[(name, key, how)] += 1
                    odd = how.isupper() or "MISMATCH" in how or "not found" in how
                    if odd and len(examples[how]) < 5:
                        examples[how].append(f"{name}.{key}: raw={texts.get(key)!r} got={value!r}")
    print(f"run: {run_dir.name}")
    print(f"turns with tool calls: {turns}; calls: {calls}")
    print("\nstatus per tool (native message kept?):")
    for (name, status, kept), count in sorted(statuses.items()):
        print(f"  {count:5d}  {name:14s} {status:20s} kept={kept}")
    print("\nhow arguments arrived:")
    for (name, key, how), count in sorted(arrivals.items()):
        print(f"  {count:5d}  {name}.{key}: {how}")
    for how, lines in examples.items():
        print(f"\n{how}:")
        for line in lines:
            print("  " + line)

    rollouts = run_dir / "rollouts.jsonl"
    if rollouts.exists():
        rewards: dict[str, list[float]] = defaultdict(list)
        for line in rollouts.read_text().splitlines():
            row = json.loads(line)
            rewards[str(row.get("task_id"))].append(float(row.get("reward") or 0.0))
        print("\nreward per task (mean over rollouts):")
        for task, values in sorted(
            rewards.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 0
        ):
            print(f"  task {task:3s}: {sum(values) / len(values):.2f} over {len(values)}")

    by_conversation: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((run_dir / "samples").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            by_conversation[row["conversation_id"]].append(row)
    pairs = sum(max(len(rows) - 1, 0) for rows in by_conversation.values())
    with_ids = [rows for rows in by_conversation.values() if rows and "prompt_ids" in rows[0]]
    if not with_ids:
        print(
            f"\nbridge: {pairs} turn pairs continue a conversation; this cookbook version "
            "records token counts, not ids, so the prefix check needs the Loops sequences"
        )
        return
    bridged = broken = 0
    first_break: str | None = None
    for rows in with_ids:
        rows.sort(key=lambda row: row["recorded_at"])
        for earlier, later in pairwise(rows):
            prefix = earlier["prompt_ids"] + earlier["completion_ids"]
            if later["prompt_ids"][: len(prefix)] == prefix:
                bridged += 1
            else:
                broken += 1
                if first_break is None:
                    position = next(
                        (i for i, (a, b) in enumerate(zip(prefix, later["prompt_ids"])) if a != b),
                        min(len(prefix), len(later["prompt_ids"])),
                    )
                    first_break = f"conversation {earlier['conversation_id']} at token {position} of {len(prefix)}"
    print(f"\nbridge: {bridged} turn pairs continue from the sampled tokens; {broken} do not")
    if first_break:
        print(f"  first break: {first_break}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
