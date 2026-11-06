"""Before and after: what the base model and a trained adapter do on held-out turns.

A smoke test for a checkpoint, not an eval framework. Validation NLL says the adapter
fits the data's tokens; this says whether its replies moved towards the data's. It
scores agreement with the reference turn (written by other models), on first turns
only, with no tool ever run and one greedy sample each -- so it cannot say whether an
agent finishes a task. For each validation conversation, the first assistant turn's prompt is
rendered with the config's renderer and sampled greedily from two standalone Loops
samplers -- the base model, and the adapter -- and each reply is parsed and scored
against the turn the data has:

- `decision`: called a tool when the reference does, answered directly when it does
  not (the `irrelevant` subset is all of the second kind);
- `tools`: the same tool names, in order, where the reference calls tools;
- `arguments`: the same names and the same arguments, where the reference calls tools;
- `well_formed`: the reply ended at a stop token and every call parsed.

    uv run python examples/sft_toucan/evaluate.py \\
        --adapter bt://loops:<run id>/sampler_weights/step-<N>

Both samplers are billable while up; they are deactivated on the way out. Results go
to evaluation.json beside this file, one record per conversation and the rates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

import httpx
from baseten.loops import ServiceClient
from baseten.loops.models import SamplingParams

from loops_cookbook.cli import config_directory, configure_logging, load_environment
from loops_cookbook.sft import ChatExample, build_renderer, load_rows
from loops_cookbook.sft.cli import load_config

HERE = Path(__file__).resolve().parent
SCORES = ("decision", "tools", "arguments", "well_formed")


def score(reference: dict, reply, stop_reason: str) -> dict[str, bool | None]:
    """``reference`` is the data's turn as the renderer takes it (arguments decoded)."""
    expected = [
        (call["function"]["name"], call["function"]["arguments"])
        for call in reference.get("tool_calls", [])
    ]
    produced = [(call.name, call.arguments) for call in reply.tool_calls]
    calls = bool(expected)
    return {
        "decision": calls == bool(produced),
        "tools": [name for name, _ in produced] == [name for name, _ in expected]
        if calls
        else None,
        "arguments": produced == expected if calls else None,
        "well_formed": stop_reason == "stop"
        and all(call.status.name == "OK" for call in reply.tool_calls),
    }


async def evaluate(label: str, sampler, renderer, cases, *, max_tokens: int) -> list[dict]:
    limit = asyncio.Semaphore(32)
    params = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, stop=list(renderer.stop_sequences)
    )

    async def one(case) -> dict:
        name, example, reference, prompt = case
        async with limit:
            response = await sampler.sample_async(prompt.prompt_input, 1, params)
        (sequence,) = response.sequences
        reply = renderer.parse(prompt.ids, sequence.tokens, example.tools)
        return {
            "model": label,
            "row": name,
            "subset": example.metadata.get("subset"),
            **score(reference, reply, sequence.stop_reason),
            "reply": {
                "content": reply.content,
                "tool_calls": [call.raw for call in reply.tool_calls],
            },
        }

    return await asyncio.gather(*(one(case) for case in cases))


def rates(records: list[dict]) -> dict[str, dict[str, float]]:
    """Per model, each score's rate over the conversations it applies to."""
    totals: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        for name in SCORES:
            if record[name] is not None:
                totals[record["model"]][name].append(record[name])
    return {
        model: {name: sum(values) / len(values) for name, values in scores.items()}
        for model, scores in totals.items()
    }


def deactivate(sampler) -> None:
    """Shut a standalone sampler down; say so loudly when that did not happen."""
    sampler_id = getattr(sampler, "_sampler_id", None)  # the SDK keeps it private
    try:
        response = httpx.post(
            f"https://api.baseten.co/v1/loops/samplers/{sampler_id}/deactivate",
            headers={"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"},
            timeout=60,
        )
        response.raise_for_status()
    except (httpx.HTTPError, KeyError) as error:
        print(f"Sampler {sampler_id} was NOT deactivated ({error}); it bills until you do.")
        return
    print(f"Deactivated sampler {sampler_id}.")


def provision(
    service: ServiceClient,
    max_seq_len: int,
    scope: str,
    *,
    base_model: str | None = None,
    model_path: str | None = None,
):
    """A standalone sampler. The SDK creates it, then waits for it in the same call, so
    a failure in that wait leaves one up that this script never got a handle to."""
    try:
        return service.create_sampling_client(
            base_model=base_model, model_path=model_path, max_seq_len=max_seq_len, scope=scope
        )
    except BaseException:
        print(
            "A sampler may have been provisioned before this failed: look for "
            "'Provisioned Loops sampler <id>' above and deactivate it "
            "(POST /v1/loops/samplers/<id>/deactivate)."
        )
        raise


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score the base model and an adapter on held-out turns."
    )
    parser.add_argument("--config", type=Path, default=HERE / "sft.toml")
    parser.add_argument("--adapter", required=True, help="bt://loops:<run>/sampler_weights/<name>")
    parser.add_argument("--limit", type=int, default=200, help="Validation conversations to score.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    # The SDK logs each sampler's id as it provisions it; that line is the way back to one
    # this script lost the handle to.
    configure_logging(verbose=False, style="plain")
    load_environment(config_directory(args.config))
    config = load_config(args.config)
    renderer = build_renderer(config)
    assert config.data.validation, "evaluate needs [data] validation"
    cases = []
    for row in load_rows(config.data.validation, config.data.format)[: args.limit]:
        example = row.example
        assert isinstance(example, ChatExample)
        messages = example.chat_messages()
        turn = next(k for k, message in enumerate(messages) if message["role"] == "assistant")
        prompt = renderer.render(messages[:turn], example.tools)
        cases.append((row.name, example, messages[turn], prompt))

    service = ServiceClient(availability_model="spot" if config.compute.spot_job else "standard")
    samplers = {}
    try:
        budget, scope = config.model.max_seq_len, config.compute.scope
        samplers["base"] = await asyncio.to_thread(
            provision, service, budget, scope, base_model=config.model.base_model
        )
        samplers["trained"] = await asyncio.to_thread(
            provision, service, budget, scope, model_path=args.adapter
        )
        records = [
            record
            for label, sampler in samplers.items()
            for record in await evaluate(
                label, sampler, renderer, cases, max_tokens=args.max_tokens
            )
        ]
    finally:
        for sampler in samplers.values():
            deactivate(sampler)

    summary = {"adapter": args.adapter, "conversations": len(cases), "rates": rates(records)}
    (HERE / "evaluation.json").write_text(json.dumps({**summary, "records": records}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
