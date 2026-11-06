"""History editing must not change the conditioning or count of sampled actions."""

import pytest
from support.builders import make_rollout
from support.fakes import ScriptedSampler
from support.leases import direct_llm

from loops_cookbook.llm.rendering import ParsedReply
from loops_cookbook.rl.grpo import build_rows


class Renderer:
    stop_sequences = ()

    def render(self, messages, tools):
        # A valid template can render reordered tool JSON or edited history
        # differently, even when some old token values recur in the result.
        return tuple(messages[0]["ids"])

    def parse(self, prompt, completion, tools):
        return ParsedReply(
            "answer", None, (), True, native_message={"role": "assistant", "content": "answer"}
        )


@pytest.mark.parametrize("replacement", [(9, 20, 21, 22, 7), (9, 22, 7), (10, 11, 20, 21, 22, 7)])
async def test_history_replacement_counts_each_sample_once_with_original_conditioning(replacement):
    sampler = ScriptedSampler([20, 21, 22], [-0.1, -0.2, -0.3], stop_reason="stop")
    llm = direct_llm(sampler, Renderer())
    chat = llm.conversation()
    first = await chat.generate(messages=[{"role": "user", "ids": [10, 11]}], max_tokens=3)
    second = await chat.generate(
        messages=[{"role": "assistant", "ids": list(replacement)}], max_tokens=3
    )
    assert first.sample.prompt_ids == (10, 11)
    assert second.sample.prompt_ids == replacement
    rows = build_rows(make_rollout("t", "g", "r", llm.samples), 1)
    assert sum(row.trained_positions for row in rows) == 6
    for sample in llm.samples:
        conditioned = sample.prompt_ids + sample.completion_ids
        matches = []
        for row in rows:
            inputs = row.datum.model_input.to_ints()
            targets = row.datum.loss_fn_inputs["target_tokens"].data
            actions = {p for start, stop in row.actions for p in range(start, stop)}
            for offset, token in enumerate(sample.completion_ids):
                position = len(sample.prompt_ids) - 1 + offset
                if (
                    inputs[: position + 1] == list(conditioned[: position + 1])
                    and targets[position] == token
                    and position in actions
                ):
                    matches.append(offset)
        assert matches == [0, 1, 2]
