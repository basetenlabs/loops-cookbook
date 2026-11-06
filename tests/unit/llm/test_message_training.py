"""Exercise the user-facing message path through capture and real datum compilation."""

from support.builders import make_rollout
from support.fakes import ScriptedSampler
from support.leases import direct_llm

from loops_cookbook.llm.rendering import ParsedReply
from loops_cookbook.rl.grpo import build_rows


def routed_sampler():
    """Stop-terminated replies, each with its own record, as a recording sampler returns."""
    return ScriptedSampler(
        lambda call: [900 + call, 99],
        [-0.125, -0.25],
        stop_reason="stop",
        policy_version=2,
        sequence_id=lambda call: f"seq-{call}",
    )


class Formatting:
    stop_sequences = (99,)

    def __init__(self):
        self.initial_calls = 0

    def render(self, messages, tools):
        self.initial_calls += 1
        if len(messages) == 1:
            return (10, 11)
        return (10, 11, 901, 99, 20)

    def parse(self, prompt_ids, completion_ids, tools):
        # This readable view intentionally cannot reconstruct the sampled IDs.
        return ParsedReply(
            "a readable view, not the training evidence",
            None,
            (),
            True,
            native_message={"role": "assistant", "content": "decoded"},
        )


async def test_message_tool_turns_compile_one_datum_without_reencoding():
    sampler, renderer = routed_sampler(), Formatting()
    llm = direct_llm(sampler, renderer)
    chat = llm.conversation()
    history = [{"role": "user", "content": "task"}]
    first = await chat.generate(messages=history, max_tokens=2)
    history += [
        {"role": "assistant", "content": "decoded"},
        {"role": "tool", "content": "observation", "tool_call_id": "call"},
    ]
    second = await chat.generate(messages=history, max_tokens=2)
    assert first.content == second.content
    assert sampler.prompts == [[10, 11], [10, 11, 901, 99, 20]]
    assert renderer.initial_calls == 2
    rollout = make_rollout("task", "group", "rollout", llm.samples, reward=1)
    (row,) = build_rows(rollout, 0.5)
    datum = row.datum
    assert datum.model_input.to_ints() == [10, 11, 901, 99, 20, 902]
    assert datum.loss_fn_inputs["target_tokens"].data == [11, 901, 99, 20, 902, 99]
    assert datum.loss_fn_inputs["logprobs"].data == [0, -0.125, -0.25, 0, -0.125, -0.25]
    assert row.actions == ((1, 3), (4, 6))
    assert {span.sequence_id for span in datum.loss_fn_input_spans} == {"seq-1", "seq-2"}


async def test_message_single_fork_continuation_merges_exact_prefix_and_keeps_replay():
    sampler, renderer = routed_sampler(), Formatting()
    llm = direct_llm(sampler, renderer)
    history = [{"role": "user", "content": "task"}]
    await llm.conversation().generate(messages=history, max_tokens=2)
    branch_history = history + [
        {"role": "assistant", "content": "decoded"},
        {"role": "tool", "content": "branch observation"},
    ]
    await llm.conversation().generate(messages=branch_history, max_tokens=2)
    rows = build_rows(make_rollout("t", "g", "r", llm.samples, reward=1), 0.5)
    assert len(rows) == 1
    assert sum(row.trained_positions for row in rows) == 4
    assert rows[0].datum.loss_fn_inputs["target_tokens"].data == [11, 901, 99, 20, 902, 99]
    assert rows[0].routed


async def test_message_sibling_forks_mask_shared_parent_once_and_keep_replay():
    sampler, renderer = routed_sampler(), Formatting()
    llm = direct_llm(sampler, renderer)
    history = [{"role": "user", "content": "task"}]
    await llm.conversation().generate(messages=history, max_tokens=2)
    prefix = history + [{"role": "assistant", "content": "decoded"}]
    left_history = prefix + [{"role": "tool", "content": "left observation"}]
    right_history = prefix + [{"role": "tool", "content": "right observation"}]
    await llm.conversation().generate(messages=left_history, max_tokens=2)
    await llm.conversation().generate(messages=right_history, max_tokens=2)
    rows = build_rows(make_rollout("t", "g", "r", llm.samples, reward=1), 0.5)
    assert len(rows) == 2
    assert sum(row.trained_positions for row in rows) == 6
    assert rows[1].actions[0][0] >= 4
    assert all(row.routed for row in rows)
