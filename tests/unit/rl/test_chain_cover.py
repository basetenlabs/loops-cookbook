"""Chain packing preserves every action's conditioning, independently of labels."""

import random
from collections import Counter
from dataclasses import replace

from support.builders import make_rollout, make_sample

from loops_cookbook.evidence import TokenBuffers
from loops_cookbook.rl.grpo import build_rows


def test_seeded_branches_preserve_actions_conditioning_and_behavior_probabilities():
    rng = random.Random(51723)
    for _ in range(100):
        samples = []
        for index in range(20):
            if samples and rng.random() < 0.8:
                parent = rng.choice(samples)
                prompt = parent.prompt_ids + parent.completion_ids
            else:
                prompt = (rng.randrange(10),)
            prompt += tuple(rng.randrange(10) for _ in range(rng.randrange(3)))
            completion = tuple(1000 + index * 4 + offset for offset in range(rng.randint(1, 4)))
            samples.append(
                make_sample(
                    sample_id=str(index),
                    prompt_ids=prompt,
                    completion_ids=completion,
                    logprobs=tuple(-0.1 * (offset + 1) for offset in range(len(completion))),
                    policy_version=rng.randrange(4),
                    conversation_id=f"conversation-{rng.randrange(3)}",
                    trainable=index == 0 or rng.random() < 0.8,
                )
            )
        rollout = make_rollout("task", "group", "rollout", tuple(samples), reward=1.0)
        rows = build_rows(rollout, 0.5)
        expected = Counter(
            (sample.prompt_ids + sample.completion_ids[:offset], token, sample.logprobs[offset])
            for sample in samples
            if sample.trainable
            for offset, token in enumerate(sample.completion_ids)
        )
        actual = Counter()
        for row in rows:
            inputs = tuple(row.datum.model_input.to_ints())
            fields = row.datum.loss_fn_inputs
            actions = {p for start, stop in row.actions for p in range(start, stop)}
            for position, target in enumerate(fields["target_tokens"].data):
                if position in actions:
                    actual[(inputs[: position + 1], target, fields["logprobs"].data[position])] += 1
                    assert fields["advantages"].data[position] == 0.5
                else:
                    assert fields["advantages"].data[position] == 0.0
        assert actual == expected
        assert rollout.minimum_policy_version == min(
            sample.policy_version for sample in samples if sample.trainable
        )
        relabeled = replace(
            rollout,
            samples=tuple(
                replace(sample, conversation_id=str(index)) for index, sample in enumerate(samples)
            ),
        )
        assert build_rows(relabeled, 0.5) == rows


def _joined_prefix_cover(samples):
    """The cover as first written, joining each candidate's prompt and completion to
    compare: the same chains, at the cost of copying a conversation per candidate."""
    result = []
    for sample in samples:
        current, longest = None, -1
        for candidate in result:
            previous = candidate[-1]
            prefix = previous.prompt_ids + previous.completion_ids
            if len(prefix) > longest and sample.prompt_ids[: len(prefix)] == prefix:
                current, longest = candidate, len(prefix)
        if current is not None:
            current.append(sample)
        else:
            result.append([sample])
    return tuple(tuple(chain) for chain in result)


def test_the_cover_compared_in_place_is_the_joined_prefix_cover_for_spans_and_tuples():
    rng = random.Random(90210)
    for _ in range(300):
        buffers, samples = TokenBuffers(), []
        for index in range(rng.randint(1, 12)):
            prompt = (rng.randrange(3),)
            if samples and rng.random() < 0.7:
                parent = rng.choice(samples)
                prompt = tuple(parent.prompt_ids) + parent.completion_ids
                cut = rng.random()
                if cut < 0.2:  # a re-render that tokenizes the parent's last token anew
                    prompt = (*prompt[:-1], prompt[-1] + 1)
                elif cut < 0.3:  # the parent's prompt and part of its completion
                    kept = len(parent.prompt_ids) + rng.randrange(len(parent.completion_ids))
                    prompt = prompt[:kept]
            prompt += tuple(rng.randrange(3) for _ in range(rng.randrange(3)))
            completion = tuple(rng.randrange(3) for _ in range(rng.randint(1, 3)))
            plain = make_sample(str(index), prompt, completion, (-0.5,) * len(completion))
            # Captured prompts are spans, over one buffer per chain or a fresh one.
            captured = buffers.span(f"conversation-{rng.randrange(2)}", prompt)
            samples.append(replace(plain, prompt_ids=captured) if rng.random() < 0.7 else plain)
        rollout = make_rollout("task", "group", "rollout", tuple(samples), reward=1.0)
        assert rollout.chains == _joined_prefix_cover(samples)
