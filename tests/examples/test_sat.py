"""The 3-SAT example: problems generated from a seed, one model call, scored on the host."""

import itertools

import pytest
from sat_episode import Options, SatEpisode, clauses, satisfied
from support.harness import Renderer

from loops_cookbook.llm.rendering import ParsedReply

SEED, VARIABLES = 7, 8


def solutions(problem, variables):
    """Every assignment, as the set of variables set to true, that satisfies ``problem``."""
    for values in itertools.product((False, True), repeat=variables):
        true = {i + 1 for i, v in enumerate(values) if v}
        if satisfied(problem, true) == len(problem):
            yield true


def answer(true):
    return "True: " + (", ".join(map(str, sorted(true))) if true else "none")


def player(monkeypatch, reply):
    """The model, replying ``reply``; it records the prompt it was shown."""
    prompts = []
    render = Renderer.render

    def record(self, messages, tools):
        prompts.append(list(messages))
        return render(self, messages, tools)

    def parse(self, prompt_ids, completion_ids, tools):
        return ParsedReply(reply, None, (), None, {"role": "assistant", "content": reply})

    monkeypatch.setattr(Renderer, "render", record)
    monkeypatch.setattr(Renderer, "parse", parse)
    return prompts


async def play(make, seed=SEED, stop_reason="stop"):
    built = await make(
        episode=SatEpisode,
        sandbox=False,
        agent_timeout=30.0,
        rollout_timeout=60.0,
        tasks=({"id": f"problem-{seed}", "seed": seed},),
        options={"variables": VARIABLES},
    )
    built.rig.sampler.stop_reason = stop_reason
    result = await built.rig.rollout("g-r1", built.task)
    await built.rig.driver.drain()
    return result


def test_every_problem_has_a_solution_and_a_seed_is_one_problem():
    for seed in range(20):
        problem = clauses(seed, VARIABLES, 4.26)
        assert len(problem) == round(4.26 * VARIABLES)
        assert all(len({abs(lit) for lit in clause}) == 3 for clause in problem)
        assert next(solutions(problem, VARIABLES), None) is not None
        assert clauses(seed, VARIABLES, 4.26) == problem
    assert clauses(1, VARIABLES, 4.26) != clauses(2, VARIABLES, 4.26)


async def test_any_satisfying_assignment_scores_one(make, monkeypatch):
    true = next(solutions(clauses(SEED, VARIABLES, 4.26), VARIABLES))
    prompts = player(monkeypatch, f"Working through the clauses.\n{answer(true)}")
    result = await play(make)
    assert result.reward == 1.0
    assert result.metadata == {"answered": True, "satisfied": 1.0, "cut_off": False}
    assert result.trains
    [[message]] = prompts
    assert message["content"].startswith("Find values for the boolean variables x1 … x8")
    assert f"{round(4.26 * VARIABLES)} clauses" in message["content"]


async def test_an_assignment_that_breaks_a_clause_scores_zero(make, monkeypatch):
    problem = clauses(SEED, VARIABLES, 4.26)
    wrong = next(
        set(true)
        for n in range(VARIABLES + 1)
        for true in itertools.combinations(range(1, VARIABLES + 1), n)
        if satisfied(problem, set(true)) < len(problem)
    )
    player(monkeypatch, answer(wrong))
    result = await play(make)
    assert result.reward == 0.0
    assert result.metadata["answered"] and 0 < result.metadata["satisfied"] < 1


@pytest.mark.parametrize(
    "reply",
    [
        "I think x1 and x3 should be true.",
        "True: 1, 3\nThat should work.",  # the answer must be the last line
        "True: one, three",
        "",
    ],
)
async def test_a_reply_without_an_answer_line_scores_zero(make, monkeypatch, reply):
    player(monkeypatch, reply)
    result = await play(make)
    assert result.reward == 0.0
    assert result.metadata["answered"] is False


async def test_a_reply_the_token_budget_cut_off_scores_zero_and_trains_nothing(make, monkeypatch):
    true = next(solutions(clauses(SEED, VARIABLES, 4.26), VARIABLES))
    player(monkeypatch, answer(true))  # even a last line that would have solved it
    result = await play(make, stop_reason="length")
    assert result.reward == 0.0 and result.metadata["cut_off"] is True
    assert not result.trains and result.minimum_policy_version is not None


async def test_a_reply_the_model_ended_without_an_answer_trains_as_zero(make, monkeypatch):
    player(monkeypatch, "Still thinking about x3 and")
    result = await play(make)
    assert result.reward == 0.0 and result.metadata["cut_off"] is False
    assert result.trains


def test_options_refuse_fewer_than_three_variables():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Options(variables=2)


def test_options_refuse_a_ratio_that_rounds_to_no_clauses():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="no clauses"):
        Options(variables=21, clause_ratio=0.01)
