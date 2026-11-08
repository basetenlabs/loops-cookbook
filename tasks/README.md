# Tasks

This directory holds the tasks that the patterns and examples share.

- [hill_climb](hill_climb/) is a Harbor task. It asks one four-choice question, and its
  verifier rewards the label `B`. The verifier runs separately from the agent's
  sandbox. Every Harbor pattern trains on this task.
- [hill_climb_steps](hill_climb_steps/) asks the same kind of question as a two-step
  Harbor task. Each step is verified in the agent's environment as soon as it ends.
- [hidden_answer.jsonl](hidden_answer.jsonl) holds one task for the custom rollout
  runtime configs that train Qwen3.5-4B. The model names a game. The reply scores 1
  when it equals the line's `answer` field, Doom, after surrounding whitespace is
  stripped. A hidden answer that the model can learn shows that the plumbing works, and
  the reward climbs where you can see it.
- [hidden_answer_35b.jsonl](hidden_answer_35b.jsonl) holds the same kind of task for the
  configs that train Qwen3.6-35B-A3B. The model chooses between Doom and Quake, and the
  reply scores 1 for Quake. A larger model needs its own task. When Qwen3.8-27B was
  asked for any classic game, it named Doom most of the time and never Quake. So with
  either game as the answer, most groups would give every reply the same reward, and
  there would be nothing to learn.
