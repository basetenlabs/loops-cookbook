# Tasks

The tasks the patterns and examples share.

- [hill_climb](hill_climb/): a Harbor task. One four-choice question whose verifier,
  run separately from the agent's sandbox, rewards the label `B`. Every Harbor pattern
  trains on it.
- [hill_climb_steps](hill_climb_steps/): the same kind of question as a two-step
  Harbor task, each step verified in the agent's environment as it ends.
- [hidden_answer.jsonl](hidden_answer.jsonl): one task for the custom runtimes that
  train Qwen3.5-4B. The model names a game, and the reply scores 1 when, stripped, it
  equals the line's `answer` field (Doom). A learnable hidden answer shows that the
  plumbing works, and the reward climbs where you can see it.
- [hidden_answer_35b.jsonl](hidden_answer_35b.jsonl): the same kind of task for the
  configs that train Qwen3.6-35B-A3B. The model chooses between Doom and Quake, and
  the reply scores 1 for Quake. A larger model needs its own task: asked for any
  classic game, Qwen3.8-27B named Doom most of the time and never Quake. With either
  answer, most groups would get the same reward for every reply, so there would be
  nothing to learn.
