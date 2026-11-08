# Roadmap

This page lists the large items that are planned. They are new recipes, and additions
to the rollout runtimes and recipes that already exist.

## Evaluation

- **Evaluating an SFT checkpoint on rollouts.** An RL run can already evaluate held-out
  task sets during training, and `loops eval` evaluates one update of a finished run.
  Supervised fine-tuning (SFT) measures only the validation negative log-likelihood
  (NLL). You will be able to run an environment's episodes against an SFT checkpoint,
  scored by the environment's grader, with no training. The evaluation will use real
  multi-turn rollouts, tools that run, and several samples for each task, so that you
  can compare a model before and after SFT. Until then,
  `examples/sft_toucan/evaluate.py` is a stopgap that measures how well first turns
  agree with the data.

## Agents

- **Remote agents.** Remote is the third agent placement, in which the agent runs inside
  the episode's sandbox. There it would call the model over HTTP, as a local agent does
  from the training machine today. With the Harbor rollout runtime, this would train
  Harbor's installed agents, such as OpenCode, without a `LoopsAgent`. With a custom
  rollout runtime, it would train any program in the sandbox.

## Models

- **Kimi vision.** This item adds image inputs for Kimi K2.5. The cookbook refuses Kimi
  K2.5's image parts today. Kimi K3 first needs a renderer in Prime
  Intellect's `renderers` package, which the cookbook renders with.

## SFT

- **Trace importers.** Importers will turn LangSmith and Langfuse exports into the
  `chat` contract. Each will merge an agent trace's calls into one conversation, for as
  long as each call extends the one before it.
- **DPO.** The SFT recipe will offer direct preference optimization (DPO) beside SFT.
