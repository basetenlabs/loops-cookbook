# Roadmap

Big-ticket items: new recipes, and additions to the runtimes and recipes that exist.

## Evaluation

- **Evaluating a checkpoint.** Run an environment's episodes against a checkpoint,
  scored by its grader, with no training: real multi-turn rollouts, tools that run,
  and several samples per task, for SFT before and after and for RL alike.
  `examples/sft_toucan/evaluate.py` is the stopgap: first-turn agreement with the data.

## Agents

- **Remote agents.** The third agent placement: an agent that runs inside the
  episode's sandbox and calls the model over HTTP, as a local agent does from the
  training machine today. On Harbor this trains its installed agents, such as OpenCode,
  without writing a `LoopsAgent`; on a custom runtime, any program in the sandbox.

## Models

- **Kimi vision.** Image inputs for Kimi K2.5, whose image parts the cookbook refuses
  today. Kimi K3 needs a Prime renderer first.

## SFT

- **Trace importers.** LangSmith and Langfuse exports to the chat contract, merging an
  agent trace's calls while each extends the last.
- **DPO**, in the SFT recipe beside SFT.
