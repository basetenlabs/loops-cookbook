# Examples

Each example is one of the [patterns](../patterns/) with a different task or different
settings. Every directory is self-contained. It holds the agent, or the custom rollout
runtime, that it trains.

The Model column also says where each example runs, as
[serverless or dedicated](../README.md#serverless-or-dedicated) explains:

- A model marked **(serverless)** runs on serverless Loops. Its config sets
  `[compute] scope = "serverless"`.
- A model with no mark runs on a dedicated deployment. Its config sets
  `scope = "dedicated"`.

| Example | Shows | Pattern it builds on | Model |
| --- | --- | --- | --- |
| [episode_basics](episode_basics/) | Small episodes show the `Episode` API. They score on the host, in a shared sandbox and in a separate verifier. | custom, in-process, direct | Qwen3.5-4B |
| [async_scheduling](async_scheduling/) | Sampling continues while the trainer updates the policy, and a rollout can train up to two steps off-policy. | custom, in-process, direct | Qwen3.6-35B-A3B (serverless) |
| [critic](critic/) | One policy plays two roles, and both roles train, through `[rollout.producers]`. | Harbor, in-process, direct | Qwen3.6-35B-A3B (serverless) |
| [multi_step](multi_step/) | A task has two steps, and each step is verified in the agent's environment. | Harbor, in-process, direct | Qwen3.6-35B-A3B (serverless) |
| [terminal_bench](terminal_bench/) | A bash agent works through a real multi-turn dataset. A second config trains it with ECHO, a custom loss. | Harbor, in-process, direct | GLM-5.3-Flash (serverless) |
| [litellm_streaming](litellm_streaming/) | An agent built on LiteLLM streams its replies. | custom, in-process, openai_http | Qwen3.6-35B-A3B (serverless) |
| [tool_agent](tool_agent/) | A multi-turn agent uses tools in a Blaxel sandbox, and the grader scores what it left there. | custom, in-process, direct | Qwen3.5-4B |
| [llm_judge](llm_judge/) | An LLM judge, reached through its own client, ranks each group's answers. This is group grading. | custom, in-process, direct | Qwen3.5-4B |
| [counting](counting/) | A vision model counts overlapping circles in an image. | custom, in-process, direct | Qwen3.6-35B-A3B (serverless) |
| [sat](sat/) | The model solves 3-SAT problems generated from seeds, with one reasoned answer each. The run evaluates held-out problems as it trains. | custom, in-process, direct | Qwen3.6-35B-A3B (serverless) |

The tasks that several examples share are in [tasks/](../tasks/).

## SFT

These examples use the supervised fine-tuning (SFT) recipe.

| Example | Shows | Model |
| --- | --- | --- |
| [sft_tool_use](sft_tool_use/) | The model learns from tool-use conversations that have a tool call and a follow-up. | Qwen3.6-35B-A3B (serverless) |
| [sft_toucan](sft_toucan/) | The model learns from real tool-agent trajectories from Hugging Face, and an evaluation compares it before and after. | Qwen3.6-35B-A3B (serverless) |
