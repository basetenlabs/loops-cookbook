# Examples

Each example is a [pattern](../patterns/) with settings or the task changed. Every
directory is self-contained, with the agent or runtime it trains.

The Model column also says where each example runs
([serverless or dedicated](../README.md#serverless-or-dedicated)):

- **(serverless)**: its config sets `[compute] scope = "serverless"`.
- **No mark**: its config sets `scope = "dedicated"`.

| Example | Shows | Pattern it builds on | Model |
| --- | --- | --- | --- |
| [episode_basics](episode_basics/) | The `Episode` API: host-only scoring, a shared sandbox, a separate verifier | custom, in-process, direct | Qwen3.5-4B |
| [async_scheduling](async_scheduling/) | Async scheduling, up to two steps off-policy | custom, in-process, direct | Qwen3.6-35B-A3B (serverless) |
| [critic](critic/) | One policy in two roles, both trained (`[rollout.producers]`) | Harbor, in-process, direct | Qwen3.6-35B-A3B (serverless) |
| [multi_step](multi_step/) | A two-step task, each step verified in the agent's environment | Harbor, in-process, direct | Qwen3.6-35B-A3B (serverless) |
| [terminal_bench](terminal_bench/) | A real multi-turn dataset, with a bash agent; ECHO, a custom loss | Harbor, in-process, direct | GLM-5.3-Flash (serverless) |
| [litellm_streaming](litellm_streaming/) | An agent built on LiteLLM, streaming its replies | custom, in-process, openai_http | Qwen3.6-35B-A3B (serverless) |
| [tool_agent](tool_agent/) | A multi-turn agent with tools in a Blaxel sandbox, graded on what it left there | custom, in-process, direct | Qwen3.5-4B |
| [llm_judge](llm_judge/) | Group grading: an LLM judge ranks each group's answers, through its own client | custom, in-process, direct | Qwen3.5-4B |
| [counting](counting/) | Images: count overlapping circles | custom, in-process, direct | Qwen3.6-35B-A3B (serverless) |
| [sat](sat/) | 3-SAT problems generated from seeds, one reasoned answer each, with held-out problems evaluated | custom, in-process, direct | Qwen3.6-35B-A3B (serverless) |

The tasks they share are in [tasks/](../tasks/).

## SFT

| Example | Shows | Model |
| --- | --- | --- |
| [sft_tool_use](sft_tool_use/) | Supervised fine-tuning on tool-use conversations with a tool call and a follow-up | Qwen3.6-35B-A3B (serverless) |
| [sft_toucan](sft_toucan/) | Supervised fine-tuning on real tool-agent trajectories from Hugging Face, with a before-and-after eval | Qwen3.6-35B-A3B (serverless) |
