# Reference

These pages give the exact rules for how each part of the cookbook behaves. Start from
the [docs](../README.md), and come here when you need the details.

- [training.md](training.md) covers the training loop, how a rollout becomes training rows, groups and advantages, the training step, resuming and forking a run, evals, the loss, the outputs, and the settings grouped by what they control.
- [model-access.md](model-access.md) explains how an agent's model call becomes training evidence, both directly and over HTTP.
- [scheduling.md](scheduling.md) covers sampling and grading slots, pre-start places, and sync and async admission.
- [resources.md](resources.md) covers provider resources, the journal and `loops cleanup`, and agent processes.
- [failures.md](failures.md) explains what each way your code can end means, along with retries, deadlines and the systemic stop.
- [episode-api.md](episode-api.md) covers the `Episode` API, typed tasks and options, the lifecycle, and group grading.
- [harbor.md](harbor.md) describes the Harbor rollout runtime.
- [daytona.md](daytona.md) describes the Daytona provider and its Harbor environment, and how to choose between Blaxel and Daytona.
- [sft.md](sft.md) describes the supervised fine-tuning (SFT) recipe, including its data contracts, which tokens train, the training step, the metrics and the run directory.
- [renderers.md](renderers.md) covers which model chat templates the cookbook supports, and how to add a renderer.
- [library.md](library.md) explains how to use the cookbook from Python, with `load()`, `check()`, `train()` and the public packages.
- [KNOB_GLOSSARY.md](KNOB_GLOSSARY.md) lists every setting, by table. It is generated from the settings models.
- [settings-that-interact.md](settings-that-interact.md) covers the settings that only make sense together.
