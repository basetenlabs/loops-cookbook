# Reference

How the pieces behave in detail. Start from the [docs](../README.md); come here for the
exact rules.

- [training.md](training.md): the loop, rollouts to training rows, groups and advantages, the training step, outputs, and settings by what they control.
- [model-access.md](model-access.md): how an agent's model call becomes training evidence, directly and over HTTP.
- [scheduling.md](scheduling.md): sampling and grading slots, pre-start places, sync and async.
- [resources.md](resources.md): provider resources, the journal and `loops cleanup`, and agent processes.
- [failures.md](failures.md): what each way your code can end means, retries, deadlines, and the systemic stop.
- [episode-api.md](episode-api.md): the `Episode` API, typed tasks and options, the lifecycle, and group grading.
- [harbor.md](harbor.md): the Harbor runtime.
- [daytona.md](daytona.md): the Daytona provider, its Harbor environment, and choosing between Blaxel and Daytona.
- [sft.md](sft.md): the SFT recipe: the data contracts, which tokens train, the training step, metrics, and the run directory.
- [renderers.md](renderers.md): model chat-template support and adding a renderer.
- [library.md](library.md): using the cookbook from Python: `load()`, `check()`, `train()` and the public packages.
- [KNOB_GLOSSARY.md](KNOB_GLOSSARY.md): every setting, by table (generated).
- [settings-that-interact.md](settings-that-interact.md): settings that only make sense together.
