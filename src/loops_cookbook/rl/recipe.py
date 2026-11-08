"""The composition root for ``loops run``: a training run plus the standard telemetry.

``assemble`` builds what a run uses -- the trainer (restoring a resume's state), the
runtime factory the config names, the renderer factory -- and hands it to ``Run``,
which builds none of it itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from loops_cookbook.cli import ConfigSource
from loops_cookbook.console import first_sentence
from loops_cookbook.evidence import TaskRef
from loops_cookbook.llm.client import probe_image_tokens
from loops_cookbook.llm.factory import create_renderer, describe, renderer_source
from loops_cookbook.llm.rendering import Renderer
from loops_cookbook.model_access.api import ModelAccessMode
from loops_cookbook.model_access.producers import DEFAULT_PRODUCER
from loops_cookbook.providers import create_adapters
from loops_cookbook.resources.api import ResourceAdapter
from loops_cookbook.resources.registry import close_adapters
from loops_cookbook.rl.config import Config, EvalSetSettings, ProducerSpec, TrainingSettings
from loops_cookbook.rl.loss import load_custom_loss
from loops_cookbook.rl.observability import observe
from loops_cookbook.rl.run import SAMPLER_WEIGHTS, Manifest, Outcome, Run, finished_hint
from loops_cookbook.rl.trainer import Trainer, TrainerPort
from loops_cookbook.run_record import finished, finished_message, manifests
from loops_cookbook.runtime.api import RuntimeCapabilities, RuntimeFactory
from loops_cookbook.runtime.context import import_factory
from loops_cookbook.trainer import Capacity, check_served_model


def configured_renderer(config: Config) -> Renderer:
    """The renderer ``[model]`` asks for; the run calls it once per reasoning effort."""
    return create_renderer(
        config.model.base_model,
        tokenizer_source=config.model.tokenizer_source,
        chat_template_kwargs=config.model.renderer_chat_template_kwargs,
        renderer=config.model.renderer,
    )


def assemble(
    config: Config,
    *,
    config_dir: Path | None = None,
    adapters: Mapping[str, ResourceAdapter] | None = None,
    environ: Mapping[str, str] | None = None,
    trainer: TrainerPort | None = None,
    runtime_factory: RuntimeFactory | None = None,
    renderer_factory: Callable[[Config], Renderer] | None = None,
    preflight_only: bool = False,
    run_directory: Path | None = None,
) -> Run:
    """A ``Run`` with everything it uses built here, from ``config``, the resume
    manifest it names, and the runtime factory and renderer it asks for.

    Each part can be supplied instead, and only what is not is built: ``loops
    check`` passes a trainer that never provisions, and tests pass fakes.
    """
    resume = Manifest.resume_from(config.training.resume_from)
    if trainer is None:
        trainer = Trainer(config, resume.trainer_state_path if resume else None)
    return Run(
        config,
        trainer,
        runtime_factory=runtime_factory or import_factory(config.rollout_runtime.factory),
        renderer_factory=renderer_factory or configured_renderer,
        environ=environ,
        resume=resume,
        config_dir=config_dir,
        adapters=adapters,
        preflight_only=preflight_only,
        run_directory=run_directory,
    )


async def train(config: Config, *, source: ConfigSource | None = None) -> Outcome:
    """Train ``config`` with its configured runtime. Same entry point as ``loops run``.

    ``source`` is where the config came from: its directory is the runtime's
    ``config_dir`` (the working directory without one), and the run's log records it.

    ``Run`` on its own is silent, so anything wanting different sinks takes one from
    ``assemble`` and observes its ``snapshot()`` and ``add_step_sink()`` itself.
    """
    adapters = await create_adapters(config.providers)
    try:
        run = assemble(config, config_dir=_directory(source), adapters=adapters)
    except BaseException:
        await close_adapters(adapters)
        raise
    async with observe(run, source.describe() if source else ()):
        return await run.execute()


@dataclass(frozen=True)
class SavedWeights:
    """What run directory ``directory`` kept of one step's weights."""

    sampler_weights: str | None = None
    """The path a sampler serves, when the run kept it."""
    policy_version: int = 0
    trainer_state: str | None = None
    """Else the trainer state, which a trainer saves for a sampler first."""


def saved_weights(directory: Path, step: int) -> SavedWeights:
    """Step ``step``'s weights in run ``directory`` (-1: the base model, which needs
    none); a ``ValueError`` naming the steps it can evaluate when it kept neither."""
    if step < 0:
        return SavedWeights()
    kept: dict[int, dict] = {}
    record = directory / SAMPLER_WEIGHTS
    if record.is_file():
        for line in record.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                kept[row["step"]] = row
    states = manifests(directory)
    manifest = json.loads(states[step].read_text()) if step in states else None
    if step in kept:
        row = kept[step]
        # A step retrained after a resume from an earlier one, keeping only its state
        # this time: the weights recorded first are the abandoned attempt's.
        kept_at = row.get("written_at")
        retrained = (
            manifest is not None
            and kept_at is not None
            and (manifest.get("written_at") or 0) > kept_at
        )
        if not retrained:
            return SavedWeights(row["path"], row["policy_version"])
    if manifest is not None:
        return SavedWeights(trainer_state=manifest["trainer_state_path"])
    steps = sorted(kept.keys() | states.keys())
    raise ValueError(
        f"{directory} kept no weights for step {step}; it can evaluate "
        + (", ".join(map(str, steps)) if steps else "only the base model")
    )


async def evaluate(
    config: Config,
    directory: Path,
    step: int,
    *,
    sets: Sequence[str] = (),
    source: ConfigSource | None = None,
    trainer: TrainerPort | None = None,
    runtime_factory: RuntimeFactory | None = None,
    renderer_factory: Callable[[Config], Renderer] | None = None,
) -> list[str]:
    """Evaluate step ``step`` of run ``directory`` (-1: the base model) with the
    config's eval sets, as ``loops eval`` does: the files and log lines of an eval
    during the run, written into that run's ``eval/``. The sets that ran it. The
    parts ``assemble`` takes can be supplied instead."""
    names = list(config.eval.sets)
    if unknown := sorted(set(sets) - set(names)):
        raise ValueError(
            f"no eval set {', '.join(map(repr, unknown))}; the config's are "
            + (", ".join(map(repr, names)) or "none")
        )
    weights = saved_weights(directory, step)
    # The run evaluated is ``directory``, whatever run the config resumes training
    # from; and only the sets named are built at all.
    chosen = {name: s for name, s in config.eval.sets.items() if not sets or name in sets}
    config = config.model_copy(
        update={
            "training": config.training.model_copy(update={"resume_from": None}),
            "eval": config.eval.model_copy(update={"sets": chosen}),
        }
    )
    adapters = await create_adapters(config.providers)
    try:
        run = assemble(
            config,
            config_dir=_directory(source),
            adapters=adapters,
            trainer=trainer,
            runtime_factory=runtime_factory,
            renderer_factory=renderer_factory,
            run_directory=directory,
        )
    except BaseException:
        await close_adapters(adapters)
        raise
    return await run.evaluate_after(
        step,
        sampler_weights=weights.sampler_weights,
        trainer_state=weights.trainer_state,
        policy_version=weights.policy_version,
    )


@dataclass(frozen=True)
class ImageSupport:
    tokens: int | None = None
    error: str | None = None

    def describe(self) -> str:
        if self.error is not None:
            return (
                f"unavailable ({self.error}); text-only training works, "
                "run `uv sync --extra vision` for image inputs"
            )
        if self.tokens is None:
            return "not supported by this renderer (text only)"
        return f"supported ({self.tokens} tokens for a 32x32 probe)"


@dataclass(frozen=True)
class EvalSetReport:
    """One eval set as preflight built it."""

    name: str
    tasks: int
    shared: int
    """Tasks it shares with training, by id or by content."""
    settings: EvalSetSettings

    def describe(self, every_steps: int, max_steps: int | None, *, no_base: str | None) -> str:
        """``no_base``: why the run has no base eval, as one that resumes or starts
        from other weights does not; ``None`` when it has one."""
        settings = self.settings
        samples = settings.samples_per_task
        # A run without max_steps has no last update to evaluate after.
        last = "after the last" if max_steps is not None else "no final eval without max_steps"
        first = "before the first" if no_base is None else f"no base eval ({no_base})"
        cadence = (
            f"every {every_steps} updates, {first} and {last}"
            if every_steps
            else f"{first}{' update' if no_base is None else ''} and {last}"
        )
        line = (
            f"Eval set {self.name!r}: {self.tasks} task{'s' if self.tasks != 1 else ''} "
            f"× {samples} sample{'s' if samples != 1 else ''}, "
            f"{settings.max_concurrent_rollouts} at once; {cadence}"
        )
        if self.shared:
            line += f"; {self.shared} task{'s' if self.shared != 1 else ''} shared with training"
        return line


@dataclass(frozen=True)
class CheckReport:
    """Preflight observations, available after its runtime and resource clients close."""

    tasks: tuple[TaskRef, ...]
    capabilities: RuntimeCapabilities
    model_access: ModelAccessMode | None
    rollout_timeout_seconds: float
    output_dir: Path
    config: Config
    model_check: str
    renderer: str
    reasoning_efforts: tuple[str, ...]
    images: ImageSupport
    resume_step: int | None = None
    """The step a resume continues at."""
    eval_sets: tuple[EvalSetReport, ...] = ()

    def describe(self) -> list[str]:
        """The same human-readable report printed by `loops check`."""
        config, capabilities = self.config, self.capabilities
        lines = [self.model_check]
        lines.append(f"Rollout runtime: {config.rollout_runtime.factory}; tasks: {len(self.tasks)}")
        mode = self.model_access.value if self.model_access else "unresolved"
        lines.append(
            "Capabilities: model access "
            + (
                f"{mode} (set in [rollout_runtime] model_access)"
                if config.rollout_runtime.model_access is not None
                else capabilities.model_access_source or mode
            )
            + "; runtime-acquired phases "
            + (
                ", ".join(sorted(phase.value for phase in capabilities.internally_acquired_phases))
                or "none"
            )
            + "; grading retries "
            + (capabilities.grading_retries_source or str(capabilities.grading_retries))
            + f"; group grading {str(capabilities.pending_group_grade).lower()}"
        )
        lines.append(
            f"Renderer: {self.renderer} "
            + f"({renderer_source(config.model.tokenizer_source, config.model.renderer)})"
            + (f"; tokenizer from {config.model.tokenizer}" if config.model.tokenizer else "")
        )
        if self.reasoning_efforts:
            lines.append("Reasoning efforts: " + ", ".join(sorted(self.reasoning_efforts)))
        lines.append(f"Images: {self.images.describe()}")
        training, rollout = config.training, config.rollout
        lines.append(
            f"Mode: {training.mode}; batch: {training.groups_per_step} × {training.group_size}"
        )
        lines.append(f"Loss: {_describe_loss(training)}")
        if training.loss_fn == "custom":
            lines.append("Loss aggregation: owned by the custom loss")
        else:
            budget = (
                f" ({training.rollouts_per_step} rollouts × {training.loss_token_budget} action tokens)"
                if training.loss_aggregation == "fixed_budget"
                else ""
            )
            lines.append(f"Loss aggregation: {training.loss_aggregation}{budget}")
        lines.append(
            f"Rollout clock: {self.rollout_timeout_seconds:g}s ("
            + (
                "derived by the runtime"
                if rollout.rollout_timeout_seconds is None
                else "[rollout] rollout_timeout_seconds"
            )
            + ")"
        )
        lines.append(
            f"Concurrency: {rollout.max_concurrent_rollouts} sampling "
            f"(+{rollout.prestart_places} started ahead); "
            f"{rollout.max_concurrent_verifiers} grading"
        )
        lines.append(_describe_producers(rollout.producers))
        max_steps = config.training.step_limit
        if self.resume_step is not None and finished(self.resume_step, max_steps):
            assert max_steps is not None
            lines.append("Resume: " + finished_message(max_steps, *finished_hint(max_steps)))
        lines.extend(
            eval_set.describe(
                config.eval.every_steps,
                max_steps,
                no_base="resume"
                if self.resume_step is not None
                else "init_from"
                if config.training.init_from
                else None,
            )
            for eval_set in self.eval_sets
        )
        lines.append(
            "Configuration and rollout runtime checked; no sandbox or Loops trainer launched."
        )
        return lines


def _directory(source: ConfigSource | None) -> Path | None:
    return source.directory if source else None


async def check(config: Config, *, source: ConfigSource | None = None) -> CheckReport:
    """Run the CLI's preflight and return its observations after releasing resources."""
    # First: a model Loops does not serve also fails the renderer's download, less legibly.
    model_check = await check_served_model(config.model)
    try:
        return await _preflight(config, _directory(source), model_check)
    except Exception as error:
        # A check that fails after this still says whether Loops serves the model.
        error.add_note(model_check)
        raise


async def _preflight(config: Config, config_dir: Path | None, model_check: str) -> CheckReport:
    adapters = await create_adapters(config.providers)
    try:
        run = assemble(
            config.model_copy(
                update={
                    "output": config.output.model_copy(
                        update={"output_dir": str(Path(config.output.output_dir) / "preflight")}
                    )
                }
            ),
            config_dir=config_dir,
            adapters=adapters,
            trainer=cast(TrainerPort, _NoTrainer(Capacity(config.compute.scope))),
            preflight_only=True,
        )
    except BaseException:
        await close_adapters(adapters)
        raise
    try:
        runtime, tasks, renderers = await run.preflight()
    except BaseException:
        await run.close_resources()
        raise
    renderer = renderers.base
    try:
        run.reconcile_resume(tasks, renderer)
        capabilities, timeout = run.capabilities, run.rollout_timeout_seconds
        if capabilities is None or timeout is None:
            raise RuntimeError("preflight returned without the runtime's capabilities")
        load_custom_loss(config.training)
        return CheckReport(
            tasks=tuple(tasks),
            capabilities=capabilities,
            model_access=run.model_access,
            rollout_timeout_seconds=timeout,
            output_dir=run.out,
            config=config,
            model_check=model_check,
            renderer=describe(renderer),
            reasoning_efforts=tuple(sorted(renderers.by_effort)),
            images=_image_support(renderer),
            resume_step=run.resume_step,
            eval_sets=tuple(
                EvalSetReport(
                    eval_set.name,
                    len(eval_set.tasks),
                    run.eval_shared.get(eval_set.name, 0),
                    eval_set.settings,
                )
                for eval_set in run.eval_sets
            ),
        )
    finally:
        try:
            await run.close_runtimes(runtime)
        finally:
            await run.close_resources()


class _NoTrainer:
    """``check`` builds no trainer; nothing here is ever called."""

    sampler = None
    sampler_version = None
    resumed_step = 0
    policy_version = 0

    def __init__(self, capacity: Capacity) -> None:
        self.capacity = capacity


def _describe_loss(training: TrainingSettings) -> str:
    if training.custom_loss is not None:
        return f"custom ({training.custom_loss.factory})"
    settings = ", ".join(f"{key} {value:g}" for key, value in training.loss_fn_config.items())
    return f"{training.loss_fn} ({settings})" if settings else training.loss_fn


def _describe_producers(producers: Mapping[str, ProducerSpec]) -> str:
    """Every declared producer, and what a call that names none uses: the table replaces
    the default one, so a table without the default refuses such a call."""
    declared = ", ".join(
        f"{name} ({'trainable' if spec.trainable else 'not trained'})"
        for name, spec in sorted(producers.items())
    )
    default = f"a call that names none uses {DEFAULT_PRODUCER}"
    if DEFAULT_PRODUCER not in producers:
        default += ", which is not declared, so such a call is refused"
    return f"Producers: {declared}; {default}"


def _image_support(renderer: Renderer) -> ImageSupport:
    try:
        return ImageSupport(tokens=probe_image_tokens(renderer))
    except Exception as error:  # noqa: BLE001 - processor errors vary by backend
        return ImageSupport(error=f"{type(error).__name__}: {first_sentence(str(error))}")
