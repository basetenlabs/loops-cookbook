"""The composition root for ``loops run``: an SFT run plus the standard telemetry.

``train`` builds the renderer first, off the event loop, so a renderer that cannot be
built fails before the run starts. ``assemble`` builds the trainer (restoring a
resume's state) and hands both to ``Run``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loops_cookbook.cancellation import see_through
from loops_cookbook.cli import ConfigSource
from loops_cookbook.llm.factory import load_tokenizer
from loops_cookbook.run_record import finished, finished_message
from loops_cookbook.sft.config import Config
from loops_cookbook.sft.run import (
    Manifest,
    Outcome,
    Plan,
    Run,
    check_resume,
    prepare,
    renderer_for,
)
from loops_cookbook.sft.sequences import SupervisedSequence, TrainingRenderer
from loops_cookbook.sft.trainer import Trainer, TrainerPort
from loops_cookbook.telemetry import telemetry
from loops_cookbook.threads import in_thread


@dataclass(frozen=True)
class CheckReport:
    """The training plan, and the first sequence's trained text to read its targets by."""

    plan: Plan
    preview: tuple[str, ...]
    resume: Manifest | None = None
    """The manifest a resume continues from, checked against the plan."""

    def describe(self) -> list[str]:
        """The same human-readable report printed by `loops check`."""
        resume, schedule = self.resume, self.plan.schedule
        return [
            *self.plan.describe(),
            *(
                [f"Resume: {resume.run_directory} at step {resume.next_step}; checked"]
                if resume
                else []
            ),
            *(
                [
                    "Resume: "
                    + finished_message(
                        schedule.total_steps,
                        schedule.bound,
                        schedule.more(resume.trainer_state_path),
                    )
                ]
                if resume and finished(resume.next_step, schedule.total_steps)
                else []
            ),
            *self.preview,
            "Configuration and data checked; no Loops trainer launched.",
        ]


async def check(config: Config, *, load: Callable[[str], Any] = load_tokenizer) -> CheckReport:
    """Validate and render the data without starting a trainer or printing a report;
    ``load`` loads the tokenizer a tokens dataset's preview is decoded with.

    A resume is held to what its run trained, as ``loops run`` holds it.

    Cancellation stops between preparation stages and rows, then waits for active workers.
    """
    resume = Manifest.resume_from(config.training.resume_from)
    planned = await prepare(config)
    if resume is not None:
        check_resume(resume, planned.digest, planned.schedule)
    if planned.renderer is not None:
        decode, note = planned.renderer.detokenize, ()
    else:
        # Loading a tokenizer may download it, so it stays off the event loop too.
        decode, note = await see_through(
            in_thread(_tokenizer_decode, config.model.tokenizer_source, load)
        )
    return CheckReport(planned, (*note, *_preview(planned.train.sequences[0], decode)), resume)


def _tokenizer_decode(
    source: str, load: Callable[[str], Any]
) -> tuple[Callable[[list[int]], str], tuple[str, ...]]:
    """The model's tokenizer, to show a tokens dataset's targets as text; their ids
    when it cannot be loaded (gated, or offline), since the data checked out regardless.
    The second item says when the ids are shown instead."""
    try:
        return load(source).decode, ()
    except OSError as error:
        return str, (f"Targets shown as token ids: the {source} tokenizer did not load ({error})",)


def _preview(sequence: SupervisedSequence, decode: Callable[[list[int]], str]) -> list[str]:
    """The first sequence's trained text, decoded, so the targets can be read."""
    spans: list[list[int]] = []
    previous = False
    for token, trained in zip(sequence.ids, sequence.trained, strict=True):
        if trained and not previous:
            spans.append([])
        if trained:
            spans[-1].append(token)
        previous = trained
    return [f"Trained text of {sequence.row}, span by span:"] + [
        f"  {decode(span)!r}" for span in spans
    ]


def assemble(
    config: Config,
    *,
    trainer: TrainerPort | None = None,
    renderer: TrainingRenderer | None = None,
) -> Run:
    """A ``Run`` with everything it uses built here, from ``config`` and the resume
    manifest it names. An optional renderer can be supplied; otherwise preparation
    builds it in a worker before rendering the dataset."""
    resume = Manifest.resume_from(config.training.resume_from)
    if trainer is None:
        trainer = Trainer(config, resume.trainer_state_path if resume else None)
    return Run(config, trainer, renderer, resume=resume)


async def train(config: Config, *, source: ConfigSource | None = None) -> Outcome:
    """Fine-tune per ``config``. Same entry point as ``loops run``. ``source`` is where
    the config came from, which the run's log records.

    ``Run`` on its own is silent; anything wanting other sinks takes one from
    ``assemble`` and passes its own to ``execute``.
    """
    # Built first: a renderer that cannot be built is the config's fault, not a run's.
    run = assemble(config, renderer=await renderer_for(config))
    preamble = source.describe() if source else ()
    async with telemetry(run.out, config, group=run.run_id, preamble=preamble) as metrics:
        return await run.execute(metrics)
