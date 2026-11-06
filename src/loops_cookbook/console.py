"""A live progress region for a terminal, and nothing at all for a log file.

Optimizer steps are far enough apart that a run in its collection phase looks
identical to one that has died. On a terminal that is worth a pinned line showing
what the pipeline is doing; in a redirected log it would be the same line repeated
forever, so there it is simply absent.

The region and the log have to share one console -- two writers on a terminal
interleave mid-redraw and corrupt both. They share it by construction here: the
application installs one handler at startup, and the region draws on whatever
console that handler is already using. Nothing reaches into the root logger at
runtime, so nothing has to guess which of the caller's handlers it may displace.

Presentation only. Nothing here knows what a run is or decides when to redraw --
``observability`` owns the timer and the wiring.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

REFRESH_PER_SECOND = 4

# The terminal is one resource, so only one region may draw on it. Ours rather than
# rich's: `Console.set_live` documents raising on a second Live but this version
# keeps a stack and allows it, and the attribute tracking it has already been
# renamed once, so it is not something to depend on.
_DRAWING = threading.Lock()

# Three views, one per line, so no line mixes units. SLOTS are the caps -- every
# denominator lives here, each labelled by the resource it belongs to. STAGES are
# where the rollouts are, in the order a rollout passes through them, as plain
# counts: without a denominator on the line, `sampling 47` cannot be misread as
# "one permit free" when `setup` holds the other.
#
# `prep` is before Harbor accepts the trial and `graded`/`ready` are after Harbor is
# done with it, so only the five between them consume a trial slot.
# Two kinds of limit, kept on separate rows because they are separate pools and one
# row invited reading them as a partition. Harbor's cap covers every rollout inside
# a trial, including the ones holding no worker permit at all (`queued`, `to-grade`),
# so `trials` is deliberately larger than the two worker caps put together and is
# not their sum.
HARBOR_SLOTS = (("runtime/harbor/live_trials", "trials", "runtime/harbor/trial_cap"),)
HARBOR_STAGES = (("runtime/harbor/preparing_trials", "prep"),)
# Harbor's rows are drawn only on a Harbor run: elsewhere they are zeros that read
# as a stalled Harbor. The runtime reports its trial cap when it prepares, before the
# first frame, so the rows never appear mid-run and change the region's height.
HARBOR_RUN = "runtime/harbor/trial_cap"

WORKER_SLOTS = (
    ("admission/active_rollouts", "rollouts", "progress/rollout_slots"),
    ("slots/verification", "verifiers", "progress/verifier_slots"),
)

STAGES = (
    ("admission/waiting", "queued"),
    ("admission/prestarted", "setup"),
    ("slots/sampling", "sampling"),
    ("queue/verification_waiting", "to-grade"),
    ("slots/verification", "grading"),
    # Graded and waiting on: more rollouts to fill the group, then the optimizer.
    ("queue/graded_rollouts", "graded"),
    ("queue/trainable_rollouts", "ready"),
)


def use_rich(setting: str, *, stream: Any = None) -> bool:
    """Whether to drive a terminal, given the setting and where output is going.

    ``auto`` asks the stream, because the cost of guessing wrong is a log full of
    control characters. An explicit ``rich`` still declines a non-terminal for the
    same reason -- there is nothing to animate in a file.

    The stream asked is the one that will be written to, and it defaults to the one
    logging uses. Deciding on ``stdout`` while drawing on ``stderr`` gets it wrong
    in both directions: `2> run.log` fills the log with a hard-wrapped transcript,
    and `> run.log` hides the region from the terminal watching it.
    """
    if setting == "plain":
        return False
    stream = sys.stderr if stream is None else stream
    if not getattr(stream, "isatty", lambda: False)():
        return False
    try:
        import rich  # noqa: F401
    except ImportError:
        if setting == "rich":
            logger.warning(
                "console = 'rich' needs the console extra; falling back to plain "
                "output. Install it with `uv sync --extra console`."
            )
        return False
    return True


def log_handler(*, stream: Any = None) -> logging.Handler:
    """The handler an application installs when it wants the pretty output.

    Tracebacks stay plain. Rich renders them synchronously on the calling thread at
    roughly a millisecond per frame, and ours unwind through asyncio, Harbor, Blaxel
    and httpx -- so a burst of rollout failures, which is exactly when the log
    matters, would block the event loop for seconds formatting its own account of it.
    """
    from rich.console import Console
    from rich.logging import RichHandler

    console = Console(file=sys.stderr if stream is None else stream)
    return RichHandler(console=console, show_path=False, rich_tracebacks=False)


def _installed_console() -> Any:
    """The console the log already writes through, if the application set one up.

    Asking the handler rather than holding a global is what makes the shared
    console structural: a region can only exist where the log is already going, so
    the two can never end up on separate writers.
    """
    for handler in logging.getLogger().handlers:
        console = getattr(handler, "console", None)
        # Console-backed is not the same as terminal-backed: an application may
        # point one at a file, and a region drawn there writes redraw frames into
        # a log nobody is watching while the terminal shows nothing at all.
        if console is not None and console.is_terminal:
            return console
    return None


def _count(gauges: dict[str, float], key: str, cap: str | None) -> str:
    value = int(gauges.get(key, 0))
    limit = int(gauges.get(cap, 0)) if cap else 0
    return f"{value}/{limit}" if limit else str(value)


def elapsed(gauges: dict[str, float]) -> str:
    """The run's own clock, as ``h:mm:ss``.

    Proof of life, and the reason it is taken from the snapshot rather than drawn as
    a spinner: a spinner is animated by rich's refresh thread, so it would keep
    turning happily while the event loop was blocked. This only advances when the
    poller that reads it actually ran.
    """
    total = int(gauges.get("pipeline/elapsed_s", 0))
    return f"{total // 3600}:{total // 60 % 60:02d}:{total % 60:02d}"


def first_sentence(text: str) -> str:
    """The first sentence of an error message; processor errors run to paragraphs."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line.split(". ", 1)[0].rstrip(".")


def _limits(gauges: dict[str, float], group: tuple) -> str:
    return "  ".join(f"{label} {_count(gauges, key, cap)}" for key, label, cap in group)


def _on_harbor(gauges: dict[str, float]) -> bool:
    return HARBOR_RUN in gauges


def harbor(gauges: dict[str, float]) -> str:
    """Harbor's own cap: every rollout inside a trial, permit or no permit.

    Empty on any other runtime, which has no trials.
    """
    return _limits(gauges, HARBOR_SLOTS) if _on_harbor(gauges) else ""


def workers(gauges: dict[str, float]) -> str:
    """The two worker pools. Their sum is not Harbor's cap and never was."""
    return _limits(gauges, WORKER_SLOTS)


def stages(gauges: dict[str, float]) -> str:
    """Where the rollouts are, in the order a rollout passes through them.

    Plain counts. The caps live on their own line, so nothing here has to carry a
    denominator that belongs to two stages at once.
    """
    walk = (*HARBOR_STAGES, *STAGES) if _on_harbor(gauges) else STAGES
    return "  ".join(f"{label} {int(gauges.get(key, 0))}" for key, label in walk)


def groups(gauges: dict[str, float]) -> str:
    """A group's life in order, then what the run has done with them.

    The first three are live and disjoint -- they sum to every group the pipeline
    holds. The last two are running totals, which is why they come last: `trained 1
    discarded 9` is the answer to whether a run is converting work into updates at
    all, and on a batch stalled by uniform rewards it is the only line that says so.
    """
    return (
        f"filling {int(gauges.get('groups/filling', 0))}  "
        f"backlog {int(gauges.get('groups/ready', 0))}  "
        f"batch {_count(gauges, 'groups/selected', 'progress/groups_needed')}  "
        f"trained {int(gauges.get('pipeline/groups_trained', 0))}  "
        f"discarded {int(gauges.get('discard/groups/total', 0))}"
    )


def trainer(gauges: dict[str, float]) -> str:
    """What the trainer is doing, and which policy the samplers are serving.

    Its own line because it is a separate component with its own state: the batch
    bar is about collection, and a run can be idle here while collection is busy --
    which is most of the time, and worth being able to see at a glance.
    """
    step = int(gauges.get("progress/step", 0))
    doing = f"training step {step}" if gauges.get("progress/training") else "idle"
    return f"{doing}  policy v{int(gauges.get('progress/policy_version', 0))}"


def note(text: str) -> None:
    """Print a line that has to survive being copied, outside the log's columns.

    A log record is laid out in fixed-width time and level columns and its message
    wrapped into whatever is left, which breaks a long path across three lines and
    mid-token. Paths and URLs are the few things a reader copies whole, so they go
    to the console unwrapped. Without a console there are no columns to escape and
    the log is the right place.
    """
    console = _installed_console()
    if console is None:
        logger.info("%s", text)
        return
    # `soft_wrap` is the one combination that emits the whole string: `no_wrap`
    # alone still crops it to the terminal width, silently losing the tail of the
    # very path the reader needs.
    console.print(text, soft_wrap=True, style="dim")


def summarise(gauges: dict[str, float]) -> str:
    """One line: how far the batch has come, and where its rollouts are sitting.

    A pure function of one snapshot, so the region, the per-step line and the
    progress record all say the same thing and can be read in a test without a run
    behind it.
    """
    limits = "  ".join(part for part in (harbor(gauges), workers(gauges)) if part)
    return (
        f"step {int(gauges.get('progress/step', 0))}  "
        f"groups {_count(gauges, 'groups/selected', 'progress/groups_needed')}  "
        f"{elapsed(gauges)}  |  {trainer(gauges)}  |  "
        f"{limits}  |  {stages(gauges)}  |  {groups(gauges)}"
    )


def region(gauges: dict[str, float], footer: str | None = None, notice: str | None = None) -> Any:
    """The pinned display: a bar for the batch, and where the rollouts are sitting.

    Rebuilt from scratch each frame rather than mutated, which is what makes it a
    pure function of its arguments -- there is no display state to drift. ``footer``
    is a line pinned under the counts, for something worth keeping on screen rather
    than scrolling away in the log.
    """
    from rich.progress_bar import ProgressBar
    from rich.table import Table
    from rich.text import Text

    def fixed(table: Any) -> Any:
        """One terminal line per row, whatever the width.

        A row that wraps changes the region's height, and its height depends on how
        wide the numbers are -- `trials 99/96` fits where `trials 101/96` does not.
        `Live` erases a fixed number of lines based on the height it last drew, so a
        frame that is a line taller or shorter strands the old one or eats a real
        log line. Truncating keeps the height constant and the accounting honest.
        """
        for column in table.columns:
            column.no_wrap = True
            column.overflow = "ellipsis"
        return table

    # `groups/selected`, not `groups/ready`: `next_batch` drains the ready queue the
    # instant a group lands, so its depth is structurally zero and a bar drawn from
    # it never moves. `selected` is what has actually been banked for this batch.
    banked = int(gauges.get("groups/selected", 0))
    needed = int(gauges.get("progress/groups_needed", 0))
    training = bool(gauges.get("progress/training"))
    head = Table.grid(padding=(0, 2))
    head.add_row(
        Text(f"step {int(gauges.get('progress/step', 0))}", style="bold white"),
        ProgressBar(total=max(needed, 1), completed=banked, width=24),
        Text(f"{banked}/{needed} groups", style="white"),
        Text(elapsed(gauges), style="dim"),
    )
    # Row labels, because the rows answer different questions and a reader should
    # not have to work out which denominators belong to which. The labels recede
    # and the numbers do not: this block is pinned under a scrolling log and has to
    # read as the thing you look at, not as something already scrolled past.
    body = Table.grid(padding=(0, 2))
    body.add_column(style="dim")
    body.add_column()
    if _on_harbor(gauges):
        body.add_row("harbor", Text(harbor(gauges), style="white"))
    body.add_row("workers", Text(workers(gauges), style="white"))
    body.add_row("stages", Text(stages(gauges), style="white"))
    body.add_row("groups", Text(groups(gauges), style="white"))
    body.add_row("trainer", Text(trainer(gauges), style="bold cyan" if training else "white"))
    stacked = Table.grid()
    stacked.add_row(fixed(head))
    stacked.add_row(fixed(body))
    if footer:
        stacked.add_row(Text(footer, style="blue"))
    if notice:
        stacked.add_row(Text(notice, style="bold yellow"))
    return fixed(stacked)


@contextlib.contextmanager
def live_region(enabled: bool) -> Iterator[Any]:
    """A pinned line below the scrolling log, or nothing at all.

    Nothing unless the application installed a console-backed handler: without one
    there is no terminal this may safely draw on, and a region on a second writer
    would corrupt both. Rich does not stop a second ``Live`` starting on the same
    console either, so a region entered while another is live also declines.
    """
    console = _installed_console()
    if not enabled or console is None or not _DRAWING.acquire(blocking=False):
        yield None
        return
    try:
        from rich.live import Live
        from rich.text import Text

        # Asymmetric on purpose. `stdout` belongs to the program -- redirecting it
        # would send `myrun > out.txt` output to the terminal instead of the file.
        # `stderr` is the stream this region and the log already share, so routing
        # it through the console is coherent, and it is the only way a raw write
        # that bypasses logging -- `warnings`, a C extension -- does not land
        # mid-redraw and strand a frame in the scrollback.
        with Live(
            Text(""),
            console=console,
            refresh_per_second=REFRESH_PER_SECOND,
            redirect_stdout=False,
            redirect_stderr=True,
        ) as live:
            yield live
    finally:
        _DRAWING.release()


def paint(live: Any, content: Any) -> None:
    """Replace the pinned display. Keeps ``rich`` from leaking into the caller.

    Takes a renderable from ``region`` or a plain string for the moments before
    there is any state to render.
    """
    if live is None:
        return
    from rich.text import Text

    live.update(Text(content, style="dim") if isinstance(content, str) else content)
