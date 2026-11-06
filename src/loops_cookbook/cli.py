"""Command-line machinery every recipe's command shares.

Each recipe's ``Config`` is the single source of its flags: every leaf setting gets a
``--section.key`` override, help comes from the field descriptions, and a misspelt
flag is answered with the setting it most resembles. Loading a config file, merging
overrides into it, reading ``.env`` and installing the log handler are the same for
every command, so they live here once. So is running a command's ``main``: a SIGTERM
stops it the way Ctrl-C does.

A recipe declares only what differs, as a ``Recipe``: its config model, how it checks
and trains, and the errors that mean its config is wrong. ``loops`` runs every
recipe's commands from it.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import logging
import signal
import sys
import tomllib
from collections.abc import Callable, Coroutine, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Union, get_args, get_origin

from dotenv import load_dotenv
from pydantic import BaseModel, TypeAdapter
from pydantic.fields import FieldInfo

from loops_cookbook import threads
from loops_cookbook.console import log_handler, use_rich
from loops_cookbook.settings import is_dedicated_only, is_local_path


def settings(model: type[BaseModel], prefix: str = "") -> Iterator[tuple[str, FieldInfo]]:
    """Every leaf setting as ``(dotted.path, field)``, in declaration order."""
    for name, info in model.model_fields.items():
        path = f"{prefix}{name}"
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            yield from settings(annotation, f"{path}.")
        else:
            yield path, info


def _is_structured(annotation: Any) -> bool:
    """Whether the command line spells this setting as JSON: a table, list, or model."""
    if get_origin(annotation) in (Union, type(int | None)):
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        annotation = members[0] if len(members) == 1 else annotation
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return True
    return get_origin(annotation) in (dict, tuple, list)


def load_config_file(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix == ".json":
        # JSON is also the serialized config written beside every run.
        return TypeAdapter(dict[str, Any]).validate_python(json.loads(text))
    return tomllib.loads(text)


def apply_override(values: dict[str, Any], path: str, value: Any) -> None:
    """Set the setting at ``path`` in the file's nested values; a table merges into it."""
    *sections, name = path.split(".")
    target = values
    for section in sections:
        child = target.setdefault(section, {})
        if not isinstance(child, dict):
            raise ValueError(f"[{section}] must be a TOML table")  # noqa: TRY004
        target = child
    target[name] = _merge(target.get(name), value)


def _merge(current: Any, override: Any) -> Any:
    """An override table keeps the keys it does not name; anything else replaces."""
    if not (isinstance(current, dict) and isinstance(override, dict)):
        return override
    merged: dict[str, Any] = dict(current)  # pyright: ignore[reportUnknownArgumentType]
    for key, value in override.items():  # pyright: ignore[reportUnknownVariableType]
        merged[key] = _merge(merged.get(key), value)
    return merged


def _metavar(annotation: Any) -> str:
    """A short type hint for --help, derived from the setting's annotation."""
    if get_origin(annotation) in (Union, type(int | None)):
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(members) == 1:
            return _metavar(members[0])
    if get_origin(annotation) is Literal:
        return "{" + ",".join(str(value) for value in get_args(annotation)) + "}"
    if _is_structured(annotation):
        return "JSON"
    return {int: "INT", float: "FLOAT", bool: "{true,false}", str: "TEXT"}.get(annotation, "VALUE")


def cli_settings(model: type[BaseModel]) -> Iterator[tuple[str, FieldInfo]]:
    """Yield leaf fields unless their ``json_schema_extra`` metadata sets ``cli=False``."""
    for path, info in settings(model):
        extra = info.json_schema_extra
        if isinstance(extra, dict) and extra.get("cli") is False:
            continue
        yield path, info


def add_setting_flags(
    parser: argparse.ArgumentParser, models: Mapping[str, type[BaseModel]]
) -> None:
    """A flag for every leaf setting of the ``models``, which are keyed by recipe.

    A setting that only some recipes have names them, and a default the recipes do not
    agree on is shown for each recipe. A setting's description is the same in every
    model that declares it, so its flag has one help text.
    """
    declared: dict[str, dict[str, FieldInfo]] = {}
    for label, model in models.items():
        for path, info in cli_settings(model):
            declared.setdefault(path, {})[label] = info
    groups: dict[str, argparse._ArgumentGroup] = {}  # pyright: ignore[reportPrivateUsage]
    for path, infos in declared.items():
        section = path.rsplit(".", 1)[0]
        group = groups.get(section)
        if group is None:
            group = groups[section] = parser.add_argument_group(f"[{section}] overrides")
        info = next(iter(infos.values()))
        help_text = info.description or ""
        if is_dedicated_only(info):
            help_text = f"{help_text} Dedicated only."
        if len(infos) < len(models):
            help_text = f"({', '.join(infos)}) {help_text}"
        shown = {label: _shown_default(each) for label, each in infos.items()}
        if all(each.is_required() for each in infos.values()):
            help_text = f"{help_text} Required." if "Required" not in help_text else help_text
        elif len(set(shown.values())) == 1:
            if (default := next(iter(shown.values()))) is not None:
                help_text = f"{help_text} (default: {default})"
        elif each := ", ".join(f"{label} {value}" for label, value in shown.items() if value):
            help_text = f"{help_text} (default: {each})"
        group.add_argument(
            f"--{path}",
            dest=path,
            default=None,
            metavar=_metavar(info.annotation),
            # argparse %-formats help text, so a description that mentions a
            # percentage is a conversion specifier and crashes `--help`.
            help=help_text.strip().replace("%", "%%"),
        )


def _shown_default(info: FieldInfo) -> str | None:
    """The default ``--help`` shows, or ``None`` when it is required or says nothing."""
    if info.is_required():
        return None
    default = info.get_default(call_default_factory=True)
    if default in (None, {}, ()):
        return None
    return str(default).lower() if isinstance(default, bool) else str(default)


def resolve_reference(value: str, *, config_path: Path, from_shell: bool) -> str:
    """A file a setting names, as an absolute path.

    A path written in the config file is relative to that file, so a config
    directory can be copied or shared without rewriting it. A path passed on the
    command line is relative to the shell instead, because that is what the caller
    typed -- resolving it against the config would look somewhere they never named.
    """
    reference = Path(value).expanduser()
    if not reference.is_absolute():
        base = Path.cwd() if from_shell else config_directory(config_path)
        reference = (base / reference).resolve()
    return str(reference)


def config_directory(config_path: Path) -> Path:
    """The directory a config's paths, ``.env`` and modules are beside: the real file's.

    A symlink is followed, so a config means the same thing however it is reached.
    """
    return config_path.expanduser().resolve().parent


# The wheel ships the repository's examples/, patterns/ and tasks/ here, at the same
# paths, so a config's ../ references reach the same files in an install as in a clone.
BUNDLED = Path(__file__).parent / "bundled"


def locate_config(path: Path) -> Path:
    """``path``, or the installed copy of the repository file at that path when nothing
    is there: ``loops run examples/counting/train.toml`` works in a clone and after
    ``pip install``. A file at ``path`` always wins, so a copied example runs as edited."""
    named = path.expanduser()
    if named.exists() or named.is_absolute():
        return named
    bundled = BUNDLED / named
    if bundled.is_file() and bundled.resolve().is_relative_to(BUNDLED.resolve()):
        return bundled
    return named


@dataclass(frozen=True)
class ConfigSource:
    """Where a config came from: its path as named, the real file that path reached when
    it was loaded, and the ``.env`` files read for it."""

    path: Path
    file: Path
    environment: tuple[Path, ...] = ()

    @classmethod
    def of(cls, path: Path, environment: Sequence[Path] = ()) -> ConfigSource:
        """``path``'s source now: a symlink repointed later does not move it."""
        named = path.expanduser().absolute()
        return cls(named, named.resolve(), tuple(environment))

    @property
    def directory(self) -> Path:
        return self.file.parent

    def describe(self) -> list[str]:
        named = str(self.path) if self.file == self.path else f"{self.path} -> {self.file}"
        read = ", ".join(str(path) for path in self.environment) or "none"
        return [
            f"Config: {named}",
            f"  paths, modules: beside {self.directory}",
            f"  .env: {read}",
        ]


def load_config[M: BaseModel](
    model: type[M],
    path: Path,
    overrides: Mapping[str, Any] | None = None,
    references: Sequence[str] = (),
    paths_or_names: Sequence[str] = (),
) -> M:
    """Settings from ``path`` with flag ``overrides`` applied, validated as ``model``.

    ``references`` are the dotted paths of settings that name files; each is made
    absolute before validation, against whoever named it (``resolve_reference``).
    A path ending in ``.*`` names a table of files, resolved entry by entry: an entry
    an override supplied is the shell's, the file's own stay the file's. A ``*``
    inside a path stands for every key of the table there, as each eval set's name.
    ``paths_or_names`` name a local path or a Hugging Face repo id; one is resolved
    the same way only when it is a path (``is_local_path``), and a repo id is kept.
    """
    known = {dotted for dotted, _ in settings(model)}
    named = [*references, *paths_or_names]
    if unknown := [ref for ref in named if ref.split(".*")[0] not in known]:
        raise ValueError(f"{model.__name__} has no setting {', '.join(unknown)}")
    path = path.expanduser()
    supplied = dict(overrides or {})
    values = load_config_file(path)
    for dotted, value in supplied.items():
        apply_override(values, dotted, value)
    for reference in references:
        for parts in _expand(values, tuple(reference.split("."))):
            if parts[-1] == "*":
                _resolve_table(values, parts[:-1], path, supplied)
            else:
                _resolve_setting(values, parts, path, supplied)
    for dotted in paths_or_names:
        _resolve_setting(values, tuple(dotted.split(".")), path, supplied, only=is_local_path)
    return model.model_validate(values)


def _table_at(values: dict[str, Any], sections: Sequence[str]) -> dict[str, Any] | None:
    """The table ``sections`` names in the raw settings, or ``None`` when absent."""
    table: Any = values
    for section in sections:
        table = table.get(section) if isinstance(table, dict) else None
    return table if isinstance(table, dict) else None


def _expand(values: dict[str, Any], parts: tuple[str, ...]) -> list[tuple[str, ...]]:
    """``parts`` with each ``*`` before the last replaced by every key of the table
    there; a trailing ``*`` is kept, for ``_resolve_table``. A key is one part, dots
    and all, as an alias like ``tasks.v1`` is."""
    inner = [at for at, part in enumerate(parts[:-1]) if part == "*"]
    if not inner:
        return [parts]
    at = inner[0]
    table = _table_at(values, parts[:at])
    return [
        expanded
        for key in (table or {})
        for expanded in _expand(values, (*parts[:at], key, *parts[at + 1 :]))
    ]


def _from_shell(supplied: Mapping[str, Any], parts: tuple[str, ...]) -> bool:
    """Whether an override supplied the setting at ``parts``: by its own dotted path,
    or inside a table an override set whole."""
    for at in range(len(parts), 0, -1):
        if (key := ".".join(parts[:at])) in supplied:
            value: Any = supplied[key]
            for part in parts[at:]:
                value = value.get(part) if isinstance(value, dict) else None
            if value is not None:
                return True
    return False


def _resolve_setting(
    values: dict[str, Any],
    parts: tuple[str, ...],
    path: Path,
    supplied: Mapping[str, Any],
    only: Callable[[str], bool] = lambda value: True,
) -> None:
    *sections, name = parts
    table = _table_at(values, sections)
    if table is not None and isinstance(table.get(name), str) and only(table[name]):
        table[name] = resolve_reference(
            table[name], config_path=path, from_shell=_from_shell(supplied, parts)
        )


def _resolve_table(
    values: dict[str, Any], parts: tuple[str, ...], path: Path, supplied: Mapping[str, Any]
) -> None:
    entries = _table_at(values, parts)
    if entries is None:
        return
    for key, value in entries.items():
        if isinstance(value, str):
            entries[key] = resolve_reference(
                value, config_path=path, from_shell=_from_shell(supplied, (*parts, key))
            )


def make_config_dir_importable(directory: Path) -> None:
    """Modules a config names by ``module:callable`` -- a runtime, an agent, a custom
    renderer -- live beside it, in ``config_directory``."""
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def load_environment(directory: Path) -> tuple[Path, ...]:
    """Load ``.env`` from the config's ``directory``, then from the working directory;
    the files read, in order.

    Variables already set in the process win. This runs before any Loops, Harbor
    or Blaxel import, because those SDKs read credentials at import time.
    """
    read: list[Path] = []
    for candidate in (directory / ".env", Path.cwd() / ".env"):
        if candidate.is_file() and all(not candidate.samefile(path) for path in read):
            load_dotenv(candidate, override=False)
            read.append(candidate)
    return tuple(read)


class ThirdPartyChatter(logging.Filter):
    """Keep third-party records below WARNING out of the run's own output.

    Setting a level on those loggers does not work: Harbor gives every per-trial
    logger an explicit ``setLevel(DEBUG)``, so their records reach our handler
    whatever their parent says. At 128 concurrent trials that buries everything the
    run itself reports -- including, unhelpfully, its own failures. Filtering at the
    handler is the one place that holds regardless of what a library does upstream.

    ``--verbose`` does not install this.
    """

    NOISY = ("harbor", "httpx", "httpcore", "blaxel", "urllib3", "opentelemetry", "botocore")

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING or not record.name.startswith(self.NOISY)


def configure_logging(*, verbose: bool, style: str) -> None:
    """Install the one handler this process logs through.

    Configuring logging belongs to the application, not to the cookbook. A library
    that reached into the root logger at runtime would have to guess which of the
    caller's handlers it was allowed to displace, and guess wrong either way --
    silencing a log file, or leaving a second writer to corrupt the progress region.
    Deciding once, here, removes the question: the region draws on the console this
    handler already owns, which is all the coordination rich needs.

    Safe to call twice. The style is only known after the training config is read,
    and a configuration error before that still has to reach the terminal.
    """
    if use_rich(style):
        # The name stays: third-party WARNINGs are exactly what the filter lets
        # through, and they are worth nothing if you cannot see who raised them.
        handler, template = log_handler(), "%(name)s: %(message)s"
    else:
        handler, template = (
            logging.StreamHandler(),
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=template,
        handlers=[handler],
        force=True,
    )
    # Warnings become log records rather than raw stderr writes, so they carry a
    # level and a timestamp and cannot interleave with a live region mid-redraw.
    logging.captureWarnings(True)
    if not verbose:
        handler.addFilter(ThirdPartyChatter())


def _known_flags(parser: argparse.ArgumentParser) -> set[str]:
    """Every flag this parser accepts, including its subcommands'.

    Asked of the parser rather than listed here: a hand-kept list drifts the moment
    a flag is added, and a flag missing from it gets "corrected" to whichever
    setting it happens to resemble.
    """
    parsers = [parser]
    for action in parser._actions:  # pyright: ignore[reportPrivateUsage]
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            parsers.extend(choices.values())  # pyright: ignore[reportUnknownArgumentType]
    return {
        name
        for one in parsers
        for action in one._actions  # pyright: ignore[reportPrivateUsage]
        for name in action.option_strings
    }


def hint(parser: argparse.ArgumentParser, argv: list[str] | None, *models: type[BaseModel]) -> None:
    """Name the nearest setting for a flag argparse rejected.

    Wrapped around ``parse_args`` rather than replacing it with
    ``parse_known_args``: argparse keeps deciding what is valid and keeps exiting
    non-zero, so if anything here is wrong the behaviour degrades to exactly what
    it was -- never to accepting a flag it should have refused.
    """
    paths = list(dict.fromkeys(path for model in models for path, _ in cli_settings(model)))
    flags = [f"--{path}" for path in paths]
    by_leaf = {path.rsplit(".", 1)[-1]: f"--{path}" for path in paths}
    known = _known_flags(parser)
    # `None` means argparse read sys.argv, so this has to read the same thing.
    for word in sys.argv[1:] if argv is None else argv:
        word = word.split("=", 1)[0]
        # A prefix of a real flag is what argparse accepts by abbreviation, so it
        # is not the one that was rejected.
        if not word.startswith("--") or any(flag.startswith(word) for flag in known):
            continue
        leaf = word.lstrip("-").replace("-", "_").rsplit(".", 1)[-1]
        if leaf in by_leaf:
            print(f"{word} is spelled {by_leaf[leaf]}", file=sys.stderr)
        elif close := difflib.get_close_matches(word, flags, n=3, cutoff=0.6):
            print(f"did you mean {' or '.join(close)}?", file=sys.stderr)
        return


def flag_overrides(args: argparse.Namespace, model: type[BaseModel]) -> dict[str, Any]:
    """The setting flags given on the command line, as ``{dotted.path: value}``."""
    overrides: dict[str, Any] = {}
    for path, info in cli_settings(model):
        supplied = getattr(args, path, None)
        if supplied is not None:
            overrides[path] = json.loads(supplied) if _is_structured(info.annotation) else supplied
    return overrides


class Terminated(Exception):
    """``run_main``'s ``main`` stopped because the process was sent SIGTERM."""


TERMINATED_EXIT_CODE = 128 + signal.SIGTERM


def run_main[T](main: Coroutine[Any, Any, T]) -> T:
    """Run ``main`` as ``asyncio.run`` does, with SIGTERM stopping it as Ctrl-C does.

    A scheduler stops a process with SIGTERM -- a spot preemption, a deleted pod, a
    cancelled job -- and by default that kills it where it stands: no drain, no
    last-gasp checkpoint, no cleanup. Here the first SIGTERM cancels ``main``, as
    ``asyncio.run`` does on the first Ctrl-C, so the run takes the same graceful stop,
    and ``Terminated`` is raised once it has. A later SIGTERM is ignored: cancelling
    again would cut that stop short, and whoever sent it sends SIGKILL when its grace
    runs out.

    The handler stays until the runner has shut down, so a later SIGTERM can't cut
    short its cancelling of leftover tasks or its wait for worker threads either.
    """
    stop = _SigtermStop()
    # Closing a loop leaves SIGTERM at its default, not at what it was.
    previous = signal.getsignal(signal.SIGTERM)
    installed = False
    try:
        with asyncio.Runner() as runner:
            try:
                try:
                    runner.get_loop().add_signal_handler(signal.SIGTERM, stop.terminate)
                    installed = True
                except (NotImplementedError, RuntimeError):
                    # No signal handlers on Windows, nor off the main thread: the
                    # default stands.
                    return runner.run(main)
                return runner.run(stop.run(main))
            finally:
                # The cookbook's own threads, waited for as the runner waits for the
                # default executor's: before the handler goes, so a write a stop left
                # running finishes rather than being cut off by a later SIGTERM. A task
                # `main` left behind is cancelled after this, on the runner's close; a
                # thread it starts then is joined only at interpreter exit.
                threads.finish()
    finally:
        if installed and previous is not None:
            signal.signal(signal.SIGTERM, previous)


class _SigtermStop:
    """Cancels ``run_main``'s ``main`` on the first SIGTERM, and only the first."""

    def __init__(self) -> None:
        self._task: asyncio.Task[Any] | None = None
        self._terminated = False

    def terminate(self) -> None:
        log = logging.getLogger(__name__)
        if self._terminated:
            log.warning("SIGTERM again; still stopping")
            return
        self._terminated = True
        if self._task is None or self._task.done():
            log.warning("SIGTERM; the run has ended, finishing its shutdown")
            return
        log.warning("SIGTERM; stopping as on Ctrl-C")
        self._task.cancel()

    async def run[T](self, main: Coroutine[Any, Any, T]) -> T:
        self._task = asyncio.current_task()
        try:
            return await main
        except asyncio.CancelledError:
            if self._terminated:
                raise Terminated from None
            raise


LOOPS_RUN_ENDED_EXIT_CODE = 75
"""``EX_TEMPFAIL``: the Loops run ended under the run, which can resume from its latest
checkpoint manifest. A wrapper that retries on it retries only what can work."""


def exit_code(status: str) -> int:
    """A finished run's exit code: 0 completed, 75 its Loops run ended, 1 otherwise."""
    if status == "completed":
        return 0
    return LOOPS_RUN_ENDED_EXIT_CODE if status == "loops_run_ended" else 1


# Settings that name a local path or a Hugging Face repo id; only a path is resolved.
# It is in the shared [model] section, so every recipe has it.
PATHS_OR_NAMES = ("model.tokenizer",)


class Report(Protocol):
    """What a recipe's check found, as the lines ``loops check`` prints."""

    def describe(self) -> list[str]: ...


class Outcome(Protocol):
    """How a recipe's run ended, as ``loops run`` reports it."""

    @property
    def status(self) -> str: ...
    @property
    def steps(self) -> int: ...
    @property
    def reason(self) -> str | None: ...
    @property
    def ok(self) -> bool: ...


@dataclass(frozen=True)
class Recipe[C: BaseModel]:
    """What one recipe's command needs; ``loops`` does the rest the same for every recipe."""

    name: str
    config: type[C]
    check: Callable[[C, ConfigSource], Coroutine[Any, Any, Report]]
    """Check a loaded config, given where it came from; launches nothing and prints
    nothing. ``loops check`` prints the report's ``describe()``."""
    train: Callable[[C, ConfigSource], Coroutine[Any, Any, Outcome]]
    """Train with a loaded config, given where it came from; returns the run's
    ``Outcome``. The run's log records the source."""
    references: tuple[str, ...]
    """Settings that name files, resolved against whoever named them."""
    errors: tuple[type[Exception], ...] = ()
    """What else means the config or its environment is wrong, beyond the errors ``loops``
    answers for every recipe: exit 2, with the message."""

    def load(self, path: Path, overrides: Mapping[str, Any] | None = None) -> C:
        """This recipe's settings from ``path``, with flag ``overrides`` (``load_config``)."""
        return load_config(self.config, path, overrides, self.references, PATHS_OR_NAMES)
