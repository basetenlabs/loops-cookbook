"""Every training config the repository ships passes ``loops check``, offline.

``check`` is the first thing a user runs: it resolves the config, imports the rollout runtime
or the Harbor job's agent, and discovers the tasks, launching nothing. A config that
names a moved file or a renamed setting fails here, not in a user's first command.
"""

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest
from support.builders import task_file

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted(
    path
    for path in [
        *ROOT.glob("patterns/*/*/train*.toml"),
        *ROOT.glob("examples/*/train*.toml"),
        *ROOT.glob("tests/live/*/train*.toml"),
    ]
    # Terminal-Bench's check downloads its dataset, so it is not offline.
    if path.parent.name != "terminal_bench"
)


def skip_without_litellm(path):
    """LiteLLM's example imports it, and only the Harbor extras bring it."""
    if "litellm" in str(path) and importlib.util.find_spec("litellm") is None:
        pytest.skip("LiteLLM comes with the Harbor extras")


@pytest.mark.parametrize("path", CONFIGS, ids=[str(p.relative_to(ROOT)) for p in CONFIGS])
def test_the_config_passes_check(path, loops_check):
    skip_without_litellm(path)
    if "harbor" in path.read_text() and importlib.util.find_spec("harbor") is None:
        pytest.skip("needs the harbor extra")
    code, out, err = loops_check(path, cwd=ROOT)
    assert code == 0, err
    assert "Rollout runtime:" in out


def test_no_shipped_config_states_the_model_access_its_agent_class_states():
    import tomllib

    stated = [
        str(path.relative_to(ROOT))
        for path in CONFIGS
        if "model_access" in tomllib.loads(path.read_text()).get("rollout_runtime", {})
    ]
    assert stated == []


# Each custom rollout runtime's dataset is its episode's own ``AnswerTask``, one per line.
ANSWER_TASK_CONFIGS = [
    "patterns/custom_runtime/in_process_direct/train.toml",
    "patterns/custom_runtime/in_process_openai_http/train.toml",
    "patterns/custom_runtime/local/train.toml",
    "examples/litellm_streaming/train.toml",
]
TASK = {"name": "b", "instruction": "Say B", "answer": "B"}


@pytest.mark.parametrize(
    "line",
    [{"name": "x", "instruction": "Say B"}, {**TASK, "anwser": "B"}],
    ids=["no_answer", "misspelt_key"],
)
@pytest.mark.parametrize("config", ANSWER_TASK_CONFIGS)
def test_check_refuses_a_dataset_line_that_is_not_a_task_naming_it(
    loops_check, tmp_path, config, line
):
    skip_without_litellm(config)
    tasks = task_file(tmp_path, [TASK, line])
    code, _, err = loops_check(
        ROOT / config, "--rollout_runtime.files", json.dumps({"tasks": str(tasks)})
    )
    assert code == 2
    assert f"{tasks}:2: the task is not a valid AnswerTask" in err


def test_every_shipped_blaxel_config_bounds_blaxel_the_same_way():
    """One provider, one set of deadlines: a value that differs between two configs is
    one of them guessing, since the Blaxel operation is the same under any runtime."""
    import tomllib

    shipped = [*ROOT.glob("patterns/*/*/train*.toml"), *ROOT.glob("examples/*/train*.toml")]
    tables = {
        str(path.relative_to(ROOT)): table
        for path in shipped
        if (table := tomllib.loads(path.read_text()).get("providers", {}).get("blaxel"))
    }
    assert len(tables) > 1
    first = next(iter(tables.values()))
    assert {name: table for name, table in tables.items() if table != first} == {}


def test_every_shipped_harbor_job_names_its_environment_once():
    """Harbor builds the class ``import_path`` names and never reads ``type`` beside it,
    so a job that sets both states one fact twice, and the second says nothing."""
    import yaml

    jobs = [
        *ROOT.glob("patterns/harbor/*/job*.yaml"),
        *ROOT.glob("examples/*/job*.yaml"),
        *ROOT.glob("examples/*/harbor.yaml"),
        *ROOT.glob("tests/live/*/job*.yaml"),
    ]
    environments = {
        str(path.relative_to(ROOT)): yaml.safe_load(path.read_text())["environment"]
        for path in jobs
    }
    assert any("import_path" in environment for environment in environments.values())
    both = [name for name, env in environments.items() if "type" in env and "import_path" in env]
    assert both == []


# A commented setting is a line ``# key = value`` inside a table: uncommenting it is
# what a reader does, so it must name a setting of that table.
COMMENTED_SETTING = re.compile(r"^#\s*([A-Za-z_][\w-]*\s*=.*)$")
TABLE = re.compile(r"^\[[^\]]+\]")
TEMPLATE = ROOT / "patterns" / "harbor" / "train_template.toml"
# The template is copied next to a Harbor job; this is the shipped job it sits beside.
TEMPLATE_HOME = ROOT / "patterns" / "harbor" / "in_process_direct"


def _uncommented(text: str) -> tuple[str, list[str]]:
    lines, settings, in_table = [], [], False
    for line in text.splitlines():
        in_table = in_table or bool(TABLE.match(line))
        if in_table and (match := COMMENTED_SETTING.match(line)):
            line = match.group(1)
            settings.append(line)
        lines.append(line)
    return "\n".join(lines) + "\n", settings


COMMENTED = [path for path in [*CONFIGS, TEMPLATE] if _uncommented(path.read_text())[1]]


def test_the_shipped_configs_comment_settings():
    assert TEMPLATE in COMMENTED


@pytest.mark.parametrize("path", COMMENTED, ids=[str(p.relative_to(ROOT)) for p in COMMENTED])
def test_the_config_passes_check_with_every_setting_uncommented(path, loops_check, tmp_path):
    """A commented line is an offer: uncommenting it must leave a config ``check``
    accepts, so each one sits in the table that owns the setting."""
    if "harbor" in path.read_text() and importlib.util.find_spec("harbor") is None:
        pytest.skip("needs the harbor extra")
    text, _ = _uncommented(path.read_text())
    home = TEMPLATE_HOME if path == TEMPLATE else path.parent
    # A copy of the config's directory, where it sits in the repository, so its
    # relative files and imports resolve as shipped; the shared tasks are the repository's.
    mirror = tmp_path / "repository"
    copy = mirror / home.relative_to(ROOT)
    shutil.copytree(home, copy, ignore=shutil.ignore_patterns("__pycache__"))
    (mirror / "tasks").symlink_to(ROOT / "tasks")
    variant = copy / "uncommented.toml"
    variant.write_text(text)
    code, out, err = loops_check(variant, cwd=ROOT)
    assert code == 0, err
    assert "Rollout runtime:" in out
