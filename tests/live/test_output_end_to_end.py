"""A real trial's agent gets a long command output whole: one Loops step on Blaxel.

``loops run`` trains the fixture job in ``output_job/``. Its agent prints 200 KB
through Harbor's ``exec`` and records whether every byte arrived; the verifier gives
half the reward for that. So every graded rollout scores at least 0.5 exactly when
the output reached the agent whole. The Loops run is deactivated and the run's
resources proven released, whatever the outcome.
"""

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.live_loops

JOB = Path(__file__).parent / "output_job"
LOOPSRL = str(Path(sys.executable).parent / "loops")


def deactivate(run_id: str) -> None:
    request = urllib.request.Request(
        f"https://api.baseten.co/v1/loops/runs/{run_id}/deactivate",
        method="POST",
        headers={"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"},
    )
    try:
        urllib.request.urlopen(request, timeout=60).close()
    except urllib.error.HTTPError as error:
        if error.code != 400:  # 400: the run is already inactive
            raise


def graded_rewards(run: Path) -> list[float]:
    rewards = []
    for result in run.glob("runtime/trials/*/*/result.json"):
        verifier = json.loads(result.read_text()).get("verifier_result") or {}
        reward = (verifier.get("rewards") or {}).get("reward")
        if reward is not None:
            rewards.append(reward)
    return rewards


def test_every_graded_rollout_received_its_command_output_whole(tmp_path):
    log = tmp_path / "run.log"
    with log.open("w") as out:
        finished = subprocess.run(
            [
                LOOPSRL,
                "run",
                str(JOB / "train.toml"),
                "--output.output_dir",
                str(tmp_path),
            ],
            stdout=out,
            stderr=subprocess.STDOUT,
            timeout=2700,
            check=False,
        )
    text = log.read_text()
    provisioned = re.search(r"Provisioned Loops run (\w+)", text)
    if provisioned:
        deactivate(provisioned.group(1))
    (run,) = [path for path in tmp_path.iterdir() if path.is_dir()]
    cleanup = subprocess.run(
        [
            LOOPSRL,
            "cleanup",
            str(JOB / "train.toml"),
            "--state-dir",
            str(run / "state"),
        ],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert cleanup.returncode == 0, cleanup.stdout + cleanup.stderr
    assert "infrastructure failures" not in text, log
    rewards = graded_rewards(run)
    assert rewards, f"no rollout was graded; see {log}"
    assert all(reward >= 0.5 for reward in rewards), (rewards, str(log))
    # One step trains, or the run stops once every group scores alike; either is a pass.
    assert finished.returncode == 0 or "consecutive unusable groups" in text, str(log)
