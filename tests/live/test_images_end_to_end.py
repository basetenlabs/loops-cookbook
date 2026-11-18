"""Images reach the policy and train exactly: one GLM-5.3-Flash step on Loops, on Blaxel.

``loops run`` trains the job in ``vision_job/``: the agent's ``look`` tool returns
each card's PNG as an image part, over several turns, and the verifier scores the
answer. Loops cannot record an image prompt, so the job declares its images and trains
without router replay. The step training, or the run stopping on groups that all
scored the same, is the pass; either needs every graded rollout to have gone through
the image path. The Loops run is deactivated and its resources proven released, whatever the
outcome.
"""

import re
import subprocess
from pathlib import Path

import pytest

from .test_output_end_to_end import LOOPSRL, deactivate, graded_rewards

pytestmark = pytest.mark.live_loops

JOB = Path(__file__).parent / "vision_job"


def test_a_step_trains_on_images_returned_by_tools(tmp_path):
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
            timeout=3600,
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
    assert graded_rewards(run), f"no rollout was graded; see {log}"
    assert finished.returncode == 0 or "consecutive unusable groups" in text, str(log)
