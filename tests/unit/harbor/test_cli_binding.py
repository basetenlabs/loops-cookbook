"""loops check on a Harbor job: an agent that cannot be bound is a usage error."""

from pathlib import Path

import yaml
from support.configs import MINIMAL, write_toml

HILL_CLIMB = Path(__file__).resolve().parents[3] / "tasks" / "hill_climb"


def test_check_refuses_a_job_whose_agent_cannot_be_bound(loops_check, tmp_path):
    job = tmp_path / "job.yaml"
    job.write_text(
        yaml.safe_dump(
            {
                "tasks": [{"path": str(HILL_CLIMB)}],
                "agents": [{"import_path": "harbor_test_agents:UnboundAgent"}],
                "environment": {"import_path": "baseline_environment:LocalEnvironment"},
                "retry": {"max_retries": 0},
            }
        )
    )
    config = write_toml(
        tmp_path / "train.toml",
        MINIMAL,
        {
            "runtime": {
                "factory": "loops_cookbook.harbor:create_runtime",
                "files": {"job": job},
            },
            "model": {"base_model": "m"},
            "rollout": {
                "rollout_timeout_seconds": 600,
                "setup_timeout_seconds": 600,
                "grading": {"timeout_seconds": 900},
            },
            "output": {"run_name": "run", "output_dir": tmp_path / "runs"},
        },
    )
    code, _, err = loops_check(config)
    assert code == 2
    assert "must implement" in err and "Traceback" not in err
