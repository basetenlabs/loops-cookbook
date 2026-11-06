from pathlib import Path

import pytest
import yaml
from live.test_trainer_normalization import contract_groups, explicit_cispo
from support.builders import make_config

from loops_cookbook.rl.aggregation import BUILTIN_NORMALIZATION_CONTRACT, aggregate_loss


def test_live_fixture_distinguishes_action_active_and_masked_tokens():
    groups = contract_groups()
    counts, active, zero, ignored = [], 0, 0, 0
    for group in groups:
        count = 0
        for row in group:
            for target, advantage in zip(
                row.datum.loss_fn_inputs["target_tokens"].data,
                row.datum.loss_fn_inputs["advantages"].data,
                strict=True,
            ):
                if target == -100:
                    ignored += advantage != 0
                else:
                    count += 1
                    active += advantage != 0
                    zero += advantage == 0
        counts.append(count)
        assert count == sum(row.trained_positions for row in group)
    assert counts == [6, 8]
    assert active == 12 and zero == 2 and ignored > 0
    batch = aggregate_loss(groups, make_config().training)
    assert batch.report.backend_normalization == BUILTIN_NORMALIZATION_CONTRACT


def test_live_oracle_masks_ignored_and_zero_advantage_positions():
    torch = pytest.importorskip("torch")
    rows = [row.datum for group in contract_groups() for row in group]
    probabilities = [
        torch.full((datum.model_input.length,), -0.1, requires_grad=True) for datum in rows
    ]
    denominators = [12] * 3 + [16] * 2
    loss, _ = explicit_cispo(rows, probabilities, denominators)
    loss.backward()
    for datum, lp, denominator in zip(rows, probabilities, denominators, strict=True):
        expected = [
            -advantage * torch.exp(torch.tensor(-0.1)).item() / denominator if target != -100 else 0
            for target, advantage in zip(
                datum.loss_fn_inputs["target_tokens"].data,
                datum.loss_fn_inputs["advantages"].data,
                strict=True,
            )
        ]
        assert lp.grad.tolist() == pytest.approx(expected)


def test_live_normalization_is_manual_and_separate_from_releasing():
    root = Path(__file__).resolve().parents[3]
    release = root / ".github/workflows/release.yml"
    assert "test_trainer_normalization.py" not in release.read_text()
    workflow = root / ".github/workflows/trainer-normalization.yml"
    config = yaml.safe_load(workflow.read_text())
    # PyYAML's YAML 1.1 reader treats the unquoted GitHub Actions key 'on' as True.
    triggers = config.get("on", config.get(True))
    assert set(triggers) == {"workflow_dispatch"}
    steps = config["jobs"]["normalization"]["steps"]
    gate = next(step for step in steps if "test_trainer_normalization.py" in step.get("run", ""))
    assert "if" not in gate and not gate.get("continue-on-error")
    assert gate["env"]["LOOPS_LIVE_LOOPS"] == "1"
    assert gate["env"]["BASETEN_API_KEY"] == "${{ secrets.BASETEN_API_KEY }}"
    assert "--extra custom-loss" in gate["run"]
    assert any('[[ -z "$BASETEN_API_KEY" ]]' in step.get("run", "") for step in steps)
