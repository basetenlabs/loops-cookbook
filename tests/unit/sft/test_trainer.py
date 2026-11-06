"""NLL from Loops' logprobs, and what one update sends."""

import pytest
from baseten.loops.models import (
    Datum,
    ForwardBackwardOutput,
    ModelInput,
    TensorData,
)
from support.fakes import Future

from loops_cookbook.sft.trainer import Trainer, score


def datum(weights):
    length = len(weights)
    return Datum(
        model_input=ModelInput.from_ints(list(range(length))),
        loss_fn_inputs={
            "target_tokens": TensorData(data=[1] * length, dtype="int64", shape=[length]),
            "weights": TensorData(data=weights, dtype="float32", shape=[length]),
        },
    )


def output(*logprobs, metrics=None):
    return ForwardBackwardOutput(
        loss_fn_outputs=[
            {"logprobs": TensorData(data=list(values), dtype="float32", shape=[len(values)])}
            for values in logprobs
        ],
        metrics=metrics or {},
    )


def test_nll_is_the_mean_over_every_trained_token_not_over_datums():
    batch = [datum([0.0, 1.0]), datum([0.0, 1.0, 1.0, 1.0])]
    measured = score(batch, output([None, -4.0], [None, -1.0, -1.0, -1.0]))
    assert measured.trained_tokens == 4
    assert measured.nll == pytest.approx(7.0 / 4)


def test_an_untrained_position_returned_as_null_is_skipped():
    assert score([datum([0.0, 1.0, 0.0])], output([None, -2.0, None])).nll == 2.0


def test_a_trained_position_without_a_logprob_is_an_error():
    with pytest.raises(RuntimeError, match="no finite logprob"):
        score([datum([0.0, 1.0])], output([None, None]))


class Client:
    def __init__(self):
        self.calls = []
        self.policy_version = 0

    async def forward_backward_async(self, datums, loss_fn, loss_fn_config=None):
        self.calls.append(("forward_backward", loss_fn, len(datums)))
        self.sent = (datums, loss_fn_config)
        summed = (loss_fn_config or {}).get("skip_token_normalization", True)
        return Future(
            output(
                *[[None] + [-1.0] * (d.model_input.length - 1) for d in datums],
                metrics={"skip_token_normalization": float(summed)},
            )
        )

    async def forward_async(self, datums, loss_fn):
        self.calls.append(("forward", loss_fn, len(datums)))
        return Future(output(*[[None, -3.0]] * len(datums)))

    async def optim_step_async(self, params):
        self.calls.append(("optim_step", params))
        version = self.policy_version = self.policy_version + 1
        metrics = {"grad_norm": 0.25, "tps": 9.0, "other": 1.0, "step": version}
        return Future(type("Result", (), {"metrics": metrics})())


async def test_an_update_accumulates_the_batch_then_steps_once(sft_config):
    optimizer = {"adam_beta1": 0.8, "adam_beta2": 0.99, "weight_decay": 0.01}
    trainer = Trainer(
        sft_config(training={"grad_clip_norm": 0.5, "adam_epsilon": 1e-8, **optimizer})
    )
    trainer.client = client = Client()
    update = await trainer.update(3, [datum([0.0, 1.0])] * 2, learning_rate=2e-4)

    assert [call[0] for call in client.calls] == ["forward_backward", "optim_step"]
    assert client.calls[0][1] == "cross_entropy"
    params = client.calls[1][1]
    assert (params.learning_rate, params.grad_clip_norm, params.eps) == (2e-4, 0.5, 1e-8)
    assert (params.beta1, params.beta2, params.weight_decay) == (0.8, 0.99, 0.01)
    assert update.score.nll == 1.0 and update.sequences == 2
    assert update.optimizer == {"grad_norm": 0.25, "tps": 9.0}


@pytest.mark.parametrize(
    ("training", "skip"), [({}, True), ({"loss_aggregation": "token_mean"}, False)]
)
async def test_an_update_sends_its_rows_and_asks_loops_for_its_reduction(
    sft_config, training, skip
):
    trainer = Trainer(sft_config(training=training))
    trainer.client = client = Client()
    batch = [datum([0.0, 1.0]), datum([0.0, 1.0, 1.0, 1.0])]
    await trainer.update(1, batch, learning_rate=2e-4)
    sent, settings = client.sent
    assert settings == {"skip_token_normalization": skip}
    assert sent == batch


async def test_validation_runs_forward_only(sft_config):
    trainer = Trainer(sft_config())
    trainer.client = client = Client()
    assert (await trainer.evaluate([datum([0.0, 1.0])])).nll == 3.0
    assert [call[0] for call in client.calls] == ["forward"]
