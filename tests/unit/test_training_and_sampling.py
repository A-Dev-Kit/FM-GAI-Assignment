# ADDED: new file, not part of the upstream SR3 codebase.
from collections.abc import Iterator, Mapping

import pytest
import torch
from torch import nn

from sr3_ablation.sampling import ReverseSampler, TrajectoryRecorder, save_trajectory
from sr3_ablation.training import (
    FineTuner,
    LossCsvLogger,
    NonFiniteLossError,
    TrainingCallback,
)


class FakeModel:
    """One-parameter model with loss (w - 3)^2, enough to exercise the loop."""

    def __init__(self, loss_override: float | None = None) -> None:
        self.weight = nn.Parameter(torch.zeros(()))
        self._training = False
        self._loss_override = loss_override

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def is_training(self) -> bool:
        return self._training

    def train_mode(self) -> None:
        self._training = True

    def eval_mode(self) -> None:
        self._training = False

    def parameters(self) -> Iterator[nn.Parameter]:
        return iter([self.weight])

    def training_loss(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self._loss_override is not None:
            return self.weight * 0 + self._loss_override
        return (self.weight - 3.0) ** 2


class CountingOptimizer(torch.optim.SGD):
    def __init__(self, params) -> None:
        super().__init__(params, lr=0.1)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


class Recorder(TrainingCallback):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_train_start(self) -> None:
        self.events.append("start")

    def on_epoch_end(self, result) -> None:
        self.events.append(f"epoch{result.epoch}")

    def on_train_end(self, results) -> None:
        self.events.append("end")


BATCHES = [{"HR": torch.zeros(1)}] * 4


def test_trainer_runs_epochs_and_callbacks(tmp_path):
    model, recorder = FakeModel(), Recorder()
    optimizer = CountingOptimizer(model.parameters())
    csv_path = tmp_path / "loss.csv"
    results = FineTuner(
        model, optimizer, BATCHES, epochs=3, callbacks=[recorder, LossCsvLogger(csv_path)]
    ).fit()
    assert recorder.events == ["start", "epoch1", "epoch2", "epoch3", "end"]
    assert optimizer.steps == 12
    assert results[-1].train_loss < results[0].train_loss
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "epoch,train_loss"
    assert [line.split(",")[0] for line in lines[1:]] == ["1", "2", "3"]


def test_gradient_accumulation_and_iteration_cap():
    model = FakeModel()
    optimizer = CountingOptimizer(model.parameters())
    trainer = FineTuner(model, optimizer, BATCHES, epochs=5, grad_accum_steps=3, max_iterations=6)
    results = trainer.fit()
    assert trainer.iteration == 6
    assert [r.iterations for r in results] == [4, 2]
    assert optimizer.steps == 2 + 1


def test_non_finite_loss_stops_training():
    model = FakeModel(loss_override=float("nan"))
    with pytest.raises(NonFiniteLossError):
        FineTuner(model, CountingOptimizer(model.parameters()), BATCHES, epochs=1).fit()


class FakeDenoiser(FakeModel):
    """Halves its input at every step, so the state after the step at index i is x_T / 2^(T-i)."""

    num_timesteps = 4

    def reverse_step(self, x_t, step_index, condition):
        assert not self.is_training
        return x_t * 0.5 + torch.rand_like(x_t) * 0


def test_sampler_visits_every_timestep_from_t_to_zero():
    denoiser = FakeDenoiser()
    seen: list[int] = []

    class Spy:
        def on_state(self, timestep, x_t):
            seen.append(timestep)

    ReverseSampler(denoiser, seed=0, show_progress=False).sample(torch.zeros(1, 3, 2, 2), [Spy()])
    assert seen == [4, 3, 2, 1, 0]


def test_sampler_is_seeded_restores_rng_and_mode(tmp_path):
    denoiser = FakeDenoiser()
    denoiser.train_mode()
    sampler = ReverseSampler(denoiser, seed=5, show_progress=False)
    condition = torch.zeros(2, 3, 2, 2)

    torch.manual_seed(123)
    expected_next = torch.rand(1)
    torch.manual_seed(123)
    recorder = TrajectoryRecorder([4, 2, 0])
    first = sampler.sample(condition, [recorder])
    assert torch.equal(torch.rand(1), expected_next)
    assert torch.equal(first, sampler.sample(condition))
    assert denoiser.is_training

    states = recorder.states
    assert list(states) == [4, 2, 0]
    torch.testing.assert_close(states[0], states[4] / 16)
    written = save_trajectory(states, "runA", tmp_path)
    assert sorted(p.name for p in written)[:2] == [
        "runA_t0000_sample1.png",
        "runA_t0000_sample2.png",
    ]
    assert len(written) == 6
