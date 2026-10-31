# ADDED: new file, not part of the upstream SR3 codebase.
from collections.abc import Iterator, Mapping  # ADDED

import pytest  # ADDED
import torch  # ADDED
from torch import nn  # ADDED

from sr3_ablation.sampling import (  # ADDED
    PURE_NOISE,  # ADDED
    ForwardNoisedStart,  # ADDED
    ReverseSampler,  # ADDED
    TrajectoryRecorder,  # ADDED
    save_trajectory,  # ADDED
)  # ADDED
from sr3_ablation.training import (  # ADDED
    FineTuner,  # ADDED
    LossCsvLogger,  # ADDED
    NonFiniteLossError,  # ADDED
    TrainingCallback,  # ADDED
)  # ADDED


class FakeModel:  # ADDED
    """One-parameter model with loss (w - 3)^2, enough to exercise the loop."""  # ADDED

    def __init__(self, loss_override: float | None = None) -> None:  # ADDED
        self.weight = nn.Parameter(torch.zeros(()))  # ADDED
        self._training = False  # ADDED
        self._loss_override = loss_override  # ADDED

    @property  # ADDED
    def device(self) -> torch.device:  # ADDED
        return torch.device("cpu")  # ADDED

    @property  # ADDED
    def is_training(self) -> bool:  # ADDED
        return self._training  # ADDED

    def train_mode(self) -> None:  # ADDED
        self._training = True  # ADDED

    def eval_mode(self) -> None:  # ADDED
        self._training = False  # ADDED

    def parameters(self) -> Iterator[nn.Parameter]:  # ADDED
        return iter([self.weight])  # ADDED

    def training_loss(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:  # ADDED
        if self._loss_override is not None:  # ADDED
            return self.weight * 0 + self._loss_override  # ADDED
        return (self.weight - 3.0) ** 2  # ADDED


class CountingOptimizer(torch.optim.SGD):  # ADDED
    def __init__(self, params) -> None:  # ADDED
        super().__init__(params, lr=0.1)  # ADDED
        self.steps = 0  # ADDED

    def step(self, closure=None):  # ADDED
        self.steps += 1  # ADDED
        return super().step(closure)  # ADDED


class Recorder(TrainingCallback):  # ADDED
    def __init__(self) -> None:  # ADDED
        self.events: list[str] = []  # ADDED

    def on_train_start(self) -> None:  # ADDED
        self.events.append("start")  # ADDED

    def on_epoch_end(self, result) -> None:  # ADDED
        self.events.append(f"epoch{result.epoch}")  # ADDED

    def on_train_end(self, results) -> None:  # ADDED
        self.events.append("end")  # ADDED


BATCHES = [{"HR": torch.zeros(1)}] * 4  # ADDED


def test_trainer_runs_epochs_and_callbacks(tmp_path):  # ADDED
    model, recorder = FakeModel(), Recorder()  # ADDED
    optimizer = CountingOptimizer(model.parameters())  # ADDED
    csv_path = tmp_path / "loss.csv"  # ADDED
    results = FineTuner(  # ADDED
        model,  # ADDED
        optimizer,  # ADDED
        BATCHES,  # ADDED
        epochs=3,  # ADDED
        callbacks=[recorder, LossCsvLogger(csv_path)],  # ADDED
    ).fit()  # ADDED
    assert recorder.events == ["start", "epoch1", "epoch2", "epoch3", "end"]  # ADDED
    assert optimizer.steps == 12  # ADDED
    assert results[-1].train_loss < results[0].train_loss  # ADDED
    lines = csv_path.read_text(encoding="utf-8").splitlines()  # ADDED
    assert lines[0] == "epoch,train_loss"  # ADDED
    assert [line.split(",")[0] for line in lines[1:]] == ["1", "2", "3"]  # ADDED


def test_gradient_accumulation_and_iteration_cap():  # ADDED
    model = FakeModel()  # ADDED
    optimizer = CountingOptimizer(model.parameters())  # ADDED
    trainer = FineTuner(model, optimizer, BATCHES, epochs=5, grad_accum_steps=3, max_iterations=6)  # ADDED
    results = trainer.fit()  # ADDED
    assert trainer.iteration == 6  # ADDED
    assert [r.iterations for r in results] == [4, 2]  # ADDED
    assert optimizer.steps == 2 + 1  # ADDED


def test_non_finite_loss_stops_training():  # ADDED
    model = FakeModel(loss_override=float("nan"))  # ADDED
    with pytest.raises(NonFiniteLossError):  # ADDED
        FineTuner(model, CountingOptimizer(model.parameters()), BATCHES, epochs=1).fit()  # ADDED


class FakeDenoiser(FakeModel):  # ADDED
    """Halves its input at every step, so the state after the step at index i is x_T / 2^(T-i)."""  # ADDED

    num_timesteps = 4  # ADDED

    def reverse_step(self, x_t, step_index, condition):  # ADDED
        assert not self.is_training  # ADDED
        return x_t * 0.5 + torch.rand_like(x_t) * 0  # ADDED


def test_sampler_visits_every_timestep_from_t_to_zero():  # ADDED
    denoiser = FakeDenoiser()  # ADDED
    seen: list[int] = []  # ADDED

    class Spy:  # ADDED
        def on_state(self, timestep, x_t):  # ADDED
            seen.append(timestep)  # ADDED

    ReverseSampler(denoiser, seed=0, show_progress=False).sample(torch.zeros(1, 3, 2, 2), [Spy()])  # ADDED
    assert seen == [4, 3, 2, 1, 0]  # ADDED


def test_sampler_is_seeded_restores_rng_and_mode(tmp_path):  # ADDED
    denoiser = FakeDenoiser()  # ADDED
    denoiser.train_mode()  # ADDED
    sampler = ReverseSampler(denoiser, seed=5, show_progress=False)  # ADDED
    condition = torch.zeros(2, 3, 2, 2)  # ADDED

    torch.manual_seed(123)  # ADDED
    expected_next = torch.rand(1)  # ADDED
    torch.manual_seed(123)  # ADDED
    recorder = TrajectoryRecorder([4, 2, 0])  # ADDED
    first = sampler.sample(condition, [recorder])  # ADDED
    assert torch.equal(torch.rand(1), expected_next)  # ADDED
    assert torch.equal(first, sampler.sample(condition))  # ADDED
    assert denoiser.is_training  # ADDED

    states = recorder.states  # ADDED
    assert list(states) == [4, 2, 0]  # ADDED
    torch.testing.assert_close(states[0], states[4] / 16)  # ADDED
    written = save_trajectory(states, "runA", tmp_path)  # ADDED
    assert sorted(p.name for p in written)[:2] == [  # ADDED
        "runA_t0000_sample1.png",  # ADDED
        "runA_t0000_sample2.png",  # ADDED
    ]  # ADDED
    assert len(written) == 6  # ADDED


def test_default_start_is_seeded_pure_noise():  # ADDED
    recorder = TrajectoryRecorder([4])  # ADDED
    condition = torch.full((2, 3, 2, 2), 0.7)  # ADDED
    ReverseSampler(FakeDenoiser(), seed=9, show_progress=False).sample(condition, [recorder])  # ADDED
    torch.manual_seed(9)  # ADDED
    assert torch.equal(recorder.states[4], torch.randn(condition.shape))  # ADDED


def test_forward_noised_start_shares_the_noise_of_the_pure_start():  # ADDED
    condition = torch.full((2, 3, 2, 2), 0.7)  # ADDED
    starts = [PURE_NOISE, ForwardNoisedStart(alpha_bar_t=0.9943)]  # ADDED
    initial = {}  # ADDED
    for start in starts:  # ADDED
        recorder = TrajectoryRecorder([4])  # ADDED
        sampler = ReverseSampler(FakeDenoiser(), seed=9, show_progress=False, start=start)  # ADDED
        sampler.sample(condition, [recorder])  # ADDED
        initial[start.name] = recorder.states[4]  # ADDED
    forward = starts[1]  # ADDED
    assert forward.signal_coefficient == pytest.approx(0.9943**0.5)  # ADDED
    assert forward.noise_coefficient == pytest.approx(0.0057**0.5)  # ADDED
    torch.testing.assert_close(  # ADDED
        initial["forward_noised"],  # ADDED
        forward.signal_coefficient * condition + forward.noise_coefficient * initial["pure_noise"],  # ADDED
    )  # ADDED


@pytest.mark.parametrize("alpha_bar_t", [0.0, 1.0, -0.1])  # ADDED
def test_forward_noised_start_rejects_degenerate_alpha_bar(alpha_bar_t):  # ADDED
    with pytest.raises(ValueError, match="alpha_bar_T"):  # ADDED
        ForwardNoisedStart(alpha_bar_t)  # ADDED
