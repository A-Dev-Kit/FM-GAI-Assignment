# ADDED: new file, not part of the upstream SR3 codebase.
"""The vendored SR3 code with its one-line change, driven through the adapter on a CPU."""

import numpy as np
import pytest
import torch

from sr3_ablation.config import load_config
from sr3_ablation.model import Sr3Backend, build_backend, read_denoiser_state, save_denoiser_state
from sr3_ablation.model.checkpoints import extract_denoiser_state
from sr3_ablation.schedule import (
    NoiseSchedule,
    ScheduleMismatchError,
    transform_for,
    verify_schedule,
)
from tests.helpers import DECLARED_EXPONENT

CPU = torch.device("cpu")
UPSTREAM_SCHEDULE_BUFFERS = (
    "betas",
    "alphas_cumprod",
    "alphas_cumprod_prev",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
    "log_one_minus_alphas_cumprod",
    "sqrt_recip_alphas_cumprod",
    "sqrt_recipm1_alphas_cumprod",
    "posterior_variance",
    "posterior_log_variance_clipped",
    "posterior_mean_coef1",
    "posterior_mean_coef2",
)


@pytest.fixture
def backend(tiny_config) -> Sr3Backend:
    return build_backend(
        tiny_config.model, tiny_config.schedule, DECLARED_EXPONENT, CPU, weights=None
    )


def test_upstream_without_exponent_key_keeps_original_schedule(tiny_config):
    backend = Sr3Backend.create(tiny_config.model, tiny_config.schedule, CPU)
    options = tiny_config.schedule.to_upstream()
    del options["exponent"]
    backend.network.set_new_noise_schedule(options, CPU)
    expected = NoiseSchedule.from_config(tiny_config.schedule, transform_for(1.0))
    assert verify_schedule(backend.schedule_buffers(), expected)["verified"]


def test_added_line_applies_the_power(tiny_config, backend):
    expected = NoiseSchedule.from_config(tiny_config.schedule, transform_for(DECLARED_EXPONENT))
    buffers = backend.schedule_buffers()
    assert verify_schedule(buffers, expected)["verified"]
    np.testing.assert_allclose(
        buffers.betas, np.linspace(1e-6, 1e-2, 10) ** DECLARED_EXPONENT, rtol=1e-5
    )
    with pytest.raises(ScheduleMismatchError):
        verify_schedule(
            buffers, NoiseSchedule.from_config(tiny_config.schedule, transform_for(1.0))
        )


def test_declared_schedule_buffers_are_finite_in_float32(tiny_config):
    """beta_1 ** x = 1e-15 must not produce NaN or inf in any float32 buffer SR3 uses."""
    full_schedule = load_config(local_config=None).schedule
    backend = build_backend(tiny_config.model, full_schedule, DECLARED_EXPONENT, CPU, None)
    expected = NoiseSchedule.from_config(full_schedule, transform_for(DECLARED_EXPONENT))
    assert verify_schedule(backend.schedule_buffers(), expected)["verified"]
    for name in UPSTREAM_SCHEDULE_BUFFERS:
        buffer = getattr(backend.network, name)
        assert buffer.dtype == torch.float32
        assert torch.isfinite(buffer).all(), name


def test_training_loss_is_normalised_and_differentiable(backend):
    backend.train_mode()
    batch = {"HR": torch.rand(2, 3, 16, 16) * 2 - 1, "SR": torch.rand(2, 3, 16, 16) * 2 - 1}
    loss = backend.training_loss(batch)
    loss.backward()
    assert loss.ndim == 0 and 0.0 < loss.item() < 5.0
    assert any(p.grad is not None for p in backend.parameters())


def test_reverse_step_keeps_shape(backend):
    backend.eval_mode()
    x = torch.randn(2, 3, 16, 16)
    assert backend.reverse_step(x, backend.num_timesteps - 1, torch.zeros_like(x)).shape == x.shape


def test_loading_weights_never_restores_checkpoint_schedule(tiny_config, tmp_path):
    """An upstream-style checkpoint stores schedule buffers; only UNet weights may be used."""
    source = build_backend(tiny_config.model, tiny_config.schedule, 1.0, CPU, weights=None)
    upstream_file = tmp_path / "gen.pth"
    torch.save(source.network.state_dict(), upstream_file)
    assert "betas" in torch.load(upstream_file, weights_only=True)

    loaded = build_backend(
        tiny_config.model, tiny_config.schedule, DECLARED_EXPONENT, CPU, upstream_file
    )
    expected = NoiseSchedule.from_config(tiny_config.schedule, transform_for(DECLARED_EXPONENT))
    assert verify_schedule(loaded.schedule_buffers(), expected)["verified"]
    for name, tensor in source.denoiser_state().items():
        torch.testing.assert_close(loaded.denoiser_state()[name], tensor)


def test_checkpoint_round_trip_and_prefix_handling(backend, tmp_path):
    path = tmp_path / "epoch01_gen.pth"
    save_denoiser_state(backend.denoiser_state(), path, {"epoch": 1})
    state = read_denoiser_state(path)
    assert state.keys() == backend.denoiser_state().keys()
    parallel = {f"module.denoise_fn.{k}": v for k, v in state.items()}
    assert extract_denoiser_state(parallel).keys() == state.keys()
    with pytest.raises(KeyError):
        extract_denoiser_state({"betas": torch.zeros(1)})
