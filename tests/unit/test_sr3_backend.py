# ADDED: new file, not part of the upstream SR3 codebase.
"""The vendored SR3 code with its one-line change, driven through the adapter on a CPU."""  # ADDED

import numpy as np  # ADDED
import pytest  # ADDED
import torch  # ADDED

from sr3_ablation.config import load_config  # ADDED
from sr3_ablation.model import Sr3Backend, build_backend, read_denoiser_state, save_denoiser_state  # ADDED
from sr3_ablation.model.checkpoints import extract_denoiser_state  # ADDED
from sr3_ablation.runs import run_spec  # ADDED
from sr3_ablation.schedule import (  # ADDED
    NoiseSchedule,  # ADDED
    ScheduleMismatchError,  # ADDED
    transform_for,  # ADDED
    verify_schedule,  # ADDED
)  # ADDED
from tests.helpers import DECLARED_EXPONENT  # ADDED

CPU = torch.device("cpu")  # ADDED
UPSTREAM_SCHEDULE_BUFFERS = (  # ADDED
    "betas",  # ADDED
    "alphas_cumprod",  # ADDED
    "alphas_cumprod_prev",  # ADDED
    "sqrt_alphas_cumprod",  # ADDED
    "sqrt_one_minus_alphas_cumprod",  # ADDED
    "log_one_minus_alphas_cumprod",  # ADDED
    "sqrt_recip_alphas_cumprod",  # ADDED
    "sqrt_recipm1_alphas_cumprod",  # ADDED
    "posterior_variance",  # ADDED
    "posterior_log_variance_clipped",  # ADDED
    "posterior_mean_coef1",  # ADDED
    "posterior_mean_coef2",  # ADDED
)  # ADDED


@pytest.fixture  # ADDED
def backend(tiny_config) -> Sr3Backend:  # ADDED
    return build_backend(  # ADDED
        tiny_config.model,  # ADDED
        tiny_config.schedule,  # ADDED
        run_spec("B", DECLARED_EXPONENT),  # ADDED
        CPU,  # ADDED
        weights=None,  # ADDED
    )  # ADDED


def test_upstream_without_exponent_key_keeps_original_schedule(tiny_config):  # ADDED
    backend = Sr3Backend.create(tiny_config.model, tiny_config.schedule, CPU)  # ADDED
    options = tiny_config.schedule.to_upstream()  # ADDED
    del options["exponent"]  # ADDED
    backend.network.set_new_noise_schedule(options, CPU)  # ADDED
    expected = NoiseSchedule.from_config(tiny_config.schedule, transform_for(1.0))  # ADDED
    assert verify_schedule(backend.schedule_buffers(), expected)["verified"]  # ADDED


def test_added_line_applies_the_power(tiny_config, backend):  # ADDED
    expected = NoiseSchedule.from_config(tiny_config.schedule, transform_for(DECLARED_EXPONENT))  # ADDED
    buffers = backend.schedule_buffers()  # ADDED
    assert verify_schedule(buffers, expected)["verified"]  # ADDED
    np.testing.assert_allclose(  # ADDED
        buffers.betas,  # ADDED
        np.linspace(1e-6, 1e-2, 10) ** DECLARED_EXPONENT,  # ADDED
        rtol=1e-5,  # ADDED
    )  # ADDED
    with pytest.raises(ScheduleMismatchError):  # ADDED
        verify_schedule(  # ADDED
            buffers,  # ADDED
            NoiseSchedule.from_config(tiny_config.schedule, transform_for(1.0)),  # ADDED
        )  # ADDED


def test_declared_schedule_buffers_are_finite_in_float32(tiny_config):  # ADDED
    """beta_1 ** x = 1e-15 must not produce NaN or inf in any float32 buffer SR3 uses."""  # ADDED
    full_schedule = load_config(local_config=None).schedule  # ADDED
    backend = build_backend(  # ADDED
        tiny_config.model,  # ADDED
        full_schedule,  # ADDED
        run_spec("B", DECLARED_EXPONENT),  # ADDED
        CPU,  # ADDED
        None,  # ADDED
    )  # ADDED
    expected = NoiseSchedule.from_config(full_schedule, transform_for(DECLARED_EXPONENT))  # ADDED
    assert verify_schedule(backend.schedule_buffers(), expected)["verified"]  # ADDED
    for name in UPSTREAM_SCHEDULE_BUFFERS:  # ADDED
        buffer = getattr(backend.network, name)  # ADDED
        assert buffer.dtype == torch.float32  # ADDED
        assert torch.isfinite(buffer).all(), name  # ADDED


def test_training_loss_is_normalised_and_differentiable(backend):  # ADDED
    backend.train_mode()  # ADDED
    batch = {"HR": torch.rand(2, 3, 16, 16) * 2 - 1, "SR": torch.rand(2, 3, 16, 16) * 2 - 1}  # ADDED
    loss = backend.training_loss(batch)  # ADDED
    loss.backward()  # ADDED
    assert loss.ndim == 0 and 0.0 < loss.item() < 5.0  # ADDED
    assert any(p.grad is not None for p in backend.parameters())  # ADDED


def test_reverse_step_keeps_shape(backend):  # ADDED
    backend.eval_mode()  # ADDED
    x = torch.randn(2, 3, 16, 16)  # ADDED
    assert backend.reverse_step(x, backend.num_timesteps - 1, torch.zeros_like(x)).shape == x.shape  # ADDED


def test_loading_weights_never_restores_checkpoint_schedule(tiny_config, tmp_path):  # ADDED
    """An upstream-style checkpoint stores schedule buffers; only UNet weights may be used."""  # ADDED
    source = build_backend(  # ADDED
        tiny_config.model,  # ADDED
        tiny_config.schedule,  # ADDED
        run_spec("A", DECLARED_EXPONENT),  # ADDED
        CPU,  # ADDED
        weights=None,  # ADDED
    )  # ADDED
    upstream_file = tmp_path / "gen.pth"  # ADDED
    torch.save(source.network.state_dict(), upstream_file)  # ADDED
    assert "betas" in torch.load(upstream_file, weights_only=True)  # ADDED

    loaded = build_backend(  # ADDED
        tiny_config.model,  # ADDED
        tiny_config.schedule,  # ADDED
        run_spec("B", DECLARED_EXPONENT),  # ADDED
        CPU,  # ADDED
        upstream_file,  # ADDED
    )  # ADDED
    expected = NoiseSchedule.from_config(tiny_config.schedule, transform_for(DECLARED_EXPONENT))  # ADDED
    assert verify_schedule(loaded.schedule_buffers(), expected)["verified"]  # ADDED
    for name, tensor in source.denoiser_state().items():  # ADDED
        torch.testing.assert_close(loaded.denoiser_state()[name], tensor)  # ADDED


def test_checkpoint_round_trip_and_prefix_handling(backend, tmp_path):  # ADDED
    path = tmp_path / "epoch01_gen.pth"  # ADDED
    save_denoiser_state(backend.denoiser_state(), path, {"epoch": 1})  # ADDED
    state = read_denoiser_state(path)  # ADDED
    assert state.keys() == backend.denoiser_state().keys()  # ADDED
    parallel = {f"module.denoise_fn.{k}": v for k, v in state.items()}  # ADDED
    assert extract_denoiser_state(parallel).keys() == state.keys()  # ADDED
    with pytest.raises(KeyError):  # ADDED
        extract_denoiser_state({"betas": torch.zeros(1)})  # ADDED
