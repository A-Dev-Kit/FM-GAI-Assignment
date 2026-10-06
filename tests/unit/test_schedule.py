# ADDED: new file, not part of the upstream SR3 codebase.
import numpy as np
import pytest

from sr3_ablation.config import load_config
from sr3_ablation.schedule import (
    IdentityTransform,
    NoiseSchedule,
    PowerTransform,
    ScheduleBuffers,
    ScheduleMismatchError,
    assert_open_unit_interval,
    base_betas,
    transform_for,
    verify_schedule,
)


@pytest.fixture
def schedule_config():
    return load_config(local_config=None).schedule


def test_base_schedule_matches_pretrained_checkpoint(schedule_config):
    betas = base_betas(schedule_config)
    assert betas.shape == (2000,)
    assert betas[0] == pytest.approx(1e-6)
    assert betas[-1] == pytest.approx(1e-2)
    assert np.all(np.diff(betas) > 0)


def test_power_transform_values_and_range():
    theta = np.array([1e-6, 0.25, 1e-2])
    np.testing.assert_allclose(PowerTransform(1.6).apply(theta), theta**1.6)
    with pytest.raises(ValueError):
        PowerTransform(1.5)
    with pytest.raises(ValueError):
        PowerTransform(3.0)
    with pytest.raises(ValueError):
        PowerTransform(2.0).apply(np.array([0.5, 1.0]))


def test_transform_for_selects_strategy():
    assert isinstance(transform_for(1.0), IdentityTransform)
    assert isinstance(transform_for(1.6), PowerTransform)
    assert transform_for(1.6).exponent == 1.6


def test_identity_returns_a_copy():
    theta = np.array([0.1, 0.2])
    result = IdentityTransform().apply(theta)
    result[0] = 0.9
    assert theta[0] == 0.1


def test_open_unit_interval_check():
    assert_open_unit_interval(np.array([1e-9, 0.999]))
    with pytest.raises(ValueError):
        assert_open_unit_interval(np.array([0.0, 0.5]))


def test_declared_numbers_from_phase1(schedule_config):
    original = NoiseSchedule.from_config(schedule_config, IdentityTransform())
    modified = NoiseSchedule.from_config(schedule_config, PowerTransform(1.6))
    assert original.summary().alpha_bar_last == pytest.approx(4.386e-5, rel=1e-3)
    assert original.first_t_snr_below_one() == 527
    assert modified.summary().alpha_bar_last == pytest.approx(0.6153, abs=1e-4)
    assert modified.summary().snr_last == pytest.approx(1.60, abs=1e-2)
    assert modified.first_t_snr_below_one() is None


def test_power_schedule_raises_snr_everywhere(schedule_config):
    original = NoiseSchedule.from_config(schedule_config, IdentityTransform())
    modified = NoiseSchedule.from_config(schedule_config, PowerTransform(1.6))
    assert np.all(modified.snr > original.snr)
    assert np.all(np.diff(original.snr) < 0)


def test_timesteps_are_one_indexed(schedule_config):
    schedule = NoiseSchedule.from_config(schedule_config, IdentityTransform())
    assert schedule.alpha_bar_at(1) == pytest.approx(1 - 1e-6)
    assert schedule.sqrt_alphas_cumprod_prev[0] == 1.0
    assert schedule.sqrt_alphas_cumprod_prev.shape == (2001,)
    with pytest.raises(IndexError):
        schedule.snr_at(0)


def _buffers(schedule: NoiseSchedule) -> ScheduleBuffers:
    return ScheduleBuffers(
        betas=schedule.betas.astype(np.float32),
        alphas_cumprod=schedule.alphas_cumprod.astype(np.float32),
        sqrt_alphas_cumprod_prev=schedule.sqrt_alphas_cumprod_prev,
    )


def test_verification_accepts_float32_copy(schedule_config):
    schedule = NoiseSchedule.from_config(schedule_config, PowerTransform(1.6))
    evidence = verify_schedule(_buffers(schedule), schedule)
    assert evidence["verified"] is True
    assert evidence["description"] == "theta_t ** 1.60"


def test_verification_rejects_other_schedule(schedule_config):
    original = NoiseSchedule.from_config(schedule_config, IdentityTransform())
    modified = NoiseSchedule.from_config(schedule_config, PowerTransform(1.6))
    with pytest.raises(ScheduleMismatchError):
        verify_schedule(_buffers(original), modified)
