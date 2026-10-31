# ADDED: new file, not part of the upstream SR3 codebase.
import numpy as np  # ADDED
import pytest  # ADDED

from sr3_ablation.config import load_config  # ADDED
from sr3_ablation.schedule import (  # ADDED
    AlphaBarPowerTransform,  # ADDED
    IdentityTransform,  # ADDED
    NoiseSchedule,  # ADDED
    PowerTransform,  # ADDED
    ScheduleBuffers,  # ADDED
    ScheduleMismatchError,  # ADDED
    assert_open_unit_interval,  # ADDED
    base_betas,  # ADDED
    transform_for,  # ADDED
    verify_schedule,  # ADDED
)  # ADDED
from tests.helpers import DECLARED_EXPONENT  # ADDED


@pytest.fixture  # ADDED
def schedule_config():  # ADDED
    return load_config(local_config=None).schedule  # ADDED


def test_base_schedule_matches_pretrained_checkpoint(schedule_config):  # ADDED
    betas = base_betas(schedule_config)  # ADDED
    assert betas.shape == (2000,)  # ADDED
    assert betas[0] == pytest.approx(1e-6)  # ADDED
    assert betas[-1] == pytest.approx(1e-2)  # ADDED
    assert np.all(np.diff(betas) > 0)  # ADDED


def test_power_transform_values_and_range():  # ADDED
    theta = np.array([1e-6, 0.25, 1e-2])  # ADDED
    np.testing.assert_allclose(  # ADDED
        PowerTransform(DECLARED_EXPONENT).apply(theta),  # ADDED
        theta**DECLARED_EXPONENT,  # ADDED
    )  # ADDED
    with pytest.raises(ValueError):  # ADDED
        PowerTransform(1.5)  # ADDED
    with pytest.raises(ValueError):  # ADDED
        PowerTransform(3.0)  # ADDED
    with pytest.raises(ValueError):  # ADDED
        PowerTransform(2.0).apply(np.array([0.5, 1.0]))  # ADDED


def test_transform_for_selects_strategy():  # ADDED
    assert isinstance(transform_for(1.0), IdentityTransform)  # ADDED
    assert isinstance(transform_for(DECLARED_EXPONENT), PowerTransform)  # ADDED
    assert transform_for(DECLARED_EXPONENT).exponent == DECLARED_EXPONENT  # ADDED
    assert isinstance(  # ADDED
        transform_for(DECLARED_EXPONENT, power_on="alpha_bar"),  # ADDED
        AlphaBarPowerTransform,  # ADDED
    )  # ADDED


def test_alpha_bar_transform_matches_independent_schedule(schedule_config):  # ADDED
    betas = base_betas(schedule_config)  # ADDED
    from_betas = AlphaBarPowerTransform(DECLARED_EXPONENT).apply(betas)  # ADDED
    from_config = NoiseSchedule.from_config(  # ADDED
        schedule_config,  # ADDED
        AlphaBarPowerTransform(DECLARED_EXPONENT),  # ADDED
    ).betas  # ADDED
    np.testing.assert_allclose(from_betas, from_config, rtol=1e-5)  # ADDED


def test_identity_returns_a_copy():  # ADDED
    theta = np.array([0.1, 0.2])  # ADDED
    result = IdentityTransform().apply(theta)  # ADDED
    result[0] = 0.9  # ADDED
    assert theta[0] == 0.1  # ADDED


def test_open_unit_interval_check():  # ADDED
    assert_open_unit_interval(np.array([1e-9, 0.999]))  # ADDED
    with pytest.raises(ValueError):  # ADDED
        assert_open_unit_interval(np.array([0.0, 0.5]))  # ADDED


def test_declared_numbers_from_phase1(schedule_config):  # ADDED
    original = NoiseSchedule.from_config(schedule_config, IdentityTransform())  # ADDED
    modified = NoiseSchedule.from_config(schedule_config, PowerTransform(DECLARED_EXPONENT))  # ADDED
    assert original.summary().alpha_bar_last == pytest.approx(4.386e-5, rel=1e-3)  # ADDED
    assert original.first_t_snr_below_one() == 527  # ADDED
    assert modified.summary().beta_first == pytest.approx(1e-15)  # ADDED
    assert modified.summary().beta_last == pytest.approx(1e-5)  # ADDED
    assert modified.summary().alpha_bar_last == pytest.approx(0.9943, abs=1e-4)  # ADDED
    assert modified.summary().snr_last == pytest.approx(174.4, abs=0.1)  # ADDED
    assert modified.first_t_snr_below_one() is None  # ADDED


def test_power_schedule_raises_snr_everywhere(schedule_config):  # ADDED
    original = NoiseSchedule.from_config(schedule_config, IdentityTransform())  # ADDED
    modified = NoiseSchedule.from_config(schedule_config, PowerTransform(DECLARED_EXPONENT))  # ADDED
    assert np.all(modified.snr > original.snr)  # ADDED
    assert np.all(np.diff(original.snr) < 0)  # ADDED


def test_timesteps_are_one_indexed(schedule_config):  # ADDED
    schedule = NoiseSchedule.from_config(schedule_config, IdentityTransform())  # ADDED
    assert schedule.alpha_bar_at(1) == pytest.approx(1 - 1e-6)  # ADDED
    assert schedule.sqrt_alphas_cumprod_prev[0] == 1.0  # ADDED
    assert schedule.sqrt_alphas_cumprod_prev.shape == (2001,)  # ADDED
    with pytest.raises(IndexError):  # ADDED
        schedule.snr_at(0)  # ADDED


def _buffers(schedule: NoiseSchedule) -> ScheduleBuffers:  # ADDED
    return ScheduleBuffers(  # ADDED
        betas=schedule.betas.astype(np.float32),  # ADDED
        alphas_cumprod=schedule.alphas_cumprod.astype(np.float32),  # ADDED
        sqrt_alphas_cumprod_prev=schedule.sqrt_alphas_cumprod_prev,  # ADDED
    )  # ADDED


def test_verification_accepts_float32_copy(schedule_config):  # ADDED
    schedule = NoiseSchedule.from_config(schedule_config, PowerTransform(DECLARED_EXPONENT))  # ADDED
    evidence = verify_schedule(_buffers(schedule), schedule)  # ADDED
    assert evidence["verified"] is True  # ADDED
    assert evidence["description"] == f"theta_t ** {DECLARED_EXPONENT:.2f}"  # ADDED


def test_verification_rejects_other_schedule(schedule_config):  # ADDED
    original = NoiseSchedule.from_config(schedule_config, IdentityTransform())  # ADDED
    modified = NoiseSchedule.from_config(schedule_config, PowerTransform(DECLARED_EXPONENT))  # ADDED
    with pytest.raises(ScheduleMismatchError):  # ADDED
        verify_schedule(_buffers(original), modified)  # ADDED
