# ADDED: new file, not part of the upstream SR3 codebase.
import pytest  # ADDED

from sr3_ablation.config import ConfigError, build_config, load_config  # ADDED
from sr3_ablation.config.loader import deep_merge  # ADDED
from sr3_ablation.paths import CONFIG_DIR, PROJECT_ROOT  # ADDED
from sr3_ablation.runs import declared_exponent, run_spec  # ADDED
from sr3_ablation.schedule import EXPONENT_LOWER, EXPONENT_UPPER  # ADDED
from tests.helpers import DECLARED_EXPONENT, tiny_raw_config  # ADDED


def test_base_config_matches_plan():  # ADDED
    config = load_config(local_config=None)  # ADDED
    assert config.train.batch_size == 8  # ADDED
    assert config.train.learning_rate == 3e-5  # ADDED
    assert config.train.epochs == 60  # ADDED
    assert config.train.checkpoint_epochs == (12, 24, 36, 48, 60)  # ADDED
    assert (config.data.val_size, config.data.test_size) == (100, 100)  # ADDED
    assert config.sampling.trajectory_timesteps == (2000, 1500, 1000, 500, 0)  # ADDED
    assert config.model.unet.channel_multiplier == (1, 2, 4, 8, 8)  # ADDED
    assert config.paths.data_root == (PROJECT_ROOT / "data").resolve()  # ADDED


def test_smoke_config_overrides_base():  # ADDED
    config = load_config([CONFIG_DIR / "base.toml", CONFIG_DIR / "smoke.toml"], local_config=None)  # ADDED
    assert config.train.max_iterations == 200  # ADDED
    assert config.train.batch_size == 8  # ADDED
    assert config.paths.output_root.name == "smoke"  # ADDED


def test_local_config_is_merged_last(tmp_path):  # ADDED
    local = tmp_path / "local.toml"  # ADDED
    local.write_text('[paths]\ndata_root = "D:/elsewhere"\n', encoding="utf-8")  # ADDED
    config = load_config(local_config=local)  # ADDED
    assert config.paths.data_root.as_posix() == "D:/elsewhere"  # ADDED


def test_deep_merge_keeps_unrelated_keys():  # ADDED
    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": 1}, {"a": {"y": 3}})  # ADDED
    assert merged == {"a": {"x": 1, "y": 3}, "b": 1}  # ADDED


def test_unknown_and_missing_keys_are_reported(tmp_path):  # ADDED
    raw = tiny_raw_config(tmp_path)  # ADDED
    raw["train"]["lr"] = 1.0  # ADDED
    with pytest.raises(ConfigError, match="unknown keys"):  # ADDED
        build_config(raw, tmp_path)  # ADDED
    raw = tiny_raw_config(tmp_path)  # ADDED
    del raw["train"]["epochs"]  # ADDED
    with pytest.raises(ConfigError, match="missing keys"):  # ADDED
        build_config(raw, tmp_path)  # ADDED


def test_inconsistent_values_are_rejected(tmp_path):  # ADDED
    raw = tiny_raw_config(tmp_path)  # ADDED
    raw["train"]["checkpoint_epochs"] = [3]  # ADDED
    with pytest.raises(ConfigError):  # ADDED
        build_config(raw, tmp_path)  # ADDED
    raw = tiny_raw_config(tmp_path)  # ADDED
    raw["sampling"]["trajectory_timesteps"] = [11]  # ADDED
    with pytest.raises(ConfigError):  # ADDED
        build_config(raw, tmp_path)  # ADDED


def test_resolved_config_is_json_friendly(tiny_config):  # ADDED
    data = tiny_config.to_dict()  # ADDED
    assert isinstance(data["paths"]["data_root"], str)  # ADDED
    assert data["model"]["unet"]["channel_multiplier"] == [1, 2]  # ADDED


def test_run_specs():  # ADDED
    assert run_spec("0", DECLARED_EXPONENT).exponent == 1.0  # ADDED
    assert run_spec("a", DECLARED_EXPONENT).exponent == 1.0  # ADDED
    run_b = run_spec("runB", DECLARED_EXPONENT)  # ADDED
    assert (run_b.name, run_b.label, run_b.exponent, run_b.fine_tuned) == (  # ADDED
        "runB",  # ADDED
        "Run B",  # ADDED
        DECLARED_EXPONENT,  # ADDED
        True,  # ADDED
    )  # ADDED
    with pytest.raises(ValueError):  # ADDED
        run_spec("C", DECLARED_EXPONENT)  # ADDED


def test_supplementary_run_s_uses_alpha_bar_power():  # ADDED
    run_s = run_spec("S", DECLARED_EXPONENT)  # ADDED
    assert (run_s.name, run_s.power_target, run_s.fine_tuned) == ("runS", "alpha_bar", True)  # ADDED


def test_exponent_is_declared_in_finetune_script():  # ADDED
    exponent = declared_exponent()  # ADDED
    assert exponent == DECLARED_EXPONENT  # ADDED
    assert EXPONENT_LOWER < exponent < EXPONENT_UPPER  # ADDED
    assert round(exponent, 2) == exponent  # ADDED
