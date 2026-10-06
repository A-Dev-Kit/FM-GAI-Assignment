# ADDED: new file, not part of the upstream SR3 codebase.
import pytest

from sr3_ablation.config import ConfigError, build_config, load_config
from sr3_ablation.config.loader import deep_merge
from sr3_ablation.paths import CONFIG_DIR, PROJECT_ROOT
from sr3_ablation.runs import declared_exponent, run_spec
from sr3_ablation.schedule import EXPONENT_LOWER, EXPONENT_UPPER
from tests.helpers import tiny_raw_config


def test_base_config_matches_plan():
    config = load_config(local_config=None)
    assert config.train.batch_size == 8
    assert config.train.learning_rate == 1e-5
    assert config.train.checkpoint_epochs == (2, 4, 6, 8, 10)
    assert config.sampling.trajectory_timesteps == (1600, 1200, 800, 400, 0)
    assert config.model.unet.channel_multiplier == (1, 2, 4, 8, 8)
    assert config.paths.data_root == (PROJECT_ROOT / "data").resolve()


def test_smoke_config_overrides_base():
    config = load_config([CONFIG_DIR / "base.toml", CONFIG_DIR / "smoke.toml"], local_config=None)
    assert config.train.max_iterations == 200
    assert config.train.batch_size == 8
    assert config.paths.output_root.name == "smoke"


def test_local_config_is_merged_last(tmp_path):
    local = tmp_path / "local.toml"
    local.write_text('[paths]\ndata_root = "D:/elsewhere"\n', encoding="utf-8")
    config = load_config(local_config=local)
    assert config.paths.data_root.as_posix() == "D:/elsewhere"


def test_deep_merge_keeps_unrelated_keys():
    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": 1}, {"a": {"y": 3}})
    assert merged == {"a": {"x": 1, "y": 3}, "b": 1}


def test_unknown_and_missing_keys_are_reported(tmp_path):
    raw = tiny_raw_config(tmp_path)
    raw["train"]["lr"] = 1.0
    with pytest.raises(ConfigError, match="unknown keys"):
        build_config(raw, tmp_path)
    raw = tiny_raw_config(tmp_path)
    del raw["train"]["epochs"]
    with pytest.raises(ConfigError, match="missing keys"):
        build_config(raw, tmp_path)


def test_inconsistent_values_are_rejected(tmp_path):
    raw = tiny_raw_config(tmp_path)
    raw["train"]["checkpoint_epochs"] = [3]
    with pytest.raises(ConfigError):
        build_config(raw, tmp_path)
    raw = tiny_raw_config(tmp_path)
    raw["sampling"]["trajectory_timesteps"] = [11]
    with pytest.raises(ConfigError):
        build_config(raw, tmp_path)


def test_resolved_config_is_json_friendly(tiny_config):
    data = tiny_config.to_dict()
    assert isinstance(data["paths"]["data_root"], str)
    assert data["model"]["unet"]["channel_multiplier"] == [1, 2]


def test_run_specs():
    assert run_spec("0", 1.6).exponent == 1.0
    assert run_spec("a", 1.6).exponent == 1.0
    run_b = run_spec("runB", 1.6)
    assert (run_b.name, run_b.label, run_b.exponent, run_b.fine_tuned) == (
        "runB",
        "Run B",
        1.6,
        True,
    )
    with pytest.raises(ValueError):
        run_spec("C", 1.6)


def test_exponent_is_declared_in_finetune_script():
    exponent = declared_exponent()
    assert exponent == 1.60
    assert EXPONENT_LOWER < exponent < EXPONENT_UPPER
    assert round(exponent, 2) == exponent
