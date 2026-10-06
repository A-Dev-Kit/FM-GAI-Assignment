# ADDED: new file, not part of the upstream SR3 codebase.
"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from sr3_ablation.config import ExperimentConfig, build_config
from tests.helpers import make_fake_afhq, tiny_raw_config


@pytest.fixture
def tiny_config(tmp_path: Path) -> ExperimentConfig:
    return build_config(tiny_raw_config(tmp_path), tmp_path)


@pytest.fixture
def fake_afhq(tiny_config: ExperimentConfig) -> Path:
    return make_fake_afhq(tiny_config.paths.data_root / "raw")
