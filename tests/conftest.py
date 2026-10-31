# ADDED: new file, not part of the upstream SR3 codebase.
"""Shared fixtures."""  # ADDED

from __future__ import annotations  # ADDED

from pathlib import Path  # ADDED

import pytest  # ADDED

from sr3_ablation.config import ExperimentConfig, build_config  # ADDED
from sr3_ablation.data import KvasirSegSource  # ADDED
from tests.helpers import make_fake_kvasir, tiny_raw_config  # ADDED


@pytest.fixture  # ADDED
def tiny_config(tmp_path: Path) -> ExperimentConfig:  # ADDED
    return build_config(tiny_raw_config(tmp_path), tmp_path)  # ADDED


@pytest.fixture  # ADDED
def fake_kvasir(tiny_config: ExperimentConfig) -> Path:  # ADDED
    source = KvasirSegSource(tiny_config.data.source_url, tiny_config.paths.data_root)  # ADDED
    return make_fake_kvasir(source.raw_dir)  # ADDED
