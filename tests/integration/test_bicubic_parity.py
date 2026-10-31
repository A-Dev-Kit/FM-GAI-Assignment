# ADDED: new file, not part of the upstream SR3 codebase.
"""Bicubic PSNR on teammate split and declaration preprocessing (~24.03 dB reference)."""  # ADDED

from __future__ import annotations  # ADDED

import numpy as np  # ADDED
import pytest  # ADDED
from PIL import Image  # ADDED

from sr3_ablation.config.loader import load_config  # ADDED
from sr3_ablation.data import layout_for, read_manifest  # ADDED
from sr3_ablation.evaluation.metrics import Psnr  # ADDED
from sr3_ablation.paths import PROJECT_ROOT  # ADDED

pytestmark = pytest.mark.integration  # ADDED

KVASIR_MANIFEST = PROJECT_ROOT / "outputs" / "dataset" / "manifest.csv"  # ADDED
SONALIKA_REFERENCE_PSNR = 24.03  # ADDED
TOLERANCE_DB = 0.35  # ADDED


def _kvasir_ready() -> bool:  # ADDED
    if not KVASIR_MANIFEST.is_file():  # ADDED
        return False  # ADDED
    entries = read_manifest(KVASIR_MANIFEST)  # ADDED
    return len(entries) == 1000 and sum(1 for e in entries if e.split == "test") == 100  # ADDED


@pytest.mark.skipif(  # ADDED
    not _kvasir_ready(),  # ADDED
    reason="full Kvasir manifest and processed data not present",  # ADDED
)  # ADDED
def test_bicubic_psnr_matches_teammate_reference():  # ADDED
    config = load_config([PROJECT_ROOT / "configs" / "base.toml"], local_config=None)  # ADDED
    layout = layout_for(config.paths.data_root, "test", 16, 128)  # ADDED
    if not layout.hr_dir.is_dir():  # ADDED
        pytest.skip("processed test split missing; run prepare-data first")  # ADDED
    psnr = Psnr()  # ADDED
    scores: list[float] = []  # ADDED
    for image_id in layout.image_ids():  # ADDED
        hr = np.asarray(Image.open(layout.hr_path(image_id)).convert("RGB"))  # ADDED
        sr = np.asarray(Image.open(layout.sr_path(image_id)).convert("RGB"))  # ADDED
        scores.append(psnr(sr, hr))  # ADDED
    mean = float(np.mean(scores))  # ADDED
    assert mean == pytest.approx(SONALIKA_REFERENCE_PSNR, abs=TOLERANCE_DB)  # ADDED
