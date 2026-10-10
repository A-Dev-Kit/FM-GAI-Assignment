# ADDED: new file, not part of the upstream SR3 codebase.
"""Whole pipeline on a tiny SR3 UNet (T = 10, 16x16 images) on the CPU."""

from __future__ import annotations

import csv

import pytest
import torch

from sr3_ablation.io.jsonio import read_json
from sr3_ablation.io.run_paths import RunPaths
from sr3_ablation.model import Sr3Backend
from sr3_ablation.paths import REPORT_DIR
from sr3_ablation.pipelines.data_prep import prepare_data
from sr3_ablation.pipelines.evaluate import evaluate_run, record_trajectories
from sr3_ablation.pipelines.finetune import finetune
from sr3_ablation.reporting import ReportBuilder
from sr3_ablation.runs import run_spec
from tests.helpers import FAKE_DUPLICATE

pytestmark = pytest.mark.integration
EXPONENT = 1.6


@pytest.fixture
def prepared(tiny_config, fake_kvasir):
    prepare_data(tiny_config, workers=2)
    upstream_style = Sr3Backend.create(tiny_config.model, tiny_config.schedule, torch.device("cpu"))
    upstream_style.set_schedule(tiny_config.schedule, 1.0)
    weights = tiny_config.paths.checkpoint_dir / tiny_config.model.pretrained_file
    weights.parent.mkdir(parents=True)
    torch.save(upstream_style.network.state_dict(), weights)
    return tiny_config


def test_full_pipeline(prepared, tmp_path):
    config = prepared
    output = config.paths.output_root
    summary = read_json(output / "dataset" / "split_summary.json")
    assert summary["counts"] == {"train": 9, "val": 3, "test": 3}
    assert summary["images_found"] == 16
    assert summary["duplicates_removed"] == [FAKE_DUPLICATE]

    for key in ("A", "B"):
        results = finetune(config, run_spec(key, EXPONENT))
        assert [r.epoch for r in results] == [1, 2]

    run_b = RunPaths.for_run(output, "runB")
    with run_b.loss_csv.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["epoch", "train_loss"] and len(rows) == 3
    assert {p.name for p in run_b.samples.iterdir()} == {
        "runB_epoch01_sample1.png",
        "runB_epoch02_sample1.png",
    }
    assert {p.name for p in run_b.checkpoints.iterdir()} == {
        "epoch01_gen.pth",
        "epoch02_gen.pth",
        "final_gen.pth",
    }
    evidence = read_json(run_b.schedule_evidence)
    assert evidence["verified"] and evidence["exponent"] == EXPONENT
    assert evidence["stage"] == "after_training"
    assert read_json(run_b.run_record)["exponent"] == EXPONENT
    with run_b.validation_csv.open(encoding="utf-8") as handle:
        assert next(csv.DictReader(handle)).keys() >= {"epoch", "psnr", "ssim"}
    assert read_json(RunPaths.for_run(output, "runA").schedule_evidence)["exponent"] == 1.0

    for key in ("0", "A", "B"):
        summary = evaluate_run(config, run_spec(key, EXPONENT))
        assert summary.images == 3 and summary.mean["psnr"] > 0
    assert read_json(run_b.sampling_schedule_evidence)["verified"]
    assert len(list(run_b.test_outputs.glob("runB_test_*.png"))) == 3
    assert read_json(RunPaths.for_run(output, "bicubic").metrics_summary)["label"] == "Bicubic"

    written = record_trajectories(config, run_spec("B", EXPONENT))
    assert sorted(p.name for p in written) == [
        "runB_t0000_sample1.png",
        "runB_t0004_sample1.png",
        "runB_t0008_sample1.png",
    ]
    record_trajectories(config, run_spec("A", EXPONENT))

    report = ReportBuilder(
        config, EXPONENT, template=REPORT_DIR / "report_template.md", build_dir=tmp_path / "build"
    ).build(pdf=False)
    text = report.read_text(encoding="utf-8")
    assert "{{" not in text and "Pending" not in text
    assert "Run B" in text and "theta_t ** 1.60" in text
    assert (output / "report_assets" / "trajectories.png").is_file()
    assert (output / "report_assets" / "test_comparison.png").is_file()


def test_sampling_refuses_changed_exponent(prepared):
    finetune(prepared, run_spec("B", EXPONENT))
    with pytest.raises(ValueError, match="fine-tuned with x"):
        evaluate_run(prepared, run_spec("B", 2.0))
