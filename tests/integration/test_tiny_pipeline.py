# ADDED: new file, not part of the upstream SR3 codebase.
"""Whole pipeline on a tiny SR3 UNet (T = 10, 16x16 images) on the CPU."""  # ADDED

from __future__ import annotations  # ADDED

import csv  # ADDED

import pytest  # ADDED
import torch  # ADDED

from sr3_ablation.io.jsonio import read_json  # ADDED
from sr3_ablation.io.run_paths import RunPaths  # ADDED
from sr3_ablation.model import Sr3Backend  # ADDED
from sr3_ablation.paths import REPORT_DIR  # ADDED
from sr3_ablation.pipelines.common import expected_schedule_for_exponent  # ADDED
from sr3_ablation.pipelines.data_prep import prepare_data  # ADDED
from sr3_ablation.pipelines.diagnostics import diagnose_start  # ADDED
from sr3_ablation.pipelines.evaluate import evaluate_run, record_trajectories  # ADDED
from sr3_ablation.pipelines.finetune import finetune  # ADDED
from sr3_ablation.reporting import ReportBuilder  # ADDED
from sr3_ablation.runs import run_spec  # ADDED
from tests.helpers import DECLARED_EXPONENT, FAKE_DUPLICATE  # ADDED

pytestmark = pytest.mark.integration  # ADDED
EXPONENT = DECLARED_EXPONENT  # ADDED
OTHER_EXPONENT = 2.0  # ADDED


@pytest.fixture  # ADDED
def prepared(tiny_config, fake_kvasir):  # ADDED
    prepare_data(tiny_config, workers=2)  # ADDED
    upstream_style = Sr3Backend.create(tiny_config.model, tiny_config.schedule, torch.device("cpu"))  # ADDED
    upstream_style.set_schedule(tiny_config.schedule, run_spec("0", DECLARED_EXPONENT))  # ADDED
    weights = tiny_config.paths.checkpoint_dir / tiny_config.model.pretrained_file  # ADDED
    weights.parent.mkdir(parents=True)  # ADDED
    torch.save(upstream_style.network.state_dict(), weights)  # ADDED
    return tiny_config  # ADDED


def test_full_pipeline(prepared, tmp_path):  # ADDED
    config = prepared  # ADDED
    output = config.paths.output_root  # ADDED
    summary = read_json(output / "dataset" / "split_summary.json")  # ADDED
    assert summary["counts"] == {"train": 9, "val": 3, "test": 3}  # ADDED
    assert summary["images_found"] == 16  # ADDED
    assert summary["duplicates_removed"] == [FAKE_DUPLICATE]  # ADDED

    for key in ("A", "B"):  # ADDED
        results = finetune(config, run_spec(key, EXPONENT))  # ADDED
        assert [r.epoch for r in results] == [1, 2]  # ADDED

    run_b = RunPaths.for_run(output, "runB")  # ADDED
    with run_b.loss_csv.open(encoding="utf-8") as handle:  # ADDED
        rows = list(csv.reader(handle))  # ADDED
    assert rows[0] == ["epoch", "train_loss"] and len(rows) == 3  # ADDED
    assert {p.name for p in run_b.samples.iterdir()} == {  # ADDED
        "runB_epoch01_sample1.png",  # ADDED
        "runB_epoch02_sample1.png",  # ADDED
    }  # ADDED
    assert {p.name for p in run_b.checkpoints.iterdir()} == {  # ADDED
        "epoch01_gen.pth",  # ADDED
        "epoch02_gen.pth",  # ADDED
        "final_gen.pth",  # ADDED
        "training_last.pth",  # ADDED
    }  # ADDED
    evidence = read_json(run_b.schedule_evidence)  # ADDED
    assert evidence["verified"] and evidence["exponent"] == EXPONENT  # ADDED
    assert evidence["stage"] == "after_training"  # ADDED
    assert read_json(run_b.run_record)["exponent"] == EXPONENT  # ADDED
    with run_b.validation_csv.open(encoding="utf-8") as handle:  # ADDED
        assert next(csv.DictReader(handle)).keys() >= {"epoch", "psnr", "ssim"}  # ADDED
    assert read_json(RunPaths.for_run(output, "runA").schedule_evidence)["exponent"] == 1.0  # ADDED

    for key in ("0", "A", "B"):  # ADDED
        summary = evaluate_run(config, run_spec(key, EXPONENT))  # ADDED
        assert summary.images == 3 and summary.mean["psnr"] > 0  # ADDED
    assert read_json(run_b.sampling_schedule_evidence)["verified"]  # ADDED
    assert len(list(run_b.test_outputs.glob("runB_test_*.png"))) == 3  # ADDED
    assert read_json(RunPaths.for_run(output, "bicubic").metrics_summary)["label"] == "Bicubic"  # ADDED

    written = record_trajectories(config, run_spec("B", EXPONENT))  # ADDED
    assert sorted(p.name for p in written) == [  # ADDED
        "runB_t0000_sample1.png",  # ADDED
        "runB_t0004_sample1.png",  # ADDED
        "runB_t0008_sample1.png",  # ADDED
    ]  # ADDED
    record_trajectories(config, run_spec("A", EXPONENT))  # ADDED

    diagnostic = diagnose_start(config, run_spec("B", EXPONENT))  # ADDED
    assert list(diagnostic) == ["bicubic", "pure_noise", "forward_noised"]  # ADDED
    record = read_json(run_b.start_diagnostic_record)  # ADDED
    assert record["timesteps"] == [10, 8, 4, 0]  # ADDED
    assert record["alpha_bar_T"] == pytest.approx(  # ADDED
        expected_schedule_for_exponent(config, EXPONENT).alpha_bar_at(10)  # ADDED
    )  # ADDED
    assert len(list(run_b.diagnostics.glob("runB_*_sample1.png"))) == 8  # ADDED

    report = ReportBuilder(  # ADDED
        config,  # ADDED
        EXPONENT,  # ADDED
        template=REPORT_DIR / "report_template.md",  # ADDED
        build_dir=tmp_path / "build",  # ADDED
    ).build(pdf=False)  # ADDED
    text = report.read_text(encoding="utf-8")  # ADDED
    assert "{{" not in text and "Pending" not in text  # ADDED
    assert "Run B" in text and f"theta_t ** {EXPONENT:.2f}" in text  # ADDED
    for figure in (  # ADDED
        "trajectories.png",  # ADDED
        "test_comparison.png",  # ADDED
        "run0_samples.png",  # ADDED
        "checkpoint_comparison.png",  # ADDED
        "start_diagnostic.png",  # ADDED
    ):  # ADDED
        assert (output / "report_assets" / figure).is_file(), figure  # ADDED
    sections = [line for line in text.splitlines() if line.startswith("### 6.")]  # ADDED
    assert [s.split(" ", 2)[2] for s in sections] == [  # ADDED
        "Run 0: the pretrained model on our inputs",  # ADDED
        "Fine-tuning loss",  # ADDED
        "Samples at the checkpoint epochs",  # ADDED
        "Inference trajectories",  # ADDED
        "Test-set scores",  # ADDED
    ]  # ADDED


def test_sampling_refuses_changed_exponent(prepared):  # ADDED
    finetune(prepared, run_spec("B", EXPONENT))  # ADDED
    with pytest.raises(ValueError, match="fine-tuned with x"):  # ADDED
        evaluate_run(prepared, run_spec("B", OTHER_EXPONENT))  # ADDED
