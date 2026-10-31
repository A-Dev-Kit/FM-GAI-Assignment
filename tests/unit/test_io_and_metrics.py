# ADDED: new file, not part of the upstream SR3 codebase.
import math  # ADDED

import numpy as np  # ADDED
import pytest  # ADDED
import torch  # ADDED

from sr3_ablation.evaluation import Psnr, Ssim, build_metrics, summarise  # ADDED
from sr3_ablation.io.images import tensor_to_uint8, uint8_to_tensor  # ADDED
from sr3_ablation.io.naming import (  # ADDED
    checkpoint_name,  # ADDED
    epoch_sample_name,  # ADDED
    super_resolved_name,  # ADDED
    trajectory_name,  # ADDED
)  # ADDED
from sr3_ablation.reporting import fill_template  # ADDED
from sr3_ablation.reporting.markdown_html import markdown_body  # ADDED


def test_file_names_follow_the_brief():  # ADDED
    assert epoch_sample_name("runB", 4, 1) == "runB_epoch04_sample1.png"  # ADDED
    assert trajectory_name("runA", 1600, 1) == "runA_t1600_sample1.png"  # ADDED
    assert trajectory_name("runA", 0, 3) == "runA_t0000_sample3.png"  # ADDED
    assert super_resolved_name("run0", "test_00007") == "run0_test_00007.png"  # ADDED
    assert checkpoint_name(10) == "epoch10_gen.pth"  # ADDED


def test_image_tensor_round_trip():  # ADDED
    image = np.random.default_rng(0).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)  # ADDED
    tensor = uint8_to_tensor(image)  # ADDED
    assert tensor.shape == (3, 8, 8)  # ADDED
    np.testing.assert_array_equal(tensor_to_uint8(tensor.unsqueeze(0))[0], image)  # ADDED
    clipped = tensor_to_uint8(torch.full((1, 3, 2, 2), 5.0))[0]  # ADDED
    assert clipped.max() == 255  # ADDED


def test_psnr():  # ADDED
    image = np.full((8, 8, 3), 100, dtype=np.uint8)  # ADDED
    assert math.isinf(Psnr()(image, image))  # ADDED
    shifted = image + 10  # ADDED
    assert Psnr()(shifted, image) == pytest.approx(20 * math.log10(255 / 10))  # ADDED


def test_ssim_bounds():  # ADDED
    rng = np.random.default_rng(1)  # ADDED
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)  # ADDED
    noisy = np.clip(image + rng.normal(0, 40, image.shape), 0, 255).astype(np.uint8)  # ADDED
    assert Ssim()(image, image) == pytest.approx(1.0)  # ADDED
    assert 0.0 < Ssim()(noisy, image) < 0.9  # ADDED


def test_metric_registry():  # ADDED
    names = [metric.name for metric in build_metrics(["psnr", "ssim"], torch.device("cpu"))]  # ADDED
    assert names == ["psnr", "ssim"]  # ADDED
    with pytest.raises(ValueError):  # ADDED
        build_metrics(["fid"], torch.device("cpu"))  # ADDED


def test_summary_ignores_infinite_psnr():  # ADDED
    metrics = [Psnr()]  # ADDED
    rows = [{"image_id": "a", "psnr": math.inf}, {"image_id": "b", "psnr": 30.0}]  # ADDED
    summary = summarise(rows, metrics, "run0", "Run 0")  # ADDED
    assert summary.mean["psnr"] == 30.0  # ADDED
    assert summary.images == 2  # ADDED


def test_markdown_keeps_maths_and_code():  # ADDED
    body = markdown_body(  # ADDED
        "Text $\\bar\\alpha_t$ and `a_b`\n\n$$x_t = y$$\n\n```python\nx = 1 * 2\n```"  # ADDED
    )  # ADDED
    assert "$\\bar\\alpha_t$" in body  # ADDED
    assert "<code>a_b</code>" in body  # ADDED
    assert "$$x_t = y$$" in body  # ADDED
    assert '<pre><code class="language-python">x = 1 * 2' in body  # ADDED


def test_fill_template():  # ADDED
    assert fill_template("x = {{EXPONENT}}", {"EXPONENT": "2.50"}) == "x = 2.50"  # ADDED
    with pytest.raises(KeyError):  # ADDED
        fill_template("{{MISSING}}", {})  # ADDED
