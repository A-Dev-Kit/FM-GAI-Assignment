# ADDED: new file, not part of the upstream SR3 codebase.
import math

import numpy as np
import pytest
import torch

from sr3_ablation.evaluation import Psnr, Ssim, build_metrics, summarise
from sr3_ablation.io.images import tensor_to_uint8, uint8_to_tensor
from sr3_ablation.io.naming import (
    checkpoint_name,
    epoch_sample_name,
    super_resolved_name,
    trajectory_name,
)
from sr3_ablation.reporting import fill_template
from sr3_ablation.reporting.markdown_html import markdown_body


def test_file_names_follow_the_brief():
    assert epoch_sample_name("runB", 4, 1) == "runB_epoch04_sample1.png"
    assert trajectory_name("runA", 1600, 1) == "runA_t1600_sample1.png"
    assert trajectory_name("runA", 0, 3) == "runA_t0000_sample3.png"
    assert super_resolved_name("run0", "test_00007") == "run0_test_00007.png"
    assert checkpoint_name(10) == "epoch10_gen.pth"


def test_image_tensor_round_trip():
    image = np.random.default_rng(0).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    tensor = uint8_to_tensor(image)
    assert tensor.shape == (3, 8, 8)
    np.testing.assert_array_equal(tensor_to_uint8(tensor.unsqueeze(0))[0], image)
    clipped = tensor_to_uint8(torch.full((1, 3, 2, 2), 5.0))[0]
    assert clipped.max() == 255


def test_psnr():
    image = np.full((8, 8, 3), 100, dtype=np.uint8)
    assert math.isinf(Psnr()(image, image))
    shifted = image + 10
    assert Psnr()(shifted, image) == pytest.approx(20 * math.log10(255 / 10))


def test_ssim_bounds():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    noisy = np.clip(image + rng.normal(0, 40, image.shape), 0, 255).astype(np.uint8)
    assert Ssim()(image, image) == pytest.approx(1.0)
    assert 0.0 < Ssim()(noisy, image) < 0.9


def test_metric_registry():
    names = [metric.name for metric in build_metrics(["psnr", "ssim"], torch.device("cpu"))]
    assert names == ["psnr", "ssim"]
    with pytest.raises(ValueError):
        build_metrics(["fid"], torch.device("cpu"))


def test_summary_ignores_infinite_psnr():
    metrics = [Psnr()]
    rows = [{"image_id": "a", "psnr": math.inf}, {"image_id": "b", "psnr": 30.0}]
    summary = summarise(rows, metrics, "run0", "Run 0")
    assert summary.mean["psnr"] == 30.0
    assert summary.images == 2


def test_markdown_keeps_maths_and_code():
    body = markdown_body(
        "Text $\\bar\\alpha_t$ and `a_b`\n\n$$x_t = y$$\n\n```python\nx = 1 * 2\n```"
    )
    assert "$\\bar\\alpha_t$" in body
    assert "<code>a_b</code>" in body
    assert "$$x_t = y$$" in body
    assert '<pre><code class="language-python">x = 1 * 2' in body


def test_fill_template():
    assert fill_template("x = {{EXPONENT}}", {"EXPONENT": "1.60"}) == "x = 1.60"
    with pytest.raises(KeyError):
        fill_template("{{MISSING}}", {})
