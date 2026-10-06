# ADDED: new file, not part of the upstream SR3 codebase.
"""Image-quality metrics on 8-bit RGB images and a name -> metric registry.

PSNR and SSIM follow the usual definitions on the full RGB range [0, 255]; SSIM uses an
11x11 Gaussian window (sigma 1.5), K1 = 0.01, K2 = 0.03, valid filtering and the mean over
channels, matching the scikit-image / MATLAB convention.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F

from sr3_ablation.io.images import uint8_to_tensor

MAX_PIXEL = 255.0


class ImageMetric(Protocol):
    name: str
    higher_is_better: bool

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Compare two ``(H, W, 3)`` uint8 images."""
        ...


class Psnr:
    name = "psnr"
    higher_is_better = True

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        _check_pair(prediction, target)
        error = np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2)
        return math.inf if error == 0 else 20.0 * math.log10(MAX_PIXEL / math.sqrt(error))


class Ssim:
    name = "ssim"
    higher_is_better = True

    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        coords = torch.arange(window_size, dtype=torch.float64) - (window_size - 1) / 2
        gauss = torch.exp(-(coords**2) / (2 * sigma**2))
        gauss /= gauss.sum()
        self._window = torch.outer(gauss, gauss).view(1, 1, window_size, window_size)
        self._c1 = (0.01 * MAX_PIXEL) ** 2
        self._c2 = (0.03 * MAX_PIXEL) ** 2

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        _check_pair(prediction, target)
        x = _as_channels(prediction)
        y = _as_channels(target)
        mu_x, mu_y = self._filter(x), self._filter(y)
        var_x = self._filter(x * x) - mu_x**2
        var_y = self._filter(y * y) - mu_y**2
        cov = self._filter(x * y) - mu_x * mu_y
        numerator = (2 * mu_x * mu_y + self._c1) * (2 * cov + self._c2)
        denominator = (mu_x**2 + mu_y**2 + self._c1) * (var_x + var_y + self._c2)
        return float((numerator / denominator).mean())

    def _filter(self, channels: torch.Tensor) -> torch.Tensor:
        return F.conv2d(channels, self._window)


class Lpips:
    """Learned perceptual distance (AlexNet backbone); requires ``pip install lpips``."""

    name = "lpips"
    higher_is_better = False

    def __init__(self, device: torch.device) -> None:
        try:
            import lpips
        except ImportError as error:
            raise ImportError("LPIPS needs the optional extra: pip install -e .[lpips]") from error
        self._device = device
        self._model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        _check_pair(prediction, target)
        with torch.no_grad():
            a = uint8_to_tensor(prediction).unsqueeze(0).to(self._device)
            b = uint8_to_tensor(target).unsqueeze(0).to(self._device)
            return float(self._model(a, b).item())


MetricFactory = Callable[[torch.device], ImageMetric]

METRIC_REGISTRY: dict[str, MetricFactory] = {
    Psnr.name: lambda _device: Psnr(),
    Ssim.name: lambda _device: Ssim(),
    Lpips.name: Lpips,
}

HIGHER_IS_BETTER: dict[str, bool] = {
    metric.name: metric.higher_is_better for metric in (Psnr, Ssim, Lpips)
}


def build_metrics(names: Sequence[str], device: torch.device) -> list[ImageMetric]:
    unknown = [name for name in names if name not in METRIC_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}; available: {sorted(METRIC_REGISTRY)}.")
    return [METRIC_REGISTRY[name](device) for name in names]


def _check_pair(prediction: np.ndarray, target: np.ndarray) -> None:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}.")


def _as_channels(image: np.ndarray) -> torch.Tensor:
    """``(H, W, C)`` uint8 -> ``(C, 1, H, W)`` float64 so each channel is filtered alone."""
    return torch.from_numpy(image.astype(np.float64)).permute(2, 0, 1).unsqueeze(1)
