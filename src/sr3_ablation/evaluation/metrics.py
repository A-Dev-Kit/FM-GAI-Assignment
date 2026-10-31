# ADDED: new file, not part of the upstream SR3 codebase.
"""Image-quality metrics on 8-bit RGB images and a name -> metric registry.  # ADDED

PSNR and SSIM follow the usual definitions on the full RGB range [0, 255]; SSIM uses an  # ADDED
11x11 Gaussian window (sigma 1.5), K1 = 0.01, K2 = 0.03, valid filtering and the mean over  # ADDED
channels, matching the scikit-image / MATLAB convention.  # ADDED
"""  # ADDED

from __future__ import annotations  # ADDED

import math  # ADDED
from collections.abc import Callable, Sequence  # ADDED
from typing import Protocol  # ADDED

import numpy as np  # ADDED
import torch  # ADDED
import torch.nn.functional as F  # ADDED

from sr3_ablation.io.images import uint8_to_tensor  # ADDED

MAX_PIXEL = 255.0  # ADDED


class ImageMetric(Protocol):  # ADDED
    """A full-reference image metric and the direction in which it improves."""  # ADDED

    name: str  # ADDED
    higher_is_better: bool  # ADDED

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:  # ADDED
        """Compare two ``(H, W, 3)`` uint8 images."""  # ADDED
        ...  # ADDED


class Psnr:  # ADDED
    """Peak signal-to-noise ratio in dB over the full 8-bit RGB range."""  # ADDED

    name = "psnr"  # ADDED
    higher_is_better = True  # ADDED

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:  # ADDED
        _check_pair(prediction, target)  # ADDED
        error = np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2)  # ADDED
        return math.inf if error == 0 else 20.0 * math.log10(MAX_PIXEL / math.sqrt(error))  # ADDED


class Ssim:  # ADDED
    """Structural similarity with a Gaussian window, averaged over channels."""  # ADDED

    name = "ssim"  # ADDED
    higher_is_better = True  # ADDED

    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:  # ADDED
        coords = torch.arange(window_size, dtype=torch.float64) - (window_size - 1) / 2  # ADDED
        gauss = torch.exp(-(coords**2) / (2 * sigma**2))  # ADDED
        gauss /= gauss.sum()  # ADDED
        self._window = torch.outer(gauss, gauss).view(1, 1, window_size, window_size)  # ADDED
        self._c1 = (0.01 * MAX_PIXEL) ** 2  # ADDED
        self._c2 = (0.03 * MAX_PIXEL) ** 2  # ADDED

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:  # ADDED
        _check_pair(prediction, target)  # ADDED
        x = _as_channels(prediction)  # ADDED
        y = _as_channels(target)  # ADDED
        mu_x, mu_y = self._filter(x), self._filter(y)  # ADDED
        var_x = self._filter(x * x) - mu_x**2  # ADDED
        var_y = self._filter(y * y) - mu_y**2  # ADDED
        cov = self._filter(x * y) - mu_x * mu_y  # ADDED
        numerator = (2 * mu_x * mu_y + self._c1) * (2 * cov + self._c2)  # ADDED
        denominator = (mu_x**2 + mu_y**2 + self._c1) * (var_x + var_y + self._c2)  # ADDED
        return float((numerator / denominator).mean())  # ADDED

    def _filter(self, channels: torch.Tensor) -> torch.Tensor:  # ADDED
        return F.conv2d(channels, self._window)  # ADDED


class Lpips:  # ADDED
    """Learned perceptual distance (AlexNet backbone); requires ``pip install lpips``."""  # ADDED

    name = "lpips"  # ADDED
    higher_is_better = False  # ADDED

    def __init__(self, device: torch.device) -> None:  # ADDED
        try:  # ADDED
            import lpips  # ADDED
        except ImportError as error:  # ADDED
            raise ImportError("LPIPS needs the optional extra: pip install -e .[lpips]") from error  # ADDED
        self._device = device  # ADDED
        self._model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()  # ADDED

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:  # ADDED
        _check_pair(prediction, target)  # ADDED
        with torch.no_grad():  # ADDED
            a = uint8_to_tensor(prediction).unsqueeze(0).to(self._device)  # ADDED
            b = uint8_to_tensor(target).unsqueeze(0).to(self._device)  # ADDED
            return float(self._model(a, b).item())  # ADDED


MetricFactory = Callable[[torch.device], ImageMetric]  # ADDED

METRIC_REGISTRY: dict[str, MetricFactory] = {  # ADDED
    Psnr.name: lambda _device: Psnr(),  # ADDED
    Ssim.name: lambda _device: Ssim(),  # ADDED
    Lpips.name: Lpips,  # ADDED
}  # ADDED

HIGHER_IS_BETTER: dict[str, bool] = {  # ADDED
    metric.name: metric.higher_is_better  # ADDED
    for metric in (Psnr, Ssim, Lpips)  # ADDED
}  # ADDED


def build_metrics(names: Sequence[str], device: torch.device) -> list[ImageMetric]:  # ADDED
    """Instantiate the named metrics from ``METRIC_REGISTRY``."""  # ADDED
    unknown = [name for name in names if name not in METRIC_REGISTRY]  # ADDED
    if unknown:  # ADDED
        raise ValueError(f"Unknown metrics {unknown}; available: {sorted(METRIC_REGISTRY)}.")  # ADDED
    return [METRIC_REGISTRY[name](device) for name in names]  # ADDED


def _check_pair(prediction: np.ndarray, target: np.ndarray) -> None:  # ADDED
    if prediction.shape != target.shape or prediction.ndim != 3:  # ADDED
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}.")  # ADDED


def _as_channels(image: np.ndarray) -> torch.Tensor:  # ADDED
    """``(H, W, C)`` uint8 -> ``(C, 1, H, W)`` float64 so each channel is filtered alone."""  # ADDED
    return torch.from_numpy(image.astype(np.float64)).permute(2, 0, 1).unsqueeze(1)  # ADDED
