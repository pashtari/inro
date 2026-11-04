"""Evaluation metrics for images, volumes and compressed representations.

Thin wrappers around torchmetrics that score a single sample and return a Python float.
Image metrics take ``(channels, H, W)`` images, and ``data_range`` defaults to the dtype
convention of torchvision and scikit-image: ``255`` for ``uint8`` images and ``1`` for
floating-point images in ``[0, 1]``. Pass it explicitly for other ranges, as
``data_range=target.max()`` for MRI magnitudes.
"""

import math
from collections.abc import Callable
from functools import cache
from typing import Any, Literal, TypeAlias

import torch
from torch import Tensor, nn
from torchmetrics.functional.classification import binary_jaccard_index
from torchmetrics.functional.image import (
    multiscale_structural_similarity_index_measure,
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

__all__ = [
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "nmse",
    "iou",
    "bits_per_pixel",
    "luma",
    "super_resolution_metrics",
]

_LPIPSNet: TypeAlias = Literal["alex", "vgg", "squeeze"]


def _data_range(target: Tensor, data_range: float | None) -> float:
    if data_range is not None:
        return float(data_range)
    return 1.0 if target.is_floating_point() else float(torch.iinfo(target.dtype).max)


def psnr(pred: Tensor, target: Tensor, data_range: float | None = None) -> float:
    """Return the peak signal-to-noise ratio in dB, pooling all elements.

    Args:
        pred: Reconstruction.
        target: Reference signal.
        data_range: Peak value; ``None`` uses ``1`` for floating-point targets and the
            dtype's maximum for integer ones.

    Examples:
        >>> round(psnr(torch.zeros(3, 4, 4), torch.full((3, 4, 4), 0.1)), 2)
        20.0
    """
    return peak_signal_noise_ratio(
        pred.float(), target.float(), data_range=_data_range(target, data_range)
    ).item()


def ssim(
    pred: Tensor, target: Tensor, data_range: float | None = None, **kwargs: Any
) -> float:
    """Return the structural similarity of two ``(channels, H, W)`` images.

    Uses an 11x11 Gaussian window by default.

    Args:
        pred: Reconstruction.
        target: Reference signal.
        data_range: Peak value; ``None`` uses ``1`` for floating-point targets and the
            dtype's maximum for integer ones.
        **kwargs: Passed to ``structural_similarity_index_measure`` of torchmetrics.
    """
    return structural_similarity_index_measure(
        pred.float().unsqueeze(0),
        target.float().unsqueeze(0),
        data_range=_data_range(target, data_range),
        **kwargs,
    ).item()


def ms_ssim(
    pred: Tensor, target: Tensor, data_range: float | None = None, **kwargs: Any
) -> float:
    """Return the multi-scale structural similarity of two ``(channels, H, W)`` images.

    The five default scales require both image sides to be at least 176 pixels.

    Args:
        pred: Reconstruction.
        target: Reference signal.
        data_range: Peak value; ``None`` uses ``1`` for floating-point targets and the
            dtype's maximum for integer ones.
        **kwargs: Passed to ``multiscale_structural_similarity_index_measure`` of
            torchmetrics.
    """
    return multiscale_structural_similarity_index_measure(
        pred.float().unsqueeze(0),
        target.float().unsqueeze(0),
        data_range=_data_range(target, data_range),
        **kwargs,
    ).item()


@cache
def _lpips_network(net: _LPIPSNet, device: torch.device) -> nn.Module:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    return LearnedPerceptualImagePatchSimilarity(net, normalize=True).to(device).eval()


@torch.no_grad()
def lpips(
    pred: Tensor,
    target: Tensor,
    data_range: float | None = None,
    net: _LPIPSNet = "vgg",
) -> float:
    """Return the learned perceptual image patch similarity of two ``(3, H, W)`` images.

    Lower is better. Inputs are clipped to the data range. The network, which matches
    the reference ``lpips`` package, is loaded once per backbone and device.

    Args:
        pred: Reconstruction.
        target: Reference signal.
        data_range: Peak value; ``None`` uses ``1`` for floating-point targets and the
            dtype's maximum for integer ones.
        net: Backbone of the network.

    References:
        Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual
        Metric", CVPR 2018.
    """
    peak = _data_range(target, data_range)
    pred, target = ((x.float() / peak).clamp(0, 1).unsqueeze(0) for x in (pred, target))
    return _lpips_network(net, pred.device)(pred, target).item()


def nmse(pred: Tensor, target: Tensor) -> float:
    """Return the normalized MSE ``||pred - target||^2 / ||target||^2``, as in fastMRI.

    Args:
        pred: Reconstruction.
        target: Reference signal.

    Examples:
        >>> nmse(torch.zeros(4), torch.ones(4))
        1.0
    """
    pred, target = pred.float(), target.float()
    return ((pred - target).square().sum() / target.square().sum()).item()


def iou(pred: Tensor, target: Tensor) -> float:
    """Return the intersection over union of occupancies, occupied above zero.

    Args:
        pred: Reconstruction.
        target: Reference signal.

    Examples:
        >>> iou(torch.tensor([1.0, 1.0, -1.0]), torch.tensor([1.0, -1.0, -1.0]))
        0.5
    """
    return binary_jaccard_index((pred > 0).int(), (target > 0).int()).item()


def bits_per_pixel(encoded: bytes, image: Tensor) -> float:
    """Return the encoded size in bits per pixel of a ``(channels, *spatial)`` image.

    Args:
        encoded: Encoded signal.
        image: Image whose spatial size counts the pixels.

    Examples:
        >>> bits_per_pixel(b"123", torch.zeros(3, 4, 6))
        1.0
    """
    return len(encoded) * 8 / math.prod(image.shape[1:])


def luma(image: Tensor) -> Tensor:
    """Return the BT.601 luma of a uint8 RGB image, as MATLAB's ``rgb2ycbcr``.

    Super-resolution reports PSNR and SSIM on this channel, with ``data_range=255``.

    Args:
        image: Image ``(3, H, W)``.

    Returns:
        Float image ``(1, H, W)`` in ``[16, 235]``.

    Examples:
        >>> white = torch.full((3, 2, 2), 255, dtype=torch.uint8)
        >>> round(luma(white).max().item(), 2)
        235.0
    """
    weights = torch.tensor([65.481, 128.553, 24.966], device=image.device) / 255
    return 16 + torch.einsum("c...,c->...", image.float(), weights).unsqueeze(0)


def super_resolution_metrics(
    border: int = 0, device: str | torch.device | None = None
) -> dict[str, Callable[[Tensor, Tensor], float]]:
    """Return the super-resolution protocol's metrics, as ``metrics`` arguments.

    These are PSNR and SSIM of the :func:`luma` channel and LPIPS of RGB.
    :func:`~inro.train` and :func:`~inro.evaluate` call them as
    ``fn(reconstruction, original)`` on uint8 ``(3, H, W)`` images.

    Args:
        border: Pixels shaved off each side before scoring, by convention the scale
            factor.
        device: Device to score on; ``None`` keeps the images where they are.

    Returns:
        The functions under the keys ``psnr``, ``ssim`` and ``lpips``.

    Examples:
        >>> sorted(super_resolution_metrics(border=4))
        ['lpips', 'psnr', 'ssim']
    """

    def shave(image: Tensor) -> Tensor:
        height, width = image.shape[-2:]
        return image.to(device)[:, border : height - border, border : width - border]

    def of_luma(metric: Callable[..., float]) -> Callable[[Tensor, Tensor], float]:
        return lambda pred, target: metric(
            luma(shave(pred)), luma(shave(target)), data_range=255
        )

    return {
        "psnr": of_luma(psnr),
        "ssim": of_luma(ssim),
        "lpips": lambda pred, target: lpips(shave(pred), shave(target)),
    }
