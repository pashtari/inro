"""Training losses and differentiable entropy estimates.

A loss follows ``loss_fn(batch, model) -> (loss, metrics)``: it runs the model on
``batch["input"]`` and returns a scalar loss and logging values, kept as scalar tensors
so that :func:`~inro.train` reads them only when it logs.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .datasets import downsample

__all__ = [
    "rate_distortion_loss",
    "sdf_loss",
    "super_resolution_loss",
    "soft_entropy",
    "laplace_entropy",
]


def _iqr(x: Tensor) -> Tensor:
    """Return the interquartile range over all elements."""
    q1 = torch.quantile(x, 0.25)
    q3 = torch.quantile(x, 0.75)
    return q3 - q1


def soft_entropy(
    x: Tensor,
    num_bins: int | None = None,
    temperature: float = 1.0,
    eps: float = 1e-16,
) -> Tensor:
    """Estimate entropy in bits with differentiable soft bin assignments.

    Bin centers span the detached input range. Each element gets a softmax assignment
    from its squared distance to them, and the entropy is that of the mean assignment.
    Gradients flow through the assignments, not the bin layout.

    Args:
        x: Input tensor; all elements are pooled.
        num_bins: Number of bins; ``None`` uses the Freedman–Diaconis width
            ``2 * IQR(x) / N**(1/3)``, rounding the count up, or Sturges' rule when the
            IQR vanishes.
        temperature: Assignment kernel width relative to bin width; larger is softer.
        eps: Constant added inside the logarithm.

    Returns:
        Scalar entropy, zero for a constant input.

    Examples:
        >>> x = torch.tensor([-1.0, 1.0])
        >>> soft_entropy(x, num_bins=2, temperature=0.1).item()
        1.0
    """
    x_flat = x.reshape(-1, 1)

    x_min = x_flat.detach().min()
    x_max = x_flat.detach().max()

    if x_max == x_min:  # a constant has no entropy
        return (x_flat * 0).sum()

    if num_bins is None:
        iqr_value = _iqr(x_flat.detach())
        num_elements = x_flat.numel()
        bin_width = 2 * iqr_value / (num_elements ** (1 / 3))
        if bin_width > 0:
            num_bins = int(torch.ceil((x_max - x_min) / bin_width).item())
        else:  # no Freedman-Diaconis width: Sturges' rule
            num_bins = math.ceil(math.log2(num_elements)) + 1
            bin_width = (x_max - x_min) / num_bins
    else:
        bin_width = (x_max - x_min) / num_bins

    bin_centers = torch.linspace(x_min, x_max, num_bins, dtype=x.dtype, device=x.device)

    squared_distances = (x_flat - bin_centers) ** 2
    sigma = bin_width * temperature
    logits = -squared_distances / (2 * sigma**2)
    assignments = F.softmax(logits, dim=1)

    probabilities = assignments.mean(dim=0)
    return -(probabilities * torch.log2(probabilities + eps)).sum()


def laplace_entropy(x: Tensor, eps: float = 1e-16) -> Tensor:
    """Return the differential entropy of a zero-mean Laplace fit, in bits.

    Uses the maximum-likelihood scale ``b = mean(abs(x))`` and the entropy
    ``log2(2 * b * e + eps)``.

    Args:
        x: Input tensor; all elements are pooled.
        eps: Constant added inside the logarithm.
    """
    scale = x.abs().mean()
    return torch.log2(2 * scale * math.e + eps)


def rate_distortion_loss(
    batch: dict[str, Any],
    model: nn.Module,
    alpha: float = 0.0,
    eps: float = 1e-16,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Balance reconstruction MSE with estimated parameter entropy.

    The loss is ``MSE + alpha * entropy``, the entropy being the size-weighted mean of
    :func:`laplace_entropy` over the trainable parameters, evaluated only when
    ``alpha > 0``.

    Args:
        batch: Batch with ``input`` and ``target``.
        model: INR to evaluate.
        alpha: Weight of the rate term.
        eps: Constant added inside the logarithm.

    Returns:
        Scalar loss and the metrics ``mse``, ``psnr`` (for targets in ``[-1, 1]``),
            ``loss``, and ``entropy`` when ``alpha > 0``, as scalar tensors.
    """
    output = model(batch["input"])
    target = batch["target"]
    metrics: dict[str, Tensor] = {}

    mse = F.mse_loss(output, target)
    metrics["mse"] = mse.detach()
    metrics["psnr"] = 10 * torch.log10(4.0 / metrics["mse"])

    if alpha > 0:
        entropy = 0
        total_params = 0
        for param in model.parameters():
            if not param.requires_grad:
                continue
            num_params = param.numel()
            total_params += num_params
            entropy += num_params * laplace_entropy(param, eps=eps)

        entropy /= total_params
        loss = mse + alpha * entropy
        metrics["entropy"] = entropy.detach()
    else:
        loss = mse

    metrics["loss"] = loss.detach()
    return loss, metrics


def sdf_loss(
    batch: dict[str, Any], model: nn.Module
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return the L1 signed-distance loss and the metrics ``loss`` and ``l1``.

    Args:
        batch: Batch with ``input`` and ``target``.
        model: INR to evaluate.
    """
    output = model(batch["input"])
    target = batch["target"]

    loss = F.l1_loss(output, target)
    value = loss.detach()
    return loss, {"loss": value, "l1": value}


def super_resolution_loss(
    batch: dict[str, Any], model: nn.Module
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return the L1 loss through the degradation and the metrics ``l1`` and ``psnr``.

    Renders each block's footprint, downsamples it with the block's kernel rows along
    both axes and matches the low-resolution pixels. The PSNR is for targets in
    ``[-1, 1]``.

    Args:
        batch: Batch of :class:`~inro.SuperResolutionDataset` items.
        model: INR to evaluate.
    """
    weights = batch["weight_y"], batch["weight_x"]
    output = downsample(model(batch["input"]), weights)
    target = batch["target"]

    loss = F.l1_loss(output, target)
    mse = F.mse_loss(output.detach(), target)
    return loss, {"l1": loss.detach(), "psnr": 10 * torch.log10(4.0 / mse)}
