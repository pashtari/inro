"""Symmetric per-tensor quantization for training and compression."""

import torch
from torch import Tensor, nn

__all__ = ["uniform_quantize", "uniform_dequantize"]


@torch.no_grad()
def uniform_quantize(
    model: nn.Module, quant_max: int
) -> tuple[nn.Module, dict[str, dict[str, Tensor]]]:
    """Fake-quantize model parameters in place and return their integer codes.

    Each parameter ``w`` becomes ``clamp(round(s * w), -quant_max, quant_max) / s``,
    with ``s = quant_max / max(abs(w))``, or ``s = 1`` for an all-zero tensor. Complex
    parameters are unsupported.

    Args:
        model: Module to quantize in place; pass a copy to keep the original weights.
        quant_max: Largest integer level, less than ``2**63``; codes use the smallest
            signed integer dtype that holds it.

    Returns:
        The same model and a dictionary mapping parameter names to
            ``{"quant": codes, "scale": s}``, both detached CPU tensors.

    Examples:
        >>> model, codes = uniform_quantize(nn.Linear(2, 3), quant_max=127)
        >>> codes["weight"]["quant"].dtype
        torch.int8
        >>> restored = uniform_dequantize(nn.Linear(2, 3), codes)
        >>> restored.weight.equal(model.weight)
        True
    """
    for dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        if quant_max <= torch.iinfo(dtype).max:
            break
    else:
        raise ValueError(f"quant_max must be less than 2**63, got {quant_max}")

    model_dict: dict[str, dict[str, Tensor]] = {}
    for name, param in model.named_parameters():
        peak = param.abs().max()
        scale = quant_max / peak if peak > 0 else peak.new_ones(())
        codes = torch.clamp(torch.round(scale * param), -quant_max, quant_max)
        param.copy_(codes / scale)

        model_dict[name] = {
            "quant": codes.cpu().to(dtype),
            "scale": scale.cpu(),
        }

    return model, model_dict


@torch.no_grad()
def uniform_dequantize(
    model: nn.Module, model_dict: dict[str, dict[str, Tensor]]
) -> nn.Module:
    """Load parameters from :func:`uniform_quantize` codes in place.

    Args:
        model: Module to overwrite in place.
        model_dict: Codes and scales of every parameter of ``model``, by name.

    Returns:
        The same model, each parameter being ``quant.float() / scale.float()``.
    """
    for name, param in model.named_parameters():
        codes = model_dict[name]["quant"].float()
        scale = model_dict[name]["scale"].float()
        param.copy_(codes / scale)

    return model
