"""Image codecs: PIL formats (JPEG, WebP, PNG, ...) and FUTON.

An encoder maps an image to bytes and a decoder maps the bytes back to an image tensor
``(channels, H, W)``, so :func:`~inro.compression.evaluate_compression` can benchmark
any pair.
"""

import io
import os
from functools import partial
from typing import IO, Any

import torch
import torchvision.transforms.functional as FT
from PIL import Image
from torch import Tensor

from ..datasets import ImageCoordinateDataset, create_coordinates
from ..losses import rate_distortion_loss
from ..models import FUTON
from ..quantization import uniform_dequantize
from ..training import train
from ..utils import decode_dict, encode_dict

__all__ = ["encode_pil", "decode_pil", "encode_futon", "decode_futon"]


def encode_pil(
    image: str | os.PathLike[str] | IO[bytes] | Tensor, **kwargs: Any
) -> bytes:
    """Encode an image with a PIL codec.

    Args:
        image: Path or binary file object of an image that PIL can open, or an image
            tensor ``(channels, H, W)``, ``uint8`` or floating-point in ``[0, 1]``.
        **kwargs: Arguments of :meth:`PIL.Image.Image.save`. Must include ``format``,
            such as ``"JPEG"``, ``"WEBP"`` or ``"PNG"``, because PIL cannot infer it for
            an in-memory buffer, and may include codec options such as ``quality``.

    Returns:
        The encoded image file.

    Raises:
        ValueError: If ``image`` is not a path, a binary file object or a tensor.

    Examples:
        >>> image = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
        >>> encoded = encode_pil(image, format="JPEG", quality=50)
        >>> decode_pil(encoded).shape
        torch.Size([3, 32, 32])
    """
    if isinstance(image, (str, os.PathLike, io.IOBase)):
        pil_image = Image.open(image)
    elif isinstance(image, Tensor):
        pil_image = FT.to_pil_image(image)
    else:
        raise ValueError("image must be a string, Path, IO[bytes] or Tensor")

    buffer = io.BytesIO()
    pil_image.save(buffer, **kwargs)
    return buffer.getvalue()


def decode_pil(encoded: bytes) -> Tensor:
    """Decode an image file held in memory, such as the output of :func:`encode_pil`.

    Args:
        encoded: Bytes of an image file in any format PIL can read.

    Returns:
        Image ``(channels, H, W)``, with the channels of the stored PIL mode and
            ``uint8`` values for 8-bit images.
    """
    return FT.pil_to_tensor(Image.open(io.BytesIO(encoded)))


def _check_module_spec(name: str, spec: Any) -> None:
    """Raise ``ValueError`` unless ``spec`` is a key or a ``(key, dict)`` pair."""
    is_key = isinstance(spec, str)
    is_pair = (
        isinstance(spec, (tuple, list))
        and len(spec) == 2
        and isinstance(spec[0], str)
        and isinstance(spec[1], dict)
    )
    if not (is_key or is_pair):
        raise ValueError(
            f"encode_futon requires `{name}` as a registry key or a (key, dict) "
            "pair, so that the bitstream can be loaded with "
            f"torch.load(weights_only=True); got {spec!r}"
        )


def encode_futon(
    image: str | os.PathLike[str] | Image.Image | Tensor,
    *,
    subsample: float = 1.0,
    lr: float = 0.1,
    num_steps: int = 500,
    alpha: float = 0.005,
    quant_max: int = 127,
    quant_interval: int = -1,
    log_interval: int = 100,
    device: str | torch.device | None = None,
    seed: int = 0,
    **kwargs: Any,
) -> bytes:
    """Encode an image by fitting a FUTON and serializing its quantized parameters.

    Training minimizes :func:`~inro.losses.rate_distortion_loss`.

    Args:
        image: Path or PIL image, converted to RGB, or ``uint8`` tensor
            ``(channels, H, W)``.
        subsample: Fraction of pixels sampled at each step, in ``(0, 1]``.
        lr: Initial learning rate of Adam.
        num_steps: Number of training steps.
        alpha: Weight of the parameters' Laplace entropy estimate in the loss.
        quant_max: Largest integer level. Each parameter tensor is scaled by
            ``quant_max / max(abs(w))`` and rounded.
        quant_interval: Steps between fake quantizations of the model; non-positive
            values disable them. The final quantization always runs.
        log_interval: Steps between history records; see :func:`train`.
        device: Device to run on; ``None`` uses CUDA when available, else the CPU.
        seed: Seed of the global PyTorch RNG, set after the model is built, so it does
            not control its initialization.
        **kwargs: Arguments of :class:`FUTON`. ``basis``, ``combiner`` and ``decoder``
            are required, each a registry key or a ``(key, dict)`` pair, and every value
            must load with ``torch.load(weights_only=True)``. ``in_features`` and
            ``out_features`` default to 2 and the channel count.

    Returns:
        The integer codes, their scales, the model config, the image ID and the spatial
            shape, serialized with ``torch.save`` and gzip.

    Raises:
        ValueError: If ``basis``, ``combiner`` or ``decoder`` is missing, or is neither
            a registry key nor a ``(key, dict)`` pair.

    Examples:
        >>> image = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)
        >>> encoded = encode_futon(
        ...     image,
        ...     num_steps=2,
        ...     basis=("cosine", {"num_components": 8}),
        ...     combiner=("cp", {"rank": 8}),
        ...     decoder="linear",
        ... )
        >>> decode_futon(encoded).shape
        torch.Size([3, 16, 16])
    """
    for name in ("basis", "combiner", "decoder"):
        if name not in kwargs:
            raise ValueError(f"encode_futon requires the FUTON argument `{name}`")
        _check_module_spec(name, kwargs[name])

    dataset = ImageCoordinateDataset(image=image, subsample=subsample)
    model_config = {
        "in_features": dataset.input.shape[-1],
        "out_features": dataset.target.shape[-1],
        **kwargs,
    }
    model = FUTON(**model_config)
    train_results = train(
        model,
        dataset,
        loss_fn=partial(rate_distortion_loss, alpha=alpha),
        lr=lr,
        num_steps=num_steps,
        quantize=True,
        quant_max=quant_max,
        quant_interval=quant_interval,
        log_interval=log_interval,
        eval_interval=-1,
        device=device,
        seed=seed,
    )

    metadata = {
        "image": {"id": dataset.id, "shape": tuple(dataset.grid_size)},
        "model": {"config": model_config, "state": train_results["model_dict"]},
    }
    return encode_dict(metadata)


def decode_futon(encoded: bytes, *, device: str | torch.device | None = None) -> Tensor:
    """Reconstruct an image from the bytes of :func:`encode_futon`.

    Rebuilds the stored FUTON, dequantizes its parameters and evaluates the whole
    coordinate grid in one pass. The output is clamped to ``[-1, 1]`` and mapped to
    ``uint8``.

    Args:
        encoded: Bytes returned by :func:`encode_futon`.
        device: Device to run on; ``None`` uses the default device.

    Returns:
        Reconstructed ``uint8`` image ``(channels, H, W)`` on ``device``.
    """
    metadata = decode_dict(encoded)
    model_config = metadata["model"]["config"]
    model_dict = metadata["model"]["state"]
    image_shape = metadata["image"]["shape"]

    coordinates = create_coordinates(image_shape).to(device)

    model = FUTON(**model_config)
    uniform_dequantize(model, model_dict)

    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(coordinates)

    return ImageCoordinateDataset.postprocess(output)
