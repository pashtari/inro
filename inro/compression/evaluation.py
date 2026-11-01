"""Rate-distortion evaluation of image codecs."""

import time
from collections.abc import Callable
from typing import Any

from torch import Tensor

from ..metrics import bits_per_pixel, psnr

__all__ = ["evaluate_compression"]


def evaluate_compression(
    image: Tensor,
    encode_fn: Callable[..., bytes],
    decode_fn: Callable[[bytes], Tensor],
    /,
    *,
    return_reconstruction: bool = False,
    **kwargs: Any,
) -> dict[str, float | Tensor]:
    """Measure an image codec's PSNR, bit rate, and encoding and decoding times.

    PSNR uses ``data_range=255`` and is infinite for lossless codecs. Timings are
    wall-clock, from ``perf_counter``, without CUDA synchronization.

    Args:
        image: Original ``uint8`` image ``(channels, H, W)``.
        encode_fn: Encoder, called as ``encode_fn(image, **kwargs)``.
        decode_fn: Decoder, called as ``decode_fn(encoded)``; its result must be on the
            device of ``image``.
        return_reconstruction: Whether to include the decoded image in the result.
        **kwargs: Arguments of the encoder.

    Returns:
        Dictionary with the floats ``"PSNR (dB)"``, ``"bit rate (bpp)"``,
            ``"encoding time (ms)"`` and ``"decoding time (ms)"``, and, if requested,
            the decoded tensor ``"reconstructed"``.

    Examples:
        >>> import torch
        >>> from inro.compression import decode_pil, encode_pil
        >>> image = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
        >>> results = evaluate_compression(image, encode_pil, decode_pil, format="PNG")
        >>> results["PSNR (dB)"]
        inf
    """
    start_time = time.perf_counter()
    encoded = encode_fn(image, **kwargs)
    encoding_time = 1000 * (time.perf_counter() - start_time)

    start_time = time.perf_counter()
    reconstruction = decode_fn(encoded)
    decoding_time = 1000 * (time.perf_counter() - start_time)

    results: dict[str, float | Tensor] = {
        "PSNR (dB)": psnr(reconstruction, image),
        "bit rate (bpp)": bits_per_pixel(encoded, image),
        "encoding time (ms)": encoding_time,
        "decoding time (ms)": decoding_time,
    }

    if return_reconstruction:
        results["reconstructed"] = reconstruction

    return results
