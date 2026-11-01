"""Chunked inference and evaluation of implicit neural representations (INRs)."""

import os
import time
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .utils import (
    _format_metrics,
    count_parameters,
    encode_dict,
    setup_experiment_loggers,
)

__all__ = ["evaluate", "chunked_inference"]


@torch.no_grad()
def chunked_inference(
    model: nn.Module,
    x: Tensor,
    *,
    chunk_size: int | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    """Evaluate coordinates in chunks under ``torch.no_grad()``.

    Moves ``model`` to ``device`` and leaves it in evaluation mode. Chunking flattens
    the leading dimensions, which the output restores.

    Args:
        model: INR to evaluate.
        x: Coordinates.
        chunk_size: Coordinates per forward pass; ``None`` or a non-positive value
            evaluates all at once.
        device: Device to run on; ``None`` uses CUDA when available, else the CPU.

    Returns:
        Output on ``device``.

    Shape:
        - Input: ``(..., C)``.
        - Output: ``(..., D)``.

    Examples:
        >>> x = torch.rand(4, 5, 2)
        >>> chunked_inference(nn.Linear(2, 3), x, chunk_size=8, device="cpu").shape
        torch.Size([4, 5, 3])
    """
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    if chunk_size is None or chunk_size <= 0:
        return model(x.to(device))

    *batch_shape, _ = x.shape
    x = x.flatten(0, -2)

    outputs = [model(chunk.to(device)) for chunk in x.split(chunk_size, dim=0)]
    return torch.cat(outputs, dim=0).reshape(*batch_shape, -1)


def evaluate(
    model: nn.Module,
    dataset: Dataset,
    *,
    model_dict: dict[str, Any] | None = None,
    metrics: Mapping[str, Callable[[Tensor, Tensor], float]] | None = None,
    rate_metrics: Mapping[str, Callable[[bytes, Tensor], float]] | None = None,
    chunk_size: int | None = None,
    device: str | torch.device = "cpu",
    ckpt_path: str | os.PathLike[str] | None = None,
    log_dir: str | os.PathLike[str] | None = None,
    save_reconstruction: bool = True,
) -> dict[str, Any]:
    """Reconstruct each dataset item, compute metrics and optionally save the result.

    Items must hold ``id``, ``input`` and ``_original``. The output passes through
    ``dataset.postprocess``, when it exists, and moves to the CPU for metrics. The model
    moves to ``device`` and stays in evaluation mode.

    Args:
        model: INR to evaluate.
        dataset: Dataset evaluated one item at a time.
        model_dict: Parameters encoded for rate metrics; ``None`` uses
            ``model.state_dict()``.
        metrics: Distortion metrics by name, called as ``fn(reconstruction, original)``.
        rate_metrics: Rate metrics by name, called as ``fn(encoded_bytes, original)``.
        chunk_size: Coordinates per forward pass; ``None`` or a non-positive value
            evaluates all at once.
        device: Device to run on.
        ckpt_path: State dict to load first, if the file exists.
        log_dir: Directory for ``log.txt`` and ``log.json`` (both appended) and
            reconstructions; ``None`` writes no files.
        save_reconstruction: Whether to save each reconstruction with ``dataset.save``
            as ``log_dir/{id}_reconstructed``.

    Returns:
        Dictionary with these keys.

        - ``config``: Holds ``model``, ``num_params`` and ``chunk_size``.
        - ``history``: One record per item, with ``id``, ``elapsed`` (seconds since
          the evaluation started), ``duration`` (seconds of its reconstruction,
          metrics left out), ``peak_memory`` (the most bytes it allocated beyond
          what the device already held, zero off CUDA) and the metric values.

    Examples:
        >>> from inro import ImageCoordinateDataset, psnr
        >>> dataset = ImageCoordinateDataset(torch.zeros(3, 8, 8, dtype=torch.uint8))
        >>> results = evaluate(nn.Linear(2, 3), dataset, metrics={"psnr": psnr})
        >>> sorted(results["history"][0])
        ['duration', 'elapsed', 'id', 'peak_memory', 'psnr']
    """
    model_dict = model_dict or model.state_dict()
    metrics = metrics or {}
    rate_metrics = rate_metrics or {}

    logger, json_logger = setup_experiment_loggers(log_dir, __name__, mode="a")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=not getattr(getattr(dataset, "input", None), "is_cuda", False),
        num_workers=0,
    )

    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    chunk_size = chunk_size or -1
    config: dict[str, Any] = {
        "model": type(model).__name__,
        "num_params": count_parameters(model),
        "chunk_size": chunk_size,
    }

    # Parameters are fixed during evaluation, so every sample shares one bitstream.
    encoded = encode_dict(model_dict) if rate_metrics else b""
    postprocess = getattr(dataset, "postprocess", None)
    cuda = torch.device(device).type == "cuda"

    history: list[dict[str, Any]] = []
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        item_id = batch["id"][0]
        x = batch["input"].to(device)
        original = batch["_original"][0]

        if cuda:  # the sample's own time and peak, after any work queued before
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        resident = torch.cuda.memory_allocated(device) if cuda else 0
        sample_start_time = time.time()

        output = chunked_inference(model, x, chunk_size=chunk_size, device=device)
        reconstruction = (
            torch.vmap(postprocess)(output) if postprocess is not None else output
        )
        reconstruction = reconstruction.cpu()[0]

        sample_duration = time.time() - sample_start_time
        elapsed_time = time.time() - start_time
        peak = torch.cuda.max_memory_allocated(device) - resident if cuda else 0

        results = {
            "id": item_id,
            "elapsed": elapsed_time,
            "duration": sample_duration,
            "peak_memory": peak,
        }

        for metric_name, metric_fn in metrics.items():
            results[metric_name] = metric_fn(reconstruction, original)
        for metric_name, metric_fn in rate_metrics.items():
            results[metric_name] = metric_fn(encoded, original)

        logger.debug(
            "Evaluation [%d/%d] %s - %s",
            i + 1,
            len(dataloader),
            item_id,
            _format_metrics(results),
        )
        json_logger.log(results)
        history.append(results)

        if log_dir and save_reconstruction:
            reconstruction_path = os.path.join(log_dir, f"{item_id}_reconstructed")
            dataset.save(reconstruction, reconstruction_path)
            logger.debug("Saved the reconstruction to %s", reconstruction_path)

    return {"config": config, "history": history}
