"""Rendering and held-out-view evaluation for radiance fields."""

import time
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from ..metrics import lpips, psnr, ssim
from .datasets import BlenderDataset
from .renderers import VolumeRenderer

__all__ = ["render_image", "evaluate", "render_views"]


@contextmanager
def _eval_mode(
    field: nn.Module, renderer: VolumeRenderer
) -> Generator[None, None, None]:
    """Put both modules in eval mode, restoring training mode if ``field`` had it."""
    was_training = field.training
    field.eval()
    renderer.eval()
    try:
        yield
    finally:
        if was_training:
            field.train()
            renderer.train()


@torch.no_grad()
def render_image(
    field: nn.Module,
    renderer: VolumeRenderer,
    rays_o: Tensor,
    rays_d: Tensor,
    *,
    background: Tensor | None = None,
    chunk_size: int | None = None,
) -> dict[str, Tensor]:
    """Render rays in chunks, without gradients or stratified jitter.

    Both modules enter eval mode, and return to training mode if ``field`` was training.

    Args:
        field: Radiance field; rendering runs on the device of its parameters.
        renderer: Volume renderer on the device of ``field``, such as from
            :func:`create_renderer`.
        rays_o: Ray origins ``(..., 3)``, such as ``(H, W, 3)`` for a view.
        rays_d: Unit ray directions ``(..., 3)``.
        background: Background color ``(3,)``; ``None`` uses white.
        chunk_size: Rays per forward pass; ``None`` fits ``2**19`` marching samples.

    Returns:
        Dictionary with ``rgb`` ``(..., 3)``, ``opacity`` ``(..., 1)`` and ``depth``
            ``(..., 1)``, with the leading dimensions of ``rays_o``.
    """
    if chunk_size is None:
        # Marching buffers are allocated before the per-ray sample cap is applied.
        chunk_size = max(2**19 // getattr(renderer, "num_marching_steps", 1024), 1)
    ray_shape = rays_o.shape[:-1]
    device = next(field.parameters()).device
    rays_o = rays_o.reshape(-1, 3).to(device)
    rays_d = rays_d.reshape(-1, 3).to(device)
    outputs: dict[str, list[Tensor]] = {"rgb": [], "opacity": [], "depth": []}
    with _eval_mode(field, renderer):
        for start in range(0, rays_o.shape[0], chunk_size):
            result = renderer(
                field,
                rays_o[start : start + chunk_size],
                rays_d[start : start + chunk_size],
                background=background,
            )
            for key in outputs:
                outputs[key].append(result[key])
    return {
        key: torch.cat(values).reshape(*ray_shape, -1)
        for key, values in outputs.items()
    }


@torch.no_grad()
def evaluate(
    field: nn.Module,
    renderer: VolumeRenderer,
    dataset: BlenderDataset,
    *,
    indices: Iterable[int] | None = None,
    metrics: Mapping[str, Callable[[Tensor, Tensor], float]] | None = None,
    chunk_size: int | None = None,
    return_images: bool = False,
) -> dict[str, Any]:
    """Score held-out views, composited onto the dataset's background.

    Renders are clamped to ``[0, 1]`` before scoring. Both modules enter eval mode, and
    return to training mode if ``field`` was training.

    Args:
        field: Radiance field; rendering runs on the device of its parameters.
        renderer: Volume renderer on the device of ``field``, such as from
            :func:`create_renderer`.
        dataset: Posed RGBA images and the background they are composited onto.
        indices: View indices; ``None`` evaluates all views.
        metrics: Metrics by name, called as ``fn(pred, target)`` on images ``(3, H, W)``
            in ``[0, 1]``; ``None`` uses PSNR, SSIM and LPIPS (VGG).
        chunk_size: Rays per forward pass; ``None`` fits ``2**19`` marching samples.
        return_images: Whether to include the rendered images in the result.

    Returns:
        Dictionary with these keys.

        - ``views``: One record per view, with ``view`` (its index), ``duration``
          (seconds of its render), ``peak_memory`` (the most bytes it allocated beyond
          what the device already held, zero off CUDA) and the metric values.
        - ``mean``: Average of these entries except the index, empty without views.
        - ``images``: Rendered CPU tensors ``(H, W, 3)``, if ``return_images``.
    """
    if metrics is None:
        metrics = {"psnr": psnr, "ssim": ssim, "lpips": lpips}
    indices = range(len(dataset)) if indices is None else indices
    device = next(field.parameters()).device
    background = dataset.background.to(device)

    cuda = device.type == "cuda"
    views: list[dict[str, int | float]] = []
    images: list[Tensor] = []
    with _eval_mode(field, renderer):
        for index in indices:
            batch = dataset[index]
            if cuda:  # the view's own time and peak, after any work queued before
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            resident = torch.cuda.memory_allocated(device) if cuda else 0
            start_time = time.time()
            rendered = render_image(
                field,
                renderer,
                batch["rays_o"],
                batch["rays_d"],
                background=background,
                chunk_size=chunk_size,
            )
            if cuda:
                torch.cuda.synchronize(device)
            duration = time.time() - start_time
            peak = torch.cuda.max_memory_allocated(device) - resident if cuda else 0
            pred = rendered["rgb"].clamp(0.0, 1.0)
            target = dataset.composite(batch["rgba"].to(device), background)
            entry: dict[str, int | float] = {
                "view": int(index),
                "duration": duration,
                "peak_memory": peak,
            }
            for name, metric in metrics.items():
                entry[name] = metric(pred.permute(2, 0, 1), target.permute(2, 0, 1))
            views.append(entry)
            if return_images:
                images.append(pred.cpu())

    keys = [key for key in views[0] if key != "view"] if views else []
    mean = {key: sum(view[key] for view in views) / len(views) for key in keys}
    result: dict[str, Any] = {"views": views, "mean": mean}
    if return_images:
        result["images"] = images
    return result


@torch.no_grad()
def render_views(
    field: nn.Module,
    renderer: VolumeRenderer,
    dataset: BlenderDataset,
    poses: Sequence[Tensor],
    *,
    chunk_size: int | None = None,
) -> list[NDArray[np.uint8]]:
    """Render camera poses, such as an orbit, as ``uint8`` frames for a video.

    Both modules enter eval mode, and return to training mode if ``field`` was training.

    Args:
        field: Radiance field; rendering runs on the device of its parameters.
        renderer: Volume renderer on the device of ``field``, such as from
            :func:`create_renderer`.
        dataset: Source of the camera intrinsics and the background color.
        poses: Camera-to-world poses ``(4, 4)``, such as from :func:`pose_spherical`.
        chunk_size: Rays per forward pass; ``None`` fits ``2**19`` marching samples.

    Returns:
        List of ``uint8`` NumPy arrays ``(H, W, 3)``, one per pose.
    """
    background = dataset.background.to(next(field.parameters()).device)
    frames: list[NDArray[np.uint8]] = []
    with _eval_mode(field, renderer):
        for pose in poses:
            rays_o, rays_d = dataset.rays_for_pose(pose)
            rendered = render_image(
                field,
                renderer,
                rays_o,
                rays_d,
                background=background,
                chunk_size=chunk_size,
            )
            rgb = (rendered["rgb"].clamp(0, 1) * 255).round().to(torch.uint8)
            frames.append(rgb.cpu().numpy())
    return frames
