"""Per-scene NeRF training with random ray batches and mean squared error.

Adam keeps PyTorch's defaults, as in :func:`inro.train`, so the learning rate is the one
setting to tune. As in torch-ngp, it decays exponentially to ``0.1 * lr`` and each step
composites onto a fresh random background, while evaluation uses the dataset's. Results
and logging follow :func:`inro.train`.
"""

import math
import os
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm

from ..utils import _format_metrics, count_parameters, setup_experiment_loggers
from .datasets import BlenderDataset
from .evaluation import evaluate
from .renderers import VolumeRenderer

__all__ = ["train"]


def train(
    field: nn.Module,
    renderer: VolumeRenderer,
    train_dataset: BlenderDataset,
    eval_dataset: BlenderDataset | None = None,
    *,
    num_steps: int = 25000,
    num_rays: int = 4096,
    max_samples_per_ray: int | None = 1024,
    lr: float = 1e-2,
    color_lr: float | None = None,
    weight_decay: float = 0.0,
    adam_betas: tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    random_background: bool = True,
    metrics: Mapping[str, Callable[[Tensor, Tensor], float]] | None = None,
    eval_interval: int | None = None,
    eval_indices: Sequence[int] | None = None,
    log_interval: int = 100,
    device: str | torch.device | None = None,
    log_dir: str | os.PathLike[str] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit a radiance field to one scene with random ray batches.

    Each step updates the occupancy grid, renders ``num_rays`` rays and minimizes their
    RGB mean squared error with Adam. A batch without occupied samples skips the
    optimizer update and counts towards ``skipped_steps``.

    Args:
        field: Radiance field, moved to ``device``.
        renderer: Volume renderer, moved to ``device``.
        train_dataset: Posed images, moved to ``device`` in place.
        eval_dataset: Held-out views; ``None`` disables evaluation.
        num_steps: Number of training steps.
        num_rays: Rays sampled per step.
        max_samples_per_ray: Most occupied samples per ray, front to back; ``None``
            removes the cap. Overrides the renderer's setting.
        lr: Initial learning rate of Adam, decayed exponentially to ``0.1 * lr``.
        color_lr: Initial learning rate of ``field.color_net``, decayed alike; ``None``
            uses ``lr``.
        weight_decay: Weight decay of Adam.
        adam_betas: Moment decay rates of Adam.
        adam_eps: Epsilon of Adam.
        random_background: Whether to composite predictions and targets onto a fresh
            random color per ray, which discourages white haze in empty space.
            Otherwise, and always in evaluation, they use the dataset's background.
        metrics: Metrics passed to :func:`evaluate`; ``None`` uses PSNR, SSIM and LPIPS.
        eval_interval: Steps between evaluations, the final step always included;
            ``None`` evaluates only there and non-positive values never.
        eval_indices: View indices to evaluate; ``None`` evaluates all views.
        log_interval: Steps between history records; final and evaluation steps are
            always recorded, and non-positive values record only those.
        device: Device to run on; ``None`` uses CUDA when available, else the CPU.
        log_dir: Directory for ``log.txt`` (truncated first), ``log.json`` (appended),
            the final ``checkpoint.pt`` of ``field`` and ``renderer.pt``, the renderer's
            state with its occupancy grid; ``None`` writes no files.
        seed: Seed of the global PyTorch RNG, set after ``field`` and ``renderer`` are
            built, so it does not control their initialization.

    Returns:
        Dictionary with these keys.

        - ``config``: Settings of the run, with ``skipped_steps`` and ``peak_memory``,
          the most bytes a training step allocated beyond what the device already held
          (zero off CUDA).
        - ``model_dict``: Final state dict of ``field``.
        - ``history``: One record per logged step, with ``step``, ``elapsed`` (seconds
          spent training so far), ``loss``, ``psnr``, ``num_samples`` and, if the step
          evaluated, the mean held-out metrics under ``eval``.
    """
    logger, json_logger = setup_experiment_loggers(log_dir, __name__)
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    field = field.to(device)
    renderer = renderer.to(device)
    train_dataset.to(device)
    renderer.max_samples_per_ray = (
        None if max_samples_per_ray is None else int(max_samples_per_ray)
    )
    field.train()
    renderer.train()

    num_parameters = count_parameters(field)
    config = {
        "model": type(field).__name__,
        "density_net": type(getattr(field, "density_net", None)).__name__,
        "renderer": type(renderer).__name__,
        "num_params": num_parameters,
        "num_steps": num_steps,
        "num_rays": num_rays,
        "max_samples_per_ray": max_samples_per_ray,
        "lr": lr,
        "color_lr": color_lr,
        "weight_decay": weight_decay,
        "adam_betas": tuple(adam_betas),
        "adam_eps": adam_eps,
        "random_background": random_background,
        "seed": seed,
        "device": str(device),
        "num_train_views": len(train_dataset),
        "image_size": (train_dataset.height, train_dataset.width),
        "aabb": renderer.aabb.tolist(),
        "near": renderer.near,
        "far": renderer.far,
        "render_step_size": renderer.render_step_size,
        "grid_resolution": renderer.grid_resolution,
        "density_threshold": renderer.density_threshold,
    }
    logger.debug("Config: %s", config)

    groups = [{"params": field.parameters()}]
    if color_lr is not None:
        named = field.named_parameters()
        groups = [
            {"params": [p for n, p in named if not n.startswith("color_net.")]},
            {"params": field.color_net.parameters(), "lr": color_lr},
        ]
    optimizer = torch.optim.Adam(
        groups,
        lr=lr,
        betas=adam_betas,
        eps=adam_eps,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.1 ** min(step / num_steps, 1.0)
    )

    if eval_interval is None:
        eval_interval = num_steps
    history: list[dict[str, Any]] = []
    status = ""
    elapsed = 0.0
    skipped_steps = 0
    progress = tqdm(range(1, num_steps + 1), desc="Training")
    cuda = device.type == "cuda"
    resident = torch.cuda.memory_allocated(device) if cuda else 0
    peak_memory = 0

    start_time = time.perf_counter()
    for step in progress:
        if cuda:  # each step's own peak, without evaluations
            torch.cuda.reset_peak_memory_stats(device)

        # Step zero fills nerfacc's initially empty occupancy grid.
        renderer.update_occupancy(field, step - 1)

        batch = train_dataset.sample_rays(num_rays)
        rays_o, rays_d, rgba = batch["rays_o"], batch["rays_d"], batch["rgba"]

        background = (
            torch.rand_like(rgba[..., :3])
            if random_background
            else train_dataset.background
        )
        target = train_dataset.composite(rgba, background)

        result = renderer(field, rays_o, rays_d, background=background)
        loss = F.mse_loss(result["rgb"], target)

        optimizer.zero_grad()
        # Pure-background batches do not depend on the parameters.
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
        else:
            skipped_steps += 1
        scheduler.step()

        if cuda:
            peak = torch.cuda.max_memory_allocated(device) - resident
            peak_memory = max(peak_memory, peak)

        should_log = (log_interval > 0 and step % log_interval == 0) or (
            step == num_steps
        )
        should_eval = (
            eval_dataset is not None
            and eval_interval > 0
            and (step % eval_interval == 0 or step == num_steps)
        )
        if not (should_log or should_eval):
            continue  # no host sync, so the GPU never waits for Python

        if cuda:
            # CUDA kernels run asynchronously; wait so timings are wall-clock.
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - start_time
        mse = loss.detach().item()
        entry: dict[str, Any] = {
            "step": step,
            "elapsed": elapsed,
            "loss": mse,
            "psnr": 10 * math.log10(1 / max(mse, 1e-12)),
            "num_samples": result["num_samples"],
        }
        status = _format_metrics(entry)
        progress.set_postfix_str(status)

        if should_eval:
            entry["eval"] = evaluate(
                field,
                renderer,
                eval_dataset,
                indices=eval_indices,
                metrics=metrics,
            )["mean"]
            status += " - eval " + _format_metrics(entry["eval"])

        logger.info("Step [%d/%d] - %s", step, num_steps, status)
        json_logger.log(entry)
        history.append(entry)
        start_time = time.perf_counter()  # evaluating and logging is not training

    logger.info("Training completed in %.1f s - %s", elapsed, status)
    if skipped_steps:
        warnings.warn(
            f"{skipped_steps} of {num_steps} steps had no occupied samples and "
            f"were skipped; the field's occupancy grid is empty or nearly so. "
            f"Treat this run's metrics as invalid rather than as a score.",
            stacklevel=2,
        )
    config["skipped_steps"] = skipped_steps
    config["peak_memory"] = peak_memory

    if log_dir:
        checkpoint_path = os.path.join(log_dir, "checkpoint.pt")
        torch.save(field.state_dict(), checkpoint_path)
        # The grid decides which samples a render queries, so rendering the saved
        # radiance field needs the grid the run was scored with.
        torch.save(renderer.state_dict(), os.path.join(log_dir, "renderer.pt"))
        logger.info("Saved checkpoint to %s", checkpoint_path)

    return {"config": config, "model_dict": field.state_dict(), "history": history}
