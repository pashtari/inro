"""Image, volume and MRI datasets for implicit neural representations (INRs).

A dataset has length one and ignores the item index. Its item is a dictionary with
``id``, ``input``, ``target`` and ``_original``, the reference signal for metrics.
Coordinate inputs lie in ``[-1, 1]``.

Dataset methods follow ``load(path, ...)``, ``save(x, path, ...)``, ``preprocess(x)``
(raw signal to target) and ``postprocess(x)`` (model output to reconstruction). Deep
Image Prior uses a fixed noise input instead of coordinates. Super-resolution observes
the image through a separable linear degradation, written out by :func:`resize_matrix`
and applied by :func:`downsample`.
"""

import math
import os
import random
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as FT

__all__ = [
    "create_coordinates",
    "mask_to_boundary",
    "resize_matrix",
    "downsample",
    "ImageCoordinateDataset",
    "MaskedImageCoordinateDataset",
    "SuperResolutionDataset",
    "OccupancyCoordinateDataset",
    "DIPImageDataset",
    "MRICoordinateDataset",
]


def create_coordinates(
    size: tuple[int, ...], domain: tuple[float, float] = (-1.0, 1.0), **kwargs: Any
) -> Tensor:
    """Return an evenly spaced coordinate grid of shape ``(*size, len(size))``.

    Coordinate ``c`` varies along axis ``c``, as with ``indexing="ij"``.

    Args:
        size: Number of points on each axis.
        domain: Endpoints of every axis, both included.
        **kwargs: Passed to ``torch.linspace``, such as ``dtype`` and ``device``.

    Examples:
        >>> create_coordinates((2, 3)).shape
        torch.Size([2, 3, 2])
    """
    axes = [torch.linspace(*domain, length, **kwargs) for length in size]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


def _load_image(
    image: str | os.PathLike[str] | Image.Image | Tensor,
    mode: str,
    item_id: str | None,
) -> tuple[Tensor, str]:
    """Return a ``(channels, H, W)`` tensor and ID from a path, PIL image or tensor.

    Paths and PIL images are converted to ``mode``; tensors are used as is. The ID
    defaults to the path stem, or an empty string for in-memory images.
    """
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as pil_image:
            return (
                FT.pil_to_tensor(pil_image.convert(mode)),
                item_id or Path(image).stem,
            )
    if isinstance(image, Image.Image):
        return FT.pil_to_tensor(image.convert(mode)), item_id or ""
    if isinstance(image, Tensor):
        return image, item_id or ""
    raise ValueError("image must be a str, os.PathLike, PIL.Image or Tensor")


def _random_subset(n: int, k: int, device: torch.device) -> Tensor:
    """Return indices of a uniformly random ``k``-subset of ``range(n)``, in no order.

    Takes the ``k`` largest of ``n`` uniform keys, distributed as ``randperm(n)[:k]``
    but without sorting all ``n``: sixteen times faster on a 3.8M-voxel volume.
    """
    return torch.rand(n, device=device).topk(k, sorted=False).indices


def _save_png(image: Tensor, path: str | os.PathLike[str]) -> None:
    """Save a ``(channels, H, W)`` image as PNG, appending ``.png`` if needed."""
    path = str(path)
    if not path.lower().endswith(".png"):
        path += ".png"
    FT.to_pil_image(image).save(path, format="PNG")


class ImageCoordinateDataset(Dataset):
    """Image as coordinate-pixel pairs, with optional random subsampling.

    Subsampling flattens ``input`` and ``target`` to ``(N, 2)`` and ``(N, channels)``
    with shared indices, a new uniformly random subset of the pixels on each access.

    Args:
        image: Path, PIL image, or uint8 tensor ``(channels, H, W)``.
        grid_size: Size of the coordinate grid; ``None`` uses the signal's size. Any
            other size serves evaluation at that resolution, without subsampling.
        mode: PIL conversion mode of paths and PIL images; ignored for tensors.
        max_samples: Cap on the coordinates drawn on each access; ``None`` sets no cap.
        subsample: Fraction of the coordinates drawn on each access, in ``(0, 1]``.
        item_id: Item identifier; ``None`` uses the path stem, or an empty string
            without a path.
        seed: Seed of Python's global RNG; sampling uses PyTorch's and ignores it.

    Shape:
        - ``input``: ``(H, W, 2)``, coordinates in ``[-1, 1]``.
        - ``target``: ``(H, W, channels)``, pixels in ``[-1, 1]``.
        - ``_original``: ``(channels, H, W)``, uint8 image.

    Examples:
        >>> image = torch.zeros(3, 4, 6, dtype=torch.uint8)
        >>> item = ImageCoordinateDataset(image)[0]
        >>> item["input"].shape, item["target"].shape
        (torch.Size([4, 6, 2]), torch.Size([4, 6, 3]))
        >>> ImageCoordinateDataset(image, subsample=0.5)[0]["input"].shape
        torch.Size([12, 2])
    """

    def __init__(
        self,
        image: str | os.PathLike[str] | Image.Image | Tensor,
        grid_size: tuple[int, int] | None = None,
        mode: str = "RGB",
        max_samples: int | None = None,
        subsample: float = 1.0,
        item_id: str | None = None,
        seed: int = 0,
    ) -> None:
        if not 0 < subsample <= 1:
            raise ValueError(f"subsample must be in (0, 1], got {subsample}")
        if max_samples is not None and max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")

        self.original, self.id = _load_image(image, mode, item_id)
        self.grid_size = grid_size or self.original.shape[1:]
        self.input = self.create_input()
        self.target = self.preprocess(self.original)
        self.subsample = subsample
        self.max_samples = max_samples
        self.seed = seed
        random.seed(self.seed)

    # Tensors moved by ``to``; ``original`` stays on the CPU.
    _device_attrs: tuple[str, ...] = ("input", "target", "side", "mask", "boundary")

    def to(self, device: str | torch.device) -> Self:
        """Move the training tensors to ``device`` in place and return self.

        The reference ``original`` stays on the CPU.

        Args:
            device: Device to move them to.
        """
        for name in self._device_attrs:
            value = getattr(self, name, None)
            if isinstance(value, Tensor):
                setattr(self, name, value.to(device))
        return self

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = {"id": self.id, "_original": self.original}
        if self.subsample < 1 or self.max_samples:
            num_coords = math.prod(self.grid_size)
            num_samples = int(num_coords * self.subsample)
            num_samples = min(num_samples, self.max_samples or num_samples)

            indices = _random_subset(num_coords, num_samples, self.input.device)

            data["input"] = self.input.flatten(end_dim=-2)[indices, :]
            data["target"] = self.target.flatten(end_dim=-2)[indices, :]

        else:
            data["input"] = self.input
            data["target"] = self.target

        return data

    @staticmethod
    def load(path: str | os.PathLike[str], **kwargs: Any) -> Tensor:
        """Load an image as a tensor, uint8 for eight-bit modes.

        Args:
            path: Image file.
            **kwargs: Passed to PIL's ``convert``, such as ``mode="RGB"``.

        Returns:
            Image ``(channels, H, W)``.
        """
        with Image.open(path) as image:
            return FT.pil_to_tensor(image.convert(**kwargs))

    @staticmethod
    def save(x: Tensor, path: str | os.PathLike[str]) -> None:
        """Save an image as PNG.

        Args:
            x: Image ``(channels, H, W)``.
            path: Output path; ``.png`` is appended if needed.
        """
        _save_png(x, path)

    def create_input(self) -> Tensor:
        """Return the coordinates ``(*grid_size, len(grid_size))`` in ``[-1, 1]``."""
        return create_coordinates(self.grid_size, (-1.0, 1.0))

    @staticmethod
    def preprocess(x: Tensor) -> Tensor:
        """Map uint8 pixels to float values in ``[-1, 1]``, channels last.

        Args:
            x: Image ``(channels, H, W)``.

        Returns:
            Target ``(H, W, channels)``.
        """
        x = x.float() * 2 / 255 - 1
        # Contiguous, as a loader's batches are, so that a graph compiled on an item
        # serves them; the moved view keeps the channel stride.
        return x.movedim(0, -1).clone(memory_format=torch.contiguous_format)

    @staticmethod
    def postprocess(x: Tensor) -> Tensor:
        """Map float predictions to uint8 pixels, clamping them to ``[-1, 1]`` first.

        Args:
            x: Predictions ``(H, W, channels)``.

        Returns:
            Image ``(channels, H, W)``.
        """
        x = x.movedim(-1, 0).clamp(-1, 1)
        return ((x + 1) * 255 / 2).round().to(torch.uint8)


def mask_to_boundary(
    mask: Tensor,
    kernel_size: int = 3,
    mode: Literal["inner", "outer", "thick"] = "thick",
) -> Tensor:
    """Extract the boundary of a binary mask by dilation and erosion.

    Args:
        mask: Binary tensor ``(H, W)``, ``(B, H, W)`` or ``(B, 1, H, W)``.
        kernel_size: Odd side of the pooling neighborhood.
        mode: ``"inner"`` (mask minus erosion), ``"outer"`` (dilation minus mask) or
            ``"thick"`` (dilation minus erosion).

    Returns:
        Binary boundary with the shape, dtype and device of ``mask``.

    Examples:
        >>> mask = torch.zeros(5, 5, dtype=torch.bool)
        >>> mask[1:4, 1:4] = True
        >>> mask_to_boundary(mask, mode="inner").sum().item()
        8
    """
    original_ndim = mask.ndim
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4:
        if mask.shape[1] != 1:
            raise ValueError("mask should be single-channel (C=1) or 2/3-D tensor")
    else:
        raise ValueError("mask must be 2D, 3D (B,H,W), or 4D (B,1,H,W)")

    float_mask = mask.float()
    padding = kernel_size // 2

    dilation = F.max_pool2d(float_mask, kernel_size, stride=1, padding=padding)

    erosion = 1.0 - F.max_pool2d(
        1.0 - float_mask, kernel_size, stride=1, padding=padding
    )

    if mode == "inner":
        boundary = float_mask - erosion
    elif mode == "thick":
        boundary = dilation - erosion
    elif mode == "outer":
        boundary = dilation - float_mask
    else:
        raise ValueError("mode must be 'inner', 'outer' or 'thick'")

    boundary = (boundary > 0.0).to(mask.dtype)

    if original_ndim == 2:
        return boundary[0, 0]
    if original_ndim == 3:
        return boundary[:, 0]
    return boundary


class MaskedImageCoordinateDataset(ImageCoordinateDataset):
    """Image as coordinate-pixel pairs, with an optional guide image and binary mask.

    The preprocessed guide extends the coordinates as extra input channels, and a custom
    loss can use the mask and its thick boundary. Inputs always use the image
    resolution. Subsampling flattens every tensor but ``_original`` with shared indices.

    Args:
        image: Path, PIL image (converted to RGB), or uint8 tensor ``(channels, H, W)``.
        side: Guide image as a path, PIL image (converted to grayscale), or uint8 tensor
            ``(side_channels, H, W)``; ``None`` uses no guide.
        mask: Binary mask as a path, PIL image (converted to grayscale), or tensor
            ``(1, H, W)`` or ``(H, W)``, nonzero values being true; ``None`` uses no
            mask.
        subsample: Fraction of the coordinates drawn on each access, in ``(0, 1]``.
        item_id: Item identifier; ``None`` uses the path stem, or an empty string
            without a path.
        seed: Seed of Python's global RNG; sampling uses PyTorch's and ignores it.

    Shape:
        - ``input``: ``(H, W, 2)``, or ``(H, W, 2 + side_channels)`` with a guide.
        - ``target``: ``(H, W, channels)``, pixels in ``[-1, 1]``.
        - ``side``: ``(H, W, side_channels)``, guide in ``[-1, 1]``, with a guide only.
        - ``mask``, ``boundary``: ``(H, W)``, bool, with a mask only.
        - ``_original``: ``(channels, H, W)``, uint8 image.

    Examples:
        >>> image = torch.zeros(3, 4, 6, dtype=torch.uint8)
        >>> item = MaskedImageCoordinateDataset(image, mask=torch.ones(4, 6))[0]
        >>> item["input"].shape, item["mask"].shape, item["boundary"].dtype
        (torch.Size([4, 6, 2]), torch.Size([4, 6]), torch.bool)
    """

    def __init__(
        self,
        image: str | os.PathLike[str] | Image.Image | Tensor,
        side: str | os.PathLike[str] | Image.Image | Tensor | None = None,
        mask: str | os.PathLike[str] | Image.Image | Tensor | None = None,
        subsample: float = 1.0,
        item_id: str | None = None,
        seed: int = 0,
    ) -> None:
        if not 0 < subsample <= 1:
            raise ValueError(f"subsample must be in (0, 1], got {subsample}")

        self.original, self.id = _load_image(image, "RGB", item_id)
        if side is not None:
            side, _ = _load_image(side, "L", None)
        if mask is not None:
            mask, _ = _load_image(mask, "L", None)

        self.grid_size = self.original.shape[-2:]
        self.input, self.side, self.mask, self.boundary = self.create_input(side, mask)
        self.target = self.preprocess(self.original)
        self.subsample = subsample
        self.seed = seed
        random.seed(self.seed)

    def create_input(
        self, side: Tensor | None, mask: Tensor | None
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        """Return ``(input, side, mask, boundary)`` at the image resolution.

        Args:
            side: Uint8 guide ``(side_channels, H, W)``, or ``None``.
            mask: Mask ``(1, H, W)`` or ``(H, W)``, or ``None``.

        Returns:
            Coordinates with the preprocessed guide appended, that guide, the bool mask
                and its thick boundary; an absent guide or mask gives ``None``.
        """
        coords = create_coordinates(self.grid_size, (-1.0, 1.0))

        if isinstance(side, Tensor):
            side = self.preprocess(side)
            inputs = torch.cat([coords, side], dim=-1)
        else:
            side = None
            inputs = coords

        if isinstance(mask, Tensor):
            mask = mask.squeeze(0).bool()
            boundary = mask_to_boundary(mask, kernel_size=3, mode="thick")
        else:
            mask = None
            boundary = None

        return inputs, side, mask, boundary

    def __getitem__(self, index: int) -> dict[str, Any]:
        fields = {
            "input": self.input,
            "target": self.target,
            "side": self.side,
            "mask": self.mask,
            "boundary": self.boundary,
        }
        fields = {key: value for key, value in fields.items() if value is not None}
        if self.subsample < 1:
            num_coords = self.input.shape[0] * self.input.shape[1]
            num_samples = int(num_coords * self.subsample)
            indices = _random_subset(num_coords, num_samples, self.input.device)
            # All share the (H, W) pixel grid in their leading dimensions.
            fields = {
                key: value.flatten(0, 1)[indices] for key, value in fields.items()
            }
        return {"id": self.id, "_original": self.original, **fields}


def resize_matrix(
    size_in: int, size_out: int, device: str | torch.device | None = None
) -> Tensor:
    """Return the matrix of antialiased bicubic resizing on one axis.

    It is the operator of ``F.interpolate(..., mode="bicubic", antialias=True)``, whose
    kernel is that of MATLAB's ``imresize`` and PIL's bicubic filter, with which
    super-resolution benchmarks such as DIV2K make their low-resolution images.

    Args:
        size_in: Size of the input axis.
        size_out: Size of the output axis.
        device: Device of the matrix; ``None`` uses the default device.

    Returns:
        Matrix ``(size_out, size_in)``.

    Examples:
        >>> kernel = resize_matrix(64, 16)
        >>> kernel.shape
        torch.Size([16, 64])
        >>> (kernel != 0).sum(1).max().item()  # taps of one low-resolution pixel
        16
    """
    # Resizing the rows of an identity image writes the matrix out, borders included.
    identity = torch.eye(size_in, device=device)[None, None]
    size = (size_out, size_in)
    return F.interpolate(identity, size=size, mode="bicubic", antialias=True)[0, 0]


def downsample(image: Tensor, kernels: Sequence[Tensor]) -> Tensor:
    """Apply one kernel per axis to an image.

    Args:
        image: Image ``(..., H, W, channels)``.
        kernels: Matrices ``(..., h, H)`` and ``(..., w, W)``: those of
            :func:`resize_matrix`, or batches of their rows, such as each block of
            :class:`SuperResolutionDataset` carries.

    Returns:
        Image ``(..., h, w, channels)``.

    Examples:
        >>> kernels = resize_matrix(8, 4), resize_matrix(16, 4)
        >>> downsample(torch.ones(8, 16, 3), kernels).shape
        torch.Size([4, 4, 3])
    """
    return torch.einsum("...il,...lmc,...jm->...ijc", kernels[0], image, kernels[1])


class SuperResolutionDataset(ImageCoordinateDataset):
    """Low-resolution image as blocks of pixels and the coordinates under them.

    An INR is fitted to the low-resolution (LR) image through its degradation from high
    resolution (HR), one matrix per axis. Each access draws ``B`` random square blocks
    of ``S`` LR pixels a side, with the global PyTorch RNG, and
    :func:`~inro.super_resolution_loss` renders their footprints of ``L`` HR pixels a
    side. Neighboring pixels share most of their footprint: at 4x, an 8 x 8 block takes
    30 INR evaluations per pixel against 256 for a single pixel.

    Args:
        image: LR image as a path, PIL image (converted to RGB), or uint8 tensor
            ``(channels, h, w)``.
        kernels: Degradation matrices ``(h, H)`` and ``(w, W)``, such as those of
            :func:`resize_matrix`. They set the HR grid and the device of the training
            tensors.
        num_blocks: Blocks ``B`` drawn on each access.
        block_size: LR pixels ``S`` on a side of a block.
        item_id: Item identifier; ``None`` uses the path stem, or an empty string
            without a path.

    Shape:
        - ``input``: ``(B, L, L, 2)``, HR coordinates of each block's footprint.
        - ``weight_y``, ``weight_x``: ``(B, S, L)``, kernel rows of each block on its
          footprint.
        - ``target``: ``(B, S, S, channels)``, LR pixels in ``[-1, 1]``.
        - ``_original``: ``(channels, h, w)``, uint8 LR image.

    Examples:
        >>> image = torch.zeros(3, 8, 8, dtype=torch.uint8)
        >>> kernels = resize_matrix(32, 8), resize_matrix(32, 8)
        >>> dataset = SuperResolutionDataset(image, kernels, num_blocks=2, block_size=2)
        >>> item = dataset[0]
        >>> item["input"].shape, item["weight_y"].shape, item["target"].shape
        (torch.Size([2, 20, 20, 2]), torch.Size([2, 2, 20]), torch.Size([2, 2, 2, 3]))
    """

    def __init__(
        self,
        image: str | os.PathLike[str] | Image.Image | Tensor,
        kernels: Sequence[Tensor],
        num_blocks: int,
        block_size: int = 8,
        item_id: str | None = None,
    ) -> None:
        device = kernels[0].device
        self.original, self.id = _load_image(image, "RGB", item_id)  # (channels, h, w)
        size = tuple(self.original.shape[1:])
        if tuple(kernel.shape[0] for kernel in kernels) != size:
            raise ValueError(
                f"kernels of {[tuple(k.shape) for k in kernels]} do not map onto "
                f"the {size[0]} x {size[1]} LR image"
            )
        if not 0 < block_size <= min(size):
            raise ValueError(
                f"block_size must be in [1, {min(size)}], got {block_size}"
            )
        self.kernels = list(kernels)
        self.num_blocks, self.block_size = num_blocks, block_size
        self.grid_size = tuple(kernel.shape[1] for kernel in kernels)  # (H, W)
        self.input = self.create_input().to(device)  # HR coordinates, (H, W, 2)
        self.target = self.preprocess(self.original).to(device)  # (h, w, channels)

        # Each kernel row's first tap and one past its last. A block's footprint L is
        # the widest row's span, zeros inside included as at 3x and 5x, plus scale per
        # further row.
        self.first = [(kernel != 0).int().argmax(1) for kernel in kernels]
        ends = [k.shape[1] - (k != 0).flip(1).int().argmax(1) for k in kernels]
        taps = max(int((end - first).max()) for end, first in zip(ends, self.first))
        scale = self.grid_size[0] // self.target.shape[0]
        self.footprint = scale * (block_size - 1) + taps
        if self.footprint > min(self.grid_size):
            raise ValueError(
                f"a block's footprint, {self.footprint} HR pixels, exceeds the "
                f"{self.grid_size[0]} x {self.grid_size[1]} HR grid"
            )

    def to(self, device: str | torch.device) -> Self:
        """Move the training tensors and kernels to ``device`` in place and return self.

        The reference ``original`` stays on the CPU.

        Args:
            device: Device to move them to.
        """
        super().to(device)
        self.kernels = [kernel.to(device) for kernel in self.kernels]
        self.first = [first.to(device) for first in self.first]
        return self

    def __getitem__(self, index: int) -> dict[str, Any]:
        device = self.target.device
        h, w = self.target.shape[:2]
        offsets = torch.arange(self.block_size, device=device)
        span = torch.arange(self.footprint, device=device)

        pixels, taps, weights = [], [], []
        for size, full, kernel, first in zip(
            (h, w), self.grid_size, self.kernels, self.first
        ):
            start = torch.randint(
                size - self.block_size + 1, (self.num_blocks, 1), device=device
            )
            pixel = start + offsets  # (B, S)
            tap = first[start].clamp(max=full - self.footprint) + span  # (B, L)
            weights.append(kernel[pixel[:, :, None], tap[:, None, :]])  # (B, S, L)
            pixels.append(pixel)
            taps.append(tap)

        return {
            "id": self.id,
            "input": self.input[taps[0][:, :, None], taps[1][:, None, :]],
            "weight_y": weights[0],
            "weight_x": weights[1],
            "target": self.target[pixels[0][:, :, None], pixels[1][:, None, :]],
            "_original": self.original,
        }


class OccupancyCoordinateDataset(ImageCoordinateDataset):
    """Voxelized shape as coordinate-occupancy pairs.

    The shape is centered and scaled into ``[-0.45, 0.45]^3``, as in BACON and FINER.
    Voxel centers lie at multiples of ``1 / resolution``, and each holds whether it
    lies inside the shape; the volume is cropped to its occupied bounds, as in WIRE.
    A mesh's inside is decided by rays: a center is inside when most of five rays from
    it cross the surface an odd number of times, which tolerates the small holes of
    scanned meshes. A point cloud's is the sign of BACON's signed distance. The volume
    is cached beside the source in ``.cache/<stem>_<resolution>_centers.pt``, which is
    reused even if the source changes. Subsampling flattens ``input`` and ``target`` to
    ``(N, 3)`` and ``(N, 1)`` with shared indices.

    Args:
        path: ``.xyz`` point cloud (columns ``x y z nx ny nz``) or a mesh readable by
            trimesh.
        resolution: Voxels per unit length after normalization into
            ``[-0.45, 0.45]^3``.
        grid_size: Size of the coordinate grid; ``None`` uses the signal's size. Any
            other size serves evaluation at that resolution, without subsampling.
        max_samples: Cap on the coordinates drawn on each access; ``None`` sets no cap.
        subsample: Fraction of the coordinates drawn on each access, in ``(0, 1]``.
        item_id: Item identifier; ``None`` uses the path stem, or an empty string
            without a path.
        seed: Seed of Python's global RNG; sampling uses PyTorch's and ignores it.

    Shape:
        - ``input``: ``(depth, H, W, 3)``, coordinates in ``[-1, 1]``.
        - ``target``: ``(depth, H, W, 1)``, floats in ``{-1, 1}``.
        - ``_original``: ``(1, depth, H, W)``, float occupancy in ``{0, 1}``.

    Examples:
        >>> dataset = OccupancyCoordinateDataset("lucy.ply")  # doctest: +SKIP

    References:
        Lindell et al., "BACON: Band-limited Coordinate Networks for Multiscale Scene
        Representation", CVPR 2022.
        Saragadam et al., "WIRE: Wavelet Implicit Neural Representations", CVPR 2023.
        Liu et al., "FINER: Flexible Spectral-bias Tuning in Implicit NEural
        Representation by Variable-periodic Activation Functions", CVPR 2024.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        resolution: int = 256,
        grid_size: tuple[int, int, int] | None = None,
        max_samples: int | None = None,
        subsample: float = 1.0,
        item_id: str | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            self.load(path, resolution),
            grid_size=grid_size,
            max_samples=max_samples,
            subsample=subsample,
            item_id=item_id or Path(path).stem,
            seed=seed,
        )

    def load(self, path: str | os.PathLike[str], resolution: int) -> Tensor:
        """Load, or build and cache, a float occupancy volume.

        Meshes need the ``3d`` extra (trimesh reads them, Open3D casts the rays).

        Args:
            path: ``.xyz`` point cloud (columns ``x y z nx ny nz``) or a mesh readable
                by trimesh.
            resolution: Voxels per unit length after normalization into
                ``[-0.45, 0.45]^3``.

        Returns:
            Occupancy ``(1, depth, H, W)``, cropped to its occupied bounds.
        """
        path = Path(path)
        cache_path = path.parent / ".cache" / f"{path.stem}_{resolution}_centers.pt"
        if cache_path.exists():
            return torch.load(cache_path, weights_only=True).float()

        if path.suffix.lower() == ".xyz":
            vertices, inside = self._point_cloud(path)
        else:
            vertices, inside = self._mesh(path)
        grid = self._voxelize(vertices, inside, resolution)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(grid, cache_path)
        return grid.float()

    @staticmethod
    def _normalize(vertices: np.ndarray) -> np.ndarray:
        """Center on the mean and scale the largest extent to fill ``[-0.45, 0.45]``."""
        vertices = vertices - vertices.mean(axis=0)
        low, high = vertices.min(), vertices.max()
        return (vertices - low) / (high - low) * 0.9 - 0.45

    @staticmethod
    def _voxelize(
        vertices: np.ndarray,
        inside: Callable[[np.ndarray], np.ndarray],
        resolution: int,
    ) -> Tensor:
        """Mark the voxel centers ``inside`` the shape, cropped to the occupied ones.

        The centers, at multiples of ``1 / resolution``, span the shape's bounds and are
        tested a few slabs at a time, which bounds memory at any resolution.

        Returns:
            Occupancy ``(1, depth, H, W)`` as ``uint8``.
        """
        low = np.floor(vertices.min(axis=0) * resolution).astype(int)
        high = np.ceil(vertices.max(axis=0) * resolution).astype(int)
        axes = [np.arange(lo, hi + 1) / resolution for lo, hi in zip(low, high)]
        plane = np.stack(np.meshgrid(axes[1], axes[2], indexing="ij"), axis=-1)
        plane = plane.reshape(-1, 2)
        grid = np.zeros([len(axis) for axis in axes], dtype=np.uint8)
        slabs = max(1, 2**21 // len(plane))
        for start in range(0, len(axes[0]), slabs):
            x = axes[0][start : start + slabs]
            points = np.column_stack(
                [np.repeat(x, len(plane)), np.tile(plane, (len(x), 1))]
            )
            occupied = inside(points.astype(np.float32))
            grid[start : start + len(x)] = occupied.reshape(len(x), *grid.shape[1:])

        bounds = []
        for axis in range(3):
            filled = np.flatnonzero(grid.any(axis=tuple({0, 1, 2} - {axis})))
            bounds.append(slice(filled[0], filled[-1] + 1))
        return torch.from_numpy(np.ascontiguousarray(grid[tuple(bounds)]))[None]

    @staticmethod
    def _mesh(path: Path) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        """Return a mesh's normalized vertices and its inside test, by five rays."""
        import trimesh

        os.environ.setdefault("OPEN3D_DISABLE_WEB_VISUALIZER", "true")
        import open3d as o3d

        mesh = trimesh.load(str(path), force="mesh")
        vertices = OccupancyCoordinateDataset._normalize(mesh.vertices)
        geometry = o3d.t.geometry.TriangleMesh()
        geometry.vertex.positions = o3d.core.Tensor(vertices.astype(np.float32))
        geometry.triangle.indices = o3d.core.Tensor(mesh.faces.astype(np.int32))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(geometry)

        def inside(points: np.ndarray) -> np.ndarray:
            occupancy = scene.compute_occupancy(o3d.core.Tensor(points), nsamples=5)
            return occupancy.numpy() > 0

        return vertices, inside

    @staticmethod
    def _point_cloud(
        path: Path,
    ) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        """Return a point cloud's normalized points and its inside test, BACON's.

        A query's signed distance is its offset from the nearest point, projected onto
        the mean normal of its three nearest neighbors; negative values are inside.
        """
        from scipy.spatial import cKDTree

        cloud = np.genfromtxt(path)
        vertices = OccupancyCoordinateDataset._normalize(cloud[:, :3])
        normals = cloud[:, 3:]
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        normals = normals / norms
        tree = cKDTree(vertices)

        def inside(points: np.ndarray) -> np.ndarray:
            _, neighbors = tree.query(points, k=3, workers=-1)
            mean_normals = normals[neighbors].mean(axis=1)
            mean_normals /= np.linalg.norm(mean_normals, axis=-1, keepdims=True) + 1e-10
            offsets = points - vertices[neighbors[:, 0]]
            return (offsets * mean_normals).sum(axis=-1) < 0

        return vertices, inside

    def save(
        self,
        x: Tensor,
        path: str | os.PathLike[str],
        smooth: bool = True,
        size: int = 1024,
        up: int = 1,
        mesh: bool = True,
    ) -> None:
        """Export occupancy as a ``.dae`` mesh and a ``.png`` render (``3d`` extra).

        The render needs no display; where the renderer cannot start, a warning replaces
        the ``.png``. Its camera is fitted to the grid rather than the mesh, so that
        reconstructions of one shape are framed alike.

        Args:
            x: Occupancy ``(1, depth, H, W)`` or ``(depth, H, W)``.
            path: Output path, whose suffix ``.dae`` and ``.png`` replace.
            smooth: Whether to apply Gaussian smoothing (sigma 1) before marching cubes
                at iso-level zero.
            size: Side of the square render, in pixels.
            up: Grid axis drawn upwards; the view looks down the shortest of the other
                two, which shows the widest silhouette.
            mesh: Whether to write the ``.dae``.
        """
        import mcubes

        # Imported in Jupyter, Open3D enables its web visualizer, which offscreen
        # renders do not use, and announces it.
        os.environ.setdefault("OPEN3D_DISABLE_WEB_VISUALIZER", "true")
        import open3d as o3d

        dae_path = Path(path).with_suffix(".dae")
        png_path = Path(path).with_suffix(".png")

        volume = x.squeeze(0).detach().cpu().numpy()
        if smooth:
            volume = mcubes.smooth(volume.copy(), method="gaussian", sigma=1)

        vertices, triangles = mcubes.marching_cubes(volume, 0)
        if mesh:
            mcubes.export_mesh(vertices, triangles, dae_path)

        geometry = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
        )
        geometry.compute_vertex_normals()
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        # Quietly: each renderer announces its EGL context.
        quiet = o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning)
        try:
            with quiet:
                renderer = o3d.visualization.rendering.OffscreenRenderer(size, size)
        except RuntimeError as error:  # a machine without EGL or a GPU
            warnings.warn(f"No renderer ({error}); no {png_path.name}.")
            return
        renderer.scene.add_geometry("mesh", geometry, material)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        renderer.scene.show_skybox(False)
        # Tone mapping would gray the white background and flatten the shading.
        renderer.scene.view.set_post_processing(False)
        extent = np.array(volume.shape, dtype=float)
        center = (extent - 1) / 2
        sideways = min({0, 1, 2} - {up}, key=lambda axis: extent[axis])
        across = ({0, 1, 2} - {up, sideways}).pop()
        # Far enough that the grid's nearest face, 4 % larger, fills the 45-degree
        # view along its longer side: a tall shape is never cut at the frame.
        fov = 45.0
        half = 1.04 * max(extent[up], extent[across]) / 2
        distance = half / math.tan(math.radians(fov / 2)) + extent[sideways] / 2
        eye = center + np.eye(3)[sideways] * distance
        renderer.setup_camera(fov, center, eye, np.eye(3)[up])
        o3d.io.write_image(str(png_path), renderer.render_to_image())

    @staticmethod
    def preprocess(x: Tensor) -> Tensor:
        """Map occupancy in ``{0, 1}`` to float targets in ``{-1, 1}``, channels last.

        Args:
            x: Occupancy ``(1, depth, H, W)``.

        Returns:
            Target ``(depth, H, W, 1)``.
        """
        x = x.float() * 2 - 1
        # As in ImageCoordinateDataset.preprocess.
        return x.movedim(0, -1).clone(memory_format=torch.contiguous_format)

    @staticmethod
    def postprocess(x: Tensor) -> Tensor:
        """Threshold predictions at zero into a float occupancy in ``{0, 1}``.

        Args:
            x: Predictions ``(depth, H, W, 1)``.

        Returns:
            Occupancy ``(1, depth, H, W)``.
        """
        x = x.movedim(-1, 0)
        return (x > 0).float()


class DIPImageDataset(Dataset):
    """Image paired with fixed noise for Deep Image Prior.

    The noise is ``0.1 * randn(noise_shape)``, and each access perturbs it anew.

    Args:
        image: Path, PIL image (converted to RGB), or uint8 tensor ``(channels, H, W)``.
        noise_shape: Shape of the noise; ``None`` uses ``(noise_channels, H, W)``.
        noise_channels: Channels of the noise when ``noise_shape`` is ``None``.
        reg_noise_std: Standard deviation of the Gaussian perturbation, drawn with the
            global PyTorch RNG; zero disables it, as for evaluation.
        item_id: Item identifier; ``None`` uses the path stem, or an empty string
            without a path.
        seed: Seed of the noise's own generator; the global RNGs are untouched.

    Shape:
        - ``input``: ``(noise_channels, H, W)`` or ``noise_shape``, which may exceed the
          image, as for super-resolution.
        - ``target``: ``(channels, H, W)``, floats in ``[0, 1]``.
        - ``_original``: ``(channels, H, W)``, uint8 image.

    Examples:
        >>> item = DIPImageDataset(torch.zeros(3, 4, 6, dtype=torch.uint8))[0]
        >>> item["input"].shape, item["target"].shape
        (torch.Size([16, 4, 6]), torch.Size([3, 4, 6]))

    References:
        Ulyanov et al., "Deep Image Prior", CVPR 2018.
    """

    def __init__(
        self,
        image: str | os.PathLike[str] | Image.Image | Tensor,
        noise_shape: tuple[int, ...] | None = None,
        noise_channels: int = 16,
        reg_noise_std: float = 1 / 30,
        item_id: str | None = None,
        seed: int = 0,
    ) -> None:
        self.original, self.id = _load_image(image, "RGB", item_id)
        self.target = self.preprocess(self.original)

        noise_shape = noise_shape or (noise_channels, *self.original.shape[1:])
        generator = torch.Generator().manual_seed(seed)
        self.input = torch.randn(noise_shape, generator=generator) * 0.1

        self.reg_noise_std = reg_noise_std

    def to(self, device: str | torch.device) -> Self:
        """Move the training tensors to ``device`` in place and return self.

        The reference ``original`` stays on the CPU.

        Args:
            device: Device to move them to.
        """
        self.input, self.target = self.input.to(device), self.target.to(device)
        return self

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        noise = self.input
        if self.reg_noise_std > 0:
            noise = noise + torch.randn_like(noise) * self.reg_noise_std
        return {
            "id": self.id,
            "input": noise,
            "target": self.target,
            "_original": self.original,
        }

    @staticmethod
    def save(x: Tensor, path: str | os.PathLike[str]) -> None:
        """Save an image as PNG.

        Args:
            x: Image ``(channels, H, W)``.
            path: Output path; ``.png`` is appended if needed.
        """
        _save_png(x, path)

    @staticmethod
    def preprocess(x: Tensor) -> Tensor:
        """Map uint8 pixels to float values in ``[0, 1]``, keeping the shape.

        Args:
            x: Image ``(channels, H, W)``.
        """
        return x.float() / 255.0

    @staticmethod
    def postprocess(x: Tensor) -> Tensor:
        """Map float predictions to uint8 pixels, clamping them to ``[0, 1]`` first.

        Args:
            x: Predictions ``(channels, H, W)``.
        """
        return (x.clamp(0, 1) * 255).round().to(torch.uint8)


class MRICoordinateDataset(Dataset):
    """Undersampled multi-coil MRI k-space for coordinate-based reconstruction.

    The model predicts interleaved real and imaginary coil images with ``num_coils * 2``
    channels. A custom loss pads these to the k-space size, Fourier-transforms them and
    compares acquired lines against ``target``.

    Args:
        kspace: Complex tensor ``(num_coils, H, W)`` or fastMRI HDF5 path; a file also
            provides ``reconstruction_rss`` when it has one.
        crop_size: Reconstruction field of view ``(crop_H, crop_W)``; ``None`` uses the
            size of the reference RSS image. Required for tensor inputs.
        mask: Mask of ``W`` elements, reshaped to bool ``(1, 1, W)``; ``None`` generates
            one.
        acceleration: Target acceleration of the generated mask.
        pattern: Line pattern of the generated mask, ``"random"`` or ``"uniform"``.
        center_fraction: Fraction of lines the generated mask keeps at the center.
        slice_index: Slice of the file to load; ``None`` uses the middle one.
        item_id: Item identifier; ``None`` uses the path stem, or an empty string
            without a path.
        seed: Seed of the mask's own generator.

    Shape:
        - ``input``: ``(crop_H, crop_W, 2)``, coordinates in ``[-1, 1]``.
        - ``target``: ``(num_coils, H, W)``, complex masked, normalized k-space.
        - ``mask``: ``(1, 1, W)``, bool sampling mask.
        - ``scale_factor``: Scalar normalization factor from :meth:`preprocess`.
        - ``zero_filled``: ``(crop_H, crop_W)``, RSS reconstruction in original units.
        - ``_original``: ``(crop_H, crop_W)``, reference RSS image, when one is loaded.

    Examples:
        >>> kspace = torch.randn(2, 8, 8, dtype=torch.complex64)
        >>> item = MRICoordinateDataset(kspace, crop_size=(4, 4))[0]
        >>> item["input"].shape, item["target"].shape, item["mask"].shape
        (torch.Size([4, 4, 2]), torch.Size([2, 8, 8]), torch.Size([1, 1, 8]))

    References:
        Zbontar et al., "fastMRI: An Open Dataset and Benchmarks for Accelerated MRI",
        arXiv 2018.
    """

    def __init__(
        self,
        kspace: str | os.PathLike[str] | Tensor,
        crop_size: tuple[int, int] | None = None,
        mask: Tensor | None = None,
        acceleration: float = 4.0,
        pattern: Literal["uniform", "random"] = "random",
        center_fraction: float = 0.08,
        slice_index: int | None = None,
        item_id: str | None = None,
        seed: int = 0,
    ) -> None:
        if isinstance(kspace, (str, os.PathLike)):
            self.id = item_id or Path(kspace).stem
            kspace, rss = self.load(kspace, slice_index=slice_index)
        elif isinstance(kspace, Tensor):
            self.id = item_id or ""
            rss = None
        else:
            raise ValueError("kspace must be a str, os.PathLike or Tensor")
        if kspace.ndim != 3 or not kspace.is_complex():
            raise ValueError(
                "kspace must be a complex (num_coils, H, W) tensor, got "
                f"{kspace.dtype} tensor of shape {tuple(kspace.shape)}"
            )

        self.kspace = kspace
        self.num_coils = kspace.shape[0]
        self.kspace_size = tuple(kspace.shape[-2:])
        if not crop_size and rss is None:
            raise ValueError(
                "crop_size must be given when no reconstruction_rss is available "
                "(e.g. for k-space tensor inputs)"
            )
        self.crop_size = crop_size or tuple(rss.shape[-2:])

        if mask is None:
            mask = self.create_undersampling_mask(
                num_cols=kspace.shape[-1],
                acceleration=acceleration,
                pattern=pattern,
                center_fraction=center_fraction,
                seed=seed,
            )
        # Shape (1, 1, W), which broadcasts against k-space (num_coils, H, W).
        self.mask = mask.reshape(1, 1, -1).bool()

        kspace_normalized, self.scale_factor = self.preprocess(
            self.kspace, self.mask, self.crop_size
        )

        self.input = create_coordinates(self.crop_size, (-1.0, 1.0))
        self.target = kspace_normalized * self.mask
        self.original = rss

        self.zero_filled = self.zero_filled_reconstruction(
            kspace, self.mask, self.crop_size
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {
            "id": self.id,
            "input": self.input,
            "target": self.target,
            "mask": self.mask,
            "scale_factor": self.scale_factor,
            "zero_filled": self.zero_filled,
        }
        if self.original is not None:  # a None would not collate into a batch
            item["_original"] = self.original
        return item

    @staticmethod
    def load(
        path: str | os.PathLike[str], slice_index: int | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Load one fastMRI slice (requires ``h5py``).

        Args:
            path: fastMRI HDF5 file.
            slice_index: Slice of the file to load; ``None`` uses the middle one.

        Returns:
            K-space ``(num_coils, H, W)`` and the reference RSS image
                ``(crop_H, crop_W)``, ``None`` when the file has none.
        """
        import h5py

        with h5py.File(path, "r") as data_file:
            kspace_all = torch.from_numpy(data_file["kspace"][()])
            if slice_index is None:
                slice_index = kspace_all.shape[0] // 2
            rss_target = None
            if "reconstruction_rss" in data_file:
                rss_target = torch.from_numpy(
                    data_file["reconstruction_rss"][slice_index]
                )

        return kspace_all[slice_index], rss_target

    @staticmethod
    def save(x: Tensor, path: str | os.PathLike[str]) -> None:
        """Min-max normalize a magnitude image and save it as an 8-bit PNG.

        Args:
            x: Magnitude image ``(H, W)``.
            path: Output path; ``.png`` is appended if needed.
        """
        image = x.detach().cpu().float()
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        _save_png((image * 255).round().to(torch.uint8).unsqueeze(0), path)

    @staticmethod
    def create_undersampling_mask(
        num_cols: int,
        acceleration: float,
        pattern: Literal["uniform", "random"] = "random",
        center_fraction: float = 0.08,
        seed: int = 0,
    ) -> Tensor:
        """Return a bool Cartesian sampling mask of shape ``(num_cols,)``.

        Center lines are always kept. ``"random"`` adds lines up to
        ``round(num_cols / acceleration)`` in total, unless the center alone exceeds it.
        ``"uniform"`` adds every ``round(acceleration)``-th line from a random offset,
        on top of the center.

        Args:
            num_cols: Number of lines.
            acceleration: Target acceleration of the generated mask.
            pattern: Line pattern of the generated mask, ``"random"`` or ``"uniform"``.
            center_fraction: Fraction of lines the generated mask keeps at the center.
            seed: Seed of the mask's own generator.
        """
        generator = torch.Generator().manual_seed(seed)

        mask = torch.zeros(num_cols, dtype=torch.bool)

        num_center = round(num_cols * center_fraction)
        center_start = (num_cols - num_center) // 2
        mask[center_start : center_start + num_center] = True

        if pattern == "uniform":
            step = round(acceleration)
            offset = torch.randint(0, step, (1,), generator=generator).item()
            mask[offset::step] = True
        elif pattern == "random":
            num_outer = max(round(num_cols / acceleration) - num_center, 0)
            outer_indices = torch.where(~mask)[0]
            permutation = torch.randperm(len(outer_indices), generator=generator)[
                :num_outer
            ]
            mask[outer_indices[permutation]] = True
        else:
            raise ValueError(f"Unknown pattern '{pattern}'. Use 'uniform' or 'random'.")

        return mask

    @staticmethod
    def zero_filled_reconstruction(
        kspace: Tensor, mask: Tensor, crop_size: tuple[int, int] | None = None
    ) -> Tensor:
        """Return the RSS magnitude of masked k-space, optionally center-cropped.

        The orthonormal inverse FFT is centered: ``ifftshift`` before it and
        ``fftshift`` after, which differ for odd sizes.

        Args:
            kspace: Complex k-space ``(num_coils, H, W)``.
            mask: Mask that broadcasts to ``kspace``.
            crop_size: Reconstruction field of view ``(crop_H, crop_W)``; ``None`` keeps
                ``(H, W)``.
        """
        kspace_masked = kspace * mask

        coil_images = torch.fft.fftshift(
            torch.fft.ifft2(
                torch.fft.ifftshift(kspace_masked, dim=(-2, -1)), norm="ortho"
            ),
            dim=(-2, -1),
        )

        if crop_size is not None:
            height, width = coil_images.shape[-2:]
            crop_height, crop_width = crop_size
            coil_images = coil_images[
                ...,
                (height - crop_height) // 2 : (height + crop_height) // 2,
                (width - crop_width) // 2 : (width + crop_width) // 2,
            ]

        return (coil_images.real**2 + coil_images.imag**2).sum(dim=-3).sqrt()

    def preprocess(
        self, x: Tensor, mask: Tensor, crop_size: tuple[int, int], eps: float = 1e-16
    ) -> tuple[Tensor, Tensor]:
        """Return normalized k-space and its scalar scale factor.

        Divides k-space by the maximum of its masked, cropped zero-filled RSS image. By
        linearity of the FFT, the same scale applies in both domains.

        Args:
            x: Complex k-space ``(num_coils, H, W)``.
            mask: Mask that broadcasts to ``x``.
            crop_size: Reconstruction field of view ``(crop_H, crop_W)``.
            eps: Lower bound of the scale factor.
        """
        zero_filled = self.zero_filled_reconstruction(x, mask, crop_size)
        scale_factor = torch.clamp(zero_filled.max(), min=eps)
        return x / scale_factor, scale_factor

    def postprocess(self, x: Tensor) -> Tensor:
        """Convert interleaved coil predictions to RSS magnitude in original units.

        The norm over the last dimension gives the RSS without complex tensors. It is
        clamped to ``[0, 1]``, then multiplied by ``scale_factor``.

        Args:
            x: Predictions ``(crop_H, crop_W, num_coils * 2)``.

        Returns:
            Magnitude ``(crop_H, crop_W)``.
        """
        return torch.clamp(torch.norm(x, dim=-1), min=0.0, max=1.0) * self.scale_factor
