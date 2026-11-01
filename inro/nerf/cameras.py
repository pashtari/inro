"""Camera utilities for novel view synthesis.

Follows the Blender and OpenGL convention of the NeRF synthetic dataset: in the camera
frame, ``x`` points right, ``y`` up, and the camera looks along ``-z``. A
camera-to-world pose is a ``(4, 4)`` matrix whose upper ``(3, 4)`` block is ``[R | t]``.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "focal_from_fov",
    "generate_ray_directions",
    "generate_rays",
    "pose_spherical",
]


def focal_from_fov(fov: float, size: int) -> float:
    """Convert a full field of view to a focal length in pixels.

    Args:
        fov: Full field of view in radians, such as ``camera_angle_x`` of the Blender
            transforms files, which pairs with the image width.
        size: Image size in pixels along the axis of ``fov``.
    """
    return 0.5 * size / math.tan(0.5 * fov)


def generate_ray_directions(
    height: int,
    width: int,
    focal: float,
    cx: float | None = None,
    cy: float | None = None,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Generate camera-frame ray directions ``(H, W, 3)`` through the pixel centers.

    Directions have ``z = -1``. They stay unnormalized, so that poses with the same
    intrinsics can share them.

    Args:
        height: Image height ``H`` in pixels.
        width: Image width ``W`` in pixels.
        focal: Focal length in pixels.
        cx: Principal point x in pixels; ``None`` uses ``width / 2``.
        cy: Principal point y in pixels; ``None`` uses ``height / 2``.
        device: Device to run on; ``None`` uses the default device.
        dtype: Floating-point dtype of the directions.
    """
    cx = 0.5 * width if cx is None else cx
    cy = 0.5 * height if cy is None else cy
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    return torch.stack(
        [(pixel_x - cx) / focal, -(pixel_y - cy) / focal, -torch.ones_like(pixel_x)],
        dim=-1,
    )


def generate_rays(
    c2w: Tensor, directions: Tensor, normalize: bool = True
) -> tuple[Tensor, Tensor]:
    """Generate world-space rays from camera poses and camera-frame directions.

    Args:
        c2w: Camera-to-world poses ``(..., 4, 4)`` or ``(..., 3, 4)``.
        directions: Camera-frame directions ``(..., 3)``, whose leading dimensions
            broadcast with those of ``c2w``: one pose ``(4, 4)`` with a grid
            ``(H, W, 3)``, or poses ``(N, 4, 4)`` with directions ``(N, 3)``.
        normalize: Whether to normalize the directions, so that distances along a ray
            are in world units.

    Returns:
        Ray origins and directions, each ``(..., 3)``. The origins are an expanded view
            of the translations of ``c2w``.
    """
    rotation = c2w[..., :3, :3]
    translation = c2w[..., :3, 3]
    # Elementwise multiplication supports a different pose for each ray.
    rays_d = (rotation * directions[..., None, :]).sum(-1)
    rays_o = translation.expand_as(rays_d)
    if normalize:
        rays_d = F.normalize(rays_d, dim=-1)
    return rays_o, rays_d


def pose_spherical(theta: float, phi: float, radius: float) -> Tensor:
    """Compute a camera-to-world pose on a sphere, looking at the origin.

    Uses the original NeRF convention, with world ``+z`` up. At ``theta = phi = 0`` the
    camera sits at ``(0, radius, 0)``.

    Args:
        theta: Azimuth in degrees from ``+y``, clockwise about ``+z`` seen from above:
            ``90`` places the camera on ``+x``.
        phi: Pitch in degrees. Negative values place the camera above the ``xy`` plane;
            the original NeRF render path uses ``-30``.
        radius: Distance from the origin.

    Returns:
        Pose ``(4, 4)`` on the default device, with the default dtype.
    """
    c2w = torch.eye(4)
    c2w[2, 3] = radius

    phi = math.radians(phi)
    pitch = torch.eye(4)
    pitch[1, 1] = math.cos(phi)
    pitch[1, 2] = -math.sin(phi)
    pitch[2, 1] = math.sin(phi)
    pitch[2, 2] = math.cos(phi)
    c2w = pitch @ c2w

    theta = math.radians(theta)
    yaw = torch.eye(4)
    yaw[0, 0] = math.cos(theta)
    yaw[0, 2] = -math.sin(theta)
    yaw[2, 0] = math.sin(theta)
    yaw[2, 2] = math.cos(theta)
    c2w = yaw @ c2w

    # Convert the orbit's rotation axes to Blender's world frame.
    blender_axes = torch.tensor(
        [[-1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]]
    )
    return blender_axes @ c2w
