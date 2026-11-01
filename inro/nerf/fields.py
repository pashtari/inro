"""Radiance fields assembled from spatial, color and direction modules.

As in Instant-NGP, a spatial network predicts density logits and geometry features, and
a color network combines those features with an encoded view direction. Holding the
color network and the renderer fixed compares spatial representations directly.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from ..models.mlp import MLP
from ..utils import ModuleSpec, build_module

__all__ = [
    "trunc_exp",
    "SphericalHarmonicsEncoding",
    "IdentityEncoding",
    "RadianceField",
]


class _TruncExp(torch.autograd.Function):
    """Exponential whose backward pass clamps its input, avoiding large gradients."""

    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        return grad_output * torch.exp(x.clamp(-15.0, 15.0))


def trunc_exp(x: Tensor) -> Tensor:
    """Compute ``exp(x)``, clamping the input of its gradient to ``[-15, 15]``.

    The clamp limits large density gradients early in training.

    Args:
        x: Input of any shape.
    """
    return _TruncExp.apply(x)


class SphericalHarmonicsEncoding(nn.Module):
    """Real spherical-harmonics encoding of unit directions.

    Evaluates the bands ``l = 0, ..., degree - 1``, so ``degree`` counts bands, as in
    tiny-cuda-nn, and gives ``degree**2`` features. ``degree=4`` matches the
    view-direction encoding of Instant-NGP.

    Args:
        degree: Number of bands, an integer from 1 to 4.

    Shape:
        - Input: ``(..., 3)`` unit directions.
        - Output: ``(..., degree**2)``.
    """

    def __init__(self, degree: int = 4) -> None:
        super().__init__()
        if not isinstance(degree, int) or not 1 <= degree <= 4:
            raise ValueError(f"Expected an integer 1 <= degree <= 4, got {degree!r}")
        self.degree = degree

    @property
    def out_features(self) -> int:
        """Number of output features, ``degree**2``."""
        return self.degree**2

    def forward(self, x: Tensor) -> Tensor:
        x, y, z = x.unbind(-1)
        features = [torch.full_like(x, 0.28209479177387814)]
        if self.degree > 1:
            features += [
                -0.48860251190291987 * y,
                0.48860251190291987 * z,
                -0.48860251190291987 * x,
            ]
        if self.degree > 2:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            features += [
                1.0925484305920792 * xy,
                -1.0925484305920792 * yz,
                0.94617469575755997 * zz - 0.31539156525251999,
                -1.0925484305920792 * xz,
                0.54627421529603959 * (xx - yy),
            ]
        if self.degree > 3:
            features += [
                0.59004358992664352 * y * (-3.0 * xx + yy),
                2.8906114426405538 * xy * z,
                0.45704579946446572 * y * (1.0 - 5.0 * zz),
                0.3731763325901154 * z * (5.0 * zz - 3.0),
                0.45704579946446572 * x * (1.0 - 5.0 * zz),
                1.4453057213202769 * z * (xx - yy),
                0.59004358992664352 * x * (-xx + 3.0 * yy),
            ]
        return torch.stack(features, dim=-1)


class IdentityEncoding(nn.Module):
    """Identity encoding of unit directions, as in FINER's NeRF network.

    Shape:
        - Input: ``(..., 3)`` unit directions.
        - Output: ``(..., 3)``.
    """

    out_features: int = 3

    def forward(self, x: Tensor) -> Tensor:
        return x


DIRECTION_ENCODINGS: dict[str, type[nn.Module]] = {
    "sh": SphericalHarmonicsEncoding,
    "identity": IdentityEncoding,
}

COLOR_NETS: dict[str, type[nn.Module]] = {"linear": nn.Linear, "mlp": MLP}


class RadianceField(nn.Module):
    """Radiance field with separate density, color and direction modules.

    The density network maps points to one density logit followed by geometry features,
    and the color network maps geometry and direction features to RGB logits. This class
    applies the density activation and the RGB sigmoid, so neither network should have
    an output activation. Each module is a :data:`~inro.utils.ModuleSpec`: an instance,
    a registry key, a class or a ``(key_or_class, kwargs)`` pair.

    Args:
        density_net: Spatial network spec, without registry keys. A class receives
            ``in_features=3`` and ``out_features=1 + geometry_features``. Its inputs are
            coordinates in ``[-1, 1]``.
        color_net: Color network spec, with registry keys ``"linear"`` and ``"mlp"``,
            returning three RGB logits. A class receives the input width and
            ``out_features=3``.
        geometry_features: Number of features passed to the color network, excluding
            density; at least 1.
        direction_encoding: Direction encoding spec, exposing ``out_features``, with
            registry keys ``"sh"``, degree-4 spherical harmonics as in Instant-NGP, and
            ``"identity"``, raw directions as in FINER.
        aabb: Scene box ``(xmin, ymin, zmin, xmax, ymax, zmax)``, kept in the state
            dict. World points map affinely to ``[-1, 1]`` and clamp at its boundary.
        density_activation: Callable from density logits to nonnegative densities.
        density_bias: Offset added to the density logits before the activation.

    Shape:
        - Input: ``x`` and unit ``directions``, each ``(N, 3)``.
        - Output: ``(rgb, sigma)``, RGB ``(N, 3)`` in ``[0, 1]`` and density ``(N,)``.

    Examples:
        >>> import inro
        >>> field = RadianceField(
        ...     (inro.InstantNGP, {"log2_hashmap_size": 14}),
        ...     ("mlp", {"hidden_features": 64, "hidden_layers": 2}),
        ... )
        >>> directions = torch.tensor([0.0, 0.0, 1.0]).expand(8, 3)
        >>> rgb, sigma = field(torch.rand(8, 3), directions)
        >>> rgb.shape, sigma.shape
        (torch.Size([8, 3]), torch.Size([8]))
    """

    def __init__(
        self,
        density_net: ModuleSpec,
        color_net: ModuleSpec,
        *,
        geometry_features: int = 15,
        direction_encoding: ModuleSpec = "sh",
        aabb: Tensor | Sequence[float] = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        density_activation: Callable[[Tensor], Tensor] = trunc_exp,
        density_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if geometry_features < 1:
            raise ValueError(f"geometry_features must be >= 1, got {geometry_features}")
        self.geometry_features = geometry_features
        self.density_net = build_module(
            density_net, {}, in_features=3, out_features=1 + geometry_features
        )
        self.register_buffer("aabb", torch.as_tensor(aabb, dtype=torch.float32))
        self.density_activation = density_activation
        self.density_bias = density_bias

        self.direction_encoding = build_module(direction_encoding, DIRECTION_ENCODINGS)
        color_in_features = geometry_features + self.direction_encoding.out_features
        self.color_net = build_module(
            color_net, COLOR_NETS, in_features=color_in_features, out_features=3
        )
        with torch.no_grad():
            # Supplied module instances may already use CUDA or half precision.
            parameter = next(self.color_net.parameters(), None)
            probe = torch.zeros(
                1,
                color_in_features,
                device=None if parameter is None else parameter.device,
                dtype=None if parameter is None else parameter.dtype,
            )
            out_features = int(self.color_net(probe).shape[-1])
        if out_features != 3:
            raise ValueError(
                f"color_net must map ({color_in_features},) -> (3,) RGB logits, but it "
                f"returns {out_features} features"
            )

    def normalize_coordinates(self, x: Tensor) -> Tensor:
        """Map world coordinates from :attr:`aabb` to ``[-1, 1]``, clamping outliers.

        Args:
            x: World coordinates ``(..., 3)``.
        """
        box_min, box_max = self.aabb[:3], self.aabb[3:]
        return (2.0 * (x - box_min) / (box_max - box_min) - 1.0).clamp(-1.0, 1.0)

    def _density_and_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return the densities and geometry features at world coordinates."""
        output = self.density_net(self.normalize_coordinates(x))
        sigma = self.density_activation(output[..., 0] + self.density_bias)
        return sigma, output[..., 1:]

    def query_density(self, x: Tensor) -> Tensor:
        """Return densities ``(N,)``, skipping the color network for occupancy updates.

        Args:
            x: World coordinates ``(N, 3)``; the batch may be empty.
        """
        if x.shape[0] == 0:
            return x.new_zeros((0,))
        return self._density_and_features(x)[0]

    def forward(self, x: Tensor, directions: Tensor) -> tuple[Tensor, Tensor]:
        """Return RGB and density at world points seen along unit directions.

        Args:
            x: World coordinates ``(N, 3)``; the batch may be empty.
            directions: Unit view directions ``(N, 3)``.
        """
        if x.shape[0] == 0:
            return x.new_zeros((0, 3)), x.new_zeros((0,))
        sigma, features = self._density_and_features(x)
        direction_features = self.direction_encoding(directions)
        rgb = torch.sigmoid(
            self.color_net(torch.cat([features, direction_features], dim=-1))
        )
        return rgb, sigma
