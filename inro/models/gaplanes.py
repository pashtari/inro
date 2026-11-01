"""GA-Planes: geometric algebra planes, feature grids with an MLP decoder."""

from collections.abc import Callable, Sequence

from torch import Tensor, nn

from .mlp import MLP
from .multivector import MultiVector

__all__ = ["GAPlanes"]


class GAPlanes(nn.Module):
    """Line, plane and volume feature grids with a ReLU MLP decoder.

    The default :class:`MultiVector` expression combines a low-rank term with
    lower-resolution grids: in 2D it concatenates the product of two line grids and a
    plane; in 3D it also includes line-plane products and a volume.

    Args:
        in_features: Number of coordinate axes ``C``, ``2`` or ``3``.
        out_features: Number of output channels ``D``.
        expr: :class:`MultiVector` expression; ``None`` selects all grids.
        features: Feature channels per grid: an int for all grids, or
            ``(line, plane, volume)`` values, a short sequence repeating its last value.
            Operands of ``o`` and ``+`` must have equal channel counts.
        resolution: Resolution per axis per grid, given as ``features`` is. Planes are
            square and volumes are cubic.
        hidden_features: Hidden width.
        hidden_layers: Number of hidden layers.
        output_activation: Callable applied to the output; ``None`` uses the identity.

    Shape:
        - Input: ``(..., C)`` coordinates in ``[-1, 1]``.
        - Output: ``(..., D)``.

    References:
        Sivgin et al., "Geometric Algebra Planes: Convex Implicit Neural Volumes",
        ICML 2025.
    """

    _DEFAULT_EXPR: dict[int, str] = {
        2: "[e1 o e2, e12]",
        3: "[e1 o e2 o e3, e1 o e23, e2 o e13, e3 o e12, e123]",
    }

    def __init__(
        self,
        in_features: int,
        out_features: int,
        expr: str | None = None,
        features: int | Sequence[int] = 32,
        resolution: int | Sequence[int] = (256, 64, 16),
        hidden_features: int = 64,
        hidden_layers: int = 2,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        if in_features not in (2, 3):
            raise ValueError(f"in_features must be 2 or 3, got {in_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.output_activation = (
            output_activation if output_activation is not None else nn.Identity()
        )

        self.mv = MultiVector(
            in_features=in_features,
            expr=self._DEFAULT_EXPR[in_features] if expr is None else expr,
            features=features,
            resolution=resolution,
        )
        self.decoder = MLP(
            self.mv.out_features, out_features, hidden_features, hidden_layers
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.mv(x.reshape(-1, self.in_features))
        out = self.output_activation(self.decoder(features))
        return out.reshape(*x.shape[:-1], self.out_features)
