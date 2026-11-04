"""F-INR: a functional tensor decomposition, one SIREN per input axis."""

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

from .siren import SineLayer

__all__ = ["FINR"]


class FINR(nn.Module):
    r"""Functional tensor decomposition with one SIREN per coordinate axis.

    Each axis's network maps its coordinate alone to factors, and a tensor format
    combines the axes' factors per output channel, with no decoder after it:

    - ``"cp"``: :math:`y_d = \sum_r \prod_c a_{c,dr}`, for any ``C``.
    - ``"tt"``: the tensor train
      :math:`y_d = \boldsymbol{u}_d^\top \boldsymbol{G}_d \boldsymbol{v}_d`, for
      ``C = 3``, whose ``R x R`` core comes from the first axis's network, as in the
      authors' code.

    Each network has ``hidden_layers`` sine layers and a linear head.

    Args:
        in_features: Number of coordinate axes ``C``.
        out_features: Number of output channels ``D``.
        rank: Rank ``R`` of the decomposition.
        hidden_features: Hidden width.
        hidden_layers: Number of hidden layers.
        mode: Tensor format, ``"cp"``, or ``"tt"`` for ``C = 3``.
        first_omega: Frequency multiplier of the first sine layer.
        hidden_omega: Frequency multiplier of the later sine layers.
        positional_encoding: Size ``n`` of the authors' ``pos_enc``, which replaces each
            network's raw coordinate by ``sin(2^k x)`` and then ``cos(2^k x)`` for ``k``
            in ``range(int(-(n - 1) / 2), int((n + 1) / 2))``, so ``n = 10`` gives the
            nine bands ``2^-4, ..., 2^4``. ``0`` keeps the raw coordinate.
        grid: Whether to run each network once per distinct coordinate, when at most
            half of a batch's are distinct, and gather the factors per point; the output
            is the same. Turn it off for scattered points, such as a radiance field's
            samples, where finding the distinct ones only costs time.
        output_activation: Callable applied to the output; ``None`` uses the identity.

    Shape:
        - Input: ``(..., C)`` coordinates in ``[-1, 1]``.
        - Output: ``(..., D)``.

    References:
        Vemuri et al., "F-INR: Functional Tensor Decomposition for Implicit Neural
        Representations", WACV 2026.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        hidden_features: int = 256,
        hidden_layers: int = 3,
        mode: str = "cp",
        first_omega: float = 100.0,
        hidden_omega: float = 30.0,
        positional_encoding: int = 0,
        grid: bool = True,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("cp", "tt"):
            raise ValueError(f"mode must be 'cp' or 'tt', got {mode!r}")
        if mode == "tt" and in_features != 3:
            raise ValueError("the tensor train combines exactly three inputs")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least one")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.mode = mode
        self.hidden_omega = hidden_omega
        self.grid = grid
        self.output_activation = output_activation

        bands = torch.empty(0)
        if positional_encoding > 0:
            low = int(-(positional_encoding - 1) / 2)  # int rounds toward zero
            bands = 2.0 ** torch.arange(low, int((positional_encoding + 1) / 2))
        self.register_buffer("bands", bands, persistent=False)
        encoded = 2 * len(bands) if len(bands) else 1

        sizes = [out_features * rank] * in_features
        if mode == "tt":
            sizes[0] = out_features * rank * rank
        self.axes = nn.ModuleList(
            nn.Sequential(
                SineLayer(encoded, hidden_features, first_omega),
                *(
                    SineLayer(hidden_features, hidden_features, hidden_omega)
                    for _ in range(hidden_layers - 1)
                ),
                nn.Linear(hidden_features, size),
            )
            for size in sizes
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize as the authors' image code does, biases at zero.

        Weights are drawn from ``U(-1/n, 1/n)`` in the first layer and from
        ``U(-sqrt(6/n) / hidden_omega, sqrt(6/n) / hidden_omega)`` after it, for fan-in
        ``n``.
        """
        for network in self.axes:
            for index, layer in enumerate(network):
                linear = layer.linear if isinstance(layer, SineLayer) else layer
                fan_in = linear.in_features
                bound = (
                    1 / fan_in
                    if index == 0
                    else math.sqrt(6 / fan_in) / self.hidden_omega
                )
                linear.weight.uniform_(-bound, bound)
                linear.bias.zero_()

    def _encode(self, coordinate: Tensor) -> Tensor:
        """Return a coordinate ``(N,)`` as a network's input ``(N, features)``."""
        if not len(self.bands):
            return coordinate[:, None]
        angles = coordinate[:, None] * self.bands
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def _factors(self, network: nn.Module, coordinate: Tensor) -> Tensor:
        """Run ``network`` on each point, once per distinct coordinate on a grid."""
        if self.grid and not coordinate.requires_grad:
            values, inverse = torch.unique(coordinate, return_inverse=True)
            if 2 * len(values) <= len(coordinate):
                return network(self._encode(values))[inverse]
        return network(self._encode(coordinate))

    def forward(self, x: Tensor) -> Tensor:
        points = x.reshape(-1, self.in_features)
        count, channels, rank = len(points), self.out_features, self.rank
        factors = [
            self._factors(network, points[:, axis])
            for axis, network in enumerate(self.axes)
        ]
        if self.mode == "cp":
            out = factors[0].view(count, channels, rank)
            for factor in factors[1:]:
                out = out * factor.view(count, channels, rank)
            out = out.sum(-1)
        else:
            core = factors[0].view(count, channels, rank, rank)
            left = factors[1].view(count, channels, rank)
            right = factors[2].view(count, channels, rank)
            out = torch.einsum("...pq,...p,...q->...", core, left, right)
        out = out.reshape(*x.shape[:-1], channels)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out
