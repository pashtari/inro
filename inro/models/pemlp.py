"""PE-MLP: ReLU MLP with a NeRF-style positional encoding."""

from collections.abc import Callable

import torch
from torch import Tensor, nn

from .mlp import MLP, ReLULayer

__all__ = ["PositionalEncoding", "PEMLP"]


class PositionalEncoding(nn.Module):
    r"""Positional encoding ``[x, sin(2**k * x), cos(2**k * x)]`` of :class:`PEMLP`.

    The bands are ``k = 0, ..., num_frequencies - 1``, without the :math:`\pi` of NeRF's
    Eq. (4), and the buffer ``freqs`` holds ``2**k``. The input comes first, then all
    sines, then all cosines; within each block, bands vary faster than coordinates.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_frequencies: Number of frequency bands.

    Shape:
        - Input: ``(..., C)``.
        - Output: ``(..., C * (2 * num_frequencies + 1))``.
    """

    def __init__(self, in_features: int, num_frequencies: int = 10) -> None:
        super().__init__()

        freqs = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freqs", freqs)

        self.out_features = in_features * (2 * num_frequencies + 1)

    def forward(self, x: Tensor) -> Tensor:
        projection = x.unsqueeze(-1) * self.freqs
        return torch.cat(
            [x, projection.sin().flatten(-2), projection.cos().flatten(-2)], dim=-1
        )


class PEMLP(nn.Module):
    """ReLU MLP with a NeRF-style positional encoding.

    Args:
        in_features: Number of coordinate axes ``C``.
        out_features: Number of output channels ``D``.
        hidden_features: Hidden width.
        hidden_layers: Number of hidden layers.
        num_frequencies: Number of frequency bands.
        output_activation: Callable applied to the output; ``None`` uses the identity.

    Shape:
        - Input: ``(..., C)`` coordinates in ``[-1, 1]``.
        - Output: ``(..., D)``.

    References:
        Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View
        Synthesis", ECCV 2020.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int = 3,
        num_frequencies: int = 10,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()

        self.encoding = PositionalEncoding(in_features, num_frequencies)
        self.mlp = MLP(
            self.encoding.out_features,
            out_features,
            hidden_features,
            hidden_layers,
            layer_class=ReLULayer,
            output_activation=output_activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self.encoding(x))
