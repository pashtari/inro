"""RFF: ReLU MLP with a random Fourier feature input encoding."""

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

from .mlp import MLP, ReLULayer

__all__ = ["RFFEncoding", "RFF"]


class RFFEncoding(nn.Module):
    r"""Random Fourier feature encoding of :class:`RFF`.

    Computes :math:`\sin(2\pi \boldsymbol{B}\boldsymbol{x})`, then
    :math:`\cos(2\pi \boldsymbol{B}\boldsymbol{x})`. The fixed buffer ``B`` of shape
    ``(num_frequencies, C)`` has its entries drawn from
    :math:`\mathcal{N}(0, \sigma^2)` at construction.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_frequencies: Number of random frequency vectors.
        sigma: Standard deviation of the frequencies.

    Shape:
        - Input: ``(..., C)``.
        - Output: ``(..., 2 * num_frequencies)``.
    """

    def __init__(
        self, in_features: int, num_frequencies: int = 256, sigma: float = 10.0
    ) -> None:
        super().__init__()

        frequencies = torch.randn(num_frequencies, in_features) * sigma
        self.register_buffer("B", frequencies)

        self.out_features = 2 * num_frequencies

    def forward(self, x: Tensor) -> Tensor:
        projection = 2 * math.pi * x @ self.B.T
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


class RFF(nn.Module):
    """ReLU MLP with random Fourier features.

    Args:
        in_features: Number of coordinate axes ``C``.
        out_features: Number of output channels ``D``.
        hidden_features: Hidden width.
        hidden_layers: Number of hidden layers.
        num_frequencies: Number of random frequency vectors.
        sigma: Standard deviation of the frequencies.
        output_activation: Callable applied to the output; ``None`` uses the identity.

    Shape:
        - Input: ``(..., C)`` coordinates in ``[-1, 1]``.
        - Output: ``(..., D)``.

    References:
        Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in
        Low Dimensional Domains", NeurIPS 2020.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int = 3,
        num_frequencies: int = 256,
        sigma: float = 10.0,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()

        self.encoding = RFFEncoding(in_features, num_frequencies, sigma)
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
