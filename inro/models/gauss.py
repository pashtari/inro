"""Gauss: coordinate MLP with Gaussian activations."""

from collections.abc import Callable

import torch
from torch import Tensor, nn

from .mlp import MLP

__all__ = ["GaussLayer", "Gauss"]


class GaussLayer(nn.Module):
    """Gaussian layer ``exp(-(scale * linear(x))**2)`` of :class:`Gauss`.

    Args:
        in_features: Input width.
        out_features: Output width.
        scale: Inverse Gaussian width.

    Shape:
        - Input: ``(..., in_features)``.
        - Output: ``(..., out_features)``.
    """

    def __init__(
        self, in_features: int, out_features: int, scale: float = 10.0
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.scale = scale

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(-((self.scale * self.linear(x)) ** 2))


class Gauss(MLP):
    """Gaussian coordinate MLP.

    An :class:`MLP` with :class:`GaussLayer` hidden layers, a linear output layer and
    PyTorch's default initialization.

    Args:
        in_features: Number of coordinate axes ``C``.
        out_features: Number of output channels ``D``.
        hidden_features: Hidden width.
        hidden_layers: Number of hidden layers.
        scale: Inverse Gaussian width.
        output_activation: Callable applied to the output; ``None`` uses the identity.

    Shape:
        - Input: ``(..., C)`` coordinates in ``[-1, 1]``.
        - Output: ``(..., D)``.

    References:
        Ramasinghe and Lucey, "Beyond Periodicity: Towards a Unifying Framework for
        Activations in Coordinate-MLPs", ECCV 2022.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int = 3,
        scale: float = 10.0,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            hidden_features,
            hidden_layers,
            layer_class=GaussLayer,
            scale=scale,
            output_activation=output_activation,
        )
