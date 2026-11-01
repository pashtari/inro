"""SIREN: coordinate MLP with sine activations."""

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

from .mlp import MLP

__all__ = ["SineLayer", "SIREN"]


class SineLayer(nn.Module):
    """Sine layer ``sin(omega * linear(x))`` of :class:`SIREN`.

    Args:
        in_features: Input width.
        out_features: Output width.
        omega: Frequency multiplier.

    Shape:
        - Input: ``(..., in_features)``.
        - Output: ``(..., out_features)``.
    """

    def __init__(
        self, in_features: int, out_features: int, omega: float = 30.0
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.omega = omega

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega * self.linear(x))


class SIREN(MLP):
    """Sinusoidal representation network.

    An :class:`MLP` with :class:`SineLayer` hidden layers, a linear output layer and
    SIREN weight initialization. Biases keep PyTorch's defaults.

    Args:
        in_features: Number of coordinate axes ``C``.
        out_features: Number of output channels ``D``.
        hidden_features: Hidden width.
        hidden_layers: Number of hidden layers.
        omega: Frequency multiplier.
        first_omega: Frequency multiplier of the first layer alone, the one
            :func:`inro.fresh_select` chooses; ``None`` uses ``omega``. The later
            layers' initialization follows ``omega``.
        output_activation: Callable applied to the output; ``None`` uses the identity.

    Shape:
        - Input: ``(..., C)`` coordinates in ``[-1, 1]``.
        - Output: ``(..., D)``.

    References:
        Sitzmann et al., "Implicit Neural Representations with Periodic Activation
        Functions", NeurIPS 2020.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int = 3,
        omega: float = 30.0,
        first_omega: float | None = None,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            hidden_features,
            hidden_layers,
            layer_class=SineLayer,
            omega=omega,
            output_activation=output_activation,
        )
        self.omega = omega
        if first_omega is not None:
            self.layers[0].omega = first_omega
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights with SIREN bounds, leaving biases unchanged."""
        _siren_init_(self.layers, self.omega)


@torch.no_grad()
def _siren_init_(layers: nn.ModuleList, omega: float) -> None:
    """Initialize SIREN weights in place (Sitzmann et al., Sec. 3.2).

    The first layer draws from ``U(-1/n, 1/n)`` and the later ones, the linear output
    included, from ``U(-sqrt(6/n)/omega, sqrt(6/n)/omega)``, for a layer's input width
    ``n``. Biases are left unchanged.
    """
    for i, layer in enumerate(layers):
        if i == 0:
            bound = 1 / layer.in_features
        else:
            bound = math.sqrt(6 / layer.in_features) / omega
        getattr(layer, "linear", layer).weight.uniform_(-bound, bound)
