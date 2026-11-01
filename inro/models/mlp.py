"""Multi-layer perceptrons with standard or custom hidden layers."""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

__all__ = ["MLP", "ReLULayer"]


class ReLULayer(nn.Module):
    """Linear projection followed by ReLU.

    Args:
        in_features: Input width.
        out_features: Output width.

    Shape:
        - Input: ``(..., in_features)``.
        - Output: ``(..., out_features)``.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(self.linear(x))


class MLP(nn.Module):
    """MLP with standard or custom hidden layers and a linear output layer.

    A standard hidden layer is a linear projection followed by ``activation``; a custom
    one is an instance of ``layer_class``. Subclasses apply custom initialization after
    ``super().__init__``.

    Args:
        in_features: Input width.
        out_features: Output width.
        hidden_features: Hidden width; ``None`` uses ``in_features``.
        hidden_layers: Number of hidden layers.
        activation: Callable applied after each standard hidden layer; ``None`` uses
            ReLU. A module instance is shared across the layers.
        layer_class: Class of the custom hidden layers, built as
            ``layer_class(in_width, out_width, **layer_kwargs)``. No other activation
            is applied.
        output_activation: Callable applied to the output; ``None`` uses the identity.
        **layer_kwargs: Constructor arguments of the hidden layers only.

    Raises:
        ValueError: If ``hidden_layers < 1``, or if ``activation`` and ``layer_class``
            are both given.

    Shape:
        - Input: ``(..., in_features)``.
        - Output: ``(..., out_features)``.

    Examples:
        >>> from inro import SineLayer
        >>> model = MLP(2, 3, hidden_features=64, layer_class=SineLayer, omega=30.0)
        >>> model(torch.rand(8, 2)).shape
        torch.Size([8, 3])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int | None = None,
        hidden_layers: int = 2,
        activation: Callable[[Tensor], Tensor] | None = None,
        *,
        layer_class: type[nn.Module] | None = None,
        output_activation: Callable[[Tensor], Tensor] | None = None,
        **layer_kwargs: Any,
    ) -> None:
        super().__init__()

        if hidden_layers < 1:
            raise ValueError(f"hidden_layers must be >= 1, got {hidden_layers}")
        if layer_class is None:
            layer_class = nn.Linear
            activation = torch.relu if activation is None else activation
        elif activation is not None:
            raise ValueError("Custom layer_class supplies its own activation.")

        if hidden_features is None:
            hidden_features = in_features

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.activation = activation if activation is not None else nn.Identity()

        layers = [layer_class(in_features, hidden_features, **layer_kwargs)]
        for _ in range(hidden_layers - 1):
            layers.append(layer_class(hidden_features, hidden_features, **layer_kwargs))
        layers.append(nn.Linear(hidden_features, out_features))
        self.layers = nn.ModuleList(layers)

        self.output_activation = (
            output_activation if output_activation is not None else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.output_activation(self.layers[-1](x))
