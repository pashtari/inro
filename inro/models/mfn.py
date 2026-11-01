"""MFN: multiplicative filter network with Gabor filters."""

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

__all__ = ["GaborFilter", "MFN"]


class GaborFilter(nn.Module):
    r"""Gabor filter ``sin(Wx + b) * exp(-gamma * ||x - mu||**2 / 2)`` of :class:`MFN`.

    The centers ``mu`` start uniformly in ``[-1, 1]`` and the inverse squared widths
    ``gamma`` from a Gamma distribution, both learnable. The weights ``W`` are scaled by
    ``input_scale * sqrt(gamma)`` and the biases ``b`` start in :math:`[-\pi, \pi]`.

    Args:
        in_features: Number of coordinate axes ``C``.
        out_features: Number of filters.
        input_scale: Frequency scale.
        alpha: Gamma shape.
        beta: Gamma rate.

    Shape:
        - Input: ``(..., C)``, with at least one leading dimension.
        - Output: ``(..., out_features)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        input_scale: float,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.Gamma(alpha, beta).sample((out_features,))
        )
        with torch.no_grad():
            self.linear.weight *= input_scale * self.gamma[:, None].sqrt()
            self.linear.bias.uniform_(-math.pi, math.pi)

    def forward(self, x: Tensor) -> Tensor:
        squared_distance = (
            (x**2).sum(-1)[..., None]
            + (self.mu**2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.linear(x)) * torch.exp(
            -0.5 * squared_distance * self.gamma[None, :]
        )


class MFN(nn.Module):
    """Multiplicative Gabor filter network.

    Each hidden layer multiplies a :class:`GaborFilter` of the raw coordinates by a
    linear transform of the previous hidden state; the first takes its filter alone. A
    linear layer gives the output.

    Args:
        in_features: Number of coordinate axes ``C``.
        out_features: Number of output channels ``D``.
        hidden_features: Hidden width.
        hidden_layers: Number of hidden layers, one filter each.
        input_scale: Frequency scale, divided by ``sqrt(hidden_layers)`` per filter.
        weight_scale: Scale of the hidden linear weights, drawn uniformly within
            ``sqrt(weight_scale / hidden_features)`` of zero.
        alpha: Gamma shape, divided by ``hidden_layers`` per filter.
        beta: Gamma rate.
        output_activation: Callable applied to the output; ``None`` uses the identity.

    Shape:
        - Input: ``(..., C)`` coordinates in ``[-1, 1]``, with at least one leading
          dimension.
        - Output: ``(..., D)``.

    References:
        Fathony et al., "Multiplicative Filter Networks", ICLR 2021.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int = 4,
        input_scale: float = 256.0,
        weight_scale: float = 1.0,
        alpha: float = 6.0,
        beta: float = 1.0,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError(f"hidden_layers must be >= 1, got {hidden_layers}")

        self.filters = nn.ModuleList(
            [
                GaborFilter(
                    in_features,
                    hidden_features,
                    input_scale / math.sqrt(hidden_layers),
                    alpha / hidden_layers,
                    beta,
                )
                for _ in range(hidden_layers)
            ]
        )
        self.linears = nn.ModuleList(
            [
                nn.Linear(hidden_features, hidden_features)
                for _ in range(hidden_layers - 1)
            ]
        )
        self.output_linear = nn.Linear(hidden_features, out_features)
        self.output_activation = (
            output_activation if output_activation is not None else nn.Identity()
        )

        bound = math.sqrt(weight_scale / hidden_features)
        with torch.no_grad():
            for linear in self.linears:
                linear.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        features = self.filters[0](x)
        for gabor, linear in zip(self.filters[1:], self.linears):
            features = gabor(x) * linear(features)
        return self.output_activation(self.output_linear(features))
