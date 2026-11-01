"""WIRE: coordinate MLPs with Gabor wavelet activations (real and complex)."""

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

from .mlp import MLP

__all__ = ["RealGaborLayer", "ComplexGaborLayer", "RealWIRE", "WIRE"]


class RealGaborLayer(nn.Module):
    """Real Gabor layer ``cos(omega * z1) * exp(-(scale * z2)**2)``.

    A linear projection gives ``z1`` and ``z2``, with ``out_features`` channels each.

    Args:
        in_features: Input width.
        out_features: Output width.
        omega: Frequency multiplier.
        scale: Inverse Gaussian width.

    Shape:
        - Input: ``(..., in_features)``.
        - Output: ``(..., out_features)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega: float = 20.0,
        scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.omega = omega
        self.scale = scale

        self.linear = nn.Linear(in_features, 2 * out_features)

    def forward(self, x: Tensor) -> Tensor:
        projection = self.linear(x)
        frequency_projection, scale_projection = projection.chunk(2, dim=-1)
        return torch.cos(self.omega * frequency_projection) * torch.exp(
            -((self.scale * scale_projection) ** 2)
        )


class ComplexGaborLayer(nn.Module):
    """Complex Gabor layer ``exp(1j * omega * z - abs(scale * z)**2)``.

    Here ``z = linear(x)``.

    Args:
        in_features: Input width.
        out_features: Output width.
        omega: Frequency multiplier.
        scale: Inverse Gaussian width.
        is_first: Whether to use real weights, for real inputs, rather than complex.

    Shape:
        - Input: ``(..., in_features)``, real if ``is_first``, else complex.
        - Output: ``(..., out_features)``, complex.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega: float = 20.0,
        scale: float = 10.0,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.omega = omega
        self.scale = scale

        dtype = torch.float if is_first else torch.cfloat
        self.linear = nn.Linear(in_features, out_features, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        projection = self.linear(x)
        return torch.exp(
            1j * self.omega * projection - (self.scale * projection).abs().square()
        )


class RealWIRE(MLP):
    """Real-valued variant of :class:`WIRE` with :class:`RealGaborLayer` hidden layers.

    Uses a linear output layer and PyTorch's default initialization. Each hidden layer
    projects to twice the reduced width, to form its two wavelet terms.

    Args:
        in_features: Number of coordinate axes ``C``.
        out_features: Number of output channels ``D``.
        hidden_features: Hidden width, divided by ``sqrt(2)`` and rounded down.
        hidden_layers: Number of hidden layers.
        omega: Frequency multiplier.
        scale: Inverse Gaussian width.
        output_activation: Callable applied to the output; ``None`` uses the identity.

    Shape:
        - Input: ``(..., C)`` coordinates in ``[-1, 1]``.
        - Output: ``(..., D)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int = 3,
        omega: float = 20.0,
        scale: float = 10.0,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        hidden_features = int(hidden_features / math.sqrt(2))

        super().__init__(
            in_features,
            out_features,
            hidden_features,
            hidden_layers,
            layer_class=RealGaborLayer,
            omega=omega,
            scale=scale,
            output_activation=output_activation,
        )


class WIRE(nn.Module):
    """Wavelet implicit neural representation with complex Gabor layers.

    Uses :class:`ComplexGaborLayer` hidden layers, the first with real weights, and a
    complex linear output layer, all with PyTorch's default initialization. The real
    part of the output goes through ``output_activation``. Dividing the width keeps the
    parameter count comparable to real-valued networks.

    Args:
        in_features: Number of coordinate axes ``C``.
        out_features: Number of output channels ``D``.
        hidden_features: Hidden width, divided by ``sqrt(2)`` and rounded down.
        hidden_layers: Number of hidden layers.
        omega: Frequency multiplier.
        scale: Inverse Gaussian width.
        output_activation: Callable applied to the output; ``None`` uses the identity.

    Shape:
        - Input: ``(..., C)`` coordinates in ``[-1, 1]``, real.
        - Output: ``(..., D)``, real.

    References:
        Saragadam et al., "WIRE: Wavelet Implicit Neural Representations", CVPR 2023.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int = 3,
        omega: float = 20.0,
        scale: float = 10.0,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError(f"hidden_layers must be >= 1, got {hidden_layers}")

        # Account for the two real components of each complex parameter.
        hidden_features = int(hidden_features / math.sqrt(2))

        self.layers = nn.ModuleList()
        self.layers.append(
            ComplexGaborLayer(
                in_features, hidden_features, omega=omega, scale=scale, is_first=True
            )
        )
        for _ in range(hidden_layers - 1):
            self.layers.append(
                ComplexGaborLayer(
                    hidden_features, hidden_features, omega=omega, scale=scale
                )
            )
        self.layers.append(nn.Linear(hidden_features, out_features, dtype=torch.cfloat))

        self.output_activation = (
            output_activation if output_activation is not None else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.output_activation(x.real)
