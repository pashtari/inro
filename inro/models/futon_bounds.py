"""Bounds on how well a FUTON with a linear head can fit a sampled signal.

On a signal's sampling grid, the error of a FUTON with a CP combiner and a linear
head, both without bias, splits exactly in two: the error of the best approximation
in the basis's span, which no rank removes, and the CP misfit of the signal's
coefficients in an orthonormal basis of that span. :func:`futon_bound` computes the
first exactly, brackets the best second, and returns the factors of a FUTON that attains
the upper bound. The documentation's error bound page states and proves the theorem.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, permutations

import torch
import torch.nn.functional as F
from torch import Tensor

from .futon import FUTON
from .futon_inits import _als, _cp_parts, _expansion, _project, _residual

__all__ = ["FUTONBound", "futon_bound"]


@dataclass
class FUTONBound:
    r"""Bounds on the best mean squared error of a FUTON with a linear head.

    Errors are float64 means over the grid's points and the signal's channels, as
    :func:`inro.train` reports them.

    Attributes:
        ranks: CP ranks, ascending.
        truncation: Error of the best approximation in the basis's span, which no rank
            goes below; exact.
        lower: Per rank, a lower bound on the error of every FUTON of that rank; exact
            for a matrix of coefficients, that is, one axis, or two and one channel.
        upper: Per rank, the error of the FUTON in ``factors``, and so an upper bound on
            the best one's.
        factors: Per rank, the factors :math:`\boldsymbol{U}^{(c)}`, ``(K_c, R)``, and
            the head :math:`\boldsymbol{V}`, ``(D, R)``, that attain ``upper``.
    """

    ranks: list[int]
    truncation: float
    lower: Tensor
    upper: Tensor
    factors: list[tuple[list[Tensor], Tensor]]

    @torch.no_grad()
    def initialize(self, model: FUTON) -> None:
        """Give ``model`` the factors and the head that attain ``upper`` at its rank.

        The model's error on the bound's grid is then the bound, up to its precision.

        Args:
            model: FUTON with the bound's basis, a CP combiner and a linear head
                without bias.

        Raises:
            ValueError: If ``model`` has another combiner or head, or a rank without
                a bound.
        """
        factors, output, _ = _cp_parts(model)
        bound_factors, bound_output = self.factors[self.ranks.index(output.shape[1])]
        for parameter, factor in zip(
            [*factors, output], [*bound_factors, bound_output]
        ):
            parameter.copy_(factor)


def _tails(tensor: Tensor, ranks: Sequence[int]) -> Tensor:
    """Return, per rank, the most energy beyond that rank in any matricization.

    A CP tensor of rank ``R`` has matricizations of rank at most ``R``, so by Eckart and
    Young none comes closer to ``tensor`` than the energy of the singular values a
    matricization has beyond the ``R``-th.
    """
    best = tensor.new_zeros(len(ranks))
    modes = range(tensor.ndim)
    for size in range(1, tensor.ndim // 2 + 1):
        for rows in combinations(modes, size):
            if 2 * size == tensor.ndim and 0 not in rows:  # each split once
                continue
            columns = [mode for mode in modes if mode not in rows]
            matrix = tensor.permute(*rows, *columns).flatten(0, size - 1).flatten(1)
            if matrix.shape[0] > matrix.shape[1]:
                matrix = matrix.T
            energies = torch.linalg.eigvalsh(matrix @ matrix.T).flip(0).clamp_min(0)
            tails = torch.stack([energies[rank:].sum() for rank in ranks])
            best = torch.maximum(best, tails)
    return best


@torch.no_grad()
def futon_bound(
    signal: Tensor,
    features: Sequence[Tensor],
    ranks: int | Sequence[int],
    sweeps: int = 0,
    tol: float = 3e-5,
) -> FUTONBound:
    """Bound the best grid error of a FUTON with a linear head on a signal.

    Every FUTON with a CP combiner and a linear head, both without bias, with these
    per-axis features and rank ``R`` has a mean squared error on the grid of at least
    the bound's ``lower`` at ``R``, and the one given by its ``factors`` has exactly its
    ``upper``; no FUTON goes below ``truncation``. The bounds depend on the basis only
    through each axis's span.

    Args:
        signal: Samples ``(N_1, ..., N_C, D)`` on a grid.
        features: Per axis ``c``, the basis ``(N_c, K_c)`` at that axis's grid
            coordinates, orthonormal or not. For a FUTON's basis on a coordinate
            dataset's grid, it is
            ``basis.axis_features(torch.linspace(-1, 1, N_c), c)``.
        ranks: CP ranks ``R`` to bound.
        sweeps: Maximum number of alternating least-squares sweeps that refine each
            upper bound, in float32, from its SVD form, the ``R`` heaviest terms of the
            nested SVDs. The default, ``0``, returns the SVD form; a sweep never
            raises the bound.
        tol: Fraction of the bound below which a sweep's decrease stops the sweeps,
            when there are any.

    Returns:
        The bounds and the factors that attain the upper ones, on the CPU.

    Raises:
        ValueError: If ``features`` is not one matrix of ``N_c`` rows per axis.

    Examples:
        >>> from inro import SincBasis
        >>> signal = torch.rand(32, 32, 3) * 2 - 1
        >>> basis = SincBasis(2, num_components=16)
        >>> grid = torch.linspace(-1, 1, 32)
        >>> features = [basis.axis_features(grid, c) for c in range(2)]
        >>> bound = futon_bound(signal, features, ranks=[4, 8])
        >>> psnr = 10 * torch.log10(4 / bound.upper)  # guaranteed, in dB
        >>> psnr.shape
        torch.Size([2])
    """
    ranks = sorted({ranks} if isinstance(ranks, int) else set(ranks))
    signal = signal.double()
    *sizes, channels = signal.shape
    if len(features) != len(sizes) or any(
        len(phi) != size for phi, size in zip(features, sizes)
    ):
        raise ValueError(
            f"expected one feature matrix of N_c rows per axis of {tuple(sizes)}"
        )

    # Phi_c / sqrt(N_c) = Q_c B_c with Q_c an orthonormal basis of its span;
    # directions the grid cannot tell from zero are left out. The small
    # decompositions here and below run on the CPU, where they are fastest.
    bases, inverses = [], []
    for phi in features:
        phi = phi.double().cpu() / math.sqrt(len(phi))
        q, s, vh = torch.linalg.svd(phi, full_matrices=False)
        keep = s > s[0] * max(phi.shape) * torch.finfo(s.dtype).eps
        bases.append(q[:, keep].to(signal))
        inverses.append(vh[keep].T / s[keep])  # B_c^+, (K_c, K_c')

    # The signal's coefficients in the orthonormal basis of the span.
    coefficients = _project(signal, bases) / math.sqrt(math.prod(sizes))
    energy = float(coefficients.square().sum())
    truncation = max(float(signal.square().mean()) * channels - energy, 0.0)
    small = coefficients.cpu()
    lower = _tails(small, ranks)

    # The upper bounds' SVD form: per rank, the R heaviest terms of the nested SVDs
    # along the mode order that keeps the most energy in them. Modes of size one are
    # peeled first, at no cost, and the last two of an order share one SVD.
    single = [mode for mode in range(small.ndim) if small.shape[mode] == 1]
    rest = [mode for mode in range(small.ndim) if small.shape[mode] > 1]
    orders = [
        single + list(order)
        for order in permutations(rest)
        if len(order) < 2 or order[-2] < order[-1]
    ]
    expansions = []
    for order in orders:
        weights, vectors = _expansion(small, order)
        weights, index = weights.sort(descending=True)
        expansions.append((weights, [vector[index] for vector in vectors]))

    # The search runs in float32 on the signal's device; each bound is the
    # exact float64 error of the factors it finds.
    working = coefficients.float()
    upper, factors = [], []
    for rank in ranks:
        kept = [float(weights[:rank].square().sum()) for weights, _ in expansions]
        weights, vectors = expansions[kept.index(max(kept))]
        start = [vector[:rank].T.to(working) for vector in vectors]
        start[0] = start[0] * weights[:rank].to(working)
        chosen = [f.double() for f in _als(working, start, sweeps, tol, truncation)]
        residual = _residual(coefficients, chosen)
        # A rank's FUTON is one of every larger rank, with zero components added.
        if upper and upper[-1] < residual:
            residual, chosen = upper[-1], factors[-1]
        upper.append(residual)
        factors.append([F.pad(f, (0, rank - f.shape[1])) for f in chosen])

    # Back to the basis: U^(c) = B_c^+ U~^(c), with the head unchanged.
    mapped = [
        ([inverse @ f.cpu() for inverse, f in zip(inverses, chosen)], chosen[-1].cpu())
        for chosen in factors
    ]
    to_mse = 1 / channels
    return FUTONBound(
        ranks=ranks,
        truncation=truncation * to_mse,
        lower=(truncation + lower) * to_mse,
        upper=(truncation + torch.tensor(upper, dtype=torch.float64)) * to_mse,
        factors=mapped,
    )
