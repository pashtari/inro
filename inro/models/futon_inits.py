"""Starts of FUTON fitted to a signal, and the CP algebra its bounds and solvers share.

On a grid, a CP FUTON, a CP combiner with a linear head, is a CP tensor: its factor of
axis ``c`` is the axis's basis times the factor matrix, and its factor of the channels
is the head's weight. A start sets
these blocks of a built FUTON from the signal, in place, before training:

- :class:`SVDInit` gives the ``R`` components the ``R`` heaviest rank-one terms of the
  signal's low frequencies, found by nested SVDs;
- :class:`ALSInit` refines those terms by alternating least squares first.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from typing import Any

import torch
from torch import Tensor, nn

from .futon import FUTON, CosineBasis, CPCombiner

__all__ = ["SVDInit", "ALSInit"]


# CP algebra ---------------------------------------------------------------------------


def _cp(factors: Sequence[Tensor]) -> Tensor:
    """Return the CP tensor ``(I_1, ..., I_M)`` of factor matrices ``(I_m, R)``."""
    r = len(factors)  # einsum subscripts, each after its operand: m for mode m, r for R
    operands = chain(*((factor, [m, r]) for m, factor in enumerate(factors)))
    return torch.einsum(*operands, list(range(r)))


def _mttkrp(tensor: Tensor, factors: Sequence[Tensor], mode: int) -> Tensor:
    """Return the matricized tensor times Khatri-Rao product (MTTKRP) along ``mode``.

    The tensor's unfolding along ``mode`` times the Khatri-Rao product of the other
    factor matrices, ``(I_mode, R)``: the right-hand side of the mode's least-squares
    problem in CP-ALS (Kolda and Bader, SIAM Review 2009).
    """
    r = tensor.ndim  # einsum subscripts, each after its operand: m for mode m, r for R
    others = chain(*((factor, [m, r]) for m, factor in enumerate(factors) if m != mode))
    return torch.einsum(tensor, list(range(r)), *others, [mode, r])


def _residual(tensor: Tensor, factors: Sequence[Tensor]) -> float:
    """Return ``||tensor - [[factors]]||_F^2``, with the CP tensor formed in full."""
    return float((tensor - _cp(factors)).square().sum())


def _ridged(gram: Tensor, ridge: float) -> Tensor:
    """Return a Gram matrix plus ``ridge`` times its mean diagonal on the diagonal."""
    eye = torch.eye(len(gram), dtype=gram.dtype, device=gram.device)
    return gram + ridge * gram.diagonal().mean() * eye


def _balanced(factors: Sequence[Tensor], scale: float = 1.0) -> list[Tensor]:
    """Return the factors balanced per component, their CP tensor times ``scale``."""
    tiny = torch.finfo(factors[0].dtype).tiny
    norms = torch.stack([factor.norm(dim=0) for factor in factors]).clamp_min(tiny)
    share = norms.log().mean(0).exp() * scale ** (1 / len(factors))
    return [factor / norm * share for factor, norm in zip(factors, norms)]


def _project(signal: Tensor, bases: Sequence[Tensor]) -> Tensor:
    """Return the coefficients of a signal in orthonormal bases of its axes."""
    for axis, basis in enumerate(bases):
        signal = torch.tensordot(signal, basis, dims=([axis], [0])).movedim(-1, axis)
    return signal


def _expansion(tensor: Tensor, order: Sequence[int]) -> tuple[Tensor, list[Tensor]]:
    """Return the orthonormal rank-one expansion of a tensor, by nested SVDs.

    The SVD of the unfolding along ``order[0]`` writes the tensor as a sum of orthogonal
    terms, each a unit vector of that mode times a unit tensor of the others. The SVDs
    along ``order[1]`` split those tensors in turn, all at once, and so on; the last two
    modes share one SVD.

    Returns:
        The terms' weights ``(L,)``, products of singular values, and per mode, in the
        tensor's order, their unit vectors ``(L, I_m)``.
    """
    weights, vectors = tensor.new_ones(1), []
    rest = tensor.permute(*order).reshape(1, -1)  # per term, its tensor of modes left
    for mode in order[:-1]:
        u, s, vh = torch.linalg.svd(
            rest.unflatten(1, (tensor.shape[mode], -1)), full_matrices=False
        )
        weights = (weights[:, None] * s).flatten()
        vectors = [v.repeat_interleave(s.shape[1], 0) for v in vectors]
        vectors.append(u.mT.flatten(0, 1))
        rest = vh.flatten(0, 1)
    vectors.append(rest)
    return weights, [vectors[order.index(mode)] for mode in range(tensor.ndim)]


def _svd_terms(tensor: Tensor, rank: int) -> list[Tensor]:
    """Return the CP factors ``(I_m, T)`` of a tensor's ``T <= rank`` heaviest terms.

    The nested SVDs split the shortest mode off first, which leaves the most terms; the
    terms' weights scale the first factor.
    """
    order = sorted(range(tensor.ndim), key=lambda mode: tensor.shape[mode])
    weights, vectors = _expansion(tensor, order)
    weights, index = weights.topk(min(rank, len(weights)))
    factors = [vector[index].T for vector in vectors]
    factors[0] = factors[0] * weights
    return factors


def _sweep(tensor: Tensor, factors: Sequence[Tensor]) -> list[Tensor]:
    """Run one sweep of alternating least squares, then balance the factors.

    Each factor in turn takes the exact solution of its least-squares problem with the
    others held, under a ridge of ``1e-12`` of its mean diagonal.
    """
    factors = list(factors)
    for mode in range(len(factors)):
        others = [factor for k, factor in enumerate(factors) if k != mode]
        gram = reduce(torch.mul, (factor.T @ factor for factor in others))
        right = _mttkrp(tensor, factors, mode)
        factors[mode] = torch.linalg.solve(_ridged(gram, 1e-12), right.T).T
    return _balanced(factors)


def _als(
    tensor: Tensor,
    factors: Sequence[Tensor],
    sweeps: int,
    tol: float = 3e-5,
    floor: float = 0.0,
) -> list[Tensor]:
    """Refine CP factors of a tensor by alternating least squares.

    After sweep ``k``, the step it took is extended by ``k^(1/3)`` times its length and
    kept if that lowers the misfit: Bro's line search, which shortens the slow stretches
    of ALS. The sweeps stop after ``sweeps``, or once one lowers the misfit by less than
    ``tol`` of ``floor`` plus the misfit.
    """
    factors = list(factors)
    if not sweeps:
        return factors
    error = _residual(tensor, factors)
    for k in range(1, sweeps + 1):
        update = _sweep(tensor, factors)
        trial = [new + k ** (1 / 3) * (new - old) for new, old in zip(update, factors)]
        candidates = [(_residual(tensor, c), c) for c in (update, trial)]
        # A NaN misfit, of factors that overflowed, sorts last and stops the sweeps.
        best_error, best = min(candidates, key=lambda c: (math.isnan(c[0]), c[0]))
        if not best_error <= error:  # rounding at convergence, or a NaN
            break
        settled = error - best_error < tol * (floor + best_error)
        error, factors = best_error, best
        if settled:
            break
    return factors


def _cp_parts(
    model: FUTON, allow_bias: bool = False
) -> tuple[nn.ParameterList, Tensor, Tensor | None]:
    """Return a CP FUTON's factor matrices, its output factor and its bias, if any: its
    linear head's weight and bias.

    Raises:
        ValueError: If ``model`` is not such a FUTON, or has a bias that ``allow_bias``
            forbids.
    """
    if isinstance(model, FUTON) and isinstance(model.encoding.combiner, CPCombiner):
        head = model.head
        if isinstance(head, nn.Linear) and (allow_bias or head.bias is None):
            return model.encoding.combiner.factors, head.weight, head.bias
    bias = "" if allow_bias else " without bias"
    raise ValueError(f"expected a FUTON with a CP combiner and a linear head{bias}")


# Starts -------------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class SVDInit:
    """Start a FUTON from the heaviest rank-one terms of a signal's low frequencies.

    Called on a built FUTON and a signal ``(N_1, ..., N_C, D)`` on the grid
    ``linspace(-1, 1, N_c)`` of each axis, such as a coordinate dataset's ``target``,
    it sets the model's parameters in place:

    1. Project the signal onto the lowest ``K_c // cut`` cosines of each axis ``c``,
       normalized per point as in :class:`~inro.models.futon.CosineBasis`.
    2. Expand the projection, a tensor of the ``C`` axes and the channels, into
       orthonormal rank-one terms by nested SVDs: the first splits off the shortest
       mode, and each next one the shortest mode left, which leaves the most terms.
    3. Give the ``R`` heaviest terms, as :meth:`refine` returns them, to the ``R``
       components: a term's vector of axis ``c``, fitted to the model's basis on the
       grid, becomes a column of factor ``c``, and its vector of the channels a column
       of the output factor, the head's weight.

    Each term's weight, times ``scale``, is shared evenly by its component's factors
    and output column. Components beyond the terms keep their draw behind a zero
    output column, silent until trained, and a head bias starts at zero. The start
    runs on the model's device, in its precision, and raises a ``ValueError`` for a
    model that is not a CP FUTON.

    Args:
        cut: Divisor of each axis's component count ``K_c`` that leaves the
            frequencies kept.
        scale: Factor of the output before the activation. With ``tanh``, ``2`` starts
            a signal of ``-1`` and ``1`` near ``-0.96`` and ``0.96``.

    Examples:
        >>> model = FUTON(
        ...     in_features=2,
        ...     out_features=3,
        ...     basis=("sinc", {"num_components": 32}),
        ...     combiner=("cp", {"rank": 16}),
        ...     head="linear",
        ... )
        >>> SVDInit(scale=2.0)(model, torch.rand(64, 64, 3) * 2 - 1)
    """

    cut: int = 4
    scale: float = 1.0

    @torch.no_grad()
    def __call__(self, model: FUTON, target: Tensor) -> None:
        blocks, output, bias = _cp_parts(model, allow_bias=True)
        basis: Any = model.encoding.basis  # any basis with axis_features
        signal = target.to(output)
        grids = [
            torch.linspace(-1, 1, n, dtype=signal.dtype, device=signal.device)
            for n in signal.shape[:-1]
        ]

        # 1. An orthonormal basis of each axis's lowest cosines, and the signal's
        # coefficients in them.
        cosine = CosineBasis(len(grids), basis.num_components)
        lows = [
            torch.linalg.qr(cosine.axis_features(grid, c)[:, : max(1, k // self.cut)]).Q
            for c, (grid, k) in enumerate(zip(grids, cosine.num_components))
        ]
        coefficients = _project(signal, lows)

        # 2. The projection's R heaviest terms, refined: a factor per axis, then the
        # channels'.
        terms = _svd_terms(coefficients, output.shape[1])
        *factors, channels = self.refine(coefficients, terms)

        # 3. Each axis's factor on the grid, fitted to the model's basis, and every
        # component's scale shared evenly by its blocks.
        parts = [
            torch.linalg.lstsq(basis.axis_features(grid, c), low @ factor).solution
            for c, (grid, low, factor) in enumerate(zip(grids, lows, factors))
        ]
        filled = channels.shape[1]
        for block, part in zip(
            [*blocks, output], _balanced([*parts, channels], self.scale)
        ):
            block[:, :filled] = part
        output[:, filled:] = 0  # components left: their draw, silent at first
        if bias is not None:
            bias.zero_()

    def refine(self, coefficients: Tensor, factors: list[Tensor]) -> list[Tensor]:
        """Return the terms' CP factors as the start loads them: here, as they are.

        Args:
            coefficients: The projection ``(K_1 // cut, ..., K_C // cut, D)``.
            factors: CP factors ``(I_m, T)`` of its ``T`` heaviest terms, a factor per
                axis and then the channels', the terms' weights in the first.
        """
        return factors


@dataclass(frozen=True, kw_only=True)
class ALSInit(SVDInit):
    """Start a FUTON as :class:`SVDInit` does, its terms refined by ALS first.

    Each sweep of alternating least squares solves every factor's least-squares problem
    on the projection exactly, the channels' included, the others held, as
    :func:`~inro.models.futon_bounds.futon_bound` refines its upper bound, so the
    misfit never rises. A sweep works on the projection, not on the samples.

    Args:
        sweeps: Most sweeps; they stop sooner once one lowers the misfit by less than
            ``3e-5`` of it.
        cut: See :class:`SVDInit`.
        scale: See :class:`SVDInit`.
    """

    sweeps: int = 10

    def refine(self, coefficients: Tensor, factors: list[Tensor]) -> list[Tensor]:
        """Return the factors after at most ``sweeps`` sweeps of ALS on the projection."""
        return _als(coefficients, factors, self.sweeps)
