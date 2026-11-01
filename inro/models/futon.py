"""FUTON: a fixed basis per axis, a tensor-network combiner and a decoder.

FUTON evaluates ``decoder(combiner(basis(x)))`` in three stages:

1. A **basis** maps coordinates ``(N, C)`` in ``[-1, 1]`` to one feature matrix
   ``(N, K_c)`` per axis.
2. A **combiner** projects each axis and fuses the projections into
   ``(N, out_features)``.
3. A **decoder**, ``nn.Linear`` or :class:`MLP`, maps the result to ``(N, D)``.

The first two stages make up :class:`FUTONEncoding`.

A feature matrix is a dense tensor, :class:`SharedRows` of a grid's table, or
:class:`Taps` of a local basis. All three multiply a matrix with ``@``, so every
combiner treats every axis alike.
"""

import math
from collections.abc import Callable, Sequence
from functools import reduce
from typing import NamedTuple, Self

import opt_einsum as oe
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import forward_ad

from ..utils import ModuleSpec, build_module
from .mlp import MLP

__all__ = [
    "SharedRows",
    "Taps",
    "CosineBasis",
    "SincBasis",
    "LegendreBasis",
    "ChebyshevBasis",
    "TriangleBasis",
    "LanczosBasis",
    "HadamardCombiner",
    "CPCombiner",
    "TRCombiner",
    "TuckerCombiner",
    "FUTONEncoding",
    "FUTON",
]


def _broadcast(value: int | Sequence[int], length: int, name: str) -> list[int]:
    """Expand a shared int to ``length`` values, or check a per-axis sequence."""
    values = [value] * length if isinstance(value, int) else list(value)
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}, got {len(values)}")
    return values


def _grid_position(x: Tensor, size: int) -> Tensor:
    """Map coordinates in ``[-1, 1]`` to continuous indices in ``[0, size - 1]``."""
    return (x + 1) / 2 * (size - 1)


def _offsets(x: Tensor, size: int) -> Tensor:
    """Return offsets ``(..., size)``, in grid steps, from ``linspace(-1, 1, size)``."""
    centers = torch.arange(size, device=x.device, dtype=x.dtype)
    return _grid_position(x, size).unsqueeze(-1) - centers


def _differentiated(x: Tensor) -> bool:
    """Return whether coordinates carry derivatives, which shared rows would drop.

    Detects reverse mode and, eagerly, forward mode (``torch.func.jvp``, ``jacfwd``):
    a compiled graph cannot unpack a dual tensor outside its level.
    """
    if x.requires_grad:
        return True
    if torch.compiler.is_compiling():
        return False
    return forward_ad.unpack_dual(x).tangent is not None


@torch.compiler.disable()  # sizes depend on the data; compiled, a graph per axis
def _shared(x: Tensor) -> list[tuple[Tensor, Tensor]] | None:
    """Return each axis's distinct coordinates and every point's row among them.

    Returns ``None`` unless the batch is flat, ``(N, C)``, needs no gradients, and
    has at most ``N / 2`` distinct coordinates on every axis, the rule of
    :class:`~inro.models.finr.FINR`.
    """
    if x.ndim != 2 or _differentiated(x):
        return None
    shared = []
    for coordinate in x.unbind(-1):
        values, rows = torch.unique(coordinate, return_inverse=True)
        if 2 * len(values) > len(coordinate):
            return None
        shared.append((values, rows))
    return shared


class SharedRows(NamedTuple):
    """Features of one axis on a grid: a table of ``S`` rows and each point's row.

    Behaves as the ``(N, K_c)`` matrix ``table[rows]``, but a product with a factor
    projects ``S`` rows rather than ``N``.

    Attributes:
        table: Features of the axis's ``S`` coordinates, dense or as :class:`Taps`.
        rows: Row of each of the ``N`` points.
    """

    table: "Tensor | Taps"
    rows: Tensor

    def __matmul__(self, other: Tensor) -> Tensor:
        return (self.table @ other)[self.rows]

    def dense(self) -> Tensor:
        """Return the ``(N, K_c)`` features."""
        table = self.table if isinstance(self.table, Tensor) else self.table.dense()
        return table[self.rows]


class Taps(NamedTuple):
    """Sparse features of one axis: the ``L`` nonzero values of each coordinate.

    Behaves as the ``(N, size)`` matrix whose row ``n`` holds ``values[n]`` in columns
    ``start[n]`` to ``start[n] + L - 1``.

    Attributes:
        values: Nonzero features ``(N, L)``.
        start: First column ``(N,)`` of each row's window.
        size: Number of columns ``K_c``.
        compiled: Whether a compiled graph made the taps, whose product then gathers
            windows of rows. Set where the taps are made: a product traced after a
            graph break may no longer see ``torch.compiler.is_compiling()``.
    """

    values: Tensor
    start: Tensor
    size: int
    compiled: bool = False

    def __matmul__(self, other: Tensor) -> Tensor:
        if self.compiled:
            # The compiler fuses this gather of L rows per point with the weighted
            # sum into one kernel.
            windows = other.unfold(0, self.values.shape[1], 1)
            return (windows[self.start] * self.values.unsqueeze(1)).sum(-1)
        # Eagerly, the same sum in one kernel, without gathering the windows.
        return F.embedding_bag(
            self._columns(), other, per_sample_weights=self.values, mode="sum"
        )

    def dense(self) -> Tensor:
        """Return the ``(N, size)`` features, zero outside the taps."""
        zeros = self.values.new_zeros(len(self.values), self.size)
        return zeros.scatter(1, self._columns(), self.values)

    def _columns(self) -> Tensor:
        taps = torch.arange(self.values.shape[1], device=self.start.device)
        return self.start.unsqueeze(1) + taps


# Features of one axis: dense, the rows a grid shares, or a local basis's taps.
Features = Tensor | SharedRows | Taps


# Bases --------------------------------------------------------------------------------


class _Basis(nn.Module):
    """Base class of the bases; subclasses implement ``_features`` for one axis."""

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_components = _broadcast(num_components, in_features, "num_components")
        self.normalize = normalize
        self._tabulate(grid_size)

    def axis_features(self, x: Tensor, axis: int) -> Tensor:
        """Return the dense features ``(..., K_c)`` of axis ``c``.

        Args:
            x: Coordinates ``(...,)`` of the axis, in ``[-1, 1]``.
            axis: Index ``c`` of the axis.
        """
        return self._normalize(self._features(x, axis))

    def forward(self, x: Tensor) -> tuple[Features, ...]:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {x.shape[-1]}"
            )
        if self.grid_size is not None and not _differentiated(x):
            # A point's row is a rounding: no sort, no synchronization and no
            # evaluation of the basis.
            return tuple(
                SharedRows(self._grid_table(c), _grid_position(xc, n).round().long())
                for c, (xc, n) in enumerate(zip(x.unbind(-1), self.grid_size))
            )
        shared = _shared(x)
        if shared is None:
            return tuple(self._evaluate(xc, c) for c, xc in enumerate(x.unbind(-1)))
        return tuple(
            SharedRows(self._evaluate(values, c), rows)
            for c, (values, rows) in enumerate(shared)
        )

    def _features(self, x: Tensor, axis: int) -> Tensor:
        """Return the features of one axis before normalization."""
        raise NotImplementedError

    def _evaluate(self, x: Tensor, axis: int) -> Tensor | Taps:
        """Return the features of one axis at coordinates ``(N,)``, dense or as taps."""
        return self.axis_features(x, axis)

    def _normalize(self, features: Tensor) -> Tensor:
        return F.normalize(features, dim=-1) if self.normalize else features

    def _tabulate(
        self,
        grid_size: int | Sequence[int] | None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Evaluate each axis once on its grid ``linspace(-1, 1, size)``.

        Registers the buffers ``_table{axis}``, dense or the taps' values, and
        ``_start{axis}``, the taps' first columns or ``None``. Subclasses whose
        features need state of their own call it again once that is set.
        """
        self.grid_size = None
        if grid_size is not None:
            self.grid_size = _broadcast(grid_size, self.in_features, "grid_size")
        for c, size in enumerate(self.grid_size or ()):
            grid = torch.linspace(-1, 1, size, device=device, dtype=dtype)
            table, start = self._evaluate(grid, c), None
            if isinstance(table, Taps):
                table, start = table.values, table.start
            self.register_buffer(f"_table{c}", table, persistent=False)
            self.register_buffer(f"_start{c}", start, persistent=False)

    def _grid_table(self, axis: int) -> Tensor | Taps:
        """Return the table of one axis on its declared grid."""
        table, start = getattr(self, f"_table{axis}"), getattr(self, f"_start{axis}")
        if start is None:
            return table
        size = self.num_components[axis]
        return Taps(table, start, size, torch.compiler.is_compiling())

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        # Evaluate the tables again where a move or cast takes them, as nn.RNNBase
        # rebuilds its flat weights: a cast would keep the rounding of the old dtype.
        super()._apply(fn, recurse)
        if self.grid_size is not None:
            table = getattr(self, "_table0")
            self._tabulate(self.grid_size, table.device, table.dtype)
        return self


class CosineBasis(_Basis):
    r"""Cosine basis :math:`\cos(k \pi u)` with :math:`u = (x + 1) / 2`.

    Axis ``c`` uses frequencies :math:`k = 0, \dots, K_c - 1`. This is the paper's
    basis without its :math:`\sqrt{2}` factor, which the factors absorb.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of frequencies ``K_c``, shared or per axis.
        normalize: Whether to L2-normalize each feature vector.
        grid_size: Sizes of the grid ``linspace(-1, 1, size)`` all inputs lie on, shared
            or per axis, tabulated once; see :class:`FUTONEncoding`.

    Shape:
        - Input: ``(..., C)``.
        - Output: ``C`` features ``(..., K_c)``.
    """

    def _features(self, x: Tensor, axis: int) -> Tensor:
        u = ((x + 1) / 2).unsqueeze(-1)
        k = torch.arange(self.num_components[axis], device=x.device, dtype=x.dtype)
        return torch.cos(math.pi * k * u)


class SincBasis(_Basis):
    r"""Cardinal sine basis centered on a uniform grid over ``[-1, 1]``.

    Component ``k`` is :math:`\operatorname{sinc}((x - \mu_k) / w)`, with centers
    :math:`\mu_k = -1 + k w` and spacing :math:`w = 2 / (K_c - 1)`. Each component is
    one at its own center and zero at the others, so ``normalize=False`` gives
    cardinal (Whittaker) interpolation.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of centers ``K_c``, shared or per axis.
        normalize: Whether to L2-normalize each feature vector.
        grid_size: Sizes of the grid ``linspace(-1, 1, size)`` all inputs lie on, shared
            or per axis, tabulated once; see :class:`FUTONEncoding`.

    Shape:
        - Input: ``(..., C)``.
        - Output: ``C`` features ``(..., K_c)``.
    """

    def _features(self, x: Tensor, axis: int) -> Tensor:
        return torch.sinc(_offsets(x, self.num_components[axis]))


def _chebyshev_to_legendre(count: int) -> Tensor:
    """Return the ``(count, count)`` matrix ``M`` with ``P_n = sum_k M[n, k] T_k``.

    Runs Bonnet's recurrence, :math:`k P_k = (2k - 1) x P_{k-1} - (k - 1) P_{k-2}`, on
    Chebyshev coefficients with :math:`x T_j = (T_{j+1} + T_{j-1}) / 2`. The entries
    are nonnegative and each row sums to one, so the map is well conditioned.
    """
    matrix = torch.zeros(count, count, dtype=torch.float64)
    matrix[0, 0] = 1.0
    if count > 1:
        matrix[1, 1] = 1.0
    for n in range(2, count):
        previous = matrix[n - 1]
        times_x = torch.zeros_like(previous)
        times_x[1:] += previous[:-1] / 2
        times_x[:-1] += previous[1:] / 2
        times_x[1] += previous[0] / 2  # x T_0 = T_1
        matrix[n] = ((2 * n - 1) * times_x - (n - 1) * matrix[n - 2]) / n
    return matrix.to(torch.get_default_dtype())


class ChebyshevBasis(_Basis):
    r"""Chebyshev polynomials of the first kind on ``[-1, 1]``.

    Degree ``k`` is :math:`T_k(x) = \cos(k \arccos x)`. The unscaled polynomials are
    orthogonal under the weight :math:`(1 - x^2)^{-1/2}`.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of polynomials ``K_c``, shared or per axis.
        normalize: Whether to L2-normalize each feature vector.
        grid_size: Sizes of the grid ``linspace(-1, 1, size)`` all inputs lie on, shared
            or per axis, tabulated once; see :class:`FUTONEncoding`.

    Shape:
        - Input: ``(..., C)``.
        - Output: ``C`` features ``(..., K_c)``.
    """

    def _features(self, x: Tensor, axis: int) -> Tensor:
        count = self.num_components[axis]
        degrees = torch.arange(count, dtype=x.dtype, device=x.device)
        return torch.cos(degrees * torch.arccos(x.unsqueeze(-1)))


class LegendreBasis(ChebyshevBasis):
    r"""Legendre polynomials on ``[-1, 1]``.

    The standard, unscaled orthogonal polynomials, computed as fixed combinations of
    Chebyshev polynomials with nonnegative coefficients that sum to one.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of polynomials ``K_c``, shared or per axis.
        normalize: Whether to L2-normalize each feature vector.
        grid_size: Sizes of the grid ``linspace(-1, 1, size)`` all inputs lie on, shared
            or per axis, tabulated once; see :class:`FUTONEncoding`.

    Shape:
        - Input: ``(..., C)``.
        - Output: ``C`` features ``(..., K_c)``.
    """

    _from_chebyshev: Tensor

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        grid_size: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__(in_features, num_components, normalize)
        matrix = _chebyshev_to_legendre(max(self.num_components))
        self.register_buffer("_from_chebyshev", matrix, persistent=False)
        self._tabulate(grid_size)

    def _features(self, x: Tensor, axis: int) -> Tensor:
        count = self.num_components[axis]
        matrix = self._from_chebyshev[:count, :count]
        return super()._features(x, axis) @ matrix.T


class _LocalBasis(_Basis):
    """Base class of compact kernels centered on ``linspace(-1, 1, K_c)``.

    Subclasses implement ``_kernel`` on offsets ``t`` in grid steps. It must vanish
    for ``|t| >= radius``, so each coordinate touches at most ``2 * radius``
    consecutive components, the only ones sparse mode evaluates, as :class:`Taps`.
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        radius: int,
        normalize: bool = True,
        sparse: bool = True,
        grid_size: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__(in_features, num_components, normalize)
        if radius < 1 or min(self.num_components) < 2 * radius:
            raise ValueError(
                "radius must be >= 1 and num_components >= 2 * radius, got "
                f"radius {radius} and num_components {self.num_components}"
            )
        self.radius = int(radius)
        self.sparse = bool(sparse)
        self._tabulate(grid_size)

    def _kernel(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def _features(self, x: Tensor, axis: int) -> Tensor:
        return self._kernel(_offsets(x, self.num_components[axis]))

    def _evaluate(self, x: Tensor, axis: int) -> Tensor | Taps:
        if not self.sparse:
            return self.axis_features(x, axis)
        # Clamping keeps each window of 2 * radius taps inside the basis; taps it
        # shifts lie outside the kernel's support and evaluate to zero.
        size = self.num_components[axis]
        position = _grid_position(x, size)
        start = (position.floor() - (self.radius - 1)).clamp(0, size - 2 * self.radius)
        taps = torch.arange(2 * self.radius, device=x.device, dtype=x.dtype)
        values = self._normalize(self._kernel((position - start).unsqueeze(-1) - taps))
        return Taps(values, start.long(), size, torch.compiler.is_compiling())


class TriangleBasis(_LocalBasis):
    r"""Triangle (hat) basis for piecewise-linear interpolation on ``[-1, 1]``.

    Component ``k`` is :math:`\max(0, 1 - |x - \mu_k| / w)` with centers spaced by
    :math:`w = 2 / (K_c - 1)`, so at most two components are nonzero. With
    ``normalize=False``, a CP combiner and an MLP decoder give the TensoRF-CP
    parametrization: a linear map of these features samples a 1D feature grid.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of centers ``K_c >= 2``, shared or per axis.
        normalize: Whether to L2-normalize each feature vector.
        sparse: Whether to evaluate only the two taps of each coordinate, as
            :class:`Taps`.
        grid_size: Sizes of the grid ``linspace(-1, 1, size)`` all inputs lie on, shared
            or per axis, tabulated once; see :class:`FUTONEncoding`.

    Shape:
        - Input: ``(..., C)``, flat ``(N, C)`` in sparse mode.
        - Output: ``C`` features ``(..., K_c)``, as :class:`Taps` in sparse mode.

    References:
        Chen et al., "TensoRF: Tensorial Radiance Fields", ECCV 2022.
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        normalize: bool = True,
        sparse: bool = True,
        grid_size: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__(in_features, num_components, 1, normalize, sparse, grid_size)

    def _kernel(self, t: Tensor) -> Tensor:
        return (1.0 - t.abs()).clamp(min=0.0)


class LanczosBasis(_LocalBasis):
    r"""Lanczos (windowed sinc) basis centered on a uniform grid over ``[-1, 1]``.

    With offset :math:`t = (x - \mu_k) / w` and radius :math:`a`, the kernel is
    :math:`\operatorname{sinc}(t) \operatorname{sinc}(t / a)` for :math:`|t| < a` and
    zero elsewhere. Each component is one at its own center and zero at the others,
    and at most ``2 * radius`` are nonzero.

    Args:
        in_features: Number of coordinate axes ``C``.
        num_components: Number of centers ``K_c >= 2 * radius``, shared or per axis.
        radius: Kernel radius in grid steps: ``2`` gives Lanczos-2 (four taps), ``3``
            Lanczos-3 (six taps).
        normalize: Whether to L2-normalize each feature vector.
        sparse: Whether to evaluate only the ``2 * radius`` taps of each coordinate,
            as :class:`Taps`.
        grid_size: Sizes of the grid ``linspace(-1, 1, size)`` all inputs lie on, shared
            or per axis, tabulated once; see :class:`FUTONEncoding`.

    Shape:
        - Input: ``(..., C)``, flat ``(N, C)`` in sparse mode.
        - Output: ``C`` features ``(..., K_c)``, as :class:`Taps` in sparse mode.
    """

    def __init__(
        self,
        in_features: int,
        num_components: int | Sequence[int],
        radius: int = 3,
        normalize: bool = True,
        sparse: bool = True,
        grid_size: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            in_features, num_components, radius, normalize, sparse, grid_size
        )

    def _kernel(self, t: Tensor) -> Tensor:
        window = torch.sinc(t) * torch.sinc(t / self.radius)
        return torch.where(t.abs() < self.radius, window, 0.0)


# Combiners ----------------------------------------------------------------------------


class _Combiner(nn.Module):
    """Base class of the combiners; ``in_features[c]`` is the width ``K_c``."""

    out_features: int

    def __init__(self, in_features: Sequence[int]) -> None:
        super().__init__()
        self.in_features = list(in_features)


class HadamardCombiner(_Combiner):
    """Elementwise product of per-axis features that share one width ``K``.

    Shared rows and taps are expanded to dense features first.

    Args:
        in_features: Feature width of each axis; all entries must be equal.

    Shape:
        - Input: ``C`` features ``(..., K)``.
        - Output: ``(..., K)``.
    """

    def __init__(self, in_features: Sequence[int]) -> None:
        super().__init__(in_features)
        if len(set(self.in_features)) != 1:
            raise ValueError(
                f"Expected all in_features entries to be equal, got {self.in_features}"
            )
        self.out_features = self.in_features[0]

    def forward(self, features: Sequence[Features]) -> Tensor:
        dense = (f if isinstance(f, Tensor) else f.dense() for f in features)
        return reduce(torch.mul, dense)


class CPCombiner(_Combiner):
    r"""Canonical polyadic (CP) combiner.

    Projects each axis to ``R`` channels with its factor
    :math:`\boldsymbol{U}^{(c)}`, ``factors[c]`` of shape ``(K_c, R)``, and multiplies
    the projections elementwise (Eqs. 10 and 11 of the paper).

    Args:
        in_features: Feature width ``K_c`` of each axis.
        rank: CP rank ``R``, which is also the output width.

    Shape:
        - Input: ``C`` features ``(N, K_c)``.
        - Output: ``(N, R)``.

    References:
        Kolda and Bader, "Tensor Decompositions and Applications", SIAM Review 2009.
    """

    def __init__(self, in_features: Sequence[int], rank: int) -> None:
        super().__init__(in_features)
        self.rank = rank
        self.out_features = rank
        self.factors = nn.ParameterList(
            nn.Parameter(torch.empty(width, rank)) for width in self.in_features
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the factors Kaiming-uniform, with fan-in ``K_c``."""
        for factor in self.factors:
            nn.init.kaiming_uniform_(factor.T)

    def forward(self, features: Sequence[Features]) -> Tensor:
        return reduce(torch.mul, (f @ u for f, u in zip(features, self.factors)))


class TRCombiner(_Combiner):
    r"""Tensor-ring (TR) combiner.

    Axis ``c`` contracts its features with a core ``(R, K_c, R)`` into an ``R x R``
    matrix. The ordered product of these matrices is flattened, leaving the closure
    of the ring, such as a trace, to the decoder. Extends the paper's CP model.

    Args:
        in_features: Feature width ``K_c`` of each axis.
        rank: Tensor-ring rank ``R``; the output width is ``R**2``.

    Shape:
        - Input: ``C`` features ``(N, K_c)``.
        - Output: ``(N, R**2)``.

    References:
        Zhao et al., "Tensor Ring Decomposition", arXiv 2016.
    """

    def __init__(self, in_features: Sequence[int], rank: int) -> None:
        super().__init__(in_features)
        self.rank = rank
        self.out_features = rank**2
        self.cores = nn.ParameterList(
            nn.Parameter(torch.empty(rank, width, rank)) for width in self.in_features
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the cores with Kaiming-uniform weights."""
        for core in self.cores:
            nn.init.kaiming_uniform_(core)

    def forward(self, features: Sequence[Features]) -> Tensor:
        rank = self.rank
        # The product with a core folded to (K_c, R * R) is one matmul per axis.
        matrices = (
            (f @ core.transpose(0, 1).flatten(1)).unflatten(-1, (rank, rank))
            for f, core in zip(features, self.cores)
        )
        return reduce(torch.matmul, matrices).flatten(start_dim=-2)


class TuckerCombiner(_Combiner):
    r"""Tucker combiner.

    Projects axis ``c`` to ``rank[c]`` channels with its factor ``(K_c, rank[c])`` and
    contracts the projections with a core of shape
    ``(rank[0], ..., rank[C - 1], out_features)``. With a linear decoder the weight
    tensor is in Tucker format, and CP is the case of a superdiagonal core. Extends
    the paper's CP model.

    Args:
        in_features: Feature width ``K_c`` of each axis.
        rank: Multilinear rank, shared or per axis.
        out_features: Output width, the core's last mode; ``None`` uses the largest
            rank.

    Shape:
        - Input: ``C`` features ``(N, K_c)``.
        - Output: ``(N, out_features)``.

    References:
        Tucker, "Some mathematical notes on three-mode factor analysis",
        Psychometrika 1966.
    """

    def __init__(
        self,
        in_features: Sequence[int],
        rank: int | Sequence[int],
        out_features: int | None = None,
    ) -> None:
        super().__init__(in_features)
        modes = len(self.in_features)
        self.rank = _broadcast(rank, modes, "rank")
        self.out_features = max(self.rank) if out_features is None else out_features
        self.factors = nn.ParameterList(
            nn.Parameter(torch.empty(width, r))
            for width, r in zip(self.in_features, self.rank)
        )
        self.core = nn.Parameter(torch.empty(*self.rank, self.out_features))
        # The core absorbs one projection at a time, largest rank first, which keeps
        # the intermediates small; opt_einsum's own optimum starts with an outer
        # product of the projections, slower and larger. Each result goes to the end
        # of the operands, hence the path, and the points' count of 1 is a placeholder.
        self.order = sorted(range(modes), key=self.rank.__getitem__, reverse=True)
        self.contract = oe.contract_expression(
            tuple(self.core.shape),
            [*range(modes + 1)],
            *(arg for c in self.order for arg in ((1, self.rank[c]), [..., c])),
            [..., modes],
            optimize=[(0, 1)] + [(0, n) for n in range(modes - 1, 0, -1)],
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the factors (fan-in ``K_c``) and the core Kaiming-uniform."""
        for factor in self.factors:
            nn.init.kaiming_uniform_(factor.T)
        nn.init.kaiming_uniform_(self.core.view(-1, self.out_features))

    def forward(self, features: Sequence[Features]) -> Tensor:
        projections = [f @ factor for f, factor in zip(features, self.factors)]
        return self.contract(self.core, *(projections[c] for c in self.order))


# FUTON --------------------------------------------------------------------------------


class FUTONEncoding(nn.Module):
    r"""FUTON's encoding: the features of a basis, fused by a combiner.

    Computes ``combiner(basis(x))``, which :class:`FUTON` decodes.

    On a grid, each projection
    :math:`\boldsymbol{U}^{(c)\top} \boldsymbol{\varphi}(x_c)` depends on one
    coordinate, so points share it. When every axis of a batch repeats its coordinates,
    the basis evaluates each distinct coordinate once and every point gathers its rows:
    the same function and gradients at
    :math:`O(R \sum_c S_c K_c + NCR)` instead of :math:`O(NR \sum_c K_c)`, for
    :math:`S_c` distinct coordinates on axis :math:`c`.

    A basis given ``grid_size`` tabulates its features once, when built, and rounds
    each point to its row, which spares a sort and a synchronization per batch. An
    off-grid point reads the nearest row, trading exactness for speed. Coordinates
    that need gradients are always evaluated point by point. The tables are
    non-persistent buffers, evaluated again by ``to`` and ``double``.

    Args:
        in_features: Number of coordinate axes ``C``.
        basis: Basis spec, built as ``basis(in_features, **kwargs)``. A custom basis
            must expose ``num_components``, one count per axis.
        combiner: Combiner spec, built as ``combiner(basis.num_components, **kwargs)``.
            A custom combiner must expose ``out_features``.

    Shape:
        - Input: ``(N, C)`` coordinates in ``[-1, 1]``.
        - Output: ``(N, out_features)``.
    """

    def __init__(
        self, in_features: int, basis: ModuleSpec, combiner: ModuleSpec
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.basis = build_module(basis, BASES, in_features)
        self.combiner = build_module(combiner, COMBINERS, self.basis.num_components)
        self.out_features = self.combiner.out_features

    def forward(self, x: Tensor) -> Tensor:
        features = self.basis(x)
        return self.combiner(features)


class FUTON(nn.Module):
    r"""Fourier Tensor Network for implicit neural representations.

    Computes ``output_activation(decoder(combiner(basis(x))))``. Each stage is a spec:
    a module instance, a registry key, a module class, or a ``(key_or_class, kwargs)``
    pair. Registry keys:

    - Bases: ``"cosine"``, ``"sinc"``, ``"legendre"``, ``"chebyshev"``, ``"triangle"``,
      ``"lanczos"``.
    - Combiners: ``"hadamard"``, ``"cp"``, ``"tr"``, ``"tucker"``.
    - Decoders: ``"linear"``, ``"mlp"``.

    Args:
        in_features: Number of coordinate axes ``C``.
        out_features: Number of output channels ``D``.
        basis: Basis spec; see :class:`FUTONEncoding`.
        combiner: Combiner spec; see :class:`FUTONEncoding`.
        decoder: Decoder spec, built as
            ``decoder(encoding.out_features, out_features, **kwargs)``.
        output_activation: Callable applied to the output; ``None`` uses the identity.

    Shape:
        - Input: ``(..., C)`` coordinates in ``[-1, 1]``.
        - Output: ``(..., D)``.

    Note:
        The cosine basis, CP combiner and linear decoder implement Eq. (9) of the
        paper: ``encoding.combiner.factors[c]`` stores :math:`\boldsymbol{U}^{(c)}` and
        ``decoder.weight`` stores :math:`\boldsymbol{V}`. The other bases and
        combiners and the MLP decoder are extensions.

    Examples:
        >>> model = FUTON(
        ...     in_features=2,
        ...     out_features=3,
        ...     basis=("cosine", {"num_components": 256}),
        ...     combiner=("cp", {"rank": 256}),
        ...     decoder="linear",
        ...     output_activation=torch.tanh,
        ... )
        >>> model(torch.rand(64, 64, 2) * 2 - 1).shape
        torch.Size([64, 64, 3])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        basis: ModuleSpec,
        combiner: ModuleSpec,
        decoder: ModuleSpec,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.encoding = FUTONEncoding(in_features, basis, combiner)
        self.decoder = build_module(
            decoder, DECODERS, self.encoding.out_features, out_features
        )
        self.output_activation = (
            output_activation if output_activation is not None else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        # Flat, so that the basis can share or sparsify the points.
        points = x.reshape(-1, self.in_features)
        fused = self.encoding(points)
        out = self.output_activation(self.decoder(fused))
        return out.reshape(*x.shape[:-1], self.out_features)


# Registries ---------------------------------------------------------------------------

BASES: dict[str, type[nn.Module]] = {
    "cosine": CosineBasis,
    "sinc": SincBasis,
    "legendre": LegendreBasis,
    "chebyshev": ChebyshevBasis,
    "triangle": TriangleBasis,
    "lanczos": LanczosBasis,
}

COMBINERS: dict[str, type[nn.Module]] = {
    "hadamard": HadamardCombiner,
    "cp": CPCombiner,
    "tr": TRCombiner,
    "tucker": TuckerCombiner,
}

DECODERS: dict[str, type[nn.Module]] = {
    "linear": nn.Linear,
    "mlp": MLP,
}
