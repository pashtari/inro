"""FUTON: a fixed basis per axis, a tensor-network combiner and a head.

FUTON computes ``output_activation(head(combiner(basis(x))))`` for coordinates
``(N, C)`` in ``[-1, 1]``:

1. A **basis** gives each axis ``c`` features ``(N, K_c)``. Every basis takes
   ``in_features``, the number of axes ``C``; ``num_components``, the widths ``K_c``,
   one for all axes or one per axis; ``normalize``, whether to L2-normalize each
   feature vector (default ``True``); and ``grid_size``, the sizes of the grid
   ``linspace(-1, 1, size)`` all inputs lie on, if any, whose features it tabulates once.
2. A **combiner** fuses the axes with a tensor network, stated once as an einsum
   equation over the axes' projections, into ``(N, R)`` features.
3. A **head** maps them to the signal: ``"linear"``, ``"mlp"`` or ``"identity"``.

Points share the features of their shared coordinates. On a grid, a basis gives each axis
as :class:`SharedRows`, a table of its coordinates and every point's row in it, so a
combiner projects ``S_c`` rows rather than ``N`` points, and it returns its features as a
:class:`GridFeatures`, its equation and operands, which the head's first linear layer reads
off the whole grid in one einsum with its weight when the batch covers the grid. A kernel
of compact support gives scattered points only their nonzero features, as :class:`Taps`.
Coordinates that need gradients are evaluated point by point.
"""

import math
import string
from collections.abc import Callable, Sequence
from itertools import chain
from typing import NamedTuple, Self

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import forward_ad

from ..utils import ModuleSpec, build_module
from .mlp import MLP

Activation = Callable[[Tensor], Tensor]
Sizes = int | Sequence[int]  # one for all axes, or one per axis


def _broadcast(value: Sizes, length: int, name: str) -> list[int]:
    """Expand a shared int to ``length`` values, or check a per-axis sequence."""
    values = [value] * length if isinstance(value, int) else list(value)
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}, got {len(values)}")
    return values


def _grid_position(x: Tensor, size: int) -> Tensor:
    """Map coordinates in ``[-1, 1]`` to continuous indices in ``[0, size - 1]``."""
    return (x + 1) / 2 * (size - 1)


def _differentiated(x: Tensor) -> bool:
    """Return whether coordinates carry derivatives, which shared rows would drop.

    Detects reverse mode and, eagerly, forward mode (``torch.func.jvp``, ``jacfwd``):
    a compiled graph cannot unpack a dual tensor outside its level.
    """
    eager = not torch.compiler.is_compiling()
    return x.requires_grad or (eager and forward_ad.unpack_dual(x).tangent is not None)


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


# Features -----------------------------------------------------------------------------


class SharedRows(NamedTuple):
    """Features of one axis on a grid: a table of ``S`` rows and each point's row.

    Behaves as the ``(N, K_c)`` matrix ``table[rows]``, which :meth:`dense` returns. A
    product with a factor projects the ``S`` rows rather than ``N`` points, and the
    product shares the rows still.

    Attributes:
        table: Features ``(S, K_c)`` of the axis's ``S`` coordinates.
        rows: Row of each of the ``N`` points.
    """

    table: Tensor
    rows: Tensor

    def __matmul__(self, other: Tensor) -> "SharedRows":
        return SharedRows(self.table @ other, self.rows)

    def unflatten(self, dim: int, sizes: Sequence[int]) -> "SharedRows":
        """Unflatten a dimension of the features, as :meth:`torch.Tensor.unflatten`."""
        return SharedRows(self.table.unflatten(dim, sizes), self.rows)

    def dense(self) -> Tensor:
        """Return the matrix ``table[rows]``, a row per point."""
        return self.table[self.rows]


class Taps(NamedTuple):
    """Sparse features of one axis: the ``L`` nonzero values of each coordinate.

    Behaves as the ``(N, K_c)`` matrix whose row ``n`` holds ``values[n]`` in columns
    ``start[n]`` to ``start[n] + L - 1``.

    Attributes:
        values: Nonzero features ``(N, L)``.
        start: First column ``(N,)`` of each row's window.
        compiled: Whether a compiled graph made the taps, whose product then gathers
            windows of rows. Set where the taps are made: a product traced after a
            graph break may no longer see ``torch.compiler.is_compiling()``.
    """

    values: Tensor
    start: Tensor
    compiled: bool = False

    def __matmul__(self, other: Tensor) -> Tensor:
        width = self.values.shape[1]
        if self.compiled:  # the compiler fuses the gather with the weighted sum
            windows = other.unfold(0, width, 1)[self.start]
            return (windows * self.values.unsqueeze(1)).sum(-1)
        # Eagerly, the same sum in one kernel, without gathering the windows.
        cols = self.start.unsqueeze(1) + torch.arange(width, device=self.start.device)
        return F.embedding_bag(cols, other, per_sample_weights=self.values, mode="sum")


# Features of one axis: dense, the rows a grid shares, or a local basis's taps.
Features = Tensor | SharedRows | Taps


# Bases --------------------------------------------------------------------------------


class _Basis(nn.Module):
    """Base class of the bases.

    Subclasses implement ``_features``, the features of one axis before normalization,
    or ``_kernel``, a function of the offsets in grid steps from the components'
    centers ``linspace(-1, 1, K_c)``. A kernel that vanishes beyond ``radius`` steps
    gives scattered coordinates only their ``2 * radius`` nonzero features, as
    :class:`Taps`.
    """

    radius = 0  # the kernel's support in grid steps; 0 for global bases

    def __init__(
        self,
        in_features: int,
        num_components: Sizes,
        normalize: bool = True,
        grid_size: Sizes | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_components = _broadcast(num_components, in_features, "num_components")
        if min(self.num_components) < 2 * self.radius:
            raise ValueError(f"num_components must be at least {2 * self.radius}")
        self.normalize = normalize
        self.grid_size: list[int] | None = None
        if grid_size is not None:
            self.grid_size = _broadcast(grid_size, in_features, "grid_size")
        self._tabulate(torch.empty(0))

    def axis_features(self, x: Tensor, axis: int) -> Tensor:
        """Return the dense features ``(..., K_c)`` of axis ``c``.

        Args:
            x: Coordinates ``(...,)`` of the axis, in ``[-1, 1]``.
            axis: Index ``c`` of the axis.
        """
        return self._normalize(self._features(x, axis))

    def forward(self, x: Tensor) -> tuple[Features, ...]:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected {self.in_features} axes, got {x.shape[-1]}")
        if self.grid_size is not None and not _differentiated(x):
            # A point's row is a rounding: no sort, no synchronization and no
            # evaluation of the basis.
            tables = (getattr(self, f"_table{c}") for c in range(self.in_features))
            return tuple(
                SharedRows(table, _grid_position(xc, len(table)).round().long())
                for table, xc in zip(tables, x.unbind(-1))
            )
        shared = _shared(x)
        if shared is None:  # scattered: a local basis's taps, other bases' features
            evaluate = self._taps if self.radius else self.axis_features
            return tuple(evaluate(xc, c) for c, xc in enumerate(x.unbind(-1)))
        return tuple(
            SharedRows(self.axis_features(values, c), rows)
            for c, (values, rows) in enumerate(shared)
        )

    def _features(self, x: Tensor, axis: int) -> Tensor:
        size = self.num_components[axis]
        centers = torch.arange(size, device=x.device, dtype=x.dtype)
        return self._kernel(_grid_position(x, size).unsqueeze(-1) - centers)

    def _kernel(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def _taps(self, x: Tensor, axis: int) -> Taps:
        """Return the nonzero features of one axis at scattered coordinates ``(N,)``."""
        # Clamping keeps each window of 2 * radius taps inside the basis; taps it
        # shifts lie outside the kernel's support and evaluate to zero.
        size = self.num_components[axis]
        position = _grid_position(x, size)
        start = (position.floor() - (self.radius - 1)).clamp(0, size - 2 * self.radius)
        taps = torch.arange(2 * self.radius, device=x.device, dtype=x.dtype)
        values = self._normalize(self._kernel((position - start).unsqueeze(-1) - taps))
        return Taps(values, start.long(), torch.compiler.is_compiling())

    def _normalize(self, features: Tensor) -> Tensor:
        return F.normalize(features, dim=-1) if self.normalize else features

    def _tabulate(self, like: Tensor) -> None:
        """Evaluate each axis once on its grid, as the buffer ``_table{axis}``, in the
        dtype and on the device of ``like``."""
        for c, size in enumerate(self.grid_size or ()):
            grid = torch.linspace(-1, 1, size, device=like.device, dtype=like.dtype)
            self.register_buffer(f"_table{c}", self.axis_features(grid, c), False)

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        # Rebuild the buffers where a move or cast takes them, as nn.RNNBase rebuilds
        # its flat weights: a cast would keep the rounding of the old dtype. A move
        # to where they are, as every evaluation makes, keeps them.
        before = [(b.device, b.dtype) for b in self.buffers(recurse=False)]
        super()._apply(fn, recurse)
        if [(b.device, b.dtype) for b in self.buffers(recurse=False)] != before:
            self._tabulate(next(self.buffers(recurse=False)))
        return self


class CosineBasis(_Basis):
    r"""Cosines :math:`\cos(k \pi u)` with :math:`u = (x + 1) / 2`, of frequencies
    :math:`k = 0, \dots, K_c - 1`: the paper's basis without its :math:`\sqrt{2}`,
    which the factors absorb."""

    def _features(self, x: Tensor, axis: int) -> Tensor:
        u = ((x + 1) / 2).unsqueeze(-1)
        k = torch.arange(self.num_components[axis], device=x.device, dtype=x.dtype)
        return torch.cos(math.pi * k * u)


class SincBasis(_Basis):
    r"""Cardinal sines :math:`\operatorname{sinc}((x - \mu_k) / w)` centered on
    :math:`\mu_k = -1 + k w`, spaced by :math:`w = 2 / (K_c - 1)`.

    Each component is one at its own center and zero at the others, so
    ``normalize=False`` gives cardinal (Whittaker) interpolation.
    """

    def _kernel(self, t: Tensor) -> Tensor:
        return torch.sinc(t)


class ChebyshevBasis(_Basis):
    r"""Chebyshev polynomials :math:`T_k(x) = \cos(k \arccos x)` of the first kind,
    orthogonal under the weight :math:`(1 - x^2)^{-1/2}`."""

    def _features(self, x: Tensor, axis: int) -> Tensor:
        k = torch.arange(self.num_components[axis], dtype=x.dtype, device=x.device)
        return torch.cos(k * torch.arccos(x.unsqueeze(-1)))


def _chebyshev_to_legendre(count: int) -> Tensor:
    r"""Return the ``(count, count)`` matrix ``M`` with ``P_n = sum_m M[n, m] T_m``.

    From :math:`P_n(\cos t) = \sum_{k=0}^n c_k c_{n-k} \cos((n - 2k) t)`, with
    :math:`c_k = \binom{2k}{k} / 4^k`, in float64. The entries are nonnegative and
    each row sums to one, so the map is well conditioned.
    """
    k = torch.arange(count, dtype=torch.float64)
    c = torch.cat([k.new_ones(1), torch.cumprod(1 - 0.5 / k[1:], 0)])
    n, j = torch.meshgrid(k.long(), k.long(), indexing="ij")
    terms = torch.where(j <= n, c[j] * c[(n - j).clamp(min=0)], 0)
    columns = (n - 2 * j).abs().clamp(max=count - 1)  # T_{|n - 2k|}
    return torch.zeros(count, count, dtype=torch.float64).scatter_add(1, columns, terms)


class LegendreBasis(ChebyshevBasis):
    """Legendre polynomials, the standard, unscaled ones, as fixed combinations of
    Chebyshev polynomials with nonnegative coefficients that sum to one."""

    _from_chebyshev: Tensor

    def _tabulate(self, like: Tensor) -> None:
        # The tables need the matrix, built anew in their dtype.
        matrix = _chebyshev_to_legendre(max(self.num_components)).to(like)
        self.register_buffer("_from_chebyshev", matrix, persistent=False)
        super()._tabulate(like)

    def _features(self, x: Tensor, axis: int) -> Tensor:
        count = self.num_components[axis]
        return super()._features(x, axis) @ self._from_chebyshev[:count, :count].T


class TriangleBasis(_Basis):
    r"""Triangles (hats) :math:`\max(0, 1 - |x - \mu_k| / w)` centered on
    :math:`\mu_k = -1 + k w`, spaced by :math:`w = 2 / (K_c - 1)`: piecewise-linear
    interpolation, two components nonzero, ``K_c >= 2``.

    With ``normalize=False``, a CP combiner and an MLP head give the TensoRF-CP
    parametrization: a linear map of these features samples a 1D feature grid.

    References:
        Chen et al., "TensoRF: Tensorial Radiance Fields", ECCV 2022.
    """

    radius = 1

    def _kernel(self, t: Tensor) -> Tensor:
        return (1.0 - t.abs()).clamp(min=0.0)


class LanczosBasis(_Basis):
    r"""Lanczos kernels, windowed sines centered on :math:`\mu_k = -1 + k w`, spaced by
    :math:`w = 2 / (K_c - 1)`, ``K_c >= 2 * radius``.

    With offset :math:`t = (x - \mu_k) / w` and radius :math:`a`, the kernel is
    :math:`\operatorname{sinc}(t) \operatorname{sinc}(t / a)` for :math:`|t| < a` and
    zero elsewhere: one at its own center, zero at the others, and at most ``2 * a``
    components nonzero.

    Args:
        radius: Kernel radius in grid steps: ``2`` gives Lanczos-2 (four taps), ``3``
            Lanczos-3 (six taps).
    """

    def __init__(
        self,
        in_features: int,
        num_components: Sizes,
        radius: int = 3,
        normalize: bool = True,
        grid_size: Sizes | None = None,
    ):
        if radius < 1:
            raise ValueError(f"radius must be at least 1, got {radius}")
        self.radius = int(radius)  # first: the checks and the tables need it
        super().__init__(in_features, num_components, normalize, grid_size)

    def _kernel(self, t: Tensor) -> Tensor:
        window = torch.sinc(t) * torch.sinc(t / self.radius)
        return torch.where(t.abs() < self.radius, window, 0.0)


# Combiners ----------------------------------------------------------------------------

# An operand of a combiner's network: dense, or the rows a grid shares.
Operand = Tensor | SharedRows


def _fold(equation: str, operands: Sequence[Operand]) -> Tensor:
    """Contract ``equation`` at the points, pairwise in the order of its operands.

    Each step keeps the points, ``...``, and the subscripts that later operands or the
    output need, so the order of the operands is the order of contraction. The first
    operand must be an axis's.
    """
    dense = [op.dense() if isinstance(op, SharedRows) else op for op in operands]
    inputs, output = equation.split("->")
    (held, *rest), out = inputs.split(","), dense[0]
    for i, (subscripts, operand) in enumerate(zip(rest, dense[1:])):
        needed = "".join(rest[i + 1 :]) + output
        kept = [c for c in dict.fromkeys(held + subscripts) if c in needed and c != "."]
        result = output if i == len(rest) - 1 else "..." + "".join(kept)
        out, held = torch.einsum(f"{held},{subscripts}->{result}", out, operand), result
    return out


def _grid(equation: str, operands: Sequence[Operand], weight: Tensor) -> Tensor:
    """Contract ``equation`` and ``weight`` on every point of the grid, read at the points.

    Each axis's ``...`` becomes its own subscript, and ``weight``, a linear layer's
    ``(width, *shape)``, contracts the network's output: one
    :func:`torch.einsum` over the axes' tables, along a path planned on their sizes.
    """
    inputs, output = equation.split("->")
    *parts, last = inputs.split("...")
    axes = string.ascii_uppercase[: len(parts)]
    subscripts = "".join(chain(*zip(parts, axes))) + last
    tables = [op.table if isinstance(op, SharedRows) else op for op in operands]
    grid = torch.einsum(f"{subscripts},Z{output[3:]}->{axes}Z", *tables, weight)
    return grid[tuple(op.rows for op in operands if isinstance(op, SharedRows))]


class GridFeatures:
    """A combiner's features on a grid, its network not yet contracted.

    :func:`torch.nn.functional.linear`, as a head's first layer calls it, contracts the
    network with the layer's weight in one einsum over the axes' tables when the grid
    has at most ``64 / out_features`` cells per point, which spares every point's
    features and the scatter of their gradients. A sparser batch, and every other
    torch function, take the features point by point, :meth:`dense`.
    """

    def __init__(self, equation: str, operands: list[Operand], shape: tuple[int, ...]):
        self.equation, self.operands, self.shape = equation, operands, shape

    def dense(self) -> Tensor:
        """Return the features ``(N, prod(shape))``, contracted point by point."""
        return _fold(self.equation, self.operands).flatten(-len(self.shape))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.linear and isinstance(args[0], cls):
            features, weight, bias = (*args, kwargs.get("bias"))[:3]
            shared = [op for op in features.operands if isinstance(op, SharedRows)]
            cells = math.prod([len(op.table) for op in shared])  # a list, for Dynamo
            if cells <= 64 / weight.shape[0] * len(shared[0].rows):
                weight = weight.unflatten(-1, features.shape)
                out = _grid(features.equation, features.operands, weight)
                return out if bias is None else out + bias
        args = [arg.dense() if isinstance(arg, cls) else arg for arg in args]
        return func(*args, **kwargs)


class _Combiner(nn.Module):
    """Base class of the combiners: a tensor network over the axes' features.

    A subclass states its network once, as ``equation``, an einsum over
    ``operands(features)`` in which ``...`` stands for the points, and ``shape``, that of
    its output at a point. Per point, the operands are contracted in the order they are
    listed; on a grid, the features come back as :class:`GridFeatures`.
    """

    equation: str
    shape: tuple[int, ...]

    def __init__(self, in_features: Sequence[int]):
        super().__init__()
        self.in_features = list(in_features)

    @property
    def out_features(self) -> int:
        """Width of the features, ``R`` for CP."""
        return math.prod(self.shape)

    def forward(self, features: Sequence[Features]) -> Tensor | GridFeatures:
        operands = self.operands(features)
        if all(isinstance(f, SharedRows) for f in features):
            return GridFeatures(self.equation, operands, self.shape)
        return _fold(self.equation, operands).flatten(-len(self.shape))

    def operands(self, features: Sequence[Features]) -> list[Operand]:
        """Return the operands of ``equation`` for the features of every axis."""
        raise NotImplementedError


class CPCombiner(_Combiner):
    r"""Canonical polyadic (CP) combiner.

    Projects each axis to ``R`` channels with its factor
    :math:`\boldsymbol{U}^{(c)}`, ``factors[c]`` of shape ``(K_c, R)``, and multiplies
    the projections elementwise, ``...r,...r->...r`` (Eqs. 9 and 10 of the paper). A
    linear head's weight is then the CP model's output factor :math:`\boldsymbol{V}`.

    Args:
        in_features: Number of features ``K_c`` of each axis.
        rank: CP rank ``R``, the number of the network's channels.

    References:
        Kolda and Bader, "Tensor Decompositions and Applications", SIAM Review 2009.
    """

    def __init__(self, in_features: Sequence[int], rank: int):
        super().__init__(in_features)
        self.rank = rank
        self.factors = nn.ParameterList(torch.empty(k, rank) for k in in_features)
        self.equation = ",".join(["...r"] * len(self.factors)) + "->...r"
        self.reset_parameters()
        self.shape = (rank,)

    def reset_parameters(self) -> None:
        """Initialize the factors Kaiming-uniform, with fan-in ``K_c``."""
        for factor in self.factors:
            nn.init.kaiming_uniform_(factor.T)

    def operands(self, features: Sequence[Features]) -> list[Operand]:
        return [f @ u for f, u in zip(features, self.factors)]


class TRCombiner(_Combiner):
    r"""Tensor-ring (TR) combiner.

    Axis ``c`` contracts its features with a core ``(R, K_c, R)`` into an ``R x R``
    matrix, and the matrices multiply in order, ``...ab,...bc->...ac`` on two axes. The
    product is flattened, ``R**2`` wide, leaving the closure of the ring, such as a
    trace, to what follows. Extends the paper's CP model.

    Args:
        in_features: Number of features ``K_c`` of each axis.
        rank: Tensor-ring rank ``R``.

    References:
        Zhao et al., "Tensor Ring Decomposition", arXiv 2016.
    """

    def __init__(self, in_features: Sequence[int], rank: int):
        super().__init__(in_features)
        self.rank = rank
        self.cores = nn.ParameterList(torch.empty(rank, k, rank) for k in in_features)
        bonds = string.ascii_lowercase[: len(self.cores) + 1]  # axis c joins c, c + 1
        links = ",".join(f"...{a}{b}" for a, b in zip(bonds, bonds[1:]))
        self.equation = f"{links}->...{bonds[0]}{bonds[-1]}"
        self.reset_parameters()
        self.shape = (rank, rank)

    def reset_parameters(self) -> None:
        """Initialize the cores with Kaiming-uniform weights."""
        for core in self.cores:
            nn.init.kaiming_uniform_(core)

    def operands(self, features: Sequence[Features]) -> list[Operand]:
        # The product with a core folded to (K_c, R * R) is one matmul per axis.
        rank = self.rank
        return [
            (f @ core.transpose(0, 1).flatten(1)).unflatten(-1, (rank, rank))
            for f, core in zip(features, self.cores)
        ]


class TuckerCombiner(_Combiner):
    r"""Tucker combiner.

    Projects axis ``c`` to ``rank[c]`` channels with its factor ``(K_c, rank[c])`` and
    contracts the projections with a core ``(rank[0], ..., rank[C - 1], max(rank))``,
    such as ``...b,abz,...a->...z`` on two axes whose second rank is the larger. With a
    linear head the weight tensor is in Tucker format, and CP is the case of a
    superdiagonal core. Extends the paper's CP model.

    Args:
        in_features: Number of features ``K_c`` of each axis.
        rank: Multilinear rank, shared or per axis; the network has as many channels as
            its largest.

    References:
        Tucker, "Some mathematical notes on three-mode factor analysis",
        Psychometrika 1966.
    """

    def __init__(self, in_features: Sequence[int], rank: Sizes):
        super().__init__(in_features)
        modes = len(self.in_features)
        self.rank = _broadcast(rank, modes, "rank")
        widths = zip(self.in_features, self.rank)
        self.factors = nn.ParameterList(torch.empty(k, r) for k, r in widths)
        self.core = nn.Parameter(torch.empty(*self.rank, max(self.rank)))
        # Per point, the core absorbs one projection at a time, largest rank first,
        # which keeps the intermediates small; z is the core's output.
        self.order = sorted(range(modes), key=self.rank.__getitem__, reverse=True)
        ranks = string.ascii_lowercase[:modes]
        first, *others = (f"...{ranks[c]}" for c in self.order)
        self.equation = ",".join([first, f"{ranks}z", *others]) + "->...z"
        self.reset_parameters()
        self.shape = (max(self.rank),)

    def reset_parameters(self) -> None:
        """Initialize the factors (fan-in ``K_c``) and the core Kaiming-uniform."""
        for factor in self.factors:
            nn.init.kaiming_uniform_(factor.T)
        nn.init.kaiming_uniform_(self.core.view(-1, self.core.shape[-1]))

    def operands(self, features: Sequence[Features]) -> list[Operand]:
        projections = [f @ u for f, u in zip(features, self.factors)]
        first, *others = (projections[c] for c in self.order)
        return [first, self.core, *others]


# FUTON --------------------------------------------------------------------------------


class FUTONEncoding(nn.Module):
    """FUTON's encoding, ``combiner(basis(x))``, which :class:`FUTON` decodes.

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

    def __init__(self, in_features: int, basis: ModuleSpec, combiner: ModuleSpec):
        super().__init__()
        self.in_features = in_features
        self.basis = build_module(basis, BASES, in_features)
        self.combiner = build_module(combiner, COMBINERS, self.basis.num_components)
        self.out_features = self.combiner.out_features

    def forward(self, x: Tensor) -> Tensor:
        return self.combiner(self.basis(x))


class FUTON(nn.Module):
    r"""Fourier Tensor Network for implicit neural representations.

    Computes ``output_activation(head(combiner(basis(x))))``. Each stage is a spec:
    a module instance, a registry key, a module class, or a ``(key_or_class, kwargs)``
    pair. Registry keys:

    - Bases: ``"cosine"``, ``"sinc"``, ``"legendre"``, ``"chebyshev"``, ``"triangle"``,
      ``"lanczos"``.
    - Combiners: ``"cp"``, ``"tr"``, ``"tucker"``.
    - Heads: ``"linear"``, ``"mlp"``, ``"identity"``.

    Args:
        in_features: Number of coordinate axes ``C``.
        out_features: Number of output channels ``D``.
        basis: Basis spec; see :class:`FUTONEncoding`.
        combiner: Combiner spec; see :class:`FUTONEncoding`.
        head: Head spec, built as
            ``head(encoding.out_features, out_features, **kwargs)``.
        output_activation: Callable applied to the output; ``None`` uses the identity.

    Shape:
        - Input: ``(..., C)`` coordinates in ``[-1, 1]``.
        - Output: ``(..., D)``.

    Note:
        The cosine basis, CP combiner and linear head implement Eqs. (8) to (10) of the
        paper: ``encoding.combiner.factors[c]`` stores :math:`\boldsymbol{U}^{(c)}` and
        ``head.weight`` stores :math:`\boldsymbol{V}`. The other bases and combiners and
        the MLP head are extensions.

    Examples:
        >>> model = FUTON(
        ...     in_features=2,
        ...     out_features=3,
        ...     basis=("cosine", {"num_components": 256}),
        ...     combiner=("cp", {"rank": 256}),
        ...     head="linear",
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
        head: ModuleSpec,
        output_activation: Activation | None = None,
    ):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.encoding = FUTONEncoding(in_features, basis, combiner)
        width = self.encoding.out_features
        self.head = build_module(head, HEADS, width, out_features)
        self.output_activation = output_activation or nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        points = x.reshape(-1, self.in_features)  # flat, for the basis to share points
        out = self.output_activation(self.head(self.encoding(points)))
        return torch.reshape(out, (*x.shape[:-1], self.out_features))


# Registries ---------------------------------------------------------------------------

BASES = {
    "cosine": CosineBasis,
    "sinc": SincBasis,
    "legendre": LegendreBasis,
    "chebyshev": ChebyshevBasis,
    "triangle": TriangleBasis,
    "lanczos": LanczosBasis,
}
COMBINERS = {"cp": CPCombiner, "tr": TRCombiner, "tucker": TuckerCombiner}
HEADS = {"identity": nn.Identity, "linear": nn.Linear, "mlp": MLP}
