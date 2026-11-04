"""Block coordinate descent for FUTON: solvers in place of a gradient optimizer.

A FUTON with a CP combiner and a linear head, both without bias, is multilinear, so
fitting one block with the others held is a least-squares problem. :class:`BCD` sweeps
the blocks once per batch and leaves each block's update to a subclass:
:class:`LeastSquares` solves the block's problem, for any basis, and
:class:`MultiplicativeUpdate` takes Lee and Seung's step, which keeps a nonnegative
model nonnegative. :func:`inro.train` takes both in place of Adam, on the same datasets.
"""

from collections.abc import Mapping
from functools import reduce
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .futon import FUTON
from .futon_inits import _cp, _cp_parts, _mttkrp, _ridged

__all__ = ["BCD", "LeastSquares", "MultiplicativeUpdate"]


@torch.compiler.disable()  # sizes that depend on the data
def _distinct(x: Tensor) -> list[tuple[Tensor, Tensor]]:
    """Return each axis's distinct coordinates and every point's row among them."""
    return [torch.unique(u, return_inverse=True) for u in x.unbind(-1)]


class BCD(Optimizer):
    """Block coordinate descent over a FUTON's factor matrices and head.

    On a grid the model is a CP tensor, so with the other blocks held each block has
    small normal equations, which a subclass's :meth:`solve` handles. :meth:`step`
    updates the factors in order, then the head, and holds every axis's response at
    unit mean square, the head's columns taking the scales.

    A batch is taken as points of the grid the basis declares, or else of the grid of
    its distinct coordinates. Only the residual comes from its points, so a sweep costs
    about the same on any share of the grid, and a whole grid with ``momentum=0`` is the
    fast setting. On a whole grid at ``lr=1``, no update of
    :class:`MultiplicativeUpdate`, or of :class:`LeastSquares` without its ridge, raises
    the batch's error.

    Args:
        model: FUTON with a CP combiner and a linear head, both without bias, whose
            basis evaluates one axis at a time through ``axis_features``, as the
            built-in ones do. Updated in place.
        momentum: Weight of the past batches in a block's statistics, in ``[0, 1)``, as
            in online matrix factorization; ``0`` takes each batch on its own. The
            solver's ``lr``, the newest batch's weight, starts at ``1 - momentum``.
            Above ``0``, :func:`inro.train` anneals it to a hundredth, as it would a
            gradient step's; at ``0`` it stays one.
        initialize: Whether to draw the parameters anew with :meth:`reset_parameters`;
            ``False`` keeps the model's.
        **defaults: A subclass's settings, kept in ``self.defaults``.

    Raises:
        ValueError: If ``model`` is not such a FUTON or ``momentum`` is out of range.

    References:
        Mairal et al., "Online learning for matrix factorization and sparse coding",
        JMLR 2010.
    """

    def __init__(
        self,
        model: FUTON,
        momentum: float = 0.9,
        initialize: bool = True,
        **defaults: Any,
    ) -> None:
        factors, output, _ = _cp_parts(model)
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        # The blocks, the factors then the output factor, which the solver reads here
        # rather than from param_groups, whose lr changes at every step and would make
        # a compiled sweep recompile.
        self._blocks = [*factors, output]
        settings = {"lr": 1 - momentum, "initialize": initialize}
        super().__init__(self._blocks, settings | defaults)
        self.model = model
        self._sweep = self._sweep_eagerly
        if initialize:
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Draw the parameters as the solver wants them.

        A subclass draws the entries. Each factor is then divided by the sum of its
        basis's magnitudes in the middle of the domain, so that each axis's response
        there is at most about one, and the head by ``2 * R``, so that its components
        sum to about one half: the first sweep starts from an output of the signal's
        order.
        """
        basis: Any = self.model.encoding.basis  # any basis with axis_features
        *factors, weight = self._blocks
        for c, factor in enumerate(factors):
            # Magnitudes, since a signed basis's values can cancel: the cosines' sum to
            # zero when K_c is 0 or 3 modulo 4.
            middle = basis.axis_features(factor.new_zeros(1), c)
            factor.copy_(self._draw(factor) / middle.abs().sum())
        weight.copy_(self._draw(weight) / (2 * weight.shape[1]))

    def _draw(self, param: Tensor) -> Tensor:
        """Draw a parameter's entries anew, before scaling."""
        raise NotImplementedError

    def _left(self, table: Tensor) -> Tensor:
        """Return an axis's basis Gram matrix, as :meth:`solve` takes it."""
        return table.T @ table / len(table)

    def _check(self, x: Tensor, y: Tensor) -> None:
        """Check the first batch's points ``x`` and values ``y`` before its sweep.

        The base class has no checks.
        """

    def solve(
        self, value: Tensor, right: Tensor, gram: Tensor, left: Tensor | None = None
    ) -> Tensor:
        """Return a block's new value, from its problem ``left @ value @ gram = right``.

        Args:
            value: Current block, ``(K_c, R)`` for a factor and ``(D, R)`` for the
                head.
            right: Right-hand side, of the block's shape.
            gram: Elementwise product of the other blocks' Gram matrices, ``(R, R)``.
            left: Gram matrix of the basis, in the form the subclass prepares it;
                ``None`` is the identity, for the head.
        """
        raise NotImplementedError

    def compile(
        self, batch: Mapping[str, Tensor] | None = None, **options: Any
    ) -> None:
        """Compile the sweep with ``torch.compile``.

        Args:
            batch: Batch to compile on now, at its shapes, leaving the model and the
                solver as they were, so that no timed step pays for it; ``None``
                compiles at the first step.
            **options: Keyword arguments of ``torch.compile``.
        """
        self._sweep = torch.compile(self._sweep_eagerly, **options)
        if batch is not None:
            saved = [block.detach().clone() for block in self._blocks]
            self.step(batch)
            with torch.no_grad():
                for block, value in zip(self._blocks, saved):
                    block.copy_(value)
            self.state.clear()

    @torch.no_grad()
    def step(self, batch: Mapping[str, Tensor]) -> Tensor:  # type: ignore[override]
        """Run one sweep over the factor matrices and the head on a batch.

        Args:
            batch: ``input``, coordinates ``(..., C)`` in ``[-1, 1]``, and ``target``,
                their values ``(..., D)``, as a dataset yields them. The points must lie
                on a grid, as a coordinate dataset's do.

        Returns:
            The batch's mean squared error before the sweep.
        """
        x = batch["input"].reshape(-1, self.model.in_features)
        y = batch["target"].reshape(-1, self.model.out_features)
        if not self.state:
            self._start(x)
            self._check(x, y)
        elif not hasattr(self, "_lr"):  # the state came from load_state_dict
            self._lr = x.new_zeros(())
        self._lr.fill_(self.param_groups[0]["lr"])
        return self._sweep(x, y)

    def _start(self, x: Tensor) -> None:
        """Set every block's averages to zero before the first batch.

        On a declared grid the tables are fixed, so each axis's left-hand side is too,
        and is made once.
        """
        self._lr = x.new_zeros(())
        square = (self._blocks[-1].shape[1],) * 2
        for block in self._blocks:  # its averages of (right, gram)
            statistics = (torch.zeros_like(block), x.new_zeros(square))
            self.state[block] = {"total": x.new_zeros(()), "statistics": statistics}
        if getattr(self.model.encoding.basis, "grid_size", None) is not None:
            for factor, table in zip(self._blocks[:-1], self._grid(x)[0]):
                self.state[factor]["left"] = self._left(table)

    def _average(self, param: Tensor, *statistics: Tensor) -> tuple[Tensor, ...]:
        """Return a block's statistics, averaged over the batches seen."""
        state = self.state[param]
        # The batches' weights sum to one, whatever the step size was at each.
        state["total"].mul_(1 - self._lr).add_(self._lr)
        weight = self._lr / state["total"]
        for average, new in zip(state["statistics"], statistics):
            average.lerp_(new, weight)
        return state["statistics"]

    def _grid(self, x: Tensor) -> tuple[list[Tensor], tuple[Tensor, ...]]:
        """Return each axis's table and every point's row in it.

        A table covers the grid the basis declares, or else the batch's distinct
        coordinates.
        """
        basis: Any = self.model.encoding.basis  # any basis with axis_features
        if getattr(basis, "grid_size", None) is not None:
            features = basis(x)
            return [f.table for f in features], tuple(f.rows for f in features)
        distinct = _distinct(x)
        tables = [basis.axis_features(u, c) for c, (u, _) in enumerate(distinct)]
        return tables, tuple(rows for _, rows in distinct)

    def _targets(self, y: Tensor, output: Tensor) -> Tensor:
        """Return what the head's output must fit for the model's to approach ``y``.

        Without an output activation, ``y`` itself. With one, the Gauss-Newton target
        from the current ``output``: the output plus the residual over the activation's
        slope, with the squared slopes bounded by their largest so that every block
        stays an unweighted least-squares problem. It is exact for an affine activation.
        The activation acts elementwise, so its slopes are one forward derivative.
        """
        activation = self.model.output_activation
        if isinstance(activation, nn.Identity):
            return y
        value, slope = torch.func.jvp(
            activation, (output,), (torch.ones_like(output),)
        )[:2]
        return output + slope * (y - value) / slope.square().max()

    def _sweep_eagerly(self, x: Tensor, y: Tensor) -> Tensor:
        count = len(x)
        tables, rows = self._grid(x)
        *factors, weight = self._blocks
        # On the grid the output is a CP tensor, whose factors are the axes'
        # responses and the head; the points pick their entries.
        modes = [table @ factor for table, factor in zip(tables, factors)] + [weight]
        grams = [m.T @ m / len(m) for m in modes[:-1]]
        output = _cp(modes)[rows]
        loss = (self.model.output_activation(output) - y).square().mean()
        targets = self._targets(y, output)

        def missed(output: Tensor) -> Tensor:
            """Sum what the model misses at the points into the grid."""
            grid = y.new_zeros(*(len(table) for table in tables), y.shape[1])
            return grid.index_put_(rows, targets - output, accumulate=True)

        error = missed(output)
        for c, (table, factor) in enumerate(zip(tables, factors)):
            gram = reduce(torch.mul, grams[:c] + grams[c + 1 :], weight.T @ weight)
            # What the model explains, from the Gram matrices, and what it misses, from
            # the batch: exact on a whole grid, unbiased on a random share.
            explained = modes[c] @ gram / len(table)
            right = table.T @ (explained + _mttkrp(error, modes, c) / count)
            right, gram = self._average(factor, right, gram)
            left = self.state[factor].get("left")  # made once on a declared grid
            if left is None:
                left = self._left(table)
            value = self.solve(factor, right, gram, left)
            # A component scaled in a factor and back in the head leaves
            # the model as it was: unit mean square responses fix the scales.
            response = table @ value
            scale = response.square().mean(0).sqrt()
            scale = torch.where(scale > 0, scale, 1.0)
            factor.copy_(value / scale)
            weight.mul_(scale)
            modes[c] = response / scale
            grams[c] = modes[c].T @ modes[c] / len(table)
            error = missed(_cp(modes)[rows])
        gram = reduce(torch.mul, grams)
        right = weight @ gram + _mttkrp(error, modes, len(tables)) / count
        right, gram = self._average(weight, right, gram)
        weight.copy_(self.solve(weight, right, gram))
        return loss


class LeastSquares(BCD):
    """Alternating least squares: every block takes its problem's solution.

    A block's problem is solved one side at a time, each with a ridge against collinear
    functions or components, for any basis, signed or not. With an output activation the
    targets are Gauss-Newton ones, refitted every sweep, so the sweeps descend the
    activated model's error rather than minimize a fixed one. The parameters start from
    normal draws.

    Args:
        model: See :class:`BCD`.
        ridge: Added to the diagonal of each Gram matrix, relative to its mean diagonal.
        momentum: See :class:`BCD`.
        initialize: See :class:`BCD`.

    Examples:
        >>> model = FUTON(
        ...     in_features=2,
        ...     out_features=3,
        ...     basis=("sinc", {"num_components": 16}),
        ...     combiner=("cp", {"rank": 8}),
        ...     head=("linear", {"bias": False}),
        ... )
        >>> solver = LeastSquares(model, momentum=0)
        >>> axis = torch.linspace(-1, 1, 32)
        >>> x = torch.cartesian_prod(axis, axis)
        >>> loss = solver.step({"input": x, "target": torch.rand(len(x), 3)})
        >>> loss.shape
        torch.Size([])
    """

    def __init__(
        self,
        model: FUTON,
        ridge: float = 1e-3,
        momentum: float = 0.9,
        initialize: bool = True,
    ) -> None:
        super().__init__(model, momentum, initialize, ridge=ridge)

    def _draw(self, param: Tensor) -> Tensor:
        return torch.randn_like(param)

    def _left(self, table: Tensor) -> Tensor:
        gram = _ridged(super()._left(table), self.defaults["ridge"])
        return torch.cholesky_inverse(torch.linalg.cholesky_ex(gram).L)

    def solve(
        self, value: Tensor, right: Tensor, gram: Tensor, left: Tensor | None = None
    ) -> Tensor:
        """Return the block's ridged solution; see :meth:`BCD.solve`."""
        if left is not None:
            right = left @ right
        # The Gram matrix is symmetric, so the right solve is a left one transposed.
        factor = torch.linalg.cholesky_ex(_ridged(gram, self.defaults["ridge"])).L
        return torch.cholesky_solve(right.T, factor).T


class MultiplicativeUpdate(BCD):
    """Lee and Seung's multiplicative updates, for a nonnegative model.

    The update multiplies a block elementwise by the ratio of its gradient's two parts,
    ``value * right / (left @ value @ gram)``. It never worsens the block's problem and
    keeps the block nonnegative; each block takes ``inner`` of them in a row, as Gillis
    and Glineur propose.

    The first :meth:`step` checks that the basis, the targets and the parameters are
    nonnegative. Use the triangle basis with ``normalize=False``, whose functions sum to
    one, and fit a signal in ``[-1, 1]`` through the affine output activation
    ``lambda z: 2 * z - 1``, under which its targets are in ``[0, 1]``. The parameters
    start uniformly in ``[0.5, 1.5]``, before scaling.

    Args:
        model: See :class:`BCD`.
        inner: Number of updates of each block per sweep.
        floor: Lower bound on a denominator, against division by zero where a basis
            function sees no sample.
        momentum: See :class:`BCD`.
        initialize: See :class:`BCD`.

    References:
        Lee and Seung, "Algorithms for non-negative matrix factorization", NeurIPS 2001.
        Gillis and Glineur, "Accelerated multiplicative updates and hierarchical ALS
        algorithms for nonnegative matrix factorization", Neural Computation 2012.
    """

    def __init__(
        self,
        model: FUTON,
        inner: int = 16,
        floor: float = 1e-12,
        momentum: float = 0.9,
        initialize: bool = True,
    ) -> None:
        if inner < 1:
            raise ValueError("inner must be at least 1")
        super().__init__(model, momentum, initialize, inner=inner, floor=floor)

    def _draw(self, param: Tensor) -> Tensor:
        return torch.rand_like(param) + 0.5

    def _check(self, x: Tensor, y: Tensor) -> None:
        tables, rows = self._grid(x)
        *factors, weight = self._blocks
        output = _cp([t @ a for t, a in zip(tables, factors)] + [weight])[rows]
        # The targets of an affine activation are exact up to a few roundings,
        # which may leave a zero slightly negative.
        lowest = [*tables, *self._blocks, self._targets(y, output) + 1e-5]
        if torch.stack([t.amin() for t in lowest]).amin() < 0:
            raise ValueError(
                "multiplicative updates need a nonnegative basis, nonnegative"
                " targets (an affine output activation, 2 z - 1, for a signal in"
                " [-1, 1]) and nonnegative parameters"
            )

    def solve(
        self, value: Tensor, right: Tensor, gram: Tensor, left: Tensor | None = None
    ) -> Tensor:
        """Return the block after ``inner`` updates; see :meth:`BCD.solve`."""
        floor = self.defaults["floor"]
        for _ in range(self.defaults["inner"]):
            denominator = value @ gram if left is None else left @ value @ gram
            # A coefficient no sample of the batch reaches has both parts at
            # zero and keeps its value; one whose right-hand side is negative,
            # which a minibatch's estimate can be, goes to zero, the
            # constrained minimizer of its auxiliary function.
            ratio = (right / denominator.clamp_min(floor)).clamp_min(0)
            value = value * torch.where(denominator > floor, ratio, 1.0)
        return value
