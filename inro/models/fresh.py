"""FreSh: a SIREN's first-layer frequency chosen from the signal's spectrum.

References:
    Kania et al., "FreSh: Frequency Shifting for Accelerated Neural Representation
    Learning", ICLR 2025.
"""

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["fresh_distance", "fresh_select", "fresh_spectrum"]


def fresh_spectrum(signal: Tensor, size: int = 64) -> Tensor:
    """Compute the spectrum FreSh compares, as a distribution.

    As in the authors' code, the grid is first resized to a square or a cube of its
    smallest side, at most ``1024``: bilinearly with antialiasing in 2D, trilinearly in
    3D. The amplitudes of its discrete Fourier transform are summed over the channels,
    and those of nonnegative frequencies along shells of equal index sum ``d``.

    Args:
        signal: Values on a grid, ``(H, W, channels)`` or ``(depth, H, W, channels)``.
        size: Number of shells ``d = 1, ..., size``, which leave out the constant term,
            at most half the side after resizing.

    Returns:
        The shells divided by their sum, a float32 tensor ``(size,)`` that sums to one.

    Raises:
        ValueError: If the grid is neither 2D nor 3D, or ``size`` is too large.
    """
    grid = signal.detach().movedim(-1, 0).float()  # (channels, *sides)
    sides = grid.shape[1:]
    if len(sides) not in (2, 3):
        raise ValueError(f"expected a 2D or 3D grid, got {len(sides)} sides")
    side = min(min(sides), 1024)
    if any(length != side for length in sides):
        planar = len(sides) == 2
        grid = F.interpolate(
            grid[None],
            size=[side] * len(sides),
            mode="bilinear" if planar else "trilinear",
            align_corners=False,
            antialias=planar,
        )[0]
    if size > side // 2:
        raise ValueError(f"{size} shells reach past half the side, {side}")
    axes = tuple(range(1, grid.ndim))
    amplitude = torch.fft.fftn(grid, dim=axes).abs().sum(0)
    corner = amplitude[(slice(0, size + 1),) * len(sides)]
    steps = torch.arange(size + 1, device=corner.device)
    shell = sum(
        steps.view([-1 if axis == i else 1 for axis in range(len(sides))])
        for i in range(len(sides))
    )
    sums = torch.zeros(len(sides) * size + 1, device=corner.device)
    sums.scatter_add_(0, shell.flatten(), corner.flatten())
    spectrum = sums[1 : size + 1]
    return spectrum / spectrum.sum()


def fresh_distance(p: Tensor, q: Tensor) -> float:
    """Return the Wasserstein-1 distance of two spectra.

    Args:
        p: Spectrum ``(n,)`` on the points ``0, ..., n - 1``.
        q: Spectrum ``(n,)`` on the same points.
    """
    return float((p.cumsum(0) - q.cumsum(0))[:-1].abs().sum())


@torch.no_grad()
def fresh_select(
    targets: Sequence[Tensor],
    render: Callable[[float], Tensor],
    omegas: Sequence[float] = tuple(range(10, 201, 10)),
    size: int = 64,
    inits: int = 10,
    seed: int = 0,
) -> tuple[float, dict[float, float]]:
    """Choose a SIREN's first-layer frequency with FreSh.

    A candidate's score is the mean :func:`fresh_distance` between the
    :func:`fresh_spectrum` of its ``i``-th untrained output and that of
    ``targets[i % len(targets)]``.

    Args:
        targets: Signal on its grid, as :func:`fresh_spectrum` takes it, or several
            views or frames.
        render: Function that builds a new, untrained model whose first layer has the
            given frequency and returns its output on the signal's grid.
        omegas: Candidate frequencies.
        size: Number of spectrum shells compared.
        inits: Number of initializations averaged per candidate.
        seed: Seed of the global random state, set anew for every candidate.

    Returns:
        The candidate of least score, the smaller one on a tie, and every candidate's
        score.

    Examples:
        >>> from inro import SIREN, create_coordinates
        >>> x = create_coordinates((32, 32))
        >>> omega, scores = fresh_select(
        ...     [torch.rand(32, 32, 3)],
        ...     lambda omega: SIREN(2, 3, 16, first_omega=omega)(x),
        ...     omegas=(10, 20),
        ...     size=8,
        ...     inits=2,
        ... )
        >>> omega in scores
        True
    """
    goals = [fresh_spectrum(target, size) for target in targets]
    scores = {}
    for omega in omegas:
        torch.manual_seed(seed)
        distances = [
            fresh_distance(goals[i % len(goals)], fresh_spectrum(render(omega), size))
            for i in range(inits)
        ]
        scores[omega] = sum(distances) / inits
    return min(omegas, key=lambda omega: (scores[omega], omega)), scores
