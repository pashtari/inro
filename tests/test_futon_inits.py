"""Tests for the starts of FUTON fitted to a signal."""

import pytest
import torch

import inro
from inro.models.futon_inits import ALSInit, SVDInit

DOUBLE = torch.float64


def grids(sizes):
    return [torch.linspace(-1, 1, n, dtype=DOUBLE) for n in sizes]


def separable(sizes, channels, seed=0):
    """Return a product of low cosines, one per axis, times a vector of the channels."""
    generator = torch.Generator().manual_seed(seed)
    cosine = inro.CosineBasis(len(sizes), num_components=list(sizes))
    profiles = [cosine.axis_features(g, c)[:, 1] for c, g in enumerate(grids(sizes))]
    modes = "abcd"[: len(sizes)]
    product = torch.einsum(f"{','.join(modes)}->{modes}", *profiles)
    return product[..., None] * torch.randn(channels, dtype=DOUBLE, generator=generator)


def futon(sizes, channels, rank, basis="sinc"):
    """Build a FUTON with a CP combiner and a linear head with bias, in double."""
    basis = (basis, {"num_components": list(sizes)})
    combiner = ("cp", {"rank": rank})
    return inro.FUTON(len(sizes), channels, basis, combiner, "linear").double()


def output(model, sizes):
    x = torch.stack(torch.meshgrid(*grids(sizes), indexing="ij"), -1)
    with torch.no_grad():
        return model(x)


@pytest.mark.parametrize("start", [SVDInit, ALSInit])
@pytest.mark.parametrize("basis", ["sinc", "cosine"])
@pytest.mark.parametrize(
    "sizes, channels", [((24,), 2), ((12, 10), 3), ((12, 10, 8), 1), ((8, 8, 8, 8), 2)]
)
def test_a_separable_low_frequency_signal_comes_back(start, basis, sizes, channels):
    """Start on the signal itself, times the scale, for any axes and channels."""
    target = separable(sizes, channels)
    model = futon(sizes, channels, rank=2, basis=basis)
    start(scale=1.5)(model, target)
    torch.testing.assert_close(output(model, sizes), 1.5 * target, atol=1e-9, rtol=0)


def test_components_beyond_the_terms_keep_their_draw():
    """Leave the components the terms do not fill silent, behind zero head columns."""
    sizes = (8, 8)  # two low cosines per axis: two terms
    model = futon(sizes, 1, rank=5)
    drawn = [factor.detach().clone() for factor in model.encoding.combiner.factors]
    target = separable(sizes, 1)
    SVDInit()(model, target)
    for factor, before in zip(model.encoding.combiner.factors, drawn):
        assert torch.equal(factor[:, 2:], before[:, 2:])
    assert torch.all(model.head.weight[:, 2:] == 0)
    torch.testing.assert_close(output(model, sizes), target, atol=1e-9, rtol=0)


def test_als_refines_the_svd_start():
    """Fit a signal of many terms closer after the sweeps than before them."""
    sizes, channels, rank = (16, 12, 10), 2, 3
    generator = torch.Generator().manual_seed(1)
    target = torch.randn(*sizes, channels, dtype=DOUBLE, generator=generator)
    errors = []
    for start in (SVDInit(cut=2), ALSInit(cut=2)):
        model = futon(sizes, channels, rank)
        start(model, target)
        errors.append(float((output(model, sizes) - target).square().mean()))
    assert errors[1] < errors[0]


@pytest.mark.parametrize(
    "combiner, head",
    [
        (("tucker", {"rank": 2}), "linear"),
        (("cp", {"rank": 2}), ("mlp", {"hidden_layers": 1})),
        (("cp", {"rank": 1}), "identity"),
    ],
)
def test_needs_a_cp_combiner_and_a_linear_head(combiner, head):
    basis = ("sinc", {"num_components": 8})
    model = inro.FUTON(2, 1, basis, combiner, head)
    with pytest.raises(ValueError, match="CP combiner and a linear head"):
        SVDInit()(model, torch.zeros(8, 8, 1))
