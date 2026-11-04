"""Smoke tests for :mod:`inro.nerf`: construction and the protocol's constants.

Every backbone must build and run a forward pass, and the marching and dataset
constants the protocol is pinned to must not drift.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import yaml

import inro

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/nerf/blender/lego"

GEOMETRY_FEATURES = 15
COLOR_NET = ("mlp", {"hidden_features": 64, "hidden_layers": 2})
MLP_CONFIG = {"hidden_features": 182, "hidden_layers": 3}
CONFIG = ROOT / "configs/nerf.yaml"

sys.path.insert(0, str(ROOT / "scripts"))
from train_common import build  # noqa: E402  # the configs' loader, as the scripts'

# The density networks the benchmark runs, as its config gives them.
DENSITY_NETS = {
    name: build(model)
    for name, model in yaml.safe_load(CONFIG.read_text())["models"].items()
}
# Other bases at FUTON's size: the sinc of the basis ablation, and local bases, which
# take their taps along the rays.
FUTON_CLASS, FUTON_KWARGS = DENSITY_NETS["FUTON"]
DENSITY_NETS |= {
    "FUTON-sinc": (
        FUTON_CLASS,
        {**FUTON_KWARGS, "basis": ("sinc", {"num_components": 100, "grid_size": 2048})},
    ),
    "FUTON-lanczos": (
        FUTON_CLASS,
        {**FUTON_KWARGS, "basis": ("lanczos", {"num_components": 100, "radius": 3})},
    ),
    "FUTON-triangle": (
        FUTON_CLASS,
        {
            **FUTON_KWARGS,
            "basis": ("triangle", {"num_components": 100, "normalize": False}),
        },
    ),
}


def _field(density_net, **kwargs):
    if isinstance(density_net, tuple):
        cls, kw = density_net
        density_net = (cls, {"output_activation": torch.nn.Identity(), **kw})
    kwargs.setdefault("color_net", COLOR_NET)
    kwargs.setdefault("geometry_features", GEOMETRY_FEATURES)
    return inro.nerf.RadianceField(density_net=density_net, **kwargs)


@pytest.mark.parametrize("name", sorted(DENSITY_NETS))
def test_backbone_builds_and_runs(name):
    """Return finite ``(rgb, sigma)`` from every backbone."""
    torch.manual_seed(0)
    field = _field(DENSITY_NETS[name])

    x = torch.rand(8, 3) * 2.0 - 1.0
    d = torch.nn.functional.normalize(torch.randn(8, 3), dim=-1)
    rgb, sigma = field(x, d)

    assert rgb.shape == (8, 3) and sigma.shape == (8,)
    assert torch.isfinite(rgb).all() and torch.isfinite(sigma).all()
    assert (rgb >= 0).all() and (rgb <= 1).all() and (sigma >= 0).all()


@pytest.mark.parametrize("name", sorted(DENSITY_NETS))
def test_backbone_parameter_matched(name):
    """Keep the rows parameter-matched, as a fair comparison needs."""
    torch.manual_seed(0)
    num_params = inro.count_parameters(_field(DENSITY_NETS[name]))
    assert 73_000 <= num_params <= 76_000, f"{name}: {num_params:,} params"


def test_finer_reproduction_parameter_count():
    """Count the 201,842 parameters of ``network_finer.py`` for FINER end to end."""
    torch.manual_seed(0)
    field = _field(
        (inro.FINER, {**MLP_CONFIG, "omega": 30.0}),
        color_net=(
            inro.FINER,
            {**MLP_CONFIG, "omega": 30.0, "output_activation": torch.nn.Identity()},
        ),
        geometry_features=182,
        direction_encoding="identity",
    )
    assert inro.count_parameters(field) == 201_842


def test_density_net_logits_are_unbounded():
    """Leave density logits unbounded: a tanh would cap the density at ``e``."""
    field = _field(
        (inro.SIREN, {**MLP_CONFIG, "output_activation": torch.nn.Identity()})
    )
    with torch.no_grad():
        for p in field.density_net.parameters():
            p.mul_(50)
        logits = field.density_net(torch.rand(256, 3) * 2 - 1)
    assert logits.abs().max() > 1.0, "density logits must not be tanh-bounded"


def test_renderer_marching_constants():
    """Pin ray marching to torch-ngp's ``--max_steps 1024`` over the unit box."""
    renderer = inro.nerf.create_renderer(
        backend="pytorch", aabb=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0), near=0.2, far=10.0
    )
    assert renderer.num_marching_steps == 1024
    assert renderer.render_step_size == 2.0 * math.sqrt(3.0) / 1024


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="Blender lego scene not present")
def test_blender_train_split_and_radius():
    """Follow FINER's ``--trainskip 4 --downscale 4 --scale 0.8``: 25 views."""
    dataset = inro.nerf.BlenderDataset(DATA_ROOT, split="train", downsample=4, skip=4)
    assert len(dataset) == 25
    assert dataset.images.shape[1:3] == (200, 200)
    assert dataset.radius == pytest.approx(3.224903, abs=1e-6)


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="Blender lego scene not present")
def test_train_loop_runs_and_loss_is_finite(tmp_path):
    """Restore the occupancy grid a run was scored with, after a few steps."""
    torch.manual_seed(0)
    dataset = inro.nerf.BlenderDataset(DATA_ROOT, split="train", downsample=4, skip=4)
    field = _field(DENSITY_NETS["FUTON-lanczos"], aabb=dataset.aabb)
    renderer = inro.nerf.create_renderer(
        backend="pytorch", aabb=dataset.aabb, near=dataset.near, far=dataset.far
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    result = inro.nerf.train(
        field,
        renderer,
        dataset,
        device=device,
        num_steps=5,
        num_rays=256,
        log_interval=5,
        eval_interval=0,
        log_dir=tmp_path,
    )

    assert result["config"]["skipped_steps"] == 0
    assert (result["config"]["peak_memory"] > 0) == (device == "cuda")
    assert math.isfinite(result["history"][-1]["loss"])
    restored = inro.nerf.create_renderer(
        backend="pytorch", aabb=dataset.aabb, near=dataset.near, far=dataset.far
    )
    restored.load_state_dict(torch.load(tmp_path / "renderer.pt"))
    assert not restored.binaries.all()  # the warmup update pruned empty cells
    assert torch.equal(restored.binaries, renderer.binaries.cpu())
    assert torch.equal(restored.density_grid, renderer.density_grid.cpu())
