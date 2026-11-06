"""Voxelization of meshes and point clouds into occupancy volumes."""

import numpy as np
import pytest
import torch

import inro

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("open3d")


def centers_inside_sphere(resolution: int) -> np.ndarray:
    """Occupancy of the voxel centers of a sphere normalized as the dataset does.

    The sphere is centered and scaled to radius 0.45; its centers lie at multiples of
    ``1 / resolution`` within ``[-0.45, 0.45]``.
    """
    axis = np.arange(-np.ceil(0.45 * resolution), np.ceil(0.45 * resolution) + 1)
    x, y, z = np.meshgrid(*3 * [axis / resolution], indexing="ij")
    return x**2 + y**2 + z**2 < 0.45**2


@pytest.fixture
def sphere(tmp_path):
    path = tmp_path / "sphere.ply"
    trimesh.creation.icosphere(subdivisions=6, radius=2.0).export(path)
    return path


def test_mesh_voxels_are_the_centers_inside(sphere):
    """No voxel the surface merely crosses is filled: only the centers inside."""
    resolution = 32
    volume = inro.OccupancyCoordinateDataset(sphere, resolution=resolution).original
    expected = centers_inside_sphere(resolution)
    # Crop the analytic sphere to its occupied bounds, as the dataset does.
    filled = [
        np.flatnonzero(expected.any(axis=tuple({0, 1, 2} - {a}))) for a in range(3)
    ]
    expected = expected[tuple(slice(f[0], f[-1] + 1) for f in filled)]
    assert volume.shape == (1, *expected.shape)
    # Centers within a facet's sag of the sphere may fall either way.
    mismatch = (volume[0].numpy() > 0) != expected
    assert mismatch.sum() <= 0.01 * expected.sum()


def test_volume_is_cached_compactly_and_reloads(sphere):
    first = inro.OccupancyCoordinateDataset(sphere, resolution=16)
    cache = sphere.parent / ".cache" / "sphere_16_centers.pt"
    stored = torch.load(cache, weights_only=True)
    assert stored.dtype == torch.uint8
    second = inro.OccupancyCoordinateDataset(sphere, resolution=16)
    assert second.original.dtype == torch.float32
    torch.testing.assert_close(first.original, second.original, rtol=0, atol=0)


def test_point_cloud_voxels_match_its_mesh(sphere, tmp_path):
    """A mesh's vertices with their normals give the occupancy of the mesh."""
    mesh = trimesh.load(sphere)
    cloud = tmp_path / "cloud.xyz"
    np.savetxt(cloud, np.column_stack([mesh.vertices, mesh.vertex_normals]))
    from_cloud = inro.OccupancyCoordinateDataset(cloud, resolution=24).original
    from_mesh = inro.OccupancyCoordinateDataset(sphere, resolution=24).original
    assert from_cloud.shape == from_mesh.shape
    assert inro.iou(from_cloud, from_mesh) > 0.97
