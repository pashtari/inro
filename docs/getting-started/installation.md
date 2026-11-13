# Installation

Python 3.11 or newer and PyTorch 2.10 or newer, with or without CUDA. Everything runs on the CPU; a GPU makes training practical.
{ .inro-lead }

## Install

=== "From PyPI"

    ```bash
    pip install inro
    ```

=== "With the extras"

    ```bash
    pip install "inro[3d,nerf]"
    ```

=== "From source"

    For the training scripts, configs and notebooks, which live in the repository:

    ```bash
    git clone https://github.com/pashtari/inro.git
    cd inro
    pip install -e ".[3d,nerf,dev]"
    ```

Check the install:

```python
import torch, inro

print(inro.__version__, torch.cuda.is_available())
```

!!! note "PyTorch and the NVIDIA driver"

    `pip install inro` brings PyTorch's default build from PyPI, which is made for a recent CUDA release and uses the GPU only with a driver that supports it. If the check prints `False` on a machine with a GPU, install the PyTorch build that matches the driver first, as [pytorch.org](https://pytorch.org/get-started/locally/) shows, and inro after it:

    ```bash
    nvidia-smi  # the newest CUDA version the driver supports, top right
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  # drivers for CUDA 12.6 or newer
    pip install inro
    ```

## Extras

| Extra | Installs | Needed for |
| --- | --- | --- |
| `3d` | PyMCubes, pycollada, trimesh, Open3D | Meshes as occupancy volumes, and exporting and rendering reconstructed meshes |
| `nerf` | nerfacc, imageio | CUDA occupancy-grid ray marching, and writing novel-view videos |
| `dev` | black, pytest, pyinstrument | Formatting, the test suite and profiling |
| `docs` | mkdocs-material, mkdocs-glightbox, mkdocstrings | Building this site |

!!! note "nerfacc and the CUDA toolkit"

    **nerfacc** compiles a CUDA extension the first time it is used and needs a CUDA toolkit for that. Without one, `inro.nerf.is_nerfacc_available()` is `False` and `inro.nerf.create_renderer()` falls back to the built-in PyTorch renderer, which produces the same images more slowly.

## Benchmark data

The experiments use three public datasets. The scripts, in the repository, download them into `data/`, which is ignored by git.

| Dataset | Script | Content | Size | Subset |
| --- | --- | --- | --- | --- |
| Kodak | `scripts/download_kodak.py` | 24 images, 768 × 512 | about 30 MB | `--images 1 17 23` |
| Stanford | `scripts/download_meshes.py` | 5 meshes | about 3 GB | `--meshes armadillo thai_statue` |
| Blender | `scripts/download_blender.py` | 8 NeRF synthetic scenes | about 2.4 GB | `--scenes lego ship` |

Each script also takes `--output-dir` and `--force`. Interrupted downloads resume, and existing files are kept. The layout is what the training scripts and notebooks expect:

```text
data/
├── kodak/kodim01.png ... kodim24.png
├── meshes/armadillo.ply  dragon.ply  happy_buddha.ply  lucy.ply  thai_statue.ply
└── blender/<scene>/transforms_{train,val,test}.json + images
```

Occupancy volumes are voxelized on first use and cached beside the mesh in `data/meshes/.cache/<name>_<resolution>_centers.pt`.

## Development

From a clone of the repository:

```bash
pip install -e ".[dev]"
black --check inro tests scripts  # formatting, line length 88
pytest -q                         # tests that need CUDA or local data skip themselves
```
