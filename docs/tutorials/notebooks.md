# Notebooks

Five notebooks in `notebooks/` run one signal end to end. They are the interactive version of the training scripts and the fastest way to see how the models compare on your own data.
{ .inro-lead }

Four compare FUTON (cosine) against five representative baselines of a task at equal size: RFF, SIREN, FINER, Instant-NGP and TensoRF, or in super-resolution the last four and bicubic interpolation. The fifth compares the ways to fit FUTON.

| Notebook | Signal | Models | Runtime on an A100 |
| --- | --- | --- | --- |
| `image_representation.ipynb` | `data/Kodak/kodim19.png` | 7, about 197k parameters each, 1000 steps on the whole image | about 6 minutes |
| `occupancy_volume.ipynb` | `data/occupancy/lucy.ply` at $256^3$ | 7, about 131k parameters each, 1000 steps of $2^{18}$ voxels | about 4 minutes |
| `nerf.ipynb` | `data/nerf/blender/lego` under FINER's protocol, in a quick pass | 7, about 75k parameters each, 5000 steps | about 11 minutes (set `NUM_STEPS = 25_000` for the benchmark's full schedule, about 35 minutes) |
| `image_super_resolution.ipynb` | `data/DIV2K/DIV2K_valid_HR/0887.png` from its 4× bicubic downsampling | 6 INRs, about 500k parameters each, 1000 steps of 135 blocks, and bicubic interpolation | about 6 minutes |
| `futon_solvers.ipynb` | `data/Kodak/kodim19.png` | 4 FUTONs, about 197k parameters each: an MLP or a linear head, fitted by Adam (1000 steps on the whole image) or by alternating least squares or multiplicative updates (500 sweeps) | about 3 minutes |

## Run them

```bash
git clone https://github.com/pashtari/inro.git
cd inro
pip install -e ".[3d,nerf]" jupyterlab
python scripts/download_kodak.py --images 19
python scripts/download_meshes.py --meshes lucy
python scripts/download_blender.py --scenes lego
python scripts/download_div2k.py
jupyter lab notebooks/
```

The notebooks read data from `../data/`, relative to their folder, and are committed with the outputs of the runs timed above, so they read without being run; rerun them top to bottom. The NeRF notebook also renders an orbit of its best INR as a GIF with `imageio`, which the `nerf` extra provides.

## What is inside

Each notebook is self-contained and has the same sections:

| Section | Content |
| --- | --- |
| *Setup* | Imports the packages, sets the seaborn theme and each model's color and marker, and defines the plotting helpers |
| *Data* | Shows the signal and the region the comparison magnifies |
| *Models* | Every model's constructor arguments and learning rate, with `# deviation` comments where a setting departs from the authors' code |
| *Training* | Runs `inro.train` or `inro.nerf.train` |
| *Convergence* | Plots the metric against steps and against training time |
| *Summary* | Tabulates every model's size, costs and scores, the best of each column in bold and the second underlined, and plots the scores against the costs |
| *Qualitative Comparison* | Sets the reconstructions side by side, magnified, above their errors |

FUTON is drawn in black, its cosine basis dashed, and the baselines in seaborn's colorblind palette. The model settings are those of `configs/<task>.yaml`, which holds every model of the benchmark, the notebooks' and the rest.

The super-resolution notebook also walks through its forward model, which `inro.resize_matrix`, `inro.SuperResolutionDataset` and `inro.super_resolution_loss` provide: the models never see the high-resolution image, and each step renders the pixels under random blocks of low-resolution pixels, downsamples them with the benchmark's bicubic kernel, and matches the low-resolution pixels.
