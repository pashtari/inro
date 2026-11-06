# Notebooks

Three notebooks in `notebooks/` run one signal end to end, one per task. They are the interactive version of the training scripts and the fastest way to see how the models compare on your own data.
{ .inro-lead }

Each compares FUTON against five representative baselines of the task at equal size: RFF, SIREN, FINER, Instant-NGP and TensoRF.

| Notebook | Signal | Models | Runtime on an A100 |
| --- | --- | --- | --- |
| `image_representation.ipynb` | `data/Kodak/kodim19.png` | 6, about 197k parameters each, 1000 steps on the whole image | about 6 minutes |
| `occupancy_volume.ipynb` | `data/occupancy/lucy.ply` at 512 voxels per unit length | 6, about 131k parameters each, 1000 steps of $2^{20}$ voxels | about 7 minutes |
| `nerf.ipynb` | `data/nerf/blender/lego` under FINER's protocol, in a quick pass | 6, about 75k parameters each, 5000 steps | about 9 minutes (set `NUM_STEPS = 25_000` for the benchmark's full schedule, about 25 minutes) |

## Run them

```bash
git clone https://github.com/pashtari/inro.git
cd inro
pip install -e ".[3d,nerf]" jupyterlab
python scripts/download_kodak.py --images 19
python scripts/download_meshes.py --meshes lucy
python scripts/download_blender.py --scenes lego
jupyter lab notebooks/
```

The notebooks read data from `../data/`, relative to their folder, and are committed with their outputs, so they read without being run; rerun them top to bottom. The NeRF notebook also renders an orbit of its best INR as a GIF with `imageio`, which the `nerf` extra provides.

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

FUTON is drawn in black, and the baselines in seaborn's colorblind palette. The model settings are those of `configs/<task>.yaml`, which holds every model of the benchmark, the notebooks' and the rest.
