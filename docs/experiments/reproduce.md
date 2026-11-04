# Reproducing the paper

Every number of the paper comes from the scripts in `scripts/`. This page is the full pipeline, from an empty `data/` folder to the runs in `logs/`.
{ .inro-lead }

The scripts and configs live in the repository, so start from a clone: `git clone https://github.com/pashtari/inro.git`, then `pip install -e ".[3d,nerf]"`.

<ol class="inro-flow">
  <li><strong>1. Data</strong><code>scripts/download_*.py</code> &rarr; <code>data/</code></li>
  <li><strong>2. Training</strong><code>scripts/train_*.py</code> &rarr; <code>logs/</code></li>
  <li><strong>3. Ablations</strong><code>configs/ablation-futon/</code> &rarr; <code>logs/</code></li>
  <li><strong>4. Analyses</strong><code>scripts/futon_*.py</code> &rarr; <code>logs/</code></li>
  <li><strong>5. Results</strong><code>logs/**/results.json</code></li>
</ol>

## 1. Data

```bash
python scripts/download_kodak.py      # data/Kodak/, 24 images, about 30 MB
python scripts/download_meshes.py     # data/occupancy/, 5 meshes, about 3 GB
python scripts/download_blender.py    # data/nerf/blender/, 8 scenes, about 2.4 GB
```

The Kodak images come from Rich Franzen's mirror, the meshes from the Stanford 3D Scanning Repository (rotated into canonical pose where a release is not), and the Blender scenes from a pinned Hugging Face mirror of the official Google Drive release. All three scripts resume and keep existing files.

## 2. Training

`scripts/train_<task>.py` fits every model of `configs/<task>.yaml` to every signal of the task:

```bash
python scripts/train_image.py --compile            # 24 images x 13 models -> logs/image/<image>/<model>/
python scripts/train_occupancy.py --compile        # 5 shapes x 14 models  -> logs/occupancy/<shape>/<model>/
python scripts/train_nerf.py --compile             # 8 scenes x 14 models  -> logs/nerf/<scene>/<model>/
```

The three scripts share one command line:

| Option | Meaning |
| --- | --- |
| `--data PATH ...` | Signals to fit; default: every signal under `data/` for the task |
| `--models NAME ...` | Model names from the config; default: all |
| `--config FILE ...` | Configs, deep-merged in order; the last names the log directory (`configs/<name>.yaml` writes to `logs/<name>/`) |
| `--set KEY.PATH=VALUE ...` | Override one value, read as YAML, e.g. `train.num_steps=500` or `models.SIREN.train.lr=3e-4` |
| `--log-dir DIR` | Where the runs go |
| `--overwrite` | Rerun runs that already finished |
| `--compile` | Compile each model with `torch.compile` before its warm-up pass, as every run of the paper was |
| `--device DEV` | `cuda` when available, else `cpu` |

A config holds the shared `data` and `train` settings and, per model, its Inro class, constructor `kwargs` and training overrides (the learning rate). Image configs may use expressions in the image height `H` and width `W`, such as `[H // 2, W // 2]`, and the occupancy config in the voxel grid's `D`, `H` and `W`, where FUTON's entry names a parameter `budget` whose largest CP rank the script fills in. Each config's header says how its learning rates were chosen, and `# deviation` marks a setting that departs from the authors' published ones.

Finished runs are skipped, so an interrupted sweep restarts where it stopped, and a failed run writes `error.txt` and lets the sweep continue. Runs are keyed by model name, so a variant needs its own name or its own `--log-dir`:

```bash
python scripts/train_image.py --models Gauss --overwrite                       # rerun one model in place
python scripts/train_image.py --set models.SIREN.train.lr=0.001 --log-dir logs/siren-lr
python scripts/train_image.py --config configs/ablation-futon/basis.yaml       # -> logs/ablation-futon/basis/
```

??? note "What a run directory holds"

    `log.txt`, `log.json`, `checkpoint.pt`, the final reconstruction (images) or two rendered test views (NeRF), and `results.json` with the training configuration, the model's `setup` from the YAML, `num_params`, `train_time`, the final `metrics` (NeRF: the mean over the 200 test views, with every view under `test_views`), the `history`, and the `provenance`: the commit, PyTorch version and GPU of the run. The commits are those of the development history, which the release squashes; every run's configuration is the released config's. Occupancy runs write no mesh, since rendering needs EGL and a GPU driver.

Measured on A100s (1000 steps; NeRF 25,000), each signal's models in one job:

| Task | Per model | Per signal | Whole sweep |
| --- | --- | --- | --- |
| Images | 56 s | 13 min | 5.2 h |
| Occupancy | 27 s | 7 min | 34 min |
| NeRF | 5 min | 77 min | 10 h |

A run's time includes compiling its model and its 20 evaluations, with LPIPS on images; training alone takes from 3 s (Instant-NGP) to a minute (MFN) on an image. Every run uses seed 0; some GPU kernels accumulate in a nondeterministic order, so a rerun on the same hardware lands within about a tenth of a decibel of a run, and training times vary with the machine and its load.

## 3. Ablations

The FUTON ablations are configs in `configs/ablation-futon/`, run through the image script. All keep the benchmark's settings (the 24 Kodak images, 1000 steps on the whole image), so only the model changes, and all but the basis study run the cosine basis alone:

| Config | Models | Varies |
| --- | --- | --- |
| `basis.yaml` | 6 | The six bases at the benchmark's size |
| `bandwidth.yaml` | 5 | The relative bandwidth $\kappa = K / N$ over {1/8, 1/4, 1/2, 1, 2}, each at the largest rank, a multiple of 5, within the budget |
| `rank.yaml` | 5 | The CP rank over {70, 140, 280, 560, 1120} at $\kappa = 1/2$, 49,539 to 788,739 parameters |
| `tensor_net.yaml` | 3 | Tensor-ring and Tucker combiners against CP |
| `head.yaml` | 4 | MLP heads 64, 128 and 256 wide and the linear head, at equal size |

```bash
for study in basis bandwidth rank tensor_net head; do
  python scripts/train_image.py --config configs/ablation-futon/$study.yaml --compile
done
```

??? note "Runs that are shared between experiments"

    Runs that another experiment already holds are copied rather than repeated, each copy's `results.json` naming its `source`: the default model of the bandwidth and rank studies, for one, is the benchmark's FUTON.

## 4. Analyses

Two scripts study FUTON itself, from the signals rather than from trained runs:

```bash
python scripts/futon_coefficients.py  # kodim23's coefficient tensor at K = 32, in full and at rank 4 -> logs/coefficients/
python scripts/futon_bound.py         # the error bound against Adam, on the 24 images and 5 volumes -> logs/bound/
```

The first computes the tensor of the [FUTON page](../concepts/futon.md#from-a-series-to-a-network)'s figure in seconds on a CPU, without training: the image's generalized Fourier coefficients in full, and the certified FUTON at each rank (`--image`, `--components` and `--ranks` choose others). The second runs the [error bound](../concepts/error-bound.md#experiment)'s experiment, which trains each of its settings once with Adam to compare.

## 5. Results

Every run's `results.json` holds its metrics and costs, so the runs can be read directly:

```python
import json
from pathlib import Path
import pandas as pd

records = [json.loads(p.read_text()) for p in Path("logs").rglob("results.json")]
table = pd.json_normalize(records)
```

The tables aggregate them as follows:

- **Errors.** The `±` in the tables is a paired standard error over the signals: each signal's own level is removed before the error is taken, which is the uncertainty of a comparison between models rather than of an absolute level; IoU, which saturates near 100 % on every shape, takes the plain standard error over the shapes. A band in a figure spans one of them, the same error.
- **Costs.** They come from the benchmark runs themselves: the training time counts the training steps alone, after a warm-up pass that compiles the model, and every evaluation records how long it took and its peak memory. Each cost in the tables is the median over a task's signals, so a run slowed by other work on its GPU does not move it.
