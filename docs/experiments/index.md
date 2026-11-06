# Experiments

The paper's experiments are one protocol applied to three tasks, documented in enough detail to rerun them.
{ .inro-lead }

<div class="grid cards" markdown>

-   :material-trophy-outline:{ .lg .middle } **[Benchmarks](benchmarks.md)**

    ---

    Every model's size, costs and final metrics on all three tasks, with convergence plots and qualitative comparisons.

-   :material-tune-variant:{ .lg .middle } **[Ablations](ablations.md)**

    ---

    FUTON's basis, bandwidth, rank, combiner and head, varied one at a time on images.

-   :material-replay:{ .lg .middle } **[Reproducing the paper](reproduce.md)**

    ---

    The data, the training scripts and their configs, and the run directories that hold every result.

</div>

## The protocol at a glance

| Task | Signals | Budget | Steps | One step sees | Schedule | Scored on |
| --- | --- | --- | --- | --- | --- | --- |
| Images | 24 Kodak images | about 197k | 1000 | the whole image | cosine, to a hundredth | all pixels |
| Occupancy | 5 Stanford shapes | about 131k | 1000 | $2^{20}$ random voxels | cosine, to a hundredth | all voxels |
| Radiance fields | 8 Blender scenes | about 74k, its color network included | 25,000 | 4096 rays | exponential, to a tenth (torch-ngp's recipe) | all 200 test views |

Every model of a task follows the same rules:

- **Hyperparameters.** Those of its authors' code, wherever the budget allows them.
- **FUTON's size.** Its components per axis and head were chosen per task in a sweep of both, on the 24 Kodak images, the five shapes at 256 voxels per unit length and the validation views of the eight scenes: half the resolution and 64 units on images and radiance fields, where FUTON reads its basis from a table of 2048 points per axis, and half the resolution and a linear head on volumes. Its CP rank is the largest multiple of 5 that keeps it smaller than every baseline. The [bandwidth](ablations.md#bandwidth) and [head](ablations.md#head) ablations show the choice on images.
- **Learning rate.** From the grid {3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4}, chosen on two signals of the task; on volumes, a rate that leaves some shape empty, every voxel outside, is passed over.
- **Optimizer.** PyTorch's default Adam.
- **Loss.** The mean squared error, for every model.
- **Run.** Seed 0 on an NVIDIA A100, the model compiled with `torch.compile` in a warm-up pass that keeps compilation out of the clock.
- **Costs.** From the same runs: the training time counts the training steps alone, every evaluation records how long it took and its peak memory, and every training step its own peak. Each cost in the tables is the median over the signals of a task, so a run slowed by other work on its GPU does not move it.
- **Metrics.** On the full signal, LPIPS with its VGG network. The tables report the mean over the signals of a task with a within-signal standard error (a plain one for IoU, which saturates on every shape), the best value in bold and the second in italics.
