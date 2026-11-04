# Getting started

From an empty environment to a trained INR in two short pages.
{ .inro-lead }

<div class="grid cards" markdown>

-   :material-download-outline:{ .lg .middle } **[Installation](installation.md)**

    ---

    Requirements, the optional extras, and the benchmark data.

-   :material-rocket-launch-outline:{ .lg .middle } **[Your first INR](quickstart.md)**

    ---

    Fit FUTON to an image, read the result, and swap in another model.

</div>

## The package at a glance

Everything is importable from the top-level package:

```python
import inro

inro.FUTON, inro.SIREN, inro.InstantNGP                       # models
inro.ImageCoordinateDataset, inro.OccupancyCoordinateDataset  # datasets
inro.train, inro.evaluate, inro.chunked_inference             # training and evaluation
inro.psnr, inro.ssim, inro.lpips, inro.iou                    # metrics
inro.futon_bound                                              # FUTON's error bound
inro.nerf                                                     # radiance fields
```
