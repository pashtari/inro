# API reference

Every public class and function, with its signature, tensor shapes and defaults, as written in the source.
{ .inro-lead }

The public API lives in three namespaces: everything for fitting coordinate/value pairs is exported at the top level, and image compression and radiance fields have their own.

| Namespace | Contents |
| --- | --- |
| `inro` | Models, datasets, `train`, `evaluate`, `chunked_inference`, losses, metrics, quantization, plotting and utilities |
| `inro.compression` | `encode_futon`, `decode_futon`, `encode_pil`, `decode_pil`, `evaluate_compression`, arithmetic coding |
| `inro.nerf` | `BlenderDataset`, `RadianceField`, renderers, `train`, `evaluate`, `render_image`, `render_views`, camera helpers |

<div class="grid cards" markdown>

-   :material-sine-wave:{ .lg .middle } **[FUTON, bases and combiners](futon.md)**

    ---

    `FUTON`, its six bases and the features they return, the combiners, the solvers and the error bounds.

-   :material-view-grid-outline:{ .lg .middle } **[inro.models](models.md)**

    ---

    Every other architecture, with its layers and encodings.

-   :material-database-outline:{ .lg .middle } **[Datasets](datasets.md)**

    ---

    Coordinate datasets for images, volumes and MRI, Deep Image Prior, and super-resolution's blocks with their bicubic operator.

-   :material-cog-play-outline:{ .lg .middle } **[Training and evaluation](training.md)**

    ---

    `train`, `evaluate`, `chunked_inference`.

-   :material-gauge:{ .lg .middle } **[Metrics and losses](metrics.md)**

    ---

    PSNR, SSIM, MS-SSIM, LPIPS, NMSE, IoU, bits per pixel and super-resolution's metrics; rate-distortion, SDF, super-resolution and entropy losses.

-   :material-numeric:{ .lg .middle } **[Quantization](quantization.md)**

    ---

    Uniform quantization of a model's parameters.

-   :material-zip-box-outline:{ .lg .middle } **[Compression](compression.md)**

    ---

    The codecs, their evaluation, and arithmetic coding.

-   :material-rotate-orbit:{ .lg .middle } **[Radiance fields](nerf.md)**

    ---

    The `inro.nerf` package.

-   :material-tools:{ .lg .middle } **[Utilities](utils.md)**

    ---

    Module specs, parameter counts, logging, serialization and the seaborn line plot.

</div>

!!! tip "Reading the docstrings"

    `(N, C)` is a flat batch and `(..., C)` allows any leading dimensions. `C` is the number of coordinate axes, `D` of output channels, `K_c` of basis functions on axis `c`, and `R` a rank; the [conventions page](../concepts/conventions.md#notation) has the full notation.
