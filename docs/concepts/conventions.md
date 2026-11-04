# Conventions

The rules that let datasets, models, losses and metrics be combined freely.
{ .inro-lead }

## Notation

| Symbol | Meaning |
| --- | --- |
| `C` | Coordinate axes, the `in_features` of an INR |
| `D` | Output channels, the `out_features` of an INR |
| `N` | Points in a flat batch; `(N, C)` is such a batch and `(..., C)` allows any leading dimensions |
| `K_c`, `K` | Basis functions on axis `c`, or on every axis |
| `R` | CP rank |
| `H`, `W` | Image height and width |
| `channels`, `depth` | An image's channels and a volume's first grid axis, spelled out because `C` and `D` mean the above |

## Coordinates

Coordinates live in $[-1, 1]$ on every axis, both endpoints included. `inro.create_coordinates((H, W))` returns an `(H, W, 2)` grid whose column $c$ spans axis $c$ (`indexing="ij"`), and the datasets build their inputs with it.

<svg class="inro-diagram" viewBox="0 0 330 200" width="330" role="img" aria-label="An image's coordinates: the first axis runs down from -1 to 1, the second across from -1 to 1">
  <defs><marker id="coord-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#2a78d6"/></marker></defs>
  <rect x="84" y="36" width="210" height="140" rx="4" fill="#2a78d6" fill-opacity=".1" stroke="#8a94a6"/>
  <path d="M84 24h210" stroke="#2a78d6" stroke-width="1.8" marker-end="url(#coord-arrow)"/>
  <path d="M72 36v140" stroke="#2a78d6" stroke-width="1.8" marker-end="url(#coord-arrow)"/>
  <text x="189" y="16" font-size="13" text-anchor="middle">x&#8322;, the column</text>
  <text x="62" y="110" font-size="13" text-anchor="end">x&#8321;, the row</text>
  <circle cx="84" cy="36" r="3.5" fill="#ee6a36"/><text x="92" y="53" font-size="12">(&#8722;1, &#8722;1)</text>
  <circle cx="294" cy="176" r="3.5" fill="#ee6a36"/><text x="286" y="168" font-size="12" text-anchor="end">(1, 1)</text>
</svg>

FUTON's bases, the hash and feature grids, and `GridINR` patches are defined on this domain, so do not pre-normalize to $[0, 1]$. Models accept any leading shape; the taps of FUTON's local bases need a flat `(N, C)` batch, which `FUTON.forward` produces internally.

## Values

Targets are in the model's output range, and the reference signal keeps its native form.

| Dataset | `input` | `target` | `_original` |
| --- | --- | --- | --- |
| `ImageCoordinateDataset` | `(H, W, 2)` | `(H, W, channels)` float in $[-1, 1]$ | `(channels, H, W)` `uint8` |
| `MaskedImageCoordinateDataset` | `(H, W, 2 + side_channels)` | `(H, W, channels)` in $[-1, 1]$, plus `side`, `mask`, `boundary` | `(channels, H, W)` `uint8` |
| `SuperResolutionDataset` | `(B, L, L, 2)`, the high-resolution coordinates under `B` blocks of low-resolution pixels | `(B, S, S, channels)` low-resolution pixels in $[-1, 1]$, plus the kernel rows `weight_y`, `weight_x` `(B, S, L)` | `(channels, h, w)` `uint8`, the low-resolution image |
| `OccupancyCoordinateDataset` | `(depth, H, W, 3)` | `(depth, H, W, 1)` in $\{-1, 1\}$ | `(1, depth, H, W)` float in $\{0, 1\}$ |
| `MRICoordinateDataset` | `(H', W', 2)` | complex `(coils, H, W)` masked k-space, plus `mask`, `scale_factor`, `zero_filled` | RSS image `(H', W')`, absent when none is loaded |
| `DIPImageDataset` | noise `(noise_channels, H, W)` | `(channels, H, W)` float in $[0, 1]$ | `(channels, H, W)` `uint8` |
| `nerf.BlenderDataset` | rays `(H, W, 3)` origins and directions | `(H, W, 4)` RGBA in $[0, 1]$ | |

- **Subsampling** (`subsample`, `max_samples`) flattens `input` and `target` to `(N, ...)` with shared random indices.
- **`preprocess`** maps native values to targets: `uint8` to $[-1, 1]$, occupancy $\{0, 1\}$ to $\{-1, 1\}$.
- **`postprocess`** maps model output back: clipping to $[-1, 1]$ before the `uint8` cast, thresholding occupancy at zero.

Because targets are in $[-1, 1]$, the benchmark models end in `torch.tanh`, and the default training loss reports PSNR with a peak-to-peak range of 2.

## Metrics

In `inro.train` and `inro.evaluate`, metrics are called as `fn(reconstruction, original)` after `postprocess`, so they see `uint8` images and `{0, 1}` volumes.

| Metric | Input | Notes |
| --- | --- | --- |
| `inro.psnr`, `inro.ssim`, `inro.ms_ssim` | two `(channels, H, W)` images | `data_range` from the target: 255 for `uint8`, 1 for floating point |
| `inro.lpips` | two `(3, H, W)` images | VGG by default, `net="alex"` for AlexNet; its weights are downloaded once |
| `inro.iou` | two volumes | Thresholds both at zero, so it works on the `{0, 1}` original and on raw model output alike |
| `inro.nmse` | two volumes or images | fastMRI's normalized MSE |
| `inro.bits_per_pixel(encoded, image)` | a byte string and a `(channels, *spatial)` image | The string's size over the image's spatial size |
| `inro.super_resolution_metrics(border)` | two `(3, H, W)` images | Super-resolution's protocol: PSNR and SSIM on `inro.luma`, BT.601's Y as MATLAB's `rgb2ycbcr` computes it, and LPIPS on RGB, all without a border of `border` pixels |

## Widths, depths and activations

`hidden_layers=h` means `h` hidden layers and `h + 1` weight layers. `output_activation` is applied after the last layer of every model; inside `inro.nerf.RadianceField` the density network must have none, because the field applies `trunc_exp` to the density and a sigmoid to the color.

## Checkpoints

- `inro.train` saves the model's state dict as `checkpoint.pt` in `log_dir` and returns it as `model_dict`.
- With `quantize=True`, the checkpoint holds the quantized weights and `model_dict` a dictionary of integer codes and per-tensor scales, which `inro.uniform_dequantize` loads back.
- `inro.nerf.train` saves the field's state dict, including the `aabb` buffer.
- `MLP.layers` holds only weight layers (`layers.0` to `layers.h`), and grid models keep their decoder under `decoder.layers.*`.

## Seeds and devices

- `seed` in `inro.train` and `inro.nerf.train` seeds PyTorch's global RNG before training, after the model has been built. Construct the model after `torch.manual_seed` to also fix its initialization, as the scripts do.
- A dataset's `seed` reseeds Python's RNG only; subsampling draws from PyTorch's RNG.
- `device=None` selects CUDA when available. The training loop moves the model and, through their `to` method, the datasets to that device, and `_original` stays on the CPU.

## Experiment files

A benchmark run lives in `logs/<task>/<signal>/<model>/` with `log.txt`, `log.json`, `checkpoint.pt`, the final reconstruction (or two rendered test views for NeRF), and `results.json`: the training configuration, the model's `setup` from the YAML, `num_params`, `train_time`, the final `metrics`, and the `history`. Failed runs leave `error.txt`. The [reproduction page](../experiments/reproduce.md) describes the scripts that write and read them.
