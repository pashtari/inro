# Your first INR

Fit FUTON to one image, read what the training loop returns, and swap in another model. Any RGB image works; the code below uses a Kodak image from the [installation page](installation.md#benchmark-data).
{ .inro-lead }

<ol class="inro-flow">
  <li><strong>Dataset</strong>an image as coordinates and values</li>
  <li><strong>Model</strong>a function of coordinates</li>
  <li><strong>Training</strong>Adam with cosine annealing</li>
  <li><strong>Inference</strong>evaluate it anywhere</li>
</ol>

## Fit an image

```python
import torch
import inro

image = "data/Kodak/kodim19.png"
dataset = inro.ImageCoordinateDataset(image)  # (1)!
H, W, _ = dataset.target.shape

model = inro.FUTON(  # (2)!
    in_features=2,
    out_features=3,
    basis=("cosine", {"num_components": [H // 2, W // 2]}),
    combiner=("cp", {"rank": 280}),
    head=("mlp", {"hidden_layers": 1, "hidden_features": 64}),
    output_activation=torch.tanh,  # (3)!
)

result = inro.train(  # (4)!
    model,
    dataset,
    dataset,
    num_steps=1000,
    lr=1e-1,
    metrics={"psnr": inro.psnr, "ssim": inro.ssim},
    log_interval=250,
    eval_interval=250,
)
print(result["history"][-1]["eval"])
```

1.  `dataset.input` holds the pixel-center coordinates, `(H, W, 2)` in $[-1, 1]$, and `dataset.target` the RGB values, `(H, W, 3)`, also in $[-1, 1]$. Each access returns the whole image; `subsample=0.1` would draw a fresh random 10 % of the pixels per access instead, flattened to `(N, 2)` and `(N, 3)`.
2.  Three specs, each a registry key, a `(key, kwargs)` pair, a class or a module instance. The cosine basis has half as many functions per axis as the image has pixels, the CP combiner has rank 280, and the head is an MLP with one hidden layer of width 64, whose first layer reads a batch that covers the image off the whole grid in one contraction.
3.  The targets are in $[-1, 1]$, and so is the output of `tanh`.
4.  Adam with cosine annealing from `lr` down to `lr / 100`, one optimizer step per batch; this dataset's one item is the whole image, so every step sees every pixel. Every `log_interval` steps it prints the training metrics above a progress bar. Every `eval_interval` steps, and at the last, it maps predictions back to `uint8` with `dataset.postprocess` and adds the metrics to that line.

On a GPU this takes about a dozen seconds (13 s on an RTX 4060 Ti) and ends near 37.8 dB PSNR.

!!! info "This is the benchmark's FUTON"

    It holds 197,379 parameters. The benchmark also tabulates the basis on the pixel grid (`grid_size`) and compiles the model, which trains it in about 4 s on an A100.

The result is a plain dictionary:

```python
result["config"]       # seed, device, model, num_params, optimizer settings, ...
result["model_dict"]   # the final state dict
result["history"][-1]  # {"step": 1000, "elapsed": ..., "mse": ..., "psnr": ...,
                       #  "eval": {"psnr": ..., "ssim": ...}}
```

## Reconstruct and save

To rebuild the image, evaluate the model on the dataset's coordinate grid in chunks and map the output back to pixels:

```python
output = inro.chunked_inference(model, dataset.input, chunk_size=65536)  # (H, W, 3) in [-1, 1]
reconstruction = dataset.postprocess(output.cpu())                       # uint8 (3, H, W)

print(inro.psnr(reconstruction, dataset.original))
dataset.save(reconstruction, "kodim19_futon.png")
```

Coordinates need not lie on the pixel grid: `inro.create_coordinates((2 * H, 2 * W))` gives a twice-finer grid on the same domain, and `model(coords)` evaluates the continuous representation there. How well it interpolates depends on the basis and on `num_components`; see [FUTON](../concepts/futon.md).

## Log to disk

Pass `log_dir` to keep a record of the run:

```python
result = inro.train(model, dataset, dataset, num_steps=1000, lr=1e-1, log_dir="logs/quickstart")
```

| File | Content |
| --- | --- |
| `log.txt` | The console's lines, with the run's configuration |
| `log.json` | The same records as JSON, read back with `inro.read_json_records` |
| `checkpoint.pt` | The final state dict |
| `kodim19_reconstructed.png` | The evaluation's reconstruction |

## Swap the model

Every model in the [zoo](../concepts/models.md) takes `in_features` and `out_features` first and maps `(..., in_features)` coordinates in $[-1, 1]$ to `(..., out_features)` values, so the training call does not change:

```python
model = inro.SIREN(2, 3, hidden_features=256, hidden_layers=4, output_activation=torch.tanh)
result = inro.train(model, dataset, dataset, num_steps=1000, lr=1e-3, metrics={"psnr": inro.psnr})
```

## Next steps

<div class="grid cards" markdown>

-   **[Fitting an image](../tutorials/image.md)**

    ---

    The benchmark's models and learning rates, with learning curves.

-   **[Occupancy volumes](../tutorials/occupancy.md)**

    ---

    The same loop on a 3D shape, scored by IoU.

-   **[Radiance fields](../tutorials/nerf.md)**

    ---

    Any model as the density network of a NeRF.

</div>
