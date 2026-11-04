"""FUTON's coefficient tensor for a Kodak image, in full and at low rank.

With a cosine basis of ``K`` functions per axis, a FUTON with a linear head represents
an image through a weight tensor W of ``K x K x 3`` coefficients. The best one in full
holds the image's generalized Fourier coefficients, its least-squares projection on
the basis, and at rank ``R`` the certified FUTON of the error bound holds a rank-``R``
CP tensor. ``inro.futon_bound`` gives both from the image alone, without training,
with the PSNR of the images they represent: the truncation in full, and at each rank
the certified FUTON's and the ceiling that no FUTON of that rank passes.

The basis is the paper's, ``1`` and ``sqrt(2) cos(k pi u)`` on ``u`` in ``[0, 1]``,
orthonormal on the unit interval, so that W holds the series' own coefficients.

Usage:

    python scripts/futon_coefficients.py                    # kodim23, K = 32, R = 4
    python scripts/futon_coefficients.py --image kodim19 --components 64 --ranks 4 16

Writes:

    logs/coefficients/<image>.pt    the weight tensors, "full" and one per rank
    logs/coefficients/<image>.json  their parameter counts and PSNRs
"""

import argparse
import json
import math
from pathlib import Path

import torch

import inro
from train_common import ROOT


def paper_basis(components: int, size: int, axis: int) -> torch.Tensor:
    """Return the paper's cosine basis on an axis's grid, ``(size, components)``."""
    basis = inro.CosineBasis(2, num_components=components, normalize=False)
    grid = torch.linspace(-1, 1, size, dtype=torch.float64)
    scale = torch.full((components,), math.sqrt(2.0), dtype=torch.float64)
    scale[0] = 1.0
    return basis.axis_features(grid, axis).double() * scale


def psnr(mse: float) -> float:
    """Return the PSNR of a mean squared error, for signals in ``[-1, 1]``."""
    return 10 * math.log10(4 / mse)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", default="kodim23", help="Kodak image name")
    parser.add_argument("--components", type=int, default=32, help="K per axis")
    parser.add_argument("--ranks", type=int, nargs="+", default=[4], help="CP ranks")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs" / "coefficients")
    args = parser.parse_args()

    image = inro.ImageCoordinateDataset(ROOT / "data" / "Kodak" / f"{args.image}.png")
    signal = image.target.double()  # (H, W, 3) in [-1, 1]
    *sizes, channels = signal.shape
    features = [paper_basis(args.components, n, c) for c, n in enumerate(sizes)]
    # The nested SVDs' 3 K terms are the whole tensor, so the full one is that rank.
    full = channels * args.components
    bound = inro.futon_bound(signal, features, sorted({*args.ranks, full}))

    weights, ranks = {}, {}
    for rank, (factors, head), lower, upper in zip(
        bound.ranks, bound.factors, bound.lower, bound.upper
    ):
        tensor = torch.einsum("ar,br,dr->abd", *factors, head).float()
        if rank == full:
            weights["full"] = tensor
            continue
        weights[str(rank)] = tensor
        ranks[str(rank)] = {
            "parameters": rank * (sum(len(f) for f in factors) + channels),
            "certified_psnr": psnr(float(upper)),
            "ceiling_psnr": psnr(float(lower)),
        }
    record = {
        "image": args.image,
        "basis": "cosine",
        "components": args.components,
        "size": sizes,
        "values": signal.numel(),
        "coefficients": weights["full"].numel(),
        "truncation_psnr": psnr(bound.truncation),
        "ranks": ranks,
    }
    args.log_dir.mkdir(parents=True, exist_ok=True)
    torch.save(weights, args.log_dir / f"{args.image}.pt")
    (args.log_dir / f"{args.image}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(f"{args.image}, K = {args.components}: {record['coefficients']:,} coefficients, "
          f"{record['truncation_psnr']:.2f} dB in full", flush=True)  # fmt: skip
    for rank, row in ranks.items():
        print(f"  rank {rank}: {row['parameters']:,} parameters, certified "
              f"{row['certified_psnr']:.2f} dB, ceiling {row['ceiling_psnr']:.2f} dB")  # fmt: skip


if __name__ == "__main__":
    main()
