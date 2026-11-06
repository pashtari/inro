"""FUTON's error bound against trained FUTONs, on Kodak images and occupancy volumes.

Usage:

    python scripts/futon_bound.py                  # every signal and check
    python scripts/futon_bound.py --tasks image --signals kodim19 kodim01

For every signal, spectral resolution ``K = (kappa L_1, ..., kappa L_C)``, ``L_c`` the
signal's points on axis ``c``, and CP rank ``R``, the script:

- bounds the best grid error of the paper's FUTON, a cosine basis, a CP combiner
  and a linear head without bias (``inro.futon_bound``), the upper bound being
  the error of the certified FUTON, built from the nested SVDs without training,
  and times it;
- trains the same FUTON with Adam from a random start, as the benchmark trains
  every model.

A check on every signal, at ``kappa = 1/2`` and the benchmark's rank, measures how close
training comes to the ceiling: Adam for the benchmark's steps and for five times as
many. Errors are mean squared errors on the whole grid, and PSNR is taken for signals
in ``[-1, 1]``.
The theorem behind the bound is in docs/concepts/error-bound.md.

Writes:

    logs/bound/<task>/<signal>.json  one record per (kappa, R), resumable
    logs/bound/<task>_checks.json    the optimizer checks, one row per signal
"""

import argparse
import json
import os
import time

# Silence inro.train's progress bar in the study, but not where this is imported.
if __name__ == "__main__":
    os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
import yaml

import inro
from train_common import ROOT, fill_rank, resolve

# Each task's "kappas" are relative bandwidths, K_c = kappa L_c on every axis c,
# L_c its points. Kappa 1 and 2 enter the cost alone, to show how the bound's
# time grows with K; every other result is at kappa <= 1/2.
TASKS = {
    "image": {
        "signals": [f"kodim{i:02d}" for i in range(1, 25)],
        "kappas": (0.125, 0.25, 0.5, 1.0, 2.0),
        "ranks": (32, 64, 128, 256, 512, 1024),
        "data": {},  # the whole image at every step, as configs/image.yaml
        "lr": 1e-1,  # FUTON's rate in configs/image.yaml
    },
    "occupancy": {
        "signals": ["armadillo", "dragon", "happy_buddha", "lucy", "thai_statue"],
        "kappas": (0.125, 0.25, 0.5, 1.0, 2.0),
        "ranks": (32, 64, 128, 256, 512, 1024),
        "data": {"max_samples": 1048576},  # as configs/occupancy.yaml
        "lr": 3e-2,  # FUTON's rate in configs/occupancy.yaml
    },
}
# The paper's basis. Records carry it, so that settings of another basis are never
# mistaken for its own.
BASIS = "cosine"
# The benchmark's steps, at its batches and each task's "lr" above.
STEPS = 1000
# The optimizer checks at kappa = 1/2 and the benchmark's rank: Adam for STEPS and for
# LONGER times them.
LONGER = 5
LOGS = ROOT / "logs" / "bound"


def psnr(mse: float | np.ndarray) -> float | np.ndarray:
    """Return the PSNR of a mean squared error, for signals in ``[-1, 1]``."""
    return 10 * np.log10(4 / np.asarray(mse))


def dataset(task: str, name: str, **data):
    """Load a signal, whole or, with ``data``, as each training step draws it."""
    if task == "image":
        return inro.ImageCoordinateDataset(ROOT / f"data/Kodak/{name}.png", **data)
    path = ROOT / f"data/occupancy/{name}.ply"
    config = yaml.safe_load((ROOT / "configs/occupancy.yaml").read_text())
    resolution = config["data"]["resolution"]  # the benchmark's grid
    return inro.OccupancyCoordinateDataset(path, resolution=resolution, **data)


def benchmark_rank(task: str, sizes: list[int], channels: int) -> int:
    """Return the CP rank of the benchmark's FUTON on a signal of ``sizes``."""
    config = yaml.safe_load((ROOT / "configs" / f"{task}.yaml").read_text())
    names = "DHW"[-len(sizes) :]
    entry = resolve(config["models"]["FUTON"], **dict(zip(names, sizes)))
    return fill_rank(entry, outputs=channels)["kwargs"]["combiner"][1]["rank"]


def futon(sizes: list[int], channels: int, kappa: float, rank: int) -> inro.FUTON:
    """Build the paper's FUTON: cosine basis, CP combiner, linear head, no bias."""
    components = [max(2, int(kappa * size)) for size in sizes]
    basis = (BASIS, {"num_components": components, "grid_size": sizes})
    return inro.FUTON(len(sizes), channels, basis, ("cp", {"rank": rank}),
                      ("linear", {"bias": False}))  # fmt: skip


def features_of(model: inro.FUTON, sizes: list[int], device) -> list[torch.Tensor]:
    """Return the basis on each axis's grid, as ``inro.futon_bound`` takes it."""
    return [
        model.encoding.basis.axis_features(torch.linspace(-1, 1, n), c).to(device)
        for c, n in enumerate(sizes)
    ]


def evaluate(model: inro.FUTON, data, device) -> tuple[float, float | None]:
    """Return the grid MSE, and the IoU (%) at the zero level for occupancy."""
    output = inro.chunked_inference(model, data.input, chunk_size=2**20, device=device)
    target = data.target.to(device)
    mse = float((output.double() - target.double()).square().mean())
    if target.shape[-1] != 1 or not torch.all(target.abs() == 1):
        return mse, None
    inside, truth = output > 0, target > 0
    return mse, 100 * float((inside & truth).sum() / (inside | truth).sum())


def train(model: inro.FUTON, data, device, lr: float, steps: int = STEPS,
          seed: int = 0) -> float:  # fmt: skip
    """Train in place with Adam as the benchmark does, and return the seconds taken."""
    torch.manual_seed(seed)
    model.encoding.combiner.reset_parameters()
    model.head.reset_parameters()
    start = time.perf_counter()
    inro.train(model, data, num_steps=steps, lr=lr, log_interval=0,
               save_reconstruction=False, device=device, seed=seed)  # fmt: skip
    return time.perf_counter() - start


def warm_up(task: str, device: torch.device) -> None:
    """Bound and briefly train on a task's first signal, untimed.

    No timed run then pays for setting up kernels and libraries.
    """
    name = TASKS[task]["signals"][0]
    full = dataset(task, name)
    sizes, channels = list(full.target.shape[:-1]), full.target.shape[-1]
    features = features_of(futon(sizes, channels, 0.125, 1), sizes, device)
    inro.futon_bound(full.target.to(device), features, [8])
    train_data = dataset(task, name, **TASKS[task]["data"])
    train(
        futon(sizes, channels, 0.125, 8),
        train_data,
        device,
        TASKS[task]["lr"],
        steps=10,
    )


def of_basis(rows: list[dict]) -> list[dict]:
    """Return the rows of the study's basis; a row that names none is sinc's."""
    return [row for row in rows if row.get("basis", "sinc") == BASIS]


def run(task: str, name: str, device: torch.device) -> None:
    """Bound and train every setting of one signal not yet recorded.

    A ``K`` is bounded at all its ranks in one call, whose time the cost reports,
    so a ``K`` with a rank yet to bound is bounded again at every rank; a setting
    whose Adam run is recorded keeps it.
    """
    settings = TASKS[task]
    path = LOGS / task / f"{name}.json"
    record = json.loads(path.read_text()) if path.exists() else {"configs": []}
    configs = {(c["kappa"], c["R"]): c for c in of_basis(record["configs"])}
    kappas = [
        kappa
        for kappa in settings["kappas"]
        if any("upper" not in configs.get((kappa, r), {}) for r in settings["ranks"])
    ]
    if not kappas:
        return
    full = dataset(task, name)
    train_data = dataset(task, name, **settings["data"])
    sizes, channels = list(full.target.shape[:-1]), full.target.shape[-1]
    record.update(task=task, signal=name, sizes=sizes, channels=channels)
    target = full.target.to(device)
    for kappa in kappas:
        features = features_of(futon(sizes, channels, kappa, 1), sizes, device)
        start = time.perf_counter()
        bound = inro.futon_bound(target, features, settings["ranks"])
        seconds = time.perf_counter() - start
        for rank, lower, upper in zip(bound.ranks, bound.lower, bound.upper):
            config = configs.get((kappa, rank))
            if config is None:  # no Adam run yet
                model = futon(sizes, channels, kappa, rank)
                adam_seconds = train(model, train_data, device, settings["lr"])
                adam_mse, adam_iou = evaluate(model, full, device)
                config = {
                    "basis": BASIS,
                    "lr": settings["lr"],
                    "kappa": kappa,
                    "K": [max(2, int(kappa * n)) for n in sizes],
                    "R": rank,
                    "parameters": inro.count_parameters(model),
                    "adam": adam_mse,
                    "adam_iou": adam_iou,
                    "adam_seconds": adam_seconds,
                }
                configs[(kappa, rank)] = config
                record["configs"].append(config)
            config.update(
                {
                    "truncation": bound.truncation,
                    "lower": float(lower),
                    "upper": float(upper),
                    "bound_seconds": seconds,
                }
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1))
            print(f"{task} {name} kappa {kappa} R {rank}: certified "
                  f"[{psnr(upper):.2f}, {psnr(lower):.2f}] dB, Adam "
                  f"{psnr(config['adam']):.2f} dB", flush=True)  # fmt: skip


def checks(task: str, device: torch.device) -> None:
    """Check on every signal how close training comes to the bound's ceiling.

    At ``kappa = 1/2`` and the benchmark's rank: Adam for the benchmark's steps and for
    ``LONGER`` times them, against the bound.
    """
    path = LOGS / f"{task}_checks.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    for name in TASKS[task]["signals"]:
        if any(row["signal"] == name for row in of_basis(rows)):
            continue
        full = dataset(task, name)
        sizes, channels = list(full.target.shape[:-1]), full.target.shape[-1]
        rank = benchmark_rank(task, sizes, channels)
        model = futon(sizes, channels, 0.5, rank)
        bound = inro.futon_bound(
            full.target.to(device), features_of(model, sizes, device), rank
        )
        row = {"signal": name, "basis": BASIS, "lr": TASKS[task]["lr"], "R": rank,
               "upper": float(bound.upper[0]), "lower": float(bound.lower[0])}  # fmt: skip
        train_data = dataset(task, name, **TASKS[task]["data"])
        for key, steps in (("adam", STEPS), ("adam_longer", LONGER * STEPS)):
            train(model, train_data, device, TASKS[task]["lr"], steps=steps)
            row[key] = evaluate(model, full, device)[0]
        rows.append(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=1))
        print(f"{task} checks {name} done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--signals", nargs="+", help="signals to run (default: all)")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    for task in args.tasks:
        warm_up(task, device)
        for name in args.signals or TASKS[task]["signals"]:
            if name in TASKS[task]["signals"]:
                run(task, name, device)
        if not args.signals:
            checks(task, device)


if __name__ == "__main__":
    main()
