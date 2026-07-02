#!/usr/bin/env python3
"""Run the complete paper experiment suite on all three datasets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASETS = "XES3G5M,ASSISTMENT2009,ASSISTMENT2017"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--output-root", default="paper_results")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model-seeds", default="12405,12406,12407")
    parser.add_argument("--kmeans-seeds", type=int, default=10)
    parser.add_argument("--bootstrap-repeats", type=int, default=10)
    parser.add_argument("--order-shuffles", type=int, default=20)
    parser.add_argument("--metric-sample", type=int, default=5000)
    parser.add_argument("--tsne-sample", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    output_root = Path(args.output_root).resolve()
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    if not datasets:
        raise ValueError("--datasets must contain at least one dataset")

    for dataset in datasets:
        command = [
            args.python,
            str(root / "run_experiments.py"),
            "--dataset",
            dataset,
            "--output-dir",
            str(output_root / dataset),
            "--config",
            args.config,
            "--python",
            args.python,
            "--device",
            args.device,
            "--gpu",
            args.gpu,
            "--model-seeds",
            args.model_seeds,
            "--kmeans-seeds",
            str(args.kmeans_seeds),
            "--bootstrap-repeats",
            str(args.bootstrap_repeats),
            "--order-shuffles",
            str(args.order_shuffles),
            "--metric-sample",
            str(args.metric_sample),
            "--tsne-sample",
            str(args.tsne_sample),
            "--batch-size",
            str(args.batch_size),
        ]
        if args.num_epochs is not None:
            command.extend(["--num-epochs", str(args.num_epochs)])
        if args.force:
            command.append("--force")
        if args.dry_run:
            command.append("--dry-run")
        print(f"\n=== {dataset} ===")
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
