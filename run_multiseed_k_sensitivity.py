#!/usr/bin/env python3
"""Run K=3/4/5/6 sensitivity for existing latent artifacts across seeds.

This is a post-training wrapper around analyze_cluster_sensitivity.py.
It does not train models or recompute latent representations. By default it
matches the paper-results layout:

    seed 12405 -> experiment_1/runs/soft_kmeans/seed_12405
    other seeds -> experiment_2/model_seed_runs/seed_<seed>

Every per-seed report is written below --output-root. An aggregate mean/std
table is also written at that root.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analyze_cluster_sensitivity import parse_k_values, run_sensitivity


DEFAULT_DATASETS = ["ASSISTMENT2009", "ASSISTMENT2017", "XES3G5M"]
DEFAULT_SEEDS = [12405, 12406, 12407]


@dataclass(frozen=True)
class SeedPaths:
    preprocessed_csv: Path
    clustering_dir: Path
    output_dir: Path


def _render(template: str, dataset: str, seed: int) -> Path:
    return Path(template.format(dataset=dataset, seed=int(seed)))


def build_paths(
    source_root: Path,
    output_root: Path,
    dataset_root: Path,
    dataset: str,
    seed: int,
    primary_seed: int,
    primary_run_template: str,
    extra_run_template: str,
    latent_template: str,
    output_template: str,
    data_template: str,
) -> SeedPaths:
    """Build all paths for one dataset/seed without touching the filesystem."""

    run_template = (
        primary_run_template if int(seed) == int(primary_seed) else extra_run_template
    )
    run_dir = Path(source_root) / dataset / _render(run_template, dataset, seed)
    clustering_dir = run_dir / _render(latent_template, dataset, seed)
    return SeedPaths(
        preprocessed_csv=Path(dataset_root) / _render(data_template, dataset, seed),
        clustering_dir=clustering_dir,
        output_dir=Path(output_root) / _render(output_template, dataset, seed),
    )


def _required_artifacts(clustering_dir: Path, k_values: list[int]) -> list[Path]:
    common = [
        clustering_dir / "latent_features.npz",
        clustering_dir / "latent_metadata.csv",
    ]
    per_k = []
    for k in k_values:
        per_k.extend(
            [
                clustering_dir / f"cluster_assignments_k{k}.csv",
                clustering_dir / f"kmeans_model_k{k}.pkl",
                clustering_dir / f"representatives_k{k}.csv",
            ]
        )
    return common + per_k


def _check_inputs(paths: SeedPaths, config_path: Path, k_values: list[int]) -> None:
    missing = []
    if not config_path.exists():
        missing.append(config_path)
    if not paths.preprocessed_csv.exists():
        missing.append(paths.preprocessed_csv)
    missing.extend(
        path
        for path in _required_artifacts(paths.clustering_dir, k_values)
        if not path.exists()
    )
    if missing:
        preview = "\n".join(f"- {path}" for path in missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n- ... and {len(missing) - 20} more"
        raise FileNotFoundError("Missing sensitivity inputs:\n" + preview + suffix)


def _write_aggregate(
    records: list[tuple[str, int, Path]],
    output_root: Path,
    selected_k: int,
) -> None:
    frames = []
    for dataset, seed, output_dir in records:
        frame = pd.read_csv(output_dir / "cluster_metrics.csv")
        frame.insert(0, "model_seed", int(seed))
        frame.insert(0, "dataset", dataset)
        frames.append(frame)

    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(output_root / "k_sensitivity_multiseed_metrics.csv", index=False)

    summary = (
        metrics.groupby(["dataset", "k"], as_index=False)
        .agg(
            silhouette_mean=("silhouette_score", "mean"),
            silhouette_std=("silhouette_score", "std"),
            dbi_mean=("davies_bouldin_index", "mean"),
            dbi_std=("davies_bouldin_index", "std"),
            ch_mean=("calinski_harabasz_score", "mean"),
            ch_std=("calinski_harabasz_score", "std"),
            balance_mean=("cluster_size_min_max_ratio", "mean"),
            balance_std=("cluster_size_min_max_ratio", "std"),
            geometry_best_fraction=("metric_best_k", "mean"),
        )
        .sort_values(["dataset", "k"])
    )
    summary["selected_k"] = summary["k"].eq(int(selected_k))
    summary.to_csv(output_root / "k_sensitivity_multiseed_summary.csv", index=False)

    lines = [
        "# Multi-seed K sensitivity",
        "",
        "This report reuses existing latent representations and K-specific K-means artifacts; no model was retrained.",
        "",
        "The K order is evaluated with Silhouette/DBI/CH and cluster balance. Means and standard deviations are across model seeds.",
        "",
    ]
    for dataset in summary["dataset"].drop_duplicates().tolist():
        lines.extend([f"## {dataset}", ""])
        subset = summary[summary["dataset"] == dataset]
        lines.extend(
            [
                "| K | Silhouette mean ± std | DBI mean ± std | CH mean ± std | Geometry-best fraction |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in subset.iterrows():
            lines.append(
                f"| {int(row['k'])} | {row['silhouette_mean']:.4f} ± {row['silhouette_std']:.4f} | "
                f"{row['dbi_mean']:.4f} ± {row['dbi_std']:.4f} | "
                f"{row['ch_mean']:.2f} ± {row['ch_std']:.2f} | "
                f"{row['geometry_best_fraction']:.2f} |"
            )
        lines.append("")
    (output_root / "k_sensitivity_multiseed_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="paper_result/paper_results")
    parser.add_argument("--output-root", default="paper_result/k_sensitivity_by_seed")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--primary-seed", type=int, default=12405)
    parser.add_argument(
        "--primary-run-template",
        default="experiment_1/runs/soft_kmeans/seed_{seed}",
    )
    parser.add_argument(
        "--extra-run-template",
        default="experiment_2/model_seed_runs/seed_{seed}",
    )
    parser.add_argument(
        "--latent-template",
        default="latent_clustering/{dataset}/official",
    )
    parser.add_argument(
        "--output-template",
        default="{dataset}/soft_kmeans/seed_{seed}",
    )
    parser.add_argument(
        "--data-template",
        default="{dataset}/preprocessed_df.csv",
    )
    parser.add_argument("--k-values", default="3,4,5,6")
    parser.add_argument("--selected-k", type=int, default=3)
    parser.add_argument("--metric-sample", type=int, default=5000)
    parser.add_argument(
        "--timestamp-unit",
        choices=["auto", "milliseconds", "seconds", "ordinal"],
        default="auto",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    dataset_root = Path(args.dataset_root)
    config_path = Path(args.config)
    k_values = parse_k_values(args.k_values)
    output_root.mkdir(parents=True, exist_ok=True)
    records = []

    total = len(args.datasets) * len(args.seeds)
    current = 0
    for dataset in args.datasets:
        for seed in args.seeds:
            current += 1
            paths = build_paths(
                source_root=source_root,
                output_root=output_root,
                dataset_root=dataset_root,
                dataset=dataset,
                seed=seed,
                primary_seed=args.primary_seed,
                primary_run_template=args.primary_run_template,
                extra_run_template=args.extra_run_template,
                latent_template=args.latent_template,
                output_template=args.output_template,
                data_template=args.data_template,
            )
            _check_inputs(paths, config_path, k_values)
            print(
                f"[{current}/{total}] {dataset} / seed_{seed}\n"
                f"  clustering={paths.clustering_dir}\n"
                f"  output={paths.output_dir}",
                flush=True,
            )
            if (
                paths.output_dir.exists()
                and any(paths.output_dir.iterdir())
                and not args.force
            ):
                raise FileExistsError(
                    f"Output exists: {paths.output_dir}; use --force to overwrite generated files"
                )
            if not args.dry_run:
                paths.output_dir.mkdir(parents=True, exist_ok=True)
                run_sensitivity(
                    preprocessed_csv=paths.preprocessed_csv,
                    clustering_dir=paths.clustering_dir,
                    config_path=config_path,
                    output_dir=paths.output_dir,
                    k_values=k_values,
                    selected_k=args.selected_k,
                    metric_sample=args.metric_sample,
                    seed=int(seed),
                    timestamp_unit=args.timestamp_unit,
                )
                records.append((dataset, int(seed), paths.output_dir))

    if not args.dry_run:
        _write_aggregate(records, output_root, args.selected_k)
        print(f"Wrote aggregate sensitivity outputs to: {output_root}")


if __name__ == "__main__":
    main()
