#!/usr/bin/env python3
"""Train selected model seeds and produce camera-ready mean +/- std tables.

The training stage delegates to ``run_camera_ready_correction.py`` so that the
accepted training/extraction pipeline remains the single source of truth. By
default this runner evaluates Original CL4KT and Soft + KMeans on seeds
12405/12406/12407, with K=3 fixed for the reported table.

Outputs are written below ``--output-root``:

    per_seed_metrics.csv
    mean_std_metrics.csv
    paper_table.csv
    paper_table.md

Use ``--phase aggregate`` to regenerate the tables from an already completed
output root without retraining.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "camera_ready_correction.yaml"
DEFAULT_DATASETS = ["XES3G5M", "ASSISTMENT2009", "ASSISTMENT2017"]
DEFAULT_CONFIGURATIONS = ["original", "soft_kmeans"]
ALL_CONFIGURATIONS = [
    {"slug": "original", "losses": None},
    {"slug": "soft", "losses": "soft"},
    {"slug": "soft_kmeans", "losses": "soft,kmeans"},
    {"slug": "soft_separation_temporal", "losses": "soft,separation,temporal"},
    {"slug": "full", "losses": "soft,kmeans,separation,temporal"},
]
DEFAULT_SEEDS = [12405, 12406, 12407]
METHOD_LABELS = {
    "original": "Original",
    "soft_kmeans": "Soft + KMeans",
}


def parse_csv_values(value: str, cast: Callable = str) -> list:
    """Parse comma- or whitespace-separated values and reject duplicates."""

    if value is None:
        raise ValueError("value must not be None")
    tokens = str(value).replace(",", " ").split()
    if not tokens:
        raise ValueError("value must not be empty")
    try:
        values = [cast(token) for token in tokens]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid value: {value!r}") from exc
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate value in {value!r}")
    return values


def _runner_run_dir(output_root: Path, dataset: str, configuration: str, seed: int) -> Path:
    preferred = (
        Path(output_root)
        / dataset
        / "experiment_1"
        / "runs"
        / configuration
        / f"seed_{int(seed)}"
    )
    if preferred.exists():
        return preferred

    # Compatibility with the repository's earlier Experiment 2 layout.
    if configuration == "soft_kmeans" and int(seed) != 12405:
        return (
            Path(output_root)
            / dataset
            / "experiment_2"
            / "model_seed_runs"
            / f"seed_{int(seed)}"
        )
    return preferred


def _latest_prediction_log(run_dir: Path, dataset: str) -> tuple[Path, pd.DataFrame]:
    log_dir = Path(run_dir) / "train_logs" / dataset
    candidates = sorted(
        log_dir.glob("cl4kt_*.csv"),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    valid = []
    for path in candidates:
        frame = pd.read_csv(path)
        if "auc" in frame.columns and not frame.empty:
            valid.append((path, frame))
    if not valid:
        raise FileNotFoundError(f"No prediction log found under {log_dir}")
    return valid[-1]


def load_seed_metrics(
    output_root: Path,
    dataset: str,
    configuration: str,
    seed: int,
    selected_k: int,
) -> dict:
    """Read one seed's K=selected_k cluster metrics and final test AUC."""

    run_dir = _runner_run_dir(output_root, dataset, configuration, seed)
    latent_dir = run_dir / "latent_clustering" / dataset / "official"
    metrics_path = latent_dir / "cluster_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing cluster metrics for {dataset}/{configuration}/seed_{seed}: "
            f"{metrics_path}"
        )

    metrics = pd.read_csv(metrics_path)
    if "k" not in metrics.columns:
        raise ValueError(f"Missing 'k' column in {metrics_path}")
    selected = metrics[metrics["k"].astype(int) == int(selected_k)]
    if selected.empty:
        raise ValueError(f"No k={selected_k} row in {metrics_path}")
    row = selected.iloc[0]

    required_columns = {
        "silhouette_score": "silhouette",
        "davies_bouldin_index": "dbi",
        "calinski_harabasz_score": "ch",
    }
    missing = [column for column in required_columns if column not in metrics.columns]
    if missing:
        raise ValueError(f"Missing columns in {metrics_path}: {missing}")

    log_path, prediction_log = _latest_prediction_log(run_dir, dataset)
    auc = prediction_log.iloc[-1]["auc"]
    if pd.isna(auc):
        raise ValueError(f"Final AUC is NaN in {log_path}")

    return {
        "dataset": dataset,
        "configuration": configuration,
        "model_seed": int(seed),
        "k": int(selected_k),
        "silhouette": float(row["silhouette_score"]),
        "dbi": float(row["davies_bouldin_index"]),
        "ch": float(row["calinski_harabasz_score"]),
        "auc": float(auc),
        "prediction_log": str(log_path),
    }


def collect_seed_metrics(
    output_root: Path,
    datasets: Sequence[str],
    configurations: Sequence[str],
    seeds: Sequence[int],
    selected_k: int,
) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        for configuration in configurations:
            for seed in seeds:
                rows.append(
                    load_seed_metrics(
                        output_root,
                        dataset,
                        configuration,
                        int(seed),
                        selected_k,
                    )
                )
    return pd.DataFrame(rows)


def _sample_std(values: pd.Series) -> float:
    return float(values.std(ddof=1))


def aggregate_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate independent model seeds using sample standard deviation."""

    required = {
        "dataset",
        "configuration",
        "model_seed",
        "k",
        "silhouette",
        "dbi",
        "ch",
        "auc",
    }
    missing = sorted(required - set(per_seed.columns))
    if missing:
        raise ValueError(f"Missing per-seed columns: {missing}")

    summary = (
        per_seed.groupby(["dataset", "configuration", "k"], as_index=False)
        .agg(
            n_seeds=("model_seed", "nunique"),
            silhouette_mean=("silhouette", "mean"),
            silhouette_std=("silhouette", _sample_std),
            dbi_mean=("dbi", "mean"),
            dbi_std=("dbi", _sample_std),
            ch_mean=("ch", "mean"),
            ch_std=("ch", _sample_std),
            auc_mean=("auc", "mean"),
            auc_std=("auc", _sample_std),
        )
        .sort_values(["dataset", "configuration", "k"])
        .reset_index(drop=True)
    )
    if (summary["n_seeds"] < 2).any():
        raise ValueError("At least two model seeds are required to report std")
    return summary


def _format_mean_std(mean: float, std: float, decimals: int) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def format_summary_cells(summary: pd.DataFrame) -> pd.DataFrame:
    """Create human-readable cells for the camera-ready table."""

    result = summary.copy()
    result["method"] = result["configuration"].map(
        lambda value: METHOD_LABELS.get(value, str(value))
    )
    result["silhouette"] = result.apply(
        lambda row: _format_mean_std(
            row["silhouette_mean"], row["silhouette_std"], 4
        ),
        axis=1,
    )
    result["dbi"] = result.apply(
        lambda row: _format_mean_std(row["dbi_mean"], row["dbi_std"], 4),
        axis=1,
    )
    result["ch"] = result.apply(
        lambda row: _format_mean_std(row["ch_mean"], row["ch_std"], 2),
        axis=1,
    )
    result["auc"] = result.apply(
        lambda row: _format_mean_std(row["auc_mean"], row["auc_std"], 4),
        axis=1,
    )
    return result[
        [
            "dataset",
            "configuration",
            "method",
            "k",
            "n_seeds",
            "silhouette",
            "dbi",
            "ch",
            "auc",
        ]
    ]


def write_summary_outputs(
    per_seed: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_metrics(per_seed)
    table = format_summary_cells(summary)

    per_seed_path = output_dir / "per_seed_metrics.csv"
    summary_path = output_dir / "mean_std_metrics.csv"
    table_path = output_dir / "paper_table.csv"
    markdown_path = output_dir / "paper_table.md"
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    table.to_csv(table_path, index=False)

    lines = [
        "# Model-seed robustness",
        "",
        "Metrics are mean ± sample standard deviation across independent model seeds.",
        "The selected post-hoc cluster granularity is fixed to K=3.",
        "",
    ]
    for dataset in table["dataset"].drop_duplicates().tolist():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Method | Sil. ↑ | DBI ↓ | CH ↑ | AUC ↑ |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        subset = table[table["dataset"] == dataset]
        for _, row in subset.iterrows():
            lines.append(
                f"| {row['method']} | {row['silhouette']} | {row['dbi']} | "
                f"{row['ch']} | {row['auc']} |"
            )
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "per_seed": per_seed_path,
        "numeric_summary": summary_path,
        "paper_table": table_path,
        "markdown": markdown_path,
    }


def build_runner_command(
    python_executable: str,
    config_path: Path,
    datasets: Sequence[str],
    configurations: Sequence[str],
    seeds: Sequence[int],
    gpu: str,
    ddp: bool,
    ddp_gpus: int,
    ddp_master_port: int,
    force: bool,
    dry_run: bool = False,
) -> list[str]:
    command = [
        python_executable,
        "run_camera_ready_correction.py",
        "--config",
        str(config_path),
        "--phase",
        "all",
    ]
    for dataset in datasets:
        command.extend(["--dataset", str(dataset)])
    for configuration in configurations:
        command.extend(["--run-slug", str(configuration)])
    for seed in seeds:
        command.extend(["--seed", str(int(seed))])
    if gpu:
        command.extend(["--gpu", str(gpu)])
    if ddp:
        command.append("--ddp")
    if ddp_gpus is not None:
        command.extend(["--ddp-gpus", str(int(ddp_gpus))])
    if ddp_master_port is not None:
        command.extend(["--ddp-master-port", str(int(ddp_master_port))])
    if force:
        command.append("--force")
    if dry_run:
        command.append("--dry-run")
    return command


def _temporary_config_with_output_root(
    config_path: Path,
    output_root: Path,
    temporary_directory: Path,
    batch_size: int | None = None,
    num_epochs: int | None = None,
) -> Path:
    document = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    settings = document.get("camera_ready_correction")
    if not isinstance(settings, dict):
        train_config = document.get("train_config") or {}
        cl4kt_config = document.get("cl4kt_config") or {}
        settings = {
            "output_root": str(Path(output_root).resolve()),
            "datasets": list(DEFAULT_DATASETS),
            "model_seeds": list(DEFAULT_SEEDS),
            "k_values": [3, 4, 5, 6],
            "selected_k": 3,
            "metric_sample": 5000,
            "tsne_sample": 5000,
            "skip_tsne": False,
            "batch_size": int(train_config.get("batch_size", 512)),
            "eval_batch_size": int(train_config.get("eval_batch_size", 512)),
            "cpu_threads": 4,
            "cpu_affinity": "",
            "num_epochs": int(train_config.get("num_epochs", 300)),
            "device": "cuda",
            "gpu": "0",
            "ddp": False,
            "ddp_gpus": 2,
            "ddp_master_port": 29501,
            "kmeans_seeds": 10,
            "bootstrap_repeats": 10,
            "order_shuffles": 20,
            "configurations": list(ALL_CONFIGURATIONS),
            "mask_prob": float(cl4kt_config.get("mask_prob", 0.2)),
            "crop_prob": float(cl4kt_config.get("crop_prob", 0.3)),
            "permute_prob": float(cl4kt_config.get("permute_prob", 0.3)),
            "replace_prob": float(cl4kt_config.get("replace_prob", 0.3)),
            "reg_cl": float(cl4kt_config.get("reg_cl", 0.1)),
        }
        document["camera_ready_correction"] = settings
    settings["output_root"] = str(Path(output_root).resolve())
    if batch_size is not None:
        settings["batch_size"] = int(batch_size)
    if num_epochs is not None:
        settings["num_epochs"] = int(num_epochs)
    temporary_config = Path(temporary_directory) / "camera_ready_seed_config.yaml"
    temporary_config.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return temporary_config


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default="camera_ready_seed_robustness")
    parser.add_argument(
        "--summary-dir",
        help="Where to write CSV/Markdown summaries; defaults to <output-root>/summary.",
    )
    parser.add_argument("--phase", choices=["all", "aggregate"], default="all")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument(
        "--configurations", default=",".join(DEFAULT_CONFIGURATIONS)
    )
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--selected-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--ddp-gpus", type=int, default=2)
    parser.add_argument("--ddp-master-port", type=int, default=29501)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the delegated training plan without running or aggregating it.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = Path(args.output_root).resolve()
    summary_dir = (
        Path(args.summary_dir).resolve()
        if args.summary_dir
        else output_root / "summary"
    )
    datasets = parse_csv_values(args.datasets, cast=str)
    configurations = parse_csv_values(args.configurations, cast=str)
    seeds = parse_csv_values(args.seeds, cast=int)
    if len(seeds) < 2:
        raise ValueError("At least two model seeds are required to report std")
    if args.selected_k < 2:
        raise ValueError("--selected-k must be at least 2")

    if args.phase == "all":
        config_path = Path(args.config).resolve()
        with tempfile.TemporaryDirectory(prefix="seed_robustness_") as temp_dir:
            runner_config = _temporary_config_with_output_root(
                config_path,
                output_root,
                Path(temp_dir),
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
            )
            command = build_runner_command(
                python_executable=args.python,
                config_path=runner_config,
                datasets=datasets,
                configurations=configurations,
                seeds=seeds,
                gpu=args.gpu,
                ddp=args.ddp,
                ddp_gpus=args.ddp_gpus,
                ddp_master_port=args.ddp_master_port,
                force=args.force,
                dry_run=args.dry_run,
            )
            print("[train/extract]", " ".join(command), flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            if args.dry_run:
                return 0

    per_seed = collect_seed_metrics(
        output_root=output_root,
        datasets=datasets,
        configurations=configurations,
        seeds=seeds,
        selected_k=args.selected_k,
    )
    paths = write_summary_outputs(per_seed, summary_dir)
    for name, path in paths.items():
        print(f"[{name}] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
