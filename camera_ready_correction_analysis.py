#!/usr/bin/env python3
"""Aggregate the full Camera-Ready correction campaign.

This module is post-training analysis only.  It reads the isolated output
tree produced by ``run_camera_ready_correction.py`` and writes new correction
tables under ``experiment_1/camera_ready_correction``.  Existing paper
outputs are never modified.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from run_camera_ready_correction import load_camera_ready_config


PRIMARY_SLUGS = [
    "original",
    "soft",
    "soft_kmeans",
    "soft_separation_temporal",
    "full",
]
METRIC_COLUMNS = [
    "silhouette_score",
    "davies_bouldin_index",
    "calinski_harabasz_score",
    "cluster_size_min_max_ratio",
    "metric_sample_size",
]
PREDICTION_COLUMNS = ["EarlyStopEpoch", "auc", "acc", "rmse"]


def correction_run_dir(output_root, dataset, slug, seed):
    """Return the standard full-campaign run directory.

    The fallback keeps the analyzer useful with the already-produced
    ``camera_ready_outputs`` tree, where the two extra Soft+KMeans seeds were
    stored under ``experiment_2/model_seed_runs``.
    """

    root = Path(output_root)
    preferred = (
        root
        / dataset
        / "experiment_1"
        / "runs"
        / slug
        / f"seed_{int(seed)}"
    )
    if preferred.exists():
        return preferred
    if slug == "soft_kmeans" and int(seed) != 12405:
        fallback = (
            root
            / dataset
            / "experiment_2"
            / "model_seed_runs"
            / f"seed_{int(seed)}"
        )
        if fallback.exists():
            return fallback
    return preferred


def expected_latent_files(latent_dir: Path, k_values):
    latent_dir = Path(latent_dir)
    return [
        latent_dir / "latent_features.npz",
        latent_dir / "latent_metadata.csv",
        latent_dir / "cluster_metrics.csv",
        latent_dir / "kmeans_model.pkl",
        *[latent_dir / f"cluster_assignments_k{k}.csv" for k in k_values],
        *[latent_dir / f"kmeans_model_k{k}.pkl" for k in k_values],
    ]


def validate_artifact_tree(output_root, dataset, slugs, seeds, k_values):
    """Fail before writing analysis output if any required artifact is absent."""

    missing = []
    for slug in slugs:
        for seed in seeds:
            run_dir = correction_run_dir(output_root, dataset, slug, seed)
            latent_dir = run_dir / "latent_clustering" / dataset / "official"
            for path in expected_latent_files(latent_dir, k_values):
                if not path.exists():
                    missing.append(
                        f"dataset={dataset}, configuration={slug}, seed_{seed}: {path}"
                    )
    if missing:
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(
            "Missing Camera-Ready correction artifacts:\n" + preview + suffix
        )


def _latent_dir(output_root, dataset, slug, seed):
    return correction_run_dir(output_root, dataset, slug, seed) / "latent_clustering" / dataset / "official"


def _read_metrics(latent_dir: Path, dataset, slug, seed, k_values):
    metrics = pd.read_csv(latent_dir / "cluster_metrics.csv")
    rows = []
    for k in k_values:
        selected = metrics[metrics["k"].astype(int) == int(k)]
        if selected.empty:
            raise ValueError(f"cluster_metrics.csv has no k={k}: {latent_dir}")
        row = selected.iloc[0].to_dict()
        row.update(
            {
                "dataset": dataset,
                "configuration": slug,
                "model_seed": int(seed),
                "k": int(k),
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in METRIC_COLUMNS:
        if column not in frame:
            raise ValueError(f"Missing metric column {column}: {latent_dir / 'cluster_metrics.csv'}")
    return frame


def _read_assignments(latent_dir: Path, k):
    path = latent_dir / f"cluster_assignments_k{k}.csv"
    frame = pd.read_csv(path)
    required = {"user_id", "step_index", "correct", "cluster"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return frame


def _read_latent_metadata(latent_dir: Path):
    metadata = pd.read_csv(latent_dir / "latent_metadata.csv")
    required = {"user_id", "step_index", "correct"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"{latent_dir / 'latent_metadata.csv'} is missing columns: {missing}")
    return metadata


def _validate_alignment(left, right, context):
    keys = ["user_id", "step_index"]
    left_keys = left[keys].astype(str).reset_index(drop=True)
    right_keys = right[keys].astype(str).reset_index(drop=True)
    if not left_keys.equals(right_keys):
        raise ValueError(f"Timestep ordering mismatch for {context}")


def _aligned_seed_assignments(output_root, dataset, seed_a, seed_b, k):
    left_dir = _latent_dir(output_root, dataset, "soft_kmeans", seed_a)
    right_dir = _latent_dir(output_root, dataset, "soft_kmeans", seed_b)
    left = _read_assignments(left_dir, k)
    right = _read_assignments(right_dir, k)
    _validate_alignment(left, right, f"{dataset}/k={k}/seed_{seed_a} vs seed_{seed_b}")
    return left, right


def _prefix_and_streak_features(frame):
    result = frame.copy()
    result["correct_numeric"] = pd.to_numeric(result["correct"], errors="coerce").fillna(0.0)
    result = result.sort_values(["user_id", "step_index"], kind="stable")
    result["prefix_correctness"] = result.groupby("user_id")["correct_numeric"].transform(
        lambda values: values.expanding().mean()
    )
    result["correct_streak"] = 0
    result["incorrect_streak"] = 0
    for _, indices in result.groupby("user_id", sort=False).groups.items():
        correct_streak = 0
        incorrect_streak = 0
        for index in indices:
            if result.at[index, "correct_numeric"] >= 0.5:
                correct_streak += 1
                incorrect_streak = 0
            else:
                incorrect_streak += 1
                correct_streak = 0
            result.at[index, "correct_streak"] = correct_streak
            result.at[index, "incorrect_streak"] = incorrect_streak
    if "skill_id" in result.columns:
        skill_rate = result.groupby("skill_id")["correct_numeric"].transform("mean")
        result["skill_difficulty"] = 1.0 - skill_rate
    else:
        result["skill_difficulty"] = np.nan
    return result


def _ordered_profiles(assignments):
    features = _prefix_and_streak_features(assignments)
    grouped = features.groupby("cluster", dropna=False)
    profiles = grouped.agg(
        mean_correct=("correct_numeric", "mean"),
        mean_prefix_correctness=("prefix_correctness", "mean"),
        mean_skill_difficulty=("skill_difficulty", "mean"),
        mean_correct_streak=("correct_streak", "mean"),
        mean_incorrect_streak=("incorrect_streak", "mean"),
        n_steps=("cluster", "size"),
    ).reset_index()
    profiles["proportion"] = profiles["n_steps"] / profiles["n_steps"].sum()
    profiles = profiles.sort_values(
        ["mean_prefix_correctness", "mean_correct"], kind="stable"
    ).reset_index(drop=True)
    profiles.insert(0, "ordered_state", np.arange(len(profiles)))
    return profiles


def compute_multiseed_k_sensitivity(output_root, dataset, settings):
    seeds = list(settings["model_seeds"])
    k_values = list(settings["k_values"])
    metric_rows = []
    assignment_cache = {}
    for seed in seeds:
        latent_dir = _latent_dir(output_root, dataset, "soft_kmeans", seed)
        metric_rows.append(
            _read_metrics(latent_dir, dataset, "soft_kmeans", seed, k_values)
        )
        for k in k_values:
            assignment_cache[(seed, k)] = _read_assignments(latent_dir, k)
    metrics = pd.concat(metric_rows, ignore_index=True)
    metrics["silhouette_rank"] = metrics.groupby(["dataset", "model_seed"])[
        "silhouette_score"
    ].rank(ascending=False, method="min")
    metrics["davies_bouldin_rank"] = metrics.groupby(["dataset", "model_seed"])[
        "davies_bouldin_index"
    ].rank(ascending=True, method="min")
    metrics["calinski_harabasz_rank"] = metrics.groupby(["dataset", "model_seed"])[
        "calinski_harabasz_score"
    ].rank(ascending=False, method="min")
    metrics["geometry_rank"] = metrics[
        ["silhouette_rank", "davies_bouldin_rank", "calinski_harabasz_rank"]
    ].mean(axis=1)
    metrics["metric_best_k"] = metrics["geometry_rank"] == metrics.groupby(
        ["dataset", "model_seed"]
    )["geometry_rank"].transform("min")
    metrics["selected_k"] = metrics["k"].eq(int(settings["selected_k"]))

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
            geometry_rank_mean=("geometry_rank", "mean"),
            geometry_rank_std=("geometry_rank", "std"),
            geometry_best_count=("metric_best_k", "sum"),
        )
    )
    summary["geometry_best_fraction"] = summary["geometry_best_count"] / len(seeds)
    summary["selected_k"] = summary["k"].eq(int(settings["selected_k"]))
    summary["metric_best_k"] = summary["geometry_rank_mean"] == summary.groupby(
        "dataset"
    )["geometry_rank_mean"].transform("min")

    stability_rows = []
    profile_rows = []
    for k in k_values:
        for seed_a, seed_b in itertools.combinations(seeds, 2):
            left, right = _aligned_seed_assignments(
                output_root, dataset, seed_a, seed_b, k
            )
            stability_rows.append(
                {
                    "dataset": dataset,
                    "k": int(k),
                    "model_seed_a": int(seed_a),
                    "model_seed_b": int(seed_b),
                    "ari": adjusted_rand_score(left["cluster"], right["cluster"]),
                    "nmi": normalized_mutual_info_score(left["cluster"], right["cluster"]),
                }
            )
            left_profiles = _ordered_profiles(left)
            right_profiles = _ordered_profiles(right)
            profile = {
                "dataset": dataset,
                "k": int(k),
                "model_seed_a": int(seed_a),
                "model_seed_b": int(seed_b),
            }
            for column in [
                "mean_correct",
                "mean_prefix_correctness",
                "mean_skill_difficulty",
                "mean_correct_streak",
                "mean_incorrect_streak",
                "proportion",
            ]:
                left_values = left_profiles[column].to_numpy()
                right_values = right_profiles[column].to_numpy()
                profile[f"{column}_mean_abs_diff"] = float(
                    np.mean(np.abs(left_values - right_values))
                )
                correlation = spearmanr(left_values, right_values).statistic
                profile[f"{column}_spearman"] = (
                    float(correlation) if not pd.isna(correlation) else np.nan
                )
            profile_rows.append(profile)

    return {
        "k_multiseed_metrics": metrics,
        "k_multiseed_summary": summary,
        "k_multiseed_label_stability": pd.DataFrame(stability_rows),
        "k_profile_rank_stability": pd.DataFrame(profile_rows),
    }


def _latest_prediction_log(run_dir: Path, dataset):
    candidates = sorted(
        (run_dir / "train_logs" / dataset).glob("cl4kt_*.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    valid = []
    for path in candidates:
        frame = pd.read_csv(path)
        if set(PREDICTION_COLUMNS).issubset(frame.columns) and not frame.empty:
            valid.append((path, frame))
    if not valid:
        raise FileNotFoundError(
            f"No valid cl4kt prediction log under {run_dir / 'train_logs' / dataset}"
        )
    return valid[-1]


def compute_configuration_k_ablation(output_root, dataset, settings):
    k_values = list(settings["k_values"])
    seeds = list(settings["model_seeds"])
    metric_frames = []
    prediction_rows = []
    for item in settings["configurations"]:
        slug = item["slug"]
        for seed in seeds:
            latent_dir = _latent_dir(output_root, dataset, slug, seed)
            metric_frames.append(
                _read_metrics(latent_dir, dataset, slug, seed, k_values)
            )
            log_path, log_frame = _latest_prediction_log(
                correction_run_dir(output_root, dataset, slug, seed), dataset
            )
            row = log_frame.iloc[-1][PREDICTION_COLUMNS].to_dict()
            row.update(
                {
                    "dataset": dataset,
                    "configuration": slug,
                    "model_seed": int(seed),
                    "source_path": str(log_path),
                }
            )
            prediction_rows.append(row)
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics["geometry_rank_within_configuration_seed"] = metrics.groupby(
        ["dataset", "configuration", "model_seed"]
    )["silhouette_score"].rank(ascending=False, method="min")
    prediction = pd.DataFrame(prediction_rows)
    return {
        "configuration_k_metrics": metrics,
        "configuration_prediction_metrics": prediction,
    }


def _fmt(value):
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _markdown_table(frame, columns, max_rows=None):
    selected = frame[columns].copy()
    if max_rows is not None:
        selected = selected.head(max_rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in selected.iterrows():
        lines.append("| " + " | ".join(_fmt(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _write_report(output_dir, dataset, settings, multi, ablation):
    summary = multi["k_multiseed_summary"].sort_values("k")
    stability = multi["k_multiseed_label_stability"]
    config_metrics = ablation["configuration_k_metrics"]
    prediction = ablation["configuration_prediction_metrics"]
    selected_k = int(settings["selected_k"])
    geometry_best = summary.loc[summary["metric_best_k"], "k"].tolist()
    stable_by_k = stability.groupby("k")["ari"].mean().sort_values(ascending=False)
    stable_k = int(stable_by_k.index[0]) if not stable_by_k.empty else selected_k

    lines = [
        f"# Camera-Ready correction analysis: {dataset}",
        "",
        "This report is generated from the isolated full-seed campaign. It is a post-training analysis and does not change the CL4KT method, training loss, data split, or paper setting.",
        "",
        "## Experimental scope",
        "",
        f"- Configurations: {', '.join(item['slug'] for item in settings['configurations'])}",
        f"- Model seeds: {', '.join(str(seed) for seed in settings['model_seeds'])}",
        f"- Candidate K: {', '.join(str(k) for k in settings['k_values'])}",
        f"- Fixed paper choice: `selected_k={selected_k}`",
        "- The campaign does not add external pedagogical validation or instructor annotations.",
        "",
        "## K sensitivity across model seeds",
        "",
        _markdown_table(
            summary,
            [
                "k",
                "silhouette_mean",
                "silhouette_std",
                "dbi_mean",
                "dbi_std",
                "ch_mean",
                "ch_std",
                "geometry_rank_mean",
                "geometry_best_fraction",
                "selected_k",
            ],
        ),
        "",
        "## Label and profile stability",
        "",
        _markdown_table(
            stability.groupby("k", as_index=False).agg(
                ari_mean=("ari", "mean"),
                ari_std=("ari", "std"),
                nmi_mean=("nmi", "mean"),
                nmi_std=("nmi", "std"),
            ),
            ["k", "ari_mean", "ari_std", "nmi_mean", "nmi_std"],
        ),
        "",
        "## Configuration × K ablation",
        "",
        _markdown_table(
            config_metrics.groupby(["configuration", "k"], as_index=False).agg(
                silhouette_mean=("silhouette_score", "mean"),
                dbi_mean=("davies_bouldin_index", "mean"),
                ch_mean=("calinski_harabasz_score", "mean"),
                balance_mean=("cluster_size_min_max_ratio", "mean"),
            ),
            ["configuration", "k", "silhouette_mean", "dbi_mean", "ch_mean", "balance_mean"],
        ),
        "",
        "## Prediction metrics",
        "",
        _markdown_table(
            prediction.groupby("configuration", as_index=False).agg(
                auc_mean=("auc", "mean"),
                auc_std=("auc", "std"),
                acc_mean=("acc", "mean"),
                rmse_mean=("rmse", "mean"),
                early_stop_epoch_mean=("EarlyStopEpoch", "mean"),
            ),
            ["configuration", "auc_mean", "auc_std", "acc_mean", "rmse_mean", "early_stop_epoch_mean"],
        ),
        "",
        "## Interpretation of K=3",
        "",
        f"For this dataset, the geometry-best K values from the aggregate internal ranking are {geometry_best or ['none']}; the highest mean ARI stability is at K={stable_k}. These are reported as diagnostics, not used to replace the paper setting.",
        f"The paper retains `selected_k={selected_k}` as a compact common vocabulary across datasets and as the state definition used by the accepted paper. K=3 should therefore be described as a jointly interpretable and comparable choice, not as the universally geometry-optimal K.",
        "The discovered labels remain descriptive summaries of the learned representation space. This analysis does not establish diagnostic cognitive states or external pedagogical validity.",
        "",
    ]
    (output_dir / "camera_ready_correction_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    ablation_start = lines.index("## Configuration × K ablation")
    (output_dir / "configuration_k_ablation.md").write_text(
        "\n".join(lines[0:3] + lines[ablation_start:]),
        encoding="utf-8",
    )


def analyze_dataset(source_root, analysis_output_root, dataset, settings, force=False, validate_only=False):
    slugs = [item["slug"] for item in settings["configurations"]]
    seeds = list(settings["model_seeds"])
    k_values = list(settings["k_values"])
    validate_artifact_tree(source_root, dataset, slugs, seeds, k_values)
    if validate_only:
        return None
    output_dir = Path(analysis_output_root) / dataset / "experiment_1" / "camera_ready_correction"
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"{output_dir} exists; use --force to replace only this new analysis directory"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    multi = compute_multiseed_k_sensitivity(source_root, dataset, settings)
    ablation = compute_configuration_k_ablation(source_root, dataset, settings)
    for frame_name, frame in {**multi, **ablation}.items():
        frame.to_csv(output_dir / f"{frame_name}.csv", index=False)
    _write_report(output_dir, dataset, settings, multi, ablation)
    return output_dir


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/camera_ready_correction.yaml")
    parser.add_argument("--source-root", "--input-root", dest="source_root", help="Root containing checkpoints, latent artifacts, and train logs.")
    parser.add_argument("--output-root", dest="legacy_output_root", help="Backward-compatible root used for both input and output when the new split flags are omitted.")
    parser.add_argument("--analysis-output-root", dest="analysis_output_root", help="Separate root for generated correction tables and Markdown reports.")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    settings = load_camera_ready_config(args.config)
    legacy_root = args.legacy_output_root
    source_root = Path(args.source_root or legacy_root or settings["output_root"]).resolve()
    analysis_output_root = Path(args.analysis_output_root or legacy_root or settings["output_root"]).resolve()
    if source_root == analysis_output_root:
        print("[warning] source-root and analysis-output-root are identical; use separate paths when another pipeline is running.", flush=True)
    datasets = args.dataset or settings["datasets"]
    if not args.all_datasets and args.dataset is None:
        datasets = settings["datasets"]
    for dataset in datasets:
        result = analyze_dataset(
            source_root,
            analysis_output_root,
            dataset,
            settings,
            force=args.force,
            validate_only=args.validate_only,
        )
        if args.validate_only:
            print(f"Validated: {dataset}")
        else:
            print(f"Wrote: {result}")


if __name__ == "__main__":
    main()
