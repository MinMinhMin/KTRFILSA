#!/usr/bin/env python3
"""Create geometry and post-hoc interpretability evidence for several K values."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import kruskal
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from cluster_pedagogy_analysis import (
    build_timestep_features,
    cluster_summary,
    markdown_table,
    skill_difficulty_table,
    write_markdown,
)
from extract_and_cluster import parse_k_values, sample_for_metric, validate_selected_k


SUPPORTED_EFFECT_FEATURES = [
    "mean_correct",
    "previous_correct_rate",
    "avg_skill_difficulty",
    "hard_skill_ratio",
    "repeat_skill_ratio",
    "repeat_question_ratio",
    "longest_correct_streak",
    "longest_incorrect_streak",
    "correct_std",
    "median_time_gap_sec",
    "active_days",
]


def load_sequence_config(config_path):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    train_config = config["train_config"]
    return int(train_config["seq_len"]), train_config["sequence_option"]


def load_model_bundle(clustering_dir, k):
    path = Path(clustering_dir) / f"kmeans_model_k{k}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing K-specific model for k={k}: {path}. "
            "Run extract_and_cluster.py with this k value first."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "kmeans" not in bundle:
        raise ValueError(f"Invalid clustering bundle: {path}")
    return bundle


def attach_distance_to_centroid(assignments, latent_features, model_bundle):
    assignments = assignments.copy().reset_index(drop=True)
    latent_features = np.asarray(latent_features, dtype=np.float32)
    if len(assignments) != len(latent_features):
        raise ValueError(
            "Assignment and latent feature row counts differ: "
            f"{len(assignments)} != {len(latent_features)}"
        )
    scaler = model_bundle.get("scaler")
    cluster_features = scaler.transform(latent_features) if scaler is not None else latent_features
    labels = assignments["cluster"].to_numpy(dtype=int)
    kmeans = model_bundle["kmeans"]
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= len(kmeans.cluster_centers_):
        raise ValueError("Assignment contains a cluster outside the model's centroid range")
    centers = kmeans.cluster_centers_[labels]
    assignments["distance_to_centroid"] = np.linalg.norm(cluster_features - centers, axis=1)
    return assignments


def _safe_kruskal(groups):
    groups = [np.asarray(group, dtype=float) for group in groups if len(group)]
    if len(groups) < 2:
        return np.nan
    if all(np.allclose(group, group[0]) for group in groups):
        return 1.0
    return float(kruskal(*groups).pvalue)


def compute_feature_effects(feature_df):
    rows = []
    for k, k_df in feature_df.groupby("k", sort=True):
        for feature in SUPPORTED_EFFECT_FEATURES:
            if feature not in k_df.columns:
                rows.append(
                    {
                        "k": int(k),
                        "feature": feature,
                        "status": "missing",
                        "between_cluster_mean_range": np.nan,
                        "normalized_separation": np.nan,
                        "kruskal_pvalue": np.nan,
                    }
                )
                continue
            numeric = pd.to_numeric(k_df[feature], errors="coerce")
            if not numeric.notna().any():
                rows.append(
                    {
                        "k": int(k),
                        "feature": feature,
                        "status": "missing",
                        "between_cluster_mean_range": np.nan,
                        "normalized_separation": np.nan,
                        "kruskal_pvalue": np.nan,
                    }
                )
                continue
            valid = k_df.loc[numeric.notna(), ["cluster"]].copy()
            valid[feature] = numeric[numeric.notna()].to_numpy()
            groups = [group[feature].to_numpy() for _, group in valid.groupby("cluster")]
            means = valid.groupby("cluster")[feature].mean()
            mean_range = float(means.max() - means.min()) if len(means) else np.nan
            global_std = float(valid[feature].std(ddof=0))
            normalized = mean_range / max(global_std, 1e-12) if np.isfinite(mean_range) else np.nan
            rows.append(
                {
                    "k": int(k),
                    "feature": feature,
                    "status": "available",
                    "between_cluster_mean_range": mean_range,
                    "normalized_separation": normalized,
                    "kruskal_pvalue": _safe_kruskal(groups),
                }
            )
    return pd.DataFrame(rows)


def _geometry_metrics(features, labels, seed, metric_sample):
    metric_features, metric_labels = sample_for_metric(
        features, labels, seed, metric_sample
    )
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    return {
        "metric_sample_size": len(metric_features),
        "silhouette_score": silhouette_score(metric_features, metric_labels),
        "davies_bouldin_index": davies_bouldin_score(metric_features, metric_labels),
        "calinski_harabasz_score": calinski_harabasz_score(
            metric_features, metric_labels
        ),
        "cluster_size_min_max_ratio": float(counts.min() / counts.max()),
    }


def _representative_features(assignments, features):
    keys = [
        key
        for key in ["feature_index", "user_id", "step_index", "sequence_index"]
        if key in assignments.columns and key in features.columns
    ]
    if not keys:
        return assignments.copy()
    feature_columns = [
        column
        for column in features.columns
        if column not in {"cluster", "distance_to_centroid", *keys}
    ]
    return assignments.merge(features[keys + feature_columns], on=keys, how="left")


def write_sensitivity_report(metrics, profiles, effects, selected_k, output_path):
    output_path = Path(output_path)
    metric_best_k = int(
        metrics.loc[metrics["metric_best_k"], "k"].iloc[0]
        if "metric_best_k" in metrics.columns and metrics["metric_best_k"].any()
        else metrics.sort_values(
            ["silhouette_score", "davies_bouldin_index"], ascending=[False, True]
        ).iloc[0]["k"]
    )
    lines = [
        "# Cluster-K Sensitivity and Interpretability",
        "",
        f"Selected K: **{int(selected_k)}**. Geometry-ranked K: **{metric_best_k}**.",
        "",
        "The clustering states are descriptive summaries of model representations "
        "and observed interaction history; they are not diagnoses of true cognitive states.",
        "",
        "## Geometry metrics",
        "",
        "Silhouette and Calinski–Harabasz are higher-is-better; Davies–Bouldin is lower-is-better.",
        "",
        markdown_table(
            metrics[
                [
                    "k",
                    "silhouette_score",
                    "davies_bouldin_index",
                    "calinski_harabasz_score",
                    "cluster_size_min_max_ratio",
                ]
            ].sort_values("k")
        ),
        "",
        "## Profile comparison",
        "",
        markdown_table(
            profiles[
                [
                    "k",
                    "cluster",
                    "n_steps",
                    "n_users",
                    "avg_mean_correct",
                    "avg_avg_skill_difficulty",
                    "avg_correct_rate_delta",
                    "avg_longest_correct_streak",
                    "avg_longest_incorrect_streak",
                    "pedagogical_label_suggestion",
                ]
            ].sort_values(["k", "cluster"])
        ),
        "",
        "## Why K=3",
        "",
    ]

    selected_profiles = profiles[profiles["k"] == selected_k].sort_values("avg_mean_correct")
    if (
        selected_k == 3
        and len(selected_profiles) == 3
        and (selected_profiles["n_steps"] > 0).all()
        and float(selected_profiles["avg_mean_correct"].max() - selected_profiles["avg_mean_correct"].min()) >= 0.10
    ):
        lines.append(
            "K=3 is a defensible compact interpretation because its three supported "
            "clusters form an ordered low/intermediate/high correctness profile while "
            "retaining readable skill-difficulty and streak contrasts. K=4–6 split "
            "this descriptive space into finer subprofiles; those extra distinctions "
            "should be retained only if they add a clear pedagogical meaning."
        )
    else:
        lines.append(
            "The internal evidence does not establish a strong low/intermediate/high "
            "K=3 interpretation under the report thresholds. K=3 remains the selected "
            "paper setting, but this sensitivity result should be described as "
            "inconclusive rather than as external validation."
        )
    lines.extend(
        [
            "",
            "## Feature separation",
            "",
            "`normalized_separation` is the between-cluster mean range divided by the "
            "pooled feature standard deviation; larger values indicate stronger internal "
            "profile separation. Missing features are reported explicitly.",
            "",
            markdown_table(
                effects.sort_values(
                    ["k", "status", "normalized_separation"],
                    ascending=[True, True, False],
                )
            ),
            "",
            "## Limitation",
            "",
            "This analysis remains internal to the CL4KT representation and observed "
            "interaction history. It does not test later course performance, dropout, "
            "or independent instructor judgments, so pedagogical usefulness requires "
            "future external validation.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_sensitivity(
    preprocessed_csv,
    clustering_dir,
    config_path,
    output_dir,
    k_values,
    selected_k=3,
    metric_sample=5000,
    seed=0,
    timestamp_unit="auto",
):
    k_values = [int(k) for k in k_values]
    validate_selected_k(selected_k, k_values)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clustering_dir = Path(clustering_dir)
    df = pd.read_csv(preprocessed_csv, sep="\t")
    latent = np.load(clustering_dir / "latent_features.npz", allow_pickle=True)
    raw_features = latent["features"].astype(np.float32)
    metadata = pd.read_csv(clustering_dir / "latent_metadata.csv")
    if len(raw_features) != len(metadata):
        raise ValueError("latent_features.npz and latent_metadata.csv have different lengths")
    if timestamp_unit == "auto":
        if "timestamp" not in df.columns:
            timestamp_unit = "ordinal"
        else:
            timestamp_unit = (
                "milliseconds"
                if pd.to_numeric(df["timestamp"], errors="coerce").abs().max() > 1e11
                else "seconds"
            )
    timestamp_scale = 1000.0 if timestamp_unit == "milliseconds" else 1.0
    pandas_timestamp_unit = None if timestamp_unit == "ordinal" else (
        "ms" if timestamp_unit == "milliseconds" else "s"
    )
    seq_len, sequence_option = load_sequence_config(config_path)
    skill_stats = skill_difficulty_table(df)

    metric_rows = []
    profile_frames = []
    effect_frames = []
    for k in k_values:
        assignments_path = clustering_dir / f"cluster_assignments_k{k}.csv"
        if not assignments_path.exists():
            raise FileNotFoundError(f"Missing K-specific assignments: {assignments_path}")
        assignments = pd.read_csv(assignments_path)
        bundle = load_model_bundle(clustering_dir, k)
        scaler = bundle.get("scaler")
        cluster_features = scaler.transform(raw_features) if scaler is not None else raw_features
        labels = assignments["cluster"].to_numpy(dtype=int)
        metric_rows.append({"k": k, **_geometry_metrics(cluster_features, labels, seed, metric_sample)})
        assignments = attach_distance_to_centroid(assignments, raw_features, bundle)
        timestep_features = build_timestep_features(
            df,
            assignments,
            skill_stats,
            seq_len,
            sequence_option,
            timestamp_scale,
            pandas_timestamp_unit,
        )
        if timestep_features.empty:
            raise ValueError(f"No pedagogical features generated for k={k}")
        summary = cluster_summary(timestep_features)
        summary["mean_correct"] = summary["avg_mean_correct"]
        summary["avg_skill_difficulty"] = summary["avg_avg_skill_difficulty"]
        summary.insert(0, "k", k)
        profile_frames.append(summary)
        effects = timestep_features.copy()
        effects.insert(0, "k", k)
        effect_frames.append(effects)

        k_dir = output_dir / f"k_{k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        timestep_features.to_csv(k_dir / "timestep_pedagogy_features.csv", index=False)
        summary.drop(columns=["k"]).to_csv(k_dir / "cluster_pedagogy_summary.csv", index=False)
        representatives = pd.read_csv(clustering_dir / f"representatives_k{k}.csv")
        enriched = _representative_features(representatives, timestep_features)
        enriched.to_csv(k_dir / "representatives_pedagogy_enriched.csv", index=False)
        write_markdown(summary.drop(columns=["k"]), enriched, k_dir / "cluster_pedagogy_report.md")

    metrics = pd.DataFrame(metric_rows)
    metric_best_k = int(
        metrics.sort_values(
            ["silhouette_score", "davies_bouldin_index"], ascending=[False, True]
        ).iloc[0]["k"]
    )
    metrics["selected_k"] = metrics["k"].eq(selected_k)
    metrics["metric_best_k"] = metrics["k"].eq(metric_best_k)
    metrics["selected_k_value"] = int(selected_k)
    metrics["metric_best_k_value"] = int(metric_best_k)
    profiles = pd.concat(profile_frames, ignore_index=True)
    effects = compute_feature_effects(pd.concat(effect_frames, ignore_index=True))
    metrics.to_csv(output_dir / "cluster_metrics.csv", index=False)
    profiles.to_csv(output_dir / "interpretability_summary.csv", index=False)
    effects.to_csv(output_dir / "interpretability_feature_effects.csv", index=False)
    write_sensitivity_report(
        metrics,
        profiles,
        effects,
        selected_k,
        output_dir / "interpretability_sensitivity.md",
    )
    return metrics, profiles, effects


def main():
    parser = argparse.ArgumentParser(
        description="Generate geometry and interpretability sensitivity outputs for multiple K values."
    )
    parser.add_argument("--preprocessed_csv", required=True)
    parser.add_argument("--clustering_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k_values", default="3,4,5,6")
    parser.add_argument("--selected_k", type=int, default=3)
    parser.add_argument("--metric_sample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--timestamp_unit",
        choices=["auto", "milliseconds", "seconds", "ordinal"],
        default="auto",
    )
    args = parser.parse_args()
    run_sensitivity(
        preprocessed_csv=args.preprocessed_csv,
        clustering_dir=args.clustering_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        k_values=parse_k_values(args.k_values),
        selected_k=args.selected_k,
        metric_sample=args.metric_sample,
        seed=args.seed,
        timestamp_unit=args.timestamp_unit,
    )
    print(f"Saved K sensitivity outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
