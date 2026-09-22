#!/usr/bin/env python3
"""Run the four paper experiments from one resumable entry point.

The pipeline is intentionally resumable. Every expensive stage writes into an
isolated output directory and is skipped when its required artifacts already
exist, unless --force is supplied.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.linalg import orthogonal_procrustes
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from extract_and_cluster import (
    parse_k_values as parse_cluster_k_values,
    validate_selected_k,
)


EXPERIMENT_1_RUNS = [
    ("original", "Original CL4KT", None),
    ("soft", "Soft", "soft"),
    ("soft_kmeans", "Soft + KMeans", "soft,kmeans"),
    (
        "soft_separation_temporal",
        "Soft + Separation + Temporal",
        "soft,separation,temporal",
    ),
    (
        "full",
        "Full",
        "soft,kmeans,separation,temporal",
    ),
]

PEDAGOGICAL_FEATURES = [
    ("correct", "Current correctness"),
    ("previous_correct_rate", "Prefix correctness"),
    ("skill_difficulty", "Skill difficulty"),
    ("hard_skill_ratio", "Hard-skill ratio"),
    ("skill_entropy", "Skill entropy"),
    ("repeat_skill_ratio", "Repeated skill ratio"),
    ("repeat_question_ratio", "Repeated question ratio"),
    ("longest_correct_streak", "Longest correct streak"),
    ("longest_incorrect_streak", "Longest incorrect streak"),
    ("median_time_gap_sec", "Median time gap"),
    ("active_days", "Active days"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run representation, stability, interpretation, and trajectory experiments."
    )
    parser.add_argument("--output-dir", default="paper_experiment_results")
    parser.add_argument("--dataset", default="XES3G5M")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Launch each training run with torchrun and DistributedDataParallel.",
    )
    parser.add_argument(
        "--ddp-gpus",
        type=int,
        default=2,
        help="Number of processes/visible GPUs used for DDP training.",
    )
    parser.add_argument(
        "--ddp-master-port",
        type=int,
        default=29501,
        help="Rendezvous port for torchrun DDP training.",
    )
    parser.add_argument(
        "--model-seeds",
        default="12405,12406,12407",
        help="At most three comma-separated model seeds.",
    )
    parser.add_argument("--kmeans-seeds", type=int, default=10)
    parser.add_argument("--bootstrap-repeats", type=int, default=10)
    parser.add_argument("--order-shuffles", type=int, default=20)
    parser.add_argument("--k-values", default="3,4,5,6")
    parser.add_argument("--selected-k", type=int, default=3)
    parser.add_argument("--metric-sample", type=int, default=5000)
    parser.add_argument("--tsne-sample", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_seeds(value):
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--model-seeds must contain at least one seed")
    if len(seeds) > 3:
        raise ValueError("At most three model seeds are supported")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--model-seeds contains duplicate values")
    return seeds


def now_text():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output_dir).resolve()
        self.logs = self.root / "logs"
        self.manifest_path = self.root / "manifest.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_manifest()

    def _load_manifest(self):
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {
            "created_at": now_text(),
            "dataset": self.args.dataset,
            "output_dir": str(self.root),
            "stages": {},
        }

    def save_manifest(self):
        self.manifest["updated_at"] = now_text()
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def mark(self, stage, status, **extra):
        self.manifest["stages"][stage] = {
            "status": status,
            "updated_at": now_text(),
            **extra,
        }
        self.save_manifest()

    def run_command(self, stage, command, required_outputs):
        required_outputs = [Path(path) for path in required_outputs]
        if (
            not self.args.force
            and required_outputs
            and all(path.exists() for path in required_outputs)
        ):
            print(f"[resume] {stage}")
            self.mark(stage, "reused", outputs=[str(path) for path in required_outputs])
            return

        printable = " ".join(str(part) for part in command)
        print(f"\n[{stage}]\n{printable}")
        if self.args.dry_run:
            return

        log_path = self.logs / f"{stage.replace('/', '__')}.log"
        self.mark(stage, "running", command=command, log=str(log_path))
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parent,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()
            return_code = process.wait()

        if return_code != 0:
            self.mark(stage, "failed", return_code=return_code, log=str(log_path))
            raise RuntimeError(f"Stage failed: {stage}. See {log_path}")
        missing = [str(path) for path in required_outputs if not path.exists()]
        if missing:
            self.mark(stage, "failed", missing_outputs=missing, log=str(log_path))
            raise FileNotFoundError(f"{stage} did not create: {missing}")
        self.mark(stage, "completed", outputs=[str(path) for path in required_outputs])


def source_root():
    return Path(__file__).resolve().parent


def resolved_config_path(args):
    path = Path(args.config)
    return path if path.is_absolute() else source_root() / path


def resolved_dataset_root(args):
    config = yaml.safe_load(resolved_config_path(args).read_text(encoding="utf-8"))
    path = Path(config["dataset_path"])
    return path if path.is_absolute() else (source_root() / path).resolve()


def dataset_file(args):
    return resolved_dataset_root(args) / args.dataset / "preprocessed_df.csv"


def validate_inputs(args):
    root = source_root()
    required = [
        resolved_config_path(args),
        root / "main.py",
        root / "extract_and_cluster.py",
        root / "cluster_pedagogy_analysis.py",
        root / "trajectory_analysis.py",
        root / "generate_experiment3_visuals.py",
        root / "order_shuffle_control.py",
        root / "analyze_cluster_sensitivity.py",
        dataset_file(args),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")


def split_dir(base, dataset):
    return base / dataset / "official"


def checkpoint_path(run_dir, dataset):
    candidates = sorted(
        (run_dir / "checkpoints" / "cl4kt" / dataset).glob("params_*"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return None
    return candidates[-1]


def training_command(args, run_dir, seed, losses):
    training = [
        args.python,
        "main.py",
        "--config",
        args.config,
        "--model_name",
        "cl4kt",
        "--data_name",
        args.dataset,
        "--seed",
        str(seed),
        "--gpu",
        str(args.gpu),
        "--checkpoint_dir",
        str(run_dir / "checkpoints"),
        "--log_path",
        str(run_dir / "train_logs"),
        "--batch_size",
        str(args.batch_size),
        "--mask_prob",
        "0.5",
        "--crop_prob",
        "0.3",
        "--permute_prob",
        "0.5",
        "--replace_prob",
        "0.5",
        "--reg_cl",
        "0.1",
    ]
    if args.ddp:
        training.append("--ddp")
    if args.num_epochs is not None:
        training.extend(["--num_epochs", str(args.num_epochs)])
    if losses is not None:
        training.extend(
            [
                "--use_joint_training_module",
                "--joint_training_losses",
                losses,
            ]
        )
    if not args.ddp:
        return training
    return [
        args.python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(args.ddp_gpus),
        "--master_port",
        str(args.ddp_master_port),
    ] + training[1:]


def extraction_command(args, run_dir, seed, checkpoint):
    return [
        args.python,
        "extract_and_cluster.py",
        "--config",
        args.config,
        "--data_name",
        args.dataset,
        "--checkpoint",
        str(checkpoint),
        "--output_dir",
        str(run_dir / "latent_clustering"),
        "--k_values",
        ",".join(str(k) for k in args.k_values),
        "--selected_k",
        str(args.selected_k),
        "--metric_sample",
        str(args.metric_sample),
        "--tsne_sample",
        str(args.tsne_sample),
        "--device",
        args.device,
        "--seed",
        str(seed),
    ]


def ensure_model_run(pipeline, run_dir, seed, losses, stage_prefix):
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints" / "cl4kt" / pipeline.args.dataset
    existing = sorted(checkpoint_dir.glob("params_*")) if checkpoint_dir.exists() else []
    expected_checkpoint = existing[-1] if existing else checkpoint_dir / "params_PENDING"
    pipeline.run_command(
        f"{stage_prefix}/train",
        training_command(pipeline.args, run_dir, seed, losses),
        [expected_checkpoint] if existing else [],
    )
    checkpoint = checkpoint_path(run_dir, pipeline.args.dataset)
    if checkpoint is None:
        if pipeline.args.dry_run:
            checkpoint = checkpoint_dir / "params_BEST_EPOCH"
        else:
            raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")

    latent_dir = split_dir(run_dir / "latent_clustering", pipeline.args.dataset)
    pipeline.run_command(
        f"{stage_prefix}/extract",
        extraction_command(pipeline.args, run_dir, seed, checkpoint),
        [
            latent_dir / "latent_features.npz",
            latent_dir / "cluster_metrics.csv",
            latent_dir / "kmeans_model.pkl",
        ]
        + [
            latent_dir / f"cluster_assignments_k{k}.csv"
            for k in pipeline.args.k_values
        ]
        + [latent_dir / f"kmeans_model_k{k}.pkl" for k in pipeline.args.k_values],
    )
    return checkpoint, latent_dir


def metric_sample(features, labels, seed, sample_size):
    if sample_size > 0 and len(features) > sample_size:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(features), size=sample_size, replace=False)
        return features[indices], labels[indices]
    return features, labels


def load_latent_space(latent_dir):
    latent = np.load(latent_dir / "latent_features.npz", allow_pickle=True)
    raw = latent["features"].astype(np.float32)
    labels = latent["labels"].astype(int)
    bundle = joblib.load(latent_dir / "kmeans_model.pkl")
    scaler = bundle.get("scaler") if isinstance(bundle, dict) else None
    features = scaler.transform(raw) if scaler is not None else raw
    metadata = pd.read_csv(latent_dir / "latent_metadata.csv")
    return features.astype(np.float32), labels, metadata


def summarize_experiment_1(root, dataset, seed, metric_sample_size):
    rows = []
    for slug, label, _ in EXPERIMENT_1_RUNS:
        run_dir = root / "experiment_1" / "runs" / slug / f"seed_{seed}"
        latent_dir = split_dir(run_dir / "latent_clustering", dataset)
        features, labels, _ = load_latent_space(latent_dir)
        sample_x, sample_y = metric_sample(
            features, labels, seed=seed, sample_size=metric_sample_size
        )
        counts = np.bincount(labels, minlength=3)
        proportions = counts / counts.sum()
        entropy = float(
            -(proportions * np.log(proportions + 1e-12)).sum() / np.log(len(proportions))
        )
        rows.append(
            {
                "run": slug,
                "configuration": label,
                "seed": seed,
                "n_timesteps": len(labels),
                "silhouette": silhouette_score(sample_x, sample_y),
                "davies_bouldin": davies_bouldin_score(sample_x, sample_y),
                "calinski_harabasz": calinski_harabasz_score(sample_x, sample_y),
                "cluster_0_size": int(counts[0]),
                "cluster_1_size": int(counts[1]),
                "cluster_2_size": int(counts[2]),
                "cluster_0_fraction": proportions[0],
                "cluster_1_fraction": proportions[1],
                "cluster_2_fraction": proportions[2],
                "cluster_size_min_max_ratio": counts.min() / counts.max(),
                "cluster_size_normalized_entropy": entropy,
            }
        )
    output = root / "experiment_1"
    pd.DataFrame(rows).to_csv(output / "representation_quality.csv", index=False)


def analyze_k_sensitivity(latent_dir, output_path, seed=None, metric_sample_size=None):
    """Write the legacy K-sensitivity alias from extractor metrics."""
    metrics_path = Path(latent_dir) / "cluster_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing extractor metrics: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    compatibility = metrics.rename(
        columns={
            "silhouette_score": "silhouette",
            "davies_bouldin_index": "davies_bouldin",
            "calinski_harabasz_score": "calinski_harabasz",
        }
    )
    if "cluster_size_normalized_entropy" not in compatibility:
        entropy_values = []
        for _, row in compatibility.iterrows():
            k = int(row["k"])
            assignments = pd.read_csv(
                Path(latent_dir) / f"cluster_assignments_k{k}.csv"
            )
            proportions = assignments["cluster"].value_counts(normalize=True)
            entropy_values.append(
                float(
                    -(proportions * np.log(proportions + 1e-12)).sum()
                    / np.log(k)
                )
            )
        compatibility["cluster_size_normalized_entropy"] = entropy_values
    compatibility.to_csv(output_path, index=False)


def best_centroid_alignment(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError("Centroid arrays must have the same shape")
    best_mean = math.inf
    best_max = math.inf
    best_perm = None
    for permutation in itertools.permutations(range(len(candidate))):
        distances = np.linalg.norm(reference - candidate[list(permutation)], axis=1)
        mean_distance = float(distances.mean())
        if mean_distance < best_mean:
            best_mean = mean_distance
            best_max = float(distances.max())
            best_perm = permutation
    return best_mean, best_max, best_perm


def pairwise_label_stability(label_sets, prefix):
    rows = []
    keys = list(label_sets)
    for left, right in itertools.combinations(keys, 2):
        rows.append(
            {
                f"{prefix}_a": left,
                f"{prefix}_b": right,
                "ari": adjusted_rand_score(label_sets[left], label_sets[right]),
                "nmi": normalized_mutual_info_score(
                    label_sets[left], label_sets[right]
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[f"{prefix}_a", f"{prefix}_b", "ari", "nmi"],
    )


def analyze_kmeans_and_bootstrap(
    reference_latent_dir,
    output_dir,
    kmeans_seed_count,
    bootstrap_repeats,
    metric_sample_size,
    base_seed,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    features, _, metadata = load_latent_space(reference_latent_dir)
    kmeans_rows = []
    labels_by_seed = {}
    centers_by_seed = {}

    for kmeans_seed in range(kmeans_seed_count):
        model = KMeans(n_clusters=3, random_state=kmeans_seed, n_init=10)
        labels = model.fit_predict(features)
        sample_x, sample_y = metric_sample(
            features,
            labels,
            seed=base_seed,
            sample_size=metric_sample_size,
        )
        labels_by_seed[kmeans_seed] = labels
        centers_by_seed[kmeans_seed] = model.cluster_centers_
        kmeans_rows.append(
            {
                "kmeans_seed": kmeans_seed,
                "silhouette": silhouette_score(sample_x, sample_y),
                "davies_bouldin": davies_bouldin_score(sample_x, sample_y),
                "calinski_harabasz": calinski_harabasz_score(sample_x, sample_y),
            }
        )

    pd.DataFrame(kmeans_rows).to_csv(
        output_dir / "kmeans_seed_metrics.csv", index=False
    )
    pairwise = pairwise_label_stability(labels_by_seed, "seed")
    centroid_rows = []
    for left, right in itertools.combinations(labels_by_seed, 2):
        mean_distance, max_distance, permutation = best_centroid_alignment(
            centers_by_seed[left], centers_by_seed[right]
        )
        centroid_rows.append(
            {
                "seed_a": left,
                "seed_b": right,
                "mean_matched_centroid_distance": mean_distance,
                "max_matched_centroid_distance": max_distance,
                "matching": ",".join(map(str, permutation)),
            }
        )
    pairwise.to_csv(output_dir / "kmeans_pairwise_ari_nmi.csv", index=False)
    pd.DataFrame(centroid_rows).to_csv(
        output_dir / "kmeans_centroid_stability.csv", index=False
    )

    reference_labels = labels_by_seed[0]
    reference_centers = centers_by_seed[0]
    users = metadata["user_id"].to_numpy()
    unique_users = np.unique(users)
    user_to_indices = {
        user: indices
        for user, indices in metadata.groupby("user_id", sort=False).indices.items()
    }
    bootstrap_rows = []
    for repeat in range(bootstrap_repeats):
        rng = np.random.RandomState(base_seed + 1000 + repeat)
        sampled_users = rng.choice(
            unique_users, size=len(unique_users), replace=True
        )
        indices = np.concatenate([user_to_indices[user] for user in sampled_users])
        model = KMeans(
            n_clusters=3,
            random_state=base_seed + repeat,
            n_init=10,
        )
        model.fit(features[indices])
        labels = model.predict(features)
        mean_distance, max_distance, permutation = best_centroid_alignment(
            reference_centers, model.cluster_centers_
        )
        sample_x, sample_y = metric_sample(
            features, labels, base_seed, metric_sample_size
        )
        bootstrap_rows.append(
            {
                "bootstrap_repeat": repeat,
                "sampled_students": len(sampled_users),
                "unique_sampled_students": len(np.unique(sampled_users)),
                "ari_vs_reference": adjusted_rand_score(reference_labels, labels),
                "nmi_vs_reference": normalized_mutual_info_score(
                    reference_labels, labels
                ),
                "mean_matched_centroid_distance": mean_distance,
                "max_matched_centroid_distance": max_distance,
                "matching": ",".join(map(str, permutation)),
                "silhouette": silhouette_score(sample_x, sample_y),
                "davies_bouldin": davies_bouldin_score(sample_x, sample_y),
            }
        )
    pd.DataFrame(bootstrap_rows).to_csv(
        output_dir / "student_bootstrap_stability.csv", index=False
    )


def analyze_model_seed_stability(seed_latent_dirs, output_dir, metric_sample_size):
    label_sets = {}
    rows = []
    reference_metadata = None
    reference_features = None
    reference_centers = None
    reference_seed = min(seed_latent_dirs)
    aligned_centers_by_seed = {}
    common_metric_seed = min(seed_latent_dirs)
    for seed, latent_dir in sorted(seed_latent_dirs.items()):
        features, labels, metadata = load_latent_space(latent_dir)
        identity = metadata[["user_id", "step_index"]].astype(str).agg("::".join, axis=1)
        if reference_metadata is None:
            reference_metadata = identity.to_numpy()
        elif not np.array_equal(reference_metadata, identity.to_numpy()):
            raise ValueError("Model-seed runs do not contain identical timestep ordering")
        label_sets[seed] = labels
        centers = np.stack(
            [features[labels == cluster].mean(axis=0) for cluster in range(3)]
        )
        if seed == reference_seed:
            reference_features = features
            reference_centers = centers
            aligned_centers_by_seed[seed] = centers
        else:
            if reference_features is None or reference_centers is None:
                raise RuntimeError("Reference model seed must be processed first")
            sample_size = min(metric_sample_size, len(features))
            rng = np.random.RandomState(common_metric_seed)
            indices = rng.choice(len(features), size=sample_size, replace=False)
            reference_sample = reference_features[indices]
            candidate_sample = features[indices]
            reference_mean = reference_sample.mean(axis=0, keepdims=True)
            candidate_mean = candidate_sample.mean(axis=0, keepdims=True)
            rotation, _ = orthogonal_procrustes(
                candidate_sample - candidate_mean,
                reference_sample - reference_mean,
            )
            aligned_centers = (
                (centers - candidate_mean) @ rotation + reference_mean
            )
            aligned_centers_by_seed[seed] = aligned_centers
        sample_x, sample_y = metric_sample(
            features, labels, common_metric_seed, metric_sample_size
        )
        rows.append(
            {
                "model_seed": seed,
                "silhouette": silhouette_score(sample_x, sample_y),
                "davies_bouldin": davies_bouldin_score(sample_x, sample_y),
                "calinski_harabasz": calinski_harabasz_score(sample_x, sample_y),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "model_seed_metrics.csv", index=False)
    pairwise_label_stability(label_sets, "model_seed").to_csv(
        output_dir / "model_seed_pairwise_ari_nmi.csv", index=False
    )
    centroid_rows = []
    for seed_a, seed_b in itertools.combinations(sorted(aligned_centers_by_seed), 2):
        mean_distance, max_distance, permutation = best_centroid_alignment(
            aligned_centers_by_seed[seed_a],
            aligned_centers_by_seed[seed_b],
        )
        centroid_rows.append(
            {
                "model_seed_a": seed_a,
                "model_seed_b": seed_b,
                "alignment_reference_seed": reference_seed,
                "alignment": "orthogonal_procrustes",
                "alignment_sample_size": min(metric_sample_size, len(reference_features)),
                "mean_matched_centroid_distance": mean_distance,
                "max_matched_centroid_distance": max_distance,
                "matching": ",".join(map(str, permutation)),
            }
        )
    pd.DataFrame(
        centroid_rows,
        columns=[
            "model_seed_a",
            "model_seed_b",
            "alignment_reference_seed",
            "alignment",
            "alignment_sample_size",
            "mean_matched_centroid_distance",
            "max_matched_centroid_distance",
            "matching",
        ],
    ).to_csv(output_dir / "model_seed_centroid_stability.csv", index=False)


def summarize_stability(output_dir):
    rows = []
    for source, metrics_filename, pairwise_filename in [
        ("kmeans_seed", "kmeans_seed_metrics.csv", "kmeans_pairwise_ari_nmi.csv"),
        ("model_seed", "model_seed_metrics.csv", "model_seed_pairwise_ari_nmi.csv"),
    ]:
        frame = pd.read_csv(output_dir / metrics_filename)
        pairwise = pd.read_csv(output_dir / pairwise_filename)
        row = {"source": source, "n_runs": len(frame), "n_comparisons": len(pairwise)}
        for metric in ["silhouette", "davies_bouldin"]:
            if metric in frame:
                row[f"{metric}_mean"] = frame[metric].mean()
                row[f"{metric}_std"] = frame[metric].std(ddof=1)
                row[f"{metric}_variance"] = frame[metric].var(ddof=1)
        row["ari_mean"] = pairwise["ari"].mean() if len(pairwise) else np.nan
        row["ari_std"] = pairwise["ari"].std(ddof=1) if len(pairwise) else np.nan
        row["nmi_mean"] = pairwise["nmi"].mean() if len(pairwise) else np.nan
        row["nmi_std"] = pairwise["nmi"].std(ddof=1) if len(pairwise) else np.nan
        rows.append(row)

    kmeans_centroids = pd.read_csv(
        output_dir / "kmeans_centroid_stability.csv"
    )
    model_centroids = pd.read_csv(
        output_dir / "model_seed_centroid_stability.csv"
    )
    for row in rows:
        if row["source"] == "kmeans_seed":
            row["centroid_distance_mean"] = kmeans_centroids[
                "mean_matched_centroid_distance"
            ].mean()
            row["centroid_distance_std"] = kmeans_centroids[
                "mean_matched_centroid_distance"
            ].std(ddof=1)
        elif row["source"] == "model_seed":
            row["centroid_distance_mean"] = model_centroids[
                "mean_matched_centroid_distance"
            ].mean()
            row["centroid_distance_std"] = model_centroids[
                "mean_matched_centroid_distance"
            ].std(ddof=1)

    bootstrap = pd.read_csv(output_dir / "student_bootstrap_stability.csv")
    rows.append(
        {
            "source": "student_bootstrap",
            "n_runs": len(bootstrap),
            "n_comparisons": len(bootstrap),
            "silhouette_mean": bootstrap["silhouette"].mean(),
            "silhouette_std": bootstrap["silhouette"].std(ddof=1),
            "silhouette_variance": bootstrap["silhouette"].var(ddof=1),
            "davies_bouldin_mean": bootstrap["davies_bouldin"].mean(),
            "davies_bouldin_std": bootstrap["davies_bouldin"].std(ddof=1),
            "davies_bouldin_variance": bootstrap["davies_bouldin"].var(ddof=1),
            "ari_mean": bootstrap["ari_vs_reference"].mean(),
            "ari_std": bootstrap["ari_vs_reference"].std(ddof=1),
            "nmi_mean": bootstrap["nmi_vs_reference"].mean(),
            "nmi_std": bootstrap["nmi_vs_reference"].std(ddof=1),
            "centroid_distance_mean": bootstrap[
                "mean_matched_centroid_distance"
            ].mean(),
            "centroid_distance_std": bootstrap[
                "mean_matched_centroid_distance"
            ].std(ddof=1),
        }
    )
    pd.DataFrame(rows).to_csv(output_dir / "stability_summary.csv", index=False)


def safe_kruskal(groups):
    groups = [group for group in groups if len(group)]
    if len(groups) < 2:
        return np.nan, np.nan
    if all(np.all(group == group[0]) for group in groups):
        return 0.0, 1.0
    result = kruskal(*groups)
    return float(result.statistic), float(result.pvalue)


def analyze_pedagogy(feature_path, output_dir, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(feature_path)
    available = [
        (column, label)
        for column, label in PEDAGOGICAL_FEATURES
        if column in df
        and pd.to_numeric(df[column], errors="coerce").notna().any()
    ]
    profile_rows = []
    compact_rows = []
    test_rows = []

    for cluster, cluster_df in df.groupby("cluster"):
        compact = {
            "cluster": int(cluster),
            "n_steps": len(cluster_df),
            "n_users": cluster_df["user_id"].nunique(),
        }
        for column, label in available:
            values = pd.to_numeric(cluster_df[column], errors="coerce").dropna()
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            profile_rows.append(
                {
                    "cluster": int(cluster),
                    "feature": column,
                    "feature_label": label,
                    "n": len(values),
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "median": values.median(),
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1,
                }
            )
            compact[column] = (
                f"{values.mean():.4f} ± {values.std(ddof=1):.4f}; "
                f"{values.median():.4f} [{q1:.4f}, {q3:.4f}]"
            )
        compact_rows.append(compact)

    rng = np.random.RandomState(seed)
    for column, label in available:
        clean = df[["cluster", column]].copy()
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        clean = clean.dropna()
        groups = [
            group[column].to_numpy()
            for _, group in clean.groupby("cluster")
        ]
        statistic, p_value = safe_kruskal(groups)
        n = len(clean)
        k = clean["cluster"].nunique()
        epsilon_squared = (
            max(0.0, (statistic - k + 1) / (n - k))
            if n > k and not pd.isna(statistic)
            else np.nan
        )
        if len(clean) > 50000:
            indices = rng.choice(len(clean), size=50000, replace=False)
            mi_df = clean.iloc[indices]
        else:
            mi_df = clean
        if mi_df.empty or mi_df["cluster"].nunique() < 2:
            mi = np.nan
        else:
            discrete = column == "correct"
            mi = mutual_info_classif(
                mi_df[[column]].to_numpy(),
                mi_df["cluster"].to_numpy(),
                discrete_features=discrete,
                random_state=seed,
            )[0]
        test_rows.append(
            {
                "feature": column,
                "feature_label": label,
                "n": n,
                "kruskal_h": statistic,
                "p_value": p_value,
                "epsilon_squared": epsilon_squared,
                "mutual_information": mi,
                "mi_sample_size": len(mi_df),
            }
        )

    pd.DataFrame(profile_rows).to_csv(
        output_dir / "pedagogical_profile_long.csv", index=False
    )
    pd.DataFrame(compact_rows).to_csv(
        output_dir / "pedagogical_profile_table.csv", index=False
    )
    pd.DataFrame(test_rows).sort_values(
        "epsilon_squared", ascending=False
    ).to_csv(output_dir / "pedagogical_statistical_tests.csv", index=False)


def normalized_entropy(labels, n_states):
    if len(labels) == 0:
        return np.nan
    probabilities = np.bincount(labels, minlength=n_states).astype(float)
    probabilities /= probabilities.sum()
    nonzero = probabilities > 0
    value = -(probabilities[nonzero] * np.log2(probabilities[nonzero])).sum()
    return float(value / np.log2(n_states)) if n_states > 1 else 0.0


def dwell_lengths(labels):
    if len(labels) == 0:
        return []
    lengths = []
    current = labels[0]
    length = 1
    for label in labels[1:]:
        if label == current:
            length += 1
        else:
            lengths.append((int(current), length))
            current = label
            length = 1
    lengths.append((int(current), length))
    return lengths


def plot_case_study(user_df, case_type, output_path, state_names):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ordered = user_df.sort_values("step_index")
    rolling = ordered["correct"].rolling(10, min_periods=1).mean()
    axes[0].plot(ordered["step_index"], rolling, color="#2667ff", linewidth=2)
    axes[0].scatter(
        ordered["step_index"],
        ordered["correct"],
        s=12,
        alpha=0.35,
        color="#172033",
    )
    axes[0].set_ylabel("Correctness")
    axes[0].set_ylim(-0.08, 1.08)
    axes[0].set_title(f"{case_type}: user {ordered['user_id'].iloc[0]}")

    axes[1].step(
        ordered["step_index"],
        ordered["ordered_state"],
        where="post",
        color="#d1495b",
        linewidth=1.8,
    )
    axes[1].scatter(
        ordered["step_index"],
        ordered["ordered_state"],
        c=ordered["ordered_state"],
        cmap="viridis",
        s=20,
    )
    axes[1].set_yticks(range(len(state_names)), state_names)
    axes[1].set_ylabel("Learning state")
    axes[1].set_xlabel("Timestep")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_ordered_transition_heatmap(probabilities, state_names, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    matrix = pd.DataFrame(probabilities, index=state_names, columns=state_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Transition probability"},
        ax=ax,
    )
    ax.set_title("State Transition Probability Matrix")
    ax.set_xlabel("Next learning state")
    ax.set_ylabel("Current learning state")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyze_trajectories(states_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    states = pd.read_csv(states_path).sort_values(["user_id", "step_index"])
    cluster_order = (
        states.groupby("cluster_label")["correct"]
        .mean()
        .sort_values()
        .reset_index(name="current_correctness")
    )
    cluster_order["ordered_state"] = np.arange(len(cluster_order))
    names = ["Needs support", "Developing", "High-stable"]
    cluster_order["state_name"] = [
        names[index] if index < len(names) else f"State {index}"
        for index in cluster_order["ordered_state"]
    ]
    cluster_order.to_csv(output_dir / "state_order.csv", index=False)
    mapping = dict(
        zip(cluster_order["cluster_label"], cluster_order["ordered_state"])
    )
    states["ordered_state"] = states["cluster_label"].map(mapping).astype(int)
    n_states = len(mapping)

    counts = np.zeros((n_states, n_states), dtype=np.int64)
    dwell_rows = []
    user_rows = []
    stage_rows = []
    for user_id, user_df in states.groupby("user_id"):
        labels = user_df["ordered_state"].to_numpy(dtype=int)
        for source, target in zip(labels[:-1], labels[1:]):
            counts[source, target] += 1
        runs = dwell_lengths(labels)
        for state, length in runs:
            dwell_rows.append(
                {"user_id": user_id, "state": state, "dwell_time": length}
            )

        n = len(user_df)
        thirds = np.array_split(np.arange(n), 3)
        stage_names = ["early", "middle", "late"]
        for stage_name, indices in zip(stage_names, thirds):
            stage_labels = labels[indices]
            for state in range(n_states):
                stage_rows.append(
                    {
                        "user_id": user_id,
                        "stage": stage_name,
                        "state": state,
                        "fraction": float(np.mean(stage_labels == state))
                        if len(stage_labels)
                        else np.nan,
                    }
                )

        first = user_df.iloc[: max(1, n // 3)]["correct"].mean()
        last = user_df.iloc[-max(1, n // 3) :]["correct"].mean()
        changes = float(np.mean(labels[1:] != labels[:-1])) if n > 1 else 0.0
        user_rows.append(
            {
                "user_id": user_id,
                "n_steps": n,
                "early_correctness": first,
                "late_correctness": last,
                "correctness_delta": last - first,
                "state_change_rate": changes,
                "state_entropy": normalized_entropy(labels, n_states),
            }
        )

    probabilities = np.divide(
        counts,
        counts.sum(axis=1, keepdims=True),
        out=np.zeros_like(counts, dtype=float),
        where=counts.sum(axis=1, keepdims=True) > 0,
    )
    state_names = cluster_order.sort_values("ordered_state")["state_name"].tolist()
    pd.DataFrame(counts, index=state_names, columns=state_names).to_csv(
        output_dir / "transition_counts_ordered.csv"
    )
    pd.DataFrame(probabilities, index=state_names, columns=state_names).to_csv(
        output_dir / "transition_matrix_ordered.csv"
    )
    plot_ordered_transition_heatmap(
        probabilities,
        state_names,
        output_dir / "transition_heatmap_ordered.png",
    )

    transition_rows = []
    for state in range(n_states):
        row = probabilities[state]
        nonzero = row > 0
        entropy = float(-(row[nonzero] * np.log2(row[nonzero])).sum())
        transition_rows.append(
            {
                "state": state,
                "state_name": state_names[state],
                "self_transition_probability": probabilities[state, state],
                "transition_entropy": entropy,
            }
        )
    pd.DataFrame(transition_rows).to_csv(
        output_dir / "transition_metrics.csv", index=False
    )

    dwell_df = pd.DataFrame(dwell_rows)
    dwell_df.to_csv(output_dir / "dwell_times.csv", index=False)
    dwell_df.groupby("state")["dwell_time"].agg(
        ["count", "mean", "std", "median"]
    ).reset_index().to_csv(output_dir / "dwell_time_summary.csv", index=False)

    stage_df = pd.DataFrame(stage_rows)
    stage_df.groupby(["stage", "state"])["fraction"].mean().reset_index().to_csv(
        output_dir / "stage_state_distribution.csv", index=False
    )
    early = (
        stage_df[stage_df["stage"] == "early"]
        .groupby("state")["fraction"]
        .mean()
    )
    late = (
        stage_df[stage_df["stage"] == "late"]
        .groupby("state")["fraction"]
        .mean()
    )
    delta = pd.DataFrame(
        {
            "state": range(n_states),
            "early_fraction": [early.get(state, 0.0) for state in range(n_states)],
            "late_fraction": [late.get(state, 0.0) for state in range(n_states)],
        }
    )
    delta["late_minus_early"] = delta["late_fraction"] - delta["early_fraction"]
    delta["total_variation_contribution"] = (
        0.5 * delta["late_minus_early"].abs()
    )
    delta.to_csv(output_dir / "early_late_distribution_difference.csv", index=False)

    transition_total = counts.sum()
    overall_self_transition = (
        float(np.trace(counts) / transition_total) if transition_total else np.nan
    )
    source_weights = counts.sum(axis=1).astype(float)
    if source_weights.sum():
        source_weights /= source_weights.sum()
    transition_entropy_values = np.asarray(
        [row["transition_entropy"] for row in transition_rows], dtype=float
    )

    user_df = pd.DataFrame(user_rows)
    eligible = user_df[user_df["n_steps"] >= 30].copy()
    if eligible.empty:
        eligible = user_df.copy()
    improving = eligible.sort_values("correctness_delta", ascending=False).iloc[0]
    declining = eligible.sort_values("correctness_delta").iloc[0]
    oscillating_pool = eligible[
        ~eligible["user_id"].isin([improving["user_id"], declining["user_id"]])
    ].copy()
    if oscillating_pool.empty:
        oscillating_pool = eligible.copy()
    oscillating_pool["oscillation_score"] = (
        oscillating_pool["state_change_rate"]
        + oscillating_pool["state_entropy"]
        - oscillating_pool["correctness_delta"].abs()
    )
    oscillating = oscillating_pool.sort_values(
        "oscillation_score", ascending=False
    ).iloc[0]
    cases = pd.DataFrame(
        [
            {"case_type": "improving", **improving.to_dict()},
            {"case_type": "oscillating", **oscillating.to_dict()},
            {"case_type": "declining", **declining.to_dict()},
        ]
    )
    cases.to_csv(output_dir / "case_studies.csv", index=False)
    for _, case in cases.iterrows():
        if pd.api.types.is_numeric_dtype(states["user_id"]):
            selected = states[
                pd.to_numeric(states["user_id"], errors="coerce")
                == float(case["user_id"])
            ]
        else:
            selected = states[
                states["user_id"].astype(str) == str(case["user_id"])
            ]
        plot_case_study(
            selected,
            case["case_type"],
            output_dir / f"case_{case['case_type']}_user_{case['user_id']}.png",
            state_names,
        )
    pd.DataFrame(
        [
            {
                "n_students": states["user_id"].nunique(),
                "n_timesteps": len(states),
                "overall_self_transition_probability": overall_self_transition,
                "weighted_transition_entropy": float(
                    np.sum(source_weights * transition_entropy_values)
                ),
                "average_dwell_time": dwell_df["dwell_time"].mean(),
                "median_dwell_time": dwell_df["dwell_time"].median(),
                "early_late_total_variation_distance": delta[
                    "total_variation_contribution"
                ].sum(),
            }
        ]
    ).to_csv(output_dir / "trajectory_summary.csv", index=False)


def write_run_config(root, args, seeds):
    config = {
        "dataset": args.dataset,
        "model_seeds": seeds,
        "kmeans_seed_count": args.kmeans_seeds,
        "bootstrap_repeats": args.bootstrap_repeats,
        "metric_sample": args.metric_sample,
        "device": args.device,
        "gpu": args.gpu,
        "ddp": getattr(args, "ddp", False),
        "ddp_gpus": getattr(args, "ddp_gpus", 1),
        "ddp_master_port": getattr(args, "ddp_master_port", 29501),
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "training_defaults": {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "contrastive_weight": 0.1,
            "mask_probability": 0.5,
            "crop_probability": 0.3,
            "permute_probability": 0.5,
            "replace_probability": 0.5,
            "early_stopping_patience": 10,
        },
        "joint_clustering_defaults": {
            "outer_weight": 0.05,
            "soft_weight": 1.0,
            "kmeans_weight": 0.1,
            "separation_weight": 0.01,
            "temporal_weight": 0.01,
            "student_t_alpha": 1.0,
            "l2_normalize_training_states": True,
        },
        "posthoc_k_values": list(args.k_values),
        "selected_k": int(args.selected_k),
        "metric_best_k": None,
        "sensitivity_outputs": [
            "experiment_1/k_sensitivity/cluster_metrics.csv",
            "experiment_1/k_sensitivity/interpretability_summary.csv",
            "experiment_1/k_sensitivity/interpretability_feature_effects.csv",
            "experiment_1/k_sensitivity/interpretability_sensitivity.md",
        ],
        "order_shuffle_repetitions": args.order_shuffles,
        "order_shuffle_base_seed": seeds[0],
        "experiment_1_runs": [
            {"slug": slug, "label": label, "losses": losses}
            for slug, label, losses in EXPERIMENT_1_RUNS
        ],
        "reuse_policy": (
            "Experiment 1 Soft+KMeans seed 0 is reused by Experiments 2, 3, and 4. "
            "Only the remaining model seeds are trained for Experiment 2."
        ),
        "stability_model": "Soft + KMeans",
        "trajectory_scope": (
            "Per-timestep states from the configured CL4KT sequence window "
            "(the packaged paper configuration uses the 100 most recent interactions)."
        ),
    }
    config_path = root / "run_config.json"
    if config_path.exists() and not args.force:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        critical_keys = [
            "dataset",
            "model_seeds",
            "kmeans_seed_count",
            "bootstrap_repeats",
            "order_shuffle_repetitions",
            "metric_sample",
            "batch_size",
            "num_epochs",
            "posthoc_k_values",
            "selected_k",
            "ddp",
            "ddp_gpus",
            "ddp_master_port",
        ]
        changed = [
            key for key in critical_keys if existing.get(key) != config.get(key)
        ]
        if changed:
            raise ValueError(
                "Output directory already belongs to a different configuration "
                f"({', '.join(changed)} changed). Use another --output-dir or --force."
            )
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_dry_run(args, seeds):
    total_trainings = len(EXPERIMENT_1_RUNS) + len(seeds) - 1
    print("Dry-run plan")
    print(f"- Output: {Path(args.output_dir).resolve()}")
    print(
        f"- Experiment 1: {len(EXPERIMENT_1_RUNS)} model trainings + "
        f"K sensitivity (K={','.join(map(str, args.k_values))}), selected K={args.selected_k}"
    )
    print(
        f"- Experiment 2: {args.kmeans_seeds} KMeans seeds, "
        f"{args.bootstrap_repeats} student bootstraps, "
        f"{len(seeds)} model seeds ({len(seeds) - 1} additional trainings)"
    )
    print(
        "- Experiment 3: pedagogy profiles + Kruskal–Wallis/effect size/MI + "
        f"order-shuffle control ({args.order_shuffles} shuffles) + figures"
    )
    print(
        "- Experiment 4: transitions, ordered heatmap, dwell, entropy, stages, "
        "3 case studies"
    )
    print(f"- Total model trainings: {total_trainings}")


def main():
    args = parse_args()
    if isinstance(args.k_values, str):
        args.k_values = parse_cluster_k_values(args.k_values)
    validate_selected_k(args.selected_k, args.k_values)
    seeds = parse_seeds(args.model_seeds)
    if args.kmeans_seeds < 2:
        raise ValueError("--kmeans-seeds must be at least 2")
    if args.bootstrap_repeats < 1:
        raise ValueError("--bootstrap-repeats must be at least 1")
    if args.order_shuffles < 1:
        raise ValueError("--order-shuffles must be at least 1")
    if args.ddp_gpus < 1:
        raise ValueError("--ddp-gpus must be at least 1")
    if args.ddp and "," not in str(args.gpu) and args.ddp_gpus > 1:
        raise ValueError("DDP with multiple processes requires --gpu such as 0,1")
    validate_inputs(args)
    pipeline = Pipeline(args)
    write_run_config(pipeline.root, args, seeds)

    if args.dry_run:
        print_dry_run(args, seeds)
        return

    primary_seed = seeds[0]
    experiment_1_latents = {}
    for slug, _, losses in EXPERIMENT_1_RUNS:
        run_dir = (
            pipeline.root
            / "experiment_1"
            / "runs"
            / slug
            / f"seed_{primary_seed}"
        )
        _, latent_dir = ensure_model_run(
            pipeline,
            run_dir,
            primary_seed,
            losses,
            f"experiment_1/{slug}/seed_{primary_seed}",
        )
        experiment_1_latents[slug] = latent_dir
    summarize_experiment_1(
        pipeline.root, args.dataset, primary_seed, args.metric_sample
    )
    k_sensitivity_path = pipeline.root / "experiment_1" / "k_sensitivity.csv"
    if args.force or not k_sensitivity_path.exists():
        analyze_k_sensitivity(
            experiment_1_latents["soft_kmeans"],
            k_sensitivity_path,
        )
    sensitivity_dir = pipeline.root / "experiment_1" / "k_sensitivity"
    sensitivity_outputs = [
        sensitivity_dir / "cluster_metrics.csv",
        sensitivity_dir / "interpretability_summary.csv",
        sensitivity_dir / "interpretability_feature_effects.csv",
        sensitivity_dir / "interpretability_sensitivity.md",
    ]
    pipeline.run_command(
        "experiment_1/k_sensitivity",
        [
            args.python,
            "analyze_cluster_sensitivity.py",
            "--preprocessed_csv",
            str(dataset_file(args)),
            "--clustering_dir",
            str(experiment_1_latents["soft_kmeans"]),
            "--config",
            args.config,
            "--output_dir",
            str(sensitivity_dir),
            "--k_values",
            ",".join(str(k) for k in args.k_values),
            "--selected_k",
            str(args.selected_k),
            "--metric_sample",
            str(args.metric_sample),
            "--seed",
            str(primary_seed),
            "--timestamp_unit",
            "ordinal" if args.dataset.upper() == "ASSISTMENT2009" else "auto",
        ],
        sensitivity_outputs,
    )
    if all(path.exists() for path in sensitivity_outputs):
        sensitivity_metrics = pd.read_csv(sensitivity_outputs[0])
        metric_best_rows = sensitivity_metrics[
            sensitivity_metrics["metric_best_k"]
        ]
        metric_best_k = int(metric_best_rows.iloc[0]["k"])
        run_config_path = pipeline.root / "run_config.json"
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        run_config["metric_best_k"] = metric_best_k
        run_config_path.write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        pipeline.mark(
            "experiment_1/k_sensitivity",
            "completed",
            outputs=[str(path) for path in sensitivity_outputs],
            metric_best_k=metric_best_k,
            selected_k=args.selected_k,
        )
    pipeline.mark(
        "experiment_1/summary",
        "completed",
        output=str(pipeline.root / "experiment_1" / "representation_quality.csv"),
    )

    experiment_2_dir = pipeline.root / "experiment_2"
    experiment_2_dir.mkdir(parents=True, exist_ok=True)
    model_seed_latents = {primary_seed: experiment_1_latents["soft_kmeans"]}
    for seed in seeds[1:]:
        run_dir = experiment_2_dir / "model_seed_runs" / f"seed_{seed}"
        _, latent_dir = ensure_model_run(
            pipeline,
            run_dir,
            seed,
            "soft,kmeans",
            f"experiment_2/model_seed_{seed}",
        )
        model_seed_latents[seed] = latent_dir
    kmeans_outputs = [
        experiment_2_dir / "kmeans_seed_metrics.csv",
        experiment_2_dir / "kmeans_pairwise_ari_nmi.csv",
        experiment_2_dir / "kmeans_centroid_stability.csv",
        experiment_2_dir / "student_bootstrap_stability.csv",
    ]
    if args.force or not all(path.exists() for path in kmeans_outputs):
        analyze_kmeans_and_bootstrap(
            experiment_1_latents["soft_kmeans"],
            experiment_2_dir,
            args.kmeans_seeds,
            args.bootstrap_repeats,
            args.metric_sample,
            primary_seed,
        )
    else:
        print("[resume] experiment_2 KMeans/bootstrap analysis")

    model_seed_outputs = [
        experiment_2_dir / "model_seed_metrics.csv",
        experiment_2_dir / "model_seed_pairwise_ari_nmi.csv",
        experiment_2_dir / "model_seed_centroid_stability.csv",
    ]
    if args.force or not all(path.exists() for path in model_seed_outputs):
        analyze_model_seed_stability(
            model_seed_latents, experiment_2_dir, args.metric_sample
        )
    else:
        print("[resume] experiment_2 model-seed analysis")
    summarize_stability(experiment_2_dir)
    pipeline.mark("experiment_2/analysis", "completed", output=str(experiment_2_dir))

    experiment_3_dir = pipeline.root / "experiment_3"
    pedagogy_features = (
        experiment_1_latents["soft_kmeans"] / "timestep_pedagogy_features.csv"
    )
    pipeline.run_command(
        "experiment_3/build_pedagogy_features",
        [
            args.python,
            "cluster_pedagogy_analysis.py",
            "--preprocessed_csv",
            str(dataset_file(args)),
            "--clustering_dir",
            str(experiment_1_latents["soft_kmeans"]),
            "--config",
            args.config,
        ]
        + (
            ["--timestamp_unit", "ordinal"]
            if args.dataset.upper() == "ASSISTMENT2009"
            else []
        ),
        [pedagogy_features],
    )
    pedagogy_outputs = [
        experiment_3_dir / "pedagogical_profile_long.csv",
        experiment_3_dir / "pedagogical_profile_table.csv",
        experiment_3_dir / "pedagogical_statistical_tests.csv",
    ]
    if args.force or not all(path.exists() for path in pedagogy_outputs):
        analyze_pedagogy(pedagogy_features, experiment_3_dir, primary_seed)
    else:
        print("[resume] experiment_3 analysis")
    temporal_mode = "ordinal" if args.dataset.upper() == "ASSISTMENT2009" else "time"
    pipeline.run_command(
        "experiment_3/figures",
        [
            args.python,
            "generate_experiment3_visuals.py",
            "--root",
            str(experiment_3_dir),
            "--dataset",
            args.dataset,
            "--temporal-mode",
            temporal_mode,
        ],
        [
            experiment_3_dir / "figures" / "cluster_profile_cards.png",
            experiment_3_dir / "figures" / "cluster_profile_heatmap.png",
            experiment_3_dir / "figures" / "feature_effect_size_ranking.png",
            experiment_3_dir / "figures" / "difficulty_time_comparison.png",
            experiment_3_dir / "cluster_interpretation_summary.csv",
        ],
    )
    pipeline.mark("experiment_3/analysis", "completed", output=str(experiment_3_dir))

    reference_run = (
        pipeline.root
        / "experiment_1"
        / "runs"
        / "soft_kmeans"
        / f"seed_{primary_seed}"
    )
    reference_checkpoint = checkpoint_path(reference_run, args.dataset)
    order_control_dir = experiment_3_dir / "order_shuffle_control"
    pipeline.run_command(
        "experiment_3/order_shuffle_control",
        [
            args.python,
            "order_shuffle_control.py",
            "--config",
            args.config,
            "--data_name",
            args.dataset,
            "--checkpoint",
            str(reference_checkpoint),
            "--assignments_csv",
            str(
                experiment_1_latents["soft_kmeans"]
                / "cluster_assignments_k3.csv"
            ),
            "--cluster_model",
            str(experiment_1_latents["soft_kmeans"] / "kmeans_model.pkl"),
            "--latent_features",
            str(experiment_1_latents["soft_kmeans"] / "latent_features.npz"),
            "--output_dir",
            str(order_control_dir),
            "--num_shuffles",
            str(args.order_shuffles),
            "--seed",
            str(primary_seed),
            "--batch_size",
            str(args.batch_size),
            "--device",
            args.device,
        ],
        [
            order_control_dir / "order_shuffle_runs.csv",
            order_control_dir / "order_shuffle_per_user.csv",
            order_control_dir / "order_shuffle_summary.csv",
            order_control_dir / "identity_checkpoint_verification.csv",
        ],
    )
    experiment_4_dir = pipeline.root / "experiment_4"
    trajectory_root = experiment_4_dir / "trajectory"
    trajectory_dir = split_dir(trajectory_root, args.dataset)
    pipeline.run_command(
        "experiment_4/extract_trajectories",
        [
            args.python,
            "trajectory_analysis.py",
            "--config",
            args.config,
            "--data_name",
            args.dataset,
            "--checkpoint",
            str(reference_checkpoint),
            "--cluster_model",
            str(experiment_1_latents["soft_kmeans"] / "kmeans_model.pkl"),
            "--output_dir",
            str(trajectory_root),
            "--device",
            args.device,
            "--seed",
            str(primary_seed),
        ],
        [
            trajectory_dir / "sequential_states.csv",
            trajectory_dir / "transition_matrix.csv",
        ],
    )
    trajectory_outputs = [
        experiment_4_dir / "trajectory_summary.csv",
        experiment_4_dir / "case_studies.csv",
        experiment_4_dir / "transition_matrix_ordered.csv",
        experiment_4_dir / "transition_heatmap_ordered.png",
    ]
    if args.force or not all(path.exists() for path in trajectory_outputs):
        analyze_trajectories(
            trajectory_dir / "sequential_states.csv",
            experiment_4_dir,
        )
    else:
        print("[resume] experiment_4 analysis")
    pipeline.mark("experiment_4/analysis", "completed", output=str(experiment_4_dir))
    pipeline.manifest["status"] = "completed"
    pipeline.save_manifest()
    print(f"\nAll experiments completed. Results: {pipeline.root}")


if __name__ == "__main__":
    main()
